"""Render recorded IPC states in a separate CPU Genesis process.

This is explicitly a visual replay, never a second physical simulation. Native
IPC cloth vertices are copied exactly into Genesis's custom visual vertex API.
The original trajectory and its physical validation are not altered.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image
import imageio.v2 as imageio
from run_sim import build_scene,arr,caption,smooth
from trajectory import reference_positions,stage_name,arm_pose

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory')
    parser.add_argument('--width',type=int,default=960)
    parser.add_argument('--out')
    parser.add_argument('--every',type=int,default=1)
    args=parser.parse_args()
    source=Path(args.trajectory).resolve()
    out=Path(args.out).resolve() if args.out else source.parent
    out.mkdir(parents=True,exist_ok=True)
    data=np.load(source)
    report=json.loads((source.parent/'report.json').read_text())
    config=report['parameters']
    if config['mode']!='dressing':raise ValueError('This renderer expects the dressing scene.')
    scene,cloth,bear,entities,robots,cams,_=build_scene(SimpleNamespace(
        backend='cpu',dt=.01,substeps=1,spacing=.012,viewer=False,servo=False,
        width=args.width,replay_cloth=True))
    current_assets=Path(__file__).resolve().parent/'assets'
    for part in bear['parts']:
        saved=source.parent/'assets'/part['mesh']
        if not saved.exists() or saved.read_bytes()!=(current_assets/part['mesh']).read_bytes():
            raise RuntimeError(f'Replay asset differs from recorded run: {part["mesh"]}')
    initial=data['initial']
    cloth.set_vverts(None)
    visual=arr(cloth.get_vverts()).reshape(-1,3)
    distance,mapping=cKDTree(initial).query(visual)
    if distance.max()>2e-6:
        raise RuntimeError(f'Visual mesh does not match recorded topology: {distance.max():.6g}m')
    fps=1/float(np.median(np.diff(data['times']))*args.every)
    overview=imageio.get_writer(str(out/'ipc_reconstruction.mp4'),fps=fps,
        codec='libx264',quality=7,macro_block_size=2)
    detail=imageio.get_writer(str(out/'ipc_front_back.mp4'),fps=fps,
        codec='libx264',quality=7,macro_block_size=2)
    arm_parts=[part for part in bear['parts'] if part['name'].startswith('arm_')]
    arm_local_vertices=[]
    for part in arm_parts:
        entity=entities[part['name']]
        entity.set_vverts(None)
        arm_local_vertices.append(arr(entity.get_vverts()).reshape(-1,3)-part['pos'])
    for name,cam in cams.items():Image.fromarray(cam.render(force_render=True)[0]).save(out/f'initial_{name}.png')
    metrics=json.loads((source.parent/'metrics.json').read_text())
    selected_frames=sorted(set(range(0,len(data['times']),args.every))|{len(data['times'])-1})
    for k in selected_frames:
        elapsed=float(data['times'][k])
        progress=min(max((elapsed/config['dt']-1)/max(config['steps']-1,1),0.),1.)
        cloth.set_vverts(data['positions'][k,mapping])
        for j,part in enumerate(arm_parts):
            if 'arm_positions' in data and 'arm_quaternions' in data:
                pos=data['arm_positions'][k,j];quat=data['arm_quaternions'][k,j]
            elif 'arm_transforms' in data:
                transform=data['arm_transforms'][k,j]
                if transform.shape==(4,4):
                    entities[part['name']].set_vverts(
                        arm_local_vertices[j]@transform[:3,:3].T+transform[:3,3])
                    continue
                else:
                    pos=transform[:3];quat=transform[3:]
            else:
                shoulder,wrist=data['arm_segments'][k,j]
                root=np.asarray(part['shoulder_world']);paw=np.asarray(part['wrist_world'])
                rotation,_=Rotation.align_vectors([wrist-shoulder],[paw-root])
                pos=shoulder+rotation.apply(np.asarray(part['pos'])-root)
                q=rotation.as_quat();quat=q[[3,0,1,2]]
            entities[part['name']].set_pos(pos);entities[part['name']].set_quat(quat)
        for j,(side,robot) in enumerate(zip([-1,1],robots)):
            if 'robot_qpos' in data and len(data['robot_qpos'][k]):
                robot.set_qpos(data['robot_qpos'][k,j])
            else:
                field=reference_positions(initial,progress)
                edge=np.flatnonzero(initial[:,0]*side>.388)
                target=field[edge].mean(axis=0)
                if progress<.10:
                    s=smooth(progress/.10)
                    target=(1-s)*np.array([side*.28,-.05,1.43])+s*target
                if progress>.92:
                    target=target+np.array([side*.18,0,.20])*smooth((progress-.92)/.08)
                target=target+[side*.025,0,.045]
                q=robot.inverse_kinematics(robot.get_link('hand'),pos=target,
                    quat=(0,1,0,0),local_point=(0,0,.1),rot_mask=(False,False,True),
                    init_qpos=robot.get_qpos(),max_solver_iters=35)
                q=arr(q);q[-2:]=.025 if progress>.92 or progress<.08 else .006
                robot.set_qpos(q)
        row=metrics[k]
        labels=[f'Genesis native IPC / {stage_name(progress)} / {elapsed:.2f} s',
            f'Recorded cloth; illustrative robots | guides {row.get("guides_active",0)} | seam candidates {row.get("seam_activated_pairs",0)}']
        overview.append_data(caption(cams['overview'].render(force_render=True)[0],labels))
        detail.append_data(caption(np.concatenate([cams['front'].render(force_render=True)[0],cams['back'].render(force_render=True)[0]],axis=1),labels))
        if k%20==0:print('REPLAY_FRAME',k,'/',len(data['times']),flush=True)
    overview.close();detail.close()
    for name,cam in cams.items():Image.fromarray(cam.render()[0]).save(out/f'final_{name}.png')
    (out/'render_provenance.json').write_text(json.dumps({
        'source_trajectory':str(source),'renderer':'Genesis CPU rasterizer',
        'physics_recomputed':False,'maximum_initial_vertex_mapping_error_m':float(distance.max()),
        'native_vertex_positions_used_exactly':True,'robot_source':'saved qpos if present, otherwise illustrative IK',
        'procedural_bear_assets_match_recorded_run_byte_for_byte':True,
        'arms_use_full_recorded_affine_transform':True,
        'frames':len(selected_frames)},indent=2))

if __name__=='__main__':main()
