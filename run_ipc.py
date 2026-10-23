"""Native Genesis FEM thin-shell + CUDA IPC reconstruction experiment.

The smoke mode drops the two-aperture garment onto a sphere and table. Dressing
uses IPC-native soft vertex guides (released at 92 percent) and a native
SoftVertexStitch seam activated by a state-preserving world rebuild at closure.
No external post-step vertex teleport/projection
is applied. Robot arms are illustrative IK replay, not claimed force actuators.
Run with host GPU access; sandboxed processes on this host cannot see CUDA.
"""
from __future__ import annotations
import os,sys,json,time,argparse,types,faulthandler,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','0')
os.environ.setdefault('GS_CACHE_FILE_PATH',str(ROOT/'.cache'/'ipc_genesis'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
faulthandler.enable()
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image
import imageio.v2 as imageio
import genesis as gs
import genesis.utils.geom as gu
import uipc
from uipc.constitution import SoftPositionConstraint,SoftVertexStitch
from uipc.core import Engine,World,FiniteElementStateAccessorFeature,AffineBodyStateAccessorFeature
from assets_builder import build_assets
from cloth_model import build_cloth
from trajectory import reference_positions,stage_name,arm_pose


def array(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


def groups(p):
    u,v=p[:,0],p[:,1]+.25
    outer=(np.abs(u)>.388)|((v<-.087)&((np.abs(u)<.12)|(np.abs(u)>.31)))
    rims=[]
    for c in [-.205,.205]:
        du,dv=np.abs(u-c),np.abs(v-.045)
        rims.append(np.flatnonzero((du<.046)&(dv<.049)&((du>.028)|(dv>.031))))
    rim_guides=np.concatenate([r[(v[r]>.045)&(np.abs(u[r])>.205)] for r in rims])
    guides=np.flatnonzero(outer|np.isin(np.arange(len(p)),rim_guides))
    left=np.flatnonzero((u<-.388)&(np.abs(v)<.084))
    right=np.flatnonzero((u>.388)&(np.abs(v)<.084))
    i,j=linear_sum_assignment(np.abs(v[left,None]-v[right][None,:]))
    return guides,rims,np.c_[left[i],right[j]]


class IPCGuides:
    """Per-instance build adapter to native uipc constitutions and Animator."""
    def __init__(self,cloth,strength=100.,seam_strength=1e6):
        self.initial=array(cloth.init_positions).reshape(-1,3).copy()
        self.guide_ids,self.rims,self.pairs=groups(self.initial)
        self.strength=strength
        self.seam_strength=seam_strength
        self.progress=0.
        self.active=np.zeros(len(self.pairs),bool)
        self.callback_count=0
        self.seam_callback_count=0
        self.slot=None
        self.stitch_geometry=None
        self.last_guide_error=0.
        self.seam_transition=None
        self._retained_worlds=[]

    def install(self,coupler):
        original=coupler._finalize_ipc
        controller=self
        def finalize(this):
            controller._before_init(this)
            original()
        coupler._finalize_ipc=types.MethodType(finalize,coupler)

    def _before_init(self,coupler):
        ipc_scene=coupler._ipc_scene
        objects=ipc_scene.objects().find('cloth_0_0')
        if len(objects)!=1:
            raise RuntimeError(f'Expected one IPC cloth object, found {len(objects)}')
        obj=objects[0]
        ids=obj.geometries().ids()
        self.slot,rest_slot=ipc_scene.geometries().find(int(ids[0]))
        self.rest_slot=rest_slot
        self.spc=SoftPositionConstraint()
        ipc_scene.constitution_tabular().insert(self.spc)
        for slot in [self.slot,rest_slot]:
            geom=slot.geometry()
            self.spc.apply_to(geom,strength_rate=self.strength)
            uipc.view(geom.vertices().find(uipc.builtin.is_constrained))[:]=0
        def guide_callback(info):
            self.callback_count+=1
            geom=info.geo_slots()[0].geometry()
            constrained=uipc.view(geom.vertices().find(uipc.builtin.is_constrained)).reshape(-1)
            aim=uipc.view(geom.vertices().find(uipc.builtin.aim_position)).reshape(-1,3)
            constrained[:]=0
            if self.progress<.92:
                constrained[self.guide_ids]=1
                target=reference_positions(self.initial,self.progress)
                aim[self.guide_ids]=target[self.guide_ids]
                actual=np.asarray(geom.positions().view()).reshape(-1,3)
                self.last_guide_error=float(np.linalg.norm(actual[self.guide_ids]-target[self.guide_ids],axis=1).max())
        ipc_scene.animator().insert(obj,guide_callback)
        # In pyuipc0.0.7 kappa is immutable after backend initialization. A single
        # state-preserving new Engine/World at closure activates the native seam.
        self.stitch=SoftVertexStitch()
        ipc_scene.constitution_tabular().insert(self.stitch)
        self.stitch_geometry=self.stitch.create_geometry((self.slot,self.slot),
            self.pairs.astype(np.int32),kappa=0.,rest_length=.0015)
        seam_obj=ipc_scene.objects().create('image_estimated_velcro_seam')
        self.seam_slot,self.seam_rest_slot=seam_obj.geometries().create(self.stitch_geometry)
        def seam_callback(info):
            self.seam_callback_count+=1
        ipc_scene.animator().insert(seam_obj,seam_callback)

    def maybe_activate_seam(self,coupler,positions):
        """Preserve native FEM/ABD states and rest shape while enabling bonds."""
        if self.seam_transition is not None or not .90<=self.progress<.92:return
        gap=np.linalg.norm(positions[self.pairs[:,0]]-positions[self.pairs[:,1]],axis=1)
        active=gap<.020
        if not np.any(active):return
        old_world,old_engine=coupler._ipc_world,coupler._ipc_engine
        fem=old_world.features().find(FiniteElementStateAccessorFeature.FeatureName)
        state=fem.create_geometry()
        for name in ['position','velocity']:state.vertices().create(name,np.zeros(3,dtype=np.float64))
        fem.copy_to(state)
        saved={name:np.array(state.vertices().find(name).view(),copy=True) for name in ['position','velocity']}
        abd=old_world.features().find(AffineBodyStateAccessorFeature.FeatureName)
        abd_state=abd.create_geometry()
        abd_state.instances().create('transform',np.eye(4,dtype=np.float64))
        abd_state.instances().create('velocity',np.zeros((4,4),dtype=np.float64))
        abd.copy_to(abd_state)
        abd_saved={name:np.array(abd_state.instances().find(name).view(),copy=True) for name in ['transform','velocity']}
        rest_before=np.array(self.rest_slot.geometry().positions().view(),copy=True)
        for slot in [self.seam_slot,self.seam_rest_slot]:
            uipc.view(slot.geometry().instances().find('kappa')).reshape(-1)[:]=active*self.seam_strength
        new_engine=Engine('cuda',tempfile.mkdtemp(prefix='genesis-ipc-seam-'))
        new_world=World(new_engine)
        new_world.init(coupler._ipc_scene)
        if not new_world.is_valid():raise RuntimeError('Native seam transition world is invalid')
        new_fem=new_world.features().find(FiniteElementStateAccessorFeature.FeatureName)
        new_abd=new_world.features().find(AffineBodyStateAccessorFeature.FeatureName)
        new_fem.copy_from(state);new_abd.copy_from(abd_state)
        new_fem.copy_to(state);new_abd.copy_to(abd_state)
        errors={f'fem_{name}_max_error':float(np.max(np.abs(state.vertices().find(name).view()-value)))
                for name,value in saved.items()}
        errors.update({f'abd_{name}_max_error':float(np.max(np.abs(abd_state.instances().find(name).view()-value)))
                       for name,value in abd_saved.items()})
        errors['rest_position_max_error']=float(np.max(np.abs(self.rest_slot.geometry().positions().view()-rest_before)))
        if max(errors.values())>1e-9:raise RuntimeError(f'Seam transition state mismatch: {errors}')
        coupler._ipc_engine,coupler._ipc_world=new_engine,new_world
        coupler._abd_state_feature=new_abd
        self._retained_worlds.append((old_engine,old_world))
        self.active=active
        self.seam_transition=dict(progress=self.progress,pairs=int(active.sum()),
                                  native_world_rebuilt=True,state_transfer_errors=errors)
        print('IPC_NATIVE_SEAM_ACTIVATED',json.dumps(self.seam_transition),flush=True)


def create_scene(args,out):
    assets=out/'assets'
    garment=build_cloth(assets)
    bear=build_assets(assets) if args.mode=='dressing' else None
    gs.init(backend=gs.cpu,logging_level='info',seed=11)
    scene=gs.Scene(sim_options=gs.options.SimOptions(dt=args.dt,substeps=1),
        coupler_options=gs.options.IPCCouplerOptions(
            newton_max_iterations=100,newton_tolerance=.01,
            newton_semi_implicit_enable=False,n_linesearch_iterations=12,
            linear_system_tolerance=1e-3,contact_enable=True,
            contact_d_hat=.001,contact_resistance=1e7,contact_friction_enable=True,
            contact_eps_velocity=.005,enable_rigid_rigid_contact=False,
            enable_rigid_ground_contact=False,sanity_check_enable=True),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        vis_options=gs.options.VisOptions(background_color=(.033,.049,.065),
            ambient_light=(.35,.35,.35),shadow=True,plane_reflection=False),
        show_viewer=False)
    contact=gs.materials.Rigid(needs_coup=True,coup_type='ipc_only',coup_friction=.6)
    scene.add_entity(gs.morphs.Box(pos=(0,0,.725),size=(1.4,.98,.05),fixed=True),
        contact,gs.surfaces.Default(color=(.48,.32,.19,1)))
    entities={}
    if args.mode=='smoke':
        scene.add_entity(gs.morphs.Sphere(radius=.06,pos=(0,-.25,.79),fixed=True),
            contact,gs.surfaces.Default(color=(.43,.25,.12,1)))
    else:
        for part in bear['parts']:
            arm=part['name'].startswith('arm_')
            material=gs.materials.Kinematic()
            if part['collision']:
                material=(gs.materials.Rigid(needs_coup=True,coup_type='two_way_soft_constraint',
                    coup_friction=.6,gravity_compensation=1.) if arm else contact)
            entities[part['name']]=scene.add_entity(
                gs.morphs.Mesh(file=str(assets/part['mesh']),pos=part['pos'],
                    fixed=not arm,collision=part['collision'],convexify=True,
                    decimate=part['collision'],decimate_face_num=250),material,
                gs.surfaces.Default(color=part['color'],smooth=True))
    offset=(0,0,.15) if args.mode=='smoke' else (0,0,0)
    cloth=scene.add_entity(gs.morphs.Mesh(file=garment['obj'],pos=offset),
        gs.materials.FEM.Cloth(E=args.young,nu=.3,rho=240,thickness=.001,
            bending_stiffness=.01,friction_mu=.6),
        gs.surfaces.Default(double_sided=True,smooth=False))
    controller=IPCGuides(cloth,args.guide_strength,args.seam_strength) if args.mode=='dressing' else None
    if controller is not None:
        controller.install(scene.sim.coupler)
    robots=[]
    if args.mode=='dressing' and not args.no_robots:
        for side,yaw in [(-1,0),(1,180)]:
            robots.append(scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml',
                pos=(side*.55,.14,.78),euler=(0,0,yaw),collision=False),gs.materials.Kinematic()))
    cams={}
    for name,pos,look,fov in [
        ('overview',(1.7,-2.3,1.95),(0,0,1.),42),
        ('front',(0,-1.6,1.22),(0,.07,1.035),28),
        ('back',(0,1.65,1.24),(0,.08,1.04),28)]:
        if not args.no_render:
            cams[name]=scene.add_camera(res=(args.width,args.width*9//16),pos=pos,lookat=look,fov=fov,GUI=False)
    print('IPC_BUILD_BEGIN',flush=True)
    scene.build()
    print('IPC_BUILD_OK',flush=True)
    for robot in robots:
        robot.set_qpos([0,-.4,0,-1.9,0,1.6,.78,.035,.035])
    return scene,cloth,bear,entities,controller,robots,cams


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','dressing'],default='smoke')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--settle-steps',type=int,default=100)
    parser.add_argument('--max-run-steps',type=int,default=None,
                        help='Stop early for an initial-scene/constraint smoke test, without accelerating the reference.')
    parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--out',default='output/ipc_smoke')
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--render-every',type=int,default=5)
    parser.add_argument('--young',type=float,default=6e4)
    parser.add_argument('--guide-strength',type=float,default=100.)
    parser.add_argument('--seam-strength',type=float,default=1e6)
    parser.add_argument('--no-video',action='store_true')
    renderer=parser.add_mutually_exclusive_group()
    renderer.add_argument('--no-render',dest='no_render',action='store_true',
                          help='Physics only; no OpenGL context or video (default).')
    renderer.add_argument('--render',dest='no_render',action='store_false',
                          help='Attempt live GPU rendering; offline render_ipc_replay.py is the verified route.')
    parser.set_defaults(no_render=True)
    parser.add_argument('--no-robots',action='store_true')
    args=parser.parse_args()
    out=(ROOT/args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    begin=time.time()
    scene,cloth,bear,entities,controller,robots,cams=create_scene(args,out)
    initial=array(cloth.get_state().pos).reshape(-1,3).copy()
    faces=np.asarray(cloth.surface_triangles)
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    rest=np.linalg.norm(initial[edges[:,1]]-initial[edges[:,0]],axis=1)
    hist=[];times=[];metrics=[];arm_history=[];arm_transform_history=[];robot_qpos_history=[]
    arm_offsets=[]
    if controller is not None:
        for part in bear['parts']:
            if part['name'].startswith('arm_'):
                desc=entities[part['name']].base_link.desc
                arm_offsets.append(np.asarray(gu.trans_quat_to_T(desc.offset_pos,desc.offset_quat)))
    def save_trajectory():
        payload=dict(positions=np.asarray(hist),initial=initial,faces=faces,edges=edges,times=np.asarray(times),
                     arm_segments=np.asarray(arm_history),arm_transforms=np.asarray(arm_transform_history),
                     robot_qpos=np.asarray(robot_qpos_history),arm_inertia_offsets=np.asarray(arm_offsets),
                     arm_transform_frame=np.asarray('authored_mesh'))
        if controller is not None:payload.update(seam_pairs=controller.pairs,bonded=controller.active,
            guide_ids=controller.guide_ids,rim_left=controller.rims[0],rim_right=controller.rims[1],
            body_radii=np.asarray(bear['body_radii']))
        np.savez_compressed(out/'trajectory.npz',**payload)
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    for name,cam in cams.items():Image.fromarray(cam.render()[0]).save(out/f'initial_{name}.png')
    writer=None if args.no_video or args.no_render else imageio.get_writer(str(out/'ipc_reconstruction.mp4'),
        fps=1/(args.dt*args.render_every),codec='libx264',quality=7,macro_block_size=2)
    last_step=args.steps+(args.settle_steps if args.mode=='dressing' else 0)
    if args.max_run_steps is not None:last_step=min(last_step,args.max_run_steps)
    run_error=None;completed_steps=0;release_first_step=None
    for i in range(last_step):
        progress=min(i/max(args.steps-1,1),1.)
        current_arms=[];current_arm_transforms=[]
        if controller is not None:
            controller.progress=progress
            for part in bear['parts']:
                if part['name'].startswith('arm_'):
                    pos,quat,shoulder,wrist=arm_pose(part,progress)
                    entities[part['name']].set_pos(pos)
                    entities[part['name']].set_quat(quat)
                    current_arms.append([shoulder,wrist])
            if robots and i%4==0:
                target=reference_positions(controller.initial,progress)
                for side,robot,k in zip([-1,1],robots,[0,1]):
                    target_hand=target[controller.pairs[:,k]].mean(axis=0)+[side*.025,0,.045]
                    q=robot.inverse_kinematics(robot.get_link('hand'),pos=target_hand,
                        quat=(0,1,0,0),local_point=(0,0,.1),rot_mask=(False,False,True),max_solver_iters=30)
                    robot.set_qpos(q)
        try:
            scene.step()
            if not scene.sim.coupler._ipc_world.is_valid():
                raise RuntimeError('IPC world reported invalid state after advance.')
        except Exception as exc:
            run_error=repr(exc)
            print('IPC_ABORT',run_error,flush=True)
            break
        completed_steps+=1
        if controller is not None and progress>=.92 and release_first_step is None:release_first_step=i
        p=array(cloth.get_state().pos).reshape(-1,3).copy()
        # Record IPC's simulated arm transforms rather than assuming the soft
        # rigid targets were reached exactly. The geometry audit uses these.
        if controller is not None:
            actual_arms=[]
            for part in bear['parts']:
                if not part['name'].startswith('arm_'):continue
                entity=entities[part['name']]
                entry=scene.sim.coupler._abd_data_by_link.get(entity.base_link)
                if entry is None:
                    raise RuntimeError('Missing actual IPC arm transform for validation.')
                desc=entity.base_link.desc
                offset=np.asarray(gu.trans_quat_to_T(desc.offset_pos,desc.offset_quat))
                T=np.asarray(entry[0].transform)@np.linalg.inv(offset)
                current_arm_transforms.append(T.copy())
                actual_arms.append([(T[:3,:3]@np.asarray(part['shoulder_local'])+T[:3,3]).tolist(),
                                    (T[:3,:3]@np.asarray(part['wrist_local'])+T[:3,3]).tolist()])
            current_arms=actual_arms
        if not np.isfinite(p).all():
            run_error='Nonfinite IPC cloth positions';break
        if controller is not None:
            try:controller.maybe_activate_seam(scene.sim.coupler,p)
            except Exception as exc:
                run_error=f'Seam transition failed: {exc!r}'
                print('IPC_ABORT',run_error,flush=True)
                break
        if i%args.render_every==0 or i==last_step-1:
            ratio=np.linalg.norm(p[edges[:,1]]-p[edges[:,0]],axis=1)/rest
            row={'step':i,'time_s':(i+1)*args.dt,'mean_z_m':float(p[:,2].mean()),
                'min_z_m':float(p[:,2].min()),'stretch_max':float(ratio.max()),
                'stretch_p95':float(np.quantile(ratio,.95)),'finite':True}
            if controller is not None:
                gap=np.linalg.norm(p[controller.pairs[:,0]]-p[controller.pairs[:,1]],axis=1)
                row.update(guides_active=len(controller.guide_ids) if progress<.92 else 0,
                    seam_activated_pairs=int(controller.active.sum()),seam_mean_m=float(gap.mean()),
                    seam_max_m=float(gap.max()),guide_target_error_max_m=controller.last_guide_error,
                    stage=stage_name(progress))
            metrics.append(row);hist.append(p);times.append((i+1)*args.dt);arm_history.append(current_arms)
            arm_transform_history.append(current_arm_transforms)
            robot_qpos_history.append([array(robot.get_qpos()).copy() for robot in robots])
            if writer:writer.append_data(cams['overview'].render()[0])
            (out/'progress.json').write_text(json.dumps(row,indent=2))
            if i%100==0:save_trajectory()
        if i%20==0:print('IPC_STEP',i,'/',last_step,'elapsed',round(time.time()-begin,1),flush=True)
    if writer:writer.close()
    for name,cam in cams.items():Image.fromarray(cam.render()[0]).save(out/f'final_{name}.png')
    save_trajectory()
    report={'mode':args.mode,'parameters':vars(args),'native_contact':'CUDA libuipc via Genesis IPCCoupler',
        'cloth_model':'StrainLimitingBaraffWitkinShell + DiscreteShellBending',
        'post_step_particle_projection_used':False,'elapsed_s':time.time()-begin,
        'guide_callback_count':0 if controller is None else controller.callback_count,
        'stitch_callback_count':0 if controller is None else controller.seam_callback_count,
        'host_only_runtime_stitch_update_supported':False,
        'seam_activation_method':'new native IPC world with exact FEM/ABD state and rest-shape preservation',
        'seam_transition':None if controller is None else controller.seam_transition,
        'native_stitch_state_transfer_verified':bool(controller is not None and controller.seam_transition is not None),
        'robot_contact_force_control_used':False,
        'run_error':run_error,'completed_steps':completed_steps,
        'last_recorded_time_s':times[-1] if times else None,
        'release_reached':bool(metrics and controller is not None and metrics[-1]['guides_active']==0),
        'released_duration_s':0. if release_first_step is None else (completed_steps-release_first_step)*args.dt,
        'last':metrics[-1] if metrics else None,'success_certified':False,
        'limitations':['Image-inferred geometry/materials; not original simulation recovery.',
            'External soft guide actuation; robots are visual IK replay.',
            'Distance-triggered native soft stitch models Velcro approximately; not hook-and-loop microstructure.',
            'No full dressing success claim without release/topology validation.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    if controller is not None and hist:
        from validate import validate
        validation=validate(out/'trajectory.npz')
        report['last_recorded_geometric_checks']={
            'both_arm_axes_through_apertures':validation['both_arm_axes_through_intended_apertures_final'],
            'torso_coverage_at_095m':validation['torso_coverage'][1]['near_torso_angular_coverage_fraction'],
            'max_edge_stretch':validation['material_edge_stretch']['maximum_ratio'],
            'p95_edge_stretch':validation['material_edge_stretch']['p95_ratio'],
            'torso_inside_vertices':validation['torso_vertex_penetration']['vertices_inside_98_percent_ellipsoid'],
        }
        if report['release_reached']:report['release_geometric_checks']=report['last_recorded_geometric_checks']
        (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
