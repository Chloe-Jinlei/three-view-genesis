"""Image-only teddy dressing reconstruction in local Genesis.

This is an ASSISTED, scripted reconstruction, not recovery of the original
parameters and not an autonomous robot policy. See report.json for measured
outcomes. Sparse material boundary/rim guides are explicit external supports.
Only the final release phase removes these guides. A seam couples cloth to
cloth, never to a fixed world point. Robot motion uses IK pose replay by default.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('GS_CACHE_FILE_PATH', str(ROOT / '.cache' / 'genesis'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache' / 'numba'))
for cache_path in ['genesis','matplotlib','numba']:
    (ROOT / '.cache' / cache_path).mkdir(parents=True,exist_ok=True)
import argparse, json, time, faulthandler
faulthandler.enable()
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import genesis as gs
from assets_builder import build_assets
from cloth_model import build_cloth
from trajectory import reference_positions, stage_name, arm_pose
from parameters import BODY_RADII, ARMS_ABDUCTION_DEG
from contact_extension import apply_contact_extension, build_proxies
from strain_limiter import project_strain


def arr(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


def smooth(x):
    x=np.clip(x,0,1); return x*x*(3-2*x)


def caption(rgb, lines):
    im=Image.fromarray(rgb).convert('RGB'); d=ImageDraw.Draw(im)
    d.rectangle((0,0,im.width,62),fill=(13,21,27))
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,17)
    small=ImageFont.truetype(fontpath,13)
    d.text((18,8),lines[0],font=font,fill=(241,230,201))
    d.text((18,35),lines[1],font=small,fill=(170,189,197))
    return np.asarray(im)


def build_scene(args):
    assets=ROOT/'assets'; bear=build_assets(assets); garment=build_cloth(assets)
    gs.init(backend=gs.cpu if args.backend=='cpu' else gs.gpu,logging_level='warning',seed=7)
    scene=gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt,substeps=args.substeps),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        pbd_options=gs.options.PBDOptions(particle_size=args.spacing,
            lower_bound=(-1,-1,.5),upper_bound=(1,1,1.8),
            max_stretch_solver_iterations=10,max_bending_solver_iterations=2),
        vis_options=gs.options.VisOptions(background_color=(.033,.049,.065),
            ambient_light=(.33,.33,.33),shadow=True,plane_reflection=False),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.7,-2.3,1.95),
            camera_lookat=(0,0,1.0),camera_fov=42),
        show_viewer=args.viewer)
    contact=gs.materials.Rigid(needs_coup=True,coup_softness=.001)
    decorative=gs.materials.Rigid(needs_coup=False)
    def box(pos,size,color,collision=True):
        return scene.add_entity(gs.morphs.Box(pos=pos,size=size,fixed=True,collision=collision),
            material=contact if collision else decorative,surface=gs.surfaces.Default(color=(*color,1)))
    scene.add_entity(gs.morphs.Plane(),material=contact,surface=gs.surfaces.Default(color=(.16,.18,.19,1)))
    box((0,0,.725),(1.4,.98,.05),(.48,.32,.19))
    for x in [-.60,.60]:
        for y in [-.39,.39]: box((x,y,.35),(.045,.045,.70),(.10,.13,.14),False)
    bear_entities={}
    for p in bear['parts']:
        replay_arm=getattr(args,'replay_cloth',False) and p['name'].startswith('arm_')
        bear_entities[p['name']]=scene.add_entity(gs.morphs.Mesh(file=str(assets/p['mesh']),
            pos=p['pos'],fixed=True,collision=p['collision'] and not replay_arm,
            enable_custom_vverts=replay_arm,convexify=True,
            decimate=p['collision'],decimate_face_num=300),
            material=gs.materials.Kinematic() if replay_arm else contact if p['collision'] else decorative,
            surface=gs.surfaces.Default(color=p['color'],smooth=True))
    robots=[]
    for s,angle in [(-1,0),(1,180)]:
        box((s*.55,.14,.765),(.17,.17,.03),(.12,.16,.18),False)
        robot=scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml',
            pos=(s*.55,.14,.78),euler=(0,0,angle),collision=args.servo),
            material=gs.materials.Rigid(needs_coup=False) if args.servo else gs.materials.Kinematic())
        robots.append(robot)
    # Explicit texture retains material identity throughout deformation.
    clothfile=garment.get('obj_path',garment.get('mesh_path',garment.get('mesh')))
    if not Path(clothfile).is_absolute(): clothfile=str(assets/clothfile)
    if getattr(args,'replay_cloth',False):
        cloth=scene.add_entity(gs.morphs.Mesh(file=str(clothfile),collision=False,
            enable_custom_vverts=True),material=gs.materials.Kinematic(),
            surface=gs.surfaces.Default(double_sided=True,smooth=True))
    else:
        cloth=scene.add_entity(gs.morphs.Mesh(file=str(clothfile)),
            material=gs.materials.PBD.Cloth(rho=.24,stretch_compliance=2e-8,
                bending_compliance=3e-5,stretch_relaxation=.25,bending_relaxation=.10,
                air_resistance=.00005),
            surface=gs.surfaces.Default(double_sided=True,smooth=False))
    cams={}
    for name,pos,look,up,fov in [
        ('overview',(1.7,-2.3,1.95),(0,0,1.0),(0,0,1),42),
        ('front',(0,-1.6,1.22),(0,.07,1.035),(0,0,1),28),
        ('back',(0,1.65,1.24),(0,.08,1.04),(0,0,1),28),
        ('top',(0,-.015,2.65),(0,0,.75),(0,1,0),48)]:
        cams[name]=scene.add_camera(res=(args.width,args.width*9//16),pos=pos,lookat=look,up=up,fov=fov,GUI=False)
    print('Building Genesis scene',flush=True); scene.build()
    for r in robots:
        q=np.array([0,-.40,0,-1.9,0,1.6,.78,.035,.035])
        r.set_qpos(q)
        if args.servo:
            r.set_dofs_kp([450,450,350,350,180,100,70,180,180])
            r.set_dofs_kv([45,45,35,35,20,12,10,8,8])
            r.control_dofs_position(q)
    return scene,cloth,bear,bear_entities,robots,cams,garment


def select_groups(p):
    u=p[:,0]; v=p[:,1]+.25
    # Geometrically selected after Genesis remeshing; no original vertex ids assumed.
    outer=(np.abs(u)>.388)|((np.abs(v)>.087)&((np.abs(u)<.12)|(np.abs(u)>.31)))
    rims=[]
    for c in [-.205,.205]:
        du=np.abs(u-c); dv=np.abs(v-.045)
        ring=((du<.046)&(dv<.049)&((du>.028)|(dv>.031)))
        rims.append(np.flatnonzero(ring))
    # Guide only the upper aperture arc. The lower rim must conform freely to
    # the arm/torso junction; pinning it to a projected target tears the fit.
    rim_guided=np.concatenate([ids[v[ids]>.045] for ids in rims])
    guide=np.flatnonzero(outer | np.isin(np.arange(len(p)),rim_guided))
    ends=[np.flatnonzero((u<-.385)&(np.abs(v)<.084)),np.flatnonzero((u>.385)&(np.abs(v)<.084))]
    # Stable, one-to-one sewing of corresponding material heights.
    li,ri=ends
    a,b=linear_sum_assignment(np.abs(v[li,None]-v[ri][None,:]))
    pairs=np.c_[li[a],ri[b]]
    return guide,rims,pairs


def sew(p,pairs,active,threshold=.024):
    a,b=pairs.T; delta=p[b]-p[a]; dist=np.linalg.norm(delta,axis=1)
    active=active|(dist<threshold)
    # Bonded layers retain 1.05 particle diameter separation to avoid self-contact conflicts.
    if np.any(active):
        aa,bb=a[active],b[active]; dd=delta[active]; ll=dist[active]
        # A glued patch resists shear as well as opening. Use a common, freely
        # rotating layer normal instead of independent distance-only strings.
        normal=dd.mean(axis=0)
        normal=normal/max(np.linalg.norm(normal),1e-8)
        if np.linalg.norm(normal)<.5:normal=np.array([0.,1.,0.])
        midpoint=.5*(p[aa]+p[bb])
        p[aa]=midpoint-.0065*normal; p[bb]=midpoint+.0065*normal
    return p,active,dist


def metrics(p,p0,rims,pairs,active,tree_edges,rest):
    center=np.array([0,.10,.965]); radii=BODY_RADII
    ell=np.sqrt(np.sum(((p-center)/radii)**2,axis=1))
    low=(p[:,2]>.81)&(p[:,2]<1.13)
    ratios=np.linalg.norm(p[tree_edges[:,1]]-p[tree_edges[:,0]],axis=1)/rest
    enddist=np.linalg.norm(p[pairs[:,0]]-p[pairs[:,1]],axis=1)
    return {'finite':bool(np.isfinite(p).all()),'particles':len(p),
        'bonded_pairs':int(active.sum()),'candidate_pairs':len(pairs),
        'seam_mean_m':float(enddist.mean()),'seam_max_m':float(enddist.max()),
        'body_inside_particles':int(((ell<.98)&low).sum()),
        'local_stretch_p95':float(np.quantile(ratios,.95)),
        'local_stretch_max':float(ratios.max()),
        'cloth_min_z_m':float(p[:,2].min()),'cloth_mean_z_m':float(p[:,2].mean()),
        'rim_centers_m':[p[ids].mean(axis=0).tolist() for ids in rims]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps',type=int,default=480)
    ap.add_argument('--settle-steps',type=int,default=150,
                    help='Extra freely settling steps after trajectory; seam only remains')
    ap.add_argument('--dt',type=float,default=.02)
    ap.add_argument('--substeps',type=int,default=8)
    ap.add_argument('--spacing',type=float,default=.012)
    ap.add_argument('--width',type=int,default=960)
    ap.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    ap.add_argument('--viewer',action='store_true')
    ap.add_argument('--servo',action='store_true',help='PD control robot joints; default explicit IK visualization')
    ap.add_argument('--contact-extension',action='store_true',
                    help='Add explicit swept-particle ellipsoid contact and approximate Coulomb friction')
    ap.add_argument('--strain-limit',action='store_true',help='Add explicit edge-length projection; reports residual strain')
    ap.add_argument('--out',default='output/assisted')
    ap.add_argument('--initial-only',action='store_true')
    ap.add_argument('--render-every',type=int,default=4)
    ap.add_argument('--no-video',action='store_true')
    ap.add_argument('--detail-video',action='store_true',help='Record simultaneous close front/back views')
    ap.add_argument('--guide',choices=['boundary','dense','grips'],default='boundary')
    args=ap.parse_args(); out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    start=time.time(); scene,cloth,bear,entities,robots,cams,garment=build_scene(args)
    p0=arr(cloth.get_particles_pos()).copy(); np.save(out/'initial_particles.npy',p0)
    guide,rims,pairs=select_groups(p0)
    if args.guide=='dense': guide=np.arange(len(p0))
    if args.guide=='grips': guide=np.flatnonzero((np.abs(p0[:,0])>.385)&(p0[:,1]>-.19))
    # Engine remeshed topology, not OBJ vertex indices or proximity edges.
    edges=np.asarray(cloth.edges,dtype=int)
    faces=np.asarray(cloth.mesh.faces,dtype=int)
    rest=np.linalg.norm(p0[edges[:,1]]-p0[edges[:,0]],axis=1)
    active=np.zeros(len(pairs),bool); hist=[]; arm_hist=[]; robot_hist=[]; records=[]; prevstage=None
    hands=[r.get_link('hand') for r in robots]
    # Establish symmetric initial raised robot pose.
    qprev=[]
    for s,r,h in zip([-1,1],robots,hands):
        q=r.inverse_kinematics(h,pos=(s*.28,-.05,1.43),quat=(0,1,0,0),
            local_point=(0,0,.10),rot_mask=(False,False,True),max_solver_iters=40)
        r.set_qpos(q); qprev.append(arr(q))
        if args.servo:r.control_dofs_position(q)
    for name,cam in cams.items():
        Image.fromarray(cam.render()[0]).save(out/f'initial_{name}.png')
    if args.initial_only:
        print('Initial scene saved',out,flush=True); return
    cloth.fix_particles(guide.tolist())
    writer=None; detail_writer=None
    if not args.no_video:
        writer=imageio.get_writer(str(out/'reconstruction.mp4'),fps=1/(args.dt*args.render_every),codec='libx264',quality=8,macro_block_size=2)
        if args.detail_video:
            detail_writer=imageio.get_writer(str(out/'front_back.mp4'),fps=1/(args.dt*args.render_every),codec='libx264',quality=8,macro_block_size=2)
    released=False; before_release=None; ee_errors=[]
    arms=[x for x in bear['parts'] if x['name'].startswith('arm_')]
    proxies=build_proxies(bear)
    contact_records=[]; strain_records=[]
    last_step=args.steps+args.settle_steps
    for i in range(last_step+1):
        t=min(i/args.steps,1.0); stage=stage_name(t)
        target=reference_positions(p0,t)
        if stage!=prevstage:
            print(f'{i}/{args.steps} {stage} elapsed={time.time()-start:.1f}s',flush=True); prevstage=stage
        # Scripted teddy shoulder articulation, separately recorded as an assumption.
        current_arms=[]
        contact_parts=[part for part in bear['parts'] if not part['name'].startswith('arm_')]
        for part in arms:
            position,quat,shoulder,wrist=arm_pose(part,t)
            entities[part['name']].set_pos(position)
            entities[part['name']].set_quat(quat)
            current_arms.append([shoulder,wrist])
            contact_parts.append(dict(part,pos=position,shoulder_world=shoulder,wrist_world=wrist))
        if args.contact_extension:
            proxies=build_proxies(dict(bear,parts=contact_parts),particle_radius=args.spacing/2)
        if t>=.92 and not released:
            before_release=arr(cloth.get_particles_pos()).copy()
            cloth.release_particle(guide.tolist()); released=True
            print('ALL EXTERNAL CLOTH GUIDES RELEASED',flush=True)
        if not released:cloth.set_particles_pos(target[guide],particles_idx_local=guide)
        # Robot grippers follow the two manipulation handles. This is IK replay,
        # not a claim that the visual robot alone generated every guide force.
        if i%4==0:
            for k,(s,r,h) in enumerate(zip([-1,1],robots,hands)):
                ids=pairs[:,k]; handle=target[ids].mean(axis=0)
                if t<.10:
                    handle=(1-smooth(t/.10))*np.array([s*.28,-.05,1.43])+smooth(t/.10)*handle
                if t>.92:handle=handle+np.array([s*.18,0,.20])*smooth((t-.92)/.08)
                # Avoid two coincident bulky robot wrists at a sewn seam.
                handle=handle+np.array([s*.025,0,.045])
                q,err=r.inverse_kinematics(h,pos=handle,quat=(0,1,0,0),local_point=(0,0,.10),
                    rot_mask=(False,False,True),init_qpos=r.get_qpos(),return_error=True,max_solver_iters=35)
                q=arr(q); q[-2:]=.025 if t>.92 or t<.08 else .006
                if args.servo:r.control_dofs_position(q)
                else:r.set_qpos(q)
                ee_errors.append(float(np.linalg.norm(arr(err)[:3])))
        previous=arr(cloth.get_particles_pos()).copy() if args.contact_extension else None
        scene.step()
        p=arr(cloth.get_particles_pos())
        native_p=p.copy()
        free_mask=np.ones(len(p),bool)
        if not released:free_mask[guide]=False
        velocity=arr(cloth.get_particles_vel())
        if args.strain_limit:
            for projection_pass in range(4):
                if t>=.84:p,active,_=sew(p.copy(),pairs,active)
                p,sm=project_strain(p,edges,rest,free_mask,iterations=32 if released else 12,max_ratio=1.01,min_ratio=.85)
                if args.contact_extension:
                    p,velocity,inner_cm=apply_contact_extension(previous,p,velocity,args.dt,bear,
                        free_mask=free_mask,mu=.6,particle_radius=args.spacing/2,proxies=proxies)
                    contact_records.append(dict(inner_cm,step=i,projection_pass=projection_pass))
            velocity=velocity+(p-native_p)/args.dt
            strain_records.append(sm)
        if args.contact_extension:
            p,velocity,cm=apply_contact_extension(previous,p,velocity,args.dt,bear,
                free_mask=free_mask,mu=.6,particle_radius=args.spacing/2,proxies=proxies)
            touched=np.flatnonzero(free_mask)
            cloth.set_particles_pos(p[touched],particles_idx_local=touched)
            cloth.set_particles_vel(velocity[touched],particles_idx_local=touched)
            contact_records.append(cm)
        elif args.strain_limit:
            touched=np.flatnonzero(free_mask)
            cloth.set_particles_pos(p[touched],particles_idx_local=touched)
            cloth.set_particles_vel(velocity[touched],particles_idx_local=touched)
        if t>=.84:
            p,active,dist=sew(p.copy(),pairs,active)
            if np.any(active):
                sewn=np.unique(pairs[active]);vel=arr(cloth.get_particles_vel()).copy()
                aa,bb=pairs[active].T
                shared=.5*(vel[aa]+vel[bb]);vel[aa]=shared;vel[bb]=shared
                cloth.set_particles_pos(p[sewn],particles_idx_local=sewn)
                # Restore common pair motion after position API resets velocities.
                # Relative seam velocity is damped; pair translation remains free.
                cloth.set_particles_vel(vel[sewn],particles_idx_local=sewn)
        if not np.isfinite(p).all():raise RuntimeError('Nonfinite cloth state')
        if i%10==0:
            print(f'CHECK {i} bounds={p.min(axis=0).round(3)} {p.max(axis=0).round(3)}',flush=True)
            np.save(out/'checkpoint.npy',p)
        if i%args.render_every==0 or i==last_step:
            measure=metrics(p,p0,rims,pairs,active,edges,rest)
            measure.update(step=i,t_seconds=i*args.dt,stage=stage,guides_active=0 if released else len(guide))
            records.append(measure);hist.append(p.copy())
            arm_hist.append(current_arms)
            robot_hist.append([arr(r.get_qpos()).copy() for r in robots])
            if writer:
                rgb=cams['overview'].render()[0]
                rgb=caption(rgb,[f'IMAGE-ONLY GENESIS RECONSTRUCTION | {stage}',
                    f'Estimated geometry / assisted cloth guides | t={i*args.dt:4.1f}s | bonds {active.sum()}/{len(active)} | guides {0 if released else len(guide)}'])
                writer.append_data(rgb)
                if detail_writer:
                    panels=[Image.fromarray(cams[n].render()[0]).resize((args.width//2,args.width*9//32)) for n in ['front','back']]
                    canvas=Image.new('RGB',(args.width,args.width*9//32+64),(13,21,27))
                    for k,pan in enumerate(panels):canvas.paste(pan,(k*args.width//2,64))
                    detail_writer.append_data(caption(np.asarray(canvas),[
                        'FRONT / BACK | '+stage,
                        f'Genesis PBD + explicit constraints | bonds {active.sum()}/{len(active)} | external guides {0 if released else len(guide)}']))
            if i in [0,int(args.steps*.4),int(args.steps*.7),int(args.steps*.9),last_step]:
                Image.fromarray(cams['front'].render()[0]).save(out/f'stage_{i:04d}.png')
    if writer:writer.close()
    if detail_writer:detail_writer.close()
    for name,cam in cams.items():Image.fromarray(cam.render()[0]).save(out/f'final_{name}.png')
    np.savez_compressed(out/'trajectory.npz',positions=np.asarray(hist),initial=p0,
        seam_pairs=pairs,bonded=active,guide_ids=guide,rim_left=rims[0],rim_right=rims[1],
        faces=faces,edges=edges,arm_segments=np.asarray(arm_hist),robot_qpos=np.asarray(robot_hist),body_radii=BODY_RADII,
        times=np.array([r['t_seconds'] for r in records]))
    report={'reconstruction_type':'assisted image-only estimate','genesis_version':gs.__version__,
        'original_simulation_data_used':False,'inputs':'three supplied PNG images and task description',
        'parameters':vars(args),'material':{'areal_density_kg_m2':.24,'stretch_compliance':2e-8,'bending_compliance':3e-5},
        'estimated_body_radii_m':BODY_RADII.tolist(),
        'scripted_bear_arm_abduction_degrees':ARMS_ABDUCTION_DEG,
        'limitations':['Dimensions and material properties estimated, not identified.',
            'Teddy body fixed rigid; shoulders scripted, no recovered plush mechanics.',
            'Sparse cloth boundary and sleeve rim guides are external manipulation assistance.',
            'Robots use IK replay by default; guide forces are not all transmitted through grippers.',
            'Legacy Genesis PBD rigid coupling lacks tangential friction and continuous cloth collision.',
            'Velcro is a distance-triggered cloth-to-cloth seam approximation, not microscopic hooks.',
            'Visual and centroid checks alone do not prove sleeve threading topology.'],
        'guide_particle_count':len(guide),'total_particles':len(p0),
        'release_duration_seconds':round((args.steps*.08+args.settle_steps)*args.dt,3),
        'robot_IK_position_error_max_m':float(max(ee_errors)),
        'final':records[-1], 'elapsed_seconds':time.time()-start,
        'success_certified':False,
        'success_note':'Inspect final views and release metrics; this run does not certify autonomous physical dressing.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'metrics.json').write_text(json.dumps(records,indent=2))
    (out/'contact_metrics.json').write_text(json.dumps(contact_records,indent=2))
    (out/'strain_metrics.json').write_text(json.dumps(strain_records,indent=2))
    from validate import validate
    validation=validate(out/'trajectory.npz')
    checks={
        'two_arm_axes_in_sleeves':validation.get('both_arm_axes_through_intended_apertures_final',False),
        'seam_bonded':bool(active.all()),
        'torso_coverage_at_095m':validation['torso_coverage'][1]['near_torso_angular_coverage_fraction'],
        'max_edge_stretch':validation['material_edge_stretch']['maximum_ratio'],
        'p95_edge_stretch':validation['material_edge_stretch']['p95_ratio'],
        'torso_inside_vertices':validation['torso_vertex_penetration']['vertices_inside_98_percent_ellipsoid']}
    report['release_test_checks']=checks
    report['geometric_release_test_passed']=bool(checks['two_arm_axes_in_sleeves'] and checks['seam_bonded']
        and checks['torso_coverage_at_095m']>=.8 and checks['max_edge_stretch']<=1.5
        and checks['p95_edge_stretch']<=1.15 and checks['torso_inside_vertices']==0)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
