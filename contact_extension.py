"""Optional analytic particle contact extension for the image-only reconstruction.

This supplements, rather than replaces or modifies, Genesis PBD. It uses static
estimated torso/arm ellipsoids, particle-segment collision checks, and a bounded
Coulomb velocity correction. It is not a triangle CCD solver, plush simulation,
or proof of sleeve threading. It never attaches particles to world coordinates.

The inflation is conservative for spherical particles: scaling every semiaxis
by 1 + particle_radius/min(semiaxes) contains the ellipsoid's Minkowski sum with
a particle-radius sphere. This intentionally overestimates some clearances.
Arm poses are taken from the passed manifest; pass updated shoulder/wrist
positions if the bear articulates. Motion of the colliders during dt is omitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def _align_z(axis):
    z = np.asarray(axis, dtype=float)
    z /= np.linalg.norm(z)
    if z[2] < -1 + 1e-10:
        return np.diag([1., -1., -1.])
    k = np.array([[0., 0., z[0]], [0., 0., z[1]], [-z[0], -z[1], 0.]])
    # Rodrigues: cross([0,0,1], z) = [-z_y,z_x,0].
    return np.eye(3) + k + k @ k / (1 + z[2])


def build_proxies(bear_manifest, particle_radius=0.006):
    """Return analytic collision proxies from our generated bear manifest."""
    if isinstance(bear_manifest, (str, Path)):
        bear_manifest = json.loads(Path(bear_manifest).read_text())
    if particle_radius < 0:
        raise ValueError('particle_radius must be nonnegative')
    proxies = []
    for part in bear_manifest['parts']:
        if part['name'] == 'body':
            center = np.asarray(part.get('pos', bear_manifest['body_center']), dtype=float)
            radii = np.asarray(part.get('radii', bear_manifest['body_radii']), dtype=float)
            rotation = np.eye(3)
        elif part['name'].startswith('arm_'):
            shoulder = np.asarray(part['shoulder_world'], dtype=float)
            wrist = np.asarray(part['wrist_world'], dtype=float)
            axis = wrist-shoulder
            length = np.linalg.norm(axis)
            if length <= 1e-10:
                raise ValueError('Arm shoulder and wrist must differ.')
            center = (shoulder+wrist)/2
            # Same estimated parameters as assets_builder.py, explicitly exposed
            # in the returned proxy instead of inferred from original data.
            radii = np.asarray(part.get('contact_radii', [.029, .028, length/2+.018]), dtype=float)
            rotation = _align_z(axis/length)
        else:
            continue
        if np.any(radii <= 0):
            raise ValueError('Ellipsoid radii must be positive.')
        proxies.append({'name': part['name'], 'center': center,
                        'radii': radii, 'expanded_radii': radii*(1+particle_radius/radii.min()),
                        'rotation': rotation})
    return proxies


def _normal_at(local, axes, rotation):
    normal = local / (axes*axes)
    lengths = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(lengths[:,None], 1e-15)
    return normal @ rotation.T


def apply_contact_extension(prev_positions, new_positions, velocities, dt,
                            bear_manifest, *, free_mask=None, mu=0.6,
                            particle_radius=0.006, iterations=3,
                            gravity=(0.,0.,-9.81), proxies=None):
    """Return (corrected_positions, corrected_velocities, metrics).

    Inputs and outputs are N-by-3 world arrays. ``free_mask=False`` particles are
    untouched. For a static surface, incoming normal speed is removed. Tangent
    velocity reduction is at most mu times the estimated normal impulse/mass.
    Its budget is the maximum of incoming normal speed, contact-position
    correction/dt, and the supporting component of gravity*dt. This estimate is
    explicit because the native solver does not expose its cloth contact force.

    No hidden adhesion is used. A particle with no contact is unchanged. The
    function does not prevent triangles passing between sampled particles and
    cannot restore an already incorrect cloth/arm topology.
    """
    previous = np.asarray(prev_positions, dtype=float)
    original = np.asarray(new_positions, dtype=float)
    speed = np.asarray(velocities, dtype=float)
    if previous.shape != original.shape or speed.shape != original.shape or original.ndim != 2 or original.shape[1] != 3:
        raise ValueError('All position/velocity inputs must have identical (N,3) shape.')
    if not all(np.isfinite(a).all() for a in (previous, original, speed)):
        raise ValueError('Contact inputs must be finite.')
    if dt <= 0 or mu < 0 or iterations < 1:
        raise ValueError('dt/iterations must be positive; mu must be nonnegative.')
    n_particles = len(original)
    free = np.ones(n_particles, dtype=bool) if free_mask is None else np.asarray(free_mask,dtype=bool)
    if free.shape != (n_particles,):
        raise ValueError('free_mask must have shape (N,).')
    if proxies is None:
        proxies = build_proxies(bear_manifest,particle_radius)
    p,v = original.copy(),speed.copy()
    gravity = np.asarray(gravity,dtype=float)
    contacts = [np.zeros(n_particles,dtype=bool) for _ in proxies]
    normals = [np.zeros_like(p) for _ in proxies]
    budgets = [np.zeros(n_particles) for _ in proxies]
    sweep_counts = [np.zeros(n_particles,dtype=bool) for _ in proxies]
    inside_counts = [np.zeros(n_particles,dtype=bool) for _ in proxies]
    epsilon = 1e-7
    for iteration in range(iterations):
        for k,proxy in enumerate(proxies):
            center,axes,R = proxy['center'],proxy['expanded_radii'],proxy['rotation']
            old = (previous-center) @ R
            new = (p-center) @ R
            x0,x1 = old/axes,new/axes
            delta = x1-x0
            aa = np.sum(delta*delta,axis=1)
            bb = 2*np.sum(x0*delta,axis=1)
            cc = np.sum(x0*x0,axis=1)-1
            discriminant = bb*bb-4*aa*cc
            entry = (-bb-np.sqrt(np.maximum(discriminant,0)))/(2*np.maximum(aa,1e-30))
            # Segment entry works even when both endpoints lie outside on
            # opposite sides. Initially penetrating particles are handled below.
            swept = free & (cc >= -1e-7) & (aa>1e-18) & (discriminant>=0) & (entry>=-1e-8) & (entry<=1)
            swept &= bb < 0
            if np.any(swept):
                ids=np.flatnonzero(swept)
                contact_local=old[ids]+np.clip(entry[ids,None],0,1)*(new[ids]-old[ids])
                normal=_normal_at(contact_local,axes,R)
                hit=contact_local @ R.T+center
                remaining=p[ids]-hit
                inward=np.minimum(np.sum(remaining*normal,axis=1),0)
                corrected=hit+remaining-inward[:,None]*normal+epsilon*normal
                normal_distance=np.maximum(np.sum((corrected-p[ids])*normal,axis=1),0)
                p[ids]=corrected
                contacts[k][ids]=True
                normals[k][ids]=normal
                budgets[k][ids]=np.maximum(budgets[k][ids],normal_distance/dt)
                sweep_counts[k][ids]=True
            # Resolve initial overlap, or residual overlap caused by another
            # proxy. Radial projection is conservative and remains unanchored.
            local=(p-center) @ R
            scaled=local/axes
            radius=np.linalg.norm(scaled,axis=1)
            inside=free & (radius<1-1e-8)
            if np.any(inside):
                ids=np.flatnonzero(inside)
                radial=scaled[ids]/np.maximum(radius[ids,None],1e-15)
                zero=radius[ids]<1e-12
                if np.any(zero):
                    radial[zero]=0
                    radial[zero,np.argmin(axes)]=1
                surface_local=radial*axes
                normal=_normal_at(surface_local,axes,R)
                corrected=surface_local @ R.T+center+epsilon*normal
                normal_distance=np.maximum(np.sum((corrected-p[ids])*normal,axis=1),0)
                p[ids]=corrected
                contacts[k][ids]=True
                normals[k][ids]=normal
                budgets[k][ids]=np.maximum(budgets[k][ids],normal_distance/dt)
                inside_counts[k][ids]=True
            # Persistent nearby contact can support gravity without any position
            # anchor. The test is relative to the conservative expanded proxy.
            local=(p-center) @ R
            radial=np.linalg.norm(local/axes,axis=1)
            nearby=free & (radial>=1-1e-7) & (radial<=1.001)
            if np.any(nearby):
                ids=np.flatnonzero(nearby)
                contacts[k][ids]=True
                normals[k][ids]=_normal_at(local[ids],axes,R)

    friction_particles=np.zeros(n_particles,dtype=bool)
    speed_reduction=0.
    normal_budget_sum=0.
    for active,normal,budget in zip(contacts,normals,budgets):
        if not np.any(active):
            continue
        ids=np.flatnonzero(active)
        nn=normal[ids]
        vn=np.sum(v[ids]*nn,axis=1)
        incoming=np.maximum(-vn,0)
        support=np.maximum(-(nn @ gravity)*dt,0)
        budget=np.maximum.reduce([budget[ids],incoming,support])
        tangent=v[ids]-vn[:,None]*nn
        tangent_speed=np.linalg.norm(tangent,axis=1)
        reduction=np.minimum(tangent_speed,mu*budget)
        tangent*=np.maximum(0,1-reduction/np.maximum(tangent_speed,1e-15))[:,None]
        v[ids]=tangent+np.maximum(vn,0)[:,None]*nn
        friction_particles[ids[reduction>1e-12]]=True
        speed_reduction+=float(reduction.sum())
        normal_budget_sum+=float(budget.sum())
    corrections=np.linalg.norm(p-original,axis=1)
    contact_union=np.logical_or.reduce(contacts) if contacts else np.zeros(n_particles,dtype=bool)
    remaining_inside={}
    for proxy in proxies:
        local=(p-proxy['center']) @ proxy['rotation']
        inside=np.linalg.norm(local/proxy['expanded_radii'],axis=1)<1-1e-5
        remaining_inside[proxy['name']]=int(np.count_nonzero(inside & free))
    metrics={
        'model':'custom static ellipsoid particle CCD + Coulomb velocity friction',
        'free_particles':int(free.sum()),
        'contact_particles':int(contact_union.sum()),
        'corrected_particles':int(np.count_nonzero(corrections>1e-9)),
        'swept_particle_proxy_events':sum(int(a.sum()) for a in sweep_counts),
        'overlap_particle_proxy_events':sum(int(a.sum()) for a in inside_counts),
        'friction_particles':int(friction_particles.sum()),
        'max_position_correction_m':float(corrections.max(initial=0)),
        'sum_position_correction_m':float(corrections.sum()),
        'sum_tangent_speed_removed_mps':speed_reduction,
        'sum_normal_impulse_per_mass_mps':normal_budget_sum,
        'remaining_inside_by_proxy':remaining_inside,
        'mu':float(mu), 'particle_radius_m':float(particle_radius),
        'changes_fixed_particles':bool(np.any(p[~free]!=original[~free]) or np.any(v[~free]!=speed[~free])),
    }
    return p,v,metrics


# Short alias for optional controller integrations.
correct_contact = apply_contact_extension
