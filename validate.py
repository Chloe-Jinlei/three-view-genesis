"""Independent geometric diagnostics for the reconstructed dressing rollout.

Uses the ACTUAL post-remesh Genesis material mesh. A centroid near an arm is
never counted as threading. The tests are geometric evidence at saved frames,
not proof of collision-free motion between frames or autonomous manipulation.
Run: python validate.py output/assisted
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def boundary_loops(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    graph = {}
    for a, b in unique[counts == 1]:
        graph.setdefault(int(a), []).append(int(b))
        graph.setdefault(int(b), []).append(int(a))
    if any(len(n) != 2 for n in graph.values()):
        raise ValueError('Material boundary is not a disjoint union of simple edge loops.')
    pending, loops = set(graph), []
    while pending:
        start = min(pending)
        loop, prev, curr = [], None, start
        while True:
            loop.append(curr)
            pending.discard(curr)
            nxt = graph[curr][0] if graph[curr][0] != prev else graph[curr][1]
            prev, curr = curr, nxt
            if curr == start:
                break
            if curr in loop:
                raise ValueError('Invalid repeated boundary vertex.')
        loops.append(np.array(loop, dtype=int))
    return loops, unique, bool(np.all(counts <= 2))


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def signed_area(p):
    return float(0.5 * np.sum(cross2(p, np.roll(p, -1, axis=0))))


def polygon_self_crossings(p):
    n, count = len(p), 0
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = p[j], p[(j + 1) % n]
            o1, o2 = cross2(b-a, c-a), cross2(b-a, d-a)
            o3, o4 = cross2(d-c, a-c), cross2(d-c, b-c)
            if o1*o2 < -1e-18 and o3*o4 < -1e-18:
                count += 1
    return count


def triangulate_polygon(p):
    """Ear clipping, preserving rim order; only valid for a simple polygon."""
    order = list(range(len(p)))
    if signed_area(p) < 0:
        order.reverse()
    faces = []
    attempts = 0
    while len(order) > 3:
        found = False
        for i, ib in enumerate(order):
            ia, ic = order[i-1], order[(i+1) % len(order)]
            a, b, c = p[[ia, ib, ic]]
            if cross2(b-a, c-a) <= 1e-14:
                continue
            others = [j for j in order if j not in (ia, ib, ic)]
            q = p[others]
            # Strict interior so collinear samples along a rim do not block ears.
            inside = ((cross2(b-a, q-a) > 1e-12) & (cross2(c-b, q-b) > 1e-12)
                      & (cross2(a-c, q-c) > 1e-12))
            if not np.any(inside):
                faces.append([ia, ib, ic])
                order.pop(i)
                found = True
                break
        attempts += 1
        if not found or attempts > len(p)*2:
            raise ValueError('Cannot triangulate projected rim; collapsed or non-simple polygon.')
    faces.append(order)
    return np.array(faces, dtype=int)


def segment_triangle_hits(a, b, triangles, tolerance=1e-9):
    """Return unique positions along open segment a->b intersecting triangles."""
    if not len(triangles):
        return np.empty(0)
    d = b-a
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum('ij,ij->i', e1, h)
    valid = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[valid] = 1/det[valid]
    s = a-triangles[:, 0]
    u = inv*np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv*(q@d)
    t = inv*np.einsum('ij,ij->i', e2, q)
    hits = np.sort(t[valid & (u >= -tolerance) & (v >= -tolerance)
                     & (u+v <= 1+tolerance) & (t > tolerance) & (t < 1-tolerance)])
    if not len(hits):
        return hits
    return hits[np.r_[True, np.diff(hits) > 1e-6]]


def sleeve_diagnostic(rim, arm_segment, material_triangles):
    a, b = arm_segment
    center = rim.mean(axis=0)
    _, singular, basis = np.linalg.svd(rim-center, full_matrices=False)
    plane = (rim-center)@basis[:2].T
    normal = basis[2]
    dev = (rim-center)@normal
    area = abs(signed_area(plane))
    crosses = polygon_self_crossings(plane)
    da, db = float((a-center)@normal), float((b-center)@normal)
    result = {
        'rim_vertices': len(rim),
        'rim_projected_area_m2': area,
        'rim_plane_rms_m': float(np.sqrt(np.mean(dev*dev))),
        'rim_plane_max_m': float(np.max(np.abs(dev))),
        'projected_rim_self_crossings': crosses,
        'arm_endpoint_plane_distances_m': [da, db],
        'arm_endpoints_on_opposite_sides': bool(da*db < 0),
        'cap_crossing_count': None,
        'arm_axis_crosses_material_count': int(len(segment_triangle_hits(a, b, material_triangles))),
        'axis_threads_aperture_evidence': False,
    }
    if abs(da-db) > 1e-10:
        fraction = da/(da-db)
        point = a+fraction*(b-a)
        pp = (point-center)@basis[:2].T
        v = plane-pp
        vn = np.roll(v, -1, axis=0)
        winding = np.sum(np.arctan2(cross2(v, vn), np.einsum('ij,ij->i', v, vn)))/(2*np.pi)
        result.update(plane_crossing_segment_fraction=float(fraction), projected_winding=float(winding))
    if crosses == 0 and area > 1e-7:
        try:
            cap_faces = triangulate_polygon(plane)
            hits = segment_triangle_hits(a, b, rim[cap_faces])
            result['cap_crossing_count'] = int(len(hits))
            result['cap_crossing_segment_fractions'] = hits.tolist()
            result['axis_threads_aperture_evidence'] = bool(
                len(hits) == 1 and da*db < 0 and result['arm_axis_crosses_material_count'] == 0
                and abs(result.get('projected_winding', 0)) > 0.9
                and result['rim_plane_max_m'] < 0.020
            )
        except ValueError as exc:
            result['cap_error'] = str(exc)
    # Axis-to-boundary distance is a diagnostic, not proof of finite arm clearance.
    axis = b-a
    q = np.clip(((rim-a)@axis)/(axis@axis), 0, 1)
    result['minimum_rim_vertex_axis_distance_m'] = float(np.min(np.linalg.norm(rim-(a+q[:,None]*axis), axis=1)))
    # The authored arm is a slightly non-circular ellipsoid, with transverse
    # radii 29/28 mm and axial radius half endpoint spacing + 18 mm. This
    # enclosing rotational spheroid needs no unrecorded axial-roll assumption.
    # It can flag possible penetration; it deliberately cannot certify clearance.
    length = float(np.linalg.norm(axis))
    unit = axis/length
    offsets = rim-(a+b)*0.5
    along = offsets@unit
    perpendicular = np.linalg.norm(offsets-along[:,None]*unit,axis=1)
    arm_level = np.sqrt((perpendicular/0.029)**2+(along/(length*0.5+0.018))**2)
    result['rim_vertices_inside_enclosing_arm_spheroid'] = int((arm_level<0.98).sum())
    result['minimum_rim_arm_spheroid_normalized_radius'] = float(arm_level.min())
    result['finite_arm_clearance_certified'] = False
    return result


def cross_section_coverage(vertices, faces, height, center, radii, bins=360):
    """Ray coverage of actual cloth/height-plane intersection near the torso."""
    triangles = vertices[faces]
    d = triangles[:, :, 2]-height
    candidates = triangles[(d.min(axis=1) < 0) & (d.max(axis=1) > 0)]
    segments = []
    for tri in candidates:
        hits = []
        for i, j in ((0,1), (1,2), (2,0)):
            if (tri[i,2]-height)*(tri[j,2]-height) < 0:
                q = (height-tri[i,2])/(tri[j,2]-tri[i,2])
                hits.append((tri[i]+q*(tri[j]-tri[i]))[:2])
        if len(hits) == 2:
            segments.append(hits)
    theta = np.arange(bins)*2*np.pi/bins
    directions = np.c_[np.cos(theta), np.sin(theta)]
    section = np.sqrt(max(0,1-((height-center[2])/radii[2])**2))
    body_radius = section/np.sqrt((directions[:,0]/radii[0])**2+(directions[:,1]/radii[1])**2)
    covered = np.zeros(bins, dtype=bool)
    for a, b in segments:
        e, offset = b-a, a-center[:2]
        den = cross2(directions, e)
        ok = np.abs(den) > 1e-12
        t = np.divide(cross2(offset, e), den, out=np.zeros_like(den), where=ok)
        u = np.divide(cross2(offset, directions), den, out=np.zeros_like(den), where=ok)
        covered |= ok & (u>=0) & (u<=1) & (t>=0.85*body_radius) & (t<=body_radius+0.05)
    longest, current = 0, 0
    for x in np.r_[covered, covered]:
        current = 0 if x else current+1
        longest = max(longest, current)
    front = directions[:,1] < 0
    return {
        'height_m': height, 'near_torso_angular_coverage_fraction': float(covered.mean()),
        'front_coverage_fraction': float(covered[front].mean()),
        'back_coverage_fraction': float(covered[~front].mean()),
        'largest_uncovered_angle_deg': float(min(longest,bins)*360/bins),
        'cloth_slice_segments': len(segments),
    }


def validate(path, output=None):
    path = Path(path)
    if path.is_dir():
        path = path/'trajectory.npz'
    data = np.load(path, allow_pickle=False)
    p0, history = data['initial'], data['positions']
    final = history[-1]
    report = {
        'input': str(path.resolve()), 'recorded_frames': len(history),
        'all_positions_finite': bool(np.isfinite(history).all()),
        'full_physical_task_success_certified': False,
        'limitations': [
            'Sampled-frame geometric tests cannot prove collision-free continuous motion.',
            'An arm-axis cap crossing does not by itself prove clearance for the finite-thickness arm.',
            'No force, grasp feasibility, material identification, or autonomous policy certification is performed.',
            'Material topology is taken from the saved post-remesh mesh, never from proximity-selected rim particles.',
        ],
    }
    if 'seam_pairs' in data:
        pairs = data['seam_pairs']
        distances = np.linalg.norm(final[pairs[:,0]]-final[pairs[:,1]], axis=1)
        report['seam'] = {'candidate_pairs':len(pairs), 'bonded_pairs':int(data['bonded'].sum()),
                          'mean_distance_m':float(distances.mean()), 'max_distance_m':float(distances.max()),
                          'pairs_within_20mm':int((distances<0.020).sum())}
    if 'faces' not in data:
        report['topology_available'] = False
        report['cannot_validate_reason'] = 'Missing faces from cloth.mesh.faces; ordered material rims cannot be established.'
    else:
        faces = data['faces'].astype(int)
        loops, edges, manifold = boundary_loops(faces)
        areas = [abs(signed_area(p0[loop,:2])) for loop in loops]
        outer = int(np.argmax(areas))
        holes = sorted([loop for i,loop in enumerate(loops) if i != outer], key=lambda loop:p0[loop,0].mean())
        used = np.unique(faces)
        report['material_topology'] = {'boundary_loops':len(loops), 'expected_boundary_loops':3,
            'edge_manifold':manifold, 'euler_characteristic':int(len(used)-len(edges)+len(faces)),
            'hole_rim_particle_ids':[loop.tolist() for loop in holes],
            'initial_loop_projected_areas_m2':areas,
            'topology_matches_two_aperture_sheet':bool(len(loops)==3 and manifold and len(used)-len(edges)+len(faces)==-1)}
        true_edges = data['edges'].astype(int) if 'edges' in data else edges
        rest = np.linalg.norm(p0[true_edges[:,1]]-p0[true_edges[:,0]],axis=1)
        ratios = np.linalg.norm(final[true_edges[:,1]]-final[true_edges[:,0]],axis=1)/np.maximum(rest,1e-12)
        report['material_edge_stretch'] = {'minimum_ratio':float(ratios.min()), 'p05_ratio':float(np.quantile(ratios,.05)),
            'median_ratio':float(np.median(ratios)), 'p95_ratio':float(np.quantile(ratios,.95)),
            'maximum_ratio':float(ratios.max()), 'edges_over_20_percent_extension':int((ratios>1.2).sum())}
        center = np.array([0.,.10,.965])
        radii = data['body_radii'] if 'body_radii' in data else np.array([.112,.087,.168])
        report['torso_coverage'] = [cross_section_coverage(final,faces,z,center,radii) for z in [.90,.95,1.00,1.04]]
        normalized = np.linalg.norm((final-center)/radii,axis=1)
        report['torso_vertex_penetration'] = {'vertices_inside_98_percent_ellipsoid':int((normalized<.98).sum()),
            'deepest_normalized_radius':float(normalized.min()), 'proxy':'analytic estimated torso ellipsoid'}
        if 'arm_segments' in data and len(holes)==2:
            segments = data['arm_segments']
            if segments.shape != (len(history),2,2,3):
                raise ValueError(f'arm_segments has {segments.shape}; expected {(len(history),2,2,3)}')
            frames = []
            for i, p in enumerate(history):
                sleeves = [sleeve_diagnostic(p[holes[k]], segments[i,k],p[faces]) for k in range(2)]
                frames.append({'frame':i,'time_s':float(data['times'][i]),'sleeves':sleeves})
            report['sleeves_final'] = frames[-1]['sleeves']
            report['sleeves_by_frame'] = frames
            report['sampled_frames_with_axis_material_crossing'] = [f['frame'] for f in frames
                if any(s['arm_axis_crosses_material_count'] for s in f['sleeves'])]
            report['both_arm_axes_through_intended_apertures_final'] = all(s['axis_threads_aperture_evidence'] for s in frames[-1]['sleeves'])
        else:
            report['sleeve_check_unavailable'] = 'Need two material hole loops and recorded arm_segments with shape (T,2,2,3).'
    output = Path(output) if output else path.with_name('validation.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    result = validate(args.trajectory,args.out)
    compact = {k:v for k,v in result.items() if k not in ('sleeves_by_frame','material_topology')}
    if 'material_topology' in result:
        compact['material_topology'] = {k:v for k,v in result['material_topology'].items() if k!='hole_rim_particle_ids'}
    print(json.dumps(compact,indent=2))
