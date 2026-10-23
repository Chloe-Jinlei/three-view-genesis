"""Independent finite-arm diagnostics; never replaces the strict validator.

The authored shoulder/wrist markers are interior points of the arm ellipsoid.
This additionally tests its complete principal-axis segment, with poles found
by exact line/triangle intersection on the authored mesh, and final finite
mesh/sheet intersection. Recorded full affine transforms are preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from validate import boundary_loops, signed_area, segment_triangle_hits, sleeve_diagnostic


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def principal_poles(mesh, shoulder_local, wrist_local):
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(vertices-center, full_matrices=False)
    axis = basis[0]
    axis *= np.sign(axis @ (wrist_local-shoulder_local))
    span = 2*np.max(np.abs((vertices-center)@axis))
    a, b = center-span*axis, center+span*axis
    hits = segment_triangle_hits(a, b, np.asarray(mesh.triangles))
    if len(hits) != 2:
        raise ValueError(f'Expected two physical mesh poles; found {len(hits)}.')
    return np.stack([a+hits[0]*(b-a), a+hits[1]*(b-a)])


def cap_diagnostic(rim, segment, material):
    result = sleeve_diagnostic(rim, segment, material)
    # The original helper's enclosing spheroid assumes shoulder/wrist markers
    # and adds 18 mm. It must not be applied again to already complete poles.
    for key in ('rim_vertices_inside_enclosing_arm_spheroid',
                'minimum_rim_arm_spheroid_normalized_radius',
                'finite_arm_clearance_certified'):
        result.pop(key, None)
    return result


def paired_segment_triangle_intersections(a, b, triangles, tolerance=1e-9):
    """Non-coplanar closed-segment/triangle tests for equally sized arrays."""
    d = b-a
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    h = np.cross(d, e2)
    determinant = np.einsum('ij,ij->i', e1, h)
    valid = np.abs(determinant) > 1e-12
    inv = np.zeros(len(determinant))
    inv[valid] = 1/determinant[valid]
    s = a-triangles[:, 0]
    u = inv*np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv*np.einsum('ij,ij->i', d, q)
    t = inv*np.einsum('ij,ij->i', e2, q)
    return (valid & (u >= -tolerance) & (v >= -tolerance)
            & (u+v <= 1+tolerance) & (t >= -tolerance) & (t <= 1+tolerance))


def triangle_surface_intersections(arm_triangles, cloth_triangles):
    """AABB broad phase followed by all six non-coplanar edge/face tests."""
    amin, amax = arm_triangles.min(axis=1), arm_triangles.max(axis=1)
    bmin, bmax = cloth_triangles.min(axis=1), cloth_triangles.max(axis=1)
    intersecting = []
    candidates = 0
    for start in range(0, len(arm_triangles), 128):
        stop = min(start+128, len(arm_triangles))
        overlap = np.all((amin[start:stop, None] <= bmax[None]+1e-9)
                         & (amax[start:stop, None] >= bmin[None]-1e-9), axis=2)
        ia, ib = np.nonzero(overlap)
        ia += start
        candidates += len(ia)
        if not len(ia):
            continue
        aa, bb = arm_triangles[ia], cloth_triangles[ib]
        hit = np.zeros(len(ia), dtype=bool)
        for edge in range(3):
            nxt = (edge+1) % 3
            hit |= paired_segment_triangle_intersections(aa[:, edge], aa[:, nxt], bb)
            hit |= paired_segment_triangle_intersections(bb[:, edge], bb[:, nxt], aa)
        intersecting.extend(np.stack([ia[hit], ib[hit]], axis=1).tolist())
    return {'broad_phase_candidate_pairs': candidates,
            'noncoplanar_intersecting_triangle_pairs': len(intersecting),
            'intersection_pair_examples': intersecting[:20],
            'coplanar_overlap_tested': False}


def convex_inside_vertices(mesh, world_points, matrix):
    local = transform_points(world_points, np.linalg.inv(matrix))
    lo, hi = mesh.bounds
    candidates = np.flatnonzero(np.all((local >= lo) & (local <= hi), axis=1))
    if not len(candidates):
        return []
    normals = np.asarray(mesh.face_normals)
    origins = np.asarray(mesh.triangles)[:, 0]
    distances = local[candidates]@normals.T-np.einsum('ij,ij->i', origins, normals)
    return candidates[np.max(distances, axis=1) < -1e-6].tolist()


def validate_finite_arms(run_directory, output=None):
    root = Path(run_directory).resolve()
    data = np.load(root/'trajectory.npz')
    if str(data['arm_transform_frame']) != 'authored_mesh':
        raise ValueError('Arm transforms must explicitly be in the authored mesh frame.')
    manifest = json.loads((root/'assets/bear_manifest.json').read_text())
    parts = {p['name']: p for p in manifest['parts']}
    loops, _, manifold = boundary_loops(data['faces'])
    outer = np.argmax([abs(signed_area(data['initial'][r, :2])) for r in loops])
    holes = sorted([r for i, r in enumerate(loops) if i != outer],
                   key=lambda r: data['initial'][r, 0].mean())
    if not manifold or len(holes) != 2:
        raise ValueError('Need exactly two true material aperture loops.')
    final = data['positions'][-1]
    material = final[data['faces']]
    report = {
        'input': str(root/'trajectory.npz'),
        'time_s': float(data['times'][-1]),
        'strict_validator_modified': False,
        'method': 'Exact authored arm-mesh principal-axis poles, transformed by recorded full affine matrices; original ordered material aperture caps.',
        'arms': [],
        'limitations': [
            'A complete-axis crossing and a shoulder-to-wrist crossing are different diagnostics; neither replaces the other.',
            'Final finite-surface checks use the authored 1280-triangle arm meshes; the solver may use decimated collision meshes.',
            'The final surface test includes noncoplanar triangle intersections and convex containment, not coplanar-overlap or positive-clearance certification.',
            'Sampled release-frame cap tests cannot independently certify collision-free continuous motion.',
        ],
    }
    specifications = []
    for k, side in enumerate(('left', 'right')):
        part = parts['arm_'+side]
        mesh = trimesh.load(root/'assets'/part['mesh'], process=False, force='mesh')
        shoulder = np.asarray(part['shoulder_local'])
        wrist = np.asarray(part['wrist_local'])
        poles = principal_poles(mesh, shoulder, wrist)
        specifications.append((mesh, poles))
        matrix = data['arm_transforms'][-1, k]
        complete = transform_points(poles, matrix)
        strict = data['arm_segments'][-1, k]
        strict_diagnostic = cap_diagnostic(final[holes[k]], strict, material)
        full_diagnostic = cap_diagnostic(final[holes[k]], complete, material)
        surface = triangle_surface_intersections(
            transform_points(np.asarray(mesh.vertices), matrix)[np.asarray(mesh.faces)], material)
        inside = convex_inside_vertices(mesh, final, matrix)
        surface.update(cloth_vertices_strictly_inside_authored_arm=len(inside),
                       inside_vertex_examples=inside[:20],
                       authored_mesh_is_convex=bool(mesh.is_convex))
        item = {'side': side, 'mesh': str(root/'assets'/part['mesh']),
                'mesh_vertices': len(mesh.vertices), 'mesh_triangles': len(mesh.faces),
                'authored_shoulder_to_wrist_local': [shoulder.tolist(), wrist.tolist()],
                'complete_mesh_axis_poles_local': poles.tolist(),
                'mesh_extension_past_shoulder_and_wrist_m': np.linalg.norm(poles-np.stack([shoulder, wrist]), axis=1).tolist(),
                'recorded_affine_transform': matrix.tolist(),
                'strict_shoulder_to_wrist': strict_diagnostic,
                'complete_finite_mesh_axis': full_diagnostic,
                'final_finite_mesh_surface': surface}
        hits = full_diagnostic.get('cap_crossing_segment_fractions', [])
        if len(hits) == 1:
            cap_point = complete[0]+hits[0]*(complete[1]-complete[0])
            direction = strict[1]-strict[0]
            fraction = float((cap_point-strict[0])@direction/(direction@direction))
            item['cap_crossing_point_world'] = cap_point.tolist()
            item['cap_crossing_fraction_on_strict_axis'] = fraction
            item['cap_crossing_distance_from_shoulder_toward_wrist_m'] = float(fraction*np.linalg.norm(direction))
        report['arms'].append(item)
    run_report = json.loads((root/'report.json').read_text())
    release_start = float(data['times'][-1])-float(run_report.get('released_duration_s', 0))
    release = []
    for frame in np.flatnonzero(data['times'] >= release_start-1e-8):
        vertices = data['positions'][frame]
        triangles = vertices[data['faces']]
        records = []
        for k, (_, poles) in enumerate(specifications):
            segment = transform_points(poles, data['arm_transforms'][frame, k])
            diagnostic = cap_diagnostic(vertices[holes[k]], segment, triangles)
            records.append({key: diagnostic.get(key) for key in (
                'cap_crossing_count', 'arm_axis_crosses_material_count',
                'axis_threads_aperture_evidence', 'projected_rim_self_crossings')})
        release.append({'frame': int(frame), 'time_s': float(data['times'][frame]), 'arms': records})
    report['strict_both_shoulder_to_wrist_axes_thread_final'] = all(
        a['strict_shoulder_to_wrist']['axis_threads_aperture_evidence'] for a in report['arms'])
    report['both_complete_physical_mesh_axes_thread_final'] = all(
        a['complete_finite_mesh_axis']['axis_threads_aperture_evidence'] for a in report['arms'])
    report['final_surface_nonintersection_evidence_both_arms'] = all(
        a['final_finite_mesh_surface']['noncoplanar_intersecting_triangle_pairs'] == 0
        and a['final_finite_mesh_surface']['cloth_vertices_strictly_inside_authored_arm'] == 0
        for a in report['arms'])
    report['sampled_release_frames'] = release
    report['both_complete_axes_thread_all_sampled_release_frames'] = all(
        all(a['axis_threads_aperture_evidence'] for a in f['arms']) for f in release)
    report['full_physical_task_success_certified'] = False
    output = Path(output) if output else root/'finite_arm_validation.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = validate_finite_arms(args.run_directory, args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('arms', 'sampled_release_frames', 'limitations')}, indent=2))
