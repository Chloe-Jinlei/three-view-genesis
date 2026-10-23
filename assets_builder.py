"""Procedural teddy reconstruction from the three supplied initial-state images.

All dimensions are estimates, in meters.  The teddy faces -Y, and the table is
at Z=0.75.  This file deliberately contains no simulation or controller code.
Each exported OBJ is local to ``pos`` in the JSON manifest, with unit scale.
Run this file directly to regenerate its assets, or call build_assets(out_dir).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from parameters import BODY_RADII


PALETTE = {
    "fur": (0.39, 0.225, 0.105, 1.0),
    "head": (0.43, 0.254, 0.124, 1.0),
    "ears": (0.46, 0.279, 0.141, 1.0),
    "inner_ear": (0.57, 0.374, 0.208, 1.0),
    "tan": (0.65, 0.579, 0.392, 1.0),
    "black": (0.025, 0.023, 0.019, 1.0),
    "highlight": (0.80, 0.78, 0.70, 1.0),
}


def _ellipsoid(radii, subdivisions=3):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.vertices *= np.asarray(radii, dtype=float)
    return mesh


def _tube_path(points, radius, sections=10):
    """Short, closed cylindrical pieces for the smile, in local coordinates."""
    pieces = []
    for start, end in zip(points[:-1], points[1:]):
        start, end = np.asarray(start), np.asarray(end)
        axis = end - start
        tube = trimesh.creation.cylinder(radius=radius, height=np.linalg.norm(axis), sections=sections)
        matrix = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), axis)
        matrix[:3, 3] = (start + end) * 0.5
        tube.apply_transform(matrix)
        pieces.append(tube)
    return trimesh.util.concatenate(pieces)


def _front_patch(body_radii, patch_radii, offset=0.0014, radial_steps=12, angular_steps=64):
    """Thin closed patch following the front of a curved ellipsoid."""
    bx, by, bz = body_radii
    px, pz = patch_radii
    front = [[0.0, -by - offset, 0.0]]
    for ri in range(1, radial_steps + 1):
        r = ri / radial_steps
        for ai in range(angular_steps):
            angle = ai * 2.0 * np.pi / angular_steps
            x, z = px * r * np.cos(angle), pz * r * np.sin(angle)
            y = -by * np.sqrt(max(0.001, 1.0 - (x / bx) ** 2 - (z / bz) ** 2)) - offset
            front.append([x, y, z])
    front = np.asarray(front)
    faces = []
    for ai in range(angular_steps):
        faces.append([0, 1 + ai, 1 + (ai + 1) % angular_steps])
    for ri in range(radial_steps - 1):
        inner, outer = 1 + ri * angular_steps, 1 + (ri + 1) * angular_steps
        for ai in range(angular_steps):
            an = (ai + 1) % angular_steps
            faces.extend([[inner + ai, outer + ai, outer + an], [inner + ai, outer + an, inner + an]])
    # A back surface 0.8 mm beneath the visible front makes a closed manifold.
    back = front.copy()
    back[:, 1] += 0.0008
    count = len(front)
    all_faces = list(faces) + [[c + count, b + count, a + count] for a, b, c in faces]
    edge = 1 + (radial_steps - 1) * angular_steps
    for ai in range(angular_steps):
        a, b = edge + ai, edge + (ai + 1) % angular_steps
        all_faces.extend([[a, b, b + count], [a, b + count, a + count]])
    mesh = trimesh.Trimesh(vertices=np.vstack([front, back]), faces=all_faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def build_assets(out_dir):
    """Create bear part OBJ files and bear_manifest.json; return the manifest.

    ``collision`` distinguishes structural solids from cosmetic face patches.
    Arm meshes use midpoint pivots; the shoulder and wrist locations are also
    recorded in world and local coordinates for downstream kinematic control.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []

    def add(name, mesh, pos, color, collision=True, **extra):
        filename = f"bear_{name}.obj"
        mesh.export(out_dir / filename)
        part = {
            "name": name,
            "mesh": filename,
            "pos": [float(v) for v in pos],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "scale": 1.0,
            "color": list(PALETTE.get(color, color)),
            "collision": collision,
            **extra,
        }
        parts.append(part)
        return part

    body_pos = np.array([0.0, 0.10, 0.965])
    body_radii = tuple(BODY_RADII)
    add("body", _ellipsoid(body_radii), body_pos, "fur", radii=list(body_radii))
    add("belly", _front_patch(body_radii, (0.077, 0.116)), body_pos, "tan", collision=False)
    add("head", _ellipsoid((0.098, 0.084, 0.092)), (0.0, 0.082, 1.17), "head")

    for side, sign in (("left", -1), ("right", 1)):
        add(f"ear_{side}", _ellipsoid((0.039, 0.030, 0.039), 2), (sign * 0.078, 0.082, 1.245), "ears")
        add(f"inner_ear_{side}", _ellipsoid((0.026, 0.006, 0.026), 2), (sign * 0.078, 0.054, 1.245), "inner_ear", collision=False)
        add(f"foot_{side}", _ellipsoid((0.048, 0.058, 0.037)), (sign * 0.060, 0.065, 0.788), "fur")
        shoulder = np.array([sign * 0.105, 0.095, 1.065])
        wrist = np.array([sign * 0.160, 0.070, 0.985])
        center = (shoulder + wrist) * 0.5
        axis = wrist - shoulder
        arm = _ellipsoid((0.029, 0.028, np.linalg.norm(axis) * 0.5 + 0.018))
        arm.apply_transform(trimesh.geometry.align_vectors((0.0, 0.0, 1.0), axis))
        add(
            f"arm_{side}", arm, center, "head", animated=True,
            shoulder_world=shoulder.tolist(), wrist_world=wrist.tolist(),
            shoulder_local=(shoulder - center).tolist(), wrist_local=(wrist - center).tolist(),
        )

    # Broad short muzzle with tiny muzzle details: readable at scene-camera size.
    add("muzzle", _ellipsoid((0.043, 0.021, 0.029)), (0.0, -0.003, 1.137), "tan", collision=False)
    for side, sign in (("left", -1), ("right", 1)):
        add(f"eye_{side}", _ellipsoid((0.0050, 0.0038, 0.0062), 2), (sign * 0.029, -0.0005, 1.175), "black", collision=False)
        add(f"eye_glint_{side}", _ellipsoid((0.0012, 0.0007, 0.0014), 1), (sign * 0.029 - 0.001, -0.0040, 1.177), "highlight", collision=False)

    # A rounded upside-down triangular nose.
    nose_mesh = trimesh.Trimesh(
        vertices=[[-0.008, -0.002, 0.004], [0.008, -0.002, 0.004], [0.0, -0.002, -0.006],
                  [-0.007, 0.003, 0.003], [0.007, 0.003, 0.003], [0.0, 0.003, -0.005]],
        faces=[[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]],
        process=True,
    )
    trimesh.repair.fix_normals(nose_mesh)
    add("nose", nose_mesh, (0.0, -0.024, 1.146), "black", collision=False)
    mouth_origin = np.array([0.0, -0.0236, 1.137])
    mouth_paths = [
        [[0.0, 0.0, 0.004], [0.0, 0.0, -0.004]],
        [[-0.012, 0.002, -0.005], [-0.009, 0.0006, -0.009], [-0.004, 0.0, -0.0095], [0.0, 0.0, -0.004]],
        [[0.012, 0.002, -0.005], [0.009, 0.0006, -0.009], [0.004, 0.0, -0.0095], [0.0, 0.0, -0.004]],
    ]
    mouth = trimesh.util.concatenate([_tube_path(p, 0.00085) for p in mouth_paths])
    add("mouth", mouth, mouth_origin, "black", collision=False)

    manifest = {
        "description": "Procedural teddy estimate based only on the supplied three initial-state images.",
        "units": "meters",
        "front": "-Y",
        "table_top_z": 0.75,
        "body_center": body_pos.tolist(),
        "body_radii": list(body_radii),
        "estimated": True,
        "parts": parts,
    }
    (out_dir / "bear_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "assets")
    args = parser.parse_args()
    manifest = build_assets(args.out)
    print(f"Wrote {len(manifest['parts'])} bear mesh parts to {args.out.resolve()}")
