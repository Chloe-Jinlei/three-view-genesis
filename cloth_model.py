"""Procedural garment inferred from the three supplied initial-state images.

No original simulation assets or parameters are used. OBJ coordinates are world
coordinates: (u, -0.25 + v, 0.758). The two apertures are genuine mesh boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


LENGTH = 0.80
WIDTH = 0.19
ORIGIN = np.array([0.0, -0.25, 0.758])
HOLE_CENTERS = (-0.205, 0.205)
HOLE_V = 0.045
HOLE_HALF_WIDTH = 0.035
HOLE_HALF_HEIGHT = 0.038
STRIP_WIDTH = 0.025


def _subdivide(breaks, spacing=0.008):
    values = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        values.extend(np.linspace(lo, hi, int(np.ceil((hi-lo)/spacing))+1)[:-1])
    return np.array(values + [breaks[-1]], dtype=np.float64)


def _geometry():
    xs = _subdivide(sorted([-LENGTH/2, LENGTH/2] +
                           [c+d for c in HOLE_CENTERS for d in (-HOLE_HALF_WIDTH, HOLE_HALF_WIDTH)]))
    ys = _subdivide([-WIDTH/2, HOLE_V-HOLE_HALF_HEIGHT,
                     HOLE_V+HOLE_HALF_HEIGHT, WIDTH/2])
    uv_world = np.array([(x, y) for y in ys for x in xs])
    triangles = []
    nx = len(xs)
    for j in range(len(ys)-1):
        for i in range(nx-1):
            mx, my = (xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2
            inside_hole = any(abs(mx-c) < HOLE_HALF_WIDTH-1e-10 and
                              abs(my-HOLE_V) < HOLE_HALF_HEIGHT-1e-10
                              for c in HOLE_CENTERS)
            if inside_hole:
                continue
            a = j*nx+i
            b, c, d = a+1, a+nx+1, a+nx
            triangles.extend(((a,b,c),(a,c,d)) if (i+j)%2 == 0 else
                             ((a,b,d),(b,c,d)))
    faces = np.asarray(triangles, dtype=np.int64)
    used = np.unique(faces)
    remap = np.full(len(uv_world), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    uv_world = uv_world[used]
    faces = remap[faces]
    vertices = np.column_stack((uv_world, np.zeros(len(uv_world)))) + ORIGIN
    texcoords = (uv_world + [LENGTH/2, WIDTH/2]) / [LENGTH, WIDTH]
    return vertices, faces, texcoords


def _texture(path):
    """A deterministic woven texture, with hook/loop strips and seam stitching."""
    w, h = 2048, 512
    yy, xx = np.mgrid[:h, :w]
    rng = np.random.default_rng(23)
    weave = (1.6*np.sin(xx*np.pi/2) + 1.3*np.cos(yy*np.pi/2) +
             rng.normal(0, 0.7, (h,w)))
    rgb = np.clip(np.array([197.,190.,163.])[None,None,:] + weave[:,:,None], 0,255)
    strip_px = int(round(w*STRIP_WIDTH/LENGTH))
    for sl, color in ((slice(0,strip_px), [34.,65.,76.]),
                      (slice(w-strip_px,w), [127.,82.,47.])):
        stripe_weave = (2.6*np.cos(xx[:,sl]*np.pi/2) +
                        2.3*np.sin(yy[:,sl]*np.pi/3) +
                        0.6*weave[:,sl])
        rgb[:,sl,:] = np.clip(np.array(color)[None,None,:] + stripe_weave[:,:,None],0,255)
    im = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(im)
    # Image v increases downward, OBJ v increases upward.
    def xy(u,v):
        return ((u/LENGTH+0.5)*(w-1),(0.5-v/WIDTH)*(h-1))
    def stitch_segment(a,b, color=(148,143,122), dash=7, gap=8, width=1):
        a,b = np.array(a,dtype=float),np.array(b,dtype=float)
        length = np.linalg.norm(b-a)
        if length == 0:
            return
        tangent = (b-a)/length
        for s in np.arange(0,length,dash+gap):
            draw.line([tuple(a+tangent*s),tuple(a+tangent*min(s+dash,length))],fill=color,width=width)
    seam = 0.003
    for v in (-WIDTH/2+seam, WIDTH/2-seam):
        stitch_segment(xy(-LENGTH/2+seam,v),xy(LENGTH/2-seam,v))
    for u in (-LENGTH/2+STRIP_WIDTH+0.002,LENGTH/2-STRIP_WIDTH-0.002):
        stitch_segment(xy(u,-WIDTH/2+seam),xy(u,WIDTH/2-seam))
    for hc in HOLE_CENTERS:
        x0,x1 = hc-HOLE_HALF_WIDTH-seam,hc+HOLE_HALF_WIDTH+seam
        y0,y1 = HOLE_V-HOLE_HALF_HEIGHT-seam,HOLE_V+HOLE_HALF_HEIGHT+seam
        points = [xy(x0,y0),xy(x1,y0),xy(x1,y1),xy(x0,y1)]
        for a,b in zip(points,points[1:]+points[:1]):
            stitch_segment(a,b,color=(151,143,118),dash=5,gap=6)
    im.save(path)


def classify_particles(positions, rim_tolerance=0.009):
    """Classify INITIAL post-remesh particles using their world positions.

    Call before cloth movement; save these indices for the entire rollout.
    """
    p = np.asarray(positions)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Expected unbatched positions with shape (N,3).')
    uv = p[:,:2] - ORIGIN[:2]
    u,v = uv.T
    result = {
        'material_coordinates': uv,
        'left_strip': np.flatnonzero(u <= -LENGTH/2 + STRIP_WIDTH),
        'right_strip': np.flatnonzero(u >= LENGTH/2 - STRIP_WIDTH),
        'left_edge': np.flatnonzero(u <= -LENGTH/2 + rim_tolerance),
        'right_edge': np.flatnonzero(u >= LENGTH/2 - rim_tolerance),
    }
    for name,hc in zip(('left_hole','right_hole'),HOLE_CENTERS):
        dx,dy = np.abs(u-hc),np.abs(v-HOLE_V)
        on_vertical = (np.abs(dx-HOLE_HALF_WIDTH) <= rim_tolerance) & (dy <= HOLE_HALF_HEIGHT+rim_tolerance)
        on_horizontal = (np.abs(dy-HOLE_HALF_HEIGHT) <= rim_tolerance) & (dx <= HOLE_HALF_WIDTH+rim_tolerance)
        result[name+'_rim'] = np.flatnonzero(on_vertical | on_horizontal)
    return result


def build_cloth(out_dir):
    """Write OBJ, MTL, texture, and explicit assumed-geometry metadata."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    mesh_path = out/'garment.obj'
    mtl_path = out/'garment.mtl'
    tex_path = out/'garment_fabric.png'
    vertices,faces,uvs = _geometry()
    _texture(tex_path)
    mtl_path.write_text('newmtl garment_fabric\nKa 0.3 0.3 0.3\nKd 1 1 1\nKs 0 0 0\nNs 1\nd 1\nillum 1\nmap_Kd garment_fabric.png\n')
    with mesh_path.open('w') as f:
        f.write('# Image-inferred reconstruction; vertices in world coordinates, metres.\n')
        f.write('mtllib garment.mtl\no garment\nusemtl garment_fabric\n')
        for p in vertices:
            f.write('v {:.9f} {:.9f} {:.9f}\n'.format(*p))
        for uv in uvs:
            f.write('vt {:.9f} {:.9f}\n'.format(*uv))
        f.write('s 1\n')
        for tri in faces+1:
            f.write('f '+' '.join(f'{i}/{i}' for i in tri)+'\n')
    edges = np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique_edges,edge_counts = np.unique(edges,axis=0,return_counts=True)
    boundary_edges = unique_edges[edge_counts==1]
    boundary_graph = {}
    for a,b in boundary_edges:
        boundary_graph.setdefault(int(a),set()).add(int(b))
        boundary_graph.setdefault(int(b),set()).add(int(a))
    pending = set(boundary_graph)
    boundary_loops = 0
    while pending:
        stack = [pending.pop()]
        boundary_loops += 1
        while stack:
            for n in boundary_graph[stack.pop()]:
                if n in pending:
                    pending.remove(n)
                    stack.append(n)
    assert boundary_loops == 3, f'Expected outer boundary + 2 holes, got {boundary_loops}'
    assert all(len(n)==2 for n in boundary_graph.values())
    assert len(vertices)-len(unique_edges)+len(faces)==-1
    manifest = {
        'obj': str(mesh_path), 'mesh_path': str(mesh_path),
        'mtl': str(mtl_path), 'texture': str(tex_path),
        'coordinates': 'world XYZ in metres; use morph pos=(0,0,0), scale=1',
        'origin': ORIGIN.tolist(), 'length': LENGTH, 'width': WIDTH,
        'hole_centers_local': [[c,HOLE_V] for c in HOLE_CENTERS],
        'hole_size': [2*HOLE_HALF_WIDTH,2*HOLE_HALF_HEIGHT],
        'velcro_strip_width': STRIP_WIDTH,
        'vertices': len(vertices), 'triangles': len(faces),
        'boundary_loops': boundary_loops,
        'area_m2': LENGTH*WIDTH-2*(2*HOLE_HALF_WIDTH)*(2*HOLE_HALF_HEIGHT),
        'assumptions': 'All dimensions, hole shapes, texture, and physical properties are estimates from images; no original assets used.',
    }
    manifest_path = out/'garment_manifest.json'
    manifest['manifest_path'] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',default=str(Path(__file__).parent/'assets'))
    args = parser.parse_args()
    print(json.dumps(build_cloth(args.out),indent=2))
