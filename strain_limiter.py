"""Optional explicit cloth inextensibility projection, separate from Genesis.

This adds pairwise edge-length inequalities to the native PBD result. It does
not pin particles or recover the original cloth properties. Equal free particle
masses are assumed, matching this reconstruction's Genesis cloth discretization.
Finite iterations are approximate; reported residuals expose infeasible guides.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def _cached_colors(edge_bytes, edge_count):
    edges=np.frombuffer(edge_bytes,dtype=np.int64).reshape(edge_count,2)
    used={}
    groups=[]
    for i,(a,b) in enumerate(edges):
        a,b=int(a),int(b)
        taken=used.get(a,set()) | used.get(b,set())
        color=0
        while color in taken:
            color+=1
        while len(groups)<=color:
            groups.append([])
        groups[color].append(i)
        used.setdefault(a,set()).add(color)
        used.setdefault(b,set()).add(color)
    return tuple(np.asarray(group,dtype=np.int64) for group in groups)


def _ratios(p,edges,rest):
    return np.linalg.norm(p[edges[:,1]]-p[edges[:,0]],axis=1)/rest


def _ratio_stats(ratios,min_ratio,max_ratio,prefix):
    return {
        prefix+'min_ratio':float(ratios.min()) if len(ratios) else 1.,
        prefix+'max_ratio':float(ratios.max()) if len(ratios) else 1.,
        prefix+'p95_ratio':float(np.quantile(ratios,.95)) if len(ratios) else 1.,
        prefix+'extension_violations':int(np.count_nonzero(ratios>max_ratio+1e-5)),
        prefix+'compression_violations':int(np.count_nonzero(ratios<min_ratio-1e-5)),
    }


def project_strain(pos, rest_edges, rest_lengths, free_mask=None,
                   iterations=8, max_ratio=1.03, min_ratio=.85):
    """Return (corrected_positions, stats) using colored Gauss–Seidel sweeps.

    ``rest_edges`` is an E-by-2 particle index array, ``rest_lengths`` an E-vector
    in metres. ``free_mask=False`` particles are unmodified. Edges sharing a
    vertex have different colors, so each vectorized color group has no write
    conflicts. Color order reverses on alternating sweeps to reduce directional
    bias. Impossible fixed–fixed edges are retained and explicitly counted.

    An exactly collapsed edge has no identifiable separation direction from
    these inputs; it is reported rather than assigned an invented direction.
    This function is not collision aware. Alternate with contact projection and
    measure the final remaining strain and penetrations after all corrections.
    """
    original=np.asarray(pos,dtype=float)
    edges=np.ascontiguousarray(rest_edges,dtype=np.int64)
    rest=np.asarray(rest_lengths,dtype=float)
    if original.ndim!=2 or original.shape[1]!=3 or not np.isfinite(original).all():
        raise ValueError('pos must be a finite (N,3) array.')
    if edges.ndim!=2 or edges.shape[1]!=2 or rest.shape!=(len(edges),):
        raise ValueError('rest_edges/rest_lengths must have shapes (E,2)/(E,).')
    if np.any(edges<0) or np.any(edges>=len(original)) or np.any(edges[:,0]==edges[:,1]):
        raise ValueError('Edges must contain valid distinct particle indices.')
    if np.any(rest<=0) or not np.isfinite(rest).all():
        raise ValueError('Rest lengths must be finite and positive.')
    if not (0<=min_ratio<=1<=max_ratio) or iterations<1:
        raise ValueError('Require 0<=min_ratio<=1<=max_ratio and iterations>=1.')
    free=np.ones(len(original),dtype=bool) if free_mask is None else np.asarray(free_mask,dtype=bool)
    if free.shape!=(len(original),):
        raise ValueError('free_mask must have shape (N,).')
    colors=_cached_colors(edges.tobytes(),len(edges))
    p=original.copy()
    before=_ratios(p,edges,rest)
    lower,upper=min_ratio*rest,max_ratio*rest
    wa,wb=free[edges[:,0]].astype(float),free[edges[:,1]].astype(float)
    weights=wa+wb
    largest_single_correction=0.
    iterations_used=0
    for iteration in range(int(iterations)):
        max_change=0.
        order=colors if iteration%2==0 else colors[::-1]
        for ids in order:
            a,b=edges[ids].T
            d=p[b]-p[a]
            length=np.linalg.norm(d,axis=1)
            eligible=(weights[ids]>0)&(length>1e-12)
            target=np.clip(length,lower[ids],upper[ids])
            correction=(length-target)/np.maximum(length,1e-12)
            correction[~eligible]=0
            correction=(correction/np.maximum(weights[ids],1))[:,None]*d
            ca=wa[ids,None]*correction
            cb=wb[ids,None]*correction
            p[a]+=ca
            p[b]-=cb
            change=max(np.linalg.norm(ca,axis=1).max(initial=0),
                       np.linalg.norm(cb,axis=1).max(initial=0))
            max_change=max(max_change,float(change))
        iterations_used=iteration+1
        largest_single_correction=max(largest_single_correction,max_change)
        if max_change<1e-9:
            break
    after=_ratios(p,edges,rest)
    fixed_conflict=(weights==0)&((after<min_ratio-1e-5)|(after>max_ratio+1e-5))
    displacement=np.linalg.norm(p-original,axis=1)
    stats={
        'model':'custom equal-mass colored Gauss-Seidel edge strain projection',
        'edge_count':len(edges),'color_count':len(colors),'iterations_used':iterations_used,
        'min_allowed_ratio':float(min_ratio),'max_allowed_ratio':float(max_ratio),
        'fixed_fixed_conflicts':int(fixed_conflict.sum()),
        'collapsed_edges':int(np.count_nonzero(after*rest<=1e-12)),
        'corrected_particles':int(np.count_nonzero(displacement>1e-9)),
        'max_position_correction_m':float(displacement.max(initial=0)),
        'sum_position_correction_m':float(displacement.sum()),
        'largest_single_edge_position_correction_m':largest_single_correction,
        'free_centroid_shift_m':float(np.linalg.norm((p[free]-original[free]).mean(axis=0))) if np.any(free) else 0.,
        'changes_fixed_particles':bool(np.any(p[~free]!=original[~free])),
        **_ratio_stats(before,min_ratio,max_ratio,'before_'),
        **_ratio_stats(after,min_ratio,max_ratio,'after_'),
    }
    stats['all_constraints_satisfied']=bool(stats['after_extension_violations']==0 and
                                           stats['after_compression_violations']==0)
    return p,stats
