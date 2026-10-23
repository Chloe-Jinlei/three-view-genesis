"""Image-inferred, assisted dressing reference; no recovered demonstration data.

The returned field is intended only for selected grasp/rim guides in a cloth
solver. Applying it to every vertex would be animation, not a free simulation.
Coordinates are metres. Bear faces -Y. ``initial_vertices`` are world positions.
"""

from __future__ import annotations

import numpy as np
from parameters import BODY_RADII, BACK_WRAP_Y, ARMS_ABDUCTION_DEG

INITIAL_Y = -0.25
INITIAL_Z = 0.758
HALF_LENGTH = 0.40
HOLE_U = 0.205
HOLE_V = 0.045
HOLE_HALF_WIDTH = 0.035
HOLE_HALF_HEIGHT = 0.038
BODY_CENTER = np.array([0.0, 0.10, 0.965])
ARM_ROOT = np.array([0.105, 0.095, 1.065])
ARM_PAW = np.array([0.160, 0.070, 0.985])
STAGES = (
    (0.00, 0.10, "reach_and_grasp"),
    (0.10, 0.22, "lift_clear_of_table"),
    (0.22, 0.36, "stand_in_front"),
    (0.36, 0.53, "form_open_sleeve_approach"),
    (0.53, 0.73, "guided_wrist_to_shoulder_threading"),
    (0.73, 0.87, "wrap_tails_around_back"),
    (0.87, 0.90, "align_velcro"),
    (0.90, 0.92, "hold_closed_seam"),
    (0.92, 1.00, "release_and_settle"),
)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def stage(t: float) -> str:
    for lo, hi, name in STAGES:
        if float(t) <= hi:
            return name
    return STAGES[-1][2]


def stage_name(t: float) -> str:
    return stage(t)


def arm_angle(t: float) -> float:
    """Positive abduction magnitude; use arm_pose for side-specific Y rotation."""
    return float(ARMS_ABDUCTION_DEG)*_progress(t, 0.53, 0.73)


def _rotation_y(angle_radians):
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _abduction_rotation(side, thread_progress):
    # The right arm rotates by negative Y, the left by positive Y.
    angle = -float(side)*np.deg2rad(ARMS_ABDUCTION_DEG)*float(thread_progress)
    return _rotation_y(angle)


def arm_endpoints(side, thread_progress=0.0):
    """Articulated shoulder and wrist for normalized threading progress."""
    root, paw = ARM_ROOT.copy(), ARM_PAW.copy()
    root[0] *= side
    paw[0] *= side
    rotation = _abduction_rotation(side, thread_progress)
    return root, root+rotation@(paw-root)


def arm_pose(part, t: float):
    """Return (entity_position, wxyz_quaternion, shoulder, wrist).

    Arm mesh vertices already contain their initial ellipsoid orientation.
    Consequently the entity quaternion is only the added shoulder rotation.
    The shoulder is stationary and the mesh midpoint rotates around it.
    """
    root = np.asarray(part["shoulder_world"], dtype=float)
    original_wrist = np.asarray(part["wrist_world"], dtype=float)
    original_center = np.asarray(part["pos"], dtype=float)
    side = 1.0 if root[0] >= 0 else -1.0
    progress = _progress(t, 0.53, 0.73)
    rotation = _abduction_rotation(side, progress)
    angle = -side*np.deg2rad(ARMS_ABDUCTION_DEG)*progress
    quaternion = np.array([np.cos(angle/2), 0.0, np.sin(angle/2), 0.0])
    position = root+rotation@(original_center-root)
    wrist = root+rotation@(original_wrist-root)
    return position, quaternion, root.copy(), wrist


def _progress(t, lo, hi):
    return float(smoothstep((float(t) - lo) / (hi - lo)))


def _bezier(p0, p1, p2, p3, q):
    q = np.asarray(q)[:, None]
    return ((1-q)**3*p0 + 3*(1-q)**2*q*p1
            + 3*(1-q)*q**2*p2 + q**3*p3)


def _project_outside_body(points, clearance=0.007):
    """Project nominal references out of the known analytic torso proxy.

    Collision response, including arm/head contact, still belongs to Genesis.
    This is a safeguard for the reference field, not a contact simulation.
    """
    p = np.asarray(points, dtype=float).copy()
    # Use the identical conservative envelope as contact_extension.py. A
    # narrower guide envelope causes abrupt overlap resolution on release.
    axes=BODY_RADII*(1+.006/BODY_RADII.min())
    delta=p-BODY_CENTER
    radial=np.linalg.norm(delta/axes,axis=1)
    mask=radial<1.0
    p[mask]=BODY_CENTER+delta[mask]/np.maximum(radial[mask,None],1e-9)
    return p


def _arm_proxy(side, thread_progress=0.0):
    """Match the image-inferred arm ellipsoid created by assets_builder.py."""
    root, paw = ARM_ROOT.copy(), ARM_PAW.copy()
    root[0] *= side
    paw[0] *= side
    axis = paw-root
    length = np.linalg.norm(axis)
    normal = axis/length
    # Minimal rotation from local +Z onto the limb centreline, matching the
    # mesh construction's align_vectors convention for a non-antipodal axis.
    cross = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    k = np.array([[0.0, -cross[2], cross[1]],
                  [cross[2], 0.0, -cross[0]],
                  [-cross[1], cross[0], 0.0]])
    rotation = np.eye(3)+k+k@k/(1+normal[2])
    abduction = _abduction_rotation(side, thread_progress)
    posed_paw = root+abduction@(paw-root)
    return ((root+posed_paw)/2, abduction@rotation,
            np.array([0.029, 0.028, length/2+0.018]))


def _project_outside_proxies(points, thread_progress=0.0):
    """Keep sparse nominal guides outside the torso/arm union.

    Alternating projections allow the lower sleeve rim to move around the
    armpit union, instead of requesting an impossible planar ring through the
    overlapping arm and torso. This is still guidance; Genesis owns contact.
    """
    p = np.asarray(points, dtype=float).copy()
    proxies = [_arm_proxy(-1, thread_progress), _arm_proxy(1, thread_progress)]
    for _ in range(48):
        previous = p.copy()
        p = _project_outside_body(p)
        for center, rotation, radii in proxies:
            shell_radii = radii*(1+.006/radii.min())
            local = (p-center)@rotation
            normalized = local/shell_radii
            norm = np.linalg.norm(normalized, axis=1)
            inside = norm < 1.0
            degenerate = inside & (norm < 1e-10)
            if np.any(degenerate):
                normalized[degenerate] = np.array([1.0, 0.0, 0.0])
                norm[degenerate] = 1.0
            if np.any(inside):
                local[inside] = normalized[inside]/norm[inside, None]*shell_radii
                p[inside] = local[inside]@rotation.T+center
        if np.max(np.linalg.norm(p-previous, axis=1), initial=0.0) < 1e-8:
            break
    return p


def sleeve_centers(thread_progress: float):
    """Follow the rotating arms from beyond paw tips toward forearms."""
    progress = np.clip(float(thread_progress), 0.0, 1.0)
    r = 1.20-0.90*progress
    centers = []
    for side in [-1, 1]:
        root, wrist = arm_endpoints(side, progress)
        centers.append(root+r*(wrist-root))
    return np.stack(centers)


def _curved_reference(u, v, thread_progress, wrap_progress, close_progress):
    """Open-front U with independently curling posterior tails.

    The front panel intentionally has surplus arclength. It must wrinkle in
    PBD; no assertion of an isometric or unique image reconstruction is made.
    """
    u, v = np.asarray(u), np.asarray(v)
    sign = np.where(u >= 0, 1.0, -1.0)
    au = np.abs(u)
    q = np.clip(au / HOLE_U, 0.0, 1.0)
    h = sleeve_centers(thread_progress)[1]
    arm_root, arm_wrist = arm_endpoints(1, thread_progress)
    arm_n = arm_wrist-arm_root
    arm_n /= np.linalg.norm(arm_n)
    vertical = np.array([0.0, 0.0, 1.0])
    rim_vertical = vertical - arm_n * arm_n[2]
    rim_vertical /= np.linalg.norm(rim_vertical)
    tangent = np.array([-arm_n[1], arm_n[0], 0.0])
    tangent /= np.linalg.norm(tangent)
    front_y = -0.025 + 0.030 * float(thread_progress)
    front_z = (1-thread_progress)*h[2] + thread_progress*1.045
    p0 = np.array([0.0, front_y, front_z])
    p1 = np.array([0.064, front_y, front_z])
    p2 = h - 0.055*tangent
    centers = _bezier(p0, p1, p2, h, q)
    tail = au > HOLE_U
    if np.any(tail):
        w = np.clip((au[tail]-HOLE_U)/(HALF_LENGTH-HOLE_U), 0.0, 1.0)
        open_tail = h + (au[tail]-HOLE_U)[:, None]*tangent
        # Both terminal strips cross the back centreline by approximately 8 mm.
        # Their normal separation closes from 15 to 3 mm for contact bonding.
        phi = w*(np.pi/2 + 0.065)
        closed_tail = np.empty_like(open_tail)
        closed_tail[:, 0] = h[0]*np.cos(phi)
        closed_tail[:, 1] = h[1] + (BACK_WRAP_Y-h[1])*np.sin(phi)
        closed_tail[:, 2] = h[2] + 0.004*w
        centers[tail] = ((1-wrap_progress)*open_tail
                         + wrap_progress*closed_tail)
    centers[:, 0] *= sign

    dv = v-HOLE_V
    p = centers + dv[:, None]*vertical
    # Use a locally affine sleeve patch so the aperture preserves its actual
    # material width. The global Bezier curve compresses du near its endpoint;
    # applying that compression to a rim would make a nominal 70-mm opening
    # narrower than the reconstructed arm.
    du = np.abs(au-HOLE_U)
    wu = 1-smoothstep((du-0.040)/0.040)
    wv = 1-smoothstep((np.abs(dv)-0.044)/0.046)
    tilt = wu*wv
    b = np.tile(rim_vertical, (len(u), 1))
    b[:, 0] *= sign
    affine = h+(au-HOLE_U)[:, None]*tangent
    affine[:, 0] *= sign
    affine += dv[:, None]*b
    p = (1-tilt[:, None])*p+tilt[:, None]*affine
    # Smooth separation between the two layers at the posterior seam.
    seam = smoothstep((au-(HALF_LENGTH-0.055))/0.055)*wrap_progress
    separation = 0.015*(1-close_progress)+0.003*close_progress
    p[:, 1] += sign*0.5*separation*seam
    return _project_outside_proxies(p, thread_progress)


def reference_positions(initial_vertices, t: float):
    """Return an N-by-3 guide field for normalized episode time 0 <= t <= 1.

    Retains the exact supplied initial mesh at t=0. Only sparse perimeter,
    aperture-rim, and grasp constraints should follow this field in simulation.
    """
    initial = np.asarray(initial_vertices, dtype=float)
    u = initial[:, 0]
    v = initial[:, 1]-INITIAL_Y
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.10:
        return initial.copy()
    if t <= 0.22:
        p = initial.copy()
        p[:, 2] += (0.880-INITIAL_Z)*_progress(t, 0.10, 0.22)
        return p
    if t <= 0.36:
        a = _progress(t, 0.22, 0.36)
        angle = a*np.pi/2
        return np.column_stack((
            u,
            (1-a)*INITIAL_Y+a*(-0.150)+v*np.cos(angle),
            (1-a)*0.880+a*0.970+v*np.sin(angle),
        ))
    if t <= 0.53:
        a = _progress(t, 0.36, 0.53)
        upright = np.column_stack((u, np.full_like(u, -0.150), 0.970+v))
        approach = _curved_reference(u, v, 0.0, 0.0, 0.0)
        return _project_outside_proxies((1-a)*upright+a*approach)
    threading = _progress(t, 0.53, 0.73)
    wrapping = _progress(t, 0.73, 0.87)
    closure = _progress(t, 0.87, 0.90)
    return _curved_reference(u, v, threading, wrapping, closure)


def target_positions(initial_vertices, t: float):
    """Alias for controller integrations."""
    return reference_positions(initial_vertices, t)
