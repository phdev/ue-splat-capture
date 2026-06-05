"""Convert an ingested transforms.json (OpenCV world coords) -> a COLMAP text model
(cameras.txt + images.txt + points3D.txt) so a COLMAP-based trainer (Inria 3DGS / 2DGS)
can use our EXACT UE poses and SKIP COLMAP SfM entirely (412 imgs exhaustive matching is
~12 h on CPU). Auto-detects the camera-forward convention by checking that the per-camera
forward rays converge on the scene centre. points3D = a random cloud in the camera AABB
(the trainer densifies from it).

  transforms_to_colmap.py <transforms.json> <out_sparse_dir> [n_init_points=120000]
"""
import json
import os
import sys

import numpy as np

tj, outd = sys.argv[1], sys.argv[2]
n_pts = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
os.makedirs(outd, exist_ok=True)
d = json.load(open(tj))
W, H = d["w"], d["h"]
fx, fy, cx, cy = d["fl_x"], d["fl_y"], d["cx"], d["cy"]
frames = d["frames"]
C2W = np.array([f["transform_matrix"] for f in frames], float)        # (N,4,4)
pos = C2W[:, :3, 3]

# --- detect forward convention: forward rays from all cams should converge on one point.
def convergence(fwd_col_sign):
    fwd = fwd_col_sign * C2W[:, :3, 2]                                 # +Z (OpenCV) or -Z (OpenGL)
    # least-squares point closest to all rays (pos + t*fwd)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p, f in zip(pos, fwd):
        f = f / np.linalg.norm(f); P = np.eye(3) - np.outer(f, f)
        A += P; b += P @ p
    c = np.linalg.solve(A, b)
    # mean residual distance of cams' rays to c
    res = []
    for p, f in zip(pos, fwd):
        f = f / np.linalg.norm(f); res.append(np.linalg.norm((np.eye(3) - np.outer(f, f)) @ (p - c)))
    return np.mean(res), c

resid, focus = convergence(+1.0)          # focus is the same for +/-Z (rays = infinite lines)
# directional: the TRUE forward points FROM each camera TOWARD the focus. Pick the sign
# of the +Z column whose mean dot with (focus - pos) is positive.
fz = C2W[:, :3, 2]
to_focus = focus[None, :] - pos
dot_pos = np.mean(np.sum(fz * to_focus, 1))
opencv = dot_pos > 0                        # +Z points toward focus -> OpenCV convention
print(f"ray convergence resid {resid:.2f}m, focus {focus.round(1).tolist()}; "
      f"mean dot(+Z, to_focus) {dot_pos:.1f} -> {'OpenCV (+Z fwd)' if opencv else 'OpenGL (-Z fwd)'}")

# Convert each c2w to COLMAP world-to-camera (qvec, tvec). COLMAP cam frame = OpenCV
# (+X right, +Y down, +Z forward). If the matrix is OpenGL, flip cam Y,Z to get OpenCV.
flip = np.diag([1, 1, 1]) if opencv else np.diag([1, -1, -1])

def rot2quat(R):
    # COLMAP quaternion order (w, x, y, z)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)

with open(os.path.join(outd, "cameras.txt"), "w") as f:
    f.write("# Camera list\n")
    f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

with open(os.path.join(outd, "images.txt"), "w") as f:
    f.write("# Image list\n")
    for i, fr in enumerate(frames):
        Rc2w = C2W[i, :3, :3] @ flip
        tc2w = C2W[i, :3, 3]
        Rw2c = Rc2w.T
        tw2c = -Rw2c @ tc2w
        q = rot2quat(Rw2c)
        name = os.path.basename(fr["file_path"])
        f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {tw2c[0]} {tw2c[1]} {tw2c[2]} 1 {name}\n\n")

# random init cloud in the camera AABB (+margin), random colors
lo, hi = pos.min(0), pos.max(0); m = 0.15 * (hi - lo)
rng = np.random.default_rng(0)
P = rng.uniform(lo - m, hi + m, (n_pts, 3))
col = rng.integers(80, 200, (n_pts, 3))
with open(os.path.join(outd, "points3D.txt"), "w") as f:
    f.write("# 3D point list\n")
    for j in range(n_pts):
        f.write(f"{j+1} {P[j,0]} {P[j,1]} {P[j,2]} {col[j,0]} {col[j,1]} {col[j,2]} 0\n")
print(f"wrote {len(frames)} images, 1 camera, {n_pts} init points -> {outd}")
