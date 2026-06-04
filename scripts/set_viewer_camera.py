"""Fit the SuperSplat viewer camera to a cleaned .ply's actual content and write
it into a settings.json. Applies the same -r -90,0,0 (X) rotation the SOG export
uses, so the camera matches what the viewer shows. Frames a 3/4 hero angle.

  set_viewer_camera.py <clean.ply> <settings.json> [dist_factor=0.62]
"""
import json
import sys
import numpy as np

ply, settings = sys.argv[1], sys.argv[2]
distf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85

f = open(ply, 'rb'); hdr = b''
while b'end_header\n' not in hdr: hdr += f.read(4096)
header = hdr.split(b'end_header\n', 1)[0]
lines = header.decode().splitlines()
props = [l.split()[2] for l in lines if l.startswith('property float')]
n = int(next(l for l in lines if l.startswith('element vertex')).split()[2])
arr = np.fromfile(ply, dtype=np.dtype([(p, '<f4') for p in props]), count=n,
                  offset=len(header) + len(b'end_header\n'))
xyz = np.stack([arr['x'], arr['y'], arr['z']], 1).astype(np.float64)
R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)   # -r -90 about X: (x,y,z)->(x,z,-y)
p = xyz @ R.T
ctr = np.median(p, 0)
lo, hi = np.percentile(p, 1, 0), np.percentile(p, 99, 0)
diag = float(np.linalg.norm(hi - lo))
d = diag * distf
# 3/4 view: front-right and modestly above the target (cinematic, not top-down)
direction = np.array([0.72, 0.26, 0.72]); direction /= np.linalg.norm(direction)
pos = ctr + direction * d

with open(settings) as fh:
    cfg = json.load(fh)
cfg['cameras'] = [{'initial': {
    'position': [round(float(v), 2) for v in pos],
    'target':   [round(float(v), 2) for v in ctr],
    'fov': 50}}]
with open(settings, 'w') as fh:
    json.dump(cfg, fh, indent=2)
print(f'camera -> target {ctr.round(1).tolist()}  position {pos.round(1).tolist()}  '
      f'(diag {diag:.0f}m, dist {d:.0f}m) written to {settings}')
