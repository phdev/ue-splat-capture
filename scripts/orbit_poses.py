"""Generate SuperSplat viewer settings JSONs orbiting a target, for multi-angle QA.
  orbit_poses.py <base_settings.json> <out_dir> <cx,cy,cz> <dist> <elev_deg> <n_az>
Writes pose0.json..pose{n-1}.json (copies of base with camera replaced).
"""
import json, sys, os
import numpy as np

base, out_dir, ctr_s, dist, elev, n = sys.argv[1], sys.argv[2], sys.argv[3], \
    float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
ctr = np.array([float(v) for v in ctr_s.split(',')])
cfg = json.load(open(base))
os.makedirs(out_dir, exist_ok=True)
e = np.radians(elev)
for i in range(n):
    az = np.radians(360.0 * i / n)
    d = np.array([np.cos(e) * np.sin(az), np.sin(e), np.cos(e) * np.cos(az)])
    pos = ctr + dist * d
    cfg['cameras'] = [{'initial': {
        'position': [round(float(v), 2) for v in pos],
        'target': [round(float(v), 2) for v in ctr],
        'fov': 50}}]
    json.dump(cfg, open(f'{out_dir}/pose{i}.json', 'w'))
print(f'wrote {n} poses to {out_dir} (elev {elev}deg, dist {dist}, target {ctr.tolist()})')
