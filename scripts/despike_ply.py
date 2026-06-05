"""Clean floaters from a 3DGS .ply by removing the two artifact families that
survive opacity/box/cluster filters, plus a gentle global opacity floor:

  1. SPIKES  - long thin needles (longest axis large in absolute terms AND much
               longer than the 2nd axis). Render as bright chromatic slivers.
  2. HAZE    - large soft blobs that are big AND faint (low opacity). Render as
               milky fog. Real surfaces are small+opaque, so this spares them.
  3. FLOOR   - drop near-invisible gaussians below an opacity floor.
  4. GLINT   - drop chromatic confetti: bright AND highly color-saturated specks
               (pure R/G/B). Real rock/moss is desaturated, so this spares it.
  5. SOR     - statistical outlier removal: drop gaussians whose mean distance to
               their K nearest neighbors exceeds sor_dist (sparse-region floaters;
               the dense rock surface has close neighbors). Applied LAST, on the
               survivors, so density reflects the cleaned scene. Needs scipy.
  6. CC      - keep only the largest connected component (voxelized at cc_vox m).
               The main object is one big blob; detached bright clusters / dark
               specks / colored blobs are many tiny components -> dropped. Precise
               (unlike splat-transform -D, which can fragment the main mass).

Scales in the .ply are LOG-space; opacity is a LOGIT; color is SH-DC (rgb ~=
0.5 + 0.282*f_dc). Flat 'pancake' surface splats (s2 ~= s1 >> s0) and opaque
surfaces are kept. All columns preserved.

  despike_ply.py <in.ply> <out.ply> [s_abs=0.3] [aspect=8] [haze_size=0.5] \
                 [haze_op=0.15] [op_floor=0.05] [box=x,y,z,X,Y,Z|-] [sat=0.6] \
                 [val=0.6] [sor_dist=0(off)] [sor_k=16] [cc_vox=0(off)]
"""
import sys
import numpy as np

inp, outp = sys.argv[1], sys.argv[2]
argf = lambda i, d: float(sys.argv[i]) if len(sys.argv) > i else d
s_abs, aspect = argf(3, 0.3), argf(4, 8.0)
haze_size, haze_op = argf(5, 0.5), argf(6, 0.15)
op_floor = argf(7, 0.05)
box = [float(v) for v in sys.argv[8].split(',')] if len(sys.argv) > 8 and sys.argv[8] != '-' else None
sat_thr, val_thr = argf(9, 0.6), argf(10, 0.6)
sor_dist, sor_k = argf(11, 0.0), int(argf(12, 16))
cc_vox = argf(13, 0.0)

f = open(inp, 'rb'); hdr = b''
while b'end_header\n' not in hdr: hdr += f.read(4096)
header = hdr.split(b'end_header\n', 1)[0]
lines = header.decode().splitlines()
props = [l.split()[2] for l in lines if l.startswith('property float')]
n = int(next(l for l in lines if l.startswith('element vertex')).split()[2])
dt = np.dtype([(p, '<f4') for p in props])
arr = np.fromfile(inp, dtype=dt, count=n, offset=len(header) + len(b'end_header\n'))

keep = np.all([np.isfinite(arr[p]) for p in ('x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'opacity')], 0)
if box:
    lo, hi = np.array(box[:3]), np.array(box[3:])
    xyz = np.stack([arr['x'], arr['y'], arr['z']], 1)
    keep &= np.all((xyz >= lo) & (xyz <= hi), 1)
S = np.sort(np.exp(np.stack([arr['scale_0'], arr['scale_1'], arr['scale_2']], 1)), axis=1)  # meters
s1, s2 = S[:, 1], S[:, 2]
op = 1.0 / (1.0 + np.exp(-arr['opacity']))                     # activated opacity
spike = (s2 > s_abs) & (s2 > aspect * s1)
haze = (s2 > haze_size) & (op < haze_op)
faint = op < op_floor
rgb = np.clip(0.5 + 0.28209 * np.stack([arr['f_dc_0'], arr['f_dc_1'], arr['f_dc_2']], 1), 0, 1)
mx, mn = rgb.max(1), rgb.min(1)
sat = (mx - mn) / (mx + 1e-6)
if sat_thr >= 1.0:                                 # sat_thr>=1 => DISABLE glint entirely
    glint = np.zeros(len(arr), bool)               # (incl. the hardcoded pure-hue/white clauses
else:                                              #  that eat stylized/saturated terrain — use for
    glint = (((sat > sat_thr) & (mx > val_thr))    #  2DGS surfels & no-sky MCMC; spatial filters only)
             | ((sat > 0.9) & (mx > 0.4))          # +pure-hue blobs (any hue)
             | ((sat < 0.22) & (mx > 0.82)))       # +bright near-white floaters (sky/highlight)
keep &= ~(spike | haze | faint | glint)
out = arr[keep]

n_sor = 0
if sor_dist > 0:
    from scipy.spatial import cKDTree
    p = np.stack([out['x'], out['y'], out['z']], 1).astype(np.float64)
    d, _ = cKDTree(p).query(p, k=sor_k + 1, workers=-1)
    md = d[:, 1:].mean(1)
    sor_keep = md <= sor_dist
    n_sor = int((~sor_keep).sum())
    out = out[sor_keep]

n_cc = 0
if cc_vox > 0:
    from scipy import ndimage
    p = np.stack([out['x'], out['y'], out['z']], 1)
    key = np.floor((p - p.min(0)) / cc_vox).astype(int)
    dims = key.max(0) + 3
    grid = np.zeros(dims, bool); grid[key[:, 0] + 1, key[:, 1] + 1, key[:, 2] + 1] = True
    lab, _ = ndimage.label(grid)
    gl = lab[key[:, 0] + 1, key[:, 1] + 1, key[:, 2] + 1]
    ids, counts = np.unique(gl, return_counts=True)
    biggest = ids[np.argmax(counts)]
    cc_keep = gl == biggest
    n_cc = int((~cc_keep).sum())
    out = out[cc_keep]

h = ('ply\nformat binary_little_endian 1.0\n'
     f'comment cleaned: spikes(s2>{s_abs}m,/s1>{aspect}) haze(s2>{haze_size}m,op<{haze_op}) floor(op<{op_floor})\n'
     f'element vertex {len(out)}\n' + ''.join(f'property float {p}\n' for p in props) + 'end_header\n')
with open(outp, 'wb') as g:
    g.write(h.encode()); out.tofile(g)
print(f'clean: {n:,} -> {len(out):,}  (spikes {int(spike.sum()):,}, haze {int(haze.sum()):,}, '
      f'faint {int(faint.sum()):,}, glint {int(glint.sum()):,}, sor {n_sor:,}, '
      f'cc {n_cc:,}; cheap-filter overlaps possible)  -> {outp}')
