"""Find the tight bounding box of the DENSE body of a splat (excluding sparse
floaters/wisps) so we can box-crop them. Wisps live in low-count voxels; the
rock body lives in high-count voxels. Report a box that keeps the dense mass."""
import sys
import numpy as np

inp = sys.argv[1]
op_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.12
vox = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5   # density voxel size (m)

f = open(inp, 'rb'); hdr = b''
while b'end_header\n' not in hdr: hdr += f.read(256)
header = hdr.split(b'end_header\n', 1)[0]
lines = header.decode().splitlines()
props = [l.split()[2] for l in lines if l.startswith('property float')]
n = int(next(l for l in lines if l.startswith('element vertex')).split()[2])
arr = np.fromfile(inp, dtype=np.dtype([(p, '<f4') for p in props]), count=n,
                  offset=len(header) + len(b'end_header\n'))
fin = np.all([np.isfinite(arr[p]) for p in ('x', 'y', 'z', 'opacity', 'scale_1')], 0)
arr = arr[fin]
# "real surface" subset: opaque, not huge
mask = (arr['opacity'] > op_min) & (arr['scale_1'] < 0.5) & (arr['scale_0'] < 0.5) & (arr['scale_2'] < 0.5)
a = arr[mask]
xyz = np.stack([a['x'], a['y'], a['z']], 1)
print(f'total finite {len(arr):,}; real-surface subset {len(a):,} (opacity>{op_min}, scale<0.5)')
print(f'  raw bbox min {xyz.min(0).round(1).tolist()}  max {xyz.max(0).round(1).tolist()}')
print(f'  median {np.median(xyz,0).round(1).tolist()}')

# voxel density: count gaussians per `vox`-m cell
key = np.floor(xyz / vox).astype(np.int64)
_, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
cell_count = counts[inv]               # per-gaussian: how crowded its cell is
print(f'  voxel({vox}m) count distribution: max {counts.max()}, '
      f'p50 {int(np.percentile(counts,50))}, p90 {int(np.percentile(counts,90))}')

# Keep gaussians whose cell has >= THRESH neighbors, for several thresholds.
# Report the resulting tight box + how much mass is retained.
for thresh in (3, 5, 8, 12, 20):
    dense = cell_count >= thresh
    if dense.sum() == 0: continue
    d = xyz[dense]
    lo, hi = d.min(0), d.max(0)
    frac = dense.mean()
    print(f'  cell>={thresh:>3}: keep {frac*100:5.1f}%  '
          f'box -B {lo[0]:.1f},{lo[1]:.1f},{lo[2]:.1f},{hi[0]:.1f},{hi[1]:.1f},{hi[2]:.1f}  '
          f'size {(hi-lo).round(1).tolist()}')
