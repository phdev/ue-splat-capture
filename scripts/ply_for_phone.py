"""Shrink a brush/3DGS .ply for phone viewing: drop NaNs + crop far floaters +
prune top-N by opacity + strip SH (keeps DC color)."""
import os, sys
import numpy as np
inp, outp = sys.argv[1], sys.argv[2]
keep = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000
f = open(inp, 'rb'); hdr = b''
while b'end_header\n' not in hdr: hdr += f.read(256)
header = hdr.split(b'end_header\n', 1)[0]
lines = header.decode().splitlines()
props = [l.split()[2] for l in lines if l.startswith('property float')]
n = int(next(l for l in lines if l.startswith('element vertex')).split()[2])
arr = np.fromfile(inp, dtype=np.dtype([(p, '<f4') for p in props]), count=n,
                  offset=len(header) + len(b'end_header\n'))
fin = np.all([np.isfinite(arr[p]) for p in ('x','y','z','opacity','scale_0')], 0)
arr = arr[fin]
for ax in ('x', 'y', 'z'):                       # crop far floaters
    lo, hi = np.percentile(arr[ax], [1, 99]); m = 0.6 * (hi - lo)
    arr = arr[(arr[ax] >= lo - m) & (arr[ax] <= hi + m)]
print(f'after finite+crop: {arr.shape[0]:,} of {n:,}')
if keep < arr.shape[0]:
    arr = arr[np.argpartition(arr['opacity'], arr.shape[0] - keep)[arr.shape[0] - keep:]]
op = ['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
out = np.empty(len(arr), np.dtype([(p, '<f4') for p in op]))
for p in op: out[p] = arr[p]
h = ('ply\nformat binary_little_endian 1.0\ncomment phone-optimized: SH stripped, '
     'cropped, opacity-pruned\ncomment Vertical axis: y\n'
     f'element vertex {len(arr)}\n' + ''.join(f'property float {p}\n' for p in op) + 'end_header\n')
with open(outp, 'wb') as g:
    g.write(h.encode()); out.tofile(g)
xyz = np.stack([out['x'], out['y'], out['z']], 1)
print(f'wrote {outp}: {len(arr):,} splats, {os.path.getsize(outp)/1e6:.0f} MB, '
      f'extent(m)={(xyz.max(0)-xyz.min(0)).round(1).tolist()}')
