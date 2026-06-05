"""Convert a 2D Gaussian Splatting (2DGS) .ply (surfels: 2 scales + normal) into a
standard 3DGS .ply the SuperSplat viewer / splat-transform can render. Each surfel
becomes a FLAT 3D gaussian: add a thin out-of-plane scale_2 (along the surfel normal,
which is already encoded by the rotation's 3rd axis) and drop nx,ny,nz.

  twodgs_to_3dgs.py <in_2dgs.ply> <out_3dgs.ply> [thin=1.0]   # thin = log-units thinner
"""
import sys
import numpy as np

inp, outp = sys.argv[1], sys.argv[2]
thin = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

f = open(inp, 'rb'); hdr = b''
while b'end_header\n' not in hdr: hdr += f.read(4096)
header = hdr.split(b'end_header\n', 1)[0]
lines = header.decode().splitlines()
props = [l.split()[2] for l in lines if l.startswith('property float')]
n = int(next(l for l in lines if l.startswith('element vertex')).split()[2])
arr = np.fromfile(inp, dtype=np.dtype([(p, '<f4') for p in props]), count=n,
                  offset=len(header) + len(b'end_header\n'))
assert 'scale_0' in props and 'scale_1' in props and 'scale_2' not in props, "not a 2-scale 2DGS ply"

# scale_2 (out-of-plane / normal direction): thinner than the smaller in-plane axis so
# the gaussian renders as a flat disk on the surface.
s2 = np.minimum(arr['scale_0'], arr['scale_1']) - thin
out_props = [p for p in props if p not in ('nx', 'ny', 'nz')]   # drop normals
# insert scale_2 right after scale_1 (3DGS convention: scale_0,1,2 together)
i1 = out_props.index('scale_1')
out_props = out_props[:i1 + 1] + ['scale_2'] + out_props[i1 + 1:]

dt = np.dtype([(p, '<f4') for p in out_props])
out = np.empty(n, dt)
for p in out_props:
    out[p] = s2 if p == 'scale_2' else arr[p]

h = ('ply\nformat binary_little_endian 1.0\n'
     'comment 2DGS surfels -> flat 3DGS (scale_2 = min(s0,s1)-{:.1f}, normals dropped)\n'.format(thin) +
     f'element vertex {n}\n' + ''.join(f'property float {p}\n' for p in out_props) + 'end_header\n')
with open(outp, 'wb') as g:
    g.write(h.encode()); out.tofile(g)
print(f'2DGS->3DGS: {n:,} surfels -> flat gaussians ({len(out_props)} props), thin={thin} -> {outp}')
