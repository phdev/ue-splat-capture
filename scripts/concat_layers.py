"""Layered-splat concat with per-layer dedup/filters — the scene25/26/27 recipe.

Builds one splat from independently trained passes (NEVER joint-train mixed passes —
see CLAUDE.md "LAYERED splats"). Two merge modes per added layer:

  oldest-wins (default): kdtree on everything already merged; drop added gaussians
    with an existing neighbor within --dedup R. Right for ADDITIVE layers (foliage-off
    under-canopy fill) where the base content is good and you only want what's new.

  newest-wins (--repair): the added layer is AUTHORITATIVE inside a repair zone —
    delete EXISTING gaussians within --kill R of new content in the zone, insert the
    new layer wholesale there (oldest-wins outside the zone). Right for REPAIR layers
    re-shooting a badly reconstructed region: oldest-wins would keep the old fat dark
    smudge and drop the crisp replacements (this exact failure shipped in scene26).

Per-layer add filters (safe on a layer where the island-wide versions are not):
  blue-dominance (b>r+0.06 & b>g+0.06 — moss/rock/leaves are never blue-dominant),
  glint (sat>0.6 & val>0.6, or sat>0.9), adds-only SOR (mean 16-NN dist > 0.4m).

COORDS GOTCHA: plys are in ingested coords where Y IS SIGN-FLIPPED vs UE world
(world Y=-51.9m -> ply Y=+51.9). Define zones from layer medians (self-calibrating),
not hand-typed world numbers. Viewer space after the SOG -r -90,0,0 export is
(x, z, -y) of the recentered ply. Recenter every release with the SAME ctr (we use
the prior release's median) so existing camera jsons keep framing identically.

Usage (scene27 reproduction):
  python3 scripts/concat_layers.py \
    --base  out/ed_depth_vanilla/depth_visible_on_30000.ply \
    --add   out/ed_depth_vanilla/depth_visible_off_30000.ply --dedup 0.5 \
    --add   out/ed_depth_vanilla/depth_spire_only_30000.ply  --dedup 0.35 --crop 25 \
    --add   out/ed_depth_vanilla/depth_spire_base_30000.ply  --dedup 0.35 --crop 25 \
            --repair --zone-from-median --zone-r 22 --zone-zmax 38 --kill 0.4 \
    --out /tmp/combined.ply
Then: despike_ply.py (tight knobs, box from the BASE median) -> recenter -> SOG.
"""
import argparse
import sys

import numpy as np
from scipy.spatial import cKDTree

C0 = 0.2820948  # SH DC -> rgb: rgb = 0.5 + C0 * f_dc


def read_ply(p):
    f = open(p, 'rb'); hdr = b''
    while b'end_header\n' not in hdr:
        hdr += f.read(4096)
    f.close()
    header = hdr.split(b'end_header\n', 1)[0]
    lines = header.decode().splitlines()
    props = [l.split()[2] for l in lines if l.startswith('property')]
    n = int(next(l for l in lines if l.startswith('element vertex')).split()[2])
    arr = np.fromfile(p, dtype=np.dtype([(q, '<f4') for q in props]), count=n,
                      offset=len(header) + len(b'end_header\n'))
    return arr, props


def write_ply(p, arr, props):
    with open(p, 'wb') as g:
        g.write(b'ply\nformat binary_little_endian 1.0\n')
        g.write(f'element vertex {len(arr)}\n'.encode())
        g.write((''.join(f'property float {x}\n' for x in props)).encode())
        g.write(b'end_header\n')
        arr.tofile(g)


def xyz(a):
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float32)


def color_filter(arr, label):
    rgb = np.stack([0.5 + C0 * arr['f_dc_0'], 0.5 + C0 * arr['f_dc_1'], 0.5 + C0 * arr['f_dc_2']], 1)
    blue = (rgb[:, 2] > rgb[:, 0] + 0.06) & (rgb[:, 2] > rgb[:, 1] + 0.06)
    rc = np.clip(rgb, 0, 1)
    mx, mn = rc.max(1), rc.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    glint = ((sat > 0.6) & (mx > 0.6)) | (sat > 0.9)
    drop = blue | glint
    print(f"  {label}: blue/glint dropped {int(drop.sum())}")
    return arr[~drop]


def sor(arr, ctx_xyz, label, thresh=0.4):
    a = xyz(arr)
    ctx = np.concatenate([ctx_xyz, a]) if ctx_xyz is not None else a
    d, _ = cKDTree(ctx).query(a, k=17)
    keep = d[:, 1:].mean(1) <= thresh
    print(f"  {label}: SOR dropped {int((~keep).sum())}")
    return arr[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    # repeated layer specs: parse manually so flags attach to the preceding --add
    args, layers = [], []
    argv = sys.argv[1:]
    i = 0
    base = out = None
    cur = None
    while i < len(argv):
        a = argv[i]
        if a == '--base':
            base = argv[i + 1]; i += 2
        elif a == '--out':
            out = argv[i + 1]; i += 2
        elif a == '--add':
            cur = {'path': argv[i + 1], 'dedup': 0.5, 'crop': None, 'repair': False,
                   'zone_r': 22.0, 'zone_zmax': 38.0, 'kill': 0.4}
            layers.append(cur); i += 2
        elif a == '--dedup':
            cur['dedup'] = float(argv[i + 1]); i += 2
        elif a == '--crop':
            cur['crop'] = float(argv[i + 1]); i += 2
        elif a == '--repair':
            cur['repair'] = True; i += 1
        elif a == '--zone-from-median':
            i += 1  # implied; zones always self-calibrate from medians
        elif a == '--zone-r':
            cur['zone_r'] = float(argv[i + 1]); i += 2
        elif a == '--zone-zmax':
            cur['zone_zmax'] = float(argv[i + 1]); i += 2
        elif a == '--kill':
            cur['kill'] = float(argv[i + 1]); i += 2
        else:
            sys.exit(f"unknown arg {a}")
    if not base or not out or not layers:
        sys.exit(__doc__)

    base_arr, props = read_ply(base)
    print(f"base: {len(base_arr)}")
    merged = [base_arr]
    island_xy = np.median(xyz(base_arr)[:, :2], 0)

    for L in layers:
        arr, p2 = read_ply(L['path'])
        assert p2 == props, f"prop mismatch in {L['path']}"
        print(f"layer {L['path']}: {len(arr)}")
        if L['crop']:
            a = xyz(arr); med = np.median(a, 0)
            m = np.all(np.abs(a - med) < np.array([L['crop'], L['crop'], L['crop'] + 5], np.float32), axis=1)
            arr = arr[m]
            print(f"  crop ±{L['crop']}m: {len(arr)}")
        arr = color_filter(arr, "adds")
        merged_xyz = np.concatenate([xyz(m) for m in merged])
        if not L['repair']:
            keep = cKDTree(merged_xyz).query(xyz(arr), k=1)[0] > L['dedup']
            adds = sor(arr[keep], merged_xyz, "adds")
            print(f"  oldest-wins adds: {len(adds)}")
            merged.append(adds)
        else:
            a = xyz(arr)
            feat_xy = np.median(a[:, :2], 0)
            bd = feat_xy - island_xy
            bd = bd / np.linalg.norm(bd)
            inz = (a[:, 2] < L['zone_zmax']) & (((a[:, :2] - feat_xy) @ bd) > -3.0) \
                  & (np.linalg.norm(a[:, :2] - feat_xy, axis=1) < L['zone_r'])
            new_zone = sor(arr[inz], None, "zone-adds")
            nz_xyz = xyz(new_zone)
            tree_new = cKDTree(nz_xyz)
            kept_layers = []
            for m in merged:
                mx = xyz(m)
                m_inz = (mx[:, 2] < L['zone_zmax']) & (((mx[:, :2] - feat_xy) @ bd) > -3.0) \
                        & (np.linalg.norm(mx[:, :2] - feat_xy, axis=1) < L['zone_r'])
                near = np.zeros(len(m), bool)
                if m_inz.any():
                    dd, _ = tree_new.query(mx[m_inz], k=1)
                    idx = np.where(m_inz)[0]
                    near[idx[dd < L['kill']]] = True
                print(f"  newest-wins: killed {int(near.sum())} old in zone")
                kept_layers.append(m[~near])
            merged = kept_layers
            out_zone = arr[~inz]
            merged_xyz = np.concatenate([xyz(m) for m in merged] + [nz_xyz])
            ok = cKDTree(merged_xyz).query(xyz(out_zone), k=1)[0] > L['dedup']
            out_adds = out_zone[ok]
            print(f"  zone adds {len(new_zone)} + out-of-zone adds {len(out_adds)}")
            merged += [new_zone, out_adds]

    combined = np.concatenate(merged)
    print(f"combined: {len(combined)}")
    write_ply(out, combined, props)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
