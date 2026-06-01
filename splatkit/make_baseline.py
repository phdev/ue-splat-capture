"""Freeze the current results/<tier>.json into results/baseline.json.

`make verify` then flags any future run that regresses beyond per-metric
tolerance (see splatkit/verify.py)."""
from __future__ import annotations

import json

from . import results as R


def main() -> int:
    baseline = {}
    for tier in ("t0", "t1", "t2", "t3"):
        p = R.RESULTS_DIR / f"{tier}.json"
        if p.exists():
            baseline[tier] = json.loads(p.read_text())
    out = R.RESULTS_DIR / "baseline.json"
    out.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    tiers = ", ".join(sorted(baseline))
    print(f"wrote {out} with tiers: {tiers}")
    return 0 if baseline else 1


if __name__ == "__main__":
    raise SystemExit(main())
