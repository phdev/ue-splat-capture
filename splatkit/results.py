"""Shared helpers for writing per-tier result JSON and pass/fail gating.

Each verification tier writes ``results/<tier>.json``. A tier may assert several
things; each is a *check* of the form::

    {"metric": str, "value": float, "threshold": float, "op": str, "pass": bool}

and the tier passes iff every check passes. The file also carries the single
headline ``metric/value/threshold/pass`` (the goal's required shape) drawn from
the first check, plus the full ``checks`` list.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

_OPS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
    "==": lambda v, t: v == t,
}


def check(metric: str, value, threshold, op: str, **extra) -> dict:
    value = float(value)
    threshold = float(threshold)
    passed = bool(_OPS[op](value, threshold))
    d = {"metric": metric, "value": value, "threshold": threshold,
         "op": op, "pass": passed}
    d.update(extra)
    return d


def write_tier(tier: str, checks: list[dict], **meta) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overall = all(c["pass"] for c in checks) if checks else False
    head = checks[0] if checks else {"metric": tier, "value": 0.0,
                                     "threshold": 0.0, "pass": overall}
    doc = {
        "tier": tier,
        "metric": head["metric"], "value": head["value"],
        "threshold": head["threshold"], "pass": overall,
        "checks": checks,
    }
    doc.update(meta)
    path = RESULTS_DIR / f"{tier}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return doc


def print_tier(doc: dict) -> None:
    status = "PASS" if doc["pass"] else "FAIL"
    print(f"[{doc['tier']}] {status}")
    for c in doc.get("checks", []):
        mark = "ok " if c["pass"] else "XX "
        print(f"  {mark}{c['metric']:<26} {c['value']:.5g} {c['op']} "
              f"{c['threshold']:.5g}"
              + (f"   ({c['note']})" if c.get("note") else ""))


def env_device_default() -> str:
    """Trainer/eval device. CPU forced for deterministic CI when SPLAT_CPU=1."""
    if os.environ.get("SPLAT_CPU") == "1":
        return "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
