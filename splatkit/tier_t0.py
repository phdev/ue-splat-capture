"""T0 runner: run the pure-math coordinate tests and write results/t0.json."""
from __future__ import annotations

import pathlib

import pytest

from . import results as R


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    code = int(pytest.main(["-q", str(root / "tests" / "test_convert.py")]))
    checks = [R.check("pytest_exit_code", float(code), 0.0, "==",
                      note="T0 coordinate-math tests (round-trip, handedness, reproj)")]
    doc = R.write_tier("t0", checks)
    R.print_tier(doc)
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
