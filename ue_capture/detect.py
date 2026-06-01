"""Locate UnrealEditor-Cmd inside the installed UE .app bundle (macOS).

Pure Python; safe to import anywhere. `make capture` uses this to decide whether
the UE-dependent capture runs or is skipped-with-warning.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

# Standard Epic install roots on macOS, newest-first preference handled by sort.
_SEARCH_ROOTS = [
    "/Users/Shared/Epic Games",
    "/Applications/Epic Games",
    str(Path.home() / "Applications" / "Epic Games"),
]
_REL = "Engine/Binaries/Mac/UnrealEditor-Cmd"


def find_unreal_cmd(explicit: str | None = None) -> str | None:
    """Return a path to UnrealEditor-Cmd, or None if not found.

    Resolution order: explicit arg -> $UE_CMD -> $UE_ROOT/<rel> -> scan roots.
    """
    cand = explicit or os.environ.get("UE_CMD")
    if cand and Path(cand).exists():
        return cand
    ue_root = os.environ.get("UE_ROOT")
    if ue_root:
        p = Path(ue_root) / _REL
        if p.exists():
            return str(p)
    found = []
    for root in _SEARCH_ROOTS:
        found += glob.glob(str(Path(root) / "UE_*" / _REL))
    # newest version last alphabetically -> prefer it
    found.sort()
    return found[-1] if found else None


def find_uproject(explicit: str | None = None) -> str | None:
    """Project to open for capture. $UE_PROJECT or an explicit path."""
    cand = explicit or os.environ.get("UE_PROJECT")
    if cand and Path(cand).exists():
        return cand
    return None


if __name__ == "__main__":
    cmd = find_unreal_cmd()
    proj = find_uproject()
    print(f"UnrealEditor-Cmd: {cmd or '(not found)'}")
    print(f"uproject:         {proj or '(not set; export UE_PROJECT=...)'}")
    raise SystemExit(0 if cmd else 1)
