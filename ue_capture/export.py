"""Export camera poses + intrinsics from Unreal into the neutral ue_poses.json.

Reads the AUTHORITATIVE pose straight off each camera actor's world transform
(forward/right/up vectors + location), so whatever UE actually rendered with is
what gets exported. Pure stdlib for the math + json; `unreal` is used only to
read actor vectors.
"""
from __future__ import annotations

import json
import math


def compute_intrinsics(w, h, hfov_deg):
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return {"w": int(w), "h": int(h), "fl_x": float(fx), "fl_y": float(fx),
            "cx": w / 2.0, "cy": h / 2.0}


def basis_from_actor(unreal, actor):
    """3x3 basis with columns = world directions of UE local axes (fwd,right,up)."""
    f = actor.get_actor_forward_vector()
    r = actor.get_actor_right_vector()
    u = actor.get_actor_up_vector()
    return [[f.x, r.x, u.x], [f.y, r.y, u.y], [f.z, r.z, u.z]]


def location_from_actor(unreal, actor):
    loc = actor.get_actor_location()
    return [loc.x, loc.y, loc.z]


def write_ue_poses(path, w, h, hfov_deg, frames, scene_meta):
    """frames: list of {file_path, split, location_cm, basis_ue}."""
    doc = {
        "intrinsics": compute_intrinsics(w, h, hfov_deg),
        "background": scene_meta["background"],
        "aabb_min_cm": scene_meta["aabb_min_cm"],
        "aabb_max_cm": scene_meta["aabb_max_cm"],
        "fiducials": scene_meta["fiducials"],
        "primitives": scene_meta.get("primitives", []),
        "frames": frames,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return doc
