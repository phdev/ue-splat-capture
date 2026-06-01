"""Rig math: deterministic, correct radii, interleaved held-out split."""
import math

from ue_capture import rig


def test_default_rig_deterministic():
    a = rig.default_rig()
    b = rig.default_rig()
    assert a == b
    assert len(a) > 0


def test_orbit_on_sphere_and_splits():
    center = (0.0, 0.0, 35.0)
    R = 360.0
    poses = rig.orbit_hemisphere(center, R, elevations_deg=(22.0, 48.0), n_azimuth=24)
    assert len(poses) == 48
    for p in poses:
        d = math.dist(p["location_cm"], center)
        assert abs(d - R) < 1e-6
    splits = {p["split"] for p in poses}
    assert splits == {"train", "heldout"}
    n_held = sum(p["split"] == "heldout" for p in poses)
    assert n_held == 12  # every 4th


def test_interior_walk_height_and_count():
    poses = rig.interior_walk([[200, 0, 0], [0, 200, 0], [-200, 0, 0], [0, -200, 0]],
                              n_steps=8, height_cm=130.0)
    assert len(poses) == 8
    assert all(abs(p["location_cm"][2] - 130.0) < 1e-9 for p in poses)
    assert all(p["split"] == "train" for p in poses)
