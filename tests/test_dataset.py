"""T2 robustness: schema + coverage gates must catch real defects."""
import copy
import json
from pathlib import Path

import numpy as np

from splatkit import convert, coverage, schema

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "selftest"


def _load():
    doc = json.loads((FIX / "transforms.json").read_text())
    scene = json.loads((FIX / "scene.json").read_text())
    return doc, scene


def test_schema_ok_on_fixtures():
    doc, _ = _load()
    res = schema.validate(doc, FIX)
    assert res["ok"], res["issues"]
    assert res["images_ok"] == res["n_frames"] == len(doc["frames"])
    assert res["poses_ok"] == res["n_frames"]
    assert res["n_splits"] == 2


def test_schema_detects_missing_key():
    doc, _ = _load()
    d = copy.deepcopy(doc)
    del d["fl_x"]
    assert not schema.validate(d, FIX)["ok"]


def test_schema_detects_improper_pose():
    doc, _ = _load()
    d = copy.deepcopy(doc)
    M = np.eye(4)
    M[:3, :3] = np.diag([1.0, 1.0, -1.0])  # det = -1, mirrored
    d["frames"][0]["transform_matrix"] = M.tolist()
    res = schema.validate(d, FIX)
    assert res["poses_ok"] == res["n_frames"] - 1
    assert not res["ok"]


def test_schema_detects_missing_image():
    doc, _ = _load()
    d = copy.deepcopy(doc)
    d["frames"][0]["file_path"] = "images/train/does_not_exist.png"
    res = schema.validate(d, FIX)
    assert res["images_ok"] == res["n_frames"] - 1


def test_coverage_full_on_fixtures():
    doc, _ = _load()
    cov = coverage.frustum_coverage(doc)
    assert cov["fraction"] >= 0.99
    assert cov["min_views_per_point"] >= 1


def test_camera_inside_geometry_detected():
    doc, scene = _load()
    d = copy.deepcopy(doc)
    # Put a camera dead-centre inside the big red sphere (center 70,40,45 cm).
    sphere = next(p for p in scene["primitives"]
                  if p["type"] == "sphere" and "fiducial_id" not in p)
    center_m = convert.ue_point_to_world(sphere["center"])
    M = np.asarray(d["frames"][0]["transform_matrix"], float)
    M[:3, 3] = center_m
    d["frames"][0]["transform_matrix"] = M.tolist()
    geo = coverage.cameras_inside_geometry(d, scene)
    assert geo["n_inside"] >= 1


def test_camera_below_ground_detected():
    doc, scene = _load()
    d = copy.deepcopy(doc)
    M = np.asarray(d["frames"][0]["transform_matrix"], float)
    M[:3, 3] = [0.0, 0.0, -0.5]  # half a metre underground
    d["frames"][0]["transform_matrix"] = M.tolist()
    assert coverage.cameras_inside_geometry(d, scene)["n_inside"] >= 1
