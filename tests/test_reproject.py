"""T1 robustness, exercised on the committed fixtures.

Confirms the exported poses reproject the known fiducials to sub-pixel accuracy,
AND that the reprojection gate genuinely catches an asymmetric handedness bug on
real exported data (not just synthetic unit-test cameras).
"""
import json
from pathlib import Path

import numpy as np

from splatkit import convert
from splatkit.reproject import evaluate_doc

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "selftest"


def _load():
    doc = json.loads((FIX / "transforms.json").read_text())
    scene = json.loads((FIX / "scene.json").read_text())
    fid = np.array([f["loc_cm"] for f in scene["fiducials"]], float)
    return doc, fid


def test_fixtures_reproject_subpixel():
    doc, fid = _load()
    per_pose, gmean, gmax, n = evaluate_doc(doc, fid)
    assert n > 100, "expected many fiducial observations"
    assert gmax < 1e-6, f"max per-pose mean reprojection {gmax} px"
    assert gmean < 1e-6


def test_asymmetric_flip_detected_on_fixtures():
    doc, fid = _load()
    _, _, gmax_bad, _ = evaluate_doc(doc, fid, converter=convert._bad_asymmetric_flip)
    assert gmax_bad > 5.0, f"handedness bug should diverge, got {gmax_bad} px"


def test_no_flip_converter_is_improper_on_fixtures():
    """The no-flip bug reprojects fine (global mirror) but is left-handed."""
    from splatkit import geom
    doc, _ = _load()
    fr = doc["frames"][0]
    M = convert._bad_no_handedness_flip(np.array(fr["location_cm"]),
                                        np.array(fr["basis_ue"]))
    assert not geom.is_proper_rotation(M)
