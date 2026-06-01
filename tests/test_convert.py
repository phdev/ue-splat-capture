"""T0 -- pure-math tests for the coordinate conversion. No UE, no fixtures.

Covers: world round-trip, intrinsics, proper-rotation (handedness) gate,
known-good hand-computed fixture, and reprojection agreement with an
INDEPENDENT UE-native projector defined right here in the test (so the two
implementations can't share a bug). Negative tests prove the gates have teeth.
"""
import numpy as np
import pytest

from splatkit import convert, geom
from splatkit.reproject import project_opencv


# --- independent UE-native forward projector (separate impl from convert.py) --
def project_ue_native(location_cm, basis_ue, intr, P_cm):
    rel = np.asarray(P_cm, float) - np.asarray(location_cm, float)
    f, r, u = basis_ue[:, 0], basis_ue[:, 1], basis_ue[:, 2]
    d = rel @ f
    a = rel @ r
    b = rel @ u
    px = intr["cx"] + intr["fl_x"] * (a / d)
    py = intr["cy"] - intr["fl_y"] * (b / d)
    return np.array([px, py]), d


def _cameras():
    """A spread of cameras (look-at) around a scene centre, none degenerate."""
    target = np.array([0.0, 0.0, 40.0])
    eyes = [
        (300, 0, 120), (0, 300, 120), (-280, 90, 200),
        (150, -260, 80), (-200, -200, 260), (260, 220, 60),
    ]
    cams = []
    for e in eyes:
        cams.append((np.array(e, float), geom.look_at_basis_ue(e, target)))
    return cams


def _points():
    return np.array([
        [0, 0, 40], [120, 60, 10], [-90, 110, 70], [40, -150, 5],
        [-130, -70, 130], [200, 10, 90], [10, 200, 20],
    ], float)


INTR = convert.intrinsics_from_hfov(800, 800, 90.0)


def test_world_roundtrip():
    P = np.array([[1234.5, -678.9, 42.0], [0, 0, 0], [-100, 200, -300]])
    back = convert.world_point_to_ue(convert.ue_point_to_world(P))
    assert np.allclose(back, P, atol=1e-9)


def test_world_flip_changes_handedness():
    # exactly one axis negated -> orientation-reversing linear map
    assert np.linalg.det(convert.WORLD_MAP3) < 0


def test_intrinsics_known_values():
    intr = convert.intrinsics_from_hfov(800, 800, 90.0)
    assert intr["fl_x"] == pytest.approx(400.0)
    assert intr["fl_y"] == pytest.approx(400.0)
    assert intr["cx"] == pytest.approx(400.0)
    assert intr["cy"] == pytest.approx(400.0)
    with pytest.raises(ValueError):
        convert.intrinsics_from_hfov(0, 800, 90.0)
    with pytest.raises(ValueError):
        convert.intrinsics_from_hfov(800, 800, 200.0)


def test_converted_pose_is_proper_rotation():
    for loc, basis in _cameras():
        M = convert.ue_camera_to_c2w(loc, basis)
        assert geom.is_proper_rotation(M), f"det={geom.det3(M)}"
        assert geom.det3(M) == pytest.approx(1.0, abs=1e-6)


def test_known_good_fixture():
    # camera at origin looking +X; hand-computed expected outputs.
    basis = geom.look_at_basis_ue((0, 0, 0), (1, 0, 0))
    assert np.allclose(basis, np.eye(3), atol=1e-12)
    M = convert.ue_camera_to_c2w((0, 0, 0), basis)
    expected = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    assert np.allclose(M, expected, atol=1e-12)
    Pw = convert.ue_point_to_world([1000, 200, 50])
    assert np.allclose(Pw, [10.0, -2.0, 0.5])
    uv, depth = project_opencv(M, INTR, Pw)
    assert np.allclose(uv[0], [480.0, 380.0], atol=1e-9)
    assert depth[0] == pytest.approx(10.0)


def test_reprojection_matches_independent_ue_native():
    """Production OpenCV projection == independent UE-native projection."""
    pts = _points()
    worst = 0.0
    for loc, basis in _cameras():
        M = convert.ue_camera_to_c2w(loc, basis)
        uv_cv, _ = project_opencv(M, INTR, convert.ue_point_to_world(pts))
        for j, P in enumerate(pts):
            uv_ue, d = project_ue_native(loc, basis, INTR, P)
            if d <= 1.0:
                continue  # behind/at camera
            worst = max(worst, float(np.linalg.norm(uv_cv[j] - uv_ue)))
    assert worst < 1e-6, f"worst reprojection disagreement {worst} px"


def test_rotator_matches_lookat_yaw90():
    # yaw=90deg about +Z turns forward from +X to +Y.
    basis = geom.ue_rotator_to_basis(0.0, 0.0, 90.0)
    assert np.allclose(basis[:, 0], [0, 1, 0], atol=1e-9)   # forward -> +Y
    assert geom.is_proper_rotation(basis)
    # converting a rotator-derived camera still reprojects exactly.
    loc = np.array([100.0, -50.0, 60.0])
    M = convert.ue_camera_to_c2w(loc, basis)
    P = np.array([400.0, 300.0, 20.0])
    uv_cv, _ = project_opencv(M, INTR, convert.ue_point_to_world(P))
    uv_ue, d = project_ue_native(loc, basis, INTR, P)
    assert d > 1.0
    assert np.allclose(uv_cv[0], uv_ue, atol=1e-7)


# --------------------------- negative tests (teeth) ------------------------- #
def test_no_flip_converter_is_improper_rotation():
    """The 'treat UE as already RH' bug yields det=-1 -> caught by the gate."""
    for loc, basis in _cameras():
        M = convert._bad_no_handedness_flip(loc, basis)
        assert not geom.is_proper_rotation(M)
        assert geom.det3(M) == pytest.approx(-1.0, abs=1e-6)


def test_build_transforms_rejects_improper(monkeypatch):
    monkeypatch.setattr(convert, "ue_camera_to_c2w", convert._bad_no_handedness_flip)
    loc, basis = _cameras()[0]
    frames = [{"file_path": "x.png", "location_cm": loc, "basis_ue": basis}]
    with pytest.raises(ValueError, match="handedness"):
        convert.build_transforms(INTR, frames)


def test_asymmetric_flip_breaks_reprojection():
    """Flipping points but not the basis -> large reprojection error."""
    pts = _points()
    worst = 0.0
    for loc, basis in _cameras():
        M = convert._bad_asymmetric_flip(loc, basis)
        uv_cv, _ = project_opencv(M, INTR, convert.ue_point_to_world(pts))
        for j, P in enumerate(pts):
            uv_ue, d = project_ue_native(loc, basis, INTR, P)
            if d <= 1.0:
                continue
            worst = max(worst, float(np.linalg.norm(uv_cv[j] - uv_ue)))
    assert worst > 5.0, f"asymmetric-flip bug should diverge, got {worst} px"
