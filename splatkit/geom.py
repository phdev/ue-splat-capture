"""Small, dependency-light linear-algebra helpers (numpy only).

These are deliberately self-contained (no scipy / transforms3d / etc.) so the
geometry that the whole pipeline trusts has no transitive dependencies.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(np.asarray(a, float), np.asarray(b, float))


def look_at_basis_ue(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Camera basis in UE convention (Z-up), as world-space unit vectors.

    Returns columns ``(forward, right, up)`` of the actor rotation matrix, i.e.
    the world-space directions of the camera's local +X (forward), +Y (right),
    +Z (up) axes. This matches what ``unreal.Transform.to_matrix()`` yields for
    a camera actor.

    The triple is a *proper* orthonormal basis (det = +1): UE rotators always
    produce proper rotations. UE's "left-handedness" is a convention about how
    these numbers map to the displayed world, not a property of the numeric
    triple itself. The handedness change is applied in ``convert.py``.
    """
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    world_up = np.asarray(world_up, float)
    forward = normalize(target - eye)
    # right = up_hint x forward  -> for forward=+X, up=+Z this gives +Y (UE right)
    right = normalize(cross(world_up, forward))
    up = normalize(cross(forward, right))
    R = np.stack([forward, right, up], axis=1)  # columns
    return R


def ue_rotator_to_basis(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """UE FRotator (degrees) -> basis matrix (columns forward,right,up).

    Mirrors Unreal's FRotationMatrix row layout (rows = forward, right, up basis
    vectors in world space), transposed into column form. Provided so real UE
    captures that hand us rotators (rather than a transform matrix) convert
    identically. Tested in tests/test_convert.py.
    """
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    SP, CP = np.sin(p), np.cos(p)
    SY, CY = np.sin(y), np.cos(y)
    SR, CR = np.sin(r), np.cos(r)
    forward = np.array([CP * CY, CP * SY, SP])
    right = np.array([SR * SP * CY - CR * SY, SR * SP * SY + CR * CY, -SR * CP])
    up = np.array([-(CR * SP * CY + SR * SY), CY * SR - CR * SP * SY, CR * CP])
    return np.stack([forward, right, up], axis=1)  # columns


def is_proper_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """True iff R is a proper, right-handed orthonormal rotation (det≈+1)."""
    R = np.asarray(R, float)[:3, :3]
    orthonormal = np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    return bool(orthonormal and abs(np.linalg.det(R) - 1.0) < tol)


def det3(R: np.ndarray) -> float:
    return float(np.linalg.det(np.asarray(R, float)[:3, :3]))


def invert_rigid(M: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform [[R,t],[0,1]]."""
    M = np.asarray(M, float)
    R = M[:3, :3]
    t = M[:3, 3]
    Mi = np.eye(4)
    Mi[:3, :3] = R.T
    Mi[:3, 3] = -R.T @ t
    return Mi
