# contact_geometry.py
import numpy as np
from typing import Tuple, Optional


def contact_centroid(coords_zyx: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    """
    coords_zyx: (N,3) integer voxel coords in (z,y,x).
    Returns centroid in mm in (z,y,x) coordinates.
    """
    if coords_zyx.ndim != 2 or coords_zyx.shape[1] != 3:
        raise ValueError("coords_zyx must be (N,3)")
    c_vox = coords_zyx.mean(axis=0)
    return c_vox * float(voxel_size_mm)


def pca_contact_frame(coords_zyx: np.ndarray, voxel_size_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA on point cloud (in mm). Returns:
      t1 (3,), t2 (3,), n (3,), eigvals (3,) sorted descending.
    Convention:
      - t1: first PC (largest variance)  -> tangent direction
      - n : third PC (smallest variance) -> normal direction
    """
    pts = coords_zyx.astype(np.float64) * float(voxel_size_mm)  # (N,3), in mm
    mu = pts.mean(axis=0)
    X = pts - mu

    # Covariance (3x3)
    C = (X.T @ X) / max(1, X.shape[0])

    # Eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(eigvals)[::-1]     # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    t1 = eigvecs[:, 0]
    t2 = eigvecs[:, 1]
    n  = eigvecs[:, 2]

    # Normalize for safety
    t1 = t1 / (np.linalg.norm(t1) + 1e-15)
    t2 = t2 / (np.linalg.norm(t2) + 1e-15)
    n  = n  / (np.linalg.norm(n)  + 1e-15)

    return t1, t2, n, eigvals


def planarity_score(eigvals_desc: np.ndarray, eps: float = 1e-15) -> float:
    """
    Simple planarity indicator: 1 - (lambda3/lambda2). Closer to 1 => more planar.
    Assumes eigvals sorted descending: l1 >= l2 >= l3.
    """
    l1, l2, l3 = eigvals_desc
    return float(1.0 - (l3 / (l2 + eps)))
