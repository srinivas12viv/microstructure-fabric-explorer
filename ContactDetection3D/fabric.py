# fabric.py
import numpy as np
from typing import Dict, Tuple, List


def build_adjacency_from_pairs(pairs: List[Tuple[int, int]]) -> Dict[int, set]:
    """
    pairs are unique (i<j). Returns adjacency dict: adj[i]=set(neighbors).
    """
    adj = {}
    for i, j in pairs:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    return adj


def coordination_numbers(adj: Dict[int, set]) -> Tuple[Dict[int, int], float]:
    Zi = {int(pid): int(len(nbrs)) for pid, nbrs in adj.items()}
    Zmean = float(np.mean(list(Zi.values()))) if Zi else 0.0
    return Zi, Zmean


def fabric_tensor(normals: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    normals: (Nc,3), assumed unit
    weights: (Nc,) optional (e.g., contact voxel counts)
    Returns 3x3 tensor F = sum(w n⊗n)/sum(w) or mean(n⊗n)
    """
    if normals.size == 0:
        return np.zeros((3, 3), dtype=float)

    N = normals.shape[0]
    if weights is None:
        F = np.zeros((3, 3), dtype=float)
        for k in range(N):
            n = normals[k]
            F += np.outer(n, n)
        F /= float(N)
        return F

    w = weights.astype(float)
    sw = float(np.sum(w))
    if sw <= 0:
        return fabric_tensor(normals, weights=None)

    F = np.zeros((3, 3), dtype=float)
    for k in range(N):
        n = normals[k]
        F += w[k] * np.outer(n, n)
    F /= sw
    return F


def fabric_eigensystem(F: np.ndarray):
    """
    Returns eigenvalues (descending), eigenvectors (columns aligned with eigenvalues).
    """
    evals, evecs = np.linalg.eigh(F)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]
