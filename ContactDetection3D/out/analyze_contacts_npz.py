#!/usr/bin/env python3
"""
Analyze contact_results.npz (in the same folder as this script):
- Distributions: voxel_counts, planarity
- Coordination number per particle + distribution
- Print fabric tensors (unweighted and voxel-count weighted) + eigensystems

Dependencies:
  pip install numpy matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ------------------------
# User settings
# ------------------------
NPZ_NAME = "contact_results.npz"   # must be in the same folder as this script
BINS = 5
SHOW_NORMAL_COMPONENT_HISTS = False  # quick sanity check
LOGY_VOXEL_COUNTS = False          # set True if you have a long tail


def load_npz_local(npz_name: str) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, npz_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{npz_name}' in script folder: {here}")
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def compute_coordination_from_pairs(pairs: np.ndarray):
    """
    pairs: (Nc,2) unique contacts (typically i<j). Returns:
      adj: dict pid -> set(neighbors)
      pids: sorted particle IDs with >=1 contact
      Z: coordination numbers aligned with pids
    """
    pairs = np.asarray(pairs, dtype=int)
    adj = {}
    for i, j in pairs:
        i = int(i); j = int(j)
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    pids = np.array(sorted(adj.keys()), dtype=int)
    Z = np.array([len(adj[int(pid)]) for pid in pids], dtype=int)
    return adj, pids, Z


def print_fabric(F: np.ndarray, title: str):
    print(f"\n{title}")
    with np.printoptions(precision=6, suppress=True):
        print(F)

    evals, evecs = np.linalg.eigh(F)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    print(f"Eigenvalues (desc): {evals}")
    print("Eigenvectors (columns correspond to eigenvalues):")
    with np.printoptions(precision=6, suppress=True):
        print(evecs)


def hist_plot(data, title, xlabel, bins=BINS, logy=False):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        print(f"Skipping plot '{title}' (no data).")
        return

    plt.figure(figsize=(7, 4))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    if logy:
        plt.yscale("log")
    plt.tight_layout()


def main():
    d = load_npz_local(NPZ_NAME)

    pairs = d.get("pairs", np.zeros((0, 2), dtype=int))
    voxel_counts = d.get("voxel_counts", np.array([], dtype=int))
    planarity = d.get("planarity", np.array([], dtype=float))
    normals = d.get("normals", np.zeros((0, 3), dtype=float))

    print(f"Loaded {NPZ_NAME} from the script folder.")
    print(f"  #contacts Nc = {pairs.shape[0]}")

    if voxel_counts.size:
        print(f"  voxel_counts: min={int(voxel_counts.min())} "
              f"median={float(np.median(voxel_counts)):.1f} max={int(voxel_counts.max())}")
    if planarity.size:
        print(f"  planarity: min={float(np.nanmin(planarity)):.3f} "
              f"median={float(np.nanmedian(planarity)):.3f} max={float(np.nanmax(planarity)):.3f}")

    # ---- Distributions ----
    #hist_plot(voxel_counts, "Distribution of contact voxel counts", "voxel_counts (nvox)", bins=BINS, logy=LOGY_VOXEL_COUNTS)
    #hist_plot(planarity, "Distribution of contact planarity", "planarity = 1 - (λ3/λ2)", bins=BINS)

    if SHOW_NORMAL_COMPONENT_HISTS and normals.size:
        # normals stored as (z,y,x) in your pipeline
        nz = normals[:, 0]
        ny = normals[:, 1]
        nx = normals[:, 2]
        hist_plot(nx, "Normal component distribution: nx", "nx", bins=BINS)
        hist_plot(ny, "Normal component distribution: ny", "ny", bins=BINS)
        hist_plot(nz, "Normal component distribution: nz", "nz", bins=BINS)

    # ---- Coordination numbers ----
    if pairs.size:
        adj, pids, Z = compute_coordination_from_pairs(pairs)
        print(f"\nParticles with >=1 contact: {len(pids)}")
        if Z.size:
            print("Coordination number stats:")
            print(f"  mean Z = {float(Z.mean()):.4f}")
            print(f"  median Z = {float(np.median(Z)):.1f}")
            print(f"  min Z = {int(Z.min())}  max Z = {int(Z.max())}")

            # Discrete histogram
            plt.figure(figsize=(4, 5))
            zmin, zmax = int(Z.min()), int(Z.max())
            bins = np.arange(zmin - 0.5, zmax + 1.5, 1.0)

            plt.hist(Z, bins=bins, color='orange', linewidth=1.5, edgecolor='black')

            # ---- Stats ----
            Z_mean = float(np.mean(Z))
            Z_std = float(np.std(Z, ddof=1)) if Z.size > 1 else 0.0  # sample std

            stats_txt = (
                r"$Z$" + f" = {Z_mean:.2f}\n"
                r"$\sigma_Z$" + f" = {Z_std:.2f}"
            )
            plt.text(
                5.8, 50, stats_txt,
                #ha="right", va="top",
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", alpha=1.0),
            )


            plt.xlabel(r"Coordination number $Z_i$", fontsize=16)
            plt.ylabel("Number of contacts", fontsize=16)

            plt.xticks(np.arange(zmin, zmax + 1, 1), fontsize=14)
            plt.yticks(fontsize=14)

            plt.tight_layout()
            plt.savefig('histogram.svg')
    else:
        print("\nNo pairs found; cannot compute coordination numbers.")

    # ---- Fabric tensors ----
    if "fabric_tensor" in d:
        print_fabric(d["fabric_tensor"], "Fabric tensor (unweighted) [stored]")
    elif normals.size:
        F = np.einsum("ni,nj->ij", normals, normals) / normals.shape[0]
        print_fabric(F, "Fabric tensor (unweighted) [recomputed]")
    else:
        print("\nNo normals available; cannot compute fabric tensor.")

    if "fabric_tensor_weighted" in d:
        print_fabric(d["fabric_tensor_weighted"], "Fabric tensor (voxel-count weighted) [stored]")
    elif normals.size and voxel_counts.size == normals.shape[0]:
        w = voxel_counts.astype(float)
        sw = float(w.sum())
        if sw > 0:
            Fw = np.einsum("n,ni,nj->ij", w, normals, normals) / sw
            print_fabric(Fw, "Fabric tensor (voxel-count weighted) [recomputed]")
    else:
        print("\nNo weighted fabric tensor available (missing weights or mismatch).")

    plt.show()


if __name__ == "__main__":
    main()
