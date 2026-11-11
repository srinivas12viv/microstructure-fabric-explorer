"""Detect contacts in a segmented TIFF; export geometry, fabric and toolkit inputs."""
from pathlib import Path
import argparse
import json
import sys

# Support both direct script invocation and package imports from any directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "ContactDetection3D"

import os
import numpy as np

from .config import Config
from .io_utils import load_label_tiff, ensure_dir, save_npz, save_contacts_csv
from .particle_stats import get_particle_ids, compute_bounding_boxes
from .contact_detection import find_contacts_for_particle
from .contact_geometry import contact_centroid, pca_contact_frame, planarity_score
from .fabric import build_adjacency_from_pairs, coordination_numbers, fabric_tensor, fabric_eigensystem

from . import viz


def run(cfg: Config):
    """Run detection with Config; write outputs and return summary statistics."""
    ensure_dir(cfg.out_dir)

    labels = load_label_tiff(cfg.tiff_path)
    print(f"Loaded labels: shape={labels.shape} dtype={labels.dtype}")

    particle_ids = get_particle_ids(labels, exclude=cfg.exclude_labels)
    print(f"Found {len(particle_ids)} particle IDs (excluding {cfg.exclude_labels})")

    bboxes = compute_bounding_boxes(labels, particle_ids)
    print(f"Computed bounding boxes for {len(bboxes)} particles")

    # Store contacts uniquely as (i<j) -> coords
    contacts_coords = {}  # (i,j) -> coords (N,3) global zyx

    # First loop: per particle i, detect neighbors and contact voxel clouds
    for idx, pid in enumerate(particle_ids, start=1):
        pid = int(pid)
        if pid not in bboxes:
            continue

        contacts_i, _, _ = find_contacts_for_particle(
            labels=labels,
            pid=pid,
            bbox_pid=bboxes[pid],
            shell_thickness=cfg.shell_thickness_vox,
            connectivity=cfg.connectivity,
            exclude_labels=cfg.exclude_labels,
        )

        # Store contacts only once: pid < nid
        for nid, coords in contacts_i.items():
            nid = int(nid)
            if nid <= pid:
                continue
            key = (pid, nid)

            # Combine if we somehow encounter duplicates (should be rare)
            if key in contacts_coords:
                contacts_coords[key] = np.vstack([contacts_coords[key], coords])
            else:
                contacts_coords[key] = coords

        if idx % 200 == 0 or idx == len(particle_ids):
            print(f"Processed {idx}/{len(particle_ids)} particles")

    print(f"Unique particle-pair contacts found: {len(contacts_coords)}")

    # Second loop: per contact pair, compute centroid + PCA directions
    keys = sorted(contacts_coords.keys())
    centers_mm = []
    normals = []
    tangents1 = []
    tangents2 = []
    voxel_counts = []
    planarity = []
    eigvals_all = []
    pairs = []

    contact_rows = []

    for (i, j) in keys:
        coords = contacts_coords[(i, j)]
        nvox = int(coords.shape[0])
        if nvox < cfg.min_contact_voxels:
            continue

        c_mm = contact_centroid(coords, cfg.voxel_size_mm)
        t1, t2, n, eigvals = pca_contact_frame(coords, cfg.voxel_size_mm)
        # A line or point has no uniquely determined plane normal.
        if eigvals[1] <= np.finfo(float).eps * max(eigvals[0], np.finfo(float).tiny) * 100:
            continue
        pl = planarity_score(eigvals)

        pairs.append((i, j))
        centers_mm.append(c_mm)
        normals.append(n)
        tangents1.append(t1)
        tangents2.append(t2)
        voxel_counts.append(nvox)
        planarity.append(pl)
        eigvals_all.append(eigvals)

        contact_rows.append({
            "i": i, "j": j,
            "nvox": nvox,
            "cz_mm": c_mm[0], "cy_mm": c_mm[1], "cx_mm": c_mm[2],
            "nz": n[0], "ny": n[1], "nx": n[2],
            "t1z": t1[0], "t1y": t1[1], "t1x": t1[2],
            "planarity": pl,
            "eig1": float(eigvals[0]), "eig2": float(eigvals[1]), "eig3": float(eigvals[2]),
        })

    centers_mm = np.asarray(centers_mm, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    tangents1 = np.asarray(tangents1, dtype=float).reshape(-1, 3)
    tangents2 = np.asarray(tangents2, dtype=float).reshape(-1, 3)
    voxel_counts = np.asarray(voxel_counts, dtype=int)
    planarity = np.asarray(planarity, dtype=float)
    eigvals_all = np.asarray(eigvals_all, dtype=float).reshape(-1, 3)
    pairs_arr = np.asarray(pairs, dtype=int).reshape(-1, 2)

    print(f"Contacts retained after size/rank filtering, min_contact_voxels={cfg.min_contact_voxels}: {len(pairs)}")

    # Coordination + fabric
    adj = build_adjacency_from_pairs(pairs)
    Zi, Zmean = coordination_numbers(adj)
    print(f"Mean coordination number Z = {Zmean:.4f}  (over {len(Zi)} particles with >=1 contact)")

    F = fabric_tensor(normals, weights=None)
    Fw = fabric_tensor(normals, weights=voxel_counts.astype(float))
    evalsF, evecsF = fabric_eigensystem(F)
    evalsFw, evecsFw = fabric_eigensystem(Fw)

    # Save outputs
    out_npz = os.path.join(cfg.out_dir, "contact_results.npz")
    save_npz(
        out_npz,
        particle_ids=particle_ids,
        voxel_size_mm=np.asarray(cfg.voxel_size_mm),
        pairs=pairs_arr,
        centers_mm=centers_mm,
        normals=normals,
        tangents1=tangents1,
        tangents2=tangents2,
        voxel_counts=voxel_counts,
        planarity=planarity,
        eigvals=eigvals_all,
        fabric_tensor=F,
        fabric_tensor_weighted=Fw,
        fabric_evals=evalsF,
        fabric_evecs=evecsF,
        fabric_evals_weighted=evalsFw,
        fabric_evecs_weighted=evecsFw,
    )
    print(f"Saved NPZ: {out_npz}")

    out_csv = os.path.join(cfg.out_dir, "contacts_summary.csv")
    save_contacts_csv(out_csv, contact_rows)
    # Toolkit readers expect XYZ, whereas image-space arrays above use ZYX.
    # Unit weights describe geometric orientation only; no forces are measured.
    np.savetxt(Path(cfg.out_dir) / "normals.txt",
               np.column_stack([normals[:, ::-1], np.ones(len(normals))]),
               header="nx ny nz f", comments="")
    summary = {
        "n_particles": len(particle_ids), "n_contacts": len(pairs),
        "candidate_pairs": len(keys), "discarded_pairs": len(keys) - len(pairs),
        "mean_coordination_contacting": Zmean,
        "mean_coordination_all": 2 * len(pairs) / len(particle_ids) if len(particle_ids) else 0.0,
        "coordinate_order": "zyx", "voxel_size_mm": cfg.voxel_size_mm,
        "shell_thickness_vox": cfg.shell_thickness_vox, "connectivity": cfg.connectivity,
        "min_contact_voxels": cfg.min_contact_voxels,
        "fabric_tensor": F.tolist(), "fabric_tensor_weighted": Fw.tolist(),
        "orientation_export": "normals.txt: XYZ, unit weights (not measured forces)",
    }
    (Path(cfg.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savetxt(Path(cfg.out_dir) / "coordination.csv",
               np.asarray([(int(pid), Zi.get(int(pid), 0)) for pid in particle_ids], dtype=int).reshape(-1, 2),
               delimiter=",", header="particle_id,coordination", comments="", fmt="%d")
    if cfg.plots:
        viz.save_overview(centers_mm, normals, [Zi.get(int(pid), 0) for pid in particle_ids],
                          cfg.out_dir, show=cfg.show)
    debug_key = cfg.debug_pair
    if debug_key is None and cfg.debug_contact_idx is not None:
        if not 0 <= cfg.debug_contact_idx < len(pairs):
            raise ValueError("debug-contact-index is outside the retained contact range")
        debug_key = pairs[cfg.debug_contact_idx]
    if debug_key is not None:
        debug_key = tuple(sorted(debug_key))
        if debug_key not in pairs:
            raise ValueError(f"Debug pair {debug_key} is not a retained contact")
        coords = contacts_coords[debug_key]
        viz.plot_contact_slices(labels, *debug_key, coords, coords.mean(axis=0),
                                pad=cfg.viz_pad_vox, output=Path(cfg.out_dir) / "contact-slices.png", show=cfg.show)
        viz.plot_contact_voxel_cubes(labels, *debug_key, coords, cfg.voxel_size_mm,
                                     contact_centroid(coords, cfg.voxel_size_mm),
                                     roi_pad_vox=cfg.viz_pad_vox,
                                     output=Path(cfg.out_dir) / "contact-voxels.png", show=cfg.show)
    return summary


def main():
    """Command-line entry point; defaults are independent of the working directory."""
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=defaults.tiff_path, help="Integer labeled 3D TIFF (ZYX)")
    parser.add_argument("--output-dir", default=defaults.out_dir)
    parser.add_argument("--voxel-size-mm", type=float, default=defaults.voxel_size_mm,
                        help="Isotropic voxel spacing in mm (default 1 is an uncalibrated placeholder)")
    parser.add_argument("--shell-thickness", type=int, default=1, help="Dilation iterations, >=1")
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument("--min-contact-voxels", type=int, default=3)
    parser.add_argument("--exclude-labels", type=int, nargs="+", default=[0])
    debug = parser.add_mutually_exclusive_group()
    debug.add_argument("--debug-pair", type=int, nargs=2)
    debug.add_argument("--debug-contact-index", type=int)
    parser.add_argument("--viz-pad", type=int, default=10)
    parser.add_argument("--show", action="store_true", help="Also open Matplotlib windows")
    parser.add_argument("--no-plots", action="store_true", help="Skip overview figures")
    args = parser.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    try:
        result = run(Config(tiff_path=args.input, out_dir=args.output_dir,
                            voxel_size_mm=args.voxel_size_mm, shell_thickness_vox=args.shell_thickness,
                            connectivity=args.connectivity, min_contact_voxels=args.min_contact_voxels,
                            exclude_labels=tuple(args.exclude_labels), debug_pair=args.debug_pair,
                            debug_contact_idx=args.debug_contact_index, viz_pad_vox=args.viz_pad,
                            show=args.show, plots=not args.no_plots))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
