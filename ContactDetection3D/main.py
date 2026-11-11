# main.py
import os
import numpy as np

from config import Config
from io_utils import load_label_tiff, ensure_dir, save_npz, save_contacts_csv
from particle_stats import get_particle_ids, compute_bounding_boxes
from contact_detection import find_contacts_for_particle
from contact_geometry import contact_centroid, pca_contact_frame, planarity_score
from fabric import build_adjacency_from_pairs, coordination_numbers, fabric_tensor, fabric_eigensystem

import viz


def run(cfg: Config):
    ensure_dir(cfg.out_dir)

    labels = load_label_tiff(cfg.tiff_path)
    print(f"Loaded labels: shape={labels.shape} dtype={labels.dtype}")

    particle_ids = get_particle_ids(labels, exclude=cfg.exclude_labels)
    print(f"Found {len(particle_ids)} particle IDs (excluding {cfg.exclude_labels})")

    bboxes = compute_bounding_boxes(labels, particle_ids)
    print(f"Computed bounding boxes for {len(bboxes)} particles")

    # Store contacts uniquely as (i<j) -> coords
    contacts_coords = {}  # (i,j) -> coords (N,3) global zyx
    contact_meta = {}     # (i,j) -> dict info (voxel_count, eigvals, planarity)

    # First loop: per particle i, detect neighbors and contact voxel clouds
    for idx, pid in enumerate(particle_ids, start=1):
        pid = int(pid)
        if pid not in bboxes:
            continue

        contacts_i, nbrs_i, roi_i = find_contacts_for_particle(
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

    centers_mm = np.asarray(centers_mm, dtype=float)
    normals = np.asarray(normals, dtype=float)
    tangents1 = np.asarray(tangents1, dtype=float)
    tangents2 = np.asarray(tangents2, dtype=float)
    voxel_counts = np.asarray(voxel_counts, dtype=int)
    planarity = np.asarray(planarity, dtype=float)
    eigvals_all = np.asarray(eigvals_all, dtype=float)
    pairs_arr = np.asarray(pairs, dtype=int)

    print(f"Contacts retained after min_contact_voxels={cfg.min_contact_voxels}: {len(pairs)}")

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
    if contact_rows:
        save_contacts_csv(out_csv, contact_rows)
        print(f"Saved CSV: {out_csv}")

    # Debug visualization (optional)
    # Debug visualization (optional)
    # Priority:
    #   1) debug_pair if provided
    #   2) debug_contact_idx if provided (indexes into *retained* contacts)
    debug_key = None

    # (1) Manual pair override
    if getattr(cfg, "debug_pair", None) is not None:
        di, dj = cfg.debug_pair
        debug_key = (min(int(di), int(dj)), max(int(di), int(dj)))

    # (2) Debug by retained-contact index (after min_contact_voxels)
    elif getattr(cfg, "debug_contact_idx", None) is not None:
        idx = int(cfg.debug_contact_idx)

        # pairs is the retained list in the second loop (aligned with normals, centers_mm, etc.)
        if idx < 0 or idx >= len(pairs):
            print(
                f"debug_contact_idx={idx} out of range for retained contacts. "
                f"Valid: 0..{len(pairs) - 1} (retained contacts={len(pairs)})."
            )
        else:
            i, j = pairs[idx]
            debug_key = (int(i), int(j))
            # Helpful printout
            print(
                f"Debugging retained contact idx={idx}: pair={debug_key}, "
                f"nvox={int(voxel_counts[idx])}, planarity={float(planarity[idx]):.4f}"
            )

    # If we identified a key, visualize it
    if debug_key is not None:
        if debug_key in contacts_coords:
            coords = contacts_coords[debug_key]

            # Use voxel centroid for slice indexing
            center_vox = coords.mean(axis=0)

            # Optional 2D slices
            viz.plot_contact_slices(labels, debug_key[0], debug_key[1], coords, center_vox, pad=cfg.viz_pad_vox)

            # Compute vectors for 3D plot
            c_mm = contact_centroid(coords, cfg.voxel_size_mm)
            t1, t2, n, _ = pca_contact_frame(coords, cfg.voxel_size_mm)

            # 3D scatter debug (points)
            #viz.plot_contact_3d(
            #    labels=labels,
            #    pid=debug_key[0],
            #    nid=debug_key[1],
            #    contact_coords_zyx=coords,
            #    voxel_size_mm=cfg.voxel_size_mm,
            #    center_mm=c_mm,
            #    n=n,
            #    t1=t1,
            #    stride=cfg.viz_stride,
            #    roi_pad_vox=cfg.viz_pad_vox,
            #)

            # 3D voxel-cube debug (cubes)
            # Uncomment if you want cubes instead of points (or run both)
            viz.plot_contact_voxel_cubes(
                 labels=labels,
                 pid=debug_key[0],
                 nid=debug_key[1],
                 contact_coords_zyx=coords,
                 voxel_size_mm=cfg.voxel_size_mm,
                 center_mm=c_mm,
                 roi_pad_vox=cfg.viz_pad_vox,
                 alpha_particles=0.5,
                 alpha_contact=0.9,
             )

        else:
            print(f"Debug contact {debug_key} not found in detected contacts_coords.")

    return {
        "Zmean": Zmean,
        "F": F,
        "Fw": Fw,
        "n_contacts": len(pairs),
        "n_particles": len(particle_ids),
    }


if __name__ == "__main__":
    # Edit these defaults or replace with argparse as needed
    cfg = Config(
        tiff_path="SCP_labeled_N5_ov1.tif",
        out_dir="out",
        voxel_size_mm= 1.0,  #/ 40.0,
        shell_thickness_vox=1,
        connectivity=26,
        min_contact_voxels=2,
        debug_contact_idx=50, # 0-based index into retained contacts (after min_contact_voxels)
        #debug_pair=(1,2),  # e.g., (12, 45)
    )
    run(cfg)
