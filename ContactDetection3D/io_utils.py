# io_utils.py
import os
import numpy as np
import tifffile as tiff


def load_label_tiff(path: str) -> np.ndarray:
    """
    Load a labeled TIFF stack.
    Expected axis order: (z, y, x).
    Returns an integer array labels[z,y,x].
    """
    labels = tiff.imread(path)
    if labels.ndim != 3:
        raise ValueError(f"Expected a 3D labeled TIFF. Got shape={labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labeled TIFF must contain integer particle IDs, not floats")
    if labels.size == 0 or np.any(labels < 0):
        raise ValueError("Labels must be nonempty and nonnegative")
    return labels


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz(out_path: str, **arrays) -> None:
    """
    Save results in a compressed npz.
    """
    np.savez_compressed(out_path, **arrays)


def save_contacts_csv(out_path: str, contact_rows: list) -> None:
    """
    Save a simple CSV with per-contact summary.
    contact_rows: list of dicts with same keys.
    """
    import csv
    fieldnames = (list(contact_rows[0].keys()) if contact_rows else
                  "i j nvox cz_mm cy_mm cx_mm nz ny nx t1z t1y t1x planarity eig1 eig2 eig3".split())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in contact_rows:
            w.writerow(r)
