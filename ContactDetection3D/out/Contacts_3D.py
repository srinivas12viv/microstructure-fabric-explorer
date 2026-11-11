#!/usr/bin/env python3
"""
3D plot of contact centers with symmetric contact-direction line segments (+/- normal).

Reads contacts_summary.csv produced by your pipeline.
Expected columns:
  centers (mm): cx_mm, cy_mm, cz_mm
  normals:      nx, ny, nz   (unit vectors)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------
# User inputs
# ------------------------
CSV_FILE = "contacts_summary.csv"

HALF_LEN_MM = 8   # half-length of the line segment in mm (total length = 2*HALF_LEN_MM)
STRIDE = 1           # plot every STRIDE-th contact (increase if you have many)
POINT_SIZE = 6
LINE_WIDTH = 1
ALPHA = 0.8


def set_equal_aspect(ax, x, y, z):
    """Set equal scaling for x,y,z axes in a 3D matplotlib plot."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def main():
    df = pd.read_csv(CSV_FILE)

    # Centers (mm)
    cx = df["cx_mm"].to_numpy()
    cy = df["cy_mm"].to_numpy()
    cz = df["cz_mm"].to_numpy()

    # Normals (unit vectors)
    nx = df["nx"].to_numpy()
    ny = df["ny"].to_numpy()
    nz = df["nz"].to_numpy()

    # Subsample if needed
    idx = np.arange(0, len(df), max(1, int(STRIDE)))
    cx, cy, cz = cx[idx], cy[idx], cz[idx]
    nx, ny, nz = nx[idx], ny[idx], nz[idx]

    # Build symmetric endpoints: center +/- HALF_LEN_MM * n
    x0 = cx - HALF_LEN_MM * nx
    y0 = cy - HALF_LEN_MM * ny
    z0 = cz - HALF_LEN_MM * nz

    x1 = cx + HALF_LEN_MM * nx
    y1 = cy + HALF_LEN_MM * ny
    z1 = cz + HALF_LEN_MM * nz

    # Plot
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Centers
    ax.scatter(cx, cy, cz, s=POINT_SIZE, alpha=ALPHA)

    # Line segments (one segment per contact)
    # Plotting in a loop is fine for a few thousand; if you have >50k contacts,
    # switch to a Line3DCollection approach.
    for k in range(len(cx)):
        ax.plot([x0[k], x1[k]], [y0[k], y1[k]], [z0[k], z1[k]], linewidth=LINE_WIDTH, alpha=ALPHA)

    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ax.set_zlabel("z (voxels)")
    ax.set_title("Contact centers with symmetric contact-direction lines (± normal)")

    set_equal_aspect(ax, cx, cy, cz)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
