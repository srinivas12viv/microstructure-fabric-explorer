# viz.py
import numpy as np
import matplotlib.pyplot as plt


def _clip(a, lo, hi):
    return max(lo, min(hi, a))


def plot_contact_slices(labels, pid, nid, contact_coords_zyx, center_zyx_vox, pad=10, output=None, show=True):
    """
    Quick debugging: show z-, y-, x-slices through the contact centroid.
    """
    Z, Y, X = labels.shape
    cz, cy, cx = (int(round(center_zyx_vox[0])), int(round(center_zyx_vox[1])), int(round(center_zyx_vox[2])))

    cz = _clip(cz, 0, Z - 1)
    cy = _clip(cy, 0, Y - 1)
    cx = _clip(cx, 0, X - 1)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].imshow(labels[cz, :, :], interpolation="nearest")
    axs[0].set_title(f"Z-slice z={cz}")

    axs[1].imshow(labels[:, cy, :], interpolation="nearest")
    axs[1].set_title(f"Y-slice y={cy}")

    axs[2].imshow(labels[:, :, cx], interpolation="nearest")
    axs[2].set_title(f"X-slice x={cx}")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(f"Contact debug: pid={pid} nid={nid}  (#pts={len(contact_coords_zyx)})")
    plt.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_contact_3d(
    labels,
    pid,
    nid,
    contact_coords_zyx,
    voxel_size_mm,
    center_mm,
    n,
    t1,
    stride=1,
    roi_pad_vox=12,
    particle_stride=3,
    max_particle_pts=50000,
    output=None,
    show=True,
):
    """
    3D scatter of:
      - contact voxels (highlighted)
      - voxels of the two neighboring particles (pid and nid) in a local ROI
      - PCA vectors (normal and tangent) at the contact centroid

    Notes:
      - labels expected in (z,y,x)
      - contact_coords_zyx are global voxel indices (z,y,x)
      - center_mm in (z,y,x) mm coordinates
      - n, t1 in (z,y,x) basis (unit vectors)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ---- ROI around contact (in voxel coordinates) ----
    center_vox = (np.asarray(center_mm, dtype=float) / float(voxel_size_mm))
    cz, cy, cx = np.round(center_vox).astype(int)

    Z, Y, X = labels.shape
    z0 = max(0, cz - int(roi_pad_vox)); z1 = min(Z, cz + int(roi_pad_vox) + 1)
    y0 = max(0, cy - int(roi_pad_vox)); y1 = min(Y, cy + int(roi_pad_vox) + 1)
    x0 = max(0, cx - int(roi_pad_vox)); x1 = min(X, cx + int(roi_pad_vox) + 1)

    roi = labels[z0:z1, y0:y1, x0:x1]

    # ---- Extract particle voxels in ROI ----
    pid = int(pid); nid = int(nid)

    coords_pid_local = np.argwhere(roi == pid)  # (k,3) in local ROI coords
    coords_nid_local = np.argwhere(roi == nid)

    # Convert ROI-local -> global voxel coords
    if coords_pid_local.size:
        coords_pid = coords_pid_local + np.array([z0, y0, x0], dtype=int)
    else:
        coords_pid = np.zeros((0, 3), dtype=int)

    if coords_nid_local.size:
        coords_nid = coords_nid_local + np.array([z0, y0, x0], dtype=int)
    else:
        coords_nid = np.zeros((0, 3), dtype=int)

    # ---- Downsample particle points to keep plotting responsive ----
    def downsample_coords(coords, stride_, max_pts):
        if coords.shape[0] == 0:
            return coords
        coords_ds = coords[::max(1, int(stride_))]
        if coords_ds.shape[0] > max_pts:
            idx = np.random.choice(coords_ds.shape[0], size=max_pts, replace=False)
            coords_ds = coords_ds[idx]
        return coords_ds

    coords_pid = downsample_coords(coords_pid, particle_stride, max_particle_pts)
    coords_nid = downsample_coords(coords_nid, particle_stride, max_particle_pts)

    # ---- Contact points (already global coords); subsample by stride ----
    cc = contact_coords_zyx[::max(1, int(stride))]
    pts_contact = cc.astype(float) * float(voxel_size_mm)

    # Convert particle voxel coords -> mm
    pts_pid = coords_pid.astype(float) * float(voxel_size_mm)
    pts_nid = coords_nid.astype(float) * float(voxel_size_mm)

    # ---- Plot ----
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if pts_pid.shape[0]:
        ax.scatter(pts_pid[:, 2], pts_pid[:, 1], pts_pid[:, 0], s=1, alpha=0.50)

    if pts_nid.shape[0]:
        ax.scatter(pts_nid[:, 2], pts_nid[:, 1], pts_nid[:, 0], s=1, alpha=0.50)

    ax.scatter(pts_contact[:, 2], pts_contact[:, 1], pts_contact[:, 0], s=6, alpha=0.9)

    c = np.asarray(center_mm, dtype=float)  # (z,y,x) mm

    def v_to_plot(v):
        v = np.asarray(v, dtype=float)
        return np.array([v[2], v[1], v[0]], dtype=float)

    c_plot = np.array([c[2], c[1], c[0]], dtype=float)
    scale = float(voxel_size_mm) * 30.0

    ax.quiver(c_plot[0], c_plot[1], c_plot[2], *v_to_plot(n) * scale, length=1.0, normalize=False)
    ax.quiver(c_plot[0], c_plot[1], c_plot[2], *v_to_plot(t1) * scale, length=1.0, normalize=False)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(f"Contact pid={pid} nid={nid}: particles (faint) + contact voxels + PCA vectors")

    plt.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


# --- voxel cube visualization -------------------------------------------------

def _surface_6n(mask: np.ndarray) -> np.ndarray:
    """
    Surface voxels using 6-neighborhood:
    A voxel is surface if it's inside the particle AND at least one of its 6 neighbors is outside.
    No SciPy required.
    """
    mask = mask.astype(bool)
    m = np.pad(mask, 1, mode="constant", constant_values=False)

    c  = m[1:-1, 1:-1, 1:-1]
    xp = m[1:-1, 1:-1, 2:]
    xm = m[1:-1, 1:-1, :-2]
    yp = m[1:-1, 2:, 1:-1]
    ym = m[1:-1, :-2, 1:-1]
    zp = m[2:, 1:-1, 1:-1]
    zm = m[:-2, 1:-1, 1:-1]

    interior = c & xp & xm & yp & ym & zp & zm
    return c & (~interior)


def plot_contact_voxel_cubes(
    labels,
    pid,
    nid,
    contact_coords_zyx,
    voxel_size_mm,
    center_mm,
    roi_pad_vox=12,
    alpha_particles=0.5,
    alpha_contact=0.9,
    show_contact_only_in_roi=True,
    output=None,
    show=True,
):
    """
    Voxel-cube visualization (Matplotlib ax.voxels) in a local ROI:

      - Contact voxels: RED cubes
      - Surface voxels of pid: BLUE cubes (alpha_particles)
      - Surface voxels of nid: GREEN cubes (alpha_particles)

    labels: (Z,Y,X) labeled image (z,y,x)
    contact_coords_zyx: global voxel indices (N,3) (z,y,x)
    center_mm: (z,y,x) centroid in mm
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pid = int(pid)
    nid = int(nid)

    Z, Y, X = labels.shape

    # ---- ROI around contact center (voxel coords) ----
    center_vox = (np.asarray(center_mm, dtype=float) / float(voxel_size_mm))
    cz, cy, cx = np.round(center_vox).astype(int)

    z0 = max(0, cz - int(roi_pad_vox)); z1 = min(Z, cz + int(roi_pad_vox) + 1)
    y0 = max(0, cy - int(roi_pad_vox)); y1 = min(Y, cy + int(roi_pad_vox) + 1)
    x0 = max(0, cx - int(roi_pad_vox)); x1 = min(X, cx + int(roi_pad_vox) + 1)

    roi = labels[z0:z1, y0:y1, x0:x1]

    # ---- Particle surface masks in ROI ----
    m_pid = (roi == pid)
    m_nid = (roi == nid)
    s_pid = _surface_6n(m_pid)
    s_nid = _surface_6n(m_nid)

    # ---- Contact mask (ROI-local) ----
    contact_roi = np.zeros_like(roi, dtype=bool)
    cc = np.asarray(contact_coords_zyx, dtype=int)

    if cc.size:
        if show_contact_only_in_roi:
            in_roi = (
                (cc[:, 0] >= z0) & (cc[:, 0] < z1) &
                (cc[:, 1] >= y0) & (cc[:, 1] < y1) &
                (cc[:, 2] >= x0) & (cc[:, 2] < x1)
            )
            cc = cc[in_roi]

        if cc.size:
            rr = cc - np.array([z0, y0, x0], dtype=int)
            contact_roi[rr[:, 0], rr[:, 1], rr[:, 2]] = True

    # ---- Plot voxel cubes ----
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Particle surfaces first (so contact appears on top)
    if np.any(s_pid):
        ax.voxels(s_pid.transpose(2, 1, 0), facecolors=(0.0, 0.0, 1.0, float(alpha_particles)), edgecolor=None)  # blue
    if np.any(s_nid):
        ax.voxels(s_nid.transpose(2, 1, 0), facecolors=(0.0, 1.0, 0.0, float(alpha_particles)), edgecolor=None)  # green
    if np.any(contact_roi):
        ax.voxels(contact_roi.transpose(2, 1, 0), facecolors=(1.0, 0.0, 0.0, float(alpha_contact)), edgecolor=None)  # red

    ax.set_box_aspect(roi.shape[::-1])
    ax.tick_params(labelsize=8)
    # ---- Local ROI axis labels in mm (ticks relabeled) ----
    dz = dy = dx = float(voxel_size_mm)
    nz, ny, nx = roi.shape

    ax.set_xlabel("x (mm)", labelpad=12)
    ax.set_ylabel("y (mm)", labelpad=12)
    ax.set_zlabel("z (mm)", labelpad=12)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)

    def _set_mm_ticks(set_ticks, set_ticklabels, n, dmm):
        ticks = np.linspace(0, n, 5)
        set_ticks(ticks)
        set_ticklabels([f"{t*dmm:.3f}" for t in ticks])

    _set_mm_ticks(ax.set_xticks, ax.set_xticklabels, nx, dx)
    _set_mm_ticks(ax.set_yticks, ax.set_yticklabels, ny, dy)
    _set_mm_ticks(ax.set_zticks, ax.set_zticklabels, nz, dz)

    ax.set_title(
        f"Contact {pid} / {nid}: local voxel geometry\nBlue / green: particle surfaces; red: contact patch"
    )

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.10, top=0.85)
    if output is not None:
        fig.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def save_overview(centers, normals, coordination, out_dir, show=False):
    """Save reproducible centroid/normal and all-particle coordination previews."""
    from pathlib import Path
    out_dir = Path(out_dir)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    if len(centers):
        xyz = centers[:, ::-1]
        directions = normals[:, ::-1]
        length = max(float(np.ptp(xyz, axis=0).max()) * 0.04, 1e-6)
        ax.scatter(*xyz.T, s=8, color="#087e83")
        for sign in (-1, 1):
            ax.quiver(*xyz.T, *(sign * directions).T, length=length, color="#d56e42", arrow_length_ratio=0)
        ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), length))
    else:
        ax.text2D(0.2, 0.5, "No retained contacts", transform=ax.transAxes)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title="Contact centroids and axial PCA normals")
    fig.tight_layout()
    fig.savefig(out_dir / "centroids-normals.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    values = np.asarray(coordination, dtype=int)
    ax.hist(values, bins=np.arange(-0.5, (values.max() if values.size else 0) + 1.5), color="#087e83", edgecolor="white")
    ax.set(xlabel="Coordination number", ylabel="Number of particles", title="Coordination including isolated particles")
    fig.tight_layout()
    fig.savefig(out_dir / "coordination.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
