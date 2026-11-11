# contact_detection.py
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Set
from .particle_stats import pad_bbox


def _connectivity_structure(connectivity: int) -> np.ndarray:
    """
    Returns a 3x3x3 binary structuring element for dilation.
    """
    if connectivity == 6:
        st = np.zeros((3, 3, 3), dtype=bool)
        st[1, 1, 1] = True
        st[0, 1, 1] = st[2, 1, 1] = True
        st[1, 0, 1] = st[1, 2, 1] = True
        st[1, 1, 0] = st[1, 1, 2] = True
        return st
    elif connectivity == 18:
        st = np.ones((3, 3, 3), dtype=bool)
        # remove 8 corners
        corners = [(0,0,0),(0,0,2),(0,2,0),(0,2,2),(2,0,0),(2,0,2),(2,2,0),(2,2,2)]
        for c in corners:
            st[c] = False
        return st
    elif connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    else:
        raise ValueError("connectivity must be one of {6,18,26}.")


def compute_shell(mask_i: np.ndarray, shell_thickness: int, connectivity: int) -> np.ndarray:
    """
    mask_i is boolean ROI for particle i.
    Returns the dilated mask minus the original particle mask.
    """
    if not isinstance(shell_thickness, int) or shell_thickness < 1:
        raise ValueError("shell_thickness must be a positive integer")
    st = _connectivity_structure(connectivity)
    dil = ndimage.binary_dilation(mask_i, structure=st, iterations=int(shell_thickness))
    shell = np.logical_and(dil, np.logical_not(mask_i))
    return shell


def neighbors_from_shell(labels_roi: np.ndarray, shell: np.ndarray, pid: int, exclude_labels=(0,)) -> np.ndarray:
    vals = np.unique(labels_roi[shell])
    exclude = set(int(x) for x in exclude_labels) | {int(pid)}
    nbrs = np.array([int(v) for v in vals if int(v) not in exclude], dtype=int)
    return nbrs


def contact_voxels_for_neighbor(labels_roi: np.ndarray, shell: np.ndarray, neighbor_id: int) -> np.ndarray:
    """
    Return (N,3) coords in ROI coordinates (z,y,x) where shell is occupied by neighbor_id.
    """
    m = np.logical_and(shell, labels_roi == int(neighbor_id))
    coords = np.argwhere(m)  # (z,y,x)
    return coords


def find_contacts_for_particle(
    labels: np.ndarray,
    pid: int,
    bbox_pid: Tuple[slice, slice, slice],
    shell_thickness: int,
    connectivity: int,
    exclude_labels=(0,),
    bbox_pad_extra: int = 2,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Tuple[slice, slice, slice]]:
    """
    For a given particle pid, in a padded ROI:
      - compute shell
      - find neighbor IDs
      - return dict neighbor->coords_global (N,3 in global zyx)
    """
    # Pad bbox so dilation/shell doesn't clip at ROI edge
    pad = int(shell_thickness) + int(bbox_pad_extra)
    roi = pad_bbox(bbox_pid, labels.shape, pad=pad)

    labels_roi = labels[roi]
    mask_i = (labels_roi == int(pid))

    shell = compute_shell(mask_i, shell_thickness=shell_thickness, connectivity=connectivity)
    nbrs = neighbors_from_shell(labels_roi, shell, pid=pid, exclude_labels=exclude_labels)

    # Convert ROI coords -> global coords
    z0, y0, x0 = roi[0].start, roi[1].start, roi[2].start

    contacts = {}
    for nid in nbrs:
        coords_local = contact_voxels_for_neighbor(labels_roi, shell, nid)
        if coords_local.size == 0:
            continue
        coords_global = coords_local + np.array([z0, y0, x0], dtype=int)
        contacts[int(nid)] = coords_global

    return contacts, nbrs, roi
