# particle_stats.py
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple


def get_particle_ids(labels: np.ndarray, exclude=(0,)) -> np.ndarray:
    ids = np.unique(labels)
    if exclude:
        exclude = set(exclude)
        ids = np.array([i for i in ids if int(i) not in exclude], dtype=labels.dtype)
    return ids


def compute_bounding_boxes(labels: np.ndarray, particle_ids: np.ndarray) -> Dict[int, Tuple[slice, slice, slice]]:
    """
    Returns bounding boxes as slices in (z,y,x):
      bbox[pid] = (slice(z0,z1), slice(y0,y1), slice(x0,x1))

    Uses a remapping trick so find_objects is efficient even for non-consecutive labels.
    """
    if particle_ids.size == 0:
        return {}

    # Create a sorted array of unique labels including excluded ones so searchsorted works
    all_ids_sorted = np.unique(labels)
    # Map each voxel label to an index 0..(K-1)
    idx_labels = np.searchsorted(all_ids_sorted, labels)  # same shape as labels

    # find_objects expects labels 1..max_label, so we will offset by +1
    idx_labels_1 = idx_labels + 1  # now in 1..K
    objects = ndimage.find_objects(idx_labels_1, max_label=int(idx_labels_1.max()))

    # Build map from actual pid -> index in all_ids_sorted -> object slot
    id_to_obj = {}
    id_to_pos = {int(v): int(k) for k, v in enumerate(all_ids_sorted)}  # label -> idx_labels value

    for pid in particle_ids:
        pid_int = int(pid)
        k = id_to_pos[pid_int]
        obj = objects[k]  # because idx_labels_1 = k+1 corresponds to objects[k]
        if obj is None:
            continue
        id_to_obj[pid_int] = obj

    return id_to_obj


def pad_bbox(bbox: Tuple[slice, slice, slice], shape_zyx: Tuple[int, int, int], pad: int) -> Tuple[slice, slice, slice]:
    """
    Expand bbox by pad voxels in each direction, clipped to volume bounds.
    """
    zsl, ysl, xsl = bbox
    Z, Y, X = shape_zyx

    z0 = max(0, (zsl.start or 0) - pad)
    z1 = min(Z, (zsl.stop or Z) + pad)
    y0 = max(0, (ysl.start or 0) - pad)
    y1 = min(Y, (ysl.stop or Y) + pad)
    x0 = max(0, (xsl.start or 0) - pad)
    x1 = min(X, (xsl.stop or X) + pad)

    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))
