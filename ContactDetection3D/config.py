# config.py
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence


@dataclass(frozen=True)
class Config:
    # I/O
    tiff_path: str = "labels.tif"
    out_dir: str = "out"

    # Geometry / resolution
    voxel_size_mm: float = 1.0 / 160.0  # isotropic: 1 mm per 160 voxels

    # Contact detection
    shell_thickness_vox: int = 1        # default as requested; configurable
    connectivity: int = 26              # 6, 18, or 26

    # Filtering (paper omits 1-voxel contacts; PCA needs >=2 points)
    min_contact_voxels: int = 2

    # Optional particle selection / debugging
    exclude_labels: Tuple[int, ...] = (0,)  # background label
    debug_pair: Optional[Tuple[int, int]] = None  # e.g., (i, j) to visualize
    debug_contact_idx: Optional[int] = None  # <-- ADD THIS (0-based index into retained contacts)
    debug_particle: Optional[int] = None          # visualize all contacts for i
    debug_max_contacts: int = 5                   # limit in debug plots

    # Visualization controls
    viz_pad_vox: int = 10  # padding for ROI in visualization
    viz_stride: int = 1    # downsample contact points for plotting if needed
