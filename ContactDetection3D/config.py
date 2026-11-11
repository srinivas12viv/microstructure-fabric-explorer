# config.py
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import math


@dataclass(frozen=True)
class Config:
    # I/O
    tiff_path: str = str(Path(__file__).resolve().parent / "scp_labeled_N5_ov1.tif")
    out_dir: str = str(Path(__file__).resolve().parents[1] / "outputs" / "contact-detection")

    # Geometry / resolution
    voxel_size_mm: float = 1.0  # placeholder; supply calibrated isotropic spacing

    # Contact detection
    shell_thickness_vox: int = 1        # configurable dilation iterations
    connectivity: int = 26              # 6, 18, or 26

    # Filtering: a plane normal needs at least 3 noncollinear points
    min_contact_voxels: int = 3

    # Optional particle selection / debugging
    exclude_labels: Tuple[int, ...] = (0,)  # background label
    debug_pair: Optional[Tuple[int, int]] = None  # e.g., (i, j) to visualize
    debug_contact_idx: Optional[int] = None  # 0-based retained-contact index

    # Visualization controls
    viz_pad_vox: int = 10  # padding for ROI in visualization
    viz_stride: int = 1    # downsample contact points for plotting if needed

    show: bool = False
    plots: bool = True

    def __post_init__(self):
        if not math.isfinite(self.voxel_size_mm) or self.voxel_size_mm <= 0:
            raise ValueError("voxel_size_mm must be finite and positive")
        for name, minimum in [("shell_thickness_vox", 1), ("min_contact_voxels", 3),
                              ("viz_pad_vox", 0), ("viz_stride", 1)]:
            value = getattr(self, name)
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be 6, 18 or 26")
