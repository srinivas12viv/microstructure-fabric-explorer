# ContactDetection3D

Estimate geometric contacts, contact normals, coordination and fabric from a segmented 3D image. Each particle must already have a unique positive integer ID; background is zero by default. This tool does not segment images or measure forces.

## Install and run

From the toolkit root, with Python 3.10+:

```sh
python -m pip install -r requirements.txt
python ContactDetection3D/main.py
python ContactDetection3D/main.py --help
```

The default input is the supplied `scp_labeled_N5_ov1.tif`, resolved relative to this module. Default results go to the toolkit's `outputs/contact-detection/`; the supplied `out/` figures and results are preserved. Custom relative paths resolve from the current working directory. Package invocation also works: `python -m ContactDetection3D.main`.

For your own data:

```sh
python ContactDetection3D/main.py --input path/to/labels.tif --output-dir outputs/my-contacts --voxel-size-mm 0.00625 --connectivity 26 --shell-thickness 1 --min-contact-voxels 3
```

The spacing above is only an example. The default 1 mm per voxel is an **uncalibrated placeholder** inherited from the supplied entry script; confirm your image calibration before interpreting distances. Input must be a nonempty, nonnegative integer TIFF array in `(z, y, x)` order with isotropic spacing. Floating-point intensities, RGB images, time/channel axes, anisotropic voxels, and periodic boundaries are not supported. Reorient or resample such data before use, preserving label IDs with nearest-neighbor interpolation.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | supplied TIFF | Segmented 3D label stack |
| `--output-dir` | `outputs/contact-detection` | Directory to create; same-named generated files are replaced |
| `--voxel-size-mm` | `1` | Positive isotropic spacing; requires user calibration |
| `--shell-thickness` | `1` | Positive number of binary dilation iterations |
| `--connectivity` | `26` | 6 face neighbors, 18 face/edge neighbors, or 26 face/edge/corner neighbors |
| `--min-contact-voxels` | `3` | Minimum one-sided patch size, at least 3; collinear patches are also discarded |
| `--exclude-labels` | `0` | Space-separated labels to ignore; retain 0 in this list to exclude background |
| `--no-plots` | off | Skip overview PNGs |
| `--show` | off | Also display figures interactively; requires a GUI backend |
| `--debug-pair I J` | none | Save slice and voxel-cube plots for a retained pair |
| `--debug-contact-index N` | none | Alternative zero-based index into sorted retained pairs |
| `--viz-pad` | `10` | Local voxel-view radius; slice views show full sections |

Debug selectors are mutually exclusive. Figures are saved without opening windows by default. Debug figures are still saved with `--no-plots`, which only disables overviews. Debug indices refer to the current run's retained pairs, so filtering can change them. A missing or filtered pair produces an error.

```sh
python ContactDetection3D/main.py --debug-contact-index 0 --viz-pad 10
```

## Method and interpretation

1. Find each particle's bounding box and pad it to cover the dilation shell.
2. Dilate its mask using the selected connectivity; subtract the original mask. Neighbor labels occupying this shell define candidate contacts. Store each unordered pair once with the lower ID first.
3. Preserve the original **one-sided** patch convention: patch voxels belong to the higher-ID neighbor within the lower-ID particle's shell. Disconnected patches between a pair are pooled. Counts and centroids can therefore change when particle IDs are reassigned; these are not symmetric interface areas.
4. Remove patches below the voxel threshold and patches with numerically rank less than two. PCA of physical patch coordinates gives two tangents and a normal from the smallest covariance eigenvalue. Three points are a minimum, not a guarantee of a stable normal. Nearly equal second/third eigenvalues also indicate an uncertain normal; inspect the eigenvalues and planarity.
5. Build the retained contact graph. Report coordination both over contacting particles and over all included particles, assigning zero to isolated particles.
6. Compute `F = sum(w * outer(n, n)) / sum(w)`, once with unit weights and once with patch voxel counts. Components use the ZYX basis. Empty contact sets produce zero tensors; nonempty tensors have trace approximately one. Eigenvalues descend; eigenvectors are columns.

A dilation contact is a proximity criterion on a voxel grid, not proof of mechanical contact. Changing connectivity, shell thickness, segmentation, or resolution changes results. Shell thickness is a grid-neighborhood distance, not a spherical physical cutoff. Patch voxel count is neither contact area nor force. The PCA normal is axial: its sign and tangent signs are arbitrary. Planarity is `1 - eig3/(eig2 + 1e-15)`; it is a descriptive score, not a confidence interval. Normals are not oriented from one particle toward another.

The current bounding-box remapping uses full-volume integer arrays; large stacks can need substantially more memory than their on-disk TIFF size. Per-pair voxel clouds are retained in memory during analysis. Crop appropriately or use a machine with enough RAM; tiling is not implemented.

## Output files

| File | Contents |
| --- | --- |
| `contacts_summary.csv` | One row per retained pair: `i,j,nvox`, named centroid/normal/tangent components, planarity and descending covariance eigenvalues |
| `contact_results.npz` | `particle_ids`, `voxel_size_mm`, `pairs (N,2)`, `centers_mm`, `normals`, `tangents1`, `tangents2`, `eigvals` (all `(N,3)`); `voxel_counts`, `planarity` (`(N,)`); two fabric tensors and their eigensystems |
| `normals.txt` | Toolkit-compatible whitespace table with one header: `nx ny nz f`; normals reordered to XYZ, `f=1` for every row |
| `coordination.csv` | `particle_id,coordination`, including isolated particles |
| `summary.json` | Counts, discarded candidate-pair count, both mean coordinations, settings, calibration and tensors |
| `centroids-normals.png` | Contact centroids with symmetric normal segments, axes in mm |
| `coordination.png` | All-particle coordination distribution |
| `contact-slices.png`, `contact-voxels.png` | Optional debug figures |

NPZ geometric vectors and tensors use **ZYX**; CSV columns explicitly name components. Coordinates are voxel indices multiplied by spacing, with index zero at the origin (no half-voxel shift). Covariance eigenvalues have units mm squared. Debug voxel plots show local ROI coordinates in mm. Empty runs still write header-only CSV/text tables and consistently shaped empty NPZ arrays. Existing histogram tools require nonempty orientation data.

## Use with the toolkit

```sh
python PolarHistogram2D/PolarHistogram.py --input outputs/contact-detection/normals.txt --plane xy --output outputs/image-polar.html
python SphericalHistogram3D/3DHistogram.py --input outputs/contact-detection/normals.txt --output outputs/image-sphere.html
```

Both histogram tools add antipodes. Their force-color field is constant one for this export, a unit orientation weight rather than a measured force. The force-network tool additionally needs particle positions and physical contact forces; this workflow does not generate those inputs.

For Python callers, from the repository root:

```python
from ContactDetection3D.config import Config
from ContactDetection3D.main import run

summary = run(Config(tiff_path="my_labels.tif", out_dir="outputs/my-contacts",
                     voxel_size_mm=0.00625, plots=False))
print(summary["n_contacts"])
```

## Supplied figures and legacy results

`out/typicalContact.svg`, `out/centroids_normals.svg` and `out/histogram.svg` are the original figures displayed in `index.html`. `out/Validation_contact_algorithm.pdf` and `.pptx` are supplied supporting slides, not an independent validation performed by this toolkit. Their exact generating settings and calibration were not recorded. Keep historical results separate when comparing new runs.

The old entry script used a two-voxel threshold and could retain collinear PCA patches. This integration defaults to three voxels and filters rank-deficient patches, so regenerated results may differ. The original histogram's ordinate says ?Number of contacts,? although a coordination distribution counts particles; the new plot uses ?Number of particles.? Legacy scripts inside `out/` are preserved reference material, not toolkit entry points; `Contacts_3D.py` additionally needs pandas. Use `main.py` for supported plotting without pandas.

Run `python build_gallery.py` from the root to regenerate the toolkit outputs and download bundles. It includes the contact-detection example and sample TIFF; it preserves the original SVGs. Run `python -m unittest discover -s tests -v` for synthetic contact, geometry, empty-input and toolkit-export checks. Redistribution terms are covered by the root `LICENSE-NOTE.md`.
