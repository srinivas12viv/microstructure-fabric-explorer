# Microstructure & Fabric Explorer

Four Python tools for exploring contact networks, force transmission, and directional fabric from simulation or experimental data. The supplied examples come from discrete element method (DEM) simulations; experimental inputs must follow the same file schemas and supply the quantities required by each tool. ContactDetection3D also estimates geometric contacts from an already segmented, integer-labeled 3D TIFF. Image segmentation and force inference remain outside its scope. Open **index.html** to explore the gallery, locally and offline. Keep `outputs/` beside it: the interactive exports share a local Plotly bundle. 3D interaction requires WebGL. No web server or account is needed.

Suggested GitHub repository name: `microstructure-fabric-explorer`.

## Setup and reproduce

Python 3.10+ is recommended. From this folder:

```sh
python -m pip install -r requirements.txt
python build_gallery.py
python -m unittest discover -s tests -v
```

The builder runs contact detection on the supplied TIFF and regenerates figures, CSV tables, metadata, and ZIP downloads. This requires more time and memory than the histogram examples; contact results go to `outputs/contact-detection/`. Original scripts are saved under `originals/`; the pre-existing images and `SphericalHistogram3D/figure.html` are retained. New results go to `outputs/`.

## Run individual tools

```sh
python ContactDetection3D/main.py
python ForceChain2D/ForceChain.py
python PolarHistogram2D/PolarHistogram.py --input PolarHistogram2D/normals_Aniso.txt --plane xz --bins 36 --output outputs/custom-polar.html
python SphericalHistogram3D/3DHistogram.py --input SphericalHistogram3D/normals_Iso.txt --longitude-bins 36 --latitude-bins 18 --output outputs/custom-sphere.html
```

Default paths resolve relative to the code, so scripts also work from a different working directory. Supplied custom relative paths resolve from your working directory. Each script supports `--help`. Example ZIPs contain their tool, shared module, data, and setup notes; the full toolkit also contains the gallery and generated outputs.

## Labeled-image contact detection

See [ContactDetection3D/README.md](ContactDetection3D/README.md) for the complete CLI, Python API, output schemas, axis conventions, and limitations. Run `python ContactDetection3D/main.py --help` for options. The gallery features three original SVG figures and links to the supplied validation slides; those historical figures are preserved separately from regenerated outputs.

```sh
python ContactDetection3D/main.py --input ContactDetection3D/scp_labeled_N5_ov1.tif --output-dir outputs/my-contacts --voxel-size-mm 1
python PolarHistogram2D/PolarHistogram.py --input outputs/my-contacts/normals.txt --output outputs/image-polar.html
python SphericalHistogram3D/3DHistogram.py --input outputs/my-contacts/normals.txt --output outputs/image-sphere.html
```

`--voxel-size-mm 1` is a placeholder, not a verified sample calibration. Replace it with your isotropic image spacing. Exported orientation weights are all one: these plots summarize contact counts, not measured forces. ForceChain2D additionally requires positions and forces that this image workflow does not supply.

## Input formats

UTF-8/ASCII whitespace-delimited files with exactly one header row, followed by numeric rows:

| File | Columns | Meaning |
| --- | --- | --- |
| Force-chain positions | `id x y z` | Unique integer particle ID and coordinates; only x/y are rendered |
| Force-chain contacts | `id1 id2 nx ny nz f` | Particle pair, normal, and nonnegative force magnitude |
| Orientation contacts | `nx ny nz f` | Nonzero normal and nonnegative force magnitude |

All numeric input must be finite. Normals are normalized on loading. Coordinates and forces retain the input units; no physical units were supplied. The Iso and Aniso names are inherited dataset labels, not statistical guarantees.

## Methods and interpretation

**Force chains:** line color and width encode normal force. A batched line collection replaces per-contact plotting. The original component-distance filter is preserved: a contact is excluded when either x or y separation is at least 0.1 input units. This is a visual heuristic, not a periodic-boundary reconstruction. Change it with `--max-gap`, or use `--max-gap 0` to show all contacts. Missing particle references produce an error. A minimum visible width also displays zero-force contacts.

**Polar:** project each normal into XY, XZ or YZ; exclude projections with norm at most 1e-12. Include each direction and its antipode, then histogram into 36 ten-degree bins by default. Radius is count; color is mean force per bin (empty bins use zero). The fit uses least squares on density with basis `1, cos(2θ), sin(2θ)`, then converts to count scale with `N × bin width`. The reported amplitude is `2π sqrt(cos_coefficient² + sin_coefficient²)` and preferred axis is `0.5 atan2(sin_coefficient, cos_coefficient)` modulo 180°. A near-zero amplitude has no meaningful preferred axis. This is a descriptive second harmonic, not an uncertainty estimate; strongly structured distributions can yield a negative fitted density. A projected distribution is not the full 3D fabric.

**Spherical:** normalize directions and add antipodes. Use `atan2(y,x)` for azimuth and bins uniform in azimuth and cos(colatitude), giving 648 equal-solid-angle bins at the default resolution. NumPy histogram endpoint handling includes the poles. Wedge radius and color encode bin count; this is not a particle geometry. One triangulated mesh replaces thousands of traces; empty bins are omitted without division by zero. CSVs include empty bins. A fixed-angle radius comparison across datasets also depends on sample count. The static 3D PNG is a surface through bin centers; the interactive HTML shows the actual wedges.

## Improvements

All rows are retained; arbitrary whitespace is supported; empty bins and axial vectors are safe; preferred orientation uses the full atan2 quadrant; hard-coded fit scaling is removed; outputs are explicit and portable; original data are preserved. Conservation and edge-case tests cover the numerical methods.

## Sharing

Host this entire folder as a static website, or use the full toolkit ZIP. Read `LICENSE-NOTE.md` before public redistribution. No original license or attribution information was included in the supplied folder.
