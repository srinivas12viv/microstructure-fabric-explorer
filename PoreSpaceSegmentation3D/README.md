# PoreSpaceSegmentation3D

MATLAB code for dividing binary 3D pore space into pore bodies and pore throats,
then exporting their sizes and principal orientations. One implementation runs
two synthetic examples: **VEP** (ellipsoidal void lattice) and **SCP** (simple
cubic packing).

## Run

Make this folder the MATLAB current folder, or add it to the MATLAB path:

```matlab
run_example("VEP")
run_example("SCP")
% Optional parallel processing:
run_example("VEP", UseParallel=true)
```

Requires MATLAB R2022b or later, Image Processing Toolbox, Statistics and Machine
Learning Toolbox, and Signal Processing Toolbox. Parallel Computing Toolbox is
optional. Large volumes may require several GB of memory. MATLAB syntax checks
passed; the consolidated workflow has not been numerically rerun because a
working MATLAB license was unavailable during integration.

## Your own input

```matlab
result = poreSpaceSegmentation3D("input.mat", "my-results", ...
    VoxelSize=0.00625, MeanParticleDiameter=0.25, D50=0.25, Units="mm");
```

Replace the example calibration with your own values. The input MAT file must
contain only the required binary array `vox3d`: **1 = pore, 0 = solid**.
This release supports cubic volumes with isotropic voxels. Dimensions follow
MATLAB **X,Y,Z** order. Coordinates are voxel centers. Mask external air before
inverting a solid mask; grayscale thresholding is outside this module.

`VoxelSize` is length per voxel. `MeanParticleDiameter` controls seed selection;
`D50` separately normalizes sizes. The bundled examples use spacing 1, mean
particle diameter 40 and an illustrative D50=40 in voxel units.

Results are created locally under `outputs/`: a labeled MAT volume, body/throat
CSV tables, JSON settings and diagnostics, and separate orientation text files
compatible with the toolkit histograms. Equivalent-sphere diameter is
`(6*volume/pi)^(1/3)`, not geometric throat length. Axes come from voxel-center
second moments; near-degenerate axes are flagged in CSV exports. Text orientation
weights are one and do not represent forces.

## Shared contents

- `poreSpaceSegmentation3D.m`, `run_example.m`: public entry points.
- `private/`: shared algorithm functions.
- `examples/VEP/`, `examples/SCP/`: one binary input each and selected PDF figures.
- `docs/MODIFICATIONS.md`: concise overview of the method changes.
- Citation and upstream license files.

The figures illustrate the supplied examples; they are not newly generated
validation results. Sphericity figures are included as illustrations, but the
shared runner does not recalculate sphericity. Saved research results, debug
folders, editable slide decks, and internal audit records are not included.

## Attribution

Original framework and code: [Nimisha Roy's Pore Space Architecture](https://github.com/Nimisha-Roy/Pore_Space_Architecture),
published by Roy, Frost and Roozbahani (2023), *Quantifying three-dimensional
bodies and throats of particulate system pore space*, Powder Technology, 415,
118160. The original MIT notice is preserved in `LICENSE.upstream`.

Refinement modifications: **Srinivas Vivek Bokkisa**, *Multiscale Assessment of
fabric in granular soils*, submitted thesis, unpublished, Chapter 8 and Appendix
8B. Institution and submission year are unspecified. See `REFERENCES.bib` and
`THIRD_PARTY_NOTICES.md`. The upstream code license does not establish the
redistribution terms for the example data or thesis figures.
