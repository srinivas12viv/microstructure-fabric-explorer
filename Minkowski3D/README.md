# Minkowski3D

Compute particle-void fabric from spherical particle positions and radii:

**Particle geometry -> weighted Voronoi cells -> Minkowski moments -> assembly fabric.**

Voronoi tessellation uses [pyvoro2/Voro++](https://delonecommons.github.io/pyvoro2/).
Analytic moment integration and fabric statistics are implemented in `main.py`,
based on Srinivas Vivek Bokkisa's master's thesis,
[*Fabric evolution in 3D discrete element modelling of soil liquefaction*](https://doi.org/10.14711/thesis-991012752454703412)
(2019), section 3.3, and inspired by
[Karambola](https://github.com/morphometry/karambola).
No contacts, force data, or DEM software are required.

## Files

```text
Minkowski3D/
  main.py                 # Tessellation, moments, fabric, exports, and plotting
  run_example.py          # Run the supplied sample
  __init__.py             # Python API exports
  requirements.txt
  README.md
  .gitignore
  example/
    particles.txt         # 10,000 particles: id x y z radius
    domain.json           # Sheared periodic domain and units
```

## Run the example

From this folder, with Python 3.10+:

```sh
python -m pip install -r requirements.txt
python run_example.py
```

Open `outputs/preview.html` in a browser with WebGL support. The interactive
preview shows the packing, one cell and its principal axis, the full axis
distribution, and the local elongation histogram. Large geometry views are
sampled; statistics use every particle. Results are generated locally and
excluded from Git by `.gitignore`.

The example contains a thesis packing snapshot with radii and coordinates in
**mm**. Radii were joined by particle ID before packaging. Its loading stage is
unspecified. The weighted tessellation was checked against all 10,000 original
cell volumes: maximum discrepancy was about 5e-6 mm^3, within printed precision.

The example defaults to **cell-centroid moments**, matching the original
postprocessing reference. Expected results are approximately:

```text
Cells: 10000
Ed: 0.28956483
Ad: 0.32984239
```

For moments about each particle center instead:

```sh
python run_example.py --reference particle
```

## Your own data

Supply a whitespace-separated particle file with optional exact header:

```text
id x y z radius
42 0.5 0.5 0.5 0.2
81 1.5 0.5 0.5 0.3
```

Use unique integer IDs, finite positions, positive radii, and consistent length
units. The domain configuration is separate because positions alone do not
establish specimen boundaries or periodic shear. For a fully periodic domain,
write its lattice vectors as rows:

```json
{
  "lattice_vectors": [[2, 0, 0], [-0.2, 2, 0], [0, 0, 2]],
  "units": "mm",
  "mode": "power"
}
```

For an orthorhombic domain, replace `lattice_vectors` with `bounds` containing
three `[minimum, maximum]` pairs and an explicit `periodic` array such as
`[true, true, false]`. Mode `power` uses squared radii as weights; `standard`
selects ordinary Voronoi. Choose a sensible coordinate scale for Voro++
numerical tolerances.

```sh
python main.py --input example/particles.txt --domain example/domain.json --reference centroid --output-dir outputs
```

The general CLI defaults to `--reference particle`. Four-column files require
an explicit `--uniform-radius VALUE`. Relative custom paths resolve from the
working directory. `run_example.py` locates its bundled sample automatically.

Python API from this folder:

```python
import json
from main import load_particles, analyze, save_results

ids, positions, radii = load_particles("example/particles.txt")
with open("example/domain.json") as stream:
    domain = json.load(stream)
result = analyze(ids, positions, radii, domain, reference="centroid")
save_results(result, "outputs")
```

## Computation and interpretation

Each closed convex cell is triangulated, its winding is oriented outward,
and its moments are integrated analytically using tetrahedra. Periodic cells
remain continuous around their generating sites. Invalid cells and failed
volume closure cause an error rather than silently dropping particles.

```text
W020(o) = integral_cell (x-o) outer (x-o) dV
W_normalized = W020 / radius^5
beta = lambda_min / lambda_max
ed = 1 - beta
Ed = mean(ed)
v = unit eigenvector of lambda_max
V = mean(v outer v)
D = (15/2) (V - I/3)
Ad = sqrt((3/2) sum_ab D_ab^2)
```

`Ed` describes average local elongation; `V` and `Ad` describe collective axis
alignment. Locally elongated cells can have randomly oriented axes. This Ad
convention has a perfectly aligned limit of **7.5**, not 1.

Physical moments have length^5 units and normalized moments are dimensionless.
Particle-centered and cell-centroid moments differ by a parallel-axis term.
The normalized cell describes the full particle environment: the particle's
volume is not subtracted. It is not an explicit pore-only geometry. The method
assumes spherical particles; general irregular-particle normalization is outside
its scope. One snapshot gives static fabric, not deformation history.

When the largest eigenvalues are nearly equal, the principal axis is ambiguous.
Both the literal all-cell orientation estimator and an explicitly filtered
estimator are exported, with ambiguity counts. Equal moment eigenvalues do
not uniquely imply spherical geometry; a cube also has isotropic moments.

## Outputs

| File | Contents |
| --- | --- |
| `cells.csv` | IDs, sites, radii, volumes, centroids, beta, ed, eigenvalues, axes, and ambiguity flags |
| `tensors.npz` | Physical and normalized tensors, input geometry, derived arrays, and metadata; load with `allow_pickle=False` |
| `fabric.json` | Ed, V, D, Ad, mean beta, units, reference, numerical settings, and volume closure |
| `principal_axes.txt` | Axis vectors with unit weights, compatible with orientation histograms; weights do not represent forces |
| `preview.html` | Self-contained interactive visualization |

The implementation has been checked against analytic box and tetrahedron
moments, reference shifts, rotation/scaling, reversed face winding, fabric
limits, weighted bisectors, and periodic translations.
