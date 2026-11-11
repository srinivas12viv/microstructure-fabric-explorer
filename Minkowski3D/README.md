# Minkowski_w020

`Minkowski_w020` is a Python tool to compute the Minkowski tensor $W_0^{2,0}$ for large sets of 3D convex cells, such as Voronoi cells in granular packings. The tensor $W_0^{2,0}$ characterizes the volume mass distribution and anisotropy of a convex body and is defined as

$$
W_0^{2,0} = \int_V \mathbf{r} \otimes \mathbf{r} \,\mathrm{d}V,
$$

where $\mathbf{r}$ is the position vector and $V$ is the volume of the cell.

Minkowski tensors form a family of six independent tensorial shape measures commonly denoted
$W_0^{2,0}, W_1^{2,0}, W_3^{2,0}, W_1^{0,2}, W_2^{0,2}, \ldots$, each capturing different aspects of the shape of a convex body (Schröder-Turk et al., 2010).

---

## Motivation

[Karambola](https://morphometry.org/software/karambola/) is an open-source C++ code that can compute a full set of Minkowski tensors for 3D bodies and surfaces. However:

- It requires generating a separate `.poly` file for each convex body.
- It is most convenient for a small number of cells.
- Computations become expensive when scaling to $\mathcal{O}(10^4)$ Voronoi cells.

`Minkowski_w020` focuses specifically on $W_0^{2,0}$ and is designed to handle **large ensembles of cells** efficiently.

---

## Key Features

- **Python implementation**  
  Written in Python for readability and easy modification or extension by users.

- **Parallel computation**  
  Computation of $W_0^{2,0}$ for different cells is mutually independent, so the code uses parallelization to significantly reduce runtime, even for $\sim 10{,}000+$ Voronoi cells.

- **Direct use of Voro++ outputs**  
  No need to generate `.poly` files. The code expects:
  - **Vertices** and
  - **Faces**  
  as input, which are standard outputs from [Voro++](http://math.lbl.gov/voro++/) (using custom print options such as `%t` for faces and `%P` for vertices).

- **Flexible reference point**  
  Includes a `reference_sphere_center` option so that Minkowski tensors can be computed with respect to a specified sphere center, useful for particle–void analyses.

---

## Relation to Karambola and Voro++

- **Karambola**  
  `Minkowski_w020` is inspired by the algorithms used in the C++ Karambola code, which can compute all Minkowski tensors for 3D bodies. Here, the focus is narrowed to $W_0^{2,0}$ to allow scaling to large numbers of cells with a simpler workflow.

- **Voro++**  
  Voro++ (Chris Rycroft) is a widely used library to compute Voronoi tessellations around spherical particles. The faces and vertices of each Voronoi cell can be exported using custom formats in Voro++, and these files serve as the direct input to `Minkowski_w020`.

---

## Reference

Schröder-Turk, G. E., Kapfer, S., Breidenbach, B., Beisbart, C., and Mecke, K. (2010).  
“Tensorial Minkowski functionals and anisotropy measures for planar patterns.” *Journal of Microscopy*, 238(1), 57–74.
