# Method and modifications

The five-stage framework is adapted from **Nimisha Roy's Pore Space
Architecture**, published by Roy, Frost and Roozbahani (2023). The core sequence
is pore candidates, connectivity, throat centers, initial body/throat assignment,
and geometric refinement.

**Srinivas Vivek Bokkisa's modifications** are described in *Multiscale Assessment
of fabric in granular soils*, submitted thesis, unpublished, Chapter 8 and
Appendix 8B. They primarily revise the refinement stage:

- Axis-constrained PCA plane construction and local 2D coordinate transforms.
- Adaptive concave boundaries with geometric fallbacks.
- Arc-length resampling and periodic Cartesian Savitzky-Golay smoothing.
- Inflection/knee candidates and a geometric bulge criterion for selecting
  body/throat transitions.
- Back-projection and two-pass voxel relabeling, with optional parallel processing.

Steps 1-4 retain the original conceptual framework with implementation changes
for search and voxel processing. These descriptions identify the changes;
independent accuracy or speed improvements are not asserted.

The toolkit provides one shared implementation for both examples and portable
outputs. Body-to-throat reassignment is allowed, matching the supplied serial
policy. Overlapping proposals are merged in ascending throat-row order, with
later proposals taking precedence. This is deterministic for fixed row ordering,
not invariant to arbitrary row permutations. PCA-normal projection is not
claimed to be an exact constrained least-squares plane minimizer.

MATLAB syntax checks passed. Numerical execution and serial/parallel agreement
remain unverified because a working MATLAB license was unavailable during
integration. Example figures are supplied illustrations, not regenerated results.

Original software and MIT license:
https://github.com/Nimisha-Roy/Pore_Space_Architecture.
See `../REFERENCES.bib` and `../LICENSE.upstream`.
