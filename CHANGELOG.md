# Changelog

## 2.0.0

All equi-volume depths change compared with 1.x (corrected formula and new default area model); equi-distance depths are unchanged (except for points exactly on prism boundaries, which now get a depth). To reproduce results of earlier versions, install them (e.g. `pip install "git+https://github.com/dchaimow/vdfs@v1.1.0"`).

Changes of results:
- **Equi-volume depths: corrected formula.** `alpha_from_beta` now uses the linear-area model of Waehnert et al. (2014, Eq. 10), as in surface_tools by K. Wagstyl. The formula used since the first version, alpha = (2 ap beta - ap beta^2 + aw beta^2) / (ap + aw), was the inverse of the surface_tools function (distance fraction to volume fraction, measured from the pial surface). It agrees with the correct relation only to first order in the white/pial area difference and gave too many voxels at mid-depth and too few near the surfaces; in synthetic tests (`validation/`) its depth errors were about twice as large (rms up to 0.027, max up to 0.050 at strongly curved cortex).
- **Equi-volume depths: quadratic area model (new default).** The cross-sectional area of a column is modelled as quadratic in the distance fraction, using the vertex areas of the white, midthickness and pial surfaces. This is exact for the prism geometry vdfs uses and also accounts for curvature in two directions, which the linear model leaves out (see README, Equi-volume model). The linear model remains available with `area_model="linear"`.

Performance:
- The intersection of each voxel center with the intermediate surfaces is now calculated in compiled code (numba), in parallel over voxels (numba threads instead of joblib processes): about 10-50 times faster.
  - Equi-distance: the condition that the intermediate plane passes through the point is a cubic polynomial in the depth, whose roots in [0, 1] are calculated exactly (`equidist_cubic`, `cubic_roots_in_unit_interval`) instead of numerically with scipy.
  - Equi-volume: the depth lies between the vertices' volume fractions at the equi-distance depth (closed form, `beta_from_alpha_quadratic`) and is found by a bracketed root search (`equivol_root`); near a prism's side faces a general search is used (`equivol_search`).
  - Results are the same as before to within 1e-9 (synthetic validation and a real FreeSurfer subject). If prisms overlap, the first candidate prism containing the point is still used.
- joblib is no longer a dependency. `find_depth_through_point` (scipy) is kept but no longer used.

Updating from 1.x:
- `voxeldepths_from_surfaces` and `process_voxeldepth_from_surfaces` work as before; equi-volume depths now use the quadratic area model (`area_model="linear"` for the corrected linear model).
- `calc_depth_from_surfaces_on_grid` and `calc_depth_from_surfaces_voxelidx` keep their arguments; without the new `area_mid` argument they use the linear area model. `n_jobs` now sets the number of numba threads.
- `equivol_plane` takes three additional arguments (middle coefficients of the quadratic area model).

New:
- `area_model` argument (`"quadratic"`, default, or `"linear"`) for `voxeldepths_from_surfaces` and `process_voxeldepth_from_surfaces`; `area_mid` argument for `calc_depth_from_surfaces_on_grid` and `calc_depth_from_surfaces_voxelidx` (without it, the linear model is used).
- `alpha_from_beta_quadratic()`, `calc_mid_vertex_area()`.

Fixes:
- Points lying exactly on a boundary between prisms (e.g. voxel centres on a mesh symmetry plane) could get no depth, because rounding placed them marginally outside every adjacent intermediate triangle. Points within a tiny tolerance of a triangle edge (`EDGE_TOLERANCE`, 1e-6 of the edge length) now count as inside (#10).

## 1.1.0

New:
- `voxeldepths_from_surfaces()`: new main entry function without area file arguments (vertex areas are calculated from the surfaces, identical to the freesurfer `?h.area`/`?h.area.pial` files up to float32 precision).
- Gifti surfaces (`.gii`) as input (support existed in fmri_analysis' copy of this code since 2022, but was missing in vdfs).
- `process_voxeldepth_from_surfaces()` (original interface): area files can also be gifti metric files, or `None` to calculate them.
- Labels can also be gifti files (vertices with nonzero values).
- `calc_vertex_area()`, `load_vertex_area()`, `read_surf_geometry()`, `read_label_vertices()` helper functions.
- `process_dc_voxeldepth_from_surfaces` alias (name used in fmri_analysis).
- Installable with pip (`pyproject.toml`), tests (`tests/`).

Fixes:
- The first triangle (index 0, i.e. the first left hemisphere triangle, or the first right hemisphere triangle if only the right hemisphere is used) was never used, because 0 also marked empty entries in the list of candidate prisms of each voxel. Voxels inside its prism got no depth. Empty entries are now -1 (changes the output of `numba_calc_bounding_prisms`).
- Possible `IndexError` when the candidate prism list of a voxel was completely filled and the voxel center was not inside any of the candidate prisms.
- With labels, the columns output contained indices into the label-restricted list of triangles. It now contains the original triangle indices (as without labels).
- Crash when no voxel was close to any prism.
- Division by zero for vertices with zero area on both surfaces (#3).
- Clear errors for non-corresponding white and pial surfaces, mismatching number of vertex areas, unknown `method`, and freesurfer surfaces without volume info (c_ras).

Changes of results compared to 1.0.0 (for the same inputs, using area files):
- Voxels in the prism of the first triangle (fix above).
- Columns, if labels are used (fix above).
- The equivolume formula is unchanged.

Updating a project from 1.0.0:
- Existing calls of `process_voxeldepth_from_surfaces()` with area files keep working.
- 1.0.0 cannot be installed with pip (no `pyproject.toml`), only used as a git submodule. To switch a project from a `vdfs` submodule to a pip-installed vdfs, remove the submodule (`git rm vdfs`) and install vdfs (>= 1.1.0) in the project's environment. Scripts that `import vdfs` work unchanged.

## 1.0.0

State of the code before consolidation (FreeSurfer surfaces only, area files required), tagged for reproducibility of earlier results. Not pip-installable, only usable as a git submodule.
