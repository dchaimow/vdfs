# Changelog

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
