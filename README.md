# vdfs
Voxel depth from surfaces.

Python code that uses white matter and pial surfaces to calculate voxel-wise relative cortical depths for arbitrary voxel grids. Both equivolume and equidistant approaches are supported. Surfaces can be freesurfer or gifti files.

## Installation

Install into the Python environment (or container) in which it will be used, preferably pinned to a release tag:

```bash
pip install "git+https://github.com/dchaimow/vdfs@v2.0.0"
```

Dependencies (numpy, scipy, nibabel, numba) are installed automatically. For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/dchaimow/vdfs
pip install -e "vdfs[test]"
python -m pytest vdfs/tests
```

Alternatively, the repository can be added as a git submodule named `vdfs` (`git submodule add https://github.com/dchaimow/vdfs`), which can then be imported directly from the parent directory.

Projects that use vdfs through [fmri_analysis](https://github.com/dchaimow/fmri_analysis) (`from fmri_analysis import voxeldepths_from_surfaces`) need vdfs to be installed as above.

## Usage

```python
import vdfs

nii_depths, nii_columns = vdfs.voxeldepths_from_surfaces(
    surf_white_lh_file, surf_pial_lh_file,
    surf_white_rh_file, surf_pial_rh_file,
    volume_file, depths_fname, columns_fname,
    label_lh_fname=None, label_rh_fname=None,
    method="equivol", area_model="quadratic", upsample_factor=None, n_jobs=32, force=False)
```

`surf_white_lh_file`, `surf_pial_lh_file`, `surf_white_rh_file`, `surf_pial_rh_file`: white matter and pial surfaces of the left and right hemisphere, either freesurfer (e.g. `lh.white`, `lh.pial`) or gifti (`.gii`) files. Freesurfer surface coordinates are transformed to scanner space using the c_ras stored in the surface file, gifti surface coordinates are assumed to be in scanner space already. White and pial surfaces of a hemisphere need to have corresponding vertices and triangles.

`volume_file`: filename of nifti that determines grid and affine of output

`depths_fname`: filename of output depths file (0 = white matter surface, 1 = pial surface, NaN outside of the cortical ribbon)

`columns_fname`: filename of output columns file (index of the surface triangle, i.e. the prism between white matter and pial triangle, that each voxel's center belongs to; if both hemispheres are used, right hemisphere indices are offset by the number of left hemisphere triangles)

`label_lh_fname`, `label_rh_fname`: optional freesurfer label files or gifti files (vertices with nonzero values), depth calculation will be restricted to triangles with all vertices inside the label (if provided a label file is also required for the other hemisphere, otherwise other hemisphere will be ignored altogether)

`method` : `"equivol"` or `"equidist"`

`area_model` : for equi-volume depths, how the cross-sectional area of a cortical column changes between white and pial surface: `"quadratic"` (default, exact for the geometry vdfs uses) or `"linear"` (Waehnert et al., 2014), see [Equi-volume model](#equi-volume-model)

`upsample_factor` : upsampling factor (if desired), default=None

`n_jobs`: number of threads to run in parallel

`force`: if True, generate output files even if they already exist (otherwise existing output files are loaded and returned)

### Vertex areas

The equivolume method needs the area of each vertex on the white matter and pial surface, and for the quadratic area model also on the midthickness surface (vertices halfway between corresponding white and pial vertices). `voxeldepths_from_surfaces` calculates them from the surfaces (one third of the summed areas of all triangles containing the vertex), which gives the same values as the freesurfer `?h.area` and `?h.area.pial` files (up to float32 precision). Freesurfer's `?h.area.mid` is not the midthickness area, but the mean of `?h.area` and `?h.area.pial`.

To use area files for the white and pial surface instead, use the original interface, in which an area file follows each surface file (the midthickness areas are still calculated from the surfaces):

```python
vdfs.process_voxeldepth_from_surfaces(
    surf_white_lh_file, area_white_lh_file, surf_pial_lh_file, area_pial_lh_file,
    surf_white_rh_file, area_white_rh_file, surf_pial_rh_file, area_pial_rh_file,
    volume_file, depths_fname, columns_fname,
    label_lh_fname=None, label_rh_fname=None,
    method="equivol", upsample_factor=None, n_jobs=32, force=False, area_model="quadratic")
```

Area files can be freesurfer (e.g. `lh.area`, `lh.area.pial`) or gifti metric files, any area file given as `None` is calculated from its surface. `process_dc_voxeldepth_from_surfaces` is an alias of this function (its name in fmri_analysis before vdfs became a separate package).

### Other point sets

The lower level functions `numba_calc_bounding_prisms` and `calc_depth_from_surfaces_voxelidx` (with argument `p`) can be used to calculate depths for arbitrary points, e.g. streamlines (see [streamline_positions_from_surfaces](https://github.com/dchaimow/streamline_positions_from_surfaces)).

## Equi-volume model

vdfs divides the cortical ribbon into prisms: each triangle of the white matter surface and the corresponding triangle of the pial surface span one prism, and intermediate surfaces are obtained by moving every vertex along the straight line from its white to its pial position, `white + alpha (pial - white)`, where `alpha` is the distance fraction. A voxel's depth is the depth of the intermediate surface that passes through its center.

- **Equi-distance**: all three vertices of a triangle use the same `alpha`, and the depth is `alpha`.
- **Equi-volume**: following Bok's principle that cortical segments preserve their volume across curvature (Bok, 1929; Waehnert et al., 2014), the depth `beta` is the volume fraction of the column between the white surface and the intermediate surface. At each vertex, `alpha` is chosen such that the volume of the column below the intermediate surface is the fraction `beta` of the column's total volume. This requires a model of how the column's cross-sectional area `A(alpha)` changes between the white (`a_w`) and pial (`a_p`) vertex areas.

**Linear area model** (`area_model="linear"`): `A(alpha) = a_w + (a_p - a_w) alpha` (Waehnert et al., 2014, Eq. 10; surface_tools). Solving the volume fraction `(2 a_w alpha + (a_p - a_w) alpha^2) / (a_w + a_p) = beta` gives `alpha = (-a_w + sqrt((1 - beta) a_w^2 + beta a_p^2)) / (a_p - a_w)`. Waehnert et al. describe the linear area as an approximation chosen for robustness and because it has a closed-form solution. It is exact when the cortex is curved in at most one direction (e.g. a cylinder).

**Quadratic area model** (`area_model="quadratic"`, default): since vdfs moves vertices along straight lines, the edges of an intermediate triangle are linear in `alpha`, so its area (the cross product of two edges) is quadratic in `alpha` (exactly so when the white and pial triangles are parallel, and very nearly otherwise):

`A(alpha) = a_w (1 - alpha)^2 + 2 a_m alpha (1 - alpha) + a_p alpha^2`

The middle coefficient follows from the area `a_mid` of the midthickness surface (the intermediate surface at `alpha = 0.5`): `a_m = 2 a_mid - (a_w + a_p) / 2`. The volume below `alpha` is then a cubic polynomial, `a_w alpha + (a_m - a_w) alpha^2 + (a_w - 2 a_m + a_p) alpha^3 / 3`, which is solved for `alpha` with Newton's method. The linear model is the special case `a_mid = (a_w + a_p) / 2`. Geometrically, the area along the normal of a surface with mean curvature `H` and Gaussian curvature `K` changes as `1 - 2 H t + K t^2`, so the quadratic term is the contribution of curvature in two directions (gyral crowns, sulcal fundi, saddles), which the linear model leaves out. On a real FreeSurfer subject, the midthickness area deviates from the linear model's prediction by -11% to +5% (5th-95th percentile of vertices).

**Validation**: the folder [`validation/`](validation/) tests both models on synthetic surfaces with exact equi-volume depths (sphere and cylinder shells, a folded sheet). The quadratic model's depth errors (rms 0.0003-0.002, max 0.004) are as small as the error from representing the surfaces by triangles; the linear model leaves errors up to rms 0.015 / max 0.029 where the cortex is curved in two directions.

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## References

- Bok, S. T. (1929). Der Einfluß der in den Furchen und Windungen auftretenden Krümmungen der Großhirnrinde auf die Rindenarchitektur. Zeitschrift für die gesamte Neurologie und Psychiatrie.
- Waehnert, M. D., Dinse, J., Weiss, M., Streicher, M. N., Waehnert, P., Geyer, S., Turner, R., & Bazin, P.-L. (2014). Anatomically motivated modeling of cortical laminae. NeuroImage, 93, 210–220.
- Wagstyl, K., et al. (2018). Mapping cortical laminar structure in the 3D BigBrain. Cerebral Cortex, 28(7), 2551–2562; and [surface_tools](https://github.com/kwagstyl/surface_tools) (equivolumetric surfaces).

## Acknowledgements

The linear area model follows Waehnert et al. (2014) as implemented in https://github.com/kwagstyl/surface_tools/blob/master/equivolumetric_surfaces/generate_equivolumetric_surfaces.py (which is a nice tool to calculate intermediate layer meshes).
