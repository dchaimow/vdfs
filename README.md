# vdfs
Voxel depth from surfaces.

Python code that uses white matter and pial surfaces to calculate voxel-wise relative cortical depths for arbitrary voxel grids. Both equivolume and equidistant approaches are supported. Surfaces can be freesurfer or gifti files.

## Installation

Install into the Python environment (or container) in which it will be used, preferably pinned to a release tag:

```bash
pip install "git+https://github.com/dchaimow/vdfs@v1.1.0"
```

Dependencies (numpy, scipy, nibabel, numba, joblib) are installed automatically. For development, clone the repository and install it in editable mode:

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
    method="equivol", upsample_factor=None, n_jobs=32, force=False)
```

`surf_white_lh_file`, `surf_pial_lh_file`, `surf_white_rh_file`, `surf_pial_rh_file`: white matter and pial surfaces of the left and right hemisphere, either freesurfer (e.g. `lh.white`, `lh.pial`) or gifti (`.gii`) files. Freesurfer surface coordinates are transformed to scanner space using the c_ras stored in the surface file, gifti surface coordinates are assumed to be in scanner space already. White and pial surfaces of a hemisphere need to have corresponding vertices and triangles.

`volume_file`: filename of nifti that determines grid and affine of output

`depths_fname`: filename of output depths file (0 = white matter surface, 1 = pial surface, NaN outside of the cortical ribbon)

`columns_fname`: filename of output columns file (index of the surface triangle, i.e. the prism between white matter and pial triangle, that each voxel's center belongs to; if both hemispheres are used, right hemisphere indices are offset by the number of left hemisphere triangles)

`label_lh_fname`, `label_rh_fname`: optional freesurfer label files or gifti files (vertices with nonzero values), depth calculation will be restricted to triangles with all vertices inside the label (if provided a label file is also required for the other hemisphere, otherwise other hemisphere will be ignored altogether)

`method` : `"equivol"` or `"equidist"`

`upsample_factor` : upsampling factor (if desired), default=None

`n_jobs`: number of cpus to run in parallel

`force`: if True, generate output files even if they already exist (otherwise existing output files are loaded and returned)

### Vertex areas

The equivolume method needs the area of each vertex on the white matter and pial surface. `voxeldepths_from_surfaces` calculates them from the surfaces (one third of the summed areas of all triangles containing the vertex), which gives the same values as the freesurfer `?h.area` and `?h.area.pial` files (up to float32 precision).

To use area files instead (e.g. to reproduce results of vdfs <= 1.0), use the original interface, in which an area file follows each surface file:

```python
vdfs.process_voxeldepth_from_surfaces(
    surf_white_lh_file, area_white_lh_file, surf_pial_lh_file, area_pial_lh_file,
    surf_white_rh_file, area_white_rh_file, surf_pial_rh_file, area_pial_rh_file,
    volume_file, depths_fname, columns_fname,
    label_lh_fname=None, label_rh_fname=None,
    method="equivol", upsample_factor=None, n_jobs=32, force=False)
```

Area files can be freesurfer (e.g. `lh.area`, `lh.area.pial`) or gifti metric files, any area file given as `None` is calculated from its surface. `process_dc_voxeldepth_from_surfaces` is an alias of this function (its name in fmri_analysis before vdfs became a separate package).

### Other point sets

The lower level functions `numba_calc_bounding_prisms` and `calc_depth_from_surfaces_voxelidx` (with argument `p`) can be used to calculate depths for arbitrary points, e.g. streamlines (see [streamline_positions_from_surfaces](https://github.com/dchaimow/streamline_positions_from_surfaces)).

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements

The formula for calculating the euclidean distance fraction, that will yield the desired volume fraction, given vertex areas in the white matter and pial surfaces was taken from https://github.com/kwagstyl/surface_tools/blob/master/equivolumetric_surfaces/generate_equivolumetric_surfaces.py (which is a nice tool to calculate intermediate layer meshes).
