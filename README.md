# vdfs
Voxel depth from surfaces.

Python code that uses white matter and pial surfaces to calculate voxel-wise relative cortical depths for arbitrary voxel grids. Both equivolume and equidistant approaches are supported.

## Usage:

```python
import voxeldepth_from_surfaces as vdfs

vdfs.process_voxeldepth_from_surfaces(surf_white_lh_file,area_white_lh_file,
                                      surf_pial_lh_file,area_pial_lh_file,
                                      surf_white_rh_file,area_white_rh_file,
                                      surf_pial_rh_file,area_pial_rh_file,
                                      volume_file,
                                      depths_fname,columns_fname,
                                      label_lh_fname=None,label_rh_fname=None,
                                      method='equivol',
                                      upsample_factor=None,n_jobs=32,force=False)
```

`surf_white_lh_file`: filename of left white matter surface (freesurfer of gifti)

`area_white_lh_file`: filename of left white matter vertex areas (freesurfer)

`surf_pial_lh_file`: filename of left pial surface (freesurfer of gifti)

`area_pial_lh_file`: filename of left pial vertex areas (freesurfer)

`surf_white_rh_file`: filename of right white matter surface (freesurfer of gifti)

`area_white_rh_file`: filename of right white matter vertex areas (freesurfer)

`surf_pial_rh_file`: filename of right pial surface (freesurfer of gifti)

`area_pial_rh_file`: filename of right pial vertex areas (freesurfer)

`volume_file`: filename of nifti that determines grid and affine of output

`depths_fname`: filename of output depths file

`columns_fname`: filename of output columns file (face number that each voxel's center belongs to)

`label_lh_fname`: optional freesurfer left label file, depth calculation will be restricted to label vertices

`label_rh_fname`: optional freesurfer right label file, depth calculation will be restricted to label vertices

`method` : 'equivol' or 'equidist'

`upsample_factor` : upsampling factor (if desired), default=None

`n_jobs`: number of cpus to run in parallel

`force`: if True, generate output files even if they already exist

## Acknowledgements

The formula for calculating the euclidean distance fraction, that will yield the desired volume fraction, given vertex areas in the white matter and pial surfaces was taken from https://github.com/kwagstyl/surface_tools/blob/master/equivolumetric_surfaces/generate_equivolumetric_surfaces.py (which is a nice tool to calculate intermediate layer meshes).
