"""Usage: run_laynii.py <geometry> <vdfs|large> <outdir> <LN2_LAYERS binary> [<vdfs results dir>]
Runs LN2_LAYERS -equivol (with -iter_smooth 100, the default, and 1000) on a rim built from the surfaces.

Rim (LayNii convention, see LN2_LAYERS.cpp: border labels mark grey matter voxels):
  3 = grey matter voxel (center inside the ribbon)
  2 = grey matter voxel with a face neighbour outside the grey matter, on the white side (depth < 0.5)
  1 = the same on the pial side (depth >= 0.5)
The edge of the volume does not count as a border. -incl_borders gives every grey matter voxel a depth.
  vdfs:  the vdfs voxel box; grey matter and sides from the vdfs equi-distance depths (consistent with the mesh)
  large: geometry.large_box (ribbon not cut near the patches); grey matter and sides from the exact depths"""
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

import geometry

name, mode, outdir, ln2 = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
outdir.mkdir(parents=True, exist_ok=True)
if mode == "vdfs":
    geo = geometry.make(name)
    origin, shape = geo["origin"], geo["shape"]
    depth = np.load(Path(sys.argv[5]) / f"{name}_equidist.npz")["depths"]
else:
    origin, shape = geometry.large_box(name)
    depth = geometry.exact_depths(name, geometry.voxel_centers(origin, shape))[0]
    depth[(depth <= 0) | (depth >= 1)] = np.nan
gm = ~np.isnan(depth)
border = gm & ~ndimage.binary_erosion(gm, structure=ndimage.generate_binary_structure(3, 1), border_value=1)
rim = np.zeros(shape, dtype=np.int16)
rim[gm] = 3
rim[border & (depth < 0.5)] = 2
rim[border & (depth >= 0.5)] = 1
affine = np.diag([geometry.VS] * 3 + [1.0])
affine[:3, 3] = origin
rim_file = outdir / f"{name}_rim.nii"
nib.save(nib.Nifti1Image(rim, affine), rim_file)
print(f"{name} ({mode} box {shape}): rim voxels GM {np.sum(rim == 3)}, white border {np.sum(rim == 2)}, "
      f"pial border {np.sum(rim == 1)}")
for it in (100, 1000):
    subprocess.run([ln2, "-rim", str(rim_file), "-nr_layers", "10", "-equivol", "-incl_borders",
                    "-iter_smooth", str(it), "-output", str(outdir / f"{name}_ln2_s{it}")],
                   check=True, stdout=subprocess.DEVNULL)
