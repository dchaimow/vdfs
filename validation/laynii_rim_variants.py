"""Usage: laynii_rim_variants.py <outdir> <LN2_LAYERS binary>
How LayNii's results depend on how the rim is built from the surfaces, on closed concentric shells.
LN2_LAYERS measures depths between the centers of its border voxels (rim 2 = inner, 1 = outer border), and in
LayNii's convention the border voxels are the outermost grey matter voxels. Rim variants:
  outside:  borders = voxels just outside the grey matter (centers outside the ribbon) touching it
  inside:   borders = outermost grey matter voxels (LayNii convention)
  centered: borders = voxels whose center lies within half a voxel of the surface (on either side)
Errors are evaluated on all voxels that LN2_LAYERS assigns a depth to (rim = 3), against the exact depths."""
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

import geometry

outdir, ln2 = Path(sys.argv[1]), sys.argv[2]
outdir.mkdir(parents=True, exist_ok=True)
VS = geometry.VS
nbh = ndimage.generate_binary_structure(3, 1)
cases = {
    "sphere_crown": ("sphere", 4.0, 7.0, (7.5, 7.5, 7.5)),
    "sphere_fundus": ("sphere", 6.0, 3.0, (6.5, 6.5, 6.5)),
    "cylinder_crown": ("cylinder", 4.0, 7.0, (7.5, 8.0, 7.5)),
}
runs = [(v, o) for v in ("outside", "inside", "centered") for o in ([], ["-iter_smooth", "1000"])]

for name, (kind, r_w, r_p, half) in cases.items():
    half = np.array(half)
    lo = -half + geometry.GRID_OFFSET
    shape = tuple(int(v) for v in np.floor(2 * half / VS) + 1)
    xyz = lo + np.indices(shape).transpose(1, 2, 3, 0) * VS
    r = np.linalg.norm(xyz, axis=-1) if kind == "sphere" else np.hypot(xyz[..., 0], xyz[..., 2])
    n = 3 if kind == "sphere" else 2
    ribbon = (r - r_w) * (r - r_p) < 0
    true_equidist = (r - r_w) / (r_p - r_w)
    true_equivol = (r ** n - r_w ** n) / (r_p ** n - r_w ** n)
    away_from_ends = (np.abs(xyz[..., 1]) <= 4.0) if kind == "cylinder" else np.ones(shape, bool)
    affine = np.diag([VS] * 3 + [1.0])
    affine[:3, 3] = lo
    print(f"\n=== {name}")
    print(f"    {'rim':10s} {'options':18s} {'voxels':>7s} | equidist vs exact equidist: mean, max |err| | "
          f"equivol vs exact equivol: mean, rms, max |err|")
    for variant, opts in runs:
        rim = np.zeros(shape, dtype=np.int16)
        if variant == "outside":
            border = ndimage.binary_dilation(ribbon, structure=nbh) & ~ribbon
            wm_side = (r - r_w) * (r_p - r_w) < 0
            rim[ribbon] = 3
            rim[border & wm_side], rim[border & ~wm_side] = 2, 1
        elif variant == "inside":
            border = ribbon & ~ndimage.binary_erosion(ribbon, structure=nbh)
            nearer_white = np.abs(r - r_w) < np.abs(r - r_p)
            rim[ribbon] = 3
            rim[border & nearer_white], rim[border & ~nearer_white] = 2, 1
        else:  # centered
            near_w, near_p = np.abs(r - r_w) <= VS / 2, np.abs(r - r_p) <= VS / 2
            rim[ribbon] = 3
            rim[near_w], rim[near_p] = 2, 1
        tag = f"{name}_{variant}" + ("_smooth1000" if opts else "")
        nib.save(nib.Nifti1Image(rim, affine), outdir / f"{tag}_rim.nii")
        subprocess.run([ln2, "-rim", str(outdir / f"{tag}_rim.nii"), "-nr_layers", "10", "-equivol", *opts,
                        "-output", str(outdir / f"{tag}_ln2")], check=True, stdout=subprocess.DEVNULL)
        sel = (rim == 3) & away_from_ends
        res = []
        for k, truth in (("equidist", true_equidist), ("equivol", true_equivol)):
            d = nib.load(outdir / f"{tag}_ln2_metric_{k}.nii").get_fdata()
            if np.corrcoef(d[sel], truth[sel])[0, 1] < 0:
                d = 1 - d
            e = d[sel] - truth[sel]
            res.append((e.mean(), np.sqrt((e ** 2).mean()), np.abs(e).max()))
        (m1, _, x1), (m2, s2, x2) = res
        print(f"    {variant:10s} {' '.join(opts) or 'defaults':18s} {sel.sum():7d} | {m1:+.4f} {x1:.4f}"
              f"                          | {m2:+.4f} {s2:.4f} {x2:.4f}")
