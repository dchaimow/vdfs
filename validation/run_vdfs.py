"""Usage: run_vdfs.py <geometry> <equidist|current|wagstyl> <outdir> [aligned]
Calculates depths for one geometry with one vdfs variant ("aligned": voxel grid without offset)."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import geometry

warnings.simplefilter("ignore")
name, variant, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if variant == "wagstyl":
    import vdfs_wagstyl as vdfs  # vdfs 1.1.0 with surface_tools' formula (converted to depths from white)
else:
    import vdfs
print(name, variant, "using", vdfs.__file__)

geo = geometry.make(name, aligned="aligned" in sys.argv[4:])
to_grid = lambda c: (c - geo["origin"]) / geometry.VS
surf_white = dict(nVertices=len(geo["white"]), nTris=len(geo["faces"]), tris=geo["faces"],
                  vertices=to_grid(geo["white"]))
surf_pial = dict(surf_white, vertices=to_grid(geo["pial"]))
# vertex areas from the surfaces in mm
area_white = vdfs.calc_vertex_area(geo["white"], geo["faces"])
area_pial = vdfs.calc_vertex_area(geo["pial"], geo["faces"])
method = "equidist" if variant == "equidist" else "equivol"
t = time.time()
depths, columns = vdfs.calc_depth_from_surfaces_on_grid(
    surf_white, area_white, surf_pial, area_pial, *geo["shape"], method, n_jobs=10)
print(f"  {np.sum(~np.isnan(depths))} voxels with depth, {time.time() - t:.0f} s")
np.savez_compressed(outdir / f"{name}_{variant}.npz", depths=depths, columns=columns)
