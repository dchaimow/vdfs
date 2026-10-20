"""Usage: run_vdfs.py <geometry> <equidist|current|wagstyl|quadratic> <outdir> [aligned]
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
elif variant == "quadratic":
    import vdfs_quadratic as vdfs  # vdfs 1.1.0 with the quadratic area model (see make_variants.py)
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
if variant == "quadratic":
    # middle coefficient of the quadratic area model from the area of the midthickness surface
    area_mid = vdfs.calc_vertex_area((geo["white"] + geo["pial"]) / 2, geo["faces"])
    area_white = np.column_stack((area_white, 2 * area_mid - (area_white + area_pial) / 2))
method = "equidist" if variant == "equidist" else "equivol"
t = time.time()
depths, columns = vdfs.calc_depth_from_surfaces_on_grid(
    surf_white, area_white, surf_pial, area_pial, *geo["shape"], method, n_jobs=10)
print(f"  {np.sum(~np.isnan(depths))} voxels with depth, {time.time() - t:.0f} s")
np.savez_compressed(outdir / f"{name}_{variant}.npz", depths=depths, columns=columns)
