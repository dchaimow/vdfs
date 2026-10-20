"""Usage: make_wagstyl_variant.py <output dir>
Writes <output dir>/vdfs_wagstyl.py: a copy of the installed vdfs module in which only alpha_from_beta is replaced
by the surface_tools (Wagstyl) formula, converted to depths measured from the white surface."""
import sys
from pathlib import Path

import vdfs  # noqa: F401  (imports the module, which the package name shadows with the function of the same name)

src_file = sys.modules["vdfs.voxeldepths_from_surfaces"].__file__
src = Path(src_file).read_text()
old = """    if ap + aw == 0:
        return beta
    alpha = (2 * ap * beta - ap * beta**2 + aw * beta**2) / (ap + aw)
    return alpha"""
new = """    if ap + aw == 0:
        return beta
    # surface_tools (Wagstyl) formula, area changing linearly with distance, depths measured from white
    if abs(ap - aw) <= 1e-6 * (ap + aw):
        return beta
    alpha = (-aw + np.sqrt((1 - beta) * aw**2 + beta * ap**2)) / (ap - aw)
    return alpha"""
if src.count(old) != 1:
    sys.exit(f"alpha_from_beta in {src_file} is not the expected (current) formula")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "vdfs_wagstyl.py").write_text(src.replace(old, new))
print("written", out / "vdfs_wagstyl.py", "from", src_file)
