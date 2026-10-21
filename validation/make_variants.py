"""Usage: make_variants.py <output dir>
Writes vdfs_v1formula.py, a copy of the installed vdfs module in which the linear-area formula in alpha_from_beta
is replaced by the formula used in vdfs <= 1.1, (2 ap beta - ap beta^2 + aw beta^2) / (ap + aw), for comparison.
(The linear and quadratic area models are tested with vdfs itself, see run_vdfs.py.)"""
import sys
from pathlib import Path

import vdfs  # noqa: F401  (imports the module, which the package name shadows with the function of the same name)

src_file = sys.modules["vdfs.voxeldepths_from_surfaces"].__file__
src = Path(src_file).read_text()
old = """    denominator = aw + np.sqrt((1 - beta) * aw**2 + beta * ap**2)
    if denominator == 0:  # aw = 0 and beta = 0
        return 0.0
    return beta * (aw + ap) / denominator"""
new = """    # formula of vdfs <= 1.1 (inverse of the surface_tools relation)
    return (2 * ap * beta - ap * beta**2 + aw * beta**2) / (ap + aw)"""
if src.count(old) != 1:
    sys.exit(f"alpha_from_beta in {src_file} is not the expected (linear-area) formula")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "vdfs_v1formula.py").write_text(src.replace(old, new))
print("written", out / "vdfs_v1formula.py", "from", src_file)
