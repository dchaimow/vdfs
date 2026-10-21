"""Usage: make_variants.py <output dir>
Writes vdfs_v1formula.py, a copy of the installed vdfs module in which the linear-area formula in alpha_from_beta
is replaced by the formula used in vdfs <= 1.1, (2 ap beta - ap beta^2 + aw beta^2) / (ap + aw), for comparison
(and beta_from_alpha_quadratic by its numerical inverse).
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
# vdfs brackets equi-volume depths with the inverse relation (volume fraction for a distance fraction), which
# therefore has to be the inverse of the replaced formula: calculated numerically (bisection)
old_inverse = """    c1, c2, c3 = aw, am - aw, (aw - 2 * am + ap) / 3
    total = c1 + c2 + c3
    if total <= 0:
        return alpha
    return ((c3 * alpha + c2) * alpha + c1) * alpha / total"""
new_inverse = """    # inverse of alpha_from_beta_quadratic (with the formula of vdfs <= 1.1), by bisection
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if alpha_from_beta_quadratic(mid, aw, ap, am) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)"""
if src.count(old) != 1 or src.count(old_inverse) != 1:
    sys.exit(f"alpha_from_beta / beta_from_alpha_quadratic in {src_file} are not the expected (linear-area) versions")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "vdfs_v1formula.py").write_text(src.replace(old, new).replace(old_inverse, new_inverse))
print("written", out / "vdfs_v1formula.py", "from", src_file)
