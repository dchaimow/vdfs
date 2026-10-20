"""Usage: make_variants.py <output dir>
Writes copies of the installed vdfs module in which only alpha_from_beta is replaced:
  vdfs_wagstyl.py:   surface_tools (Wagstyl) formula, area changing linearly with distance, converted to depths
                     measured from the white surface
  vdfs_quadratic.py: area changing quadratically with distance, A(alpha) = a_w (1-alpha)^2 + 2 a_m alpha (1-alpha)
                     + a_p alpha^2, which is exact for triangles whose vertices move linearly between white and pial
                     (if white and pial triangles are parallel). a_m comes from the area of the midthickness surface
                     (white + pial) / 2: a_m = 2 a_mid - (a_w + a_p) / 2. To leave all other functions unchanged, the
                     white area array passed to vdfs has two columns, (a_w, a_m), and alpha_from_beta unpacks them."""
import sys
from pathlib import Path

import vdfs  # noqa: F401  (imports the module, which the package name shadows with the function of the same name)

src_file = sys.modules["vdfs.voxeldepths_from_surfaces"].__file__
src = Path(src_file).read_text()
old = """    if ap + aw == 0:
        return beta
    alpha = (2 * ap * beta - ap * beta**2 + aw * beta**2) / (ap + aw)
    return alpha"""
variants = {
    "vdfs_wagstyl.py": """    if ap + aw == 0:
        return beta
    # surface_tools (Wagstyl) formula, area changing linearly with distance, depths measured from white
    if abs(ap - aw) <= 1e-6 * (ap + aw):
        return beta
    alpha = (-aw + np.sqrt((1 - beta) * aw**2 + beta * ap**2)) / (ap - aw)
    return alpha""",
    "vdfs_quadratic.py": """    # area changing quadratically with distance; aw holds (white area, middle coefficient a_m)
    a_w, a_m = aw[0], aw[1]
    # volume fraction below alpha is V(alpha) / V(1) with V(alpha) = c1 alpha + c2 alpha^2 + c3 alpha^3
    c1, c2, c3 = a_w, a_m - a_w, (a_w - 2 * a_m + ap) / 3
    total = c1 + c2 + c3
    if total <= 0:
        return beta
    target = beta * total
    alpha = beta
    for _ in range(50):
        v = ((c3 * alpha + c2) * alpha + c1) * alpha - target
        dv = (3 * c3 * alpha + 2 * c2) * alpha + c1
        if dv <= 0:
            break
        step = v / dv
        alpha = min(1.0, max(0.0, alpha - step))
        if abs(step) < 1e-12:
            break
    return alpha""",
}
if src.count(old) != 1:
    sys.exit(f"alpha_from_beta in {src_file} is not the expected (current) formula")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for fname, new in variants.items():
    (out / fname).write_text(src.replace(old, new))
    print("written", out / fname, "from", src_file)
