"""Synthetic white/pial surface pairs with curvature, their exact depths, and the voxel grids used to evaluate them."""
import numpy as np
from scipy.spatial import cKDTree

VS = 0.125  # voxel size in mm (3 mm thickness = 24 voxels)
THICKNESS = 3.0
GRID_OFFSET = np.array([0.0371, 0.0529, 0.0613])
GEOMETRIES = ["sphere_crown", "sphere_fundus", "cylinder_crown", "eggcrate"]
SHELLS = {"sphere_crown": ("sphere", 4.0, 7.0), "sphere_fundus": ("sphere", 6.0, 3.0),
          "cylinder_crown": ("cylinder", 4.0, 7.0)}  # kind, white radius, pial radius
EGG_AMP, EGG_WAVELENGTH = 2.0, 20.0


def icosphere(level):
    """Unit icosphere (vertices, faces) by repeated subdivision of an icosahedron."""
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    verts = [np.array(x, float) / np.linalg.norm(x) for x in v]
    faces = f
    for _ in range(level):
        cache, new_faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.array(verts), np.array(faces, dtype=np.int64)


def grid_mesh(nu, nv, periodic_u=False):
    """Triangulation of an nu x nv vertex grid (index = iu * nv + iv)."""
    idx = np.arange(nu * nv).reshape(nu, nv)
    if periodic_u:
        idx = np.vstack((idx, idx[:1]))
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
    return np.concatenate((np.column_stack((a, b, d)), np.column_stack((a, d, c)))).astype(np.int64)


def eggcrate_height(x, y):
    k = 2 * np.pi / EGG_WAVELENGTH
    return EGG_AMP * np.cos(k * x) * np.cos(k * y)


def eggcrate_derivatives(x, y):
    """Height and its first and second derivatives (f, fx, fy, fxx, fyy, fxy) of the white surface."""
    k, a = 2 * np.pi / EGG_WAVELENGTH, EGG_AMP
    cx, sx, cy, sy = np.cos(k * x), np.sin(k * x), np.cos(k * y), np.sin(k * y)
    f = a * cx * cy
    return f, -a * k * sx * cy, -a * k * cx * sy, -k * k * f, -k * k * f, a * k * k * sx * sy


def eggcrate_normal(x, y):
    _, fx, fy, *_ = eggcrate_derivatives(x, y)
    n = np.stack((-fx, -fy, np.ones_like(fx)), axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def make(name, aligned=False):
    """Returns dict with white/pial coords (mm), faces, grid origin and shape, and geometry info (vdfs voxel box).
    aligned=True leaves out the grid offset, so that voxel centers lie on mesh symmetry planes."""
    if name in ("sphere_crown", "sphere_fundus"):
        u, faces = icosphere(4)
        r_w, r_p = SHELLS[name][1:]
        white, pial = r_w * u, r_p * u
        lo, hi = (np.array([-3.5, -3.5, 3.0]), np.array([3.5, 3.5, 7.3])) if name == "sphere_crown" \
            else (np.array([-3.0, -3.0, 2.4]), np.array([3.0, 3.0, 6.3]))
    elif name == "cylinder_crown":
        n_phi = int(round(2 * np.pi * 4.0 / 0.3))
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        y = np.arange(-4.0, 4.0001, 0.3)
        pp, yy = np.meshgrid(phi, y, indexing="ij")
        u = np.column_stack((np.sin(pp.ravel()), yy.ravel(), np.cos(pp.ravel())))
        faces = grid_mesh(n_phi, len(y), periodic_u=True)
        r_w, r_p = SHELLS[name][1:]
        white = u * [r_w, 1, r_w]
        pial = u * [r_p, 1, r_p]
        lo, hi = np.array([-3.5, -2.0, 3.0]), np.array([3.5, 2.0, 7.3])
    elif name == "eggcrate":
        s = np.arange(-6.0, 16.0001, 0.3)
        xx, yy = np.meshgrid(s, s, indexing="ij")
        x, y = xx.ravel(), yy.ravel()
        white = np.column_stack((x, y, eggcrate_height(x, y)))
        pial = white + THICKNESS * eggcrate_normal(x, y)
        faces = grid_mesh(len(s), len(s))
        lo, hi = np.array([-3.0, -3.0, -2.3]), np.array([13.0, 8.0, 5.3])
    else:
        raise ValueError(name)
    if not aligned:
        # irregular offset, so that no voxel center lies exactly on a boundary between prisms (symmetry planes)
        lo = lo + GRID_OFFSET
    shape = tuple(int(v) for v in np.floor((hi - lo) / VS) + 1)
    return dict(white=white, pial=pial, faces=faces, origin=lo, shape=shape)


def large_box(name):
    """Voxel box for LayNii without the ribbon being cut near the evaluated patches: complete sphere shells,
    a long cylinder, and a folded sheet extending at least 10 mm beyond the patches. Returns origin, shape."""
    lo, hi = {"sphere_crown": ([-7.5] * 3, [7.5] * 3), "sphere_fundus": ([-6.5] * 3, [6.5] * 3),
              "cylinder_crown": ([-7.5, -8.0, -7.5], [7.5, 8.0, 7.5]),
              "eggcrate": ([-12.0, -12.0, -2.3], [22.0, 17.0, 5.3])}[name]
    lo = np.array(lo) + GRID_OFFSET
    return lo, tuple(int(v) for v in np.floor((np.array(hi) - lo) / VS) + 1)


def voxel_centers(origin, shape):
    return origin + np.indices(shape).transpose(1, 2, 3, 0) * VS


def eggcrate_foot_point(p, n_newton=8, chunk=2_000_000):
    """Nearest point on the white surface for points p (..., 3): returns foot x, y and signed distance t
    (positive on the pial side)."""
    shp = p.shape[:-1]
    p = p.reshape(-1, 3)
    xs = np.arange(p[:, 0].min() - 4, p[:, 0].max() + 4, 0.1)
    ys = np.arange(p[:, 1].min() - 4, p[:, 1].max() + 4, 0.1)
    gx, gy = [a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")]
    tree = cKDTree(np.column_stack((gx, gy, eggcrate_height(gx, gy))))
    fx_out, fy_out, t_out = (np.empty(len(p)) for _ in range(3))
    for s in range(0, len(p), chunk):
        q = p[s:s + chunk]
        nearest = tree.query(q)[1]
        x, y = gx[nearest], gy[nearest]
        for _ in range(n_newton):
            f, fx, fy, fxx, fyy, fxy = eggcrate_derivatives(x, y)
            r = f - q[:, 2]
            g1, g2 = (x - q[:, 0]) + r * fx, (y - q[:, 1]) + r * fy
            h11, h22, h12 = 1 + fx * fx + r * fxx, 1 + fy * fy + r * fyy, fx * fy + r * fxy
            det = h11 * h22 - h12 * h12
            x, y = x - (h22 * g1 - h12 * g2) / det, y - (h11 * g2 - h12 * g1) / det
        foot = np.column_stack((x, y, eggcrate_height(x, y)))
        t = np.linalg.norm(q - foot, axis=1) * np.sign(np.sum((q - foot) * eggcrate_normal(x, y), axis=1))
        fx_out[s:s + chunk], fy_out[s:s + chunk], t_out[s:s + chunk] = x, y, t
    return fx_out.reshape(shp), fy_out.reshape(shp), t_out.reshape(shp)


def exact_depths(name, xyz):
    """Exact equi-distance and equi-volume depths (0 = white, 1 = pial; values outside [0, 1] are outside the
    ribbon) and the local pial/white area ratio of the column through each point, plus the column coordinates
    used for the patches. For the folded sheet, columns are the normals of the white surface (the pial surface
    is its parallel surface at THICKNESS) and the area at distance t along a normal changes by 1 - 2 H t + K t^2
    (H, K: mean and Gaussian curvature), so the volume fraction is V(t)/V(T) with V(t) = t - H t^2 + K t^3 / 3."""
    if name in SHELLS:
        kind, r_w, r_p = SHELLS[name]
        r = np.linalg.norm(xyz, axis=-1) if kind == "sphere" else np.hypot(xyz[..., 0], xyz[..., 2])
        n = 3 if kind == "sphere" else 2
        equidist = (r - r_w) / (r_p - r_w)
        equivol = (r ** n - r_w ** n) / (r_p ** n - r_w ** n)
        ratio = np.full(r.shape, (r_p / r_w) ** (n - 1))
        return equidist, equivol, ratio, None
    fx_, fy_, t = eggcrate_foot_point(xyz)
    _, fx, fy, fxx, fyy, fxy = eggcrate_derivatives(fx_, fy_)
    w = 1 + fx * fx + fy * fy
    H = ((1 + fx * fx) * fyy - 2 * fx * fy * fxy + (1 + fy * fy) * fxx) / (2 * w ** 1.5)
    K = (fxx * fyy - fxy * fxy) / w ** 2
    T = THICKNESS
    V = lambda s: s - H * s ** 2 + K * s ** 3 / 3
    return t / T, V(t) / V(T), 1 - 2 * H * T + K * T ** 2, (fx_, fy_)


PATCH_CENTERS = {"crest (0,0)": (0, 0), "trough (10,0)": (10, 0), "flank (5,0)": (5, 0), "saddle (5,5)": (5, 5)}


def patch_masks(name, xyz, foot=None):
    """Patches as sets of whole columns (analytic): a cone of 25 deg for spheres, a 25 deg wedge for the cylinder,
    and columns whose white-surface foot point lies within 1.5 mm of a center for the folded sheet."""
    if name.startswith("sphere"):
        polar = np.degrees(np.arccos(np.clip(xyz[..., 2] / np.linalg.norm(xyz, axis=-1), -1, 1)))
        return {"cone 25deg": polar <= 25}
    if name == "cylinder_crown":
        phi = np.degrees(np.abs(np.arctan2(xyz[..., 0], xyz[..., 2])))
        return {"wedge 25deg": (phi <= 25) & (np.abs(xyz[..., 1]) <= 1.5)}
    fx_, fy_ = foot
    return {k: np.hypot(fx_ - cx, fy_ - cy) <= 1.5 for k, (cx, cy) in PATCH_CENTERS.items()}
