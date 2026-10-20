"""Usage: make_report_data.py <results dir>
Writes the report's chart data (report_data.json) and layer images (fig_*_{light,dark}.png) to
<results dir>/report, from results.json (analyze.py), the depth maps and the LayNii outputs."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import nibabel as nib
import numpy as np

import geometry as G

R = Path(sys.argv[1])
FIG = R / "report"
FIG.mkdir(exist_ok=True)
VS = G.VS
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", muted="#52514e", line="#0b0b0b", nan="#e9e8e3",
                  div=["#2a78d6", "#f0efec", "#e34948"], hole="#e34948"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", muted="#c3c2b7", line="#f0efec", nan="#2c2c2a",
                 div=["#3987e5", "#383835", "#e66767"], hole="#e66767"),
}
# ordinal layer ramp (10 layers, white -> pial)
LAYERS = ListedColormap(["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                         "#256abf", "#1c5cab", "#104281"])

# ---------------------------------------------------------------- chart data
data = {"results": json.loads((R / "results.json").read_text()), "formula": [], "rim_variants": []}
beta = np.linspace(0, 1, 41)
for ratio in (3.0, 1 / 3):
    aw, ap = 1.0, ratio
    cur = (2 * ap * beta - ap * beta ** 2 + aw * beta ** 2) / (ap + aw)
    wag = (-aw + np.sqrt((1 - beta) * aw ** 2 + beta * ap ** 2)) / (ap - aw)
    data["formula"].append({"label": f"pial/white area ratio {ratio:.2g}", "x": beta.round(3).tolist(),
                            "series": {"current": cur.round(4).tolist(), "wagstyl": wag.round(4).tolist()}})
# rim sensitivity (laynii_rim_variants.py output)
geom = None
for line in (R / "laynii_rim_variants.txt").read_text().splitlines():
    if line.startswith("==="):
        geom = line.strip("= ").strip()
    m = re.match(r"\s+(outside|inside|centered)\s+(defaults|-iter_smooth \d+)\s+(\d+) \| ([-+.\d]+) ([.\d]+)\s+\| "
                 r"([-+.\d]+) ([.\d]+) ([.\d]+)", line)
    if m:
        data["rim_variants"].append(dict(geometry=geom, rim=m[1], options=m[2], equidist_mean=float(m[4]),
                                         equidist_max=float(m[5]), mean=float(m[6]), rms=float(m[7]), max=float(m[8])))
(FIG / "report_data.json").write_text(json.dumps(data))


# ---------------------------------------------------------------- images
def load_vdfs(name, variant):
    return np.load(R / "out" / f"{name}_{variant}.npz")["depths"]


def load_ln2(name, box, it, ref):
    """LayNii equi-volume metric (0 = white), NaN outside the grey matter."""
    d = nib.load(R / f"out_ln2_{box}_box" / f"{name}_ln2_s{it}_metric_equivol.nii").get_fdata()
    gm = nib.load(R / f"out_ln2_{box}_box" / f"{name}_rim.nii").get_fdata() > 0
    if np.corrcoef(d[gm], ref[gm])[0, 1] < 0:
        d = 1 - d
    return np.where(gm, d, np.nan)


def large_cropped(name, it):
    """LayNii result and exact depths of the large box, cropped to the vdfs box."""
    geo = G.make(name)
    lo, shp = G.large_box(name)
    eqd, eqv, *_ = G.exact_depths(name, G.voxel_centers(lo, shp))
    d = load_ln2(name, "large", it, eqd)
    i0 = np.round((geo["origin"] - lo) / VS).astype(int)
    s = tuple(slice(i0[k], i0[k] + geo["shape"][k]) for k in range(3))
    return d[s], eqv[s]


def slice_y0(vol, origin):
    return vol[:, int(round((0.0 - origin[1]) / VS)), :].T  # (z, x)


def extent(lo, shape):
    return [lo[0] - VS / 2, lo[0] + (shape[0] - 0.5) * VS, lo[2] - VS / 2, lo[2] + (shape[2] - 0.5) * VS]


def show_layers(ax, d, ext, t):
    layers = np.where(np.isnan(d), np.nan, np.clip(np.floor(d * 10), 0, 9))
    ax.set_facecolor(t["nan"])
    ax.imshow(layers, origin="lower", extent=ext, cmap=LAYERS, vmin=-0.5, vmax=9.5, interpolation="nearest")


def show_error(ax, e, ext, t, lim):
    ax.set_facecolor(t["nan"])
    cmap = LinearSegmentedColormap.from_list("div", t["div"])
    return ax.imshow(e, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")


def style(ax, t, title):
    ax.set_title(title, color=t["ink"], fontsize=10, loc="left")
    ax.tick_params(colors=t["muted"], labelsize=8)
    for s in ax.spines.values():
        s.set_color(t["muted"])
    ax.set_xlabel("x (mm)", color=t["muted"], fontsize=8)
    ax.set_ylabel("z (mm)", color=t["muted"], fontsize=8)
    ax.set_aspect("equal")


def colorbar(fig, im, ax, t, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=t["muted"], labelsize=8)
    cb.outline.set_edgecolor(t["muted"])
    cb.set_label(label, color=t["muted"], fontsize=8)


def save(fig, name, theme):
    fig.savefig(FIG / f"{name}_{theme}.png", dpi=110, facecolor=THEMES[theme]["bg"], bbox_inches="tight")
    plt.close(fig)


def surface_curves(geo, y0):
    """Intersection of the white and pial meshes with the plane y = y0."""
    out = []
    for key in ("white", "pial"):
        v, pts = geo[key], []
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            p, q = v[geo["faces"][:, a]], v[geo["faces"][:, b]]
            s = (y0 - p[:, 1]) / np.where(q[:, 1] != p[:, 1], q[:, 1] - p[:, 1], np.nan)
            ok = (s >= 0) & (s <= 1)
            pts.append(p[ok] + s[ok, None] * (q[ok] - p[ok]))
        out.append(np.concatenate(pts))
    return out


cases = {}
for name in ("eggcrate", "sphere_crown", "sphere_fundus"):
    geo = G.make(name)
    xyz = G.voxel_centers(geo["origin"], geo["shape"])
    eqd, eqv, *_ = G.exact_depths(name, xyz)
    exact = np.where((eqd > 0) & (eqd < 1), eqv, np.nan)
    c = dict(geo=geo, exact=exact, ext=extent(geo["origin"], geo["shape"]),
             vd={v: load_vdfs(name, v) for v in ("equidist", "current", "wagstyl")})
    for it in (100, 1000):
        c[f"large{it}"], c[f"large_truth{it}"] = large_cropped(name, it)
    c["box1000"] = load_ln2(name, "vdfs", 1000, c["vd"]["equidist"])
    cases[name] = c

for theme, t in THEMES.items():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    s = lambda c, a: slice_y0(a, c["geo"]["origin"])

    # folded sheet and spheres: layers (top) and errors against the exact depth (bottom)
    for name, size, lim in (("eggcrate", (16, 6.4), 0.05), ("sphere_crown", (14, 6.6), 0.04),
                            ("sphere_fundus", (14, 6.6), 0.04)):
        c = cases[name]
        fig, axes = plt.subplots(2, 4, figsize=size, facecolor=t["bg"])
        tops = [(c["exact"], "exact equi-volume layers"), (c["vd"]["current"], "vdfs, current formula"),
                (c["vd"]["wagstyl"], "vdfs, Wagstyl formula"), (c["large1000"], "LayNii, smoothing 1000")]
        for ax, (d, title) in zip(axes[0], tops):
            show_layers(ax, s(c, d), c["ext"], t)
            style(ax, t, title)
        bottoms = [(c["vd"]["current"] - c["exact"], "error, current formula", lim),
                   (c["vd"]["wagstyl"] - c["exact"], "error, Wagstyl formula", lim),
                   (c["large100"] - c["large_truth100"], "error, LayNii, smoothing 100", 0.15),
                   (c["large1000"] - c["large_truth1000"], "error, LayNii, smoothing 1000", 0.15)]
        for ax, (e, title, l) in zip(axes[1], bottoms):
            im = show_error(ax, s(c, e), c["ext"], t, l)
            style(ax, t, title)
            colorbar(fig, im, ax, t, f"depth error (±{l})")
        if name == "eggcrate":
            for ax in axes.flat:
                for curve in surface_curves(c["geo"], c["geo"]["origin"][1] + round(-c["geo"]["origin"][1] / VS) * VS):
                    ax.plot(curve[:, 0], curve[:, 2], ".", ms=0.8, color=t["line"])
                ax.set_xlim(-3, 13)
                ax.set_ylim(-2.3, 5.3)
        fig.tight_layout()
        save(fig, f"fig_{name}", theme)

    # LayNii: box size
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), facecolor=t["bg"])
    for row, name in enumerate(("sphere_crown", "eggcrate")):
        c = cases[name]
        for col, (e, title) in enumerate([(c["box1000"] - c["exact"], "vdfs box (ribbon cut at the box faces)"),
                                          (c["large1000"] - c["large_truth1000"], "large box (cropped for display)")]):
            ax = axes[row, col]
            im = show_error(ax, s(c, e), c["ext"], t, 0.15)
            style(ax, t, f"LayNii error, smoothing 1000: {title}")
            colorbar(fig, im, ax, t, "depth error (±0.15)")
            if name == "eggcrate":
                ax.set_xlim(-3, 13)
    fig.tight_layout()
    save(fig, "fig_laynii_box", theme)

    # vdfs: voxels without depth on prism boundaries (aligned grid)
    name = "sphere_crown"
    aligned = G.make(name, aligned=True)
    lo, shape = aligned["origin"], aligned["shape"]
    d = np.load(R / "out_aligned_grid" / f"{name}_equidist.npz")["depths"]
    eqd = G.exact_depths(name, G.voxel_centers(lo, shape))[0]
    inside = (eqd > 0.02) & (eqd < 0.98)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=t["bg"])
    for ax, y, title in [(axes[0], 0.0, "slice y = 0 (a plane containing mesh edges)"),
                         (axes[1], 0.5, "slice y = 0.5 mm")]:
        j = int(round((y - lo[1]) / VS))
        sl, ins = d[:, j, :].T, inside[:, j, :].T
        img = np.empty(sl.shape + (3,))
        img[:] = matplotlib.colors.to_rgb(t["nan"])
        img[~np.isnan(sl)] = matplotlib.colors.to_rgb("#86b6ef")
        img[ins & np.isnan(sl)] = matplotlib.colors.to_rgb(t["hole"])
        ax.imshow(img, origin="lower", extent=extent(lo, shape), interpolation="nearest")
        style(ax, t, title)
    fig.tight_layout()
    save(fig, "fig_holes", theme)
print("written", sorted(p.name for p in FIG.iterdir()))
