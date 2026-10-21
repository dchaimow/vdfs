"""Usage: analyze.py <results dir>
Evaluates all methods in the analytic patches (whole columns) against the exact depths: voxel fraction per depth
bin (equi-volume = 10% each) and depth errors. Prints tables and writes <results dir>/results.json."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

import geometry as G

res_dir = Path(sys.argv[1])
N_BINS, N_PROFILE = 10, 20
LABELS = {"sphere_crown": "Sphere, gyral crown", "sphere_fundus": "Sphere, sulcal fundus",
          "cylinder_crown": "Cylinder, crown", "eggcrate": "Folded sheet"}
METHODS = {  # key: label
    "exact": "exact equi-volume",
    "vdfs_equidist": "vdfs equi-distance",
    "vdfs_v1formula": "vdfs <= 1.1 formula",
    "vdfs_linear": "vdfs, linear area model",
    "vdfs_quadratic": "vdfs, quadratic area model",
    "ln2_vdfs_s100": "LayNii, vdfs box, smoothing 100",
    "ln2_vdfs_s1000": "LayNii, vdfs box, smoothing 1000",
    "ln2_large_s100": "LayNii, large box, smoothing 100",
    "ln2_large_s1000": "LayNii, large box, smoothing 1000",
}


def load_ln2(path, ribbon, ref_equidist):
    """LayNii equi-distance and equi-volume metrics, oriented as 0 = white."""
    de = nib.load(f"{path}_metric_equidist.nii").get_fdata()
    dv = nib.load(f"{path}_metric_equivol.nii").get_fdata()
    if np.corrcoef(de[ribbon], ref_equidist[ribbon])[0, 1] < 0:
        de, dv = 1 - de, 1 - dv
    de[~ribbon], dv[~ribbon] = np.nan, np.nan
    return de, dv


def stats(d, truth):
    frac = 100 * np.histogram(np.clip(d, 0, 1 - 1e-12), bins=N_BINS, range=(0, 1))[0] / d.size
    e = d - truth
    which = np.clip((truth * N_PROFILE).astype(int), 0, N_PROFILE - 1)
    profile = [float(e[which == i].mean()) if np.any(which == i) else None for i in range(N_PROFILE)]
    return dict(n=int(d.size), bins=[round(float(x), 2) for x in frac],
                max_dev=round(float(np.abs(frac - 100 / N_BINS).max()), 2), err_mean=round(float(e.mean()), 5),
                err_rms=round(float(np.sqrt((e ** 2).mean())), 5), err_max=round(float(np.abs(e).max()), 5),
                profile=[None if p is None else round(p, 5) for p in profile])


out = {"methods": METHODS, "geometries": []}
for name in G.GEOMETRIES:
    geo = G.make(name)
    xyz = G.voxel_centers(geo["origin"], geo["shape"])
    eqd, eqv, ratio, foot = G.exact_depths(name, xyz)
    exact_ribbon = (eqd > 0) & (eqd < 1)
    patches = G.patch_masks(name, xyz, foot)
    depth = {}
    for v in ("equidist", "v1formula", "linear", "quadratic"):
        depth[f"vdfs_{v}"] = np.load(res_dir / "out" / f"{name}_{v}.npz")["depths"]
    vdfs_ribbon = ~np.isnan(depth["vdfs_equidist"])
    checks = {"vdfs equi-distance vs exact": depth["vdfs_equidist"][vdfs_ribbon] - eqd[vdfs_ribbon]}
    for it in (100, 1000):
        de, dv = load_ln2(res_dir / "out_ln2_vdfs_box" / f"{name}_ln2_s{it}", vdfs_ribbon, eqd)
        depth[f"ln2_vdfs_s{it}"] = dv
        checks[f"LayNii equi-distance, vdfs box, smoothing {it} vs exact"] = de[vdfs_ribbon] - eqd[vdfs_ribbon]
    # large box: own grid
    lo, shp = G.large_box(name)
    xyz_l = G.voxel_centers(lo, shp)
    eqd_l, eqv_l, _, foot_l = G.exact_depths(name, xyz_l)
    ribbon_l = (eqd_l > 0) & (eqd_l < 1)
    patches_l = G.patch_masks(name, xyz_l, foot_l)
    large = {}
    for it in (100, 1000):
        de, dv = load_ln2(res_dir / "out_ln2_large_box" / f"{name}_ln2_s{it}", ribbon_l, eqd_l)
        large[f"ln2_large_s{it}"] = dv
        checks[f"LayNii equi-distance, large box, smoothing {it} vs exact"] = de[ribbon_l] - eqd_l[ribbon_l]

    g = {"name": name, "label": LABELS[name], "equidist_checks": {}, "patches": []}
    print(f"\n=== {LABELS[name]}")
    for k, e in checks.items():
        g["equidist_checks"][k] = [round(float(np.sqrt((e ** 2).mean())), 5), round(float(np.abs(e).max()), 5)]
        print(f"  equi-distance check, {k}: rms {np.sqrt((e ** 2).mean()):.4f}, max {np.abs(e).max():.4f}")
    for pname, mask in patches.items():
        r = ratio[mask & exact_ribbon]
        label = (f"{LABELS[name]}" if name != "eggcrate" else f"{LABELS[name]}, {pname.split(' ')[0]}") \
            + (f" (pial/white area {r.min():.2f})" if np.ptp(r) < 0.01 else f" (pial/white area {r.min():.2f}–{r.max():.2f})")
        p = {"patch": pname, "label": label, "methods": {}}
        sel = mask & exact_ribbon
        p["methods"]["exact"] = stats(eqv[sel], eqv[sel])
        for key, d in depth.items():
            sel = mask & ~np.isnan(d)
            p["methods"][key] = stats(d[sel], eqv[sel])
        for key, d in large.items():
            sel = patches_l[pname] & ~np.isnan(d)
            p["methods"][key] = stats(d[sel], eqv_l[sel])
        g["patches"].append(p)
        print(f"  {label}")
        print(f"    {'method':34s} {'voxels':>7s}  voxel fraction per depth bin (%)                       "
              f"maxdev | error vs exact: mean    rms     max")
        for key, s in p["methods"].items():
            print(f"    {METHODS[key]:34s} {s['n']:7d}  " + " ".join(f"{x:4.1f}" for x in s["bins"])
                  + f"   {s['max_dev']:4.1f}  |       {s['err_mean']:+.4f} {s['err_rms']:.4f} {s['err_max']:.4f}")
    out["geometries"].append(g)
# voxel grid aligned with the mesh: ribbon voxels without depth (points on prism boundaries)
aligned_file = res_dir / "out_aligned_grid" / "sphere_crown_equidist.npz"
if aligned_file.exists():
    geo = G.make("sphere_crown", aligned=True)
    eqd = G.exact_depths("sphere_crown", G.voxel_centers(geo["origin"], geo["shape"]))[0]
    d = np.load(aligned_file)["depths"]
    out["aligned_grid_holes"] = int(np.sum((eqd > 0.02) & (eqd < 0.98) & np.isnan(d)))
    print(f"\nAligned grid (sphere crown): {out['aligned_grid_holes']} ribbon voxels without depth")
(res_dir / "results.json").write_text(json.dumps(out))
