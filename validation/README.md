# Equi-volume validation on synthetic surfaces

Checks how well vdfs equi-volume depths preserve volume, on synthetic white/pial surfaces with known geometry and exact depths, and compares them with LayNii's `LN2_LAYERS -equivol`.

## What is tested

- **Geometries** (`geometry.py`), cortical thickness 3 mm:
  - `sphere_crown`: white r = 4 mm, pial r = 7 mm (pial/white area 3.06)
  - `sphere_fundus`: pial r = 3 mm inside, white r = 6 mm outside (area ratio 0.25)
  - `cylinder_crown`: white r = 4 mm, pial r = 7 mm (area ratio 1.75)
  - `eggcrate`: folded sheet, white z = 2 cos(kx) cos(ky) (20 mm wavelength), pial = parallel surface 3 mm along the normal (crest, trough, flank, saddle)
- **Exact depths** (`geometry.exact_depths`): columns are radial in the shells and follow the white-surface normals in the folded sheet. There the area at distance t along a normal changes by 1 − 2Ht + Kt² (mean and Gaussian curvature H, K), so the exact equi-volume depth is V(t)/V(T) with V(t) = t − Ht² + Kt³/3.
- **Voxel grid**: 0.125 mm (24 voxels across the cortex), shifted by an irregular offset so that no voxel center lies exactly on a boundary between prisms. The `aligned` option leaves out the offset; `analyze.py` counts ribbon voxels without depth on this grid (points on prism boundaries, 689 in vdfs ≤ 1.1, see #10).
- **Methods**:
  - vdfs equi-distance, and equi-volume with vdfs' two area models: linear (Waehnert et al., 2014; surface_tools), and quadratic, A(α) = a_w (1 − α)² + 2 a_m α (1 − α) + a_p α² with a_m = 2 a_mid − (a_w + a_p)/2 from the area of the midthickness surface (white + pial)/2. For comparison, the formula of vdfs ≤ 1.1, in a copy of the installed vdfs module in which only `alpha_from_beta` is replaced (`make_variants.py`).
  - LayNii `LN2_LAYERS -equivol -incl_borders` with `-iter_smooth` 100 (default) and 1000 (`run_laynii.py`), in the vdfs voxel boxes (rim from the vdfs depths) and in larger boxes in which the ribbon is not cut near the patches (rim from the exact geometry). Rim in LayNii's convention: border labels 2 (white side) and 1 (pial side) on the outermost grey matter voxels, 3 on the other grey matter voxels.
  - `laynii_rim_variants.py`: how LayNii's results depend on the rim construction (borders outside the grey matter, outermost grey matter voxels, or centered on the surface) and the smoothing, on closed shells.
- **Checks** (`analyze.py`), in patches of whole columns (`geometry.patch_masks`): errors against the exact equi-volume depth, and voxel counting (equi-volume depths put 10% of the voxels into each tenth of the depth).

## Running

Requirements: a Python environment with vdfs installed (`pip install -e ..` from this folder for the checkout itself) plus `matplotlib`; and a compiled `LN2_LAYERS` from [LayNii](https://github.com/layerfMRI/LAYNII), e.g.

```bash
cd LAYNII && c++ -std=c++11 -DHAVE_ZLIB -O3 -o LN2_LAYERS src/LN2_LAYERS.cpp \
    dep/nifti2_io.cpp dep/znzlib.cpp dep/laynii_lib.cpp -I./dep -lm -lz
```

Then:

```bash
./run_all.sh results /path/to/LN2_LAYERS
```

This takes about 5 minutes (vdfs runs with 10 threads) and writes to `results/`:

- `out/`, `out_aligned_grid/`: vdfs depths and columns (`.npz`)
- `out_ln2_vdfs_box/`, `out_ln2_large_box/`, `out_rim_variants/`: LayNii rims and outputs (`.nii`)
- `analysis.txt`, `results.json`, `laynii_rim_variants.txt`: voxel fractions per depth bin and errors against the exact depths
- `report/vdfs_equivolume_report.html`: report with layer images and charts (single self-contained file)

## Results (vdfs 2.0 development version, LayNii 2.9.0)

Largest deviation from 10% in any depth bin (percentage points) and rms depth error against the exact depth; LayNii in the large box:

| Patch | equi-distance | vdfs ≤ 1.1 formula | linear area model | quadratic area model | LayNii, smoothing 100 | LayNii, smoothing 1000 |
|---|---|---|---|---|---|---|
| Sphere, gyral crown | 5.2 / 0.099 | 1.7 / 0.015 | 0.5 / 0.008 | 0.1 / 0.002 | 1.7 / 0.045 | 2.8 / 0.017 |
| Sphere, sulcal fundus | 6.4 / 0.119 | 2.6 / 0.022 | 0.8 / 0.012 | 0.2 / 0.002 | 2.2 / 0.041 | 3.3 / 0.018 |
| Cylinder, crown | 2.7 / 0.051 | 0.5 / 0.005 | 0.3 / 0.001 | 0.3 / 0.001 | 1.7 / 0.039 | 1.9 / 0.014 |
| Folded sheet, crest | 3.6 / 0.078 | 1.2 / 0.010 | 0.4 / 0.005 | 0.2 / 0.002 | 5.8 / 0.069 | 3.5 / 0.043 |
| Folded sheet, trough | 6.2 / 0.130 | 3.0 / 0.027 | 1.0 / 0.015 | 0.5 / 0.002 | 5.5 / 0.088 | 5.2 / 0.074 |
| Folded sheet, flank | 0.1 / 0.019 | 0.2 / 0.001 | 0.2 / 0.000 | 0.2 / 0.000 | 1.4 / 0.013 | 2.0 / 0.015 |
| Folded sheet, saddle | 2.0 / 0.031 | 0.8 / 0.006 | 0.5 / 0.004 | 0.3 / 0.001 | 3.3 / 0.029 | 3.7 / 0.032 |

With exact depths the bin deviations are 0.1–0.5 (counting noise), and vdfs' equi-distance depths differ from the exact ones by at most 0.0026 (approximation of the surfaces by triangles), which bounds the accuracy any vdfs variant can reach here.
