#!/bin/bash
# Runs the synthetic equi-volume validation and builds the report.
# Usage: run_all.sh <results dir> <path to LN2_LAYERS>
# Needs a Python environment with vdfs installed, plus matplotlib and scipy (see README.md).
set -euo pipefail

results=$(mkdir -p "$1" && cd "$1" && pwd)
ln2=$2
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$here:$results/variants${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$results/out" "$results/out_aligned_grid"

python "$here/make_variants.py" "$results/variants"
for g in sphere_crown sphere_fundus cylinder_crown eggcrate; do
    for v in equidist current wagstyl quadratic; do
        python "$here/run_vdfs.py" "$g" "$v" "$results/out"
    done
    # LayNii in the vdfs box (rim from the vdfs depths) and in a large box (rim from the exact geometry)
    python "$here/run_laynii.py" "$g" vdfs "$results/out_ln2_vdfs_box" "$ln2" "$results/out"
    python "$here/run_laynii.py" "$g" large "$results/out_ln2_large_box" "$ln2"
done
# voxel grid aligned with the mesh: shows voxels without depth on prism boundaries (only equi-distance needed)
python "$here/run_vdfs.py" sphere_crown equidist "$results/out_aligned_grid" aligned
# how LayNii depends on the rim construction and the smoothing (closed shells)
python "$here/laynii_rim_variants.py" "$results/out_rim_variants" "$ln2" | tee "$results/laynii_rim_variants.txt"
python "$here/analyze.py" "$results" | tee "$results/analysis.txt"
python "$here/make_report_data.py" "$results"
python "$here/build_report.py" "$results"
