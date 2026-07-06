#!/usr/bin/env bash
# H3 step 1: build v14/design_k substrate for Istanbul (check2hgi_design_k_resln_mae_l0_1), then postbuild inputs.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
export PYTHONPATH=src
PY=/home/vitor.oliveira/.venv/bin/python
BASE=docs/studies/closing_data/v17_completion/h3_istanbul
echo "=== [step1a] design_k build (resln+mae, 500ep, cuda) $(date '+%F %T') ==="
$PY scripts/probe/build_design_k_delaunay.py --state istanbul \
    --out-suffix resln_mae_l0_1 --epochs 500 --device cuda
rc=$?; echo "design_k rc=$rc"; [ $rc -ne 0 ] && exit $rc
echo "=== [step1b] postbuild inputs (next.parquet + next_region.parquet) $(date '+%F %T') ==="
$PY scripts/pre_freeze_gates/postbuild_v14.py --state istanbul
rc=$?; echo "postbuild rc=$rc"; [ $rc -ne 0 ] && exit $rc
echo "=== STEP1 DONE $(date '+%F %T') ==="
ls -la output/check2hgi_design_k_resln_mae_l0_1/istanbul/
