# BASELINE handoff — **M4 Pro** (24 GB, MPS) · self-contained · 2026-06-24

> **You are the M4 Pro. Read ONLY this file, then execute.** Phase = baselines. Two jobs: **(1) diagnose the CTLE
> frozen-below-floor — CRITICAL PATH, it gates the H100's FL CTLE-SC run; (2) finish the TX HMT-GRN safety-net
> row.** You are free now; do Task 1 first. Decisions: `../../../articles/[mobiwac]/BASELINE_HANDOFF.md`; results →
> [`RESULTS_BOARD.md`](RESULTS_BOARD.md); cross-machine map → [`BASELINE_DISTRIBUTION.md`](BASELINE_DISTRIBUTION.md).
>
> **Protocol:** seed 0 × 5 folds (n=5), gated stride-1 overlap, leak-free per-fold train-only priors, user-disjoint
> folds. MPS = fp32 (no fp16 confound); HMT-GRN on MPS is validated == CPU within 0.06 pp (PR #38). Keep ≤2–3
> concurrent processes under `scripts/closing_data/ram_watchdog.sh` (4+ → OOM-reboot zone; CA kNN once rebooted the box).

## 0 · Setup (once)
```bash
cd <M4 repo>; git checkout main && git pull
export PYTHONPATH=src
```

## 1 · Task 1 — DIAGNOSE the CTLE frozen-below-floor  ·  CRITICAL PATH (gates H100 step D)
The recorded frozen **CTLE-SC cat at AL = 17.77** is *below* the bigram floor (~19.5). Before the H100 spends FL
compute on CTLE-SC, confirm **whether that is a real CTLE weakness or a pipeline/leak-fix artifact** (e.g. the
frozen CTLE embedding not actually feeding the head, or a min_seq/fold desync). Diagnose on **AL** (small, fast):
```bash
# 1. confirm the per-fold leak-clean CTLE embedding exists + is the frozen CTLE vectors (not zeros / not c2hgi):
ls output/board_baselines/ctle/alabama/s0_f0/embeddings.parquet   # + LEAK_MARKER.txt = "TRAIN-ONLY per fold"
#    spot-check the 64 vector cols are non-degenerate and differ from check2hgi's (substrate actually swapped).
# 2. re-run the SC comparison and read the per-fold trace:
python scripts/closing_data/mac_baseline_compare.py --state alabama --baseline ctle --cells-root output --folds 5 --heads cat
#    -> docs/results/closing_data/baseline_compare/alabama_ctle.json  (expect cat ~17.8 if real)
# 3. cross-check: does the SAME head on check2hgi give ~55.6 on the same rows/folds? (comparand) — if yes, the
#    pipeline is fine and 17.8 is CTLE's true frozen weakness; if the head can't learn on ANY substrate, it's a bug.
```
**Report the verdict** (real CTLE weakness vs pipeline bug) to the orchestrator/H100 **before** the H100 runs FL
CTLE-SC. If it's a bug, identify it (the H100 fixes + reruns); if it's real, the H100 proceeds — and the paper
presents **CTLE-E2E** (~21, the fine-tuned strength) as the headline CTLE number with CTLE-SC as the
matched-frozen-capacity companion (frozen-SC undersells deep models; that's the honest reading).

## 2 · Task 2 — finish TX HMT-GRN (the external-MTL safety-net row)
HMT-GRN is the sole region-native external baseline; AL/AZ/FL/CA/Istanbul are done, **TX is the last cell**
(in-flight: "building dk_ovl → HMT"). It learns its own embeddings from scratch (does NOT consume Check2HGI), so
MPS is fine (validated == CPU). It needs only the dk_ovl **inputs**, not the 12.6 GB log_T (it builds its own
per-fold prior).
```bash
python scripts/mtl_improvement/build_overlap_probe_engine.py texas 1   # if TX dk_ovl inputs absent (big: ~22 G disk)
python scripts/baselines/b3_hmt_grn.py --state texas --engine check2hgi_dk_ovl --seed 0 --folds 5 --epochs 50 --device mps
#  -> results/baseline_b3_hmt_grn_style/texas/ (gitignored); record cat/reg in MACS_BOARD_RESULTS.md + RESULTS_BOARD §4
```
Expected: TX HMT reg well below our MTL ~67 (we lead by the same ≈ +13–16 pp margin as the other states).

## 3 · (Optional) CSLSL @ AL/AZ — only if the A40 stays busy
`python scripts/baselines/b4_cascade.py --state {alabama,arizona} --seed 0 --folds 5 --epochs 50 --device mps` —
MPS from-scratch training is trustworthy here (HMT validated MPS==CPU); **label the result `[M4/MPS]`** and
cross-check one fold vs CPU. Otherwise leave CSLSL to the A40 (CUDA, cleaner vs the champion).

## 4 · Traps + outputs
- ≤2–3 concurrent; `ram_watchdog.sh` kills the newest proc < 4 GB free. Do NOT run a CA/TX embedding-quality kNN here (that's what OOM-rebooted the box).
- All paths train-only per fold (vocab/prior/OOD/fold-split); val users disjoint from train (asserted) — don't "optimise" away.
- Outputs: CTLE diagnosis JSON (`baseline_compare/alabama_ctle.json`, NOT gitignored) + a one-line verdict; TX HMT numbers into `MACS_BOARD_RESULTS.md`. **n=5 provisional.**
- Do **NOT** run the FL block (H100) or any dropped baseline (CTLE-SC ladder / region-SC).
