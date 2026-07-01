# CA/TX v17 n=20 — H100 handoff

**Why this exists.** v17 (champion = v16 + bs8192 + per-head cat-lr 1e-3) is n=20-confirmed at
AL/AZ/FL (`perhead_lr_n20.md`) and is now `DEFAULT_CANON`. The board's CA/TX cells still need the
n=20 top-up (seeds {1,7,100} on top of the existing seed-0 board cells). This is the **H100 driver**
— the faster/parallel option. **NB (2026-07-01 correction):** an earlier claim that the A40 "cannot do
this in feasible time (~72 days)" was a **tqdm misread** — the real cost is **~53 min per FOLD**
(4.53 batch/s; 14300 batches = all 50 epochs, NOT per-epoch) → ~4.4 h/cell, **~35 h (~1.5 days) for
8 cells SERIAL, which IS feasible on the A40** (serial only — 2-wide trips the host-RAM fold-construction
guard + pins VRAM 45/46). So the H100 is the *fast* path (parallel, ~hours), not the *only* path.

**A40 in the meantime:** a **fold-0 audit** of CA + TX (`run_catx_v17_audit_1fold.sh`,
`catx_v17_audit/`) confirms the v17 recipe runs end-to-end at the big states + gives a real fold-0
number to cross-check the H100 later. It is NOT the n=20 result.

## Run it (on the H100)

```bash
# 0. repo at this commit, venv active, torch 2.11.0+cu128 (edit the venv path in the driver header if different)
# 1. substrates present (v14 + gated overlap engine) — the driver preflight-checks provenance and aborts if bad:
#    output/check2hgi_design_k_resln_mae_l0_1/{california,texas}/*.parquet
#    output/check2hgi_dk_ovl/{california,texas}/input/next_region.parquet  (stride1/emit_tail=false/min_seq10)
#    build overlap if missing:  python scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1 10
# 2. launch (MAX_PAR=3 is safe on 80GB — each cell ~22-26GB VRAM):
bash docs/studies/closing_data/run_catx_v17_n20_h100.sh 3
```

Results land in `catx_v17_n20_h100/{state}_s{seed}/` (per-cell `run.log` + `profile.json`), scored
PID-keyed by `a40_score_matched.py`, aggregated to per-state `cat±/reg±` at the end + `summary.tsv`.

## Recipe (identical to the A40 audit, minus concurrency)

- engine `check2hgi_dk_ovl` (gated stride-1 overlap of the v14 substrate, MIN_SEQ=10)
- v17: `--canon none` + explicit recipe, `--batch-size 8192`, `MTL_ONECYCLE_PER_HEAD_LR=1`
  (per-head cat/reg/shared LR 1e-3/3e-3/1e-3 actually applied), `--onecycle max-lr 3e-3`
- true **fp32** (`MTL_DISABLE_AMP=1`) — matches the fp32 board TX cell; avoids the bf16-at-large-C question
- heads `next_gru`(cat) + `next_stan_flow_dualtower`(reg, prior-OFF freeze_alpha/alpha_init=0), `geom_simple` selector
- `MTL_STRICT=1` (so `--canon none` + full explicit recipe is required — auto-v17 would pin the v14
  substrate + hard-fail the dk_ovl wrong-substrate guard)
- per-fold log_T is **inert** (prior-OFF + KD-off) → `MTL_SKIP_INERT_LOGT` default-on drops the load;
  no `region_transition_log_*.pt` needed

## When it lands
Fold the per-state `cat±/reg±` into `RESULTS_BOARD.md §1` as the CA/TX v17 n=20 cells, then (with
AL/AZ/FL already done) promote v17 into the §1 headline. Also land the flag-OFF eager-parity test
(`future_works/per_head_lr_onecycle_fix.md`).
