# HANDOFF_AUDIT — closure punch list for the A40 (2026-06-12)

> **You are the A40 executor. This file is your ENTIRE remaining scope for the `mtl_improvement`
> study.** The study is at scientific close: Tiers 1–5 + R0/R1/R1b/R2 are done, champion G stands
> (matches the STL reg ceiling on the matched metric, beats the STL cat ceiling +2.6…+4.1, 4 states ×
> 4 seeds — `results/mtl_improvement/R0_matched_metric_bar.json`). A 2026-06-12 local re-audit found
> **one data-integrity flaw (P0)** and a few cheap hardening gaps (H1–H3). Execute P0 (mandatory) and
> as much of H1–H2 as is cheap, settle the docs, push. **Then the study closes.**
>
> ⛔ **EXPLICITLY NOT IN SCOPE — do NOT start these:**
> - **T6.1 CA/TX completeness** — DEFERRED out of this study (user decision 2026-06-12). The major
>   large-state runs (v14 builds at CA/TX, G + ceilings, any final-recipe confirmation) belong to the
>   upcoming **`closing-data` study**, which opens after all improvement studies are evaluated and
>   closed, so the heavy compute runs ONCE against the final frozen recipe. The spec it inherits is
>   the T6.1 card in `INDEX.html`.
> - **T6.2 paper-canon restatement** — author-side, not yours.
> - Any new probe/tier. The discovery axes are exhausted (two rising-tide nulls + cos≈0); nothing here
>   re-opens them.

---

## P0 — FL cat-transfer multi-seed integrity (MANDATORY — a paper-bound number is at risk)

**The flaw.** `scripts/mtl_improvement/cat_transfer_manifest.tsv` rows
`catonly_cw1.0_s1|florida`, `_s7|florida`, `_s100|florida` **all point to the SAME rundir**
(`results/check2hgi_design_k_resln_mae_l0_1/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260610_031405_3670616`),
while the three AL seed rows have distinct, time-advancing rundirs (031508 / 031907 / 032306). This is
exactly the rundir-capture-under-concurrency trap `HANDOFF_TIER5.md §4` warns about. Consequence: the
reported FL 4-seed cat+trunk (reg-OFF) mean **72.09 ± 0.08** — and therefore the decomposition
**architecture +2.13 / region-transfer +1.08 at FL** — may be one run counted multiple times. These
numbers are cited by `PAPER_UPDATE.md` item 3, `CHAMPION.md` (cat-ablation banner),
`WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`, `orthogonality_intrinsic_test.md §P1`,
`cat_transfer_and_T53.md`. The AL side (transfer −0.67) has clean distinct rundirs and is NOT at risk.

**Do, in order:**
1. **Forensics on the box**: `ls -dt results/check2hgi_design_k_resln_mae_l0_1/florida/mtlnet_*2026061*`
   — do three distinct FL cat-only rundirs from ~2026-06-10 03:1x exist (seeds 1/7/100)? Check each
   rundir's recorded seed (`summary/full_summary.json` / config dump).
   - **If yes (manifest capture bug only)**: fix the manifest rows, re-aggregate the true 4-seed FL
     mean, and confirm/correct the decomposition.
   - **If no (only one run happened)**: re-run FL `catonly_cw1.0` at seeds {1,7,100} (G recipe +
     `--category-weight 1.0`, same as `cat_transfer_ablation.sh`; ~3 × 14 min), then re-aggregate.
2. **Recompute the decomposition** (4 real seeds): `architecture = cat+trunk − STL cat (69.96)`;
   `transfer = G (73.16) − cat+trunk`. Compare to the published +2.13 / +1.08.
3. **Settle the docs** with the verified numbers (even if unchanged — say "verified, manifest fixed"):
   `cat_transfer_and_T53.md` (also fix the stale "§a … Both seed0" header sentence above the 4-seed
   table), `orthogonality_intrinsic_test.md §P1`, `PAPER_UPDATE.md` item 3, `CHAMPION.md` banner,
   `WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`, INDEX T5/cat-transfer mentions, `log.md` entry.
4. **If FL transfer changes sign or moves > ~0.3pp**, flag it LOUDLY in log.md + PAPER_UPDATE (the
   paper framing "small transfer at scale" depends on it).

## H1 — Orthogonality/cosine: state the n, widen it for free (cheap, no new training)

The load-bearing cos≈0 figure (`figs/grad_cosine_tasks.png`) averages **2 runs total** (FL+AL
static_weight screen, seed 0); the P0-b intrinsic test is 1 run per state and the FL fully-shared
run's **seed is unstated** anywhere. Actions:
1. State every seed explicitly in `orthogonality_intrinsic_test.md` (incl. the FL fully-shared run).
2. The 16 existing G multistate rundirs (`R0_matched_metric_bar.json → g_rundirs`, 4 states × 4 seeds)
   log `grad_cosine_shared` per epoch in their trainer-diagnostics CSVs — re-average the cosine over
   ALL of them (extend `plot_grad_cosine.py`), re-render the figure, and update the WHY doc /
   `T4_audit_and_verdict.md §2` with the widened n ("16 runs, 4 states × 4 seeds" ≫ "2 runs"). If some
   rundirs lack the CSV, average what exists and write down the actual n.

## H2 — Commit the missing raw evidence (cheap, no training)

The Tier-4 **corrected re-run** (static cw {0.50,0.60,0.66,0.80}, gradnorm lr=0.05, nash) and the
**T4.0a scale-norm cw-grid** numbers exist only as markdown tables + remote rundir pointers — no
committed JSON (unlike `T4_full_screen.json`). Aggregate them (the `t4_agg.py` pattern) into
`docs/results/mtl_improvement/T4_corrected_rerun.json` (+ the AL wgrid numbers the verdict doc
summarizes only as "pure weight-reparam"), commit. Note: the local re-audit already corrected the
"per-method-tuned + arch-wired" phrasing across the docs (the nash "retune" at max_norm=2.2 IS the
registry default — a config-identity re-run); keep that wording, don't revert it. Optional single run
if trivial: nash at a genuinely non-default `max_norm` (e.g. 1.0), FL seed0, just to have one true
non-default nash point — not required for closure.

## H3 — Optional (no retune required)

`bayesagg_mtl` (AL cat 37.75) and `excess_mtl` (AL cat 45.97) sit in the screen as craters; the docs
now label them "misconfigured-at-defaults, undiagnosed" (`T4_audit_and_verdict.md §1`). If a 5-minute
log inspection yields a one-line diagnosis, add it; otherwise leave the label.

---

## X-SERIES — deep code-audit probes (2026-06-12; ADDED AFTER the P0/H items — these also GATE closure)

> Full evidence + mechanism: `CODE_AUDIT_2026-06-12.md` (read it first — every probe below has its
> file:line trail there). These four findings were verified at the code level by the design agent.
> They matter because X1/X2/X3 are **MTL-only levers — they CANNOT lift the STL ceiling** (exempt
> from the rising-tide magnitude rule), so they are the last honest shots at "reg beats STL" inside
> this study, AND they stress-test three published claims (mixing-dead, KD-dead-end, "matches").
> Score everything vs the R0 matched bar, multi-seed only on promotion.

### X1 — Cross-attn pairing misalignment (P0-A) — probe, then ONE aligned-training run
Trains on randomly-paired windows (two independent `shuffle=True` loaders, no shared generator —
`folds.py:1054-1080` + `mtl_cv.py:463`), evaluates aligned. Steps:
1. **Roll probe (zero training):** take any saved/retrained G fold; evaluate twice — normal vs
   task-b val batch rolled by 1 (`x_b = torch.roll(x_b, 1, 0)`). Δcat-F1 ≈ 0 → the model ignores
   pairing (mixing truly dead — publish as clean evidence FOR the current claims). Δ > 0 → eval
   depends on an alignment training never saw → step 2 is mandatory.
2. **Aligned-training run:** make both train loaders consume ONE shared permutation (single
   sampler/generator or a joint dataset yielding both modalities — note: passing the same seed to
   two `DataLoader(shuffle=True)` is NOT enough, they draw sequentially from the global RNG).
   Run G at AL+FL seed0 vs the R0 bar. **Promote gate:** lift ≥0.3pp on either head → multi-seed;
   this would re-open the cross-modal-transfer story (and partially un-explain "mixing is dead").
   Null → the misalignment was irrelevant; record and close.

### X2 — Aux-gate fix + the FIRST REAL KD-on-G test (P0-B)
`next_stan_flow_dualtower` is missing from `_HEADS_REQUIRING_AUX_MTL` (`folds.py:933-937`) →
`get_current_aux()` always None → the `c25_gv2.sh` `g_kd0.1/0.2` arms were NO-OPS and every
"prior-ON" dualtower arm was actually prior-OFF. CHAMPION §5's "KD adds nothing on the dual-tower"
is a dead-codepath artifact (already corrected doc-side). Steps:
1. Add `next_stan_flow_dualtower` (+ any `_hsm` variant) to the gate sets (`folds.py:933-937`,
   `p1_region_head_ablation.py:104`) or gate on `hasattr(head, "log_T")`.
2. Smoke: one batch, assert `get_current_aux() is not None` inside the head; assert G's metrics
   unchanged (α=0 buffer + KD 0.0 → bit-identical-by-construction; verify anyway).
3. **The real test:** G + `--log-t-kd-weight 0.2` (τ per v12) at AL+FL seed0 vs G. KD was the one
   confirmed pre-G reg lever and does NOT move the frozen ceiling. Promote gate: ≥0.3pp reg →
   multi-seed.

### X3 — β weight-decay probe (P1-C)
The `aux` fusion scalar β (init 0.1, `next_stan_flow_dualtower/head.py:221`) is weight-decayed at
0.05 (only α is peeled — `helpers.py:148-161`) and never logged. Steps: (1) log β per epoch on the
X2-smoke run (one line next to `head_alpha`, `mtl_cv.py:743-746`); (2) if β decays materially over
training, ONE run with β in a zero-WD group at FL seed0. Gate: ≥0.3pp on either head → multi-seed.

### X4 — Eval-precision parity for the "matches" verdict (P1-D)
MTL eval autocasts fp16 on CUDA (`mtl_eval.py:110-126`); the STL p1 ceiling is fp32; fp16 ties are
scored target-favorably (`metrics.py:114-147`) — and the headline Δreg is −0.09…−0.31pp. Steps:
(1) extend the AMP escape hatch to eval (env `MTL_DISABLE_AMP_EVAL=1` or widen `MTL_DISABLE_AMP`);
(2) retrain ONE G seed at FL (cheapest: reuse the X2-smoke run's weights), score the SAME weights
fp16-eval vs fp32-eval; report Δ. If |Δ| ≳ 0.1pp, re-score the R0 FL row at fp32 and update the
bar's caveat (the verb "matches" is robust either way unless Δ is wildly asymmetric — but the
NUMBER should be precision-clean in the paper).

**Sequencing within the punch list:** P0 (unchanged, first) → X2 step 1-2 + X3 step 1 (one shared
smoke run) → X1 roll probe → X1/X2/X3 full runs as gated → X4 → H1/H2. If any X-probe promotes,
STOP and report before multi-seeding beyond its gate — a positive X result re-opens the champion
question and the user decides scope.

## Guards that now exist (keep them green)

- `tests/test_regression/test_mtl_param_partition.py` now covers the **dualtower family** (G, G′,
  swiglu combo + the other crossattn variants) + a PCGrad gradient-coverage test on G's private tower
  (added 2026-06-12 — the pre-flight that T4.1 mandated but the screen skipped). Run it before
  pushing: `.venv/bin/python -m pytest tests/test_regression/test_mtl_param_partition.py -q`.
- Commit discipline: explicit pathspec only (the repo pre-stages unrelated `articles/*`); pin
  `--canon` in every new script (`CANONICAL_VERSIONS.md §The --canon selector`).

## Done-when (closure checklist)

- [ ] P0 resolved: FL cat-transfer numbers verified-or-rerun, manifest fixed, all citing docs settled.
- [ ] X1: roll probe done; aligned-training run done (or justified-skipped if roll probe Δ≈0 AND
      the aligned run is deemed redundant — record the reasoning); claims updated either way.
- [ ] X2: aux gate fixed + smoke green; REAL KD-on-G test run; CHAMPION/CLAIMS settled with the
      actual result.
- [ ] X3: β logged; no-WD run if β decays.
- [ ] X4: eval-precision Δ measured; R0 FL row re-scored if material.
- [ ] H1: cosine n stated + widened over the existing multistate rundirs; figure re-rendered.
- [ ] H2: `T4_corrected_rerun.json` (+ AL wgrid) committed.
- [ ] (H3 optional.)
- [ ] `log.md` closure entry: "AUDIT PUNCH LIST CLOSED — study CLOSED; CA/TX → closing-data study."
      (If an X-probe PROMOTED, the entry instead reports it and STOPS for the user's scope decision.)
- [ ] Push. Do NOT start CA/TX or any new experiment beyond the gated X-runs.
