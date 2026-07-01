# Freeze-readiness — what must close before `closing_data` P2 (the recipe/substrate FREEZE)

> Created 2026-06-17 from the PR #24/#26/#27 audit (24-agent workflow). This is the durable checklist of
> everything standing between "now" and the P2 freeze. The **gate ledger** lives in
> [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md); this doc records the audit's cross-cutting findings
> + the per-gate open work. The pre-freeze GPU/prep lanes are **DONE** (PR #28/#29); the board phase is complete
> and the current closing_data index (board done → baseline phase) is [`HANDOFF.md`](HANDOFF.md).

## 1 · Gate status (post-merge)

**CLOSED** (recorded in the ledger): **C1** (supportive diagnostic panel — single-model headline preserved),
**A2** (substrate claim *strengthened*), **A4** (transductivity defusal), **R1/R2/R3/R10** (all null —
`mtl_frontier` CLOSED, champion G unchanged, **nothing flows to G0.2**).

**OPEN — the freeze cannot commit while any is open:**
- **G0.1 aligned-pairing** — the lone *recipe-changing* P0 gate. Spec in [`PLAN.md §G0.1`](PLAN.md). ≥0.3 pp
  either head (multi-seed) → recipe becomes v17 + STOP for user; null → v16 freezes and "wins without
  per-sample mixing" is earned. **▶ ADVISORY RESOLVED NULL 2026-06-18 (A40, AL+FL seed0):** FL null
  (cat +0.17 / reg ±0.00), AL aligned HURTS cat (−4.77) — random pairing is a beneficial augmentation, not a
  confound. v16 stands; X1 circularity closed. The **binding** {0,1,7,100} run on the FROZEN base remains the
  formal closure (expected null). [`pre_freeze_gates/LANE1_G01_VERDICT.md`](../pre_freeze_gates/LANE1_G01_VERDICT.md).
  **Loss-scale advisory (the §8 #5 companion lever): ▶ EXCLUDED** — FL reg −37.8 pp (harmful at scale);
  [`pre_freeze_gates/LANE1_LOSSSCALE_VERDICT.md`](../pre_freeze_gates/LANE1_LOSSSCALE_VERDICT.md).
- **Overlapping-windows ADOPT/KEEP** — base change (validated AL only). ADOPT ⇒ full base rebuild + a clean
  leak re-audit + baseline re-match; default is KEEP. **User sign-off required** (it accepts the rebuild cost).
- **B1–B5 baseline triage** (P1b) — which external baselines become RUN_MATRIX rows. Reading/decision; the
  reading lane is running now. Final baseline runs fold into P3 and must mirror the adopted windowing.

## 2 · Cross-cutting systemic findings (audit 2026-06-17) — NOT per-PR bugs

### 🔴 BLOCKER — one canonical v14 substrate + hash all consumers
The frozen v14 substrate is the comparability anchor for the whole board, yet across the three studies it
was materialized **three non-identical ways** (PR #27 rebuilt locally on M4/CPU; PR #26 rebuilt locally on
MPS at default build-seed 42; PR #27 A4's train-only arm at seed 0 vs its full-corpus comparand at seed 42),
with **no hash/identity assert anywhere**. Rebuilt comparator arms diverge from the board by up to **+1.8 pp
(HGI)** and **−4/−1/+6 pp (v11, bidirectional)**. Engine-NAME identity holds (the enum key is identical on
every branch); the hazard is purely at the **on-disk artifact** level.
- **Consequence:** each study's *within-study delta* cancels its own substrate offset (so every PR verdict
  holds), but the **absolute** numbers (C1 reg/cat, A2 gap-closure denominators, A4 inflation deltas) are
  **silently incomparable** to the A40/H100 board and to each other. **Do not tabulate any of these three
  studies' absolutes next to the board.**
- **Fix (P3/M0a):** regenerate **ONE** canonical v14 per state on a **fixed machine + fixed build seed**, and
  add a **hash manifest** every consumer checks against before use. See [`M0_P3_PLAN.md`](M0_P3_PLAN.md).

> **A40 UPDATE 2026-06-18 (`study/pre-freeze-a40`):** the freshness-preflight item below is **DONE** — portable
> shared util `src/data/log_t_freshness.py` wired into the two unguarded consumers (`a4_eval.py`,
> `p1_region_head_ablation.py`) + `c1_run_g.sh` made portable (its `stat -f %m` was BSD-only, silently no-op on
> the Linux A40). AL is **NOT stale on the A40** (all reporting seeds' log_T newer than `next_region.parquet`).
> The hash-manifest BLOCKER is **CLOSED (2026-06-19)**: `V14_HASH_MANIFEST.json` records **all 6 states**
> AL/AZ/FL/GE/CA/TX (region cards 1109/1547/4703/2283/8501/6553) — CA+TX built on the A40 (seed 42). The §0
> STOP-condition is **LIFTED** (substrate-identity gate satisfied on one fixed machine+seed; CA/TX carry the
> windowing-independent substrate files, input parquets deferred to the post-windowing-gate per RUN_MATRIX M0).

### 🟠 MAJOR — centralize the stale-`log_T` freshness preflight; AL is stale now
The mandated freshness rule (`log_T` mtime > `next_region.parquet` mtime before any `--per-fold-transition-dir`
run) is an ad-hoc per-script shell snippet: **present** in `c1_run_g.sh`, **absent** in `a4_eval.py` /
`p1_region_head_ablation.py`. The invariant is **currently violated for AL** on disk
(`output/check2hgi_design_k_resln_mae_l0_1/alabama/region_transition_log_seed0_fold*.pt` older than
`next_region.parquet`). It did **not** leak in these runs (the prior is inert in champion G — KD off + α
frozen 0 — and held constant across arms in A2/A4), but P3 / any prior-ON variant would be exposed.
Historical cost of a stale log_T: **+8 pp STL / +12 pp MTL-disjoint reg**.
- **Fix:** a single shared freshness-assert utility wired into every `--per-fold-transition-dir` consumer +
  rebuild the stale AL `design_k` log_T (`compute_region_transition.py --per-fold --seed {S}`).

### 🟠 MAJOR — C1 absolute reg metric ≠ board metric
C1's reported reg Acc@10 is plain `top10_acc` under the **partial `next_forward`** (zero-cat-stream), whereas
the §0.1/RUN_MATRIX board uses `top10_acc_indist` under the **joint forward**. The within-panel Δ (+1.089 pp
pooled) is apples-to-apples and the PROMOTE-as-supportive verdict holds, but the **absolute** C1 reg number
is not board-comparable. (Also: "Δreg non-negative *by construction*" is wrong — it's *empirically*
non-negative across 40/40 folds, on a different metric+forward than the selection monitor.) If a C1 panel is
ever placed beside the board, re-score it on the board metric/forward or carry this caveat explicitly.

### ✅ DONE — gate ledger reconciled
The ledger was divergent across the three branches + main (the `mtl_frontier` R-closures were never
propagated). Reconciled on this merge: R1/R2/R3/R10 marked CLOSED-null; the freeze-readiness status block
added at the top of the ledger.

## 3 · Per-PR doc caveats to apply (non-blocking; before the freeze record cites these numbers)

- **C1_VERDICT.md (#26):** state that the reported C1 reg = plain `top10_acc` under the partial `next_forward`
  (NOT board-comparable without a note); replace "Δreg non-negative by construction" with "empirically
  non-negative across 40/40 folds"; note `cat` is *aggregate* not-hurt (6/40 AL folds show a small negative,
  max −1.16 pp, mean n.s.); add the local-rebuild substrate caveat.
- **A2_RESULTS.md / A4_RESULTS.md / STATE.md (#27):** record the substrate divergence in **strong form** (the
  v11 arm is −4/−1/+6 pp bidirectional vs the board; HGI +1.0–1.8 pp) — only within-PR ratios/deltas are
  citable. Note A4's reg null bounds only the **check-in-graph** channel (the full-corpus Delaunay POI-POI
  spatial reg lever is held fixed in both arms); note the A4 build-seed mismatch (train-only seed 0 vs full
  seed 42) makes the ~0 verdict *conservative*; pair A2 with the existing CH19 per-visit decomposition
  (`RESULTS_TABLE §0.7`) since A2 bounds feature *access*, not a feature-derived *mechanism*.

## 4 · C2 paper-memo pre-prose checklist (banners already added to both memos)

Before the C2 reframing enters `PAPER_DRAFT` §0: verify the Kurin/Xin/**Mueller** citations (Mueller's
venue/year + the *direction* of its finding); fix the **fp16 attribution** in the retraction box (fp16 was
exonerated ~0 pp — the inflated old figure was the class-weighted joint `mtl_cv` default, C25); scope CA/TX
as **expected-but-unmeasured**; say region **PARITY** never "beats"; frame the cat lift as a *deployable*
Pareto gain (head-config asymmetry). Files: `articles/[BRACIS]_Beyond_Cross_Task/{MEMO_2026-06-17…,
C2_REFRAMING_PROPOSAL}.md`.

## 5 · Recommended order before P2

1. **Reading lane (now):** B1–B5 triage + the P1b RUN_MATRIX inventory (no GPU).
2. **A40 lanes (parallel):** G0.1 → multi-seed if it fires; overlapping-windows ADOPT/KEEP recommendation +
   leak re-audit; canonical-v14 substrate builds (CA/TX) + hash manifest (windowing-independent, can start
   now). The **seeded log_T** rebuilds wait for the overlapping-windows decision (windowing-dependent).
3. **User decisions:** G0.1 result (if it fires) · overlapping-windows ADOPT/KEEP · the signed RUN_MATRIX.
4. **P2 freeze** once every ledger row is closed and §2's prerequisites are met → P3 regeneration.
