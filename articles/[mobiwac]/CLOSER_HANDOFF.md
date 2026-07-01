# CLOSER_HANDOFF — MobiWac 2026: what's left to close the paper

> **Bottom line.** The **9-page draft is submittable today** (submission sweep 2026-07-01: 0 undefined refs, 0 bibtex
> warnings, 0 overfull boxes, glossary-clean). **Exactly one data gap — P1 (n=20 multi-seed) — changes a reviewer's
> verdict; everything else is coverage/robustness.** Submission is 3 small mechanical steps. This doc is the ordered,
> executable close-out list. Numbers/paths trace to `docs/studies/closing_data/RESULTS_BOARD.md §3`.

## 0 · Status at a glance
- **Paper:** compiling 9-page IEEE two-column draft; abstract + §1–§8 + Tbl 1–3 + Fig 1–4; **26 cited refs**, all resolve.
- **Data:** Part-2 cells are **seed-0 × 5-fold (n=5)** for the 5 Gowalla states; **Istanbul is n=20**; STL **region**
  ceiling is already **n=20 at all 6 states**; all baselines in (HMT-GRN 6 states, faithful STAN AL/AZ/FL/Istanbul,
  ReHDM AL/AZ/FL, CTLE FL, feature-concat FL, CSLSL tie, Markov/POI-RGNN floors).
- **The one verdict-changer:** **P1** (n=20 top-up), **blocked on the H100 lane**.

## 1 · Submission mechanics (before the EDAS upload — not data)
1. **Restore `IEEEtran.bst`.** Currently `\bibliographystyle{ieeetr}` (local fallback; `IEEEtran.bst` absent on this
   machine, `kpsewhich` confirms). On Overleaf / full TeX Live: switch to `\bibliographystyle{IEEEtran}`, rebuild,
   re-verify 0 undefined + reference formatting. (`% NOTE (QA)` in `src/main.tex` documents this.)
2. **EDAS Step 3 manuscript upload.** Paper **#1571313639** is registered (regular track, single-blind); only the PDF
   upload remains. Select the **10-page fee variant** (draft is at 9).
3. **Reconfirm the deadline.** Notes say ~25 Jun 2026 (may be past); verify the live MobiWac/EDAS cycle. Poster cut
   (`archive/PAPER_PLAN_POSTER.md`) is the fallback.
- *(Optional, cosmetic)* standardize the Table 3 `--` "not available" markers (mixed bare `--` for ReHDM-Istanbul vs
  `--`$^{\dagger}$ for CA/TX-infeasible; both read fine).

## 2 · Data gaps to close (ordered by priority)

### ⭐ P1 — n=20 multi-seed top-up {1,7,100}: MTL champion-G + STL category ceiling, 5 Gowalla states  **[VERDICT-CHANGER]**
- **What.** Take Part-2 (Table 3) for AL/AZ/FL/CA/TX from **n=5 (seed 0 × 5f)** to **n=20 (seeds {0,1,7,100} × 5f)**.
  Only the **MTL champion-G** cell and the **STL category ceiling** are missing at each state. **Do NOT re-run the STL
  region ceiling — it is already n=20 at all 6 states.** (Istanbul is already n=20.)
- **Why it matters (the ONLY verdict-changer).** At n=5 every cell sits at the Wilcoxon floor p=0.0312 and per-cell
  Holm cannot clear 0.05. At n=20, real effects reach sub-1e-4, **per-cell Holm clears**, "provisional" becomes
  paper-grade, the pooled-fold pseudoreplication workaround is retired, and the seed-0 development-bias caveat (worst
  at FL/CA/TX, +3..+8 pp) is removed. Defends the panel's #1 attack (single-seed n=5). Affects the whole §6.2
  category-superiority family and the FL/CA/TX region beats (esp. **FL +0.57**, the cell that most needs it).
- **Machine: H100 (required).** A40 verified INFEASIBLE on three walls: FL fp32 overlap MTL ~24 min/epoch (days/seed);
  A40 bf16 backward grad-NaN at large C + fp16 overflow (`TX_A40_BF16_NAN.md`, `CA_MTL_DIVERGENCE.md`); disk (CA/TX OVL
  engines 9–21 GB vs ~16 GB free).
- **Execution.**
  - MTL cells (bare driver defaults to FORBIDDEN fp16 — you MUST export the bf16 env, and pass explicit states because
    the driver's `DEFAULT_STATES` includes out-of-scope **georgia**):
    ```
    MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1 P3_BOARD_CONFIRM=1 \
      bash scripts/closing_data/p3_board.sh \
      --states "alabama arizona florida california texas" --seeds "1 7 100"
    ```
    (bf16 train, fp32 eval; the matched scorer re-forwards in fp32.)
  - STL category ceiling is a **separate run** (`p3_board.sh` runs `--task mtl` only) — per state × seed {1,7,100}:
    ```
    python scripts/train.py --task next --state <state> --engine check2hgi_dk_ovl \
        --model next_gru --folds 5 --epochs 50 --seed <S>
    # then score with scripts/closing_data/score_stl_cat_ceiling.py <rundir> --tag <state>_...
    ```
  - **Preconditions (auto-handled by `p3_board.sh`):** it rebuilds the per-seed seed-tagged region-transition prior
    (`compute_region_transition.py --per-fold --seed <S>`), refuses to launch on a stale log_T or on `torch ≠
    2.11.0+cu128`, and builds the `check2hgi_dk_ovl` engine per state (symlinks frozen v14 embeddings; never clobbers).
    Folds are frozen once, so MTL and its STL ceiling share the same overlap folds (pairing discipline,
    `STATISTICAL_PROTOCOL §4`).
- **Acceptance.** Re-report all §6.2 cells at n=20 via the matched scorer
  (`scripts/closing_data/h100_score_matched.py` / `r0_matched_rescore.py`); re-run the two pre-registered tests
  (`scripts/closing_data/superiority_wilcoxon.py`, `region_match_tost.py`); update §5.3/§6.2 to drop
  "provisional / seed-0 / no-per-cell-Holm" and the pooled-fold fallback; recompile. **No-fold-collapse check**
  (`CLAUDE.md §2b`): reg best-epoch must land **late** (not ≤~5) with no tens-of-thousands of skipped steps — the tell
  of a bf16/fp16 collapse; healthy H100 bf16 stays finite. **Never cite a VOID `*_bf16`/`*_partial` collapse JSON.**
- **Caveats to state.** The seed-0 board is fp32 (AL/AZ/FL/TX) + clean bf16 (CA); the {1,7,100} top-up is pinned bf16,
  so the pool mixes precisions — acceptable per the board's own verdict (bf16≈fp32, Δ≤0.12 pp, eval fp32-matched,
  `RESULTS_BOARD §2`) but disclose it. `RUN_MATRIX §0` still lists 6 states incl. GE — **GE is out of paper scope.**
- **Status: BLOCKED / not-started** (H100 lane). **Non-blocking for submission** (§6.2 already labels cells n=5 provisional).

### P2 — Extend the transductive-leak audit (A4) to CA / TX / Istanbul  **[coverage, not a verdict]**
- **What.** The train-users-only rebuild audit (rebuild the representation per fold on train users only, re-run both
  heads, report Δ) covers **AL/AZ/FL**; extend to a large state (CA or TX) + Istanbul.
- **Why.** The gate is already **null on both axes** at AL/AZ/FL (reg |Δ|≤0.33 pp; cat |Δ|≤0.29 pp on the in-coverage
  subset), so §5.2's leak rebuttal is sourced today. This only answers "is the null shown at scale?" (coverage).
- **Machine: CPU** (A4 eval validated CPU≡MPS); ~3 h/fold (heavier at CA/TX). **Code add first** (`LEAK_AUDIT_EXTEND_HANDOFF §2`):
  add `Resources.TL_CA` / `TL_TX` TIGER tract shapefiles to `src/configs/paths.py` + the `SHAPEFILES` dict in
  `scripts/pre_freeze_gates/a4_build.py`; for Istanbul point at the **mahalle** geojson (not a TIGER tract).
  ```
  for f in 0 1 2 3 4; do python scripts/pre_freeze_gates/a4_build.py --state <state> --seed 0 --fold $f; done
  python scripts/pre_freeze_gates/a4_eval.py     --state <state> --seed 0
  python scripts/pre_freeze_gates/a4_cat_eval.py --state <state> --seed 0
  ```
  (seed 0, same `StratifiedGroupKFold(seed)` split; per-fold train-only log_T; smoke one fold first).
- **Acceptance.** ≥1 large state with **|Δ|≲0.5 pp on both axes** (null holds at scale); add a row to
  `docs/studies/pre_freeze_gates/A4_RESULTS.md`, extend §5.2's state list, commit JSONs under
  `docs/results/pre_freeze_gates/a4/`. If any large state shows non-trivial Δ, **disclose and re-anchor, do not hide.**
- **Caveat.** A4 tests the **design_k** substrate; Istanbul's board cell is on the **GCN** substrate, so an
  A4-Istanbul audits a substrate the board doesn't use for Istanbul — caveat if reported.
- **Status: not-started (CA/TX/Istanbul); deferred post-deadline.**

### P5 — ReHDM-faithful at CA / TX / Istanbul  **[coverage, heavy]**
- **What.** Complete the ReHDM reference row (region baseline; currently AL/AZ/FL = 66.06/54.65/65.68, own protocol).
- **Why.** Published-method reference under its own protocol (chronological 80/10/10, 5 seeds), **never a matched cell**;
  HMT-GRN (primary, 6 states) + faithful STAN already carry the region-external story. Changes no verdict.
- **Machine: heavy — CA/TX ~75–120 h/state** (GPU long-run). Istanbul needs an **FSQ→mahalle region-assignment adapter**
  (the ETL assigns regions via US geometry). Code in `research/baselines/rehdm/`; recipe/order in `STAN_REFOOTING_HANDOFF`.
- **Acceptance.** Per-state JSON under `docs/baselines/next_region/` + `docs/results/…`; Table 3 ReHDM cells filled or
  footnoted infeasible/not-available; update `comparison.md`/`rehdm.md`, `RESULTS_BOARD §4`, `PAPER_PLAN §5.4/§7`.
  **Do NOT fall back to the dropped `stl_check2hgi` variant.**
- **Status: not-started; lowest coverage tier; post-deadline.**

### P4 — Bridging-metrics re-score cells (region Acc@1/@5/MRR; category Acc@1)  **[coverage, cheap — but needs the run-machine]**
- **What.** Fill 3 ladder rows in `BRIDGING_METRICS.md`: MTL champion-G reg Acc@1/@5/MRR; HMT-GRN Acc@1/@5/MRR;
  category Acc@1 (MTL cat + STL ceiling). Re-score saved logits — **no re-training.**
- **Why.** Interpretability anchors so a reader calibrates "is 65.66 Acc@10 good?"; the §6.2 metric-calibration clause
  already gives the random/Markov/majority scales, so these are nice-to-have.
- **Blocker.** The k>10 metrics were not serialized, and the HMT-GRN raw per-fold JSONs + MTL logits are **gitignored /
  not in this checkout** — needs the run-machine artifacts (H100/A40/Mac SSD), not just the repo.
- **Status: not-started; deferred.**

### (Decision, not a ranked gap) Istanbul on the `check2hgi_dk_ovl` substrate?
- Istanbul's Part-2 cells are on the **earlier GCN / stride-1** substrate, not the design_k `dk_ovl` substrate the 5
  Gowalla states use. The paper already handles this honestly (reported as **gap-to-ceiling / lift, never absolute
  Acc@10**; the beats-cat / matches-reg pattern replicates regardless of substrate). Rebuilding on dk_ovl would remove
  the "cross-substrate" caveat and make both parts fully six-state on one windowing, but it is **net-new work (a full
  Istanbul board regeneration, H100), on no ranked backlog item**. **Recommendation: keep the honest cross-substrate
  caveat; do NOT rebuild unless the reviewers specifically ask.** (Author decision, not a to-do.)

## 3 · Already CLOSED — do NOT redo
Faithful **STAN FL** (Acc@10 **72.99**±0.34, `faithful_stan_florida_5f_200ep_v6_opt.json`; CA/TX optional→footnoted);
HGI-Istanbul → Tbl 2 (+26.64); Tbl 2 substrate contrast, all 5 Gowalla on one windowing; W6 encoder-isolation probe;
CSLSL cascade tie; FL CTLE-E2E + CTLE-SC (AL/AZ/Istanbul); HMT-GRN (6 states); feature-concat control (FL); §5.2 leak-Δ
sourcing (A4 AL/AZ/FL); reviewer-clarity + bib hygiene.

## 4 · Stale docs to fix (found by the close-out scrape)
1. `PAPER_PLAN §10` says faithful-STAN-FL "in-flight" — **it is DONE** (STAN FL = 72.99).
2. `RUN_MATRIX §0` + `p3_board.sh DEFAULT_STATES` include **georgia** — **out of paper scope**; pass explicit
   `--states` for P1.
3. The backlog's single `p3_board.sh` command runs the **MTL cell only** — the STL category ceiling at {1,7,100} is a
   **separate** `train.py --task next … --model next_gru` run.

## 5 · Priority ledger
**P1 (H100, n=20 MTL + STL-cat, VERDICT-CHANGER, blocked)** ≫ P2 (A4 leak CA/TX/Istanbul, coverage, needs a shapefile
code-add) > P5 (ReHDM CA/TX/Istanbul, coverage, ~75–120 h/state + Istanbul adapter) > P4 (3 bridging re-score cells,
cheap but needs gitignored logits). **STAN-FL is done; Istanbul-on-dk_ovl is not a gap. Only P1 changes a verdict — the
draft is submittable today.**
