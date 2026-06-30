# CLAUDE.md — MobiWac 2026 paper (working folder)

> **What this folder is.** The working folder for our **MobiWac 2026** submission,
> *"Predicting the Next Category and Region of a Visit: A Check-in-Level Multi-Task Study on Mobility Data."*
> The paper itself is in [`src/`](src/) (`main.tex`, a compiling 9-page IEEE two-column draft). This file is the
> landing: the settled state, the decisions ledger, the conventions, and the doc map. **Read this first**, then
> [`GLOSSARY.md`](GLOSSARY.md) (the writing law) and [`PAPER_PLAN.md`](PAPER_PLAN.md) (the section-by-section spine).

## 1 · Current state (settled as of 2026-06-28)

- **Paper:** complete, compiling **9-page** IEEE draft. Root: [`src/main.tex`](src/main.tex) (renamed from
  `paper_skeleton.tex`). Build: `pdflatex main → bibtex main → pdflatex main ×2` (see [`src/README.md`](src/README.md)).
  All 8 sections written, 3 tables + 4 figures wired, **0 undefined refs/citations, 0 bibtex warnings**.
- **Submission:** EDAS paper **#1571313639** is **already registered** (regular track, single-blind); only the
  **manuscript upload (EDAS Step 3)** is left. The current build is **9 pages**; the budget is the **10-page (fee)
  variant** (do NOT trim to 8). See [`EDAS_SUBMISSION.md`](EDAS_SUBMISSION.md).
- **Authors (names stay, single-blind):** Vitor H. O. Silva, Germano B. dos Santos, Fabrício A. Silva — NESPeD-LAB,
  Universidade Federal de Viçosa, Florestal, MG; `{vitor.h.oliveira, germano.santos, fabricio.asilva}@ufv.br`.
- **Bibliography:** 26 cited references, **all web-verified real and current**. `\bibliographystyle{ieeetr}` is a
  LOCAL fallback (IEEEtran.bst is absent on this machine); **restore `IEEEtran.bst` at submission** (Overleaf has it).

## 2 · The science (the thesis + headline; canonical numbers live in the board)

Two parts, one model. **Part 1:** a check-in-level representation makes next-category prediction far more learnable
than a place embedding. **Part 2:** a single joint model predicting next-category and next-region together
**outperforms** the dedicated single-task category model at **every** state (about **+4.7 to +7.7** macro-F1) and,
on next-region, **outperforms** at the large region counts (FL/TX/CA) while being **statistically non-inferior within a two-point margin
(TOST)** at the small (AL/AZ/Istanbul). The region gain rises with the number of regions.

- **Headline snapshot (verify exact cells against the board — this is a memory aid, not the source of truth):**
  category Δ (MTL − single-task) = AL +7.69 / AZ +6.26 / FL +4.68 / CA +7.07 / TX +7.56 / Istanbul +6.69;
  region Δ = FL +0.57 / TX +2.06 / CA +2.18 (**outperforms**), AL −0.18 / AZ −0.06 / Istanbul −0.52 (**matches**, TOST);
  Table 2 substrate margin (Check2HGI − HGI cat) ≈ +26.6 to +39.6 across the 6 datasets.
- **Scope:** five Gowalla states (AL/AZ/FL/CA/TX) + Istanbul (Massive-STEPS, non-U.S. external check).
- **Significance:** n=5 (seed 0) provisional for Gowalla, n=20 for Istanbul. At n=5 the one-sided Wilcoxon is
  **floored at p=0.0312** — "5/5 folds, p=0.031" is **at-ceiling for n=5, NOT "barely significant"** (do not soften
  the claim on this basis). Per-cell Holm cannot clear 0.05 at n=5, so we report a **state-level** 6/6 sign test
  (p≈0.016) as the backbone and TOST for the matches; **no per-cell Holm at n=5**. The **n=20 multi-seed top-up (P1)
  is the one thing that breaks this ceiling** and is the top open item (§5).
- **Leak rebuttal (the BRACIS-reject answer):** the A4 train-users-only transductivity audit is **null on both axes**
  at AL/AZ/FL (region ≤0.33 pp, category ≤0.29 pp), sourced in §5.2. Caveat: category is a POI-proxy on the
  in-coverage subset (the transductive substrate can't measure cold-POI visits) — state it, don't hide it.
- **Baselines:** HMT-GRN (primary region-native, 6 states), faithful STAN (AL 60.72 / AZ 49.86 / FL **72.99** /
  Istanbul 61.86; CA/TX infeasible; the old AL 34.46 / AZ 38.96 below-Markov numbers are a **superseded under-trained
  collapse — never cite**), ReHDM (own-protocol reference), POI-RGNN + Markov-9 (category), CTLE + feature-concat
  (FL representation control), CSLSL cascade (a **tie at equal cost**, framed as a defense — never "we beat the cascade").

> ⭐ **Canonical numbers are NOT duplicated here.** The single source of truth for every cell is
> [`docs/studies/closing_data/RESULTS_BOARD.md §1`](../../docs/studies/closing_data/RESULTS_BOARD.md), and the
> paper-facing claim-discipline whitelist (exact CAN-say / PROVISIONAL / must-NOT-say numbers) is
> [`PAPER_PLAN.md §3`](PAPER_PLAN.md). Pull numbers from those, not from memory or from the archived handoffs.

## 2b · Where the data lives, and how to verify any number

**The data is NOT in this folder.** Every result the paper cites is a JSON under the repo's `docs/results/` tree,
indexed by the board. To trust or change any number, trace it to its JSON — do not take a number from prose.

**The map (all paths from the repo root):**
- **The board + its file map** — [`docs/studies/closing_data/RESULTS_BOARD.md`](../../docs/studies/closing_data/RESULTS_BOARD.md):
  §1 is the headline table, **§3 is "where every result lives"** (each cell → its exact JSON path), §4 is the baselines.
  Start here for any number.
- **Joint (MTL) + single-task (STL) cells:** `docs/results/closing_data/h100/` and `a40/`. **Do not guess the
  filenames — the board's §3 file-map gives the exact path per state** (they vary: AL/AZ are
  `<state>_s0_mtl_fp32_matched_score.json`, **FL** carries a `_5f_` infix `florida_s0_mtl_fp32_5f_matched_score.json`,
  **CA**'s citable cell is `h100/california_s0_mtl/california_s0_mtl_final_score.json` (a `_final_score.json` in a
  subdir, **bf16**), **TX** is `a40/tx_ba2_fp32_s0.json`). STL cat ceilings: `*_s0_stl_cat_ceiling.json`. Also in
  `a40/`: the CSLSL cascade `*_cascade_s0.json`, the W6 probe `*_w6_freezereg_s0.json`.
- **STL region ceilings (the "Dedicated" reg column):** `docs/results/P1/region_head_<state>_region_5f_50ep_<state>_ovl_stl_reg_s0.json`
  (the second slot is the full name for AL/AZ/FL but **abbreviated for TX/CA**: `…_tx_ovl_…` / `…_ca_ovl_…`).
- **Substrate Table 2 (HGI place-level cat):** `docs/results/closing_data/baseline_compare/<state>_hgi_ovl_cat.json`.
- **Baselines:** faithful STAN / POI-RGNN → `docs/results/baselines/faithful_*`; CTLE / cascade / SC →
  `docs/results/closing_data/baseline_compare/`; consolidated per-state → `docs/baselines/next_{region,category}/results/<state>.json`.
- **Floors:** Markov-1 region + Markov-9 category → `docs/results/P0/simple_baselines/<state>/`.
- **Leak audit (A4, the §5.2 numbers):** [`docs/studies/pre_freeze_gates/A4_RESULTS.md`](../../docs/studies/pre_freeze_gates/A4_RESULTS.md)
  + the JSONs at `docs/results/pre_freeze_gates/a4/`.
- **Istanbul (the non-U.S. dataset):** `docs/results/second_dataset/istanbul/`. The board's n=20 cell is the
  **stride-1** consolidation `istanbul_stride1_multiseed_summary.json` (the bare `istanbul_s{0,1,7,100}_*` files are
  the non-overlap per-seed runs — do not use those for the board cell).
- **Model inputs / embeddings** (the substrate itself): `output/` — **gitignored**, lives only on the run machines.

**How to verify (the "comprove" recipe):** (1) find the cell in `RESULTS_BOARD §1`; (2) follow **§3** to its exact
JSON (the filenames vary by state — let §3 tell you, don't pattern-match). (3) Read the per-fold arrays and means:
category is `cat_macro_f1_mean` (from `cat_per_fold`); region is `reg_full_top10_mean` (the OOD-discounted "full"
metric `top10_acc_indist*(1-ood_frac)`, from `reg_per_fold`). (Key names vary by producer: the TX cell
`a40/tx_ba2_fp32_s0.json` uses `mtl_cat_macro_f1` / `mtl_reg_full_top10` instead — read the JSON's keys, don't assume.) The cell is the **per-task diagnostic-best** fold-mean
(cat at its f1-best epoch, reg at its indist-best epoch), NOT the joint `geom_simple` checkpoint — see the JSON's
`method` field. Confirm the mean matches the paper and **no fold collapsed**: the tell of an fp16/bf16 precision
collapse is a reg best-epoch ≤ ~5 and/or tens of thousands of skipped steps (the board flags these VOID — e.g. the
TX `*_bf16` and CA `*_partial` JSONs; never cite them). Large-state cells are **fp32 where available**; **clean bf16
is accepted as corroboration** (TX's fp32 cell is cross-checked by an H100 bf16 run to 0.03 pp; CA's main cell is
clean bf16 with no fp32 sibling). (4) For a stats claim, re-run the
generator. The cell JSONs themselves are produced by `scripts/closing_data/{a40,h100}_score_matched.py` (the matched
re-scorer; the board also credits `scripts/mtl_improvement/r0_matched_rescore.py`). The two pre-registered tests are
reproducible: `scripts/closing_data/superiority_wilcoxon.py` (per-state Wilcoxon + the state-level sign test) and
`scripts/closing_data/region_match_tost.py` (small-state TOST + power); the leak audit is
`scripts/pre_freeze_gates/a4_{build,eval,cat_eval}.py`; the paper's TOST prose/CSV is [`analysis/tost_region.{md,py}`](analysis/).

## 3 · Decisions ledger (settled; do not silently reopen)

| Decision | Ruling |
|---|---|
| **Abstract TOST wording** | Abstract stays SOFTENED: "matches it (statistically, within two points)" — NO "TOST" acronym in the abstract. The full "statistically non-inferior within a two-point margin (TOST)" appears once in the §1 contribution and in §5.3/§6.2. (GLOSSARY honesty rule, 2026-06-26.) |
| **Verdict verb** | Use **"outperforms"** (paired Wilcoxon superiority) / **"matches"** (TOST non-inferiority); keep each verb bound to its test, and never "outperforms region everywhere". **Updated 2026-06-28 per CC3 (author decision): this supersedes the earlier "keep beats / do NOT swap to outperforms" ruling.** The superiority verb is "outperforms", never "beats" / "wins". |
| **"Dedicated" wording** | Keep "dedicated"; expand to **"dedicated single-task model"** on first use (Table III's column header is literally "Dedicated"). Do not rename to bare "single-task model". |
| **FL region cell** | FL +0.57 stays a **beat** (5/5 folds, Wilcoxon) with no materiality caveat (user decision). |
| **Venue bridge** | Mobility-management is **motivation only**; no measured network result, no prefetch/coverage curve. Right-size examples to tract-level (a census tract is not a radio cell). |
| **Page budget** | **10-page fee variant.** Do not trim to 8. |
| **Dataset years** | Left out of the prose (BRACIS never raised vintage; this is a methods paper; the reference years 2011/2025 are visible). |
| **Region externals** | HMT-GRN = PRIMARY (faithful, board-matched, multi-task). STAN (faithful, from raw) + ReHDM = secondary references. STAN-on-our-representation (`stl_hgi`) is **NOT a baseline** (it sits above us at AL). |

## 4 · Conventions

The writing law is [`GLOSSARY.md`](GLOSSARY.md). In brief: American English; **no em-dash** (commas/parens/semicolons);
tasks are **next-category / next-region** (never activity/area); keep next-category/next-region/next-place distinct;
**no repo codenames** in prose (B9, v11–v16, champion-G, log_T → "region-transition prior", "substrate" → "representation");
"state of the art" not "SOTA"; expand each acronym on first use; plain words for the **networking/systems audience**.

## 5 · What is open / next

1. **P1 — n=20 multi-seed top-up (seeds {1,7,100}, MTL + STL).** The ONE lever that changes a reviewer's verdict
   (breaks the single-seed-n=5 attack, lets per-cell Holm clear 0.05). Blocked on the A40 (fp32 too slow, bf16
   grad-NaN); needs the H100 lane. Full spec in [`IMPROVEMENTS_BACKLOG.md`](IMPROVEMENTS_BACKLOG.md) §P1.
2. **Apply the accepted Germano edits.** [`REVIEW_GERMANO.md`](REVIEW_GERMANO.md) has all 70 comments answered
   (Accept 29 / Partial 29 / Reject 12), each with a concrete "Edit:". Implementing them is an open prose pass.
3. **Reconfirm the deadline.** The notes say ~25 June 2026 (now past); verify the actual cycle on MobiWac/EDAS
   before investing further. If missed, the poster cut ([`archive/PAPER_PLAN_POSTER.md`](archive/PAPER_PLAN_POSTER.md))
   is the fallback.
4. **At submission:** restore `IEEEtran.bst`; final em-dash/codename sweep; de-anonymize check (authors already named).
5. **Deferred (non-blocking, post-deadline coverage adds):** the A4 leak-audit extension to CA/TX/Istanbul (P2; AZ is
   done), ReHDM at CA/TX/Istanbul (P5), and the 3 bridging-metrics re-score cells (P4). None change a verdict — P1 is
   the only one that does.

## 6 · Doc map

**Canonical / active (top level):**
- [`PAPER_PLAN.md`](PAPER_PLAN.md) — the section-by-section spine, claim-discipline whitelist (§3), page budget, open-deps (§10).
- [`GLOSSARY.md`](GLOSSARY.md) — naming/voice law (incl. the region-wording honesty rule).
- [`EDAS_SUBMISSION.md`](EDAS_SUBMISSION.md) — the registered EDAS form (ID, authors, abstract, topics).
- [`REVIEW_GERMANO.md`](REVIEW_GERMANO.md) — co-author review, 70 comments answered (open edit worklist).
- [`IMPROVEMENTS_BACKLOG.md`](IMPROVEMENTS_BACKLOG.md) — ranked forward work; P1 is the only verdict-changing item.
- [`BRIDGING_METRICS.md`](BRIDGING_METRICS.md) — the metrics-ladder + floors supplementary record (backs §6.2).
- [`analysis/`](analysis/) — `tost_region.{md,py}`, the reproducible TOST computation behind §5.3/§6.2.
- [`docs/`](docs/) — the venue dossier (`MOBIWAC_CONFERENCE_GUIDE.md`, `SUBMISSION_CHECKLIST.md`, `BEST_PAPERS_ANALYSIS.md`, `SOURCES.md`, `exemples/`).
- [`src/`](src/) — the paper (`main.tex` + sections/tables/figs + `references.bib` + `README.md`).

**Archived ([`archive/`](archive/)) — work done, kept for provenance; do not treat as live state:**
- `CLOSE_BLOCKERS_HANDOFF.md` — the 3 submission blockers (all closed: Tbl1/Tbl2/FL-CTLE/W6/TOST).
- `STAN_REFOOTING_HANDOFF.md` — faithful-STAN re-footing (done AL/AZ/FL/Istanbul; the audit graduated to `closing_data/FAITHFUL_STAN_FINDINGS.md`).
- `ISTANBUL_BASELINES_HANDOFF.md` — Istanbul baseline run-spec (results folded into `closing_data/ISTANBUL_BASELINES_RESULTS.md`).
- `LEAK_AUDIT_EXTEND_HANDOFF.md` — A4 leak-audit extension recipe (deferred; source of truth `pre_freeze_gates/A4_RESULTS.md`).
- `BASELINE_HANDOFF.md` — the locked baseline plan (D1–D4 decisions, now mirrored in PAPER_PLAN §5.4/§7).
- `BASELINE_AUDIT.md` — the adversarial audit that produced the baseline set (decision provenance).
- `REVIEW_PANEL.md` — the early (2026-06-23) simulated reviewer panel; verdict superseded (we flipped to regular track).
- `PAPER_PLAN_POSTER.md` — the 4-page poster cut (deadline fallback only).

## 7 · The repo docs landscape (for research)

The paper draws on the whole repo, not just this folder. When you need to find something, here is where to look
(all paths from the repo root). The **`closing_data` study is the heart — it is where the paper's data was produced
and recorded.**

- ⭐ **[`docs/studies/closing_data/`](../../docs/studies/closing_data/) — THE most important folder for this paper.**
  The study that produced every Part-2 number. Read in this order:
  - `RESULTS_BOARD.md` — the board (headline §1, file-map §3, baselines §4). The one source of truth for numbers.
  - `STATISTICAL_PROTOCOL.md` — the pre-registered tests (Wilcoxon superiority + TOST non-inferiority + the n=20 plan).
  - `FAITHFUL_STAN_FINDINGS.md`, `CSLSL_CASCADE.md`, `W6_ENCODER_ISOLATION.md`, `ISTANBUL_BASELINES_RESULTS.md` —
    the per-result findings (STAN, cascade tie, trunk-not-transfer probe, Istanbul baselines). The leak-audit
    finding is `../pre_freeze_gates/A4_RESULTS.md` (a sibling study, not under closing_data).
  - `HANDOFF_A40.md` — the live worklist for the remaining GPU runs; `log.md` — the outcomes log (what happened when).
- **[`docs/results/`](../../docs/results/) — the JSON archive** (the actual numbers). For this paper: `closing_data/`
  (MTL/STL + baseline_compare), `P1/` (region ceilings), `P0/simple_baselines/` (floors), `baselines/` (faithful
  baselines), `second_dataset/istanbul/` (Istanbul), `pre_freeze_gates/a4/` (leak audit). Other `results/*` dirs
  (canonical_improvement, mtl_improvement, embedding_eval, …) are earlier studies — historical, not paper-current.
- **[`docs/baselines/`](../../docs/baselines/) — per-baseline narrative docs**: `next_region/{stan,rehdm,hmt_grn,comparison}.md`
  and `next_category/`, each with the method's framing, fairness notes, and a `results/<state>.json` index. Read these
  to understand a baseline's protocol before citing it.
- **[`docs/studies/second_dataset/`](../../docs/studies/second_dataset/)** — the Istanbul/Massive-STEPS study (ETL,
  the 7-category mapping `category_map.md`, dataset stats `STATS_T1.md`).
- **Code that PRODUCES the data:** `scripts/closing_data/` (run + score: `board_h100_*.sh`, `a40_*.sh`,
  `{a40,h100}_score_matched.py`, `build_hgi_overlap_inputs.py`, and the two stats scripts above); `research/baselines/stan/`
  (the faithful-STAN implementation + its README); `scripts/pre_freeze_gates/a4_*.py` (leak audit). The training
  entrypoint is `scripts/train.py`.
  - ⚠ **The paper's recipe is the `check2hgi_dk_ovl` BOARD recipe, NOT the repo-root default.** It is gated
    **stride-1 overlap, MIN_SEQ=10**, model `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (reg) +
    `next_gru` (cat), static_weight cw=0.75, `geom_simple` selector, **fp32**. See the recipe line in `RESULTS_BOARD §1`
    and `RUN_MATRIX.md §0`. The repo-root `/CLAUDE.md` documents the **older non-overlap champion-G** recipe and warns
    those four board values are deliberately NOT global defaults — re-running with the root defaults gives non-board
    numbers (wrong windowing/heads). Always use the board recipe to reproduce a paper cell.
- **Repo-wide guide:** [`/CLAUDE.md`](../../CLAUDE.md) — the project's master guide (architecture, the OOM/precision
  lessons, the canonical-flag traps). Read it before running anything, but use the **board** recipe above for this paper.

## 8 · The paper source (`src/` — LaTeX, in THIS folder)

> ⚠ Two different `src/`: the one in **this folder** (`articles/[mobiwac]/src/`) is the **LaTeX paper**; the repo's
> **`/src`** (root) is the **model/training codebase** (§9). Don't confuse them.

`src/main.tex` is the root (IEEE `conference`, `IEEEtran` class) and `\input`s everything; build with
`pdflatex main → bibtex main → pdflatex main ×2` (see [`src/README.md`](src/README.md)). Structure:
- `sections/01_introduction … 08_conclusion.tex` — the eight section bodies (each header comment says "Plan: PAPER_PLAN §X").
- `tables/tbl1_datasets.tex` (dataset stats), `tbl2_substrate.tex` (Check2HGI vs HGI category), `tbl3_results.tex`
  (the main one-model-two-tasks table). Cells carry mean ± fold-std via the `\sd{}` macro.
- `figs/` — `fig1_dataflow.tex` + `fig2_model.tex` are **TikZ, drawn inline** (no external image); `fig3_embquality`
  and `fig4_deltas` are **`\includegraphics` of PDFs generated by the companion `.py` scripts** (`fig3_embquality.py`,
  `fig4_deltas.py`) — edit the `.py` and re-render to change those figures; the `.pdf` is a required, committed input.
- `references.bib` — 26 cited entries render (of 33 in the file; all 26 verified); `IEEEtran.cls` is bundled for local compile.
- Build artifacts (`*.aux/.log/.bbl/.blg`) are gitignored; `main.pdf` and the figure PDFs are tracked.

## 9 · The model + training codebase (repo `/src` and `research/`)

The actual implementation that produced the results lives at the **repo root** (paths below are from the repo root;
the project guide [`/CLAUDE.md`](../../CLAUDE.md) has the full architecture). What's relevant to this paper:

- **The joint model:** [`src/models/mtlnet.py`](../../src/models/mtlnet.py) — MTLnet. The paper's model is the
  `mtlnet_crossattn_dualtower` variant (two private per-task encoders → a shared bidirectional cross-attention trunk
  → a category head + a dual-tower region head with a private spatial path). Registered in `src/models/registry.py`.
- **The two heads the paper uses:** `src/models/next/next_gru/` (category head) and
  `src/models/next/next_stan_flow_dualtower/` (region head, the dual tower). Other heads under `src/models/next/*`
  and `src/models/category/*` are alternatives explored earlier, not the paper's choice.
- **Training loops (where the cells come from):** [`src/training/runners/mtl_cv.py`](../../src/training/runners/mtl_cv.py)
  (the joint MTL 5-fold loop) + `mtl_eval.py` (eval; the matched scorer reads this), and `category_cv.py` / `next_cv.py`
  (the single-task "ceiling" runs). Entry point: [`scripts/train.py`](../../scripts/train.py) (`--task mtl --engine
  check2hgi_dk_ovl …`, the board recipe in §7).
- **The MTL loss:** `src/losses/` (registry + `_common.py`). The paper uses **static_weight, cw=0.75, unweighted CE**
  on both heads (not a gradient balancer — that finding is in §2 / PAPER_PLAN §2.3).
- **Data → model inputs:** `src/data/folds.py` (user-disjoint `StratifiedGroupKFold`), `src/data/inputs/{core,builders}.py`
  (sequence windows — the board uses **stride-1, MIN_SEQ=10, window-9**), `src/data/dataset.py`. Configs +
  the `EmbeddingEngine` enum (incl. `check2hgi_dk_ovl`) are in `src/configs/` (`paths.py`, `experiment.py`, `globals.py`).
- **The representation (Part 1):** [`research/embeddings/check2hgi/`](../../research/embeddings/check2hgi/) — the
  check-in-level Check2HGI substrate (graph build + infomax training; produces the per-visit vectors).
- **The baselines' code:** [`research/baselines/`](../../research/baselines/) — `stan/` (faithful STAN, with its own
  README), `rehdm/`, `poi_rgnn/`, `mha_pe/`. (HMT-GRN and the cascade live in `scripts/closing_data/` / the study.)
