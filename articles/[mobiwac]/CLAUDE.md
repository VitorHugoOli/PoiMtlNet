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
- **Submission:** EDAS paper **#1571313639**, **regular track**, single-blind. We adopt the **10-page (fee) variant**
  (do NOT trim to 8). Manuscript upload (EDAS Step 3) still pending; see [`EDAS_SUBMISSION.md`](EDAS_SUBMISSION.md).
- **Authors (names stay, single-blind):** Vitor H. O. Silva, Germano B. dos Santos, Fabrício A. Silva — NESPeD-LAB,
  Universidade Federal de Viçosa, Florestal, MG; `{vitor.h.oliveira, germano.santos, fabricio.asilva}@ufv.br`.
- **Bibliography:** 27 cited references, **all web-verified real and current**. `\bibliographystyle{ieeetr}` is a
  LOCAL fallback (IEEEtran.bst is absent on this machine); **restore `IEEEtran.bst` at submission** (Overleaf has it).

## 2 · The science (the thesis + headline; canonical numbers live in the board)

Two parts, one model. **Part 1:** a check-in-level representation makes next-category prediction far more learnable
than a place embedding. **Part 2:** a single joint model predicting next-category and next-region together **beats**
the dedicated single-task category model at **every** state (about **+4.7 to +7.7** macro-F1) and, on next-region,
**beats** at the large region counts (FL/TX/CA) while being **statistically non-inferior within a two-point margin
(TOST)** at the small (AL/AZ/Istanbul). The region gain rises with the number of regions.

- **Scope:** five Gowalla states (AL/AZ/FL/CA/TX) + Istanbul (Massive-STEPS, non-U.S. external check).
- **Significance:** n=5 (seed 0) provisional for Gowalla, n=20 for Istanbul. Per-state Wilcoxon sits at the n=5 floor
  (p=0.031); we report a **state-level** 6/6 sign test (p≈0.016) and TOST for the matches; **no per-cell Holm at n=5**.
- **Leak rebuttal (the BRACIS-reject answer):** the A4 train-users-only transductivity audit is **null on both axes**
  at AL/AZ/FL (region ≤0.33 pp, category ≤0.29 pp), sourced in §5.2.
- **Baselines:** HMT-GRN (primary region-native, 6 states), faithful STAN (AL/AZ/FL/Istanbul; CA/TX infeasible),
  ReHDM (own-protocol reference), POI-RGNN + Markov-9 (category), CTLE + feature-concat (FL representation control),
  CSLSL cascade (a **tie at equal cost**, framed as a defense — never "we beat the cascade").

> ⭐ **Canonical numbers are NOT duplicated here.** The single source of truth for every cell is
> [`docs/studies/closing_data/RESULTS_BOARD.md §1`](../../docs/studies/closing_data/RESULTS_BOARD.md), and the
> paper-facing claim-discipline whitelist (exact CAN-say / PROVISIONAL / must-NOT-say numbers) is
> [`PAPER_PLAN.md §3`](PAPER_PLAN.md). Pull numbers from those, not from memory or from the archived handoffs.

## 3 · Decisions ledger (settled; do not silently reopen)

| Decision | Ruling |
|---|---|
| **Abstract TOST wording** | Abstract stays SOFTENED: "matches it (statistically, within two points)" — NO "TOST" acronym in the abstract. The full "statistically non-inferior within a two-point margin (TOST)" appears once in the §1 contribution and in §5.3/§6.2. (GLOSSARY honesty rule, 2026-06-26.) |
| **Verdict verb** | Use the paper's defined **"beats / matches"** vocabulary (paired Wilcoxon = beat, TOST = match). Do **NOT** swap to "outperforms" (it decouples the verb from the test). |
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
- `LEAK_AUDIT_EXTEND_HANDOFF.md` — A4 leak-audit extension recipe (deferred; source of truth `closing_data/A4_RESULTS.md`).
- `BASELINE_HANDOFF.md` — the locked baseline plan (D1–D4 decisions, now mirrored in PAPER_PLAN §5.4/§7).
- `BASELINE_AUDIT.md` — the adversarial audit that produced the baseline set (decision provenance).
- `REVIEW_PANEL.md` — the early (2026-06-23) simulated reviewer panel; verdict superseded (we flipped to regular track).
- `PAPER_PLAN_POSTER.md` — the 4-page poster cut (deadline fallback only).

**External sources of truth (not in this folder):**
- Numbers/board: [`docs/studies/closing_data/RESULTS_BOARD.md`](../../docs/studies/closing_data/RESULTS_BOARD.md).
- The closing_data study (leak audit, baselines, findings): [`docs/studies/closing_data/`](../../docs/studies/closing_data/).
- The repo-wide project guide: [`/CLAUDE.md`](../../CLAUDE.md).
