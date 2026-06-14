# baseline_gap — close the external-baseline gaps for the new paper

> **Status:** SCAFFOLDED, not launched (2026-06-14). Machine: **A40 (training) + Mac (ETL/scoring)**.
> Position: **Level 0/1** of [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md) — the *decision* of
> which baselines enter the final tables is a pre-freeze RUN_MATRIX input (feeds `closing_data` P1b);
> the *implementation/runs* are comparison rows that fold into the P3 regeneration.
>
> **Read first:** [`docs/research/baseline_gap_analysis.md`](../../research/baseline_gap_analysis.md)
> (the full present-vs-missing inventory + the minimum-credible-table) and
> [`docs/research/literature_review.md`](../../research/literature_review.md) (what each baseline is).

## Why this study exists

`baseline_gap_analysis.md` found that, today, *every comparison that matters is between the authors' own
configurations*. A reviewer can say so in one line. The repo has strong substrate-axis probes (HGI vs
Check2HGI under matched heads) and faithful end-to-end region baselines (STAN, ReHDM) — but the
**external-validity baselines that calibrate the contribution against the published field are missing or
unowned.** This study owns them, triages which are paper-blocking vs nice-to-have, implements the chosen
set, and feeds their rows/columns into the `closing_data` RUN_MATRIX. It is the corrective for the gap the
pre-freeze scaffolding left open (A2 alone — folded into `pre_freeze_gates` — is not the whole gap).

> **Boundary with neighbors.** `pre_freeze_gates/A2` owns the *feature-concat interpretation control*
> (HGI⊕raw-features — is the cat lift learning or feature injection?). `closing_data/P1b` owns re-running
> the **existing substrate comparators** (DGI/HGI/HMRM/Time2Vec) under the frozen protocol. **This study
> owns the NET-NEW external baselines** the project has never run. No overlap.

## ⚠ Comparability regime (the binding constraint — read before scoping any run)

A baseline is only worth running if it is **comparable to the champion**, which means it must execute on the
**exact frozen base** the champion uses. Several base-level decisions are still open upstream and each one
**pins what the baselines must match**:

| Upstream decision | Owner | What baselines must inherit |
|---|---|---|
| Substrate identity (v14 / a promoted v17) | `mtl_frontier` + `closing_data` P2 | substrate-column baselines (B1/B2) read the **same frozen substrate** |
| **Overlapping vs non-overlapping windows** (stride) | `pre_freeze_gates` (ADOPT/KEEP) | **every** baseline — incl. end-to-end ones that build their own sequences — must use the **same windowing/stride** |
| Fold protocol (user-disjoint StratifiedGroupKFold; seeds {0,1,7,100}; per-fold train-only priors) | frozen at P2 | identical folds, seeds, and leak-free prior construction |
| Region/category label spaces (TIGER tracts; 7-root cat) | frozen at P2 | identical targets |

**Consequences for sequencing:**
- **Substrate-column baselines (B1 CTLE, B2 POI2Vec/skip-gram)** plug into matched heads on the frozen
  substrate, so they inherit windowing/splits automatically — but they still cannot produce *final* numbers
  until the base is frozen. (CTLE additionally pre-trains on **train-portion-only**, its own protocol — that
  is a fidelity requirement, not a regime mismatch.)
- **End-to-end baselines (B3 HMT-GRN-style, B4 cascade, B5 Flashback/DeepMove)** construct their **own**
  sequence inputs from raw check-ins. They MUST be fed the **adopted windowing (overlapping/stride) and the
  same fold/split regime** — otherwise the comparison is invalid (different effective sample counts,
  different leak surface). Their final runs are therefore **BLOCKED on the `pre_freeze_gates`
  overlapping-window ADOPT/KEEP decision and the P2 freeze.**
- **Implication for the timeline:** in this study's Level-0/1 phase you may *implement and smoke-test* each
  baseline (wire it up, confirm it trains) on the *current* base, but the **paper-grade runs wait for the
  frozen base** and re-run if overlapping-windows is adopted. Budget for one re-run of the end-to-end
  baselines after the freeze.

**Scope boundary on the second dataset:** baselines in this study target the **primary (Gowalla) frozen
base**. The `second_dataset` (Massive-STEPS NYC) validation phase is scoped lean (champion G + STL ceilings
+ Markov floor only); running the full external-baseline suite there is **explicitly out of scope** for the
closing paper unless the user pulls a specific baseline into the second-dataset validation. Record that as a
conscious decision, not an omission.

## Scope — triaged from `baseline_gap_analysis.md`, prioritized for THIS paper

### Tier 1 — paper-blocking (a credible substrate/embedding claim needs these)
| ID | Baseline | What it answers | Cost | Pre-freeze? |
|---|---|---|---|---|
| **B1** | **CTLE** (Lin et al., AAAI 2021, [code](https://github.com/Logan-Lin/CTLE)) — contextual per-visit embedding as a substrate column under matched heads | "Why is a hierarchical-infomax substrate better than CTLE's contextual-transformer embeddings?" — the single most-asked review question; today unanswered | Moderate (external codebase; per-state pretrain on train-portion-only per CTLE protocol; 64-d) | DECISION pre-freeze (is it a RUN_MATRIX substrate column?); runs can trail |
| **B2** | **POI2Vec / skip-gram** standalone substrate columns (POI2Vec already in-repo as an HGI input) | completes the canonical location-embedding baseline set (CTLE's own suite) | Low (POI2Vec exists; skip-gram on check-in sequences is cheap) | column decision pre-freeze |

### Tier 2 — expected by reviewers of the MTL story
| ID | Baseline | What it answers | Cost |
|---|---|---|---|
| **B3** | **HMT-GRN-style external MTL baseline** (Lim et al., SIGIR 2022, [code](https://github.com/poi-rec/HMT-GRN)) — shared-LSTM + per-task heads, adapted to the category+region pairing | the MTL table currently has **zero** external MTL rows — every MTL comparison is internal | Medium (adapt the shared-LSTM multi-task design to this data/tasks) |
| **B4** | **Cascade category→region baseline** (CSLSL/CatDM pattern: predict category, condition region on it) | tests the dominant *published* alternative to parallel MTL; cheap given existing heads | Low-Medium |
| **B5** | **Flashback** (sparse-trace regime, AL/AZ) and/or **DeepMove**, adapted to region targets | STAN alone is a thin sequential-baseline set by 2026 standards | Medium |

### Tier 3 — robustness / reviewer-expectation management (run only if cheap or time allows)
TALE / Geo-Teaser / CACSR substrate columns; LBSN2Vec (hypergraph check-in embedding); an LLM zero-shot
reference row (LLM-Mob/AgentMove style, one state); **true PLE** (fix the inter-level gate chain) — *only*
if any PLE claim remains in the paper (current `mtlnet_ple` is a non-canonical CGC-stack — see
`baseline_gap_analysis.md §1.4`).

## Hierarchy / blocking

- **Pre-freeze (Level 0/1):** the **triage decision** — which of B1–B5 become RUN_MATRIX rows/columns —
  must land by `closing_data` P1b so the freeze pins the right comparison set. CTLE-as-substrate-column
  (B1) specifically needs the frozen protocol to be commensurable, so its *decision* is pre-freeze even
  though its *runs* trail.
- **Implementation (parallel):** code/adapt the chosen baselines on the A40 (training) + Mac (ETL/glue)
  concurrently with `mtl_frontier`/`pre_freeze_gates`; nothing here changes the champion recipe or the
  substrate identity, so it does not gate the freeze beyond the RUN_MATRIX decision.
- **Final runs (Level 3):** the chosen baselines run at the full frozen protocol (all states × 4 seeds ×
  5 folds) as comparison rows, folded into the `closing_data` P3 regeneration / M1 baseline board — on the
  **frozen base** (substrate + adopted windowing + folds). End-to-end baselines re-run if overlapping-windows
  is adopted (see the Comparability-regime section).

## Protocol (match the family so rows are comparable)

- Frozen substrate (v14 / blessed base) and matched heads where the baseline is a *substrate column*
  (B1/B2): `next_gru` cat / `next_stan_flow` reg, same folds, per-fold per-seed train-only priors.
- Faithful architecture where the baseline is an *end-to-end system* (B3/B4/B5) — mirror the STAN/ReHDM
  fidelity bar already set in `docs/baselines/` (faithful reimpl, documented deviations).
- **Fairness ledger:** record each baseline's tuning budget vs Check2HGI's (the open
  `baseline_gap_analysis.md §1.4` HGI/Check2HGI tuning-parity caveat applies to all new baselines too).
- Multi-seed {0,1,7,100}; paired Wilcoxon; report n and p. CTLE must pre-train on **train-portion-only**
  (its own protocol) — do NOT give it the transductive full-corpus advantage the substrate has (that
  asymmetry is exactly what `pre_freeze_gates/A4` measures).

## Hand-off

Triage verdict (which baselines in/out + why) → a **RUN_MATRIX row/column block** written into
[`../closing_data/PLAN.md`](../closing_data/PLAN.md) P1b, plus a `docs/baselines/` audit doc per new
baseline (mirroring `POI_RGNN_AUDIT.md`). `STATE.md` + a `docs/studies/log.md` row on each decision/close.
