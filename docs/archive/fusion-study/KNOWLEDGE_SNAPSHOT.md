# Knowledge Snapshot — 2026-04-13

**Purpose:** a single entry-point doc that any future agent (or human) can read to understand *exactly* where the project is and why. It supersedes the older `KNOWLEDGE_BASE_2026-04-13.md` (which described pre-bugfix state) and points to the active plan.

If you read nothing else, read this. Then drop into `docs/studies/fusion/` for the plan.

---

## Project in one paragraph

MTLnet is a multi-task learning framework for Point-of-Interest (POI) intelligence on Gowalla-style LBSN data. Two tasks trained jointly: 7-class POI category classification and 7-class next-POI category prediction. The shared backbone is a stack of residual blocks (with FiLM or expert-gating), with per-task encoders and heads. We target BRACIS 2026 (deadline 2026-04-20) with a paper that corrects a conclusion from our own prior CBIC 2025 submission.

---

## Why the project is paused mid-study

In April 2026 we completed a large 5-stage ablation on the original dataset. In the middle of writing the report, we discovered a **label bug** in the dataset: some POI category labels were wrong. This invalidated:

- All comparisons with published baselines (HAVANA, POI-RGNN, PGC) because their numbers were on the correct labels while ours were on the buggy ones.
- All internal deltas (CBIC vs new config, fusion vs HGI, etc.) because both sides used buggy labels.

The user is regenerating embeddings + fusion inputs on the corrected data. Until that's done, **no scientific conclusions can be drawn** from existing results. All the numbers in `docs/archive/` are snapshots from the bug era — useful for methodology but not for evidence.

---

## Active plan

Everything lives in `docs/studies/fusion/`. Entry points:

- `docs/studies/fusion/README.md` — navigation
- `docs/studies/fusion/QUICK_REFERENCE.md` — TL;DR
- `docs/studies/fusion/MASTER_PLAN.md` — 6-phase strategy
- `docs/studies/fusion/CLAIMS_AND_HYPOTHESES.md` — claim catalog (C01..C21+)
- `docs/studies/fusion/COORDINATOR.md` — orchestrator spec
- `docs/studies/fusion/phases/Pk_*.md` — detailed phase plans

The plan is **claim-driven**: every test exists to validate or refute a specific numbered claim. We do not run experiments "to see what happens."

---

## What we (currently) believe (tier: hypothesis, not yet confirmed on clean data)

These were apparent from the pre-bugfix study and are carried as **hypotheses** to revalidate:

1. **Fusion improves over HGI-only** on category F1 (+30 p.p. on AL in pre-bug runs).
2. **Gradient-surgery optimizers accelerate convergence but may not raise the ceiling** at matched batch size — weakened by pre-bug T0.2 which showed equal_weight ≈ Aligned-MTL at 50 epochs.
3. **Expert-gating architectures (CGC / DSelectK / MMoE) outperform FiLM hard sharing**.
4. **The CBIC 2025 "MTL doesn't help" conclusion was config-specific** — the combination of DGI + FiLM + NashMTL is the culprit, not MTL itself.
5. **Heads co-adapt with the backbone** — standalone head rankings invert in the MTL setting.

**None of these are confirmed on the new data yet.** They are the hypotheses that drive P1-P5.

---

## The critical open question (#1 reviewer risk)

**Does MTL actually beat single-task on the new configuration?**

The prior CBIC paper showed MTL ≈ single-task with DGI+FiLM+NashMTL. We never ran single-task-fusion with the new (DSelectK + Aligned-MTL) config, so we cannot defend "MTL works when configured right" yet. This is claim **C06**, tested in **P2.1**, and is the single most important control in the whole study.

If C06 refutes, the paper pivots from "MTL done right" to "task-specific fusion for POI prediction" — still publishable, but a different narrative.

---

## Timing context

- **BRACIS submission deadline: 2026-04-20**
- **Today: 2026-04-13**
- Feasible minimum for BRACIS: P0 + P1 + P2 + P3 (~25-30 h compute).
- P4 (hparams) and P5 (mechanism figures) slip to journal extension if needed.

---

## Known mechanisms and caveats

- **Scale imbalance in fusion:** Sphere2Vec L2 ≈ 0.55 vs HGI L2 ≈ 8.46 (ratio ~15:1) for category; Time2Vec ≈ 1.0 vs HGI (ratio ~8.7:1) for next. Zero-ablation (pre-bug) showed the model is ~90% HGI-dependent. Per-source normalization hurts (drops accuracy 10 p.p.).
- **Batch-size confound:** CAGrad/Aligned-MTL/PCGrad require `gradient_accumulation_steps=1` (incompatible with accumulation). Pre-bug, this created a 4096 vs 8192 effective-batch asymmetry between optimizer classes. Fixed in `src/ablation/runner.py` to force matched batch for the gradient-surgery losses.
- **Bugs already fixed in-repo** (commits in git log):
  - Gradient-accumulation CLI injection for ca/al/pcgrad
  - `None`-loss fallback in `_get_weighted_loss` (gradient-surgery losses return None as reporting scalar)
  - FLOPs KeyError on Apple Silicon, EmbeddingAligner dedup, generate_sequences return_start_indices
- **Known benign warnings:** `fvcore` missing → FLOPs=0 but training proceeds; MPS fallback for `torch.linalg.eigh` used by Aligned-MTL (CPU fallback, correctness unaffected); `enable_nested_tensor` Transformer warning.

---

## What's in `docs/archive/` (and how to treat it)

Everything under `docs/archive/` is from the **pre-bugfix** era. Useful for:
- Understanding methodology that worked / failed
- Seeing how prior stages were structured
- Recognizing patterns that may or may not replicate post-fix

**Do not cite these numbers in the paper.** They are not evidence.

Files of interest:
- `docs/studies/fusion/archive/full_ablation_study/STUDY_DESIGN.md` — the 5-stage protocol that inspired the current plan
- `docs/studies/fusion/archive/full_ablation_study/runs/STAGE_*.md` — individual stage analyses (numbers invalid, methodology OK)
- `docs/studies/fusion/archive/full_ablation_study/FUSION_RATIONALE.md` — the task-specific fusion rationale (still valid as argument)
- `docs/studies/fusion/archive/ablation_studies/MTL_ABLATION_REPORT_2026-04-11.md` — Phase 1-2 report
- `docs/archive/HGI_HYPERPARAMETER_TUNING_2026-04-13.md` — HGI hparam search (may need redoing)

---

## What's in `docs/issues/` (read before running anything)

- `REPORT_CRITICAL_REVIEW.md` — the review that shaped the current master plan
- `DATA_LEAKAGE_ANALYSIS.md` / `DATE_LEAKAGE_IMPLEMENTATION_GUIDE.md` / `SPLIT_PROTOCOL.md` — data-splitting protocol (user-isolation, stratification)
- `LITERATURE_SURVEY_POI_METRICS.md` / `POI_RELATED_WORK_METRICS.md` — external benchmarks and what they measure

---

## Core numbers to remember (reference for sanity checking)

Pre-bug results (retain as *rough expected ranges*, not evidence):

| Config | State | Cat F1 | Next F1 | Joint |
|--------|-------|--------|---------|-------|
| CBIC (DGI + FiLM + NashMTL) | AL | ~46% | ~27% | ~0.36 |
| CBIC (DGI + FiLM + NashMTL) | FL | ~48% | ~34% | ~0.41 |
| HGI + CGC + equal_weight (pre-fusion best) | AL | ~71% | ~26% | ~0.49 |
| Fusion + DSelectK + Aligned-MTL (pre-bug champion) | AL | ~81% | ~27% | ~0.54 |
| Fusion + DSelectK + Aligned-MTL (pre-bug) | FL | ~78% | ~37% | ~0.58 |

External baselines (Gowalla, state-split, from published papers):

| Model | FL Cat F1 | FL Next F1 |
|-------|-----------|------------|
| HAVANA | 62.9% | — |
| POI-RGNN | — | 34.5% |
| PGC | 50.3% | — |

If post-bugfix numbers deviate by more than 5-10 p.p. from these, something is off. The first P0 sanity check replicates CBIC on new data to verify.

---

## Key commits to know about

- `e19bc9a` — gradient-surgery losses: runner injection + None-loss fallback
- `af8aa64` — master study plan (`docs/studies/fusion/`)
- `76f4009` — reorg: archive old docs, consolidate report

---

## What to do if you're Claude starting fresh here

1. Read `docs/studies/fusion/QUICK_REFERENCE.md`.
2. Check `docs/studies/fusion/state.json` if it exists (tells you current phase).
3. If state.json doesn't exist, we're in P0: waiting on embeddings, building tooling.
4. Ask the user: "are the new embeddings ready?" before running anything scientific.
5. Do NOT use numbers from `docs/archive/` as evidence. They're pre-bugfix.
6. The main paper narrative is the CBIC-correction story. Claim C06 (MTL > single-task) is the linchpin.
7. BRACIS deadline is tight — prioritize P0+P1+P2+P3 over P4+P5.

---

## What to do if you're a human collaborator

1. Skim `docs/studies/fusion/QUICK_REFERENCE.md` — 1 page.
2. Open `docs/studies/fusion/CLAIMS_AND_HYPOTHESES.md` to see the 21 claims we're validating.
3. If you want to run experiments on your machine: see `docs/studies/fusion/phases/Pk_*.md` for the command schemas.
4. Copy result directories to this machine and let the coordinator archive them via `/study import` (once the Skill is built in P0).

---

## Living doc

This file is a **snapshot** (dated in the title). Refresh it whenever the project state changes materially (phase transitions, major findings, scope changes). Keep the date in the filename or at the top so future readers can trust what they're seeing is current.
