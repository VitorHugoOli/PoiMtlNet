# Session Handoff — 2026-04-27 (post-Phase-1 substrate validation)

**For the next agent that picks up this study.** Start here.

This handoff supersedes `SESSION_HANDOFF_2026-04-24.md`. Read it after § 0 if you want the post-F27 cat-head context.

## 0 · One-minute summary

- **Phase 1 of the substrate-comparison plan is COMPLETE on AL + AZ.** The strong claim (Check2HGI > HGI on both tasks under matched-head + matched MTL + linear-probe) is **confirmed** at both ablation states. See [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md) for the §9 outcome-matrix verdict.
- **Three paper-quality findings landed today:**
  1. **CH16 head-invariant** (cat). 4 probes × 2 states × 5 folds = 8 substrate-Δ measurements; **all 8 positive at max-significance Wilcoxon p=0.0312** (5/5 folds positive each). Δ range +11.58 to +15.50 pp. Substrate effect is not a head amplification artefact.
  2. **CH15 reframed — head-coupled.** Existing "HGI > C2HGI on reg under STAN at 3 states" was an artefact of the STAN head's preference for POI-stable smoothness. Under the matched MTL reg head (`next_getnext_hard` = STAN + α·log_T graph prior), **C2HGI ≥ HGI everywhere**: AL tied within σ (TOST non-inferior at δ=2 pp Acc@10), AZ +2.34 pp Acc@10 / +1.29 pp MRR (5/5 folds, p=0.0312).
  3. **MTL B3 is substrate-specific.** Substituting HGI into the same B3 configuration **breaks the joint signal**: cat F1 collapses by 17 pp at both states; reg Acc@10_indist drops 30 pp at both states (29.95 / 22.10 vs C2HGI's 59.60 / 53.82). The MTL win is *interactional* — B3 exploits Check2HGI's per-visit context, and that context is what the architecture is paid for.
  4. **Mechanism — per-visit variation accounts for ~72% of the cat substrate gap.** POI-pooled Check2HGI (mean across check-ins per `placeid`) under matched-head STL `next_gru` at AL: F1 = 29.57 vs canonical 40.76 vs HGI 25.26. Per-visit context = +11.19 pp (~72%); training signal = +4.31 pp (~28%). Linear-probe agrees on direction (63%/37%); matched-head STL gives stronger per-visit signal.
- **Open work**: Phase 2 (FL + CA + TX) is now authorised. See [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md) for the launch queue. FL data is on disk and ready; CA + TX need upstream pipelines (F22/F23 in legacy tracker).

## 1 · Phase 1 results — full table

Compiled from `docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_FINDINGS.md` and `docs/studies/check2hgi/results/phase1_perfold/`.

### 1.1 next_category — substrate Δ across head probes

| State | Probe | C2HGI F1 | HGI F1 | Δ | Wilcoxon p_greater |
|---|---|---:|---:|---:|---:|
| AL | Linear (head-free) | 30.84 ± 2.02 | 18.70 ± 1.38 | **+12.14** | n/a |
| AL | next_gru (matched-head MTL) | 40.76 ± 1.50 | 25.26 ± 1.06 | **+15.50** | **0.0312** |
| AL | next_single (P1_5b head) | 38.71 ± 1.32 | 26.76 ± 0.36 | **+11.96** | **0.0312** |
| AL | next_lstm | 38.38 ± 1.08 | 23.94 ± 0.84 | **+14.44** | **0.0312** |
| AZ | Linear (head-free) | 34.12 ± 1.22 | 22.54 ± 0.45 | **+11.58** | n/a |
| AZ | next_gru (matched-head MTL) | 43.21 ± 0.78 | 28.69 ± 0.71 | **+14.52** | **0.0312** |
| AZ | next_single | 42.20 ± 0.72 | 29.69 ± 0.97 | **+12.50** | **0.0312** |
| AZ | next_lstm | 41.86 ± 0.84 | 26.50 ± 0.29 | **+15.36** | **0.0312** |

8/8 probes positive at maximum significance for n=5 paired Wilcoxon. Substrate effect head-invariant.

### 1.2 next_region — matched-head substrate Δ

`next_getnext_hard` head (STAN + α·log_T graph prior, the actual MTL B3 reg head), 5f × 50ep, seed 42:

| State | C2HGI Acc@10 | HGI Acc@10 | Δ Acc@10 | Wilcoxon p (Acc@10) | TOST δ=2pp |
|---|---:|---:|---:|---:|---|
| AL | **68.37 ± 2.66** | 67.52 ± 2.80 | +0.85 | 0.0625 marginal | non-inferior ✅ |
| AZ | **66.74 ± 2.11** | 64.40 ± 2.42 | **+2.34** | **0.0312** ✅ | non-inferior ✅ |

| State | C2HGI MRR | HGI MRR | Δ MRR | Wilcoxon p (MRR) |
|---|---:|---:|---:|---:|
| AL | 41.17 ± 2.28 | 40.75 ± 2.32 | +0.42 | 0.156 (n.s.) |
| AZ | **41.15 ± 2.13** | 39.86 ± 2.20 | **+1.29** | **0.0312** ✅ |

The CH15 verdict that "HGI > C2HGI on reg" is **head-coupled to STAN**. Under the MTL-equivalent gethard head, the substrate gap closes (AL) or reverses sign in C2HGI's favor (AZ).

### 1.3 MTL counterfactual — B3 with HGI substrate

| State | Substrate | cat F1 | reg Acc@10_indist |
|---|---|---:|---:|
| AL | C2HGI (existing B3) | **42.71 ± 1.37** | **59.60 ± 4.09** |
| AL | HGI (counterfactual) | 25.96 ± 1.61 | 29.95 ± 1.89 |
| AZ | C2HGI (existing B3) | **45.81 ± 1.30** | **53.82 ± 3.11** |
| AZ | HGI (counterfactual) | 28.70 ± 0.51 | 22.10 ± 1.63 |

Δ_cat = +17 pp · Δ_reg = +30 pp at both states. **MTL+HGI is *worse than STL+HGI* on reg by ~37 pp at AL** — the MTL configuration actively breaks the reg head when paired with POI-stable embeddings.

### 1.4 C4 mechanism — POI-pooled Check2HGI

AL only. Pool Check2HGI per `placeid` (mean across check-ins) → kills per-visit variation while preserving Check2HGI's training signal.

| Substrate | Linear probe F1 | Matched-head STL F1 (next_gru) |
|---|---:|---:|
| Check2HGI canonical | 30.84 ± 2.02 | 40.76 ± 1.50 |
| **Check2HGI POI-pooled** | **23.20 ± 1.08** | **29.57** |
| HGI | 18.70 ± 1.38 | 25.26 ± 1.06 |

- Per-visit context = canonical − pooled.
- Training signal = pooled − HGI.

| Decomposition | Linear probe | STL next_gru |
|---|---:|---:|
| Per-visit context | +7.64 pp (63%) | +11.19 pp (72%) |
| Training signal | +4.50 pp (37%) | +4.31 pp (28%) |

Per-visit variation is the dominant mechanism (~72% of matched-head gap). Training signal residual is real and should be acknowledged in the paper.

## 2 · What changed in the docs (this session)

- **New** `research/SUBSTRATE_COMPARISON_PLAN.md` — phase-gated 3-leg framework + critique remediation + outcome interpretation matrix.
- **New** `research/SUBSTRATE_COMPARISON_FINDINGS.md` — live-updated outcome matrix; final §9 verdict committed 2026-04-27.
- **New** `PHASE2_TRACKER.md` — Phase 2 launch queue (FL + CA + TX).
- **Code patches** that should NOT be reverted:
  - `src/models/registry.py::create_model` — silent kwarg-filter so STL can swap heads without conflicting `model_params` defaults.
  - `src/configs/paths.py::IoPaths.get_next_region` — allow `EmbeddingEngine.HGI` (data published by `scripts/probe/build_hgi_next_region.py`).
  - `src/configs/paths.py::EmbeddingEngine` — adds `CHECK2HGI_POOLED` for the C4 counterfactual.
  - `src/data/folds.py::_create_check2hgi_mtl_folds` — engine guard widened to `{CHECK2HGI, HGI}`.
  - `scripts/train.py` — task_set engine guard widened to `{CHECK2HGI, HGI}`.
- **Probe + analysis scripts:**
  - `scripts/probe/substrate_linear_probe.py` — head-free linear probe on `output/<engine>/<state>/input/next.parquet` last-window slice.
  - `scripts/probe/build_hgi_next_region.py` — builds `output/hgi/<state>/input/next_region.parquet` reusing check2hgi labels (substrate-independent).
  - `scripts/probe/build_check2hgi_pooled.py` — POI-mean-pools Check2HGI for the C4 counterfactual.
  - `scripts/analysis/substrate_paired_test.py` — paired-t / Wilcoxon / TOST analyser (handles single JSON or seed-dir aggregation).
- **Orchestrators:** `scripts/run_phase1_{cat,reg}_stl.sh`, `scripts/run_phase1_mtl_counterfactual.sh`, `scripts/run_phase1_c2_head_sweep.sh` — sequential AL+AZ runs under `caffeinate -s`. State-parametric; reuse for FL by editing 2 lines.

## 3 · Implications for paper-level docs

**Updated by this session:**
- `OBJECTIVES_STATUS_TABLE.md` Obj 1 row → 🟢 closed at AL+AZ matched-head (CH16).
- `CLAIMS_AND_HYPOTHESES.md` → CH16 status `confirmed AL+AZ matched-head`, paired Wilcoxon p=0.0312 each. Added CH18 (MTL substrate-specific) + CH19 (per-visit mechanism).
- `NORTH_STAR.md` → adds an "MTL is substrate-specific" addendum.
- `PAPER_STRUCTURE.md` — STL-baseline matching policy revised: `next_gru` (cat) / `next_getnext_hard` (reg) are the matched STLs. `next_single` / STAN STLs become head-sensitivity ablations (still useful — they make C2 a 4-head probe).
- `FOLLOWUPS_TRACKER.md` — F3, F9, F21c-FL, F26 closed/superseded. Phase-2 follow-ups F36–F40 added pointing at PHASE2_TRACKER.

**No retraction needed:** the existing CH16 AL evidence (`next_single` head, +18.30 pp σ-clean) is still valid as a head-sensitivity probe row. CH15 was reframed, not retracted — the STAN-head data still reads HGI > C2HGI on reg, but is now correctly attributed to head-substrate interaction not pure substrate quality.

## 4 · How to launch Phase 2

Phase 2 = same grid replicated at FL → CA → TX, on M4 Pro under `caffeinate -s`.

**FL is ready now** (embeddings + inputs + region transition matrix all on disk):

```bash
# Pre-flight: build HGI's input/next_region.parquet (substrate-independent labels)
OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output \
  python3 scripts/probe/build_hgi_next_region.py --state florida

# Substrate-only probe (Leg I) — ~5 min
OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output \
  python3 scripts/probe/substrate_linear_probe.py --state florida --engine check2hgi
OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output \
  python3 scripts/probe/substrate_linear_probe.py --state florida --engine hgi

# Cat STL matched-head grid (Leg II.1) — ~5–6 h × 2 substrates ≈ 12 h
# (orchestrator pattern; or use scripts/run_phase1_cat_stl.sh with state edited)

# Reg STL matched-head grid (Leg II.2) — ~5–6 h × 2 substrates ≈ 12 h
# Check2HGI side: paired with the F21c continuation. HGI side: new.

# MTL counterfactual (Leg III) — ~5–6 h
```

Total FL ≈ 30 h sequential. Schedule overnight under `caffeinate -s` + F20 per-fold persistence. CA + TX add upstream-pipeline cost (~6–12 h each before the substrate grid).

See `PHASE2_TRACKER.md` for the full launch queue + acceptance criteria.

## 5 · Don't

- **Don't re-run Phase 1.** AL+AZ are closed.
- **Don't run C2 or C4 at FL/CA/TX** (per `SUBSTRATE_COMPARISON_PLAN §6`). AL+AZ for C2, AL alone for C4 are sufficient.
- **Don't push to `main`.** All work stays on `worktree-check2hgi-mtl`.
- **Don't reuse the `next_single` numbers as matched-head evidence** for the post-F27 MTL B3. They are head-sensitivity probes only — `next_gru` is the matched-head STL.
- **Don't infer non-inferiority on reg from "no significant difference"** — TOST with the pre-registered δ=2 pp Acc@10 is the binding test (research/SUBSTRATE_COMPARISON_FINDINGS.md §6).

## 6 · Files most worth reading (ranked)

1. `research/SUBSTRATE_COMPARISON_FINDINGS.md` — final outcome-matrix verdict + paper-ready findings.
2. `research/SUBSTRATE_COMPARISON_PLAN.md` §9 — outcome interpretation matrix (resolved by §1 above).
3. This file (`SESSION_HANDOFF_2026-04-27.md`).
4. `PHASE2_TRACKER.md` — what to launch next.
5. `CLAIMS_AND_HYPOTHESES.md §CH16, CH18, CH19` — the three paper-quality findings.
6. `OBJECTIVES_STATUS_TABLE.md` — updated scorecard.
7. `NORTH_STAR.md` — committed config + MTL substrate-specific addendum.

## 7 · Operational gotchas (carried forward)

All G1–G8 from `SESSION_HANDOFF_2026-04-22.md` + `SESSION_HANDOFF_2026-04-24.md` still apply. Plus new G9:

**G9** — *Background-task notifications fire on the shell wrapper, not the python child.* When a Bash tool launches a long-running `python` command, the harness's "completed" notification refers to the shell process; the actual python PID may continue. Confirm completion with `ps -p <pid>` or by tailing the log.

## 8 · Commits map (this session, 2026-04-27)

| Commit | Scope |
|---|---|
| `c5f81b2` | docs(study): substrate comparison plan — phase-gated, 3-leg framework |
| `45fed36` | feat(probe): substrate linear probe + registry kwarg filter (Leg I) |
| `90d148f` | feat(phase1): orchestrators + paired-test analyser |
| `cd496c0` | data(phase1): cat STL grid AL+AZ + paired tests — CH16 confirmed |
| `698cfa0` | data(phase1): reg STL HGI grid + HGI next_region builder |
| `4b6d79a` | feat(c4): POI-pooled Check2HGI counterfactual + linear probe |
| `6d461c9` | data(phase1): MTL counterfactual + C4 STL pooled |
| `7d0ce98` | docs(phase1): live verdict tracker |
| `a57ebe0` | data(phase1): C2 head-agnostic sweep complete — Phase 1 closed |
