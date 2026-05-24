# Agent Onboarding Prompt — MTL Improvement Track

> **Paste this prompt (or its core block) at the start of every fresh Claude Code session that picks up this research track.** It onboards you with everything you need before touching code.

---

## Your role

You are the **implementing agent** for the MTL Improvement research track. A prior agent (design phase) has already:

- Read every preliminary file in `docs/studies/mtl-exploration/`.
- Read `canonical_improvement/` as the structural template.
- Run a three-dimensional audit of the user's 10 considerations (conceptual / technical-feasibility / metrics+baseline-robustness).
- Run 7 parallel breadth-search sub-agents (heads / backbone / loss / optimization / data-sampling / input-modality / instrumentation) and synthesized ~288 candidate directions into a final 32-experiment slate.
- Captured user alignment (8 archs faithful — T2a cheap + T2b heavy; full F49 audit; new folder + branch; HGI re-check per T2 winner; scale-conditional ship recipe preserved).
- Run a pre-write advisor pass and incorporated 5 substantive corrections (Tier 2 LR mini-sweep, per-tier decision gates, C2 right-sized, C4 row-pairing flagged, T0.7 same-machine sanity check).
- Designed the 8-tier chain in `INDEX.html` with explicit chain-break mitigations.

**You execute the experiments.** You are NOT locked into the design — if results redirect you, propose and pursue the new path. Document the redirection in `log.md` first.

## Required reading (in this order, before any code change)

| # | File | Purpose |
|---|------|---------|
| 1 | `docs/studies/mtl_improvement/log.md` | Most recent progress and decisions |
| 2 | `docs/studies/mtl_improvement/INDEX.html` | Full 8-tier design. **Read §Execution guidelines + §Falsified history + §Audit + §Chain & sequencing + §Evaluation framework cover-to-cover before any experiment.** |
| 3 | `docs/studies/mtl-exploration/INDEX.html` | State-of-the-art MTL audit; current state of B9 + every prior ablation. The grounding doc. |
| 4 | `docs/studies/mtl-exploration/EXPERIMENT_NO_ENCODERS.md` | Encoder ablation findings (Linear+LN matches 2-MLP at AZ; load-bearing for cat at AL). |
| 5 | `docs/studies/mtl-exploration/EXPERIMENT_HGI_SUBSTRATE.md` | HGI substrate finding (MTL+HGI ≡ STL+HGI under cross-attn). |
| 6 | `docs/studies/mtl-exploration/LEAK_BLAST_RADIUS_AUDIT.md` | The n_splits guard history; per-fold log_T discipline. |
| 7 | `docs/NORTH_STAR.md` | Canonical B9 invocation. |
| 8 | `docs/results/RESULTS_TABLE.md §0` | Paper-canonical v11 numbers. The shipping target. |
| 9 | `docs/findings/MTL_FLAWS_AND_FIXES.md` | MTL flaws catalogue; sections §1 (how MTL is wired) and §2 (empirical flaws). |
| 10 | `docs/MTL_ARCHITECTURE_JOURNEY.md` | F-trail narrative. Helpful but not load-bearing — INDEX.html distills what matters. |

## Hard rules (do not break)

1. **No substrate swap.** Substrate is fixed to Check2HGI. HGI re-checks are per-T2-winner sanity probes only (single seed, AL+AZ, 5f × 25ep). Do NOT use HGI as the primary substrate.

2. **No `--folds 1` with per-fold log_T.** The 2026-05-15 trainer guard hard-fails. Use `--folds 5` to match `n_splits=5` of the log_T. Anything that mutates fold composition, sequence-stride, or min-history triggers a **log_T rebuild** via `scripts/compute_region_transition.py --per-fold --seed S --n-splits 5` — document the rebuild in `log.md`.

3. **Stay at `shared_layer_size = 256`.** F51 capacity-widening is FALSIFIED (cat collapse at FL above 256). Capacity sweeps are depth/head-count only at matched D=256.

4. **No fclass-as-feature anywhere** — fclass is the linear-probe label; tautological leak.

5. **Per-arch light LR mini-sweep is mandatory** for every backbone candidate in T2a/T2b. Three regimes: constant 1e-3 everywhere / B9 per-head / arch-suggested-default. 5f × 30ep × seed=42 at AL+AZ for each regime. Then full-protocol run at the winning regime.

6. **Statistical claims need n ≥ 10.** At n=5 the Wilcoxon floor is p=0.0312. Any "wins paper-canon" claim requires multi-seed (4 seeds × 5 folds = n=20 pooled). For cheap iteration in T2-T7, single-seed n=5 paired is acceptable to gate-keep; final shipping (T8) requires multi-seed.

7. **Pre-flight T0.2 (mask audit) must pass on canonical first** — blocks Tier 1 launch.

8. **Pre-flight T0.7 (same-machine B9 re-baseline) must pass** — pins `B9_v11_repro` as the chain's tier-0 baseline.

9. **Unit-test gate before any T2a/T2b/T7 multi-fold launch.** Every architectural variant requires a test under `tests/test_models/test_mtl_improve_variants.py` confirming: (a) forward+backward shape correctness on a synthetic 100-user batch, (b) one-step loss is finite (no NaN), (c) parameter count is within ~5% of B9 at D=256, (d) `shared_parameters()` / `cat_specific_parameters()` / `reg_specific_parameters()` partition is bijective + exhaustive.

10. **Falsified history is off-limits.** See `INDEX.html §Falsified history`. If you believe you have a genuinely different formulation, document the structural distinction in `log.md` before launching.

11. **Respect the chain.** Each tier's winning recipe becomes the next tier's fixed baseline. If you ever want to break the chain (run an out-of-order experiment), document the rationale in `log.md` first AND assess re-execution risk explicitly.

## Required workflow

1. **Create the dedicated git worktree first**. Branch `mtl-improve` from `main`. Do not contaminate other ongoing work.

   ```bash
   git fetch origin
   git worktree add ../worktree-mtl-improve -b mtl-improve main
   cd ../worktree-mtl-improve
   # Symlink data + output + results from main repo root if needed:
   ln -s /path/to/main-repo/data data
   ln -s /path/to/main-repo/output output
   ln -s /path/to/main-repo/results results
   ```

2. **Use `TaskCreate` / `TaskUpdate` to break down every experiment** into sub-tasks: `unit-test → validate → launch → import → analyze`. Mark `in_progress` before starting, `completed` only when:
   - Results are filled in the experiment's `<div class="results-placeholder">` in `INDEX.html`.
   - A `log.md` entry is written.
   Never leave a task `in_progress` overnight without an explicit blocker note.

3. **Use the `/goal` slash command for autonomous experiment runs.** Each experiment's `Methodology` block in `INDEX.html` is a complete goal description. Queue tier-by-tier; steer when results come in.

4. **After each tier completes, call advisor** with that tier's results. Capture advisor feedback in `log.md`. Apply revisions before moving to the next tier.

5. **After the full track completes (or at any major redirect), spawn a mandatory advisor sub-agent** to evaluate the audit, the experiment results, the integration appendix, the documents. Capture feedback in `log.md`.

6. **Keep `log.md` current.** Append every decision, blocker, falsified hypothesis, redirection, finding. Date every entry (absolute, e.g. `2026-05-16`, never `today`).

## Execution order

1. **Tier 0 first.** Launch in parallel BUT respect the gates:
   - **T0.2 (mask audit) gates T1.** Must pass before any T1 launch.
   - **T0.7 (same-machine re-baseline) gates T1.** Must pass before any T1 launch.
   - T0.1 (docs cleanup), T0.3 (F49 audit), T0.4 (flat-checkin cat), T0.5 (instrumentation wave), T0.6 (n_splits guard ext) run in parallel.

2. **Tier 1 STL ceilings refresh** — pin `STL_v2` per state. Multi-seed-light (3 seeds) for n=15.

3. **Tier 2a (cheap backbones)** — 4 archs with per-arch LR mini-sweep. **STOP GATE**: if any T2a candidate beats B9 paper-grade on reg at ≥2 of {AL, AZ, FL} with cat non-inferior, T2b becomes OPTIONAL. Escalate to user via `AskUserQuestion` for skip-or-run decision.

4. **Tier 2b (heavy backbones)** — 4 archs, same protocol. Skipped if T2a winner cleared the gate.

5. **Tier 2 final decision** — pin the backbone winner. HGI sanity probe (1 seed × AL+AZ × 5f × 25ep) per winning arch.

6. **Tier 3 loss balancing** — under new arch. Includes T3 winner re-validation in T5 in case the optimizer change shifts the loss landscape.

7. **Tier 4 batch / data sampling** — under arch+loss fixed. Modest expected effect; closes the audit.

8. **Tier 5 LR / optimizer refinement** — re-sweep on the consolidated stack. Validates T3 winner; narrow-scope T3 re-run if invalidated.

9. **Tier 6 α formula deep-dive** — under fully consolidated recipe.

10. **Tier 7 final MTL head re-ablation** — chicken/egg resolution. If new head winner emerges, narrow-scope T5 LR re-fit on the new head only.

11. **Tier 8 multi-seed shipping run** — 4 seeds × 5 folds × 5 states; n=20 pooled paired Wilcoxon vs Table A (STL) AND Table B (B9 paper-canonical).

12. **Final advisor pass.** Spawn an advisor sub-agent to evaluate the entire chain. Capture in `log.md`. Write `PAPER_UPDATE.md` documenting which v11 RESULTS_TABLE rows change.

## What you compare against

| Stage | Baseline | Protocol |
|---|---|---|
| Tier 0 | n/a (no model changes) | n/a |
| Tier 1 STL | matched-head STL at seed=42 gate-keep → 3 seeds (n=15) ship | leak-free per-fold log_T |
| Tier 2–7 iteration | current chain baseline (after each tier's decision gate) | seed=42 n=5 gate-keep |
| Tier 8 shipping | MTL B9 Table B + STL Table A | 4 seeds × 5 folds pooled (n=20) |
| HGI sanity probe per T2 winner | STL+HGI ceiling | single seed AL+AZ 5f × 25ep |

## How to format a Results section update in INDEX.html

When an experiment completes, find its `<div class="results-placeholder">Results: <em>pending execution</em></div>` block and replace it with a real results block. Template:

```html
<div class="block">
<div class="block-title">Results (YYYY-MM-DD)</div>
<table>
<thead><tr><th>State</th><th>cat F1</th><th>reg Acc@10</th><th>Δ vs baseline</th><th>Wilcoxon p</th><th>leak-probe</th></tr></thead>
<tbody>
<tr><td>AL</td><td class="num">X.XX ± X.XX</td><td class="num">X.XX ± X.XX</td><td class="num">+X.XX cat / +X.XX reg</td><td class="num">cat p=X / reg p=X</td><td class="num">XX.X %</td></tr>
</tbody>
</table>
<p><strong>Verdict:</strong> <span class="pill good">PROMOTE</span> | <span class="pill warn">non-inferior</span> | <span class="pill bad">FALSIFIED</span> — one-sentence reason.</p>
<p class="footnote">JSON: <code>docs/results/mtl_improvement/T?-?_..._5f50ep.json</code> · Wilcoxon: <code>docs/results/paired_tests/mtl_improvement/T?-?_paired.json</code> · log_T rebuild: yes/no · compute: X GPU-h.</p>
</div>
```

Then update `log.md` with the dated findings block and `TaskUpdate` the corresponding task to `completed`.

## Result file conventions

- Raw per-fold metrics → `docs/results/mtl_improvement/T{tier}-{id}_{state}_{seed}_{folds}f{epochs}ep.json`
- Paired-test JSONs → `docs/results/paired_tests/mtl_improvement/T{tier}-{id}_paired.json`
- Plots → `docs/studies/mtl_improvement/figs/T{tier}-{id}_*.png`
- Trained model artifacts → `results/mtl_improvement_T{tier}-{id}/{state}/`
- Diagnostic dumps (gradient cosine, α trajectory, F50-D5 saturation, confusion matrices, ECE) → sit alongside the run's results dir as `diagnostics/`

## When you're done with this session

If you finish a tier, mark its tasks completed, update `log.md`, and either:
- Continue to the next tier in the same session if compute allows, OR
- Hand off cleanly: ensure `log.md` ends with a `**Next**:` bullet specifying the exact next experiment ID and any pending decisions.

If you're stopping mid-experiment: ensure the task is still `in_progress` (NOT `completed`), add a `**Blocker**:` line in `log.md` describing exactly what's incomplete and what the next agent needs to resume.

---

## Quick-reference: cited file paths

```
docs/studies/mtl_improvement/
  INDEX.html              # design + index + future results
  log.md                  # progress log (append-only)
  AGENT_PROMPT.md         # this file

docs/studies/mtl-exploration/
  INDEX.html              # state-of-the-art MTL audit
  EXPERIMENT_NO_ENCODERS.md
  EXPERIMENT_HGI_SUBSTRATE.md
  LEAK_BLAST_RADIUS_AUDIT.md
  considerations.md       # user's original 10 considerations (audited in INDEX.html)

docs/results/
  RESULTS_TABLE.md         # v11 paper-canonical multi-seed (the shipping target)
  paired_tests/            # leak-free paired Wilcoxon JSONs

docs/findings/
  MTL_FLAWS_AND_FIXES.md   # MTL flaws catalogue
  F49_LAMBDA0_DECOMPOSITION_GAP.md  # F49 audit reference for T0.3

docs/
  NORTH_STAR.md            # B9 invocation
  MTL_ARCHITECTURE_JOURNEY.md   # F-trail narrative

src/models/mtl/
  mtlnet_crossattn/model.py     # current backbone (B9)
  mtlnet_crossstitch/model.py   # scaffolded for T2a.1

src/models/next/
  next_gru/head.py              # current cat head
  next_stan_flow/head.py        # current reg head (α · log_T)

src/training/
  helpers.py                    # setup_per_head_optimizer
  runners/mtl_cv.py             # trainer + n_splits guard (2026-05-15)

src/data/
  folds.py                      # _create_check2hgi_mtl_folds (row-pairing constraint)
  inputs/core.py                # generate_sequences (window=9, stride=9)
  aux_side_channel.py           # last_region_idx publisher

src/tasks/
  presets.py                    # CHECK2HGI_NEXT_REGION

scripts/
  train.py                      # canonical CLI entrypoint
  compute_region_transition.py  # per-fold log_T builder (--per-fold --n-splits 5)
  p1_region_head_ablation.py    # downstream STL eval template for T1.2
  probe/leak_sniff_ijm.py       # leak-audit reference for T0.5
  probe/generality_probes.py    # fclass + kNN-Jaccard probes (T0.5)
```

---

**Final reminder:** the design is a strong starting point, not a cage. If T2a winner already closes the FL/CA/TX reg gap, T2b is optional. If T0.2 finds a real bug, escalate. If results suggest a new path, propose and pursue it — but any deviation from the chain must be deliberate, documented in `log.md`, and evaluated for sequencing impact.
