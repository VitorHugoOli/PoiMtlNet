# Agent Onboarding Prompt — MTL Improvement Track (v2, v14-rebased)

> **Paste this prompt (or its core block) at the start of every fresh Claude Code session that picks up this research track.** It onboards you with everything you need before touching code.

> ⚠ **2026-06-02 — this track was REBASED.** The base substrate is now **v14** (`check2hgi_design_k_resln_mae_l0_1`), compared vs canonical paper **v11**. The binding constraint is **the regime finding** (the cross-attn MTL regime washes out substrate gains; the gap is architectural — P4). Read `INDEX.html` §"What changed (v1→v2)" + §"The regime finding" + §"Metrics & comparand framework" BEFORE anything. The 6-tier v2 chain supersedes the old 8-tier v1 chain.

> ⚡ **2026-06-03 — execution started; read these first.** A `git pull` brought landed results + new defaults — read `INDEX.html` §"What already landed", §"State strategy", §"Execution & parallelization":
> - **The regime gate is CONFIRMED** (`docs/results/v14_mtl_vs_canonical.md`): v14 ≈ matched canonical in MTL, 5f × {0,1,7,100} at FL+AL+AZ → T0.2/T0.3 largely DONE; the architectural lever (Tier 2) is the path. STL ceilings replicate exactly.
> - **Selector default flipped (C21):** `geom_simple = sqrt(cat_F1·reg_Acc@10)` is now the default MTL joint selector. §0.1 is diagnostic-best (selector-independent); reproduce v11 joint-selected via `--checkpoint-selector joint_f1_mean`.
> - **States:** AL/AZ/**GE**/FL at **5-fold** are the main evidence (GE = the middle-scale bridge, NOT yet built — Tier-0 prerequisite T0.1b); CA/TX at **1-fold** optional.
> - **Hardware:** i9-14900K (32 threads) + A40 45GB. **Parallelize via CUDA MPS** (small-state runs underutilize the GPU); builds are the serial spine. See §Execution.
> - **New loss lever:** loss-scale normalization (~4.7× CE magnitude gap) is now Tier-4 T4.0 (ungated, highest-EV); FAMO is the O(1) balancer (PCGrad/Nash O(k)-infeasible at 9k).

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
| 1 | `docs/studies/mtl_improvement/log.md` | Most recent progress — read the **2026-06-02 reframe entry** first |
| 2 | `docs/studies/mtl_improvement/INDEX.html` | Full v2 6-tier design. **Read §What changed + §The regime finding + §Metrics & comparand framework + §Execution guidelines + §Falsified history + §Chain cover-to-cover before any experiment.** |
| 3 | `docs/results/CANONICAL_VERSIONS.md` §v11–v14 | What v14 IS, how to build it, the v11/v12/v13/v14 reproduction map. **Load-bearing.** |
| 4 | `docs/studies/embedding_eval/FINAL_SYNTHESIS.md` | The v14 verdict + **the regime finding** (substrate washes out in MTL). |
| 5 | `docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md` | **P4** (gap is architectural, not interference) + three-frontier + selector axis + residual-gap table. |
| 6 | `docs/findings/B9_STL_STAN_SWAP_AZ_FL.md` | The residual-skip FALSIFICATION (§6.2) + the §6.4 gap decomposition (~75% is the missing private backbone). The motivation for T2.1. |
| 7 | `docs/future_works/` (part2_routing, mtl_architecture_revisit, composite_two_substrate, substrate_adaptive_balancing, reg_head_sweep, head_window_batch, paper_canon_reevaluation) | The folded-in memos; each maps to a tier (see INDEX §Future-works folded in). |
| 8 | `docs/NORTH_STAR.md` + `docs/results/RESULTS_TABLE.md §0` | Canonical B9/v11 invocation + the paper anchor (b). |
| 9 | `docs/studies/mtl-exploration/EXPERIMENT_NO_ENCODERS.md` | Encoder ablation (d_model=256 load-bearing, MLP depth flat) — closes the encoder question. |
| 10 | `docs/findings/MTL_FLAWS_AND_FIXES.md` | How MTL is wired + empirical flaws catalogue. |

## Hard rules (do not break)

1. **Base substrate = v14** (`check2hgi_design_k_resln_mae_l0_1`). Build it per state (T0.1) — it exists only at FL today. Canonical-fresh (`gcn_ctrl`) is the matched control. HGI is a per-T2-winner sanity probe only (2 seeds × AL+AZ × 5f × 30ep; escalate if |MTL+HGI − STL+HGI| ≥ 2pp). **No substrate re-builds for the reg gap** — Part-1 is closed.

2. **Per-task disjoint + joint reporting is mandatory** for every MTL run — report BOTH tasks at TWO epochs: per-task disjoint (`cat@best-cat-epoch`, `reg@best-reg-epoch`) AND joint (`cat@joint`, `reg@joint` at the single `geom_simple` deployable checkpoint). Every MTL row is a 2×2 {cat,reg}×{disjoint,joint} + STL ceiling per task + composite reg ceiling. Never quote the b9 production selector alone (under-reports reg +7-11pp at large states). Disjoint is a capacity frontier (reporting-only — two-checkpoint deploy is OUT of scope, user-dropped).

2b. **Frozen-fold paired design** for infra-isolation experiments (Tier 2 architecture, T1.3 encoder probe, T5 head swaps): all arms use the identical `StratifiedGroupKFold(groups=userid, seed=S)` partition + matching seeded log_T, compared fold-by-fold so fold-composition variance cancels in the Δ. Controls data-difficulty variance (the dominant small-state noise); does NOT control init/batch-order (vary seed but keep the split aligned per seed). Report Δ vs the frozen-fold baseline, not bare absolutes (F51 fold-1-easiness). Do NOT freeze for the T6 ship run or T0.3 regime gate (those vary folds for generalization).

3. **Fresh-vs-frozen discipline.** Frozen v11 is a privileged draw (~2.6pp cat / ~0.5pp reg over fresh). For regime/substrate/architecture claims compare **v14-fresh vs canonical-fresh (gcn_ctrl)**. Frozen v11 §0.1 is for paper-continuity only, always labeled. Keep separate columns.

4. **log_T-KD ON** (v12 default W=0.2 τ=1.0). Per-fold **seeded** log_T mandatory (`--per-fold-transition-dir` with `region_transition_log_seed{S}_fold{N}.pt`); default log_T leaks ~+3pp. `--folds 1` + per-fold log_T hard-fails (n_splits guard). Any fold/stride/min-history change → rebuild log_T via `scripts/compute_region_transition.py --per-fold --seed S --n-splits 5`.

5. **Stay at `shared_layer_size = 256`** — F51 capacity-widening FALSIFIED. Sweeps are depth/head-count only.

6. **No fclass-as-feature** — tautological probe leak.

7. **Per-arch LR mini-sweep mandatory** for every Tier-2 architecture: 5 regimes (constant 1e-3 / B9 per-head / arch-default / **per-arch-group LR for new params** / B9+warmup-5%) × 5f × 40ep × seed=42 × AL+AZ, then full-protocol at the winning regime. The B9_STL_STAN_SWAP v1 collapse (B9 recipe + non-α head) is why.

8. **Statistical claims need n ≥ 10.** n=5 gate-keeps (Wilcoxon floor 0.0312); n=20 ships. Minimum-effect floor ≥1pp (n=5) / ≥0.5pp (n=20) on the targeted axis; TOST δ=2pp non-inferiority on the non-targeted axis. Multi-state promotion = win at ≥2 of {AL,AZ,FL}.

9. **Pre-flights gate Tier 1:** T0.1 (v14 built + STL-validated per state), T0.3 (regime entry gate at multi-seed). Window/mask audit is DONE (`substrate-protocol-cleanup/window_mask_audit.md`) — cite, don't re-run.

10. **Unit-test gate before any new arch/head multi-fold launch** — forward/backward shapes (synthetic 100-user batch), loss-finite, param count within ~5% of B9 at D=256, `shared/cat_specific/reg_specific_parameters()` partition bijective+exhaustive (the dual-tower's private backbone is a NEW param group — wire it into the partition).

11. **Falsified history off-limits** (INDEX §Falsified history): thin residual-skip, encoder-MLP-depth, class-balanced sampler, loss-balancing-for-interference, substrate/routing-in-MTL, MMoE/CGC (lose reg), PLE (collapse). Genuinely-different formulation → document the distinction in `log.md` first.

12. **Respect the chain.** Each tier's winner is the next tier's baseline. The load-bearing **T2.1 runs on BOTH v14 AND canonical-fresh** (regime×substrate 2×2). Out-of-order = document rationale + re-execution risk in `log.md` first.

13. **State strategy: AL/AZ/GE/FL at 5-fold are the main evidence; CA/TX at 1-fold optional.** GE (Georgia, the small↔huge middle bridge, ~3-4k regions) is user-flagged essential and **not yet built** — onboard it in T0.1b (raw data → v14+canonical substrates → frozen folds → seeded log_T) before the Tier-2 generality claim. Generality promotions need the band: ≥2 of {AL,AZ} (small) AND GE (middle) AND FL (large). CA/TX 1-fold numbers are directional (no paired Wilcoxon) — flag as such; run only for a shipping candidate (T6) or a band-ambiguous result.

14. **Parallelize smartly on the i9-14900K + A40 45GB.** Drivers (`scripts/_v14_run/`) are currently SERIAL — that's the headroom. (a) Collocate small-state MTL runs (AL/AZ/GE, light, ~2-3 min) via **CUDA MPS** — they underutilize the A40; cap concurrency at `min(⌊0.85×45GB / per-run-peak-VRAM⌋, ~4)`, measure per-run VRAM once. (b) FL (heavier, ~14 min) collocate ≤2. (c) ONE GPU build at a time (builds saturate). (d) CPU-parallel the prep (log_T, postbuild, aggregation) on the 32 threads, overlapped with GPU work; pin `OMP_NUM_THREADS = 32/N_concurrent`. (e) each process writes its own run dir + reads its own seeded log_T (race-safe); stage log_T BEFORE concurrent reads. Don't fill 45GB (OOM/fragmentation headroom). (f) **Tier S (STL-only) is the prime MPS-collocation filler** — run it concurrently with the Tier 2-4 MTL runs.

15. **Frozen ceiling after T1.4 — two senses.** (i) The **track-internal yardstick** (the (c)/(d) values T2-T5 Δ are measured against) is **immutable** once T1.4 closes — Tier S / T5 STL wins are T5 candidates + future-work notes, never a retroactive change or a re-score of completed T2+ deltas (the moving-baseline trap). (ii) The **paper-reported §0.1 STL baseline is refreshable at T6.2** — if Tier S / T5 yields a validated better STL head, re-run the §0.1 STL columns under it (don't publish an STL baseline you can beat). Keep the senses separate. All STL-improvement work is **ceiling/baseline integrity, NOT MTL transfer** (the regime finding predicts STL gains wash out jointly — do not re-litigate this).

16. **No next-POI model imports.** Do NOT plumb GETNext/STHGCN/Graph-Flashback/MobGT/Diff-POI in as heads — their headline contribution is the α·log_T transition prior we already own, and they require raw GPS/time/category/POI-level side-info our 9×D=256 check-in-embedding input lacks (STHGCN is also too heavy for the sweep grid). Only encoder *cores* transplant (self-attention, Mamba block, attention pooling); SASRec (ICDM'18) is the named transformer baseline. The 9k-class softmax is NOT a blocker for an STL head.

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

4. **Per-tier review cadence — MANDATORY at every tier boundary (do all three, in order):**
   1. **Advisor pass on the tier's results** — spawn an advisor / review sub-agent to evaluate the tier's outcomes (are the numbers sound? does the verdict follow? any leak/confound/over-read?). Capture the feedback in `log.md`. Apply revisions before proceeding.
   2. **Write a tier SUMMARY** — a short dated block (in `log.md` + the tier's INDEX section): what ran, the per-state per-task 2×2 numbers vs the anchors, the verdict per experiment, what was promoted/falsified, and the proposed next step.
   3. **STOP and surface the summary to the user for discussion** (`AskUserQuestion` or a plain checkpoint message). **Do not auto-roll into the next tier** — the user and the agent decide together how to proceed. The chain is sequential precisely so each tier is a deliberate decision point, not an autopilot step.

5. **After the FULL track completes — mandatory implementation-correctness review.** Before declaring done, spawn an **advisor / code-review sub-agent specifically to verify the implementation is correct for THIS case** — not just that the numbers look good: are the new modules (dual-tower, loss-scale-norm, etc.) doing what the design intends? Are the param partitions / log_T / selector / frozen-fold guards all wired right? Is anything silently mis-measured (a repeat of the F49 leak / stale-log_T / wrong-selector class of bug)? Then a second advisor pass on the whole track (audit + results + sequencing + claims). Capture both in `log.md`; write `PAPER_UPDATE.md`.

6. **Keep `log.md` AND `INDEX.html` current — continuously, not at the end.** This is load-bearing, not bookkeeping: the next agent (or you, in a month) reconstructs the entire state from these two files.
   - **`log.md`** — append every decision, blocker, falsified hypothesis, redirection, finding, and tier summary. Date every entry absolute (e.g. `2026-06-03`, never `today`). Update it *as decisions are made*, not retroactively.
   - **`INDEX.html`** — fill each experiment's `<div class="results-placeholder">` with the real Results block (per-state 2×2 + Δ vs anchors + verdict) the moment that experiment closes; record any design deviation (new experiment, dropped arm, re-scope) inline in the same session it happens.
   - A task is **not** `completed` until BOTH the INDEX Results block and the `log.md` entry are written (see item 2).

7. **Commit constantly — small, frequent, auditable commits.** Do NOT batch a whole tier into one commit. Commit at every natural unit: each experiment closing (code/results + its INDEX Results block + its `log.md` entry, together), each decision/redirection, each tier summary, each substrate build. Each commit should leave the repo in a coherent, reproducible state — so the history reads as the lab notebook and any point is recoverable.
   - **Work on the dedicated branch / worktree** (`mtl-improve`); never commit experiment work straight to `main` — open a PR and merge deliberately.
   - **Stage explicitly with a pathspec and check `git status` before every commit.** This repo frequently has unrelated pre-staged files (e.g. `articles/*`); a bare `git add -A` / `git commit` will sweep them in. Add only the files your change touches; verify `git show --stat HEAD` after.
   - **Message convention:** `mtl_improvement: <what> — <one-line why/result>`; end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
   - Push regularly so progress is durable off-box (the runs are on a remote A40 — don't let days of results live only in the working tree).

## Execution order (v2 — 6 tiers)

1. **Tier 0 first (foundations).** Order: **T0.0 (freeze the fold partition + seeded log_T — the immutable shared artifact) FIRST**, then everything else reuses it.
   - **T0.1 (build v14 + STL-validate per state, on the frozen folds) gates T1** — v14 exists only at FL; build AL+AZ, validate the dual-axis gain reproduces (state-dependent; has a fallback).
   - **T0.2 (frozen-fold reference board + equivalence) is THE calibration gate** — run all anchors (MTL v11, MTL v14, MTL canonical-fresh, STL-on-v14, composite) on the frozen folds and **prove MTL v11 ≈ paper §0.1 + three-frontier ≈ mtl-protocol-fix + v14 STL ≈ embedding_eval BEFORE any improvement work**. Divergence beyond fold-σ + the known fresh-vs-frozen offset → STOP and diagnose. This catches a harness discrepancy at Tier 0, not Tier 6.
   - **T0.3 (regime entry gate)** reads the board's v14-MTL vs canonical-fresh-MTL cells at n=20 — confirm v14≈canonical before committing to "substrate exhausted."
   - T0.4 (three-frontier + selector; cite window/mask audit), T0.5 (instrumentation) run in parallel.

2. **Tier 1 — consolidate ceilings + encoder probe + ceiling-hardening**: T1.1/T1.2 = confirm-and-pin the STL-on-v14 (c) + composite (d) ceilings already produced by the T0.2 board (no fresh runs) — **DONE**. **T1.3 encoder-isolation probe (≈5 GPU-h) gates the expensive Tier 2** (if the upstream encoder owns the residual, do the cheap encoder-bypass before the dual-tower) — **DONE (dual-tower justified)**. **T1.4 bounded ceiling-HARDENING (BLOCKS T2)** is the remaining Tier-1 gate: the STL heads were never tuned leak-free, so α re-tune per-state + imbalance losses (reg tail-loss; cat loss-only logit-adj+focal+label-smoothing) + one SASRec probe — then **FREEZE (c)/(d) for the rest of the track**. Positive-EV regardless (gain = better ceiling; null = reviewer-proof baseline). Framing = ceiling/baseline integrity, NOT MTL transfer.

2b. **Tier S — STL Frontier (deep arch search, STL-alone) — runs PARALLEL with Tiers 2-4, not in the linear chain.** S.1 reg encoder/arch search (SASRec, Mamba block, SimGCL aux, leak-free reg-head registry candidates); S.2 cat loss-first + one attention-pooling encoder. FL+AL+AZ 5f + GE confirm. Collocate under MPS alongside the MTL runs (STL-only → near-zero added wall-clock). **Feeds the T5 candidate set + future-work ONLY; NEVER re-opens the T1.4-frozen (c)/(d)** (the moving-baseline trap). De-scoping: do NOT import next-POI models — see Hard rules.

3. **Tier 2 — architecture (CENTERPIECE)**, scored against the **T1.4-frozen** (c)/(d), frozen-fold paired design, spanning the full sharing spectrum: **T2.0 hard-share anchor** (FiLM-off `mtlnet` = Caruana shared trunk + task heads — baseline anchor, expected to lose; completes hard←→soft←→private). T2.1 **reg-private backbone / dual-tower** is load-bearing — gated-fusion (b) is PRIMARY, + PCGrad-off arm, run the regime×substrate 2×2 ({v14, canonical-fresh} × {B9, dual-tower}); **gated by T1.3** (if the encoder owns the residual, do the cheap encoder-bypass first). T2.2 faithful CrossStitch; T2.3 faithful MoE (GATED — §6.3 prior says they lose reg; confirmatory only, skip PLE); T2.4 per-task-input hybrids. Per-arch LR mini-sweep for each. HGI sanity probe per promoted arch. **Pin the architecture.** If nothing recovers a meaningful fraction of the composite gap → composite is the deploy fallback (paper-grade negative). <strong>Do NOT add AdaShare/Learning-to-Branch</strong> — at 2 tasks they collapse to branch-depth, which the dual-tower already spans.

4. **Tier 3 — prior pathway**: T3.1 log_T-KD re-sweep on the new stack (does it stack with the dual-tower?); T3.2 log_T-as-supervision variants.

5. **Tier 4 — loss/optim (GATED)**: run ONLY if Tier 2+3 leave >2pp residual to composite. T4.1 loss balancers (low-EV per P4; class-balanced sampler EXCLUDED — falsified). T4.2 optimizer + per-head LR re-fit (incl. the dual-tower's private-backbone group); validates T3 winner.

6. **Tier 5 — final per-task head re-ablation**: T5.1 reg-head sweep; T5.2 cat-head sweep under the consolidated recipe. **Candidate set = the head registry + any Tier-S STL winners** (this is where the parallel STL branch rejoins; best-STL ≠ best-STL-in-MTL, so a Tier-S winner is re-judged under MTL, not auto-picked). If winner ≠ current → narrow-scope T4.2 LR re-fit on the new head.

7. **Tier 6 — ship**: T6.1 multi-seed n=20 × 5 states (build v14 at CA/TX here), all 4 anchors, three-frontier, Δm. T6.2 paper-canon re-eval (§0.1 re-run + deployable-selector column + PAPER_UPDATE.md).

8. **Review cadence (see Required-workflow §4–§6):** at EVERY tier boundary — (a) advisor pass on the tier results, (b) write the tier summary, (c) **STOP and surface it to the user to decide how to proceed** (no autopilot into the next tier). At the END — a mandatory **implementation-correctness review** (is the code right for this case, not just the numbers?) + a final whole-track advisor pass. Update `log.md` + `INDEX.html` continuously throughout. Capture every review in `log.md`.

## What you compare against (the 4 anchors)

| Anchor | What | Fresh/frozen | Used for |
|---|---|---|---|
| (a) v14-base-MTL | the build-on baseline | fresh | what each tier improves over |
| (b) canonical v11 §0.1 | paper continuity | **frozen (privileged-draw — label it)** | paper-facing Δ only |
| (c) STL-on-v14 ceiling | substrate target | fresh | how close MTL gets to its substrate |
| (d) composite ceiling (STL c2hgi cat + STL HGI reg) | deploy upper bound (+7-12pp reg) | fresh | fraction-recovered by the integrated winner |

**Fresh-vs-frozen:** regime/substrate/architecture claims compare **v14-fresh vs canonical-fresh (gcn_ctrl)**; (b) frozen v11 is paper-continuity only. **Protocol:** n=5 seed=42 gate-keep through Tiers 2-5; n=20 (4 seeds × 5 folds) ships at T6. HGI sanity probe per T2 winner = 2 seeds × AL+AZ × 5f × 30ep, escalate if |MTL+HGI − STL+HGI| ≥ 2pp.

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
