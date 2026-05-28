# Agent Onboarding Prompt — substrate-protocol-cleanup

> **Paste this prompt at the start of every fresh Claude Code session that picks up this track.** This study is intentionally narrow and small-state-only. Its purpose is to close the cheap, non-architectural items left over from `mtl-protocol-fix` while [`docs/studies/mtl_improvement/`](../mtl_improvement/) (branch `mtl-improve`) handles the architectural axis in parallel.

---

## Your role

You are the **implementing agent** for the substrate-protocol-cleanup study. The scope is everything from `mtl-protocol-fix/DEFERRED_WORK.md` that is **orthogonal to the MTL backbone** — i.e. things the parallel `mtl_improvement` study cannot or should not absorb.

You **execute** the Tiers below in order. Within each Tier, sub-experiments may run in parallel. Between Tiers, there are explicit decision gates documented in `INDEX.md`.

You are NOT locked into the design. If results redirect you, propose and pursue the new path. **Document the redirection in `log.md` first.**

## Required reading (in this order, before any code change)

| # | File | Purpose |
|---|------|---------|
| 1 | `docs/studies/substrate-protocol-cleanup/log.md` | Most recent progress and decisions |
| 2 | `docs/studies/substrate-protocol-cleanup/INDEX.md` | Full Tier A-D design + decision gates |
| 3 | `docs/studies/substrate-protocol-cleanup/considerations.md` | Why each Tier is here and what was *not* included |
| 4 | `docs/studies/mtl-protocol-fix/DEFERRED_WORK.md` | Authoritative deferred-work map; identifies every item this study owns |
| 5 | `docs/results/mtl_protocol_fix/phase3_summary.md` | Phase 3 verdicts feeding into Tier A and Tier B |
| 6 | `docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md` §Caveats and follow-ups | The closure-v6 follow-up gates |
| 7 | `docs/studies/merge_design/DESIGN_B.md`, `DESIGN_J.md`, `LEVER_4_POI2VEC_P2R.md` | Substrate variants Tier B re-evaluates |
| 8 | `docs/CONCERNS.md` C15, C21, C22, C23 | Open concerns this study touches |

## Non-goals (explicitly OUT of scope)

These belong to `mtl_improvement` (branch `mtl-improve`). **Do NOT duplicate or touch:**

- MTL backbone alternatives (MMoE, CGC, DSelect-K, cross-stitch, hybrids) — `mtl_improvement` T2a/T2b.
- Loss balancing (NashMTL revive, GradNorm, PCGrad, FAMO, Aligned-MTL) — `mtl_improvement` T3.
- Batch class-balance / weighted CE re-design / focal loss tuning — `mtl_improvement` T4.
- LR / optimizer regimes — `mtl_improvement` T5.
- α formula (`α · log_T` blend at the reg head) — `mtl_improvement` T6.
- Head re-design (next_lstm, next_transformer_pf, next_gru-as-reg, next_getnext no-hard-neg, next_stan_baseline) — `mtl_improvement` T7.
- Multi-seed champion ship — `mtl_improvement` T8.
- §0.1 v11 paper canon re-aval at n=20 — `paper_canon_reevaluation.md`, runs AFTER `mtl_improvement` lands.

These belong to other future-work memos:

- POI decoder with HGI-emb distillation — `poi_decoder_hgi_distill.md` (standby; composite preempts).
- FL/CA/TX composite productionisation — `composite_two_substrate_engine.md` (held until `mtl_improvement` lands).
- POI-level next-POI task pair — `task_pivot_memo.md`.

## Cost discipline (mandatory)

The user has explicitly restricted this study to **small states (AL, AZ)** for the main sweeps. **FL/CA/TX runs are forbidden** except as **1-fold pilots** used solely to confirm a sign-and-magnitude direction on a hypothesis already validated at AL/AZ. Any deviation requires a written justification in `log.md` and explicit user approval via AskUserQuestion.

When deciding whether to run something:

1. Cost > 4 GPU-h at a small state? → Open a decision gate in `log.md`.
2. Cost > 1 GPU-h at FL/CA/TX with > 1 fold? → Hard stop. Re-plan as 1-fold pilot.
3. Cost > 0 GPU-h on the architectural axis (backbones, loss, LR, heads, batch)? → Wrong study. Hand off to `mtl_improvement`.

## GPU evaluation + parallel execution (mandatory before any launch)

Before launching any GPU-bearing Tier item, evaluate the host:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

Record the result in `log.md` under a `**GPU snapshot**:` line at the start of each session. Decide parallel slots from this:

| Free GPU mem | Concurrent small-state MTL runs | Notes |
|---|---|---|
| ≥ 20 GB | up to 3 | each `--batch-size 2048` MTL fold uses ~5-7 GB at AL/AZ |
| 10-20 GB | up to 2 | watch swap; monitor with `nvidia-smi -l 5` |
| < 10 GB | 1 | serial only |

If multiple GPUs are visible, set `CUDA_VISIBLE_DEVICES` per launch and parallelise across devices. Otherwise share one device with the table above.

**Tier-level parallelism map (independent items — safe to run concurrently):**

| Group | Parallel-safe items | Why safe |
|---|---|---|
| G1 | A1 cells (4 seeds × {AL, AZ}) | Independent seeds; no shared state |
| G2 | B1 (Design B) + B2 (Design J) + B4 (Lever 5) | Independent substrate parquets; trained models do not interact |
| G3 | C2 (`--reg-freeze-at-epoch`) + C3 (`--zero-cat-kv`) | Independent code flags; orthogonal pilots |
| G4 | D1 (no GPU) with anything | Audit consumes CPU/IO only |

**Not parallel-safe:**
- Two folds of the SAME run-config (same state, same seed, same flags) — they share the per-fold log_T file path; serialise per state/seed.
- B3 Lever 4 — must wait for B1/B2 winner (sequential).
- C1 3-snapshot routing — must wait for the next training cycle that ships the `--save-task-best-snapshots` flag; not parallelisable with the run it depends on.

**Launch protocol:** when running multiple cells in parallel, prefer `tmux` panes (one per cell) or detached `setsid` subprocesses with stdout/stderr piped to per-cell JSON-line log files under `docs/results/substrate_protocol_cleanup/<tier>/<cell>/`. Document the launch command + PID list in `log.md` so a future agent can reattach.

**Memory safety:** if free GPU mem drops below 2 GB during a parallel sweep, kill one slot and serialise. The cost discipline rule still applies — a failed parallel run that has to be re-done burns more compute than running serial from the start.

## Workflow per Tier

For every experiment within a Tier:

1. **Validate** — read the existing code path, write down the assumed state, double-check it matches reality.
2. **Pre-flight gates** — at minimum:
   - **C22 stale log_T**: `stat -c '%y %n' output/check2hgi/{state}/region_transition_log_seed{S}_fold*.pt` and `output/check2hgi/{state}/input/next_region.parquet`. If log_T mtime < parquet mtime → rebuild before running.
   - **Per-fold seed-tagged log_T** present at the seed you intend to train? If not → rebuild via `scripts/compute_region_transition.py --state {state} --per-fold --seed {S}`.
   - **C23 dev-seed**: if you cite paper-grade numbers, multi-seed {0, 1, 7, 100}; seed=42 only for single-seed development comparisons at small states (AL/AZ).
3. **Code change** — minimal diff; never co-mix with experiments from another Tier.
4. **Unit test** — every new flag exercised at least once.
5. **Re-evaluate** — produce the same `phaseN_*_summary.json` + `.md` artefacts shape `mtl-protocol-fix` Phase 3 used.
6. **Analyse** — three-frontier table (best joint / best disjoint / STL ceiling) per state; Wilcoxon 5/5-fold strict.
7. **Decision gate** — open `INDEX.md` and check whether the Tier-exit condition is met. If not, halt and write the blocker into `log.md`.

## Sequencing summary (see INDEX.md for details)

```
Tier A — multi-seed promotion (small states, paper-grade)
  └── A1. log_T-KD §4.5 multi-seed n=20 at AL/AZ (Phase 3 Rank 1 → paper)

Tier B — substrate cross-study under F1 (MTL re-eval; small states only)
  ├── B1. Design B (POI2Vec at pool boundary) MTL under F1 at AL/AZ
  ├── B2. Design J (H + anchor λ=0.1) MTL under F1 at AL/AZ
  └── B3. Lever 4 (POI2Vec at p2r boundary; additive) at AL/AZ
        └── if B1 or B2 wins → Lever 4 on top of winner

Tier C — protocol coherence
  ├── C1. §4.1 per-task 3-snapshot routing — variant A (3 internally
  │       consistent MTL snapshots, deploy-time router by task)
  └── C2. §4.4 freeze-reg-after-peak (the one curriculum variant P4
          did not falsify) at AL/AZ small pilot

Tier D — no-GPU audit
  └── D1. Window / causal-mask audit (head_window_batch_audit §B)
```

## What "done" looks like

This study closes when:
- A1 has a Wilcoxon n=20 verdict for log_T-KD at AL/AZ.
- B1/B2/B3 have F1-selector MTL three-frontier numbers and a verdict (promoted / null / falsified) at AL/AZ.
- C1 has a 3-snapshot routing prototype with a verdict (Δreg vs joint-best at AL/AZ).
- C2 has a small-state freeze-reg-after-peak pilot verdict.
- D1 has a written audit confirming (or fixing) window/mask correctness.

The study does NOT need a champion. Its purpose is to land cleanly-closed verdicts for items orthogonal to the architectural revisit happening in parallel.

## What you must NOT do

- Touch the v6-final closure provenance of `mtl-protocol-fix`. That study is CLOSED and citable in the paper. Cross-reference only.
- Run anything on the architectural axis. Hand off to `mtl_improvement` instead.
- Expand to FL/CA/TX without explicit user approval.
- Co-schedule with `mtl_improvement` — they are intentionally independent studies on independent branches.

## Pointers

- Closure-v6 verdict for the parent study: [`docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md`](../../results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md)
- Phase 3 artefacts already produced: [`docs/results/mtl_protocol_fix/phase3_*`](../../results/mtl_protocol_fix/)
- Substrate variants source: [`docs/studies/merge_design/`](../merge_design/) (Designs B, J + Lever 4)
- Selector analyser (zero-retrain when val CSVs exist): `scripts/canonical_improvement/analyze_t64_selectors.py`
- Substrate build scripts (reusable): `scripts/probe/build_design_b_poi_pool.py`, `scripts/probe/build_design_j_anchor.py`
- log_T regenerator: `scripts/compute_region_transition.py --per-fold --seed {S}`
- The DEFERRED_WORK map: [`../mtl-protocol-fix/DEFERRED_WORK.md`](../mtl-protocol-fix/DEFERRED_WORK.md)
- Cross-study future-work routing: [`../../future_works/README.md`](../../future_works/README.md) §"2026-05-28 re-routing"
