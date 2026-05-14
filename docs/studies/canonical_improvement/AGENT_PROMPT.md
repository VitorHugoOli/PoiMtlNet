# Agent Onboarding Prompt — Canonical Check2HGI Improvement Track

> **Paste this prompt (or its core block) at the start of every fresh Claude Code session that picks up this research track.** It onboards you with everything you need before touching code.

---

## Your role

You are the **implementing agent** for the canonical Check2HGI improvement research track. A prior agent (design phase) has already:

- Audited the user's 11 considerations against the full check2hgi body of work.
- Run 5 parallel breadth-search sub-agents (feature engineering, model architecture, data flow, training/loss methodology, external literature).
- Designed an 18-experiment slate across 5 tiers in `INDEX.html`.
- Pinned baseline numbers from `results/BASELINES_AND_BEST_MTL.md` and `results/RESULTS_TABLE.md §0.1`.
- Captured the user's alignment (all 5 tiers, T5.2a+T5.2b both shipped, branch `check2hgi-canonical-improve`, falsified-history in front matter).

**You execute the experiments.** You are NOT locked into the design — if results redirect you, propose and pursue the new path. Document the redirection in `log.md` first.

## Required reading (in this order, before any code change)

| # | File | Purpose |
|---|------|---------|
| 1 | `docs/studies/canonical_improvement/log.md` | Most recent progress and decisions |
| 2 | `docs/studies/canonical_improvement/INDEX.html` | Full experiment design, top-down. **Read §Execution guidelines + §Falsified history + §Evaluation framework cover-to-cover before any experiment.** |
| 3 | `docs/studies/merge_design/STUDY_BRIEFING.html` | Big-picture context of the parallel merge-design track |
| 4 | `docs/studies/merge_design/STATE.md` | What's settled vs open in merge-design (especially Phase 11) |
| 5 | `docs/studies/merge_design/AUDIT_HGI_GAP.md` | Why HGI beats canonical on reg, and what's been falsified |
| 6 | `docs/studies/canonical_improvement/considerations.md` | User's original considerations (also audited in INDEX.html §Audit) |
| 7 | `docs/AGENT_CONTEXT.md` | Study-wide workflow |
| 8 | `docs/NORTH_STAR.md` | Canonical training recipe (B9 MTL) |
| 9 | `research/embeddings/check2hgi/CLAUDE.md` | Engine technical reference |
| 10 | `research/embeddings/check2hgi/model/variants.py` | Scaffolded but unintegrated encoder variants (`GATTimeEncoder`, `ResidualLNEncoder`, `Check2HGI_InfoNCE`) — Tier 2/3 head-start |

## Hard rules (do not break)

1. **No merge-family code paths.** Do not import `output/hgi/<state>/poi_embeddings.csv` or any HGI training artifact. Adding a feature derived from HGI's training output is forbidden. Deriving from intrinsic data (OSM `category`, coordinates, visit counts, opening hours) is fine. HGI's *preprocessing infrastructure* (Delaunay graph construction code) is allowed because it's graph construction, not a pretrained artifact.

2. **No fclass-as-feature.** `fclass` is the linear-probe label that the merge_design study uses as the official semantic-recovery gate. Using it in the encoder anywhere — input, supervision, auxiliary loss — is a tautological leak. Reframe to: `category` (coarse 7-class label, already in input), unsupervised prototypes (SwAV K=64), or fclass-correlated proxies (popularity, opening-hours, co-visit category-mix — but those need the held-out-fclass probe-leak test per T4.3).

3. **Pre-flight T1.1 (leak audit) on canonical FIRST.** Establishes the linear-probe floor for every later comparison. Mirror `scripts/probe/leak_sniff_ijm.py`.

4. **Re-run canonical baselines under matched protocol at comparison time.** Never quote pre-2026-05-01 numbers. The merge_design FL +13 pp stale-baseline incident is the cautionary tale.

5. **Statistical claims need n ≥ 10.** At n=5 the Wilcoxon floor is p=0.0312. Any "beats HGI on reg" claim requires T1.2 multi-seed runs to be in place first. Use Table 2 (STL paired-test) for cheap iteration through Tier 1-4; promote final winners to Table 3 (MTL B9 multi-seed v11) for shipping numbers.

6. **Unit-test gate before any T3.* or T5.* multi-fold launch.** Every architectural variant requires a test under `tests/test_embeddings/test_check2hgi_variants.py` confirming forward+backward shape correctness on a synthetic 100-node graph, loss-finite invariant, and parameter count within ~5 % of canonical at D=64 (F51 guardrail).

7. **Falsified history is off-limits.** See `INDEX.html §Falsified history`. Phase 11 S1 (c2p hard negs), S4 (c2p corrupted negs), S3-a (Checkin2Region 4th boundary), S3-b V2-c (per-check-in POI2Vec anchor); Designs A, D, E; Lever 6 and Design K on the merge family; F51 Tier 2 capacity scaling. If you believe you have a *genuinely different* formulation, document the structural distinction in `log.md` before launching.

## Required workflow

1. **Create the dedicated git worktree first**. Branch `check2hgi-canonical-improve` from `main`. Do not contaminate other ongoing work.

   ```bash
   git fetch origin
   git worktree add ../worktree-check2hgi-canonical-improve -b check2hgi-canonical-improve main
   cd ../worktree-check2hgi-canonical-improve
   ```

2. **Use `TaskCreate` / `TaskUpdate` to break down every experiment** into sub-tasks: `unit-test → validate → launch → import → analyze`. Mark `in_progress` before starting, `completed` only when:
   - Results are filled in the experiment's `<div class="results-placeholder">` in `INDEX.html`.
   - A `log.md` entry is written.
   Never leave a task `in_progress` overnight without an explicit blocker note.

3. **Use the `/goal` slash command for autonomous experiment runs.** Each experiment's `Methodology` block in `INDEX.html` is a complete goal description. Queue tier-by-tier; steer when results come in.

4. **After each tier completes, call advisor** with that tier's results. Capture advisor feedback in `log.md`. Apply revisions before moving to the next tier.

5. **After the full track completes (or at any major redirect), spawn a mandatory advisor sub-agent** to evaluate the audit, the experiment results, the integration appendix, the documents. Capture feedback in `log.md`.

6. **Keep `log.md` current.** Append every decision, blocker, falsified hypothesis, redirection, finding. Date every entry (absolute, e.g. `2026-05-14`, never `today`).

## Execution order

1. **Tier 1 first**, in this order: T1.1 → T1.3 → T1.4 → T1.5 → T1.6 (sequential, single-seed). **T1.2 (multi-seed) runs LAST** on the winning recipe so the multi-seed baseline reflects the tuned canonical, not the un-tuned default.

2. **Decision gate after Tier 1:** if T1.3 (α sweep) or T1.5 (optimizer hygiene) closes the AZ reg gap to within 1 pp of HGI, some later-tier experiments may become unnecessary. Report findings + propose stop/continue to user via `AskUserQuestion`.

3. **Tier 2 (loss-shape).** T2.1 (p2r hard negs) is highest-asymmetry; run first within tier. Tier 2 is mostly scaffolded.

4. **Tier 3 (architecture).** F51 guardrail: inductive-bias swaps at D=64, NOT capacity widening. Unit-test gate mandatory.

5. **Tier 4 (semantic recovery).** Attacks the 4 % fclass probe natively. T4.3 has the mandatory held-out-fclass-split leak guardrail.

6. **Tier 5 (stretch).** Only if Tier 1-4 doesn't close the gap. T5.2a (joint Node2Vec POI-POI) and T5.2b (masked POI feature) run as parallel hypotheses.

7. **Final advisor pass.** Document. Synthesize. Propose the shipping recipe (stack of canonical + Tier winners) for the future merge-design synthesis.

## What you compare against

| Stage | Baseline | Protocol |
|---|---|---|
| Tier 1-4 iteration | Canonical Check2HGI STL (Table 2 row 1 in `INDEX.html`) | seed=42, 5 folds, paired Wilcoxon |
| Tier-4 final winners | MTL B9 multi-seed v11 (Table 3) | 4 seeds × 5 folds, pooled Wilcoxon at n=20 |
| Generality | fclass probe 4 % (canonical floor) → 98 % (HGI ceiling) | `scripts/probe/generality_probes.py` |
| Reg floor | Markov-1-region (Table 1 row 3) | sanity gate; never fall below |

## How to format a Results section update in INDEX.html

When an experiment completes, find its `<div class="results-placeholder">Results: <em>pending execution</em></div>` block and replace it with a real results block. Template:

```html
<div class="block">
<div class="block-title">Results (YYYY-MM-DD)</div>
<table>
<thead><tr><th>State</th><th>cat F1</th><th>reg Acc@10</th><th>fclass probe</th><th>Δ vs canonical</th><th>Wilcoxon p</th></tr></thead>
<tbody>
<tr><td>AL</td><td class="num">X.XX ± X.XX</td><td class="num">X.XX ± X.XX</td><td class="num">X.X %</td><td class="num">+X.XX cat / +X.XX reg</td><td class="num">cat p=X / reg p=X</td></tr>
<tr><td>AZ</td><td class="num">...</td><td class="num">...</td><td class="num">...</td><td class="num">...</td><td class="num">...</td></tr>
</tbody>
</table>
<p><strong>Verdict:</strong> <span class="pill good">DOMINANCE</span> | <span class="pill warn">non-inferior</span> | <span class="pill bad">FALSIFIED</span> — one-sentence reason.</p>
<p class="footnote">JSON: <code>results/canonical_improvement/T1-1_AL_AZ_5f50ep.json</code> · Plot: <code>results/canonical_improvement/T1-1.png</code> · Leak-probe value: X.X % (canonical floor: X.X %, Δ X.X pp).</p>
</div>
```

Then update `log.md` with the dated findings block and `TaskUpdate` the corresponding task to `completed`.

## Result file conventions

Following the merge_design pattern:

- Raw per-fold metrics → `docs/results/canonical_improvement/T{tier}-{id}_{state}_{seed}_{folds}f{epochs}ep.json`
- Paired-test JSONs → `docs/results/paired_tests/canonical_improvement/T{tier}-{id}_paired.json`
- Plots → `docs/studies/canonical_improvement/figs/T{tier}-{id}_*.png`
- Trained engines → `output/check2hgi_canonical_T{tier}-{id}/{state}/`

## When you're done with this session

If you finish a tier, mark its tasks completed, update `log.md`, and either:
- Continue to the next tier in the same session if compute allows, OR
- Hand off cleanly: ensure `log.md` ends with a `**Next**:` bullet specifying the exact next experiment ID and any pending decisions.

If you're stopping mid-experiment: ensure the task is still `in_progress` (NOT `completed`), add a `**Blocker**:` line in `log.md` describing exactly what's incomplete and what the next agent needs to resume.

---

## Quick-reference: cited file paths

```
docs/studies/canonical_improvement/
  INDEX.html              # design + index + future results
  log.md                  # progress log (append-only)
  AGENT_PROMPT.md         # this file

docs/studies/canonical_improvement/
  considerations.md       # user's original considerations (audited in INDEX.html)

docs/studies/merge_design/
  STUDY_BRIEFING.html     # parallel-track big picture
  STATE.md                # Phase 11 + merge-design status
  AUDIT_HGI_GAP.md        # what HGI has that c2hgi lacks

docs/results/
  BASELINES_AND_BEST_MTL.md  # full baseline landscape
  RESULTS_TABLE.md           # v11 paper-canonical multi-seed
  paired_tests/              # leak-free paired Wilcoxon JSONs
  P1/                        # P1 experiment outputs (head ablations)
  P0/simple_baselines/       # Random/Majority/Markov floors

research/embeddings/check2hgi/
  check2hgi.py            # entry point (CLI flags)
  preprocess.py           # graph construction
  model/Check2HGIModule.py # loss + 3 boundaries
  model/CheckinEncoder.py # current 2-layer GCN
  model/Checkin2POI.py    # attention pool
  model/variants.py       # SCAFFOLDED but unintegrated: GAT/ResLN/InfoNCE
  CLAUDE.md               # agent reference

research/embeddings/hgi/
  preprocess.py           # Delaunay graph code (infra, allowed)
  model/HGIModule.py      # 25 % hard-neg recipe reference

scripts/
  p1_region_head_ablation.py        # downstream eval template (region head)
  p1_poi_head_ablation.py           # next-POI head (fixed regime: OneCycleLR
                                    #   + label_smoothing as of e99e904; the
                                    #   pre-fix T1 JSONs are pessimistic)
  probe/generality_probes.py        # fclass + kNN-Jaccard probes
  probe/leak_sniff_ijm.py           # leak-audit template (T1.1 reference)
  probe/finalize_design_ijm.py      # paired-test JSON template
  probe/calibrate_canonical_pr_norm.py  # diagnostic: confirms PMA collapse
                                    #   to mean-pool (PR-norm=1.0)
  probe/build_substrate_s1.py       # Phase-11 hard-negs at c2p (FALSIFIED)
  probe/build_substrate_s3a.py      # Phase-11 4th boundary (FALSIFIED)
  probe/build_substrate_s3b.py      # Phase-11 Checkin2Region replacement
  probe/build_substrate_s3b_v2c.py  # Phase-11 per-check-in POI2Vec anchor
                                    #   (FALSIFIED at AL: -9.95pp)
  probe/build_substrate_s4.py       # Phase-11 corrupted negs at c2p
                                    #   (FALSIFIED)
```

The Phase-11 substrate-probe scripts above are the closest existing references
for any canonical-improvement experiment that touches the c2p boundary — read
them before implementing T2.* to mirror the patch pattern. Their FALSIFIED
status is documented in `merge_design/STATE.md §Phase 11`.

---

**Final reminder:** the design is a strong starting point, not a cage. If T1 results suggest the entire problem is fixed by α sweep + optimizer hygiene, that's a legitimate finding — stop and report. If T3.1 GATv2 wins cat but breaks reg in a way that suggests a new direction, propose it. The user explicitly authorized deviation. Document everything in `log.md`.
