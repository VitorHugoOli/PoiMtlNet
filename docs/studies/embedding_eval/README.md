# Embedding Evaluation Ladder

> **⭐ OUTCOME (2026-06-02): v14 = `check2hgi_design_k_resln_mae_l0_1`** (ResLN+mae cat lever ⊕
> Delaunay-POI-GCN reg lever — orthogonal stack) is the dual-axis champion and supersedes v13 as the
> recommended STL / forward-MTL base. Leak-free multi-seed FL: next-cat 67.36 (≈ frozen-canon, ≫ HGI)
> + next-reg 0.7024 (closes ~69% of the canon→HGI gap; HGI 0.7060 keeps a −0.36pp edge). design_k
> (Delaunay) was WRONGLY discarded by a prior AL/AZ-only study — FL re-validation overturned it; the
> spatial axis is the one that moves L2-reg. STL-only: **no MTL benefit** from v14 or dual-substrate
> routing (pilot) — the MTL cross-attn regime is the binding constraint (Part-2). See
> [`FINAL_SYNTHESIS.md`](FINAL_SYNTHESIS.md) + [`CANONICAL_VERSIONS.md §v14`](../../results/CANONICAL_VERSIONS.md).
> The methodology / ladder content below is the reusable harness and remains valid.

**Goal.** Decide whether an embedding substrate (HGI, Check2HGI canonical, Check2HGI+ResLN+Design-B, …) is *good for the downstream POI tasks* — **next-cat** and **next-reg** today, **next-poi** in the future — using a graded battery instead of a single STL number.

**Why this study exists.** We currently judge substrates by training a full STL head (`next_gru`, `next_getnext_hard`) on frozen embeddings. That conflates three things — embedding quality, head capacity, and the STL≠MTL regime — and we already have a counterexample where it misleads: **ResLN/Design-B is the best STL dual-axis engine but gives NO MTL benefit** (CANONICAL_VERSIONS §v13). A single powerful-head STL score can therefore *crown the wrong winner*. This ladder separates the confounds.

---

## The four levels

| Lvl | Name | What it isolates | Head confound | Cost | Tooling |
|----|------|------------------|---------------|------|---------|
| **L0** | Geometry (training-free) | Raw geometry of the embedding space wrt task labels | **none** | seconds–min | `scripts/embedding_eval/` |
| **L1** | Linear probe | Linearly-accessible task info | minimal, **fixed** | minutes (GPU) | `scripts/embedding_eval/` |
| **L2** | Capacity ladder (sequence STL) | The real sequence task vs head capacity (a *curve*, not a point) | controlled & explicit | GPU-hours | `scripts/train.py --task next/category` |
| **L3** | MTL (deployment) | The actual joint objective we ship | it *is* the target | GPU-hours | `scripts/train.py --task mtl` |

**L0 / L1 are cheap proxies. L2 / L3 are ground truth.** L0/L1 *screen and eliminate* candidates and *explain* L2/L3 outcomes; they may **not** crown a winner unless their rank-correlation with L3 is empirically established (see Protocol §Validity chain).

> ⚠ **Two structural limits of L0/L1 (read before citing any number).**
> 1. **They measure *self*-prediction, not the *transition* task.** L0/L1 ask "can a POI's *own* category/region be recovered from its *own* static embedding?" — a different problem from "predict the *next* POI's category/region from a 9-step history." The sequence dynamics (and, at L2/L3, the `log_T` region-transition prior) are invisible to L0/L1. So L0/L1 are an **upper-bound proxy** for representational content, *not* a forecast of L2/L3 accuracy. Their ranking authority is **zero until the validity-chain ρ gate (below) is actually measured** — until then they may only flag "signal absent."
> 2. **POI-pooling is biased against check-in-level engines.** Pooling Check2HGI's per-visit vectors to one mean-per-POI discards exactly the contextual spread that is its design purpose, while HGI's POI-level vector is lossless. So pooled L0/L1 **systematically undersell** contextual engines. Pooled cross-engine comparison is a **diagnostic, not a verdict**; for any check-in-level engine, the check-in-granularity axis (`--granularity checkin`) MUST be reported alongside before any comparison.

---

## ⭐ How to analyze each task (the decision guide — start here)

This is the canonical, settled protocol (empirically validated 2026-06; full derivation in
[`L0_METHODOLOGY.md`](L0_METHODOLOGY.md)). Follow it exactly when generating paper analyses.

| task | nature | how to RANK substrates | what L0 is good for | command |
|---|---|---|---|---|
| **next-cat** | static attribute (own category lives in the geometry) | **L0 RANKS it** — kNN-LOO + silhouette + centroid-sep on `embeddings.parquet`, **label = category**. L0 tracks L2-cat; you may crown the cat axis at L0. Confirm at L2. | full ranker | `run.py --tasks cat` |
| **next-reg** | transition/sequence (signal in `log_T`, NOT region geometry; corr(region-cosine, T_ij)≈0.05) | **NO static L0 ranks it.** RANK **only at L2** (`next_stan_flow` + log_T, 5-fold, multi-seed `{0,1,7,100}`). | **diagnostics only** (never crown): region-silhouette → spatial-cohesion axis (localizes HGI's cross-substrate win); adj_coh → geographic axis (log_T-redundant); crs-align → fclass axis. Use to *explain* an L2 result. | rank: `p1_region_head_ablation.py --heads next_stan_flow --input-type region --folds 5`; diagnose: `region_eval.py --region-silhouette` |
| **next-poi** (future) | high-cardinality transition | as next-reg: rank at L2; L0 = check-in-granularity recoverability diagnostic only | diagnostic only | — |

**Three rules that prevent the mistakes this study already made:**
1. **Probe the task's real artifact.** next-cat → final `embeddings.parquet`; next-reg → `region_embeddings.parquet` (the `--task-b-input-type region` modality). Never label a final embedding "by region."
2. **Match the metric's label to the task.** Clustering by the 7 categories validates next-cat, NOT next-reg — for region use the region label space (`poi_to_region`). (But see rule 3.)
3. **A static metric can RANK only a static-attribute task.** For the transition tasks (reg, poi) every static-geometry metric is a single-axis *diagnostic* — it explains, it never crowns. The crown is L2. (We tested 8 transition-aware metrics; none ranks reg concordantly — `L0_METHODOLOGY.md §SETTLED`.)

---

### L0 — Geometry (training-free)
> ⚠ **L0 is TASK-SPECIFIC — see [`L0_METHODOLOGY.md`](L0_METHODOLOGY.md) (2026-06-01 audit).** L0 is a
> legitimate **ranker for next-cat** (static-attribute task: own-category lives in the geometry) but
> **diagnostic-only for next-reg** (transition task: the signal lives in log_T, not the region geometry).
> **adjacency-coherence does NOT rank region substrates** (it anti-ranked v13); region rankings start at L2.

Operates on the static per-item embedding table with labels `{category, region, poi}`. No training, so **zero head confound**. Metrics:
- **kNN-LOO accuracy + macro-F1** (cosine, k=10, leave-one-out) — local label coherence.
- **Silhouette** (by category) — global cluster separation.
- **Centroid separability ratio** — mean intra-class / inter-class cosine; cheap global cohesion-vs-separation.
- **Linear CKA vs reference engine** — representational similarity (is variant X just HGI rotated?). *Caveat:* across different-dim/scale engines a low CKA is expected and is not by itself evidence of quality — read it only as "different space," not "better space."

### L1 — Linear probe
Frozen embeddings → a **single `Linear` softmax**, fixed budget, multi-seed. Reports accuracy + macro-F1 (+ top-k / MRR for high-cardinality region). Measures how much task signal is *linearly* readable. The gap L1→L2 tells you whether the info is "easy" or "needs a heavy head."

### L2 — Capacity ladder (sequence STL)
The real next-cat / next-reg **sequence** task at increasing head capacity: linear-ish → `next_gru` → `next_stan_flow`(`next_getnext_hard`). Report the **curve**. A substrate that only wins at the top of the ladder is winning on head capacity, not on representation. Reuses `scripts/train.py`.

### L3 — MTL (deployment)
The joint next-cat + next-reg run under the NORTH_STAR / H3-alt recipe. **The only valid ground truth for what we ship.** Reuses `scripts/train.py --task mtl`. See CLAUDE.md for the canonical invocation (do not trust bare defaults).

---

## The three tasks

| Task | Label | Cardinality | L0/L1 today | L2/L3 today |
|------|-------|-------------|-------------|-------------|
| **next-cat** | target POI category | 7 | ✅ | ✅ |
| **next-reg** | target POI region | ~1k–5k (state-dependent; FL=4703) | ✅ | ✅ |
| **next-poi** | target POI id | O(10⁴–10⁵) | ⚠ checkin-granularity only | ⛔ future work |

The two live tasks are **always reported jointly** — a substrate that helps cat but hurts reg is not an improvement. `next-poi` is a selectable label axis (the per-item `placeid`) but is **meaningless at POI granularity** (one item per POI ⇒ identity, auto-skipped). At check-in granularity it measures POI *recoverability* (do a POI's visits cluster?), which is a representational diagnostic, not the next-poi forecasting task itself.

---

## Protocol invariants (non-negotiable)

1. **Fixed protocol across engines.** Same head, hyperparameters, budget, seeds for every engine. *Never* tune per-engine — that measures tuning effort, not the embedding.
2. **Declared granularity.** Check2HGI is check-in-level (~1.4M rows); HGI is POI-level (~76k). POI-pooled (mean per `placeid`) is the only way to align them item-for-item, but it is **biased against contextual engines** (see ⚠ above), so it is a **diagnostic**, not the primary verdict. Check-in granularity MUST be run for any check-in-level engine. Standardize probe inputs on the train split (engines differ in scale).
3. **Multi-seed.** Report mean ± SD over seeds `{0,1,7,100}`. With only 4 seeds, SD is a spread estimate, **not** a 95% CI (a t-based CI is ~3× wider). Never single-seed, and never the **development seed 42** (it overshoots paper §0.1 by +3 pp CA / +8 pp TX — CLAUDE.md). L0 metrics are single-shot point estimates (no seed variance).
4. **Splits.** Sequence tasks (L2/L3) use StratifiedGroupKFold on `userid` (as in `FoldCreator`). L0/L1 per-item probes use a stratified random POI split — a *different* (and acceptable, no sequence leakage at item granularity) split; this proxy-gap is intentional and must be stated when comparing L1 to L2/L3. High-cardinality region has many singleton classes ⇒ stratification falls back to random and some test regions are unseen in train (depresses absolute region scores; `test_class_coverage` is reported per run).
5. **Region partition is shared.** Region labels come from one geographic partition (`check2hgi/<state>/temp/checkin_graph.pt`: `placeid_to_idx`+`poi_to_region`) applied to every engine by `placeid`, so HGI and Check2HGI are scored against the *same* regions.
6. **L3 is the sole ranking authority; L0/L1 are permanent screens/explainers.** L0/L1 are proxies. We originally framed their promotion to "ranking" as conditional on a Spearman ρ vs L3 — but with only ~5 engines (HGI exists at FL only) and a near-saturated, near-identical Check2HGI family (cat 0.984–0.988), a pooled ρ has near-zero power, a restricted range, and a state confound; it can never legitimately "unlock" ranking. So: **L0/L1 never crown a winner.** They (a) *eliminate* — flag "signal absent"; (b) *explain* L3 outcomes post-hoc; (c) report **concordant/discordant calls** on the few discrete decisions that matter (e.g. "does the proxy correctly place HGI below the Check2HGI family?") as *descriptive* color, never an inferential p-value. The winner is always blessed by L3.

7. **Each task is screened on the embedding artifact it actually consumes (CRITICAL — corrected 2026-05-31).** The two tasks take *different inputs*, so they must be probed on different artifacts:
   - **next-cat** consumes the **final per-item embedding** (`embeddings.parquet`) → probe it (run.py). POI-pooled denoises this POI-constant label, so pooled is fine and the HGI comparison is legitimate.
   - **next-reg** consumes the **region embedding** (`region_embeddings.parquet`, `reg_0..reg_D`), looked up per sequence step via placeid→region (the `--task-b-input-type region` modality — the check-in modality is the *weak* one, ~20% vs ~53% Acc@10). **Both HGI and Check2HGI produce region embeddings and both run next-reg STL** (via `scripts/p1_region_head_ablation.py --region-emb-source {hgi,check2hgi}`); only the `next_region` *label* builder is Check2HGI-only. **The real STL result has HGI winning next-reg at all 5 states** (RESULTS_TABLE §0.3, FL 71.3 vs 69.2).
   - ⚠ The earlier run.py next-reg numbers (HGI 0.074 vs C2HGI 0.051 pooled; 0.05→0.23 check-in) probed the **wrong artifact** (final embedding labelled by region). They are **superseded** by `scripts/embedding_eval/region_eval.py`, which probes the actual region embedding. Its L0 adjacency-coherence *agrees with* HGI's region win (0.326 vs 0.274) but this is **coincidental concordance, NOT a region screen** — within the Check2HGI family adj_coh anti-ranks (v13 lowest adj_coh 0.231, yet best L2-region); adj_coh is a **diagnostic, not a ranker** (see [`L0_METHODOLOGY.md`](L0_METHODOLOGY.md)). Its 1-step transition probe is a near-tie across engines — too crude (self-transition rate 0.49, no 9-window/log_T prior) to resolve the ~2 pp gap, which therefore remains an **L2/L3-only** verdict.
   - **Data caveat:** HGI `region_embeddings.parquet` exists only at FL; AL/AZ HGI inputs were byte-identical to Check2HGI (2026-05-20 pipeline bug, `docs/findings/PHASE3_INCIDENTS.md`), so the cross-engine region screen is clean only at FL.

8. **Per-metric normalization (read when comparing L0 to L1).** The metrics intentionally preprocess differently: kNN-LOO and centroid use **unit-norm cosine** geometry; the linear probe **z-scores** features on the train split (cross-scale fairness); CKA only **mean-centers**. So an "L0 says X but L1 says Y" gap can partly reflect preprocessing, not representation — compare like-for-like.

---

## Outputs

`scripts/embedding_eval/run.py` writes tidy results to `docs/results/embedding_eval/`:
- `metrics_long.csv` — one row per (engine, state, level, task, granularity, metric, seed, value).
- `summary.md` — engine × task comparison table (mean ± CI) per level.

L2/L3 are launched via `scripts/train.py` (the harness emits the exact commands); their metrics are pulled from the standard `results/<engine>/<state>/` tree.

## Status
- **2026-06-02 — Part-1 (substrate) CLOSED.** Champion = **v14 = `check2hgi_design_k_resln_mae_l0_1`** (see OUTCOME banner at top). design_k (Delaunay) reopened and re-validated at FL (overturned the prior AL/AZ-only K≡J falsification); leak-free multi-seed confirms design_k +0.9–1.1pp reg over canonical (closes 54% AL / 78% FL of the HGI gap; v14 with resln+mae closes ~69% at +2.5pp cat); HGI keeps a small significant reg edge. **No MTL benefit** from v14 or dual-substrate routing (2-fold seed42 pilots) — the MTL cross-attn regime is the wall (Part-2). v13/v14 mechanisms graduated into `Check2HGIModule` (`reg_poi_mode`). Authority: [`FINAL_SYNTHESIS.md`](FINAL_SYNTHESIS.md) 2026-06-02 sections + [`CANONICAL_VERSIONS.md §v14`](../../results/CANONICAL_VERSIONS.md).
- **2026-05-31** — Study created; L0+L1 harness built, advisor-reviewed, and run on `{hgi, check2hgi, check2hgi_design_b, check2hgi_resln, check2hgi_resln_design_b}` (FL pooled + FL check-in + AL/AZ pooled). Headline: category saturated across the Check2HGI family (HGI far behind); the pooled "HGI wins region" result is a **pooling artifact** (check2hgi region probe 0.05→0.23 at check-in granularity); ResLN is the strongest substrate on the proxy axis but its transfer to MTL is the open ρ-gate. **L0/L1 carry screening authority only — ρ-vs-L3 not yet measured.** Findings + caveats in `log.md`; numbers in `docs/results/embedding_eval/{fl_poi,fl_checkin,smallstates_poi}/`.
