# substrate-protocol-cleanup — Considerations

**Date drafted:** 2026-05-28
**Purpose:** Capture the reasoning behind every Tier in `INDEX.md` and, more importantly, **what was rejected from scope and why**. Without this file, a future agent cannot tell whether something was forgotten or deliberately excluded.

---

## What this study is

A targeted cleanup of items from `mtl-protocol-fix/DEFERRED_WORK.md` that are **orthogonal to the MTL backbone architecture**. The architectural axis (backbones, loss balancing, batch, LR, α, heads, multi-seed champion) is owned by `mtl_improvement` on the parallel branch `mtl-improve`.

## What this study is NOT

A re-opening of `mtl-protocol-fix`. The parent study's closure (v6 final, 2026-05-24) stands and remains the citable artefact. This study cites it but does not modify its verdicts or its provenance.

A substrate competition. Even if Tier B promotes Design B or Design J under F1, the result is a substrate "free upgrade" that any architectural champion from `mtl_improvement` can adopt — it is NOT a separate substrate-axis paper claim. The substrate axis closure from `canonical_improvement` Tier 1-6 (±0.8 pp) still holds at the canonical-vs-canonical-variants framing; B and J are substrate-axis variants from `merge_design` that happened to never get the MTL+F1 evaluation, which is the only reason this study touches them.

## Three-way ownership map (per Tier)

| Item | Owner | Why this owner |
|---|---|---|
| log_T-KD multi-seed n=20 (Tier A) | this study | Phase 3 PROMOTED at n=5; paper-grade upgrade is cheap (no code) and orthogonal to backbone. |
| Designs B / J MTL under F1 (Tier B1/B2) | this study | Substrate variants, no backbone change. Sits cleanly in the gap between `merge_design` (substrate STL) and `mtl_improvement` (backbone). |
| Lever 4 POI2Vec at p2r (Tier B3) | this study | Additive substrate lever, no backbone change. |
| §4.1 per-task 3-snapshot routing (Tier C1) | this study | Deploy-time protocol/serving concern. Touches `BestTracker` not the backbone. |
| §4.4 freeze-reg-after-peak pilot (Tier C2) | this study | Cheap pilot of the one curriculum variant P4 did not falsify. Single `--reg-freeze-at-epoch N` flag, no architectural change. |
| Window/mask audit (Tier D1) | this study | No GPU, no architectural change. |
| MMoE/CGC/DSelect-K/cross-stitch (T2a/T2b) | `mtl_improvement` | Backbone alternatives. |
| Loss balancers (T3) | `mtl_improvement` | Co-designed with backbone. |
| Batch class-balance / focal-loss-only (T4) | `mtl_improvement` | Coupled to head + loss; sampler form already falsified Phase 3 Rank 2. |
| LR / optimizer regimes (T5) | `mtl_improvement` | Tunes to whichever backbone wins. |
| α formula at reg head (T6) | `mtl_improvement` | Couples to log_T usage; KD form lives here (Tier A), input-blend form is arch axis. |
| Head re-design (T7) | `mtl_improvement` | Coupled to backbone capacity. |
| FL/CA/TX composite productionisation | `composite_two_substrate_engine.md` | Held until a champion lands; large-state work. |
| §0.1 n=20 paper canon re-aval | `paper_canon_reevaluation.md` | Standby until `mtl_improvement` lands champion. |
| POI decoder distill (§4.8) | `poi_decoder_hgi_distill.md` | Standby; composite preempts. |

## Why the §4.1 variant decision matters

`DEFERRED_WORK.md` writes §4.1 as "ship cat head from cat-best, reg head from reg-best, shared backbone from joint-best." Taken literally, this is **variant C** — three different epochs contribute to one served model. The cat head was never trained against the joint-best epoch's backbone features; the reg head likewise. The intermediate representations the backbone produces at epoch X have a different distribution than what the cat head learned to expect at epoch Y. Variant C is mechanistically incoherent.

The user's intent (confirmed 2026-05-28) is **variant A** — three independent full MTL snapshots, one per "best-epoch axis", routed by task at inference. Each snapshot is internally consistent (the head it serves was trained against that snapshot's backbone). Cost: 2-3× storage, 2-3× param load at deploy. Acceptable.

Variant B (the F1 selector at a single snapshot) is what `mtl-protocol-fix` already shipped — no further work.

**Variant C-prime (deferred re-open trigger).** A hybrid not in DEFERRED_WORK: load the variant-C mixed snapshot (backbone from joint-best, heads from per-task best) and then run **1-2 epochs of joint fine-tune** on the train fold to re-align the heads to the new backbone. This resolves the coherence objection at the cost of a short fine-tune pass per fold; storage stays at 1×, deploy stays at 1×, only retraining cost is added at calibration time. **Not in this study's scope**, but worth re-opening if (a) variant A wins under Tier C1 AND (b) the 2-3× deploy storage of variant A becomes the binding constraint. If variant A loses under C1, variant C-prime is unlikely to recover (heads are not the bottleneck if 3-snapshot routing already doesn't help).

## The P4 frozen-cat residual hole (motivation for Tier C3)

P4 (`phase1_phase2_verdict_v6_final.md` §P4) froze the cat encoder *parameters* from epoch 0 and zeroed `cat_weight`. The verdict was: MTL reg still peaks at ep 2 and degrades by ep 11, therefore cat-vs-reg interference is NOT the residual mechanism — the residual is architectural.

The advisor pass (2026-05-28) flagged a residual hole: under cross-attention MTL (C20, CH20), the cat encoder *output* still flows through K/V into the shared backbone even when its parameters are frozen. P4 isolated cat-parameter contributions but not cat-activation contributions. If silenced K/V from the cat path changes reg dynamics, the residual mechanism is **K/V capacity stealing**, not the shared-backbone parameters per se — a different architectural mechanism that `mtl_improvement`'s arch axis can target more specifically.

Tier C3 is a near-zero-compute pilot of this stricter form (`--zero-cat-kv` flag). It's not a paper claim by itself; it either fully closes P4's conclusion or surfaces a sub-mechanism worth handing to `mtl_improvement`.

## Why §4.4's surviving variant is the freeze-reg-after-peak form

P4 frozen-cat falsified the "cat-vs-reg interference" hypothesis at its strongest reading: even with cat completely frozen and `cat_weight=0` from epoch 0, MTL reg still peaks at ep 2 and degrades by ep 11. So the gradient-conflict framing of curriculum (let cat warm up before reg, or let reg warm up before cat) is closed.

The surviving variant is asymmetric: let reg reach its ep ≈ 2-4 peak with the standard joint loss, then **freeze the reg head + its specific encoder** and continue training cat-only. The hypothesis is not "cat hurts reg" (already falsified) but "freezing reg at its peak preserves the reg capacity while allowing cat to continue improving." This is a pure deploy-side outcome, not an interference-resolution claim. Cheap (4 GPU-h) and either it works or it definitively closes §4.4.

## Why §4.7 is a substrate re-evaluation and not a substrate competition

Designs B and J are merge_design's STL champions (the only designs that achieved DOMINANCE under TOST + Wilcoxon at AL/AZ with STL `next_stan_flow`). They were never evaluated MTL+F1. The hypothesis is narrow: **does the STL lift survive MTL training under the F1 selector?** If yes, that lift can be added to the architectural axis as a free upgrade; if no, the substrate axis is even more closed than canonical_improvement Tier 6 declared.

The negative outcome here is informative: it would tighten the case that backbone is the binding constraint, since not even the strongest STL-improving substrate survives MTL training.

## Why Levers 4 and 5 are in this study, Lever 6 is not

- **Lever 4** is additive (region-prior term on `L_p2r`), independent of substrate engine, ~4 GPU-h. Fits this study (Tier B3).
- **Lever 5** (KL distill on top-k softmax over neighbours, ~3 GPU-h on MPS at AL+AZ per `merge_design/LEVER_5_DIST_DISTILL.md`) was originally deferred to `merge_design`, but `merge_design` closed (STATE.md 2026-05-06) leaving Lever 5 as an orphan. **Advisor pass 2026-05-28 absorbed it as Tier B4** because (i) it modifies only the substrate-build script, not the backbone, so champion architecture is irrelevant; (ii) the cost is below the Tier C2 pilot.
- **Lever 6** (two-output integrated engine, 16-24 GPU-h) was FALSIFIED 2026-05-06 in its original framing; the deploy composite (§4.2) covers the same goal without the integration cost. Stays in `merge_design/LEVER_6_FINDINGS.md` as the falsified-history record.

## Why FL/CA/TX is forbidden for main sweeps

User directive 2026-05-28: this is a study phase, not a production phase. Large-state runs commit ~5-10 GPU-h per single-state single-seed cell, which dwarfs the entire small-state Tier A budget. The architectural axis at `mtl_improvement` is where the FL/CA/TX paper-grade numbers should come from once a champion lands. This study informs that work but does not pay its compute.

1-fold pilots at FL/CA/TX are permitted only to **confirm sign-and-magnitude** for a hypothesis already validated at AL/AZ multi-seed — i.e. as a sanity check, not as a measurement.

## What is being deliberately not done (and what would un-pause each)

| Item | Why not now | What would un-pause it |
|---|---|---|
| FL/CA/TX productionisation | Large-state compute | A champion lands in `mtl_improvement` |
| POI decoder distill | Composite preempts; substrate-axis closure | Composite paper-rejected on a non-recoverable point |
| n=20 paper canon re-aval | Awaiting champion | `mtl_improvement` lands champion |
| §4.4 cat-first curriculum | Falsified via P4 | (nothing) — closed |
| Class-balanced sampler | Falsified via Phase 3 Rank 2 | (nothing) — closed |
| Lever 5 KL distill | ~~Better baseline first~~ — **moved into scope** as Tier B4 (advisor pass 2026-05-28) | — |
| Lever 6 two-output engine | Falsified 2026-05-06; composite covers the goal | Composite paper-rejected |
| §4.1 variant C-prime (mixed snapshot + 1-2 ep joint fine-tune) | Variant A first; storage cost only matters if A wins | Variant A promotes AND deploy storage becomes binding constraint |

## Risk registry

| Risk | Mitigation |
|---|---|
| `mtl_improvement` lands a champion before Tier B completes, making Tier B substrate work dilutive polish | OK — Tier B is cheap and informative regardless. The substrate verdicts strengthen any paper claim about backbone-vs-substrate decomposition. |
| Variant A 3-snapshot routing fails (Δreg @ task-best vs joint-best < 2 pp) | This is itself a useful finding — closes §4.1 entirely, strengthens the case that F1 already extracts most of the per-task capacity at a single checkpoint. |
| Window/mask audit finds a leak | Halts Tier B/C immediately; fix + re-run any affected runs; file a CONCERN. |
| User wants to expand to FL/CA/TX mid-study | Hard stop in `AGENT_PROMPT.md`; requires explicit re-design and budget approval. |
| Merge collision with `mtl_improvement` branch | Touching points: `BestTracker` (C1), `--reg-freeze-at-epoch` flag (C2), C22 preflight (all). All others are localised to substrate scripts or this study's analysers. Periodic cross-branch rebase plan. |

## Pointers

- Parent closed study: [`../mtl-protocol-fix/`](../mtl-protocol-fix/) (read `DEFERRED_WORK.md`, `phase1_phase2_verdict_v6_final.md`, `phase3_summary.md`).
- Parallel study: [`../mtl_improvement/`](../mtl_improvement/) (read `INDEX.html`, `AGENT_PROMPT.md`, `log.md`).
- Substrate source: [`../merge_design/`](../merge_design/) (Designs B, J + Lever 4).
- Future-works re-routing: [`../../future_works/README.md`](../../future_works/README.md) §"2026-05-28 re-routing".
