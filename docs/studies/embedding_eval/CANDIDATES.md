# Dropped/falsified improvements to re-screen with the ladder

Candidates from other studies that were dropped/absorbed/falsified — often judged on a
single MTL metric, single seed, or **only on the final embedding (never on the region-
embedding axis)**. The ladder can now re-screen them cheaply (L0/L1 on the right artifact,
L2 for head/training tweaks) to check whether a real, exploitable signal was missed —
especially one that a future MTL improvement could build on. Source: study-mining sweep
2026-06-01 (canonical_improvement, merge_design, substrate-protocol-cleanup, mtl-protocol-fix).

> **Re-screen principle:** a candidate "passes" only if it improves the artifact the target
> task consumes (next-cat→`embeddings.parquet`; next-reg→`region_embeddings.parquet`) on the
> *correct axis*, multi-state, above fold σ. L0/L1 = eliminate/flag; L2 = confirm; MTL = adopt.

## Tier A — L0/L1-screenable (rebuild embedding, probe in minutes)
| # | Candidate | Source | Changes | Original verdict | Why re-screen | Artifact |
|---|---|---|---|---|---|---|
| 1 | **v3c weight-decay 5e-2** | canonical_improvement T1.5 | AdamW WD=0.05 on the **embedding trainer** (`check2hgi.py --weight-decay 0.05`) | promoted-provisional, then **absorbed by ResLN** in the stack (drop-v3c Δ_reg −0.02, NS) | **Never isolated on the region-embedding axis** — reg was a single MTL cell (FL +0.63). Probe region_emb geometry + region linear/transition, all 5 states. | rebuild `output/check2hgi/` w/ flag |
| 2 | **T2.4 DropEdge** | canonical_improvement T2.4 | epoch-local edge mask on user-seq edges | falsified ("doesn't stack with v3c") | tested only on final embedding; region-emb adjacency structure untested (adj_coh) | rebuild |
| 3 | **T4.3 POI side-features** | canonical_improvement T4.3 | popularity/hours input augmentation | falsified (AL cat +0.63 didn't replicate at FL) | single-state/seed signal; multi-state L0/L1 kNN-purity may reveal a real pattern | rebuild (lightweight) |
| 4 | **T3.1 GATv2 / T3.3 R-GCN / D-design heterograph** | canonical_improvement T3.1/3.3, merge_design D | encoder/graph swaps | **catastrophic cat leak** (cat→99%) | use L0 as a *leak-sniff*: quantify the leak signature vs canonical; document, don't adopt | rebuild |
| 5 | **T6.1 log_T-KD-λ (InfoNCE @ POI-POI)** | canonical_improvement T6.1 | contrastive 4th boundary in embedding loss | failed selector at all λ (within σ) | orthogonal to the **promoted** reg-loss log_T-KD (substrate-protocol-cleanup A1); check if the variant is geometrically sound at L0/L1 even if joint-training killed it | rebuild |

## Tier B — L2-screenable (short STL run)
| # | Candidate | Source | Changes | Original verdict | Why re-screen |
|---|---|---|---|---|---|
| 6 | **T3.4 Time2Vec input** | canonical_improvement T3.4 | temporal positional features | cat −0.56 pp (reg-only trade) | STL next-cat/next-reg separately may differ from the joint-training verdict |

## MTL-only (NOT cheaply screenable — note, don't re-run here)
Design A late-fusion; freeze-reg-after-peak curriculum (C2); zero cat→reg cross-attn (C3); freeze-cat-encoder (P4); class-balanced reg sampler — all manifest only in joint training and were falsified there.

## Recommended first batch
**#1 v3c** (top ROI — the region-axis gap is real and the rebuild is one flag), then **#2 DropEdge** and **#3 POI side-features** (cheap, multi-state). Protocol: rebuild each variant's `embeddings.parquet` + `region_embeddings.parquet`, run `run.py` (next-cat) + `region_eval.py` (next-reg) across FL/AL/AZ, compare to canonical on the *correct axis* with the concordance-call framing. Only promote to an L2/MTL trial if a variant beats canonical above σ on an axis the original eval didn't measure.
