# F50 T4 — Broader Leakage Audit (2026-04-29 19:50 UTC)

**Trigger:** Independent agent audit commissioned after C4 (`α · log_T`) was found to inflate FL val top10 by ~16 pp at convergence — much larger than the original 0.5-2 pp estimate. Needed to know whether other artefacts in the codebase carry similar val→train leakage with trainable amplifiers.

**Output:** 1 severe finding (C4 propagates to 4 more heads), 1 moderate (region_hierarchy weak coupling), low/none for everything else. The TGSTAN head is flagged as **potentially worse than C4** because it has TWO trainable amplifiers on the leaky prior.

Source: agent transcript `/tmp/.../afc3db0e52c9389e0.output`.

---

## Executive scoreboard

| Artefact / Mechanism | Loaded by training? | Built from full data? | Trainable amplifier? | Severity |
|---|---|---|---|---|
| `region_transition_log.pt` (`α·log_T`) | yes (default in legacy scripts) | **yes** (full data) | **yes (α grows ~18×)** | **severe (~16 pp)** — C4 already diagnosed |
| `region_transition_log_fold{N}.pt` (per-fold) | yes (when `--per-fold-transition-dir`) | no, train userids only | yes (α) | none — fix |
| Class weights (`compute_class_weights`) | yes (when `use_class_weights`) | no — `dataloader_*.train.y` only | no | none ✅ |
| `WeightedRandomSampler` (`folds.py:267`) | yes | no — only fed `y_train` slice | no | none ✅ |
| `region_hierarchy.pt` (kmeans clusters) | `next_getnext_hard_hsm` only | label-free kmeans, but on full-data region embeddings | indirect (cluster assignment is fixed buffer) | low (<1 pp; not C4-class) |
| `checkin_graph.pt` | yes (label/region derivation) | full data, **label-free** | n/a | none ✅ |
| HGI / check2HGI / poi2HGI / DGI / time2vec / sphere2vec embeddings | yes — input features | **yes, full data, no fold split** | no (frozen lookup) | low–moderate (structural; no growth term) |
| POI-user mapping / sequence-poi mapping | folds.py diagnostic POI-isolation only | full data | no | none ✅ |
| Markov / simple / floor baselines | reference baselines | **per-fold train-only (verified)** | no | none ✅ |
| `next_stahyper.cluster_priors` | yes | label-free random init | yes (learnable) but starts from random | none ✅ |
| **`next_tgstan` per-sample gate × log_T** | yes | log_T loaded same way as C4 | **YES, BOTH α and gate amplify** | **same severity as C4 (or worse)** |

---

## Severe finding: C4 propagates to 4 more heads

The C4 mechanism applies identically to **every head that loads `transition_path`**:

| head | file | trainable amplifiers | run scripts that may carry leak |
|---|---|---|---|
| `next_getnext_hard` (diagnosed) | `src/models/next/next_getnext_hard/head.py:90,94,134` | `α` | most F50 / runpod / champion scripts |
| `next_getnext` (soft probe) | `src/models/next/next_getnext/head.py:121-152` | `α`, soft-probe MLP | F44 / earlier ablations |
| `next_getnext_hard_hsm` | `src/models/next/next_getnext_hard_hsm/head.py:146-191` | `α`, parent/child classifier | T1.2 HSM (FL run `_0019`) |
| **`next_tgstan`** ⚠ | `src/models/next/next_tgstan/head.py:81,93-121` | **`α` AND `gate = sigmoid(MLP(last_emb))`** | F4× / probe runs |
| `next_stahyper` | `src/models/next/next_stahyper/head.py:79,93-126` | `α`, `alpha_cluster` | rare; scattered |

Any FL run using one of these heads + the full-data `region_transition_log.pt` is potentially inflated by ≥10 pp. The 30+ run scripts under `scripts/run_f4*.sh`, `scripts/run_f50_*.sh`, `scripts/runpod_train_fl_h3alt.sh`, etc. all reference the legacy file.

### Special concern: TGSTAN's compound amplifier

`next_tgstan/head.py:121` computes:
```
final_logits = stan_logits + α * gate * transition_prior
gate = sigmoid(MLP(last_emb))     # per-sample, learnable
α    = nn.Parameter(scalar)        # global, learnable, grows ~18×
```

Both `α` (global scalar, F63 confirmed grows 18×) and `gate` (per-sample, learned by MLP) multiply the leaked prior. Unlike `next_getnext_hard` where only α amplifies globally, TGSTAN's gate can route the leak **selectively** to exactly the (last_region, target_region) pairs that occur in val — a strictly stronger amplification mechanism.

**Hypothesis:** TGSTAN with full-data log_T inflates val top10 by ≥ C4's 16 pp, possibly substantially more. Conclusively settling requires re-running TGSTAN with `--per-fold-transition-dir` (or running the parallel `next_tgstan + per-fold` smoke test ~3 min on GPU).

---

## Moderate finding: `region_hierarchy.pt` weak coupling

`scripts/build_region_hierarchy.py:67-87` runs kmeans over `region_embeddings.parquet` (full data, contrastive). The clustering is label-free (no `region_idx` / `next_category` enters kmeans), but the per-cluster classifier in `next_getnext_hard_hsm` (head.py:125-191) is trainable and conditioned on the cluster assignment.

Mechanism severity: **low**. Cluster assignment is a fixed `register_buffer`, not a scaled prior — no C4-style trainable amplification. Estimated <1 pp impact.

Recommended action: rebuild `region_hierarchy.pt` per fold from per-fold region embeddings; verify empirically. Lower priority than the TGSTAN/getnext re-runs.

---

## Recommended fix priority (top 5)

1. **Audit + re-run all log_T-consuming runs with `--per-fold-transition-dir`.** The diagnosed 16 pp on `next_getnext_hard` likely also applies to `next_getnext`, `next_getnext_hard_hsm`, `next_tgstan`, `next_stahyper`. Re-running these is mechanical (CLI flag) but necessary before any cross-head comparison numbers in the paper.
2. **Verify TGSTAN under per-fold log_T.** Highest hidden-risk leak: if its per-sample gate routes the leak selectively, the inflation could be > 16 pp. Run a smoke 1f×10ep test with `next_tgstan` + per-fold log_T vs full log_T.
3. **Per-fold rebuild of `region_hierarchy.pt`** for `next_getnext_hard_hsm`. Likely <1 pp impact but closes a related concern.
4. **Update docstrings** in `scripts/compute_region_transition.py` to enumerate ALL heads that consume the artefact (currently only mentions one).
5. **Guard in head constructors**: warn when `transition_path` resolves to `region_transition_log.pt` (legacy) rather than `region_transition_log_fold{N}.pt` (per-fold).

---

## Second-most-likely "C4-class" leak (one paragraph)

`next_tgstan` paired with full-data `log_T` is the single highest-risk leak still in the codebase. The head computes `stan_logits + α · gate · log_T[last_region]` where both α (scalar `nn.Parameter`, grows globally) AND gate (per-sample sigmoid-MLP, learned per-input) multiply the leaked prior. Unlike `next_getnext_hard` where only α amplifies globally, TGSTAN's gate can route the leak *selectively* to exactly the (last_region, target_region) pairs that occur in val — a strictly stronger amplifier. If TGSTAN was used in any FL run with the full-data log_T (no `--per-fold-transition-dir`), expect a leak ≥ the C4 16 pp and possibly larger. **Quick triage**: grep run logs for `next_tgstan` + absence of `--per-fold-transition-dir` and re-run those folds clean before quoting any TGSTAN numbers in the paper.

---

## What's CONFIRMED clean (✅)

- Class weights — fed `train.y` only (`folds.py`)
- Weighted sampler — train-only
- Markov / simple / floor baselines — fold-aware (verified `sgkf.split(...)` then index)
- POI-user mapping — diagnostic only (does not feed training labels)
- `next_stahyper.cluster_priors` — random init, no data leak path

---

## Cross-references

- C4 root-cause diagnosis: `F50_T4_C4_LEAK_DIAGNOSIS.md`
- Original audit: `F50_T3_AUDIT_FINDINGS.md` §C4
- Per-fold log_T impl: `scripts/compute_region_transition.py`, `src/training/runners/mtl_cv.py:755-789`
- Embeddings train policy: `research/embeddings/check2hgi/CLAUDE.md` ("No Validation Split — Uses training loss for model selection")
