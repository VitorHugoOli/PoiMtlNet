# T3.3 R-GCN + T3.4 Time2Vec — pre-launch code/design audit

**Date**: 2026-05-15
**Triggered by**: T3.2 multi-seed wrap-up advisor (verdict ACCEPT_WITH_AL_AZ_PRECONDITION), §4 Tier-3 audit recommendation.

**Mandate**: design clean encoder variants before implementation; clear them through the F51 guardrail unit test (param delta gate) and the random-init linear-probe sanity test (T3.1 leak-discriminator) before any FL launch.

---

## T3.4 Time2Vec — CLEAN, low-risk, launch-ready after implementation

**Mechanism (proposed)**: replace the 4 sin/cos temporal features (`hour_sin, hour_cos, dow_sin, dow_cos`) in the check-in node features `data.x[..., -4:]` with a learned `Time2Vec(t)` representation of dimension d_t (paper default 8 or 16). Encoder body unchanged (canonical 2-layer GCN). Input dim grows from `(num_categories + 4)` to `(num_categories + d_t)`.

**Risk surface (advisor §4)**: input-side feature swap. The only leak corner is if Time2Vec features inadvertently preserve a verbatim copy of the category one-hot through a residual or skip path. **Audit**:
- T3.4 does NOT change the encoder body → no new residuals.
- The category one-hot is left untouched in `data.x[..., :num_categories]`.
- Time2Vec is computed from a single scalar `t` (epoch time or seconds-since-midnight) — no category input.
→ **No leak path. Clean.**

**Implementation cost**: ~30 lines.
- Add `Time2VecEncoder(in_dim=1, out_dim=d_t)` in `variants.py`: `f(t) = [ω₀·t + φ₀, sin(ω_i·t + φ_i) for i=1..d_t-1]`.
- Modify `preprocess.py:_build_node_features` to skip sin/cos columns OR keep them and append Time2Vec (additive).
- Modify `regen_emb_t3.py` to pass `--time2vec --time2vec-dim d_t` and replace the 4 temporal cols.

**F51 unit test**: forward shape (N, D=64) preserved. Param delta vs canonical = +2·d_t·(d_t+1) for Time2Vec parameters (≈100 params at d_t=8) ≈ +1 % — well under 50 % gate.

**Random-init probe sanity check**: train a logistic-regression on a 2000-node sub-sample of init-time embeddings to predict the next-category. If F1 > 0.20 at init (canonical ≈ 0.14, GAT was 0.x — pending) → refuse to launch. Run as part of `T3_unit_test_encoders.py`.

**Verdict**: ✅ **GREEN-LIT for implementation + launch**. Defer to user-confirmation only on the choice of d_t (recommend 8 to keep param delta tiny).

---

## T3.3 R-GCN — DESIGN ISSUE FLAGGED, recommend deferring launch

**Mechanism (proposed)**: replace `GCNConv` with PyG `RGCNConv(num_relations=K, aggr='sum', num_bases=None)`. Separate weight matrix per edge relation, sum aggregation, no attention, no edge_attr at forward (`edge_type` is a long tensor of relation indices, NOT continuous weights).

**Where does K come from?** This is the design issue. The current canonical graph has only ONE edge type (`user_sequence` with continuous `edge_weight` for temporal decay). For R-GCN to learn relation-typed aggregation, we need K ≥ 2 structurally distinct relations.

### Option A — switch to `edge_type='both'` (CLEAN, but introduces a graph confound)

`preprocess.py` already supports `edge_type='both'` which adds `same_poi` edges (different check-ins at the same POI) alongside `user_sequence`. This gives K=2:
- relation 0: user_sequence (temporal adjacency)
- relation 1: same_poi (co-visit)

These are structurally distinct relations. No temporal conditioning leaked into the relation typing.

**Confound**: the graph itself changes vs canonical/v3c/T3.2 which all used `user_sequence` only. T3.3's result would conflate **edge-type richness** with **relation-typed aggregation**. To isolate the relation-typing contribution, we'd need to ALSO run "canonical GCN on `edge_type='both'`" as a new baseline → +1 experiment.

### Option B — bucket `user_sequence` edges by `edge_weight` (RISKY)

Define K=2 relations from continuous edge weights:
- relation 0: recent (edge_weight > median)
- relation 1: old (edge_weight ≤ median)

**Risk**: this is exactly the (attention × temporal-edge-attr) corner of the T3.1 leak triangle, just discretized. The R-GCN gets to learn separate weight matrices per temporal bucket — the same effective conditioning as GAT-with-edge-attr, just in long-tensor form. **Probable leak.** Falsified before launch.

### Option C — synthetic edge typing from node category (LEAK BY CONSTRUCTION)

Define K=2 relations by whether src and tgt share a category. Direct category-label leakage. **Reject.**

### Recommendation

- **Option A is the only clean path.** Required experiments:
  1. New baseline: canonical GCN on `edge_type='both'` (5 seeds × FL).
  2. T3.3: R-GCN on `edge_type='both'` (5 seeds × FL).
  Compare T3.3 against the new `both`-graph baseline to isolate relation-typing. Compare new baseline against canonical(user_sequence) to characterise the graph-change effect.
- Cost: ~doubled vs the planned T3.3 slot (2 experiment columns instead of 1).
- F51 unit test: param delta with K=2, num_bases=None ≈ +100 % (one extra in×out weight matrix per layer). Use `num_bases=2` basis decomposition to bring to ~+50 %.
- Random-init probe: same protocol as T3.1, threshold F1 > 0.20.

**Verdict**: 🟡 **YELLOW — needs user sign-off** on whether the 2-experiment cost is acceptable. If yes, implement Option A clean. If we want to keep T3.3 to a 1-experiment slot, fall back to "skip T3.3, do T3.4 only".

---

## Recommended sequencing (assuming user agrees)

1. **Now** (in flight): T3.2 AL+AZ multi-seed (5 sweeps running, ETA ~30 min total).
2. **Next** (after AL+AZ lands): implement T3.4 Time2Vec, F51 unit test, random-init probe, FL launch (5 seeds, ETA ~3 h with 2-parallel).
3. **Decision point**: user confirms whether to do Option A (2-experiment T3.3) or skip T3.3 entirely. If Option A: implement and unit test both R-GCN variants, then `both`-graph baseline FL multi-seed, then T3.3 FL multi-seed.
4. **Tier 3 wrap-up advisor** after T3.4 (and T3.3 if pursued).

---

## What I am NOT doing without sign-off

- ❌ Launching T3.3 R-GCN with `edge_type='user_sequence'` and bucketed edge weights (Option B) — would re-introduce the GAT leak corner.
- ❌ Launching T3.3 against a `user_sequence` graph with only K=1 relation (degenerates to canonical GCN — pointless).
- ❌ Skipping the random-init probe before either T3.3 or T3.4 FL launch — that's the screen the T3.1 advisor recommended after the GAT 99 % cat F1 surprise.
