# ReHDM STL underperformance — diagnosis (2026-05-01)

## The puzzle

| Architecture | Faithful AL Acc@10 | STL c2hgi AL Acc@10 | STL hgi AL Acc@10 | Δ (substrate − raw) |
|---|---:|---:|---:|---:|
| **STAN**  | 34.46 | 59.20 | **62.88** | **+24.7 / +28.4 pp** ✅ |
| **ReHDM** | 66.06 | 26.22 | 42.78 | **−39.8 / −23.3 pp** ❌ |

Same operation (replace raw inputs with pre-trained substrate, study protocol). STAN gains huge from substrate; ReHDM **loses nearly half its absolute Acc@10**. The 51-pp swing direction-flip implies a missing architectural ingredient in the ReHDM-STL adapter.

## Root cause

**ReHDM's `POILevelEncoder` has no positional encoding** (`research/baselines/rehdm/model.py:76-95`):

```python
class POILevelEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        self.attn = nn.MultiheadAttention(d_model, n_heads, ..., batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(Linear, ReLU, Linear)
        self.ln2 = nn.LayerNorm(d_model)
```

In the **faithful** model, position is encoded implicitly via the 6-ID stack (`EmbeddingLayer` at `model.py:49-73`):

```python
torch.cat([user(u), poi(p), cat(c), hour(h), day(d), qk(qk)], dim=-1)
```

`hour(h)` and `day(d)` differ across the 9 check-ins of a trajectory (each visit has its own hour-of-day and day-of-week), so the input vectors at positions 0..8 are distinguishable. The MSA can attend to "earliest", "most recent" by content even without an explicit positional encoding.

When the **STL adapter** (`model_stl.py`) replaces `EmbeddingLayer` with `Linear(emb_dim=64, d_model=192)` over a per-position substrate embedding, those 9 substrate vectors are structurally similar across positions (each is a 64-dim Check2HGI / HGI embedding of the visited POI). The MSA loses position discriminability, and the **theta-query pooling** (`initial_trajectory_rep`, line 73-81 of `model_stl.py`) — which uses a single learned global query attending to `[v_0, ..., v_8]` — collapses to a near-mean of the substrate sequence with no positional weighting.

This is why theta-pooling fails for substrate input but works for the 6-ID stack.

## Why STAN-STL doesn't have this problem

`NextHeadMTL` (used for STAN-STL via `p1_region_head_ablation.py --heads next_stan`) — see `src/models/heads/next.py`:

1. **Explicit positional encoding** at the input (sinusoidal `PositionalEncoding`).
2. **4-layer transformer** (vs ReHDM-STL's 1 layer).
3. **Attention-based last-position pooling**: queries position 8 (most recent) against positions 0..7. The query is per-batch (the most recent visit's representation), not a single global parameter. Strong inductive bias for next-step prediction.

ReHDM-STL has none of these.

## Sanity check vs the Lightning H100 results

The faithful version's published AL=66.06, AZ=54.65, FL=65.68 numbers were achieved with the 6-ID stack. The "stl_hgi AL=42.78" was achieved with the same architecture stripped of the 6-ID stack. The drop is 23 pp — well above any fold-noise band, and consistent at AZ (−20.7) and FL (−11.2).

## What we fixed (2026-05-01)

Two patches landed in earlier commits:

1. **target_mask bug** (`688cd3b`): `target_mask = torch.ones(b, T)` → derived from `poi_seq >= 0`. Definitive 5f×50ep on GA c2hgi: Acc@10 = 21.22 vs broken 22.31. Mask was a real latent bug but **not the cause of underperformance**. Within fold-σ noise.

## What we tested + negative result

Two opt-in architectural variants added to `ReHDMSTL` 2026-05-01:

1. **Learnable positional embedding** (`use_positional`, ~1.7K params):
   ```python
   self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
   x = self.input_proj(feats) + self.pos_emb[:, :T, :]
   ```

2. **Per-batch last-position-as-query pooling** (`pool_last_pos`):
   New `HGTransformerLayer.attn_local_perbatch` method that supports
   `[B, d]` queries (instead of the original single-global `[d]` target).
   `initial_trajectory_rep` uses the last-valid-position vertex as the
   per-batch query, mirroring STAN's last-position-attention convention.

**Smoke 5f×8ep on GA c2hgi (definitive comparison points):**

| Variant | fold0 ep8 Acc@10 | fold1 ep8 | fold2 ep6 |
|---|---:|---:|---:|
| Broken target_mask + theta-pool (5f×50ep ref) | 0.22 | — | — |
| mask-fix only (5f×8ep) | 0.20 | — | — |
| pos-only (fold0 ep6, killed early) | 0.198 ↓ | — | — |
| **pos + last-pos** | **0.180 ↓** | **0.156 ↓** | **0.185 ↓** |

**Both fixes do NOT help.** pos+last-pos is *slightly worse* than even
mask-fix-only across all 3 folds completed. Trajectory is converging to a
*lower* plateau than the broken baseline.

## Conclusion — architecture-bound

The architectural diagnosis is correct (ReHDM-STL has weaker positional
discriminability than STAN-STL), but the fix surgery is insufficient:

- ReHDM has a **1-layer POI encoder** vs STAN's **4-layer**.
- ReHDM operates on **substrate-as-input** for cold-user StratifiedGroupKFold
  windows, but its hypergraph machinery was designed for **warm-user 24h
  trajectories** with the 6-ID stack as input. The protocol shift is
  load-bearing on top of the architecture shift.
- Even with positional encoding + last-position pooling restored, the
  single-layer transformer + theta/last-pos readout cannot compensate for
  the lost 6-ID stack capacity (192d learnable concat → 192d frozen-projected).

Both flags are kept in `model_stl.py` as **opt-in for future ablation**
(default `False`; published numbers reproduce). Cross-state ReHDM-STL re-run
not warranted; published headline `comparison.md` numbers stand.

## Decision gate (closed)

| Outcome | Action |
|---|---|
| ~~≥ 0.30~~ | ~~Definitive 5f×50ep + 7-cell re-run~~ |
| ~~0.22–0.30~~ | ~~Add last-position query as secondary fix~~ |
| **≤ 0.22 ✅ matched** | **Architecture-bound. Document. No further action.** |

## What this means for the paper

ReHDM-STL underperformance is genuinely architecture-bound and is **honestly
reportable** as an observation: "ReHDM's hypergraph-with-theta-query design
was tuned for warm-user trajectories with rich ID features; under cold-user
StratifiedGroupKFold + frozen substrate input, its 1-layer POI encoder +
single-query readout is outperformed by STAN's 4-layer + last-position
attention by 20–37 pp Acc@10. The faithful variant remains a strong
external SOTA reference under the paper's own protocol."

This is already documented in `next_region/comparison.md` §"Within-baseline
pattern (ReHDM, AL Acc@10)" and the "two honest takeaways" blurb at AL.
