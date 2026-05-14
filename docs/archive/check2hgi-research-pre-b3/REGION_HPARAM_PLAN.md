# Region hyperparameter investigation plan

**Context:** Cross-attn MTL shows monotone widening reg gap: AL −4.53 → AZ −7.81 → FL −10.73 pp. Hypothesis: MTL region path is capacity-starved because the shared backbone splits budget across tasks while region class count grows with state.

The region path components inside MTL:
1. **Task-b (next) encoder**: `Linear(64→256) + ReLU + LayerNorm + Dropout`, then another Linear(256→256) block.
2. **Cross-attention blocks (2× in mtlnet_crossattn)**: 4 heads, 256-dim.
3. **GRU head**: `hidden_dim=256, num_layers=2, dropout=0.3` consumes `[B, 9, 256]`.

## Hypotheses ranked by expected lift × cost

### H-R1 — GRU hidden_dim=512 (HIGH priority, cheap)

**Claim:** doubling the GRU head's hidden_dim gives the region-specific path more capacity to compensate for MTL's shared-backbone dilution. Adds ~200K params to the region head only.

**Why it matters:** the GRU consumes the FINAL pre-head representation [B, 9, 256]. If the shared backbone has stripped region signal, a bigger GRU can't recover what's lost — but it can better exploit what remains. On AL's GRU standalone, hd=384 gave 54.68 at 1f × 50ep (similar to hd=256's 53.33) — so diminishing returns on STL. In MTL, the story may differ because the effective input has more of both tasks' signal interleaved.

**Command:**
```bash
--model mtlnet_crossattn --mtl-loss pcgrad \
  --reg-head-param hidden_dim=512 \
  --max-lr 0.003
```

**Expected:** +1–3 pp on reg Acc@10 if capacity is the bottleneck.

### H-R2 — GRU num_layers=3 (MEDIUM priority)

**Claim:** deeper recurrence improves sequential reasoning on the 9-step region history.

**Evidence against:** P1 scaled GRU (hd=384, nl=3) was only +1.4 pp over default on STL. Already tried on STL without big win.

**Expected:** ≤1 pp in MTL.

### H-R3 — GRU dropout=0.15 (MEDIUM priority, cheap)

**Claim:** MTL pipeline already adds regularization (FiLM/cross-attn dropout 0.15 in shared layers). The GRU head's extra dropout=0.3 may over-regularize, especially for region which has less effective signal.

**Expected:** +0–1 pp.

### H-R4 — Region-heavy loss weights via static_weight cat=0.3 (MEDIUM-HIGH)

**Claim:** our Phase 2 λ sweep established equal-weight (λ=0.5) as best among pcgrad alternatives. But cross-attn wasn't in that sweep. Region-heavy weight (cat=0.3 = reg 0.7) might close region gap at small cat cost.

**Command:**
```bash
--model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.3 \
  --max-lr 0.003
```

**Expected:** reg +1–3 pp, cat −0.5 to −1 pp.

### H-R5 — TCN-residual as reg head (LOW priority, different architecture)

**Claim:** TCN has different inductive bias (convolutional receptive field over sequence). P1 tied GRU on standalone (56.11 vs 56.94).

**Blocker:** requires changing `task_b.head_factory` in the preset or adding another CLI flag.

**Expected:** ≤1 pp vs GRU; mostly variance check.

## Proposed execution sequence (after Nash-MTL completes)

1. **H-R1** (GRU hidden_dim=512) — cleanest test of capacity hypothesis. ~45 min on AZ. **Do first.**
2. **H-R4** (static_weight cat=0.3) — cheap, orthogonal axis. ~45 min on AZ.
3. Combine winners: if H-R1 and H-R4 both help, combine. ~45 min.
4. **If significant lift found on AZ:** replicate on FL 1f × 50ep.

Total budget: ~2–3 h on AZ, then maybe 1 h on FL if promising.

## Decision rule

If any single region-hparam change on AZ gives ≥2 pp reg Acc@10 lift (vs the 41.07 cross-attn baseline we already have), it's worth testing at scale. Smaller lifts may be noise (AZ reg std ±3.46).
