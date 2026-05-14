# STAN — Critical Review & SOTA Search

**Date:** 2026-04-20. Written after the MTL-STAN AZ regression ([`MTL_WITH_STAN_HEAD.md`](MTL_WITH_STAN_HEAD.md)).

## 1 · The AZ regression

Recap of the problem:

| State | STL STAN | STL GRU | MTL-GRU | MTL-STAN | Δ_MTL (STAN−GRU) |
|---|---:|---:|---:|---:|---:|
| AL (10 K rows) | 59.20 | 56.94 | 45.09 | **50.27** | **+5.18 pp** |
| AZ (26 K rows) | 52.24 | 48.88 | 41.07 | 37.47 | **−3.60 pp** |

STAN STL dominates at both scales, but swapping STAN into MTL as the region head **reverses direction** between AL and AZ. We need to understand why, in priority order of testability.

## 2 · Root-cause analysis

### 2.1 Primary hypothesis — dimensional bottleneck

**Claim:** In the MTL setting, the input to the region head is `[B, 9, shared_layer_size]` where `shared_layer_size=256`. Our STAN head projects this to `d_model=128` via an input linear layer. That is a **50% dimensional compression before the first attention layer.** GRU (`hidden_dim=256`) has no such projection.

Evidence:
- STL STAN runs on `[B, 9, embed_dim=64]` (raw check2HGI embeddings) and projects *up* to `d_model=128`. Dimensional expansion, no information loss.
- MTL STAN runs on `[B, 9, 256]` (backbone-refined representation, already enriched by FiLM / cross-attention). Projects *down* to 128 — discarding signal that the backbone has already computed.
- At AL scale (10 K rows), the backbone's extra dimensions may carry noisy or redundant information; compressing to 128 still retains the useful part, and STAN's attention wins on inductive bias.
- At AZ scale (26 K rows), the backbone learns richer 256-dim representations (more data → more signal packed into the wider vector); compressing half of that to 128 discards usable signal, and GRU's 256→256 path preserves it.

**Prediction:** Setting `d_model=256` (matching the backbone output dimension, eliminating the projection bottleneck) should lift MTL-STAN on AZ closer to — or above — MTL-GRU.

**Cost:** Parameter count grows ~3× (STAN d_model=256 is ~1.2 M params vs 417 K at d_model=128). Still comparable to `next_gru` (~770 K) and `next_mtl` transformer (~1.3 M), so not a fairness issue.

### 2.2 Secondary hypothesis — PCGrad interaction with attention gradients

**Claim:** PCGrad computes per-task gradients then projects pairwise conflicting components. Attention weights (Q, K, V, pairwise bias) produce gradients with a **different correlation structure** than GRU's weight-tied recurrent gradients. The projection step may zero out more useful signal for STAN than for GRU.

Evidence (indirect):
- Cat F1 is **unchanged** between MTL-GRU and MTL-STAN on both states (38.58 vs 39.07 on AL; 43.13 vs 42.64 on AZ — all within σ). That tells us PCGrad's balancing of the category stream is head-independent. The head-specific effect is on the region side only.
- The GradNorm / NashMTL / static-weight alternatives were tested in P2 with GRU head only; we haven't paired them with STAN.

**Prediction:** MTL-STAN with `--mtl-loss static_weight` (no gradient projection) should differ from MTL-STAN with pcgrad. If the static-weight version *beats* pcgrad on AZ, PCGrad×STAN interaction is real.

**Cost:** ~1 extra 5f×50ep run per state (~25 min AL, ~40 min AZ).

### 2.3 Tertiary hypothesis — per-position bias overfit

**Claim:** STAN's pairwise bias is `[num_heads=4, 9, 9] = 324` free parameters. At AZ's larger per-fold training set, these per-(i,j) biases may overfit to fold-specific patterns that don't generalize.

Evidence (weak):
- AZ has more samples per fold (~21 K vs ~10 K on AL). More data usually *helps* overfitting, so this cuts the other way — the bias should benefit from more samples, not suffer.
- But: the 324 bias params are **not shared across check-ins**. Unlike the Q/K/V projections which see every token, each bias cell `B[h, i, j]` is trained only on pairs where a 9-window has data at positions `i` and `j`. At 9-step padding prevalence, not all 81 cells see the same amount of data.

**Prediction:** Increasing dropout on the attention path (0.3 → 0.4) should help more at AZ than AL if this hypothesis is correct. Alternatively, switching the pairwise bias init from `std=0.02` Gaussian to ALiBi-decay (recency prior) would reduce the effective DOF.

**Cost:** 2 extra runs per state.

### 2.4 Quaternary hypothesis — backbone-head distribution mismatch

**Claim:** The cross-attention backbone's output distribution was implicitly shaped during P2 architecture ablation by the GRU head's gradients (every P2 run used `next_gru`). STAN's attention expects input statistics that subtly disagree with what the cross-attn block has learned to produce.

Evidence: purely circumstantial — P2 grid-searched 5 architectures × 4 optimizers, all with the GRU head. The backbone parameters survived into this experiment via the architecture choice.

**Prediction:** Training the MTL system from scratch with STAN head from step 1 (not swapping it in after the backbone is "set") should lift MTL-STAN at AZ. We already do this (no pretrained backbone), so the mismatch would have to come from *architectural* choices (shared_layer_size, num_shared_layers, etc. tuned for GRU).

**Cost:** Expensive to test directly; would require re-running P2 with STAN. De-prioritize.

## 3 · SOTA alternatives — did we pick the best baseline?

### 3.1 Papers surveyed

| Paper | Year | Venue | Primary contribution | Fit to our data |
|---|---|---|---|---|
| **STAN** | 2021 | WWW | Bi-layer self-attn + ST interval bias | ✓ implemented (`next_stan`) |
| **GETNext** | 2022 | SIGIR | Transformer + trajectory flow graph (global POI-POI transition matrix) | ✓✓ — we have check2HGI's region-transition graph as ready-made flow map |
| **HMT-GRN** | 2022 | SIGIR | Hierarchical MTL (region + POI, hierarchical beam search) | ✗ sibling MTL — overlaps with our contribution |
| **Graph-Flashback** | 2022 | KDD | Graph memory + flashback for repeat-visit patterns | ~ overkill; adds temporal flashback signal we don't have labels for |
| **MGCL** | 2024 | Frontiers | Multi-graph + multi-granularity contrastive learning | ✗ MTL inherently; needs contrastive pretraining |
| **ImNext** | 2024 | KBS | Irregular interval attention + MTL | ~ requires raw timestamps (we have check2HGI's contextual encoding, not Δt) |
| **SGRec** | 2021 | IJCAI | Session graph + self-attn | ~ closer to session recommendation, different data model |
| **STGN / Zhao et al.** | 2019 | — | Spatio-temporal gates on LSTM | Simpler than STAN, weaker baseline |
| **MRP-LLM** | 2024 | arXiv | Multitask Reflective LLM for next POI | ✗ out of scope (LLM finetune) |
| **LSCNP** | 2026 | MDPI | BERT-guided spatio-temporal context | ✗ out of scope |
| **TLR-M (Huang)** | 2024 | arXiv | Mobility tree for time-slot preferences | ~ orthogonal signal (time-slot) |

### 3.2 Why GETNext is the strongest unused candidate

GETNext (Yang et al., SIGIR 2022) is the next most-cited next-POI baseline after STAN and is **directly applicable** to our setup:

- **Core contribution:** a *Trajectory Flow Map* — a global directed graph of POI-to-POI transition frequencies, built from the training set and used as a **learnable soft prior** on attention.
- **Architecture:** Transformer encoder over check-in sequence; flow-map embeddings added to POI embeddings at each step; category embeddings shared across the sequence.
- **Our data provides:** check2HGI's preprocessing artifact already contains a POI-POI transition graph at training time (it's what check2HGI's encoder is trained on). Region-level transition graph is a simple aggregation.

**Expected gain over STAN:** GETNext's graph prior explicitly encodes which regions-transition-to-which, which STAN must learn from scratch via attention. At the region task (~1.1 K classes on AL, 4.7 K on FL), this prior is more useful than a raw-attention baseline because many region pairs are rare.

**Cost to implement:** ~150 LOC. The graph is already built; we need to look up transition scores during forward.

### 3.3 Why ImNext is interesting but costly

ImNext (He et al., KBS 2024) adds:
- Irregular-interval attention (explicit Δt_ij between check-ins)
- Multi-task heads for next-POI + time-interval prediction

We don't have raw Δt in `next_region.parquet`. Building it requires extending the preprocessing pipeline — non-trivial (~400 LOC in `pipelines/create_inputs_check2hgi.pipe.py` + data pipeline). Deferred.

## 4 · Improvement ideas for our STAN implementation

Ordered by expected gain × cost:

### 4.1 Match d_model to backbone output (highest priority, cheapest test)

Change: pass `--reg-head-param d_model=256 --reg-head-param num_heads=8` when running MTL with STAN. No code changes required.

Expected: closes the dimensional bottleneck on AZ. Parameter count: ~1.2 M (vs 770 K GRU, 417 K STAN-128), still fair.

### 4.2 Pairwise bias initialization with recency prior

Change in `next_stan/head.py`:

```python
# Instead of std=0.02 Gaussian init:
with torch.no_grad():
    positions = torch.arange(seq_length).float()
    rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    for h in range(self.num_heads):
        slope = 1.0 / (2 ** ((h + 1) * 8.0 / self.num_heads))
        self.pair_bias.data[h] = -slope * rel_dist  # ALiBi-style
```

Expected: smaller effective DOF; stronger recency prior (newer tokens attended more by default). Same init as `next_transformer_relpos` for consistency.

### 4.3 Residual connection from input to classifier

Add a `self.input_res = nn.Linear(embed_dim, d_model)` skip path so the head output is `classifier(attn_output + input_res(last_input))`. Preserves the raw backbone signal as a fallback if attention layers disagree.

### 4.4 Attention-weighted mean pooling instead of last-position

Current: matching layer takes only the last position's output. STAN paper uses this for "target recall". For our 7-step-into-the-future region prediction, an **attention-weighted mean over all 9 positions** (with a learned query) may capture more context.

Cost: replace matching-layer with `AttentionPool1d(d_model, num_heads)`. Tested in `next_single` already.

### 4.5 Implement GETNext-style graph prior (biggest architectural bet)

Add `next_getnext` head:
- Load check2HGI's region-transition graph at construction time
- For each attention layer, add a bias `B_graph[i, j] = log(P(region_j | region_i))` (smoothed) to the attention logits
- Compare against STAN on AL/AZ/FL

Cost: ~150 LOC for the head + a ~30 LOC helper to load the transition matrix.

Expected: graph prior especially helps on the region task at the tail (rare regions). Could beat STAN on AL/AZ.

## 5 · Experimental plan

**Immediate (testable today):**

1. **Test 4.1 alone** — `--reg-head next_stan --reg-head-param d_model=256 --reg-head-param num_heads=8` on AL + AZ. Cost: ~1 h. Answers: is the bottleneck the whole story?
2. If AZ is fixed: celebrate, update docs, move on.
3. If AZ still regresses: proceed to 4.2 (ALiBi init) as a code change + rerun.

**Short-term (this week, if time allows):**

4. Implement 4.5 (GETNext head) and compare on AL + AZ.
5. Test 2.2 hypothesis: `--mtl-loss static_weight` instead of pcgrad.

**Deferred:**

6. ImNext-style Δt (needs pipeline extension).
7. Graph-flashback style repeat-visit module (needs behavioural labels).

## 6 · One-paragraph summary for the paper

> "We benchmark four candidate region heads inside our MTL framework: `next_gru` (the legacy choice, tuned for recurrence), `next_tcn_residual` (CNN-based), `next_stan` (Luo WWW'21, bi-layer self-attention), and (if implemented) `next_getnext` (Yang SIGIR'22, transformer + trajectory-flow-graph prior). STAN and GETNext beat the GRU ceiling in single-task evaluation; in MTL, the story is scale-dependent — at smaller scale (AL, 10 K) STAN-in-MTL exceeds GRU-in-MTL by +5 pp Acc@10, while at moderate scale (AZ, 26 K) the direction reverses because the d_model=128 projection from the 256-dim backbone output loses usable signal. With `d_model=256` matched to the backbone, **[TBD: result]**. We conclude that the MTL region ceiling is layered — shared-backbone capacity + head-backbone dimensional compatibility + head inductive bias — and the optimal head inside MTL does not equal the optimal standalone head at all scales."

## 7 · Follow-up checklist

- [ ] Run MTL-STAN AL with `d_model=256, num_heads=8`
- [ ] Run MTL-STAN AZ with `d_model=256, num_heads=8`
- [ ] Update `MTL_WITH_STAN_HEAD.md` with the hp-tuned numbers
- [ ] If hp tune fixes AZ: close the investigation
- [ ] If hp tune does NOT fix AZ: proceed to PCGrad × static_weight ablation (2.2)
- [ ] Decide whether GETNext (4.5) is worth 150 LOC for BRACIS timeline

## References

- Luo, Liu, Liu, *STAN*, WWW 2021 — [arXiv:2102.04095](https://arxiv.org/abs/2102.04095).
- Yang, Liu, Zhao, *GETNext*, SIGIR 2022 — [arXiv:2203.14583](https://arxiv.org/abs/2203.14583).
- He, Wang, Sun, *ImNext*, KBS 2024.
- Lim et al., *HMT-GRN*, SIGIR 2022 — positioning doc `POSITIONING_VS_HMT_GRN.md`.
- Prior MTL-STAN results: `MTL_WITH_STAN_HEAD.md`.
- STAN STL baseline: `SOTA_STAN_BASELINE.md`.
