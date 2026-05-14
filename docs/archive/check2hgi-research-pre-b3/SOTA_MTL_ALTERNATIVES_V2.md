# Follow-up research: alternatives beyond the tested ablation families

**Date:** 2026-04-18. Triggered by the completed 4-family ablation showing shared-backbone capacity-ceiling property.

## The key strategic insight

> **Reframe the paper.** "STL beats MTL by 8 pp" is a weaker headline than what our decomposition shows. The real finding is:
>
> **5.4 pp of the apparent MTL penalty is pipeline-wrapper architectural overhead, not task interference.**
>
> Most MTL papers ignore this and attribute the whole gap to task conflict. Our λ=0.0 isolation measures it directly.

Proposed headline: *"Check2HGI as substrate for POI MTL: characterising the shared-backbone capacity ceiling and disentangling architectural overhead from task dilution."*

Keep STL-vs-MTL as a *table*. A "break the ceiling" experiment (direction 1 below) is the natural closing story.

## Five directions ranked by expected lift × cost

### 1. Cross-attention between task encoders (pre-backbone) — TOP PICK

**Claim.** Replace the single residual stack bottleneck with **two parallel encoder streams that cross-attend before the backbone**. Each task's queries attend to the other task's keys/values, producing task-conditioned representations that share *information* instead of *parameters*. FiLM modulates with scalars; cross-attention modulates with content.

**Why it targets our failure.** We already have per-task modality inputs and per-task encoders. Cross-attention is the natural generalisation of FiLM when the two streams carry genuinely different signals. The region stream can query "what category context is relevant here?" without being averaged into a shared vector.

**Anchors.**
- **MulT** (Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences", ACL 2019, arXiv 1906.00295) — canonical cross-modal transformer. Mechanism fits directly.
- **InvPT** (Ye & Xu, "Inverted Pyramid Multi-task Transformer for Dense Scene Understanding", ECCV 2022, arXiv 2203.07997) — task-interaction blocks in MTL dense prediction.

**Cost.** Medium. One `MultiheadAttention` per direction, no new hyperparameters beyond head count.
**Expected lift:** **3–5 pp on region Acc@10** if it works. **Highest-probability ceiling breaker.**

### 2. Separate towers + minimal shared regulariser

Inverting the architecture: mostly per-task, small shared piece as regulariser.

**Caveat.** The ceiling on lift here is ≈2.7 pp (the dilution component of our gap). This direction can't recover the 5.4 pp architectural overhead — the overhead is intrinsic to any shared-backbone formulation, and an inverted architecture still has one in miniature.

**Anchor.** Navon et al., "Auxiliary Learning by Implicit Differentiation" (ICLR 2021, arXiv 2007.02693).

**Use:** report as a *baseline* table row, not a lift vehicle. ≤2 pp expected.

### 3. Task-specific depth / prefix-mask AdaShare

Each task picks how many shared blocks to traverse (region uses all 4, category exits after 2). This is **AdaShare with monotone-prefix gates** — a variant we can test inside the AdaShare framework we're implementing.

**Cost.** Low marginal. Constrain `adashare_logits` so gates form a prefix mask.
**Expected lift:** ≤1.5 pp, mostly on category.

### 4. Contrastive pretraining of shared backbone

Pretrain the shared stack with a task-agnostic objective (masked trajectory modelling / next-check-in contrastive), then fine-tune with MTL heads.

**Anchors.**
- CACSR (Gong et al., AAAI 2023) — contrastive POI self-supervision.
- CTLE (Lin et al., AAAI 2021) — check-in sequence representation.

Both are *representation* papers, not MTL-backbone pretraining directly. Porting the objective is needed.

**Cost.** High (new training stage + data pipeline). **Expected lift 2–4 pp**, but reframes the paper from "MTL ceiling" to "pretraining recovers MTL ceiling" — bigger contribution, different paper scope.

### 5. Bi-level / task-priority optimisation

Outer loop on region loss, inner loop uses category as implicit regulariser via hypergradient. Matches NashMTL/PCGrad empirically. **Expected lift ≤1 pp**, de-prioritise.

## Direct strategic answer: is STL-vs-MTL still worth comparing?

**Yes, but not as the headline.** Keep the comparison as a table (it's necessary context), but the headline becomes the **decomposition**. Cross-attention (direction 1) is the "we tried to break the ceiling and here's what happens" experiment that gives the paper a positive close.

## Recommended next actions

1. **Complete AdaShare** (in progress — base MTLnet). Test prefix-mask variant too.
2. **Implement cross-attention MTL** (direction 1) — highest expected lift. ~200-300 LOC: a `CrossAttentionMTLnet` that replaces the FiLM+shared stack with `(enc_cat ↔ enc_next) × N` cross-attention blocks, then per-task heads.
3. **Skip directions 2, 4, 5** unless for paper completeness — they either can't close the overhead (2, 5) or are a different paper (4).
4. **Restructure the paper** around the decomposition finding. STL-vs-MTL table stays; it's no longer the headline.

## Sources (verify IDs before final cite)

- MulT: https://arxiv.org/abs/1906.00295
- InvPT: https://arxiv.org/abs/2203.07997
- Navon et al., Auxiliary Learning by Implicit Differentiation: https://arxiv.org/abs/2007.02693
- CACSR: check AAAI 2023 proceedings for exact ID
- CTLE: check AAAI 2021 proceedings for exact ID
