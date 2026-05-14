# Strategic framing — three user questions answered

## Q1: "Is STL-vs-MTL comparison worth doing?"

**Short answer:** Yes, but not as the headline. Keep the comparison as a supporting table.

**Why:** "STL beats MTL by 8 pp" is a mixed-to-negative headline. Our actual finding is stronger:

> **5.4 pp of the apparent MTL penalty is pipeline-wrapper architectural overhead, not task interference.** (Measured by the λ=0.0 region-only isolation.)

Most MTL papers comparing STL to MTL attribute the whole performance gap to task conflict. Our ablation *disentangles* the two components and quantifies them. That's a methodological contribution absent from prior POI MTL work (HMT-GRN, MGCL, etc.).

**Reframed paper headline (proposed):** *"Check2HGI as a substrate for POI MTL: characterising the shared-backbone capacity ceiling and disentangling architectural overhead from task dilution."*

**Paper structure (revised):**

1. **Primary substrate contribution (CH16):** Check2HGI check-in-level embeddings improve next-category F1 over POI-level HGI by +18.30 pp on Alabama (5f × 50ep, user-disjoint folds). Robust under fair evaluation: HGI loses 3.2 pp dropping from leaky to fair folds; Check2HGI loses only 0.6 pp. → "Check2HGI enforces user-agnostic generalisation."

2. **Architectural contribution (CH03):** Per-task input modality (check-in seq → category head, region seq → region head) is the only Pareto-bidirectional design. Shared modalities collapse the opposite head (shared_checkin → 2.3% reg Acc@10; shared_region → 20.2% cat F1). Concat is strictly dominated.

3. **MTL characterisation (new primary section):** MTL on Check2HGI is a task-asymmetric tradeoff. On FL (127K rows): cat +1.61 pp, reg −11.28 pp. The λ=0.0 decomposition localises the gap: 5.4 pp architectural overhead + 2.7 pp category-induced dilution.

4. **Ablation table:** 11-config ablation shows the capacity-ceiling property — none of 4 intervention families (6 optimizers, 5 loss weights, gated skip, MTLoRA rank sweep) closes more than 2 pp.

5. **Break the ceiling (future work + one experiment):** AdaShare per-task routing (testing) + cross-attention MTL (proposed) as directions that leave the shared-backbone inductive bias behind.

## Q2: "Is AdaShare worth testing even if we gain only 2 pp?"

**Short answer:** Yes. Running now.

**Why:**
- **Validates the characterisation.** If AdaShare *also* plateaus at +2 pp, it strengthens the capacity-ceiling claim: "5 intervention families tested, none breaks 2 pp — the ceiling is real."
- **Closes a research gap.** Protocol listed AdaShare; having not tested it leaves a reviewer question.
- **Cheap once implemented.** ~200 LOC of implementation (done — `src/models/mtl/mtlnet/model.py`), ~25 min training.
- **Even +2 pp is publishable.** Combined with cross-attention's potential +3–5 pp, we have a 5–7 pp composite lift that closes most of the dilution + some overhead.

## Q3: "Other architectures that may improve shared knowledge?"

Research synthesised five candidates (see `research/SOTA_MTL_ALTERNATIVES_V2.md`). Ranked by expected lift × cost:

| # | Direction | Anchor | Expected lift | Cost | Decision |
|---|-----------|--------|--------------:|------|----------|
| 1 | **Cross-attention between task encoders (pre-backbone)** | MulT 2019 + InvPT 2022 | **3–5 pp** | Medium (~200 LOC) | **TOP PICK — implement next** |
| 2 | Separate towers + small shared reg | Navon ICLR 2021 | ≤2 pp | Medium | Report as baseline row only |
| 3 | Task-specific depth (prefix-mask AdaShare) | AdaShare variant | ≤1.5 pp | Low (inside AdaShare framework) | Test if AdaShare passes |
| 4 | Contrastive pretraining | CACSR AAAI 2023, CTLE AAAI 2021 | 2–4 pp | High (new pipeline) | Different paper scope — defer |
| 5 | Bi-level task priority | Navon ICLR 2021 line | ≤1 pp | Medium | Skip (matches NashMTL) |

**Concrete recommendation for this paper's experimental arc:**

1. **Finish AdaShare** (step 5, in progress) — ~25 min. Expected ≤2 pp but validates ceiling.
2. **Implement cross-attention MTL** (new step 6) — ~4 h. Expected 3–5 pp. **The "break the ceiling" experiment.**
3. **Skip directions 2, 4, 5** unless for completeness.

### Why cross-attention specifically matches our failure mode

Our architecture already has per-task modality inputs and per-task encoders. The shared backbone forces a *parameter-averaged* representation both heads must share. Cross-attention between task encoders replaces this with *content-based* sharing: each task queries the other, producing task-conditioned representations that share information without sharing parameters.

Analogy: FiLM modulates the shared representation with a scalar per task. Cross-attention modulates with content. The latter is strictly more expressive and avoids the capacity bottleneck.

If cross-attention lands at +3–5 pp on region (57-58% vs current 49%), it closes 4–6 pp of the 8 pp gap — a real positive result for the paper.

## TL;DR action plan

1. ✅ Research complete (v2 saved).
2. 🟡 AdaShare running (step 5, ~25 min).
3. ⏭️ After AdaShare: implement cross-attention MTL (step 6, ~4 h).
4. ⏭️ After step 6: reframe paper around the decomposition finding. STL-vs-MTL stays as a table; the ablation becomes the characterisation; cross-attention (if it lifts) becomes the closing story.
