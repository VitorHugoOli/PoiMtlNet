# Project Positioning — Where This Repository Sits in the Literature

> Compiled 2026-06-12. Evidence base: repo inspection (file paths cited inline) + the literature survey in [`literature_review.md`](literature_review.md). This document is deliberately critical: it separates what is novel, what is incremental, and what is already known, and proposes the strongest defensible framing.

---

## 1. What this project actually is

Two coupled artifacts:

1. **Check2HGI** (`research/embeddings/check2hgi/`): a check-in-level (per-visit, contextual) embedding built by extending Hierarchical Graph Infomax (HGI, Huang et al., ISPRS 2023) from a static POI→region→city hierarchy to a **4-level check-in→POI→region→city hierarchy**. Check-in nodes carry category one-hot + hour/day-of-week sin/cos features; edges are user-sequence transitions with temporal decay; the encoder is a 2-layer GCN; POIs are aggregated from check-ins by multi-head attention pooling (PMA-style); training maximizes MI at three boundaries (c2p, p2r, r2c) with bilinear discriminators. Output: one 64-dim embedding **per check-in** (same POI ⇒ different vectors at different visits).
2. **MTLnet** (`src/models/mtl/`): a two-task MTL system for **next-category** (7 merged Gowalla classes, macro-F1) and **next-region** (TIGER census tracts, ~1.1k–8.5k classes, Acc@10) over 9-check-in windows, on five US-state Gowalla corpora, user-disjoint stratified 5-fold CV, 4 reporting seeds. The current champion **G** (`mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` reg tower + `next_gru` cat head, unweighted CE, static weighting) matches the STL region ceiling (−0.1…−0.3 pp) while beating the STL category ceiling by +2.6…+4.1 pp at 4 states (`docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md`).

The BRACIS 2026 paper ("Substrate Carries, Architecture Pays") currently tells the **v11** story: substrate lifts category +14–29 pp everywhere, region is substrate-neutral, and MTL pays a 7–17 pp region cost. **That last leg is now known to be confound-driven** (C25 class-weighting, `docs/CONCERNS.md §C25`) and is inverted by champion G.

> **Update 2026-06-12 (user decision)**: the v11 draft will be **dropped and rewritten almost from the ground up** on the v14 substrate + the new MTL architecture (champion G / v16+), with `docs/studies/closing_data/` as the experimental engine. This resolves the paper-contradicts-repo problem by decision. Until the rewrite lands, no v11-narrative claim should be cited externally; the analysis below is written for the *new* paper.

---

## 2. Novelty assessment, axis by axis

### 2.1 The embedding (Check2HGI) — novel as a combination; incremental in mechanism

**What the literature scan found** ([`literature_review.md §6–7`](literature_review.md)):

- Contextual per-visit embeddings exist: **CTLE** (AAAI 2021) — transformer over the trajectory, masked location + masked hour objectives; also CASTLE (2023), Geo-Tokenizer (ECML PKDD 2023). None uses a graph or an infomax objective.
- Hierarchy + infomax exists: **HGI** (ISPRS 2023) — but on static POI data, no mobility, region-level output. None of its 96 citing papers moves it to check-in granularity or recommendation.
- The intersection — *contextual check-in embedding from a hierarchical check-in→POI→region graph trained with infomax* — **is unoccupied as of 2026-06-12**. No scooping detected.

**Verdict**: the embedding is **novel as a combination, not as a mechanism**. Every component is borrowed (GCN encoder; PMA pooling from Set Transformer; DGI/HGI MI objective; sin/cos temporal features; the region encoder is literally reused from the HGI codebase — `research/embeddings/hgi/model/RegionEncoder.py`). The project's own claims registry already (correctly) forbids "We propose Check2HGI" as a headline (`docs/CLAIMS_AND_HYPOTHESES.md`, forbidden-claims list). What *can* be claimed: the first check-in-granularity hierarchical-infomax substrate, with evidence that per-visit context is the active ingredient (CH19, single-state).

**Two threats to the embedding claim, both currently unaddressed**:

1. **CTLE is not a baseline.** A reviewer's first question — "why is a hierarchical-infomax substrate better than CTLE's contextual transformer embeddings?" — currently has no empirical answer in this repo.
2. **The feature-injection confound.** Check2HGI's node features *are* category one-hot + time encodings, while the HGI comparator's features are POI2Vec vectors. The headline +14–29 pp category advantage may substantially reflect "the embedding carries explicit category/time features" rather than "hierarchical infomax learns something". The CH19 POI-pooled counterfactual (~72% of the gap is per-visit context, AL only) partially addresses this, but the cheap control — **HGI (or POI2Vec) embeddings concatenated with the same raw per-visit features, same head** — has never been run. Until it is, "substrate carries" is exposed.

### 2.2 The task pairing (next-category + next-region, no next-POI) — unpublished as a framing; needs an affirmative defense

The survey found **no published work with this pairing as the headline formulation**. Closest: KGTB (arXiv 2025) uses exactly category+region — as auxiliaries for generative next-POI; HMT-GRN (SIGIR 2022) supervises next-region at 5 geohash granularities — as auxiliary/search-space tools for next-POI; MCARNN/iMTL/CSLSL/HAMTL pair category with *POI-level* location.

**Verdict**: the framing is claimable, but the predictable reviewer attack is *"next-POI MTL with the main task amputated"* — every published use of these two tasks treats them as stepping stones to POI. The defense cannot be "nobody did it"; it must be affirmative:
- **Sparsity/well-posedness**: HMT-GRN's own data shows region targets are dramatically better-posed than POI targets at LBSN sparsity (97–98% vs 99.8% matrix sparsity); on thin state-level corpora (AL: ~10k check-ins), POI-level prediction may be statistically meaningless while region/category are not.
- **Deployment sufficiency**: geo-targeting, urban planning, demand forecasting need *where (coarse) + what kind*, not venue IDs.
- **Privacy**: category+region prediction avoids venue-level user profiling.
None of these arguments is currently articulated in `PAPER_DRAFT.md` with citations; they should be.

### 2.3 The MTL architecture — known components in general-MTL terms; a genuine *domain* frontier in LBSN

`mtlnet_crossattn` is MulT-adapted (Tsai et al., ACL 2019); the registry includes faithful/adapted Cross-stitch, MMoE, PLE/CGC-lite, DSelect-k (`src/models/mtl/*`); the loss registry covers ~19 published weighting/balancing methods (`src/losses/registry.py`). The champion's dual-tower design (private STAN tower for region, shared encoder harvested by category) is an asymmetric-privatization instance of ideas present in PLE (task-specific experts) and — most directly — **STEM's all-forward/task-specific-backward gating** (AAAI 2024, [arXiv:2308.13537](https://arxiv.org/abs/2308.13537)), which the industrial RecSys frontier converged on independently.

**Verdict, two-sided**:
- In **general-MTL terms**: not architecturally novel — every mechanism has a 2018–2024 antecedent, and the paper should not claim otherwise. The STEM/AdaTT/HoME convergence is *corroboration to cite*, not anticipation to fear (none of those works touches spatio-temporal/LBSN tasks).
- In **LBSN terms**: this *is* a frontier contribution. The published LBSN-MTL literature (MCARNN, iMTL, HMT-GRN, CSLSL) is architecturally early-2018 — hard sharing or cascades, static weights, no gradient diagnostics, no optimizer evaluation, no STL-ceiling controls. **No published LBSN work has run the post-2022 MTL research program**; this repo is, to our survey's knowledge, the first rigorous MTL *regime study* in LBSN POI prediction. That domain-frontier claim is defensible and should be made explicitly — see [`mtl_frontier.md §3`](mtl_frontier.md) for the full argument and the supporting citations.

### 2.4 The empirical findings — the actual contribution

Three findings are, in combination, a defensible empirical contribution:

1. **Task-asymmetric substrate effect** (CH16/CH15): per-visit contextual embeddings lift next-category massively (+14–29 pp, 5 states, head-invariant at ablation scale, p=0.0312 per state) while per-place embeddings tie or marginally win on next-region (TOST non-inferiority at CA/TX). Leak-audited, multi-seed, n=20/state. No published equivalent of this substrate-axis decomposition exists for LBSN tasks.
2. **The C25 confound autopsy**: the apparent "MTL sacrifices region" gap (−7…−17 pp) was an objective/metric mismatch (class-weighted CE trained vs unweighted Acc@10 reported), not negative transfer. This is a cautionary-methods finding with general value — analogous in spirit to Luca et al.'s protocol critiques — and it is fully documented (`docs/CONCERNS.md §C25`, `docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md`).
3. **Orthogonal-gradient regime + scalarization suffices**: pooled cos(∇cat, ∇reg) ≈ 0 on the shared trunk; all ~19 MTL-optimizer arms null; the win comes from architectural asymmetry (private reg tower + shared encoder for cat), yielding a single model that matches the STL region ceiling and beats the STL category ceiling at 4/4 tested states. This independently replicates Kurin et al. and Xin et al. (NeurIPS 2022) in the mobility domain — citable, and rare in this literature, where nobody even tries adaptive MTL optimizers.

**Caveats that bound these claims**: champion G is validated at AL/AZ/GE/FL only (CA/TX pending — exactly what `docs/studies/closing_data/` is scoped to close); the cat gain decomposition shows the +3 pp beat is architecture-dominated, with genuine region→cat transfer only +0.93 pp at FL and −0.67 pp at AL (`FINAL_SYNTHESIS.md` Finding 5) — so "MTL helps category" must be phrased as "the joint architecture is a better category encoder, with small transfer at scale"; CH19's mechanism evidence is single-state.

---

## 3. What is already known (and must not be claimed)

| Tempting claim | Why it's not claimable | Source |
|---|---|---|
| "We propose a novel MTL architecture" | Cross-attention MTL = MulT; dual-tower = asymmetric PLE-style privatization | §2.3 |
| "MTL beats STL on mobility tasks" | Known since MCARNN/iMTL/HMT-GRN — and in this repo the cat gain is mostly architecture, not transfer | Finding 5, `FINAL_SYNTHESIS.md` |
| "Adaptive MTL optimizers don't help" as a new discovery | Kurin et al. 2022, Xin et al. 2022 published the general result; this repo contributes a domain replication | [`literature_review.md §5`](literature_review.md) |
| "First contextual check-in embedding" | CTLE (AAAI 2021), CASTLE, Geo-Tokenizer precede it | §2.1 |
| "First use of region prediction for LBSN" | HMT-GRN (SIGIR 2022), DRRGNN (TKDD 2022) | §2.2 |
| Any cross-paper numeric comparison (Acc@k vs published tables) | Protocol incompatibility: random user-disjoint CV vs temporal splits; 7-class macro-F1 vs 200+-class Acc@k; census tracts vs geohash | [`evaluation_protocol_review.md`](evaluation_protocol_review.md) |

---

## 4. Strongest defensible framing

**An empirical substrate-and-regime study**, not a method paper:

> *"What does the embedding substrate contribute to joint next-category/next-region prediction — and what does joint training cost? A controlled five-state study."*
> Contributions: (C1) the first check-in-granularity hierarchical-infomax substrate, shown to carry next-category prediction (+14–29 pp) while being region-neutral — a task-asymmetric substrate effect; (C2) a confound autopsy showing the apparent MTL region cost was an objective/metric mismatch, and that after the fix a single asymmetric dual-tower model matches the STL region ceiling while exceeding the category ceiling; (C3) evidence that in this orthogonal-gradient regime, adaptive MTL optimizers are uniformly null and architectural asymmetry is what pays — replicating Kurin/Xin in the mobility domain.

This framing (a) survives the novelty audit, (b) converts the project's unusual honesty infrastructure (CONCERNS registry, leak audits, multi-seed protocol) into a selling point, and (c) does not require beating published SOTA numbers it cannot legitimately be compared with.

**Framings to avoid**: "novel embedding method" (mechanism is borrowed), "novel MTL architecture in general-ML terms" (it isn't — claim the LBSN domain frontier instead, §2.3), "SOTA next-POI" (the project does not do next-POI). The old title's implication that MTL intrinsically "pays" is superseded by champion G; since the paper is being rewritten from the ground up (§1 update), the new title should encode the inverted finding — substrate carries category, architecture *recovers* region.

---

## 5. Bottom line

- **Strongest asset**: the controlled, leak-audited, multi-seed substrate×architecture×objective decomposition across five states — nobody in this literature does ablations at this rigor — plus the first-in-LBSN MTL regime study (§2.3).
- **Weakest point** (post the ground-up-rewrite decision): external validity — single dataset (Gowalla), no temporal-split bridge, no CTLE comparison, no feature-concat control. These are now the binding constraints, not internal consistency.
- **The work is publishable at a national venue (BRACIS) once the ground-up rewrite lands on the v16+/closing_data base**; for an international venue (SIGIR/CIKM/KDD-track), it additionally needs the CTLE baseline, the feature-concat control, a protocol bridge, and the second dataset (see [`experiment_roadmap.md`](experiment_roadmap.md), [`future_work.md §8`](future_work.md)). The studies program also supports a companion lessons/negative-results contribution ([`community_insights.md`](community_insights.md)) and a forward MTL-gain program ([`mtl_frontier.md §4`](mtl_frontier.md)).
