# netcore — Paper Draft v0 (STORY + STRUCTURE lock)

> **STATUS — v0 SCAFFOLD, 2026-06-18.** This draft locks the **story and the structure**, NOT the
> numbers. Every model-derived number is marked **[provisional — regenerate on the frozen
> overlapping-window board]**. Per the user's critical decision (2026-06-18), the project adopts
> **overlapping windows** as a base change and executes ALL open experiments BEFORE final writing.
> Consequently *every windowing-dependent artifact* (sequences, per-fold transition priors, frozen
> folds, and all model runs) WILL be regenerated on the frozen overlapping-window base; the substrate
> embedding is per-check-in and windowing-independent (it can pre-stage now). Cells with no available
> number read **[TODO: P3 board]**.
>
> **Terminology discipline enforced throughout:** "SOTA" (never "SOAT"). The three prediction targets
> are kept strictly distinct — **next-POI** (not studied here), **next-category** (7-class macro-F1),
> **next-region** (~520–8,501-class Acc@10). This paper does NOT do next-POI.
>
> **Reconciliation note for future editors.** Two competing number sets exist in the repo and tell
> OPPOSITE region stories. (1) The frozen v11 BRACIS paper-canon (`docs/results/RESULTS_TABLE.md §0.1`,
> class-WEIGHTED CE) → the OLD "−7…−17 pp region cost / classic tradeoff." (2) The champion-G R0
> matched-metric bar (unweighted CE, v14 substrate, dual-tower) → "region PARITY + category lift /
> Pareto gain." **netcore adopts story (2); story (1) is explicitly SUPERSEDED** (decomposed into
> ~half a class-weighting confound + ~half config) and must NOT be cited as the current result.

---

## Title (provisional)

**Substrate Carries, Architecture Pays Nothing: A Check-in-Level Hierarchical-Infomax Substrate and an Orthogonal-Gradient Multi-Task Regime for Joint Next-Category and Next-Region Prediction**

*(Working title — netcore. Alternatives to weigh at camera-ready: "When Multi-Task Learning Is a Free Lunch: A Substrate-and-Regime Study of Joint POI Category and Region Prediction.")*

---

## Abstract (provisional, ~150 words)

We study joint multi-task learning (MTL) of two location-based social-network (LBSN) prediction tasks from raw check-ins: **next-category** (7-class macro-F1) and **next-region** (Acc@10 over hundreds-to-thousands of regions). We make four contributions. **(C1) The substrate carries.** A check-in-level hierarchical-infomax representation (Check2HGI: checkin→POI→region→city) lifts next-category macro-F1 by +14–29 pp over a POI-level baseline at matched head and compute, while being substrate-neutral on next-region; controls show the lift is *learned*, not feature injection, and is not transductively inflated. **(C2) MTL is a Pareto gain, not a tradeoff.** A single jointly-trained model beats the single-task category ceiling by +2.6–4.1 pp at *parity* on region (within 0.3 pp); the previously reported region "cost" was a class-weighting confound plus configuration. **(C3) The regime is orthogonal-gradient and frontier-negative:** the entire post-2022 MTL frontier replicates but does not exceed our champion. **(C4) External validity** holds directionally on a second LBSN corpus. *(All numbers provisional pending the overlapping-window rebuild.)*

---

## 1. Introduction

**The problem.** Mobile users leave check-in trails on LBSN platforms. Two downstream predictions matter for recommendation and urban analytics: the **next-category** a user will visit (a coarse 7-class semantic target) and the **next-region** they will move to (a fine spatial target with hundreds to thousands of classes). These are usually modeled separately. We ask whether a *single* model can serve both — one model, one forward pass, two heads — and at what cost. We deliberately do **not** model next-POI: the contribution is about the category/region pairing, which has no published headline precedent (the closest prior work treats category or region as *auxiliaries* to a next-POI main task — e.g. HMT-GRN/KGTB).

**The gap.** Prior LBSN MTL work is architecturally early (≈2018, low-capacity single-run, no single-task (STL) ceiling controls, no modern MTL-optimizer evaluation). The standard intuition — that jointly training an easy task (category) and a hard task (region) forces a tradeoff on the hard one — has, in our own earlier measurements, *appeared* to hold (a −7…−17 pp region cost). This paper shows that intuition is wrong here, and explains why.

**The single-model framing (the spine).** Our primary object is a *single jointly-trained model* satisfying **one model, one forward, N tasks**. Any deployment that breaks this property — task-conditional routing, a two-model composite, a multi-forward cascade — is treated as **supportive/diagnostic only**, never the headline; such a deploy concedes the MTL thesis. We will show the single model is not merely competitive but *Pareto-improving*.

**The four contributions.**

- **C1 — SUBSTRATE CARRIES (task-asymmetric substrate effect).** A check-in-level hierarchical-infomax substrate (Check2HGI) lifts **next-category** macro-F1 by **+14–29 pp** over a POI-level HGI baseline at matched head/compute across all states, while being **substrate-neutral on next-region** (the POI-level baseline is marginally ahead / statistically tied). The lift is hierarchical-infomax *learning*, not feature injection (a feature-concat control closes <10% of the gap), and is not transductively inflated (a train-users-only substrate rebuild reads ≈0 inflation on both axes).
- **C2 — MTL IS A PARETO GAIN, NOT A TRADEOFF (the headline reframe).** On a confound-free champion (unweighted CE, the v14 substrate, a cross-attention dual-tower with a *private* region tower), a single jointly-trained model **beats the STL category ceiling by +2.6–4.1 pp** while **matching the STL region ceiling within 0.3 pp** (parity, not a win) — measured at **AL/AZ/FL/GE** (4 seeds each); **CA/TX are expected-but-unmeasured**. The earlier "−7…−17 pp region cost" was **~half a class-weighting confound + ~half config**, not a representational tradeoff.
- **C3 — ORTHOGONAL-GRADIENT REGIME, FRONTIER-NEGATIVE (the LBSN MTL regime study).** The two task gradients are orthogonal on the shared trunk (cos≈0, tested intrinsic). In this regime no adaptive MTL optimizer helps, and the entire post-2022 MTL frontier (optimizers, gating, output-priors, asymmetric sharing, model-merging) **replicates but does not exceed** the champion. The win comes from *architectural asymmetry* (a private region tower + a shared encoder the category task harvests), not from balancing.
- **C4 — EXTERNAL VALIDITY (directional).** On a second LBSN source (Massive-STEPS: New York City + Istanbul, US and non-US), both defining champion behaviours replicate in a dry-run: MTL beats the STL category ceiling (+9–10 pp), matches the STL region ceiling (±1 pp), and clears a Markov-1 region floor.

**Positioning.** This is an **empirical substrate-and-regime study**, not a method paper. We do not claim a novel embedding method nor a general-ML-novel MTL architecture (the cross-attention trunk is MulT-family; the dual-tower is asymmetric PLE/STEM-family). We claim a *domain frontier*: the first rigorous substrate + MTL-regime study in LBSN POI prediction, with STL-ceiling controls, modern-optimizer evaluation, and confound decomposition.

---

## 2. Related work

> **Citation banner (TO-VERIFY before camera-ready).** Several framing citations carry verification
> flags from the source memos. **Kurin et al. NeurIPS'22** (*In Defense of the Unitary Scalarization*)
> and **Xin et al. NeurIPS'22** (*Do Current MTO Methods Even Help?*) support the
> "scalarization-suffices / parity-not-conflict" framing. The **Mueller TMLR** citation's venue, year,
> AND direction must be re-checked — an audit flagged its finding may lean toward "gradient conflict
> *does* affect generalization," opposite to how it is used in the C2/C3 memos. Do not ship the Mueller
> citation without re-reading the paper. [TODO: P3 board — finalize related-work citations.]

**2.1 Next-POI vs next-category vs next-region.** We separate the three targets explicitly. *Next-POI* prediction (Flashback, STAN, GETNext, etc.) is the dominant LBSN task and is **out of scope** here. *Next-category* is a coarse 7-class semantic target; *next-region* is a fine spatial target (≈520–8,501 classes depending on the city). The category+region-without-POI pairing as a *headline* MTL objective has no published precedent; the closest prior work uses category and/or region as **auxiliaries** to a next-POI main task.

**2.2 Location/check-in embeddings (incl. the head-on substrate competitor).** Static POI embeddings (POI2Vec, skip-gram-style) encode a fixed vector per POI. Hierarchical graph-infomax methods (HGI) operate at the POI level over a region hierarchy. Our substrate, **Check2HGI**, instead embeds at the *check-in* level (contextual per visit) over a four-level graph (checkin→POI→region→city). The head-on competitor is **CTLE** (a contextual check-in transformer embedding) — currently *absent* from our baseline table; the "why not CTLE?" question is unanswered until baseline B1 runs. We flag this as a known gap (see §4, §6).

**2.3 Mobility MTL.** Prior LBSN MTL is architecturally early-2018 (e.g. MCARNN), single-run, low-capacity, and lacks STL-ceiling controls and modern-optimizer evaluation. Where "MTL wins" are claimed in the next-POI literature, they typically reduce on inspection to conditional coupling (iMTL), input-side category features (GETNext), or coarse-to-fine cascades (CSLSL/CatDM lineage). **HMT-GRN** is the canonical next-region precedent (multi-task next-POI + next-region with equal-weight CE) and is reproduced as baseline B3.

**2.4 The post-2022 MTL-optimization consensus.** A line of work argues that tuned *scalarization* (static weighting) is competitive with or superior to adaptive multi-task optimizers in the small-task-count regime (Kurin/Xin, NeurIPS'22 — TO-VERIFY per the banner above). Our C3 result is a domain-specific replication of this consensus: at cos(∇)≈0 there is nothing for a balancer to resolve. We position C3 as the *first rigorous MTL regime study in LBSN POI prediction*, not as a general-ML optimizer claim.

---

## 3. Method

### 3.1 The Check2HGI substrate (check-in-level hierarchical infomax)

Check2HGI is a **check-in-level** hierarchical graph-infomax representation over a four-level graph: **checkin → POI → region → city**. Unlike POI-level HGI (one vector per POI), Check2HGI produces a **contextual embedding per visit**, so the same POI visited at different times/contexts receives different vectors. This is the property the next-category lift exploits (CH19: per-visit context accounts for 64–90% of the category gap).

The paper substrate is **v14 = `check2hgi_design_k_resln_mae_l0_1`**: a ResLN encoder, a Delaunay POI-POI GCN region lever, and a masked-POI MAE category lever, stacked orthogonally on the canonical Check2HGI module. The substrate embedding is computed *per check-in* and is therefore **windowing-independent** — it can be pre-staged before the overlapping-window rebuild. (Substrate training detail, graph construction, and the ResLN encoder spec: [TODO: P3 board — port from `docs/context/check2hgi_overview.tex` + `research/embeddings/check2hgi/`].)

### 3.2 MTLnet champion G (the single-model architecture)

**Champion G (= canon v16)** is the single jointly-trained model. One forward pass yields both predictions:

```
model                : mtlnet_crossattn_dualtower      # cross-attention shared trunk + two towers
cat head (task A)     : next_gru                        # category, 7-class
reg head (task B)     : next_stan_flow_dualtower        # region — PRIVATE STAN tower, additive (aux) fusion,
                                                        #          graph-prior coefficient α frozen OFF (alpha_init=0.0)
loss                  : static_weight, category_weight = 0.75   # tuned scalarization, k=2
class weighting       : UNWEIGHTED CE on BOTH heads     # the C25 fix — load-bearing
schedule              : OneCycle, max-lr 3e-3, 50 epochs, 5 folds, batch 2048
checkpoint selector   : geom_simple = sqrt(cat_macroF1 · reg_Acc@10)
task A input          : check-in embeddings (windowed)
task B input          : region embeddings (windowed)
```

**The single-model property (the spine).** Champion G is *one model, one forward, two tasks*. Its architecture *exploits* the orthogonal-gradient regime rather than fighting it: the region head lives in a **private, un-diluted tower** (so the hard task is insulated from the shared trunk — its learned shared-fusion coefficient β→0 at the large state FL), while the easy category task **harvests the shared cross-attention encoder**. Additive (`aux`) fusion adds the shared pathway without diluting the private tower; the biased α·log_T region prior is OFF. Architecture *capacity* is not the lever (MoE/CGC, SwiGLU, MulT, cross-stitch, deeper trunks were all null on region — falsified five ways).

**What is supportive/diagnostic only.** A two-model STL *composite* (route each task to its own single-task model at deploy) breaks the single-model property; once the class-weighting confound is fixed, its old +7–12 pp region edge collapses to +0.53 pp at FL while G wins category by +3.2 pp at roughly half the deploy footprint — so the composite is *dominated on the joint reading* and is reported only as a diagnostic ceiling.

---

## 4. Experimental setup

**States.** Six US states: **AL, AZ, FL, CA, TX** + **GE (Georgia)**. GE is a netcore-era state (it appears in the champion-G bar but has no legacy v11 number). *Scope:* the champion-G region-parity + category-lift result is currently measured at **AL/AZ/FL/GE** only; **CA/TX are expected-but-unmeasured** (no v14 substrate built yet; the CA/TX class-weighting A/B is hardware-infeasible on the A40 — TX OOMs, CA diverges at small batch). The `closing_data` plan builds CA/TX v14 and measures parity there. [TODO: P3 board — CA/TX cells.]

**Tasks & metrics.** *Next-category*: 7-class, **macro-F1**. *Next-region*: ≈520–8,501 classes per city, **Acc@10** (matched metric: FULL `top10_acc` on both the MTL and STL sides — never in-distribution-vs-full, the B-A2 correction). MRR/Acc@1/Acc@5 reported in the appendix.

**Splits.** User-disjoint cross-validation (StratifiedGroupKFold), 5 folds, so no user appears in both train and validation.

**Windowing — NEW base change (load-bearing).** This paper adopts **overlapping windows** (sliding windows that overlap, vs the prior non-overlapping windows). This is a **base change** decided 2026-06-18. *Dependency ordering:* the Check2HGI substrate embedding is per-check-in and windowing-**independent** (pre-stageable now); but **sequences, per-fold log_T region-transition priors, frozen folds, and all model runs are windowing-DEPENDENT** and must wait for the overlapping-window base to land — otherwise they are throwaway. **Every model number in this draft is therefore provisional** and will be regenerated on the frozen overlapping-window board.

**Seeds.** Reporting seeds **{0, 1, 7, 100}** (4 seeds). Seed 42 is the *development* seed and is NOT used for paper-grade numbers (development-seed contamination overshoots the canon by up to +8 pp at large states).

**The frozen base.** Once overlapping windows + the v14 substrate + champion G are pinned, the project performs a single full regeneration: STL baselines re-run + champion + suite cells, all states × 4 seeds × 5 folds (the `closing_data` re-baseline). The draft locks the recipe and substrate; the board fills the cells.

**Baselines.** Existing (faithful reproductions, windowing-dependent runs):
- *Next-category floors*: Majority-class; Markov-1-POI.
- *Next-category*: POI-RGNN (Capanema 2022; non-user-disjoint published caveat); MHA+PE; the Check2HGI substrate linear probe (C2HGI / HGI / Δ); our STL ceiling (matched-head `next_gru`, Check2HGI); our MTL row (champion G).
- *Next-region floor*: Markov-1-region.
- *Next-region*: STAN (faithful + stl_check2hgi + stl_hgi variants; AL/AZ/FL faithful, CA/TX scoped out); ReHDM (faithful; AL/AZ/FL only); our STL GRU ceiling; our STL STAN-Flow ceiling (matched-head reg ceiling); our MTL row (champion G).

**NEW netcore baselines** (B1–B5 — *implementation is windowing-independent and can proceed now; all baseline RUNS are windowing-dependent and wait for the overlapping-window base*):
- **B1 — CTLE** (contextual check-in transformer embedding) — the head-on substrate competitor, currently absent; answers "why not CTLE?".
- **B2 — POI2Vec / skip-gram** (static POI-embedding baseline).
- **B3 — HMT-GRN-style MTL** (next-POI + next-region multi-task, equal-weight CE — the canonical next-region precedent).
- **B4 — cascade** (category-then-location cascade, iMTL/CSLSL/CatDM lineage).
- **B5 — Flashback** (spatiotemporal-weighted RNN next-location baseline).

**External validity (Massive-STEPS).** A second LBSN corpus (NYC = 1,912 TIGER tracts; Istanbul = 520 mahalle, non-US). ETL is leak-free and bit-parity verified. Compare *gap-to-ceiling / lift-over-floor*, never absolute Acc@k (region counts differ). Phase V paper numbers (champion G + STL ceilings + Markov floor, 4 seeds, frozen substrate, CUDA) are PENDING; only a dry-run (1 seed, ResLN-80ep non-frozen substrate, MPS) exists today.

---

## 5. Results

> **All tables below are SKELETONS with PROVISIONAL/placeholder cells.** Numbers shown are from the
> pre-overlapping-window measurements and WILL be regenerated. Each is tagged accordingly.

### Beat 1 — The substrate carries (C1)

**Table T-substrate. Check2HGI vs HGI, matched-head STL, by state.** *[provisional — regenerate on the frozen overlapping-window board]*

| State | next-category macro-F1 Δ (C2HGI − HGI) | next-region Acc@10 Δ (C2HGI − HGI) |
|---|---|---|
| AL | **+15.50** (p=0.0312, 5/5 folds) | −2.71 (HGI marginally ahead) |
| AZ | **+14.52** (p=0.0312) | −3.13 |
| FL | **+29.02** (p=0.0312) | −2.12 (TOST-tied at δ=3pp) |
| CA | **+28.81** (p=0.0312) | −1.85 (TOST-tied at δ=2pp) |
| TX | **+28.34** (p=0.0312) | −1.59 (TOST-tied at δ=2pp) |

*Two-band lift: ~15 pp at small states, ~28–29 pp at large states. Region: substrate-neutral (HGI nominally ahead, statistically tied at the large states). Source: `RESULTS_TABLE.md §0.3`.*

**Controls (the lift is learned, not injected, and not inflated).** *[provisional]*
- **A2 feature-concat control** — HGI ⊕ Check2HGI's *exact* node features (category one-hot + hour/dow sin/cos) closes only **<10%** of the v14 category gap (concat lift +0.83…+2.02 pp; gap closed: AL 8.3% / AZ 7.1% / FL 2.4% — *shrinks as the lift grows*). Region: concat inert. → the lift is hierarchical-infomax learning, not feature access.
- **A4 transductivity bound** — rebuilding the substrate per fold on **train-users-only** (vs the full-corpus substrate) shows downstream inflation ≈0 on **both** axes (region −0.12…−0.33 pp; category POI-proxy +0.00…+0.29 pp; AL+FL, seed 0). → not transductively inflated.
- **CH19 per-visit counterfactual** — per-visit context = **64–90%** of the category gap (AL 72% / AZ 64% / FL 90% / CA 89% / TX 90%).

> *A2/A4 caveat: M4-Pro locally-rebuilt substrates (machine-drift, empirically spot-checked), on a
> subset of states/seeds; A4 category is a POI-level proxy on the in-coverage subset, not the exact
> check-in setup. The contextual/cold-POI residual is bounded by inductive-Check2HGI future work.
> **CTLE (B1) is not yet a baseline — "why not CTLE?" is open until B1 runs.***

### Beat 2 — MTL is a Pareto gain (C2 — THE HEADLINE)

**Table T-MTL-vs-STL. Champion G (single model) vs same-substrate STL ceilings, matched metric, 4 states × 4 seeds {0,1,7,100}.** *[provisional — regenerate on the frozen overlapping-window board]*

| State | G region Acc@10 | STL region ceiling | Δ region | G cat macro-F1 | STL cat ceiling | Δ category |
|---|---|---|---|---|---|---|
| AL | 62.57 ± 0.10 | 62.67 ± 0.13 | **−0.09** (matches) | 52.91 ± 0.27 | 50.35 | **+2.56** |
| AZ | 54.68 ± 0.24 | 54.80 ± 0.22 | **−0.12** (matches) | 54.48 ± 0.74 | 50.39 | **+4.08** |
| GE | 58.35 ± 0.04 | 58.44 ± 0.06 | **−0.09** (matches) | 61.43 ± 0.26 | 57.50 | **+3.93** |
| FL | 72.97 ± 0.06 | 73.27 ± 0.06 | **−0.31** (matches) | 73.16 ± 0.04 | 69.96 | **+3.20** |
| CA | [TODO: P3 board] | [TODO: P3 board] | [expected parity, unmeasured] | [TODO] | [TODO] | [expected lift, unmeasured] |
| TX | [TODO: P3 board] | [TODO: P3 board] | [expected parity, unmeasured] | [TODO] | [TODO] | [expected lift, unmeasured] |

*Headline: a single jointly-trained model **beats the STL category ceiling by +2.6…+4.1 pp** at **region PARITY** (within 0.3 pp). Source: `R0_matched_metric_bar.json` / `mtl_improvement/FINAL_SYNTHESIS.md §2`.*

**Decomposing the old "−7…−17 pp region cost" (the retired story).** *[provisional]*
- The SUPERSEDED v11 paper-canon (class-WEIGHTED CE, GCN substrate, B9 recipe) reported Δ_region: AL −11.04 / AZ −12.27 / FL −7.34 / CA −9.50 / TX −16.59 pp (all p≤2e-06, n=20). **This is NOT the current result** — it is being retired.
- **Controlled class-weighting A/B at FL** (flip *only* the class-weight flag; v11 GCN/B9 recipe; seeds {0,1,7,100}): unweighting recovers **+3.15 pp region** (63.91 → 67.06), moving the gap −6.71 → −3.56 — **halved, not closed**. The remainder closes under the confound-free champion G (v14 + dual-tower). CA/TX A/B is hardware-infeasible (predicted by C25 class-count scaling, not measured).

> *Honest disclosure (must keep): say region **PARITY**, never "beats." Parity is a confound-fix +
> champion-G-config result, not a single flag-flip. The +2.6…+4.1 cat lift is partly the C25
> cat-unweighting fix and partly head-config asymmetry (the STL ceiling uses a 2-layer GRU, dropout
> 0.3, logit-adjust τ=0.5; G uses a 4-layer GRU, dropout 0.1, plain CE) — a real DEPLOYABLE gain, not
> a pure "MTL beats STL" effect. The category beat is architecture-dominated: the cross-attn trunk
> alone gives +2.27 (FL) / +3.22 (AL) over the STL ceiling; genuine region→category transfer is only
> +0.93 (FL) / −0.67 (AL).*

### Beat 3 — Orthogonal-gradient regime, frontier-negative (C3)

**Gradient orthogonality (the mechanism).** cos(∇cat,∇reg) on the shared trunk ≈ 0: pooled **+0.0008** over 16 runs (4 states × 4 seeds, n=3,797 epoch-fold points); it **persists in a fully-shared model** (+0.0024 AL / +0.0017 FL) even where region's shared gradient dominates → intrinsic to the task pair, not manufactured by the architecture. *(Phrase as a FIRST-ORDER AVERAGE statement; Fifty'21 lookahead-affinity is the reviewer counter.)* *[provisional]*

**Table T-frontier-negatives. The post-2022 MTL frontier replicates but does not exceed champion G.** *[provisional]*

| Lever family | Representative method | Verdict |
|---|---|---|
| Adaptive MTL optimizers (~19 arms) | Nash-MTL, PCGrad, GradNorm, CAGrad, Aligned-MTL, RLW, … | NULL (none Pareto-beat tuned static weighting) |
| Output-level priors | log_T-KD, log_C co-location KD (R1), CrossDistil (R3) | NULL (log_T-KD saturates the family; cat-harmful at FL) |
| Asymmetric / learned sharing | STEM-AFTB (R2), GRM gated read (R10, fully mapped) | NULL (cos≈0 → nothing to gate) |
| Input-side conditioning | conditional coupling (cc) | **sub-threshold POSITIVE** (FL cat +0.235 / reg +0.070, 4/4 seeds — below the 0.3 pp promote gate) |
| Model merging | merge-vs-joint (R7) | Merge < joint G (ensemble loses the joint cat lift) |
| Residual optimizer | BayesAgg-MTL (R9) | craters at the champion recipe |

*Across 10 lever-families: **9 nulls + 1 sub-threshold positive**. No lever clears the ≥0.3 pp multi-seed promote gate; champion G is unchanged. A strong citable LBSN domain-frontier negative. Source: `mtl_frontier/FINAL_SYNTHESIS.md §1,§3`.*

> *Wording discipline: the optimizer negative is a CONVERGENT-EVIDENCE negative (defaults screen +
> targeted retunes + RLW + cos≈0 + literature), NOT "every method individually hyper-tuned." Frame as
> the WEAK-AUXILIARY regime: a 7-class (~2.8-bit) category auxiliary, far below the 180–300+ class
> vocabularies behind positive category-aux results in the next-POI literature. This is a domain-frontier
> claim, not a general-ML architecture-novelty claim.*

### Beat 4 — External validity (C4, directional)

**Table T-external-validity. Massive-STEPS dry-run (DIRECTIONAL — NOT paper numbers).** *[provisional — Phase V paper numbers PENDING]*

| Corpus (regions) | MTL beats STL **category** ceiling | MTL matches STL **region** ceiling | MTL beats Markov-1 region floor |
|---|---|---|---|
| **NYC** (US, 1,912 tracts) | MTL 54.0% vs STL 44.2% → **+9.8 pp** | MTL 29.8% vs STL 30.3% → **−0.5 pp** (matches) | +5.2 pp over floor 24.6% |
| **Istanbul** (non-US, 520 mahalle) | MTL 59.4% vs STL 50.4% → **+9.0 pp** | MTL 69.6% vs STL 68.6% → **+1.0 pp** (matches) | +17.0 pp over floor 52.5% |

*Both defining champion behaviours replicate on a second LBSN source, US and non-US. Source: `second_dataset/DRY_RUN_RESULTS.md`.*

> *DRY-RUN ONLY, do NOT carry these numbers forward: 1 seed (=42 dev seed), ResLN-80ep non-frozen
> substrate (NOT the frozen v14), MPS, reduced epochs; Istanbul substrate slightly under-trained.
> Compare gap-to-ceiling / lift-over-floor, never absolute Acc@k. Drop any temporal-bridge language —
> the shipped Massive-STEPS split is user-stratified RANDOM over trails, not temporal. Phase V paper
> numbers (frozen substrate, 4 seeds, CUDA, champion G + ceilings + Markov floor) are PENDING.*

---

## 6. Discussion

**The single-model property is the spine.** Everything in the paper is read against "one model, one forward, two tasks." C2 is a property of that single model. The composite and any multi-forward route are supportive/diagnostic — they concede the thesis, and once the confound is fixed they no longer even win (the composite's old reg edge collapses to +0.53 pp at FL).

**Why parity, not a region win, is the *expected* and *earned* outcome.** Theory and literature predict parity here: at cos(∇)≈0 the auxiliary contributes no first-order progress (Du'18); MTL gains concentrate in data-starved regimes (Bingel & Søgaard'17); k=2 tuned scalarization is unbeatable (Kurin/Xin'22 — TO-VERIFY); and a 7-class auxiliary is far below the vocabularies behind positive category-aux results. The claim is *earned*, not assumed: an X-series exercised every structurally-disabled MTL-only lever (all null), the numbers are pairing-safe, and the eval precision is clean. We say region **PARITY**, never "beats."

**Honest caveats (kept, not buried).**
- *Transductivity defused (A4):* full-corpus substrate training does NOT inflate downstream numbers (≈0 on both axes).
- *Substrate strengthened, not weakened, by the control (A2):* a wider-input HGI⊕exact-features baseline still leaves >90% of the gap unclosed — a conservative test the substrate passes.
- *Region is parity, the category lift is a head-config-aware deployable gain:* disclose the recipe/head-config asymmetry between G and the STL ceiling; the cat beat is architecture-dominated, not region→category transfer.
- *CA/TX scope:* parity is measured at AL/AZ/FL/GE; CA/TX are expected-but-unmeasured (no v14 substrate yet; the confound A/B is hardware-infeasible on the A40 and is C25-scaling-*predicted*, not measured).
- *Selector caveat (C21/CH23-B):* the SUPERSEDED v11 §0.1 used a JOINT-best selector that under-reports the substrate's real region capacity by ~10 pp at FL; champion G uses the corrected `geom_simple` selector. Any region conclusion from v11 §0.1 is selector-dependent and is not the current result.

---

## 7. Conclusion, limitations, and future work

**Conclusion.** For joint next-category + next-region prediction on LBSN check-ins, (C1) a check-in-level hierarchical-infomax substrate carries a large, *learned*, task-asymmetric category lift; (C2) a single jointly-trained model is a Pareto gain — it beats the STL category ceiling at region parity, dissolving the apparent tradeoff into a class-weighting confound plus configuration; (C3) the regime is orthogonal-gradient and frontier-negative — the modern MTL frontier replicates but does not exceed the champion, which wins by architectural asymmetry rather than balancing; and (C4) both behaviours hold directionally on a second LBSN corpus.

**Limitations.** We do not model next-POI. Region parity is established at four states; CA/TX are expected-but-unmeasured. CTLE (the head-on contextual-embedding competitor, B1) is not yet a baseline. The substrate is transductive in construction (A4 bounds the downstream inflation to ≈0, but an inductive variant is future work). External-validity numbers are a dry-run; Phase V paper numbers are pending. **All model numbers are provisional pending the overlapping-window rebuild.**

**Future work.** Inductive Check2HGI (cold-POI / out-of-corpus generalization, removing the transductive caveat by construction); beyond-parity region mechanisms ruled in by the regime analysis but untested here (conditional coupling, category-conditioned logit prior, semantic-ID/coarse-to-fine region-vocabulary factorization, a region→category consistency loss); Fifty-style lookahead-affinity hardening of the orthogonality claim; and a next-POI extension of the substrate.

---

## Appendix A — Pending before final numbers ("the board")

This v0 locks STORY + STRUCTURE. The following must complete before the numbers freeze, in dependency order. *The substrate embedding is per-check-in → windowing-independent (pre-stageable now). Sequences, per-fold log_T priors, frozen folds, and ALL model runs are windowing-dependent → they wait for the overlapping-window base, else they are throwaway. Baseline IMPLEMENTATION is windowing-independent; baseline RUNS are windowing-dependent.*

1. **Overlapping-windows base change** — adopt overlapping sliding windows; rebuild sequences, per-fold region-transition (log_T) priors, and frozen folds on the new base. *(Load-bearing; everything below depends on it.)*
2. **Frozen base regeneration (the `closing_data` §0 re-baseline)** — STL baselines re-run + champion G + suite cells, **ALL states × 4 seeds {0,1,7,100} × 5 folds**, including **building the CA/TX v14 substrate** so champion-G region-parity is *measured* (not just expected) at CA/TX.
3. **New external baselines B1–B5** — implement now (windowing-independent code), RUN on the frozen overlapping-window base: B1 CTLE (the head-on substrate competitor), B2 POI2Vec/skip-gram, B3 HMT-GRN-style MTL, B4 cascade, B5 Flashback.
4. **Massive-STEPS Phase V** — champion G + STL ceilings + Markov floor, 4 seeds, frozen substrate, CUDA, at NYC + Istanbul (replace the dry-run numbers).
5. **Re-fill every table cell** marked [provisional] or [TODO: P3 board]; re-run the statistical audit (paired Wilcoxon, matched metric, matched seeds/folds/precision) against the regenerated board.

*Verification flags to clear before camera-ready: the Mueller TMLR citation (venue/year/direction); the Kurin/Xin framing citations; the CTLE "why not" answer (B1); the CA/TX measured parity.*
