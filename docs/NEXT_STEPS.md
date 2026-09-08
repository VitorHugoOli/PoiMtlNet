# NEXT_STEPS.md — Future-work command centre

> **What this is.** The single, analytical index of *where this line of work goes next* — both the
> thesis-facing scientific directions (feeding the dissertation's **§6.3 Future Work**, each tied to a
> §6.2 limitation) and the research/engineering backlog (the deferred levers in
> [`future_works/`](future_works/)). New ideas from ongoing thesis-review conversations land here first,
> get pressure-tested against what is already settled, then get routed.
>
> **What this is NOT.** Not a replacement for [`future_works/`](future_works/) (the self-contained memos
> stay the acceptance-criterion source of truth) and not a study tracker (that's
> [`studies/README.md`](studies/README.md)). This doc *ranks and argues*; the memos *specify*.

**Maintained by:** Vitor + agent, during the 2026-07 thesis-review pass.
**Last updated:** 2026-07-20.

---

## 0 · How to use this doc

1. **Adding an idea** → drop it in [§5 Parking lot](#5-parking-lot--new-ideas-inbox) as a one-liner.
   The agent then (a) checks it against the *settled-and-closed* ledger (§4) so we don't re-propose a
   falsified direction, (b) researches external validity if it's a novel method claim, (c) assigns a
   track + priority, and (d) promotes it to §2/§3 with a status line.
2. **Reading for the thesis** → §2 is the ranked list that feeds dissertation §6.3. Every item there
   names the §6.2 limitation it discharges.
3. **Reading for research** → §3 is the engineering backlog, ranked by *expected value given the wall*
   (see §1). Most items here are gated on the regime finding.

**The one discipline (from the dissertation NORTH_STAR §6):** a future-work item that does not tie to a
concrete limitation or a concrete open lever is noise. Every entry below has either a **Limitation tie**
(thesis track) or a **Gate** (research track).

---

## 1 · The wall — what any future work must respect

Before ranking anything, the load-bearing settled facts. Re-proposing work that ignores these is the
single most common failure mode when brainstorming from a fresh read of old memos.

| # | Settled fact | Consequence for future work | Source |
|---|---|---|---|
| W1 | **The MTL cross-attn joint-training regime washes out substrate/encoder gains on both tasks.** Even HGI (the STL reg winner) ≈ canonical in MTL; the four design substrates close 0 % of the MTL gap. | Any "better embedding → better MTL" idea is **dead on arrival unless it also changes the joint-training regime**. Substrate work is STL/representation-quality only. | `substrate-protocol-cleanup` CLOSURE; NORTH_STAR "regime finding" |
| W2 | **The champion is already a single joint model that beats both STL ceilings** (G = v16 → v17): cat +3 pp at all states, reg *matches* the STL ceiling (Pareto-non-inferior). The MTL tradeoff is inverted, not negative. | "Make MTL non-negative" is **already done**. New MTL work must beat *G/v17*, not STL. The bar moved. | `mtl_improvement` FINAL_SYNTHESIS; NORTH_STAR v16/v17 banners |
| W3 | **The residual reg gap (where it exists) is architectural** — localized to the joint cross-attn harness, not interference / prior / weight-decay / substrate / long-tail. Falsified 5 ways. | Loss-balancer and substrate tweaks are polish. Only *architecture* or *prior-pathway* (log_T-KD) levers have ever moved MTL reg. | P4 verdict; `mtl_improvement` Tier 2 |
| W4 | **The C25 unweighting fix + the fp16→bf16 harness fix** dissolved the two biggest apparent "MTL costs." Several older memos were written *before* these fixes. | Treat any pre-2026-06-05 memo's magnitudes as suspect; re-baseline before citing. | CONCERNS §C25; `mtl_fp16_autocast_rootcause` |
| W5 | **The single-model property is the thesis headline.** Any deploy that breaks "one model, one forward, N tasks" (composite, C1 routing, dual-substrate) is *supportive/diagnostic at most*. | Composite-engine and routing work can improve *numbers* but **cannot be the headline** — it concedes the MTL thesis. Frame accordingly. | memory `feedback_single_model_property_primary` |

**Reading:** the interesting frontier is no longer "does MTL help" (settled: yes, W2) or "which
embedding" (settled: regime-bound, W1). It is **(a) generalization beyond the current data/task regime**
(the thesis limitations) and **(b) whether a *different joint architecture* can convert the reg "matches"
into a reg "beats"** (W3). Everything else is polish.

---

## 2 · Thesis track — scientific directions (feeds dissertation §6.3)

Ranked by scientific weight for the defense. Each ties 1:1 to a §6.2 limitation. These are the items that
belong in the *written* future-work section — broad, honest, limitation-anchored.

| Rank | Direction | §6.2 Limitation it discharges | Maturity | Notes |
|---|---|---|---|---|
| T1 | **Exact next-place prediction** (predict the next POI id, not just category+region) | *No next-place task* | Idea | The natural completion of the task hierarchy: category → region → place. The whole framework is built to slot a third head. Highest reviewer-expected item; name it explicitly. |
| T2 | **Inductive / transferable representations** (replace the transductive Check2HGI substrate with one that generalizes to unseen POIs/users/cities) | *Transductive-representation caveat* | Idea + partial evidence | Directly attacks the substrate's biggest weakness. Cross-city transfer (train Gowalla → test Istanbul) is the concrete experiment. Ties to T4. |
| T3 | **Newer / denser mobility traces** (beyond Gowalla 2009–2010) | *Dataset vintage 2009–2010* | Idea | Validity/recency. Cheap to state, expensive to execute (data access). Frame as "the method is data-agnostic; the evidence base is a decade old." |
| T4 | **Broader geographic + cultural coverage** (more non-US cities beyond Istanbul) | *Single non-US city* | Idea | Complements T3. The Istanbul result already gestures at generality; make it a program. |
| T5 | **Finer category taxonomy** (beyond the 7-category Foursquare-derived scheme) | *7-category taxonomy* | Idea | The coarse taxonomy inflates category accuracy and limits downstream utility. A finer taxonomy is a harder, more useful task. |
| T6 | **Matured cascade coupling** (region → place, or category → region as an explicit cascade rather than parallel heads) | *Single-model constraint* framing | Idea + prior art | The "one model" can become a principled cascade. Related to the composite work (W5) but framed as *architecture*, not deploy-time routing, so it keeps the single-model property. |

> These six are already the spine named in dissertation NORTH_STAR §6 ("exact next-place; newer/denser
> traces; inductive representations; matured cascade coupling") plus the two taxonomy/geography items
> implied by §6.2. **§6.3 should not exceed this set** — a limitations-tied future-work section that stays
> tight reads as disciplined; a sprawling one reads as a wish-list.

---

## 3 · Research track — engineering backlog (the `future_works/` memos, ranked)

These are the deferred *levers*. Ranked by **expected value given the wall (§1)**, not by how much work
exists. Status column says whether the lever is still live or has been overtaken.

| Rank | Lever | Memo | Gate / status vs the wall | Verdict |
|---|---|---|---|---|
| R1 | **MTL architecture revisit** (faithful MMoE / CGC / DSelect-K / cross-stitch × cross-attn hybrids, per-task eval) | [`mtl_architecture_revisit.md`](future_works/mtl_architecture_revisit.md) | The *only* lever W3 leaves open: architecture is the one axis that moves MTL reg. Must beat **G/v17**, not STL. | **Live — highest-value research item.** This is the one that could turn reg "matches" → "beats". |
| R2 | **Paper-canon re-evaluation** (n=20 multi-seed §0.1 under the winning selector+arch, all 5 states) | [`paper_canon_reevaluation.md`](future_works/paper_canon_reevaluation.md) | Pay the publication-revision cost *once*, after R1 lands. Sequenced strictly after R1. | **Live but downstream** — do not run until R1/selector settle. |
| R3 | **Loss-combination study** (loss-scale normalization ÷log(C) + RLW litmus → Uncertainty Weighting / FAMO / DB-MTL) | [`joint_selection_and_loss_combination.md`](future_works/joint_selection_and_loss_combination.md) §Part 2 + [`substrate_adaptive_mtl_balancing.md`](future_works/substrate_adaptive_mtl_balancing.md) | W3 says balancers are polish, BUT loss-scale normalization ÷log(C) is a *cheap, literature-backed* first probe that was never cleanly run leak-free. | **Live, low-cost probe only.** Run the ÷log(C) + RLW litmus; defer the rest unless it moves. |
| R4 | **Per-head-LR × OneCycle fix follow-through** (the v16→v17 bug: scalar max_lr broadcasts, inert per-head LRs) | [`per_head_lr_onecycle_fix.md`](future_works/per_head_lr_onecycle_fix.md) | Already *landed as v17 default* at AL/AZ/FL; CA/TX n=20 still completing the board. | **Mostly done — finish CA/TX, then close.** |
| R5 | **Reg-head architecture sweep** (`next_stan_flow` vs `next_getnext` vs lstm/transformer_pf/gru) under per-fold log_T | [`reg_head_architecture_sweep.md`](future_works/reg_head_architecture_sweep.md) | Narrow slice of R1. Rolls into R1 if R1 launches. | **Merge into R1.** Not a standalone. |
| R6 | **Head / window / batch audit** (causal-mask correctness, class-balance form) | [`head_window_batch_audit.md`](future_works/head_window_batch_audit.md) | Diagnostic/variance-source hygiene. Sampler form already falsified; focal-only variant queued. | **Live as an audit, not a claim.** Co-schedule with R1. |
| R7 | **Cross-stitch × aligned-pairing** — the one *live positive lead* buried in the pipeline-audit memo: faithful cross-stitch (Misra α-matrix) is the **first architecture to show a positive aligned-pairing effect** (+0.60 reg vs −3.68 for full cross-attn), seed-0 only | [`pipeline_audit_quality_followups.md`](future_works/pipeline_audit_quality_followups.md) §7 | W3-compatible: it's an *architecture* lever, and it's the only aligned-pairing signal that hasn't washed out. Needs multi-seed {0,1,7,100} at AL/AZ vs a champion-grade comparator. | **Live — promote to a proper R1 sub-cell.** The rest of the memo (items 1–3 cond_coupling/G0.1) tested **null and closed**; only input-builder hygiene remains. |
| R8 | **MIN_SEQUENCE_LENGTH 5→2 probe** (residual lever after overlapping windows was adopted) | [`overlapping_windows.md`](future_works/overlapping_windows.md) | Overlapping windows (stride-1) is **already ADOPTED** — the champion engine `check2hgi_dk_ovl` is stride-1, MIN_SEQ=10, and every RESULTS_BOARD §1 number is overlap-based. The only open sub-lever is lowering MIN_SEQ (currently drops ~58 % of AL users). | **Mostly done.** Only the MIN_SEQ floor is open, and it's frozen-gated (redraws the split). |
| R9 | **Composite two-substrate engine** (STL cat ⊕ STL reg routed by task) | [`composite_two_substrate_engine.md`](future_works/composite_two_substrate_engine.md) | W5: breaks the single-model property → **cannot be a headline**. Diagnostic ceiling only (AL/AZ scored; strictly dominates MTL on reg at zero cat cost). | **Held.** Keep as the deploy-ceiling reference, not a direction. |
| R10 | **POI-decoder HGI distillation** (speculative substrate-axis re-open) | [`poi_decoder_hgi_distill.md`](future_works/poi_decoder_hgi_distill.md) | W1: substrate axis is regime-washed. Explicitly a "standby, runs only if composite is rejected." | **Standby.** Lowest priority. |
| R11 | **Substrate-adaptive MTL balancing** (NashMTL revival etc. under per-fold log_T) | [`substrate_adaptive_mtl_balancing.md`](future_works/substrate_adaptive_mtl_balancing.md) | Scope depends on R1 residual. If R1 closes the gap this is dead; if >2 pp residual it becomes load-bearing. Registry already has ~10 balancers → no new infra. | **Conditional on R1.** |
| — | **Task-pivot rationale merge** (doc polish) | [`task_pivot_memo.md`](future_works/task_pivot_memo.md) | Pure documentation debt — merge the historical "why" into NORTH_STAR/paper prose. | **Polish, not research.** |
| — | **Selector upgrades** (Pareto-restricted argmax + smoothed trajectory) | [`joint_selection_and_loss_combination.md`](future_works/joint_selection_and_loss_combination.md) §Part 1 | Core geom_simple selector already landed (C21). Rest is reviewer-proofing. | **Optional polish.** |

### 3.1 · Research → thesis bridge

Only two research items are *also* thesis-facing:
- **R1 (architecture revisit)** → if it converts reg "matches" → "beats", it strengthens the MobiWac
  chapter's headline and could seed a post-defense paper. Even null, it's a defensible §6.3 line ("we
  probed N joint architectures; the cross-attn harness remains the reg ceiling — closing it is open").
- **R9 / T6 (cascade/composite)** → the *scientific* version of composite is **T6 matured cascade
  coupling** (keeps single-model property). The *engineering* version (R9) is diagnostic only. Keep them
  separate so the thesis never cites a single-model-breaking deploy as future work.

---

## 4 · Settled-and-closed ledger — do NOT re-propose these

The graveyard. When an idea in §5 looks new, check it here first.

| Closed direction | Why it's dead | Source |
|---|---|---|
| "Better embedding lifts MTL" | Regime washes it out (W1). STL-only value. | `substrate-protocol-cleanup` |
| "MTL sacrifices reg; make it non-negative" | Dissolved by C25 fix; champion G already inverts the tradeoff (W2). | `mtl_improvement` |
| Reg-private *and* cat-private dual-tower (G′) | Cat craters −3.6…−15.3 pp at small states; no rescue config. Cat head stays shared `next_gru`. | CONCERNS §C26 |
| Capacity scaling (MoE / SwiGLU / MulT / wider backbone) as the reg lever | Falsified 5 ways; architecture *capacity* is not the lever (the STAN private tower is). | `mtl_improvement` Tier 2 |
| Class-balanced sampler for cat imbalance | Falsified (Phase 3 Rank 2). Focal-only variant still queued (R6). | mtl-protocol-fix Phase 3 |
| Aligned cross-task pairing as the default | Aligned = memorization shortcut; random = augmentation. Pairing is a *finding*, not a lever to turn on. | `research/pairing_science.md` |
| Broken F1-mean joint selector | Fixed → geom_simple default (+5.6 pp FL). | CONCERNS §C21 |
| "MTL+HGI catastrophically breaks reg (−37 pp)" | Leak artifact; retracted. Normal 3–6 pp MTL cost under leak-free. | NORTH_STAR retraction 2026-05-16 |
| BRACIS "MTL pays 7–17 pp on region" | fp16-harness artifact + old protocol; corrected by MobiWac. | `mtl_fp16_autocast_rootcause` |
| Dual-substrate MTL routing (HGI reg tower into MTL) | Pilot-falsified 2026-06-02: routed reg-Acc 47.17 ≈ v14-only 47.18 ≈ canonical 47.15. The regime, not the substrate/routing, is the wall (W1). | `part2_mtl_dual_substrate_routing.md` |
| cond_coupling × aligned-pairing / G0.1 binding | Tested null on champion (2026-07-02); de-confounded, no move ≥0.3 pp. | `pipeline_audit_quality_followups.md` items 1–3 |

---

## 5 · Parking lot — new ideas inbox

> Drop raw ideas here as one-liners with a date. The agent triages each into §2/§3, checks it against §4,
> researches external validity where needed, and records the verdict inline. Nothing is dismissed silently.

> **Status of the two entries below (2026-07-20):** described at plan altitude per the author's call —
> *"nesse primeiro momento vamos só descrever a ideia como plano futuro; em outro momento elaboramos"*.
> Both literature-research threads (LLM-POI enrichment; STHGCN-style hypergraph substrate) **have landed** —
> their novelty/white-space verdicts are folded into the **External validity** fields below (full reports
> live in two research artifacts). **Do not treat these as scoped studies yet** — no acceptance criteria,
> no experiment design; elaboration deferred by author decision.
>
> **One-line comparison:** both are legitimate thesis-facing *representation* directions (not MTL-optimizer
> plays — W1 still bounds their MTL-reg value). **Idea 1 (LLM) is the sharper, lower-risk bet** — near-empty
> white space (inductive semantic POI vector), cheaper, one clean gap to exploit (POI-Enhancer never tests
> unseen POIs). **Idea 2 (STHGCN-style hypergraph) is bigger and riskier** — defensible novelty in an
> unclaimed intersection, but the accuracy payoff on *our* Gowalla data is an open bet (the plain GNN
> sometimes wins on segmented Gowalla). Both share the same headline motive: **inductive / cold-start
> generalization** (limitation T2), which is the axis where substrate still matters despite W1.

### [2026-07-20] Idea 1 — LLM-enriched POI embeddings

- **Raw idea:** Use LLMs to enrich POI embeddings — inject semantic/world-knowledge (from POI names,
  descriptions, category text, reviews) into the representation instead of relying only on the
  co-visitation graph.
- **Track:** **thesis** (§2). This is a *representation-quality* play, which is the dissertation's spine —
  not an MTL-optimizer play.
- **Checked against §4:** Not a re-proposal of any dead substrate. It is *not* "another graph embedding"
  (W1 kills those for MTL); its value is orthogonal — textual semantics the co-visitation graph cannot
  see. **But W1 still applies to the MTL-reg story**: if pitched as "LLM substrate → better MTL," it will
  wash out. Pitch it as **STL / representation-quality / cold-start**, never as an MTL-reg lever.
- **Limitation tie:** primarily **T5 (7-category taxonomy)** — LLMs infer finer semantic structure than a
  7-class scheme — and **T2 (transductive caveat)** — a text-derived POI vector generalizes to POIs never
  seen in the training graph (inductive / cold-start). Secondary touch on **T3 (dataset vintage)**: LLM
  world-knowledge partly compensates for decade-old traces.
- **External validity (research landed 2026-07-20):** the white space aligns *exactly* with a
  Check2HGI-based contribution.
  - **Closest prior art = POI-Enhancer (AAAI 2025)** — essentially the only purpose-built "inject LLM
    semantics into an *existing* POI embedding" method: +22.4% Hit@1 across 6 base embeddings. **But it
    derives POI text from historical check-ins and never tests unseen POIs** → a clean, citable gap on the
    inductive axis. Supporting: NextLocLLM (coordinate+text form is the clearest inductive/cross-city
    evidence — ID baselines ≈ 0), Mobility-LLM (few-shot), CaLLiPer, GNPR-SID (quantized semantic IDs).
  - **Field state:** *LLM-as-predictor* (LLM4POI SIGIR 2024, +23% over STHGCN NYC, +134% on inactive users;
    Refine-POI) is **crowded/saturated**. The *"LLM produces a POI vector that makes a transductive graph
    inductive"* room is **nearly empty** — that is the opening.
  - **⚠ Load-bearing risk to design against:** Luca et al. (Springer ML 2023) — **43–72% train/test
    trajectory overlap**; models score ≤5% on truly novel trajectories vs ≥90% on memorized ones. Plus
    **LLM pretraining contamination** (a second leakage channel — the LLM may have seen these exact POIs)
    and geographic/socioeconomic bias. Any evaluation must be on genuinely unseen POIs, leak-audited.
  - Caveat: absolute Acc@k are **not** comparable across eval regimes (full-ranking vs 100-negative vs
    grid-cell) — do not cross-cite raw numbers. Full report: research artifact (LLM-POI thread).
- **Verdict + placement:** **promoted to the thesis track — the sharpest of the two ideas.** Frame:
  "an LLM-derived semantic POI vector that makes the transductive Check2HGI substrate inductive/cold-start
  capable" → discharges **T2** primarily, **T5** secondarily. Genuinely open in 2026. Elaboration deferred
  per author; the leakage protocol (Luca 2023 + pretraining contamination) is the first design constraint
  when it's picked up.

### [2026-07-20] Idea 2 — HGI ⊕ STHGCN-style spatio-temporal hypergraph substrate (cohesive multimodal fusion)

- **Raw idea:** Build a **new, more cohesive POI embedding** that borrows the STHGCN mechanism
  (Yan et al., SIGIR 2023 — spatio-temporal *hyperedges over trajectories* + hypergraph propagation) and
  fuses the best of each encoder — **POI2Vec** (geo latent), **Sphere2Vec** (spherical-distance-preserving
  coordinate encoder, already in-repo), **Time2Vec** (periodic temporal) — into ONE representation, instead
  of today's loosely-stitched
  components in Check2HGI. **Explicitly a hybrid, not a replacement:** part of the idea is to *evaluate how
  to keep HGI (hierarchical graph infomax) AND the STHGCN-style spatio-temporal hypergraph together* —
  hierarchical self-supervised structure + trajectory-level spatio-temporal higher-order structure.
- **Framing correction (author, 2026-07-20):** Check2HGI is a **task-agnostic, reusable self-supervised
  substrate**, not a category/region-only embedding. The goal is a *general* POI vector usable for ANY
  task, **including exact next-POI**. So this is a **foundation-substrate** play — adopt STHGCN's
  *mechanism* inside a *self-supervised, reusable embedding*, NOT build a supervised next-POI recommender.
- **Track:** **thesis** (§2), foundation-substrate. Also feeds **research R-new** once elaborated.
- **Checked against §4:** *Not* the falsified dual-substrate routing (that was HGI-reg routing inside MTL).
  *Not* the poi-decoder HGI distill (R10, a small aux-loss). This is a genuinely different substrate
  *architecture*. **W1 is the standing threat:** as an MTL-reg lever it will likely wash out. Its
  legitimate value is **STL representation quality + generality + enabling next-POI**, which is exactly
  where substrate still matters.
- **Limitation tie:** **T1 (no next-place task)** — a stronger general substrate is the natural enabler of
  exact next-POI — and **T2 (transductive caveat)** if the hypergraph design is made inductive. The
  HGI-⊕-STHGCN coexistence question is itself the research contribution.
- **External validity (research landed 2026-07-20):** **novelty defensible; value is the open bet.**
  - **STHGCN + its whole lineage are supervised single-task next-POI recommenders** (STHGCN SIGIR'23:
    trajectory-grained hyperedges = one hyperedge per trajectory, hypergraph transformer with
    spatio-temporal context modulating attention; then DCHL SIGIR'24, HyperMAN ICME'25, ReHDM IJCAI'25,
    MSAHG AAAI'26). **None is a self-supervised reusable embedding** — so the author's framing (borrow the
    *mechanism* into a self-supervised substrate kept alongside HGI) sits in **unclaimed space**. ReHDM &
    MSAHG are the only two modeling the region axis (nearest neighbors to next-region).
  - **The genuine white space is the intersection** *{self-supervised reusable} × {trajectory hypergraph}
    × {cohesive geo+coord+time fusion}* — **unclaimed as of 2026.** Nearest competitor **AdaptGOT (2025)**
    is self-supervised/reusable/multi-task but uses *plain* POI graphs (not hypergraphs), no continuous
    coordinate encoder, no continuous time. No published work co-embeds Time2Vec + a coordinate encoder +
    POI2Vec geo-latents in ONE (hyper)graph pass — the norm is GETNext-style dense-fuse-then-concatenate.
  - **Spatial encoder = Sphere2Vec (author's choice, 2026-07-20, swapped in for SIREN).** Sphere2Vec
    (Mai et al., ISPRS J. 2023) is an *established* POI-lineage encoder that preserves **spherical**
    distance — the principled choice for lat/lon over large geographic extents and the strongest fit for
    the **cross-city / inductive** angle (planar encoders distort at scale). **It is already implemented
    in the repo** (`research/embeddings/sphere2vec/`, sphereM variant) → near-zero implementation risk vs
    SIREN. *Trade-off:* Sphere2Vec is the incumbent norm, so it is **not itself a novelty hook** (SIREN
    would have been — it has zero POI use — but is untested/risky). This concentrates Idea 2's novelty
    where the research says it actually lives: the **cohesive self-supervised hypergraph fusion**, not the
    coordinate encoder. Net: a *better-motivated and de-risked* spatial choice at the cost of one
    speculative-novelty bullet — the right trade for a thesis future-work direction.
  - **Inductive angle = the strongest scientific motive.** HGI/Check2HGI as deployed is transductive
    (fixed per-state POI graph; a new POI/state needs a rebuild). A feature-based aggregator — which
    Sphere2Vec coords + Time2Vec + category *naturally* enable for zero-history POIs — is the redesign's
    headline motive. **Frame it as a limitation you *address*, not one you *discovered*.** (Sphere2Vec's
    spherical-distance preservation is an *asset* here: it degrades more gracefully across unseen regions
    than a planar encoder.)
  - **⚠ Load-bearing risk on OUR data:** hypergraph gains are **dataset-dependent** — STHGCN beats GETNext
    on Foursquare but **the plain GNN wins on *segmented Gowalla***, and DCHL loses on TKY. Gowalla is our
    primary dataset, so the higher-order bet is *not* guaranteed to pay off on it. Also: no published
    DGI-vs-hypergraph-infomax next-POI benchmark exists (we'd establish it), and ablations suggest
    higher-order signal is helpful-but-not-dominant — **our infomax substrate may already capture most of
    the structure** (this is W1's cousin at the STL level). Transfer to coarser category/region is
    plausible but unproven (no STHGCN-lineage paper reports it).
  - **De-risk recipe (from the report, adjusted for the Sphere2Vec swap):** make the **cohesive
    self-supervised hypergraph fusion the primary contribution** (Sphere2Vec + Time2Vec + POI2Vec geo-latents
    co-embedded in one hypergraph pass — the unclaimed intersection), **inductive generalization the
    headline motive**, and **benchmark against our own champion** (not against STHGCN's supervised numbers,
    which aren't comparable). Full report: research artifact (STHGCN/hypergraph thread).
- **Verdict + placement:** **promoted to the thesis track (T1 enabler + T2 inductive) and flagged as the
  strongest candidate for a standalone post-defense study.** It's the *bigger, riskier* of the two ideas —
  defensible novelty, but the accuracy payoff on Gowalla is an open bet, so its safest thesis role is a
  **§6.3 future-work direction with the inductive/foundation motive**, not a promised result. Plan-level
  only; **elaboration and problem-framing deferred** by author decision.

---

## 6 · Cross-references

- **Memos (acceptance-criterion source of truth):** [`future_works/`](future_works/) + its
  [`README.md`](future_works/README.md) index.
- **Deferred-work map for the mtl-protocol-fix line:**
  [`studies/archive/mtl-protocol-fix/DEFERRED_WORK.md`](studies/archive/mtl-protocol-fix/DEFERRED_WORK.md).
- **Dissertation future-work chapter:** `articles/dissertacao/NORTH_STAR.md` §6.3 (limitation-tied);
  chapter map §3. This doc's §2 is the working draft for that section.
- **The wall / settled science:** `NORTH_STAR.md` (regime finding), `CONCERNS.md` (§C21/C25/C26),
  `studies/archive/mtl_improvement/FINAL_SYNTHESIS.md`, `studies/log.md`.
- **Active experimental engine:** [`studies/closing_data/`](studies/closing_data/) (where any launched
  item runs).
