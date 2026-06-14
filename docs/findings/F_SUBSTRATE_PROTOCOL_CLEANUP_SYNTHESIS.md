# F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS — one-stop synthesis of the investigation

**Date:** 2026-05-30
**Study:** `substrate-protocol-cleanup` (CLOSED 2026-05-29) + the 2026-05-30 v11→v12 default flip.
**Purpose:** a single readable entry point for future agents/readers covering the whole
investigation: the substrate-axis null in MTL, the regime isolation, the HGI ceiling,
the log_T-KD deployable lift, the ResLN STL dual-axis, and the resulting v12 code
default. Read this before re-litigating any of these questions.

---

## TL;DR

1. **The cross-attn MTL joint-training regime washes out encoder/substrate improvements
   on BOTH axes.** Better substrates (designs B/J/L/M) and even the STL reg ceiling
   (HGI) give **no MTL reg advantage**. This is the **regime finding** — the central
   result of the study.
2. **Only the log_T PRIOR pathway moves MTL reg.** `log_T-KD` (W=0.2) is the one
   deployable MTL reg improvement: **paper-grade at AL/AZ** (+2.27/+4.91 pp, n=20,
   p=9.54e-07, leak-clean), single-seed **pilot** at FL/CA/TX (+2.40/+1.42/+1.71 pp).
3. **ResLN is the best STL cat encoder** (+0.86–1.70 pp cat) but gives **NO MTL benefit**
   — its value is STL/representation-quality/generality.
4. **The residual MTL-vs-STL reg gap is architectural** (shared-backbone), handed to
   `mtl_improvement`. This study contributes the elimination evidence, not the fix.
5. **v12 default (2026-05-30):** log_T-KD W=0.2 ON (scoped to MTL `check2hgi_next_region`)
   + ResLN encoder default (future builds). v11 (the BRACIS paper canon, no-KD/GCN)
   remains reproducible via `--log-t-kd-weight 0.0` + the frozen GCN substrate. See
   [`../results/CANONICAL_VERSIONS.md`](../results/CANONICAL_VERSIONS.md).

---

## 1. Substrate axis is NULL in MTL (Tier B + FL three-way)

Four mechanistically-distinct substrate variants — Design B (POI2Vec @ pool boundary),
Design J (H + anchor λ=0.1), Lever 4 (POI2Vec @ p2r additive), Lever 5 (KL distill /
orphan rescue) — that DOMINATE canonical Check2HGI on **STL reg** at AL/AZ/FL **do not
transfer** that advantage to MTL+F1:

- **AL/AZ:** disjoint reg flat (|Δ|≤1.22 pp, every p≥0.21), on BOTH the disjoint-oracle
  and the deployable joint/geom_simple fronts.
- **FL (B9 three-way):** designs close **0 %** of any MTL gap (disjoint |Δreg|≤0.16 pp,
  none significant). Only **J** is Wilcoxon-strict over canonical on **STL** (+1.12 pp,
  closes 53 % of the canonical→HGI gap); that STL edge vanishes under MTL.
- The **−1.7 to −2.4 pp cat** on every design build is a **build-scope confound** (each
  design build re-inits a fresh CheckinEncoder, drifting the cat-path input 100 % vs
  canonical; at α=0 the cat Δ is ~+0.19 pp), NOT a substrate cost.

**Mechanism (hedged, re-audited 2026-05-29):** under the joint config the MTL reg head is
dominated by its **α·log_T transition anchor**; with α frozen to 0 (an OOD ablation) reg
sits at near-floor for BOTH design and canonical, so the substrate-carrying encoder branch
contributes ~nothing beyond the prior. Accurate statement: *"no reg gain BEYOND the
canonical log_T anchor under the joint config"* — not *"the substrate fails to transfer."*

Sources: `../results/substrate_protocol_cleanup/tier_b/phase_b_reaudit.md`,
`tier_b/phase_b_two_front.md`, `tier_b_fl/phase_b_fl_3way.md`.

## 2. The HGI ceiling (the missing control)

Even **HGI** — the STL `next_region` ceiling (+2.12 pp STL reg over canonical at FL) —
gives **NO MTL reg advantage**: FL disjoint reg 64.49 ± 0.55 vs canonical 63.98,
**Δ+0.51, p=0.41 NS**. HGI's STL win VANISHES under B9 joint training (70.9 → 64.5 ≈
canonical ≈ designs). HGI cat collapses to 34.84 % F1 (−35.6 pp; non-viable as an MTL
substrate). So the designs are not "failing to carry HGI's advantage" — **there is no
substrate (not even HGI) whose reg advantage survives the MTL regime.** ~0.36 GPU-h.

Source: `../results/substrate_protocol_cleanup/tier_b_fl/hgi_mtl_fl.md`.

## 3. The STL↔MTL isolation cell (regime, not head/substrate)

The clean apples-to-apples experiment: identical head/config/state/embeddings.
STL `next_stan_flow` α=0 (log_T fully off) LEARNS FL region at **~73 % Acc@10**
(canonical 72.74 / design_b 73.12, Δ+0.37 p=0.0312), while the IDENTICAL head/config
under MTL FLOORS at **~0.03 %** (chance 0.213 % for 4703 FL regions). **Verdict: it is the
joint-training REGIME, not the head or substrate, that kills the MTL reg encoder.**
Hedge: α=0 is OOD → the claim is regime-and-config-scoped (B9, 50 ep), not an absolute
"the encoder can never learn region under MTL."

Source: `../results/substrate_protocol_cleanup/tier_b_fl/phase_b_fl_3way.md`.

## 4. log_T-KD — the one deployable MTL reg lift (Tier A1, NOW v12 default)

KL distillation of the per-fold **train-only** log_T (region Markov-1 prior) into the
reg-head logits — the live prior pathway in MTL.

- **Small states PAPER-GRADE (n=20, seeds {0,1,7,100}):** AL +2.27 pp, AZ +4.91 pp
  disjoint reg, 20/20 folds positive, paired Wilcoxon **p=9.54e-07** each; cat untouched.
- **Large states seed=42 PILOT (NOT paper-grade):** FL +2.40 (5-fold p=0.031), CA +1.42,
  TX +1.71 (1-fold) — sign-and-magnitude only (W=0.0 baselines overshoot §0.1 per C23).
- **Leak-audited CLEAN** (7-vector independent audit, NO LEAK): train-fold-only log_T,
  `last_region_idx` from `poi_0..8` only (never target), W=0.0 fast path byte-identical,
  reg-only lift with flat cat (opposite of the leak signature). The lift exploits a
  strong-but-bounded last→target prior (MI/H(Y) ≈ 0.58; top-1 determinism ≈ 0.35) built
  train-only — frame as "supervisory distillation of an empirical first-order region-Markov
  prior", not a novel architecture.
- **Additive on B9 at FL:** canonical + log_T-KD = best MTL reg ≈ 66.4 % (+2.40 over
  canonical); design_b+KD ≈ canonical+KD (Δ−0.03 p=0.69) — the substrate null is ROBUST
  to KD. Same regime, opposite pathways: KD works through the live log_T PRIOR; substrate
  works through the starved encoder.

Sources: `../results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md`,
`tier_a1_largestate/phase_a1_largestate_addendum.md`,
[`F_TIER_A1_PROMOTION.md`](F_TIER_A1_PROMOTION.md), [`F_TIER_A1_LEAK_AUDIT.md`](F_TIER_A1_LEAK_AUDIT.md).

## 5. ResLN STL dual-axis (canonical_improvement T3.2, NOW v12 encoder default)

`ResidualLNEncoder` is the best STL cat encoder: cat F1 **+0.86 FL / +1.48 AL / +1.70 AZ**
(5/5 seeds, p=0.03125); reg ≈0 small states / +0.71 FL (mostly v3c). Leak +2.24
IJM-verified honest. Dual-axis STL champions: **ResLN+design_b** (general) /
**ResLN+design_j** (AL-specific) — kept as **opt-in research variants** (registered,
NOT defaulted beyond the encoder). **CRITICAL: NO MTL benefit** (per §1–§3 regime
finding); the value is STL/representation-quality/generality only — never imply ResLN
improves MTL.

Source: `../results/substrate_protocol_cleanup/tier_resln/phase_resln_verdict.md`,
`../studies/archive/substrate-protocol-cleanup/canonical_improvement_coverage_audit.md`.

## 6. The deployable conclusion + the v12 default

- **Ship canonical + log_T-KD** for MTL reg (the only lever that moves it).
- **Substrate/encoder = STL / generality** (ResLN default); they do not move MTL.
- **The MTL reg bottleneck is architectural** (shared-backbone) → owned by
  `mtl_improvement` T2. This study contributes elimination evidence (substrate null +
  HGI ceiling + isolation cell + P4/C3 pathway exonerations), not a mechanism.
- **v12 code default (2026-05-30):** log_T-KD W=0.2/τ=1.0 ON, scoped to MTL
  `check2hgi_next_region`; ResLN encoder default for future builds. v11 (BRACIS paper
  canon, no-KD/GCN) reproducible via `--log-t-kd-weight 0.0` + the frozen GCN substrate
  (`output/check2hgi/<state>/`, NOT rebuilt).

See also: [`../results/CANONICAL_VERSIONS.md`](../results/CANONICAL_VERSIONS.md),
[`../results/RESULTS_TABLE.md §0.8` + `§0.9`](../results/RESULTS_TABLE.md),
[`../studies/archive/substrate-protocol-cleanup/CLOSURE.md`](../studies/archive/substrate-protocol-cleanup/CLOSURE.md),
[`../NORTH_STAR.md`](../NORTH_STAR.md), [`../CONCERNS.md` C15](../CONCERNS.md).
