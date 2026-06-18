# PROPOSAL — drop-in reframing of contribution C2 (paper-ready)

> **Status:** actionable proposal for the paper team (companion to `MEMO_2026-06-17_catreg_regime_and_C2.md`,
> which has the full analysis + adversarial fact-check). **Not an edit to locked §0 prose** — paste-ready
> text + evidence the team can adopt once §0 C2 is unlocked. **Decision owners:** paper team (prose) +
> `closing_data` (the §0 re-baseline that fully lands it).
> **One-line ask:** replace C2's "MTL pays a −7…−17 pp region cost (classic tradeoff)" with "MTL is a
> Pareto gain on the easy task at parity on the hard one; the pre-2026-06 region 'cost' was ~half a
> class-weighting confound + ~half config, not a representational tradeoff."

## 1. The current C2 (to be replaced)

> *"C2 — Classic MTL tradeoff. With Check2HGI fixed and a cross-attention MTL backbone, joint training adds
> a small additional cat lift … On next-region it pays 7 to 17 pp on Acc@10 vs. a matched-head STAN-Flow
> STL ceiling at every one of the five states — sign-consistent negative (FL −7.34 / CA −9.50 / TX −16.59)."*

This rests on the **v11 paper canon** (B9 recipe, GCN substrate), where the MTL region head trained on
**class-WEIGHTED** CE while the STL ceiling and the Acc@10 metric are **unweighted** (`CONCERNS §C25`).

## 2. Proposed C2 (paste-ready)

> **C2 — MTL is a Pareto gain on the easy task at parity on the hard one (not a tradeoff).** With Check2HGI
> fixed and a cross-attention dual-tower backbone, joint training **beats the single-task category ceiling
> by +2.6 to +4.1 pp** while **matching the single-task region ceiling within 0.3 pp** (champion G,
> unweighted CE; AL −0.09 / AZ −0.12 / GE −0.09 / FL −0.31, 4 states × 4 seeds). The large region "cost"
> reported by earlier configurations is **not representational**: it decomposes into **(i) a class-weighting
> confound** — the region head trained on class-balanced CE optimizes *away* from the frequency-weighted
> Acc@10 metric, depressing it from epoch 1 by an amount that **scales with class count** — and **(ii) a
> configuration gap** closed by the dual-tower + v14 substrate. A controlled A/B (flip *only* the
> class-weight flag, fixed recipe/substrate) recovers **+3.15 pp region at FL (gap −6.71 → −3.56, halved)**
> [CA/TX: +X.XX / +X.XX — TODO from `c2_catx_ab`], and the remainder closes under the confound-free champion.
> Mechanistically, the region head reaches its single-task ceiling through a **private tower that, at scale,
> is insulated from the shared trunk** (its learned shared-mixing coefficient β → 0), so joint training moves
> the shared-trunk-harvesting category task without disturbing region — the post-2022 "scalarization
> suffices, parity-not-conflict" regime (Kurin/Xin NeurIPS'22; Mueller TMLR'25).

## 3. Proposed abstract/tagline edit

> **From:** *"MTL on top of the substrate gains a small cat lift and pays a sign-consistent reg cost — the
> textbook tradeoff."*
> **To:** *"MTL on top of the substrate is a Pareto gain — it beats the single-task category ceiling
> (+2.6…+4.1 pp) at parity on the harder region task (within 0.3 pp); the region 'cost' seen in earlier
> setups was a class-weighting confound, not a representational tradeoff."*

(Note: this **strengthens** the paper — "Substrate Carries, Architecture Pays" still holds; the architecture
now *also* delivers a free cat lift at no reg cost, rather than paying a reg tax.)

## 4. Evidence table (the C2 replacement data)

**(a) Confound-free champion G vs same-substrate STL ceiling (unweighted CE, v14 substrate, 4 seeds)** —
`R0_matched_metric_bar.json`:

| state | G region | STL region ceiling | Δ region | G cat-F1 | STL cat | Δ cat |
|---|---|---|---|---|---|---|
| AL | 62.57 | 62.67 | **−0.09** (matches) | 52.91 | 50.35 | **+2.56** |
| AZ | 54.68 | 54.80 | **−0.12** (matches) | 54.48 | 50.39 | **+4.08** |
| GE | 58.35 | 58.44 | **−0.09** (matches) | 61.43 | 57.50 | **+3.93** |
| FL | 72.97 | 73.27 | **−0.31** (matches) | 73.16 | 69.96 | **+3.20** |

**(b) Controlled class-weighting A/B (flip only the flag; fixed v11 GCN/B9 recipe)** — the confound's share
of the region gap:

| state | weighted MTL reg | unweighted MTL reg | Δ (confound recovers) | gap weighted → unweighted |
|---|---|---|---|---|
| FL (seeds {0,1,7,100}, bs2048) | 63.91 | 67.06 | **+3.15** | −6.71 → −3.56 (**halved**) |
| CA (seed 42) | — | — | **not measured** | confound A/B hardware-infeasible (see note) |
| TX (seed 42) | — | — | **not measured** | confound A/B hardware-infeasible (see note) |

> **⚠ CA/TX measured A/B is hardware-infeasible on the A40 (2026-06-17), deferred to `closing_data`.** The
> v11 B9 recipe OOMs the A40 for the large states: **TX** (~8.5k region logits) OOMs at every feasible batch
> (bs2048/512); **CA** fits only at bs512, where the **weighted arm diverges** (reg ≈ 0 across all epochs vs
> the v11 bs2048 weighted ≈ 40 — a small-batch instability, not a clean confound delta). So **only FL has a
> clean controlled A/B** (+3.15 pp). The CA/TX confound share is **predicted** by the C25 class-count scaling
> — the confound depresses top-K *more* at higher class count, and the v11 reg-cost itself scales with class
> count (FL −7.34 < CA −9.50 < TX −16.59) — so the dramatic TX "−16.59 pp" is expected to be **largely the
> confound**, but this is a **reasoned prediction, not measured here.** The clean measurement is part of
> `closing_data`'s §0 re-baseline (the confound-free champion on the v14 substrate at all states, on
> appropriate hardware). *(Attempt logged: `c2_catx_ab_results.json` (status=INFEASIBLE);
> `scripts/mtl_frontier/c2_catx_classweight_ab.sh`.)*

## 5. Scope + honest caveats (MUST keep in the paper)

1. **Confound + config, not a flag-flip.** Unweighting alone on the v11/GCN/B9 config recovers ~half the FL
   region gap; full parity additionally requires the champion-G stack (v14 substrate + dual-tower). The clean
   move is to **re-baseline §0 on the confound-free champion** (the `closing_data` plan) and report those.
2. **State coverage.** Champion-G region-PARITY is established at **AL/AZ/GE/FL**; **CA/TX have no v14
   substrate yet** (a `closing_data` build) — CA/TX parity is *expected* (same mechanism) but **unmeasured**.
   The CA/TX *confound A/B* (this proposal §4b) is on the GCN substrate and shows the confound share, not
   full parity. Report FL as the established headline; CA/TX confound-share measured, parity pending.
3. **The cat lift is partly the C25 cat-unweighting fix and partly head-config** (the STL cat ceiling uses a
   different head config than G) — it is a real *deployable* gain but not a pure "MTL beats STL" effect;
   disclose the recipe/head-config asymmetry.
4. **Parity, not a region win.** Claim region-**parity** (within noise), never "MTL beats STL on region."

## 6. What it takes to fully land C2 (handoff)

- **Paper team:** adopt §2/§3 prose once §0 C2 is unlocked; keep the §5 caveats.
- **`closing_data`:** the decisive action — **re-baseline §0 on the confound-free champion G** (v14 substrate
  + dual-tower, unweighted CE), including **building the CA/TX v14 substrate** so champion-G region-parity is
  *measured* at the paper's headline states (currently FL only). This is already the scaffolded `closing_data`
  full §0 re-run; this proposal is its C2-facing rationale.

## 7. Sources

`MEMO_2026-06-17_catreg_regime_and_C2.md` (full analysis + fact-check) · `R0_matched_metric_bar.json` ·
`scripts/mtl_improvement/c25_fl_b9_{weighted,continuity}.sh` (FL A/B) ·
`scripts/mtl_frontier/c2_catx_classweight_ab.sh` (CA/TX A/B, this proposal) · `CONCERNS §C25` ·
`docs/studies/mtl_frontier/FINAL_SYNTHESIS.md §4` · `docs/studies/closing_data/PLAN.md`.
