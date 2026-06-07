# Future work — champion G at CA/TX (the scale-conditional completeness card, T2V.9)

**Status:** DEFERRED 2026-06-07 (user decision). The `mtl_improvement` champion **G** is validated
multi-state/multi-seed/paper-grade at AL/AZ/GE/FL (beats both STL ceilings, 4-seed). T2V.9 (CA/TX)
is the lone *completeness* extension — NOT paper-blocking — and the single most expensive step, so it
was deferred to here.

## The prediction (recorded for when it's run)
The C25 class-weighting confound scaled with class count → **CA (8501 regions) / TX (6553 regions) are
where the C25 fix's benefit is predicted LARGEST.** So G should beat both (c) ceilings at CA/TX by margins
**≥ the 4-state ones** (FL +0.26 reg / +3.2 cat is the smallest-margin large state; CA/TX should exceed it).
If instead the margin shrinks/flips at CA/TX, that bounds the scale-conditional claim — report the boundary.

## What it takes (the cost that motivated the deferral)
1. **Build the v14 `design_k` substrate** at CA + TX — `check2hgi_design_k_resln_mae_l0_1` =
   ResLN + mae cat lever ⊕ Delaunay-POI-GCN reg lever. Entry: `scripts/canonical_improvement/regen_emb_t3.py`;
   exact recipe must be reconstructed from `docs/studies/embedding_eval/` (CANDIDATES.md / FINAL_SYNTHESIS.md)
   + `docs/results/CANONICAL_VERSIONS.md §v14`. **Get the recipe right before launching — a wrong build wastes
   the large-state hours.** CA/TX have the base `output/check2hgi/<state>/` to build from.
2. **Seeded per-fold log_T** at CA/TX: `scripts/compute_region_transition.py --state <st> --per-fold --seed <S>`.
3. **(c)/(d) ceilings** at CA/TX: `scripts/mtl_improvement/t2v1_ceilings_multiseed.sh` (adapt the state list;
   HGI for (d) needs HGI region-emb at CA/TX — likely a build too, else (c)-only).
4. **Champion G** at CA/TX: the `scripts/mtl_improvement/c25_g_multistate.sh` recipe (swap the state list).
   CA/TX MTL runs are the slowest (8.5k/6.5k-class softmax) — budget accordingly; 1-fold directional is
   acceptable if 5-fold×4-seed is impractical (flag as directional, no paired Wilcoxon).

## Comparand
Same as the 4-state result: G reg/cat vs the (c) STL ceilings (reg next_stan_flow α=0; cat next_gru
logit-adjust τ=0.5) + the (d) composite. Paper-grade = multi-seed {0,1,7,100}; large states earlier
shipped at n=20 (§0.1). Trail: `docs/studies/mtl_improvement/{CHAMPION.md, HANDOFF.md, log.md, INDEX.html #T2V-9}`.
