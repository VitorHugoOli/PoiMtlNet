# T2.1 byproduct — the scale-conditional LR-recipe finding (recipe × state matrix)

**2026-06-04.** A clean, decomposed Tier-4 (optimizer) result that fell out of the T2.1 LR
mini-sweep. Baseline head (`mtlnet_crossattn` + `next_getnext_hard`), prior-ON, static_weight
cat0.75, KD-OFF, per-head LR, 5f×50ep, seeded per-fold log_T (leak-free). reg = top10_acc_indist
disjoint (capacity); cat = macro-F1 deploy.

## The three recipes
- **H3-alt** (blessed small-state): `--scheduler constant` + per-head LR; NO alt-opt / α-no-wd / min-best.
- **B9** (blessed large-state): `--scheduler cosine --max-lr 3e-3` + per-head LR + `--alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5`.
- **onecycle** (the finding): `--scheduler onecycle --max-lr 3e-3` + per-head LR; NO alt-opt = B9's aggressive schedule WITHOUT B9's alt-opt.

## Matrix — reg@10 disjoint / cat-F1 (seed42; (L)=landed multi-seed {0,1,7,100})
| recipe | AL | AZ | FL | CA | TX |
|---|---|---|---|---|---|
| H3-alt | 47.23 / 46.78 (L) | 38.27 / 48.75 (L) | **62.42 / 67.38** | — | — |
| onecycle | **56.45 / 48.51** | **44.26 / 49.43** | 61.87 / 65.82 | _pending_ | _pending_ |
| B9 | **50.96 / 42.80** | **38.32 / 46.82** | 61.28 / 70.26 (L) | _pending_ | _pending_ |

(CA/TX: B9 + onecycle pending — canonical substrate, graph maps safely regenerated 2026-06-04,
99.99% aligned; run serially CONC=1 due to 8.5k/6.5k-region VRAM ~31GB/run.)

## Decomposition (the answer)
**Small states (AL/AZ): onecycle DOMINATES both axes.**
- reg: onecycle ≫ B9 ≥ H3-alt. AL: 56.45 vs 50.96 vs 47.23. The aggressive schedule helps reg
  (B9 > H3-alt, +3.7 AL), but **no-alt-opt helps even MORE** (onecycle > B9, +5.5 AL) — alt-opt
  costs reg too, not just cat.
- cat: onecycle > H3-alt ≫ B9. AL: 48.51 vs 46.78 vs **42.80**. **B9's alt-SGD crushes small-state
  cat** (−5.7 vs onecycle) — the documented reason H3-alt was blessed over B9 (`NORTH_STAR.md:18-24`).
- So **onecycle = aggressive schedule (helps reg) + no alt-opt (helps both reg AND cat)** = the
  small-state sweet spot that H3-alt (no aggressive schedule → weak reg) and B9 (alt-opt → tanks cat)
  both missed.

**Large state (FL): reg recipe-insensitive (~61–62, all three ≈ tie); B9 wins cat (70.26).**
- At FL, alt-opt **HELPS** cat (B9 70.26 > H3-alt 67.38 > onecycle 65.82) — **opposite sign from small
  states.** reg is flat across recipes. So B9 stays the right FL recipe (its cat win is why it's blessed).

## Why this is genuinely new (not a re-discovery)
The recipe-selection study (`RESULTS_TABLE.md §0.4`, `NORTH_STAR.md`) compared H3-alt vs B9 as a
**binary** and pinned B9's small-state rejection on **alt-SGD** (cat). But H3-alt dropped BOTH the
aggressive scheduler AND alt-opt. The "aggressive-schedule-minus-alt-opt" cell was only ever tested at
**FL** (F50_T3 A1/A2, where it loses to B9) — **never at AL/AZ**. onecycle keeps the helpful lever
(schedule), drops only the harmful one (alt-opt). The alt-opt lever **flips sign by scale** (hurts
small-state cat, helps large-state cat); the aggressive schedule helps reg everywhere but matters most
at small states where H3-alt's weak constant LR left reg on the table.

## Recommendation (pending CA/TX confirmation + user decision)
- **AL/AZ: adopt onecycle** (dominates H3-alt by +6–9pp reg / +1–2pp cat, multi-seed validated).
- **FL/CA/TX: keep B9** (cat win at scale; reg insensitive). Expect CA/TX to mirror FL.
- **Consequence:** re-states the (a) MTL baseline at small states → shrinks the composite advantage
  (AL MTL→composite gap 13.44→7.44pp). NORTH_STAR small-state recipe change candidate; needs the user.
- **Does NOT change** the T2.1 architecture verdict (dual-tower falsified) or the STL (c)/(d) ceilings.

## Provenance
Drivers: `scripts/mtl_improvement/t21_recipe_matrix.sh` (+ `t21_harden.sh` validate stage for the
onecycle multi-seed). Agg: `t21_recipe_agg.py`. Graph regen: `t21_regen_catx_graph.py` (safe, verified).
Multi-seed onecycle: `T21_onecycle_validation_multiseed.txt`.
