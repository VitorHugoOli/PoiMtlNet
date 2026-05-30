# Tier B B3 verdict — Lever 4 (POI2Vec at p2r boundary, additive) on canonical c2hgi, MTL under F1 at AL/AZ

**Date**: 2026-05-28
**Phase**: Tier B B3 (Lever 4), run AFTER the Wave-1 (B1/B2/B4) verdict.
**Mechanism** (`LEVER_4_POI2VEC_P2R.md` + `scripts/substrate_protocol_cleanup/build_lever4_substrate.py`): additive region-prior term on `L_p2r`, `L = L_canonical + α·(1 − cos(region_emb, W·region_prior))`, `region_prior[r] = mean(POI2Vec[i] for i in region r)`, α=0.1. Forward pass is **canonical c2hgi** (`--base canonical`); Lever 4 is a loss-only extension. Engine `check2hgi_lever4_canonical`.
**Scope**: AL + AZ, seed=42, 5 folds, H3-alt small-state recipe, `--no-checkpoints`.
**Base / control**: the `canonical_baseline` MTL cell at the same state (same recipe, same folds) — Lever 4's gate is "vs its base", and its base is canonical c2hgi.

**Wave-1 dependency**: no Wave-1 design promoted (`phase_b1b2b4_verdict.md`), so per `INDEX.md` §B3 only the canonical+Lever4 control was built and trained; the winner-substrate stack (`check2hgi_lever4_design_b`) was SKIPPED.

**Verdict in one line: Lever 4 on canonical is NOT PROMOTED at either state — no disjoint-reg lift (Δ −0.24 AL / −0.08 AZ, both below the +0.3 pp gate) and a ~−2.6 pp cat regression.**

---

## §1 Three-frontier + Δ vs base (canonical)

STL reg ceiling = `next_stan_flow` matched head, RESULTS_TABLE §0.1 v11.

### Alabama

| Substrate | disjoint reg | geom_simple reg | disjoint cat F1 | STL reg ceiling |
|---|---:|---:|---:|---:|
| canonical c2hgi (base) | 50.82 | 48.56 | 45.76 | 61.21 |
| canonical + Lever 4 | 50.58 | 48.43 | 43.08 | 61.21 |
| **Δ (Lever4 − base)** | **−0.24** | **−0.13** | **−2.68** | — |

### Arizona

| Substrate | disjoint reg | geom_simple reg | disjoint cat F1 | STL reg ceiling |
|---|---:|---:|---:|---:|
| canonical c2hgi (base) | 41.33 | 39.60 | 48.87 | 53.06 |
| canonical + Lever 4 | 41.25 | 39.58 | 46.33 | 53.06 |
| **Δ (Lever4 − base)** | **−0.08** | **−0.02** | **−2.54** | — |

---

## §2 Decision gate (per INDEX.md §B3: +0.3 pp disjoint reg AND no cat regression vs base)

| State | disjoint reg Δ | per-fold Δ pp (f1..f5) | folds + | Wilcoxon W / p (1-sided) | cat Δ | Gate | Verdict |
|---|---:|---|---:|---:|---:|---|---|
| Alabama | −0.24 | −0.20, +0.41, −0.57, +0.24, −1.10 | 2/5 | 5.0 / 0.781 | −2.68 | reg < +0.3 AND cat regresses | **NOT PROMOTED (FALSIFIED)** |
| Arizona | −0.08 | +0.08, +0.13, −0.27, −0.21, −0.13 | 2/5 | 4.0 / 0.844 | −2.54 | reg < +0.3 AND cat regresses | **NOT PROMOTED (FALSIFIED)** |

The +0.3 pp disjoint-reg lift the Lever-4 proposal hypothesised (`LEVER_4_POI2VEC_P2R.md`: "+0.3-0.6 pp on AZ/FL reg without disturbing the cat path") does NOT materialise under MTL+F1 at small states: reg is flat-to-negative at both states. The proposal also predicted "Cat regression: zero (cat path is already detached and not touched)"; instead cat regresses ~−2.6 pp — the same uniform cat cost the Wave-1 designs show, confirming it is a substrate-embedding property (the design/Lever checkin vectors fed to the cat task differ from canonical), not specific to the p2r mechanism.

---

## §3 C18 leak-probe

NO leak signature. A leak = large disjoint-reg jump and/or cat LIFT (T3.1 pattern). Observed = reg flat/negative (Δ −0.24 / −0.08), geom reg flat (−0.13 / −0.02), cat REGRESSION (−2.68 / −2.54) — the opposite. The region-prior term neither moves reg nor lifts cat. Nothing leak-shaped; no dedicated leak audit needed (nothing promoted).

---

## §4 Cross-reference

Per `INDEX.md` §B framing + §4.2: a Lever-4 PROMOTE would have been a free additive upgrade to the architectural champion, never a project headline. It did not promote. The project reg headline remains the §4.2 deploy composite (+7–12 pp disjoint reg vs MTL). Tier B (Wave 1 + B3) produced no substrate upgrade to layer onto the architectural champion.

---

## §5 Build provenance + cost

- POI2Vec teacher present at `output/hgi/{alabama,arizona}/poi2vec_poi_embeddings_*.csv` (built 2026-05-28 19:49/19:51).
- Lever-4 substrate built via `build_lever4_substrate.py --base canonical --epochs 500 --alpha 0.1 --device cuda`: AL best_epoch=499 loss=0.4852 (~2 min); AZ best_epoch=497 loss=0.4990 (~3 min).
- Postbuild via `postbuild_design_substrate.sh check2hgi_lever4_canonical {state}`: next.parquet + next_region.parquet (AL 12709 rows / AZ 26396 rows, n_regions 1109 / 1547 matching canonical) + canonical seed=42 log_T cp'd and touched after parquet (C22 satisfied).
- Allowlist fix: added `EmbeddingEngine.CHECK2HGI_LEVER4_CANONICAL` to `src/data/folds.py:_MTL_C2HGI_ALLOWED_ENGINES` and `scripts/train.py:_ALLOWED_ENGINES_FOR_C2HGI_PRESET` (both previously listed only `LEVER4_DESIGN_B`). Same allowlist-gate class as the original Tier B incident.
- MTL train: H3-alt recipe (`--mtl-loss static_weight --category-weight 0.75 --scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 --cat-head next_gru --reg-head next_getnext_hard --task-a-input-type checkin --task-b-input-type region --no-checkpoints`), AL ~1.5 min / AZ ~3 min wall-clock.
- Total B3 compute ≪ 1 GPU-h (well under the 8 GPU-h projection). Disk held at 14 GB free throughout (never below the 3 GB STOP threshold). `--no-checkpoints` honoured.

## Artefacts

- Run dirs: `docs/results/substrate_protocol_cleanup/tier_b/lever4_canonical/{alabama,arizona}/seed42/mtlnet_*/`
- Substrate: `output/check2hgi_lever4_canonical/{alabama,arizona}/`
- Build/postbuild scripts: `scripts/substrate_protocol_cleanup/{build_lever4_substrate.py,postbuild_design_substrate.sh,build_design_next_region.py}`
- Winner-stack (B3b) SKIPPED — no Wave-1 winner.
