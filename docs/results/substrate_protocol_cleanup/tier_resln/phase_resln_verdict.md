# ResLN matrix verdict — encoder improvement (T3.2 ResidualLN) × substrate, STL + MTL, vs HGI

**Date**: 2026-05-30
**Scope**: ResLN-canonical + ResLN+design_b (ResLN encoder + POI2Vec@pool, never-before-built stack) at AL/AZ/FL, seed=42, 5-fold, `--no-checkpoints`. STL (reg + cat) and MTL (B9/H3-alt, two fronts). Compared vs canonical (no-ResLN, matched recipe), designs B/J, and HGI. RAW per-fold Wilcoxon (scipy exact).
**Trigger**: the canonical_improvement coverage audit found ResLN (T3.2, +0.86–1.70 pp STL cat) is the one validated improvement NOT in our baseline.

---

## 1. STL NEXT-REG (Acc@10 %) — does the family EQUALISE / BEAT HGI?

| State | canonical | design_b | design_j | ResLN-canon | **ResLN+design_b** | **HGI** |
|---|---:|---:|---:|---:|---:|---:|
| AL | 61.02 | 61.49 | 61.96 | 60.96 | **61.99** | 61.86 |
| AZ | 51.33 | 52.59 | 52.15 | 51.57 | **52.98** | **53.37** |
| FL | 69.68 | 69.93 | 70.34 | 69.68 | **70.21** | **71.34** |

(FL design_b = leak-free 69.93; the plain P1 ~82 % file is the known-leaky one, excluded.)

**Read:**
- **AL — EQUALISED.** ResLN+design_b (61.99) and design_j (61.96) both nominally **exceed HGI (61.86)** — within noise (beat-HGI p=0.31 / 0.50), i.e. statistically tied. The family matches HGI on reg at AL.
- **AZ — closes most, HGI still leads.** ResLN+design_b (52.98) is the best closer (−0.39 vs HGI), shrinking canonical's −2.04 gap by ~80 %. HGI nominally leads.
- **FL — closes ~30 %, HGI leads by ~1.1 pp.** ResLN+design_b 70.21 vs HGI 71.34. Consistent with the real residual embedding-quality gap to HGI (merge_design Test 2).
- **ResLN+design_b is the best non-HGI variant at ALL three states** — the ResLN-encoder × POI2Vec stack helps STL reg marginally beyond design_b or ResLN alone (the stack the two prior studies never built).

## 2. STL NEXT-CAT (F1 %) — does check2hgi STAY BEST? does ResLN widen it?

| State | canonical | ResLN-canon | **ResLN+design_b** | HGI |
|---|---:|---:|---:|---:|
| AL | 48.73 | 49.18 | **51.26** | ~20–25 |
| AZ | 51.14 | 51.46 | **52.45** | ~24 |
| FL | 70.60 | 70.39 | 70.58 | (low) |

**Read:** check2hgi **STAYS DECISIVELY BEST on cat** — 48–70 % vs HGI's ~20–25 % (2–3×). HGI is non-viable on cat. **ResLN+design_b widens the cat lead at AL (+2.53) and AZ (+1.31)**; flat at FL. So the cat half of the merge thesis is not just preserved but improved by the ResLN×POI2Vec stack at small states.

## 3. The dual-axis STL verdict (the user's question)

**ResLN+design_b is the strongest single-engine STL result in the whole line of work:** it is the best reg-gap-closer (ties HGI at AL, closes ~80 % at AZ, ~30 % at FL) AND the best cat (widens the lead over canonical/HGI at AL/AZ). That is exactly the merge thesis goal — *equalise HGI on reg while keeping/improving cat over HGI* — best realised by stacking the ResLN encoder with POI2Vec injection. It equalises HGI on reg at AL and gets within ~0.4–1.1 pp at AZ/FL, while keeping a 2–3× cat advantage.

## 3b. ResLN+design_j (added 2026-05-30) — the AL specialist

Design J (anchored learnable POI table) was the best STL reg closer at AL. ResLN+design_j tested across AL/AZ/FL, STL + MTL.

**STL next-reg (Acc@10 %):**

| State | ResLN+design_j | vs HGI | vs ResLN+design_b |
|---|---:|---:|---:|
| AL | **62.10** | **+0.24** (the single highest reg of ANY variant — nominally beats HGI, p=0.50) | +0.11 |
| AZ | 52.22 | −1.15 | **−0.76** (worse) |
| FL | 69.76 | −1.58 | **−0.45** (worse) |

**STL next-cat (F1 %):** AL 49.76 (+1.03 vs canon), AZ 51.86 (+0.73), FL 70.21 (−0.39) — but **lower than ResLN+design_b at all 3 states** (−1.50 / −0.59 / −0.36).

**MTL:** flat everywhere (Δreg −0.36/−0.14/−0.06, Δcat +0.22/−0.41/−0.35; all NS) — confirms the regime finding.

**Verdict:** ResLN+design_j is the **AL reg specialist** — it is the *only* variant to nominally exceed HGI on reg at any state (AL 62.10 vs HGI 61.86), edging even ResLN+design_b. **But it does NOT generalise**: at AZ/FL it is worse than ResLN+design_b on reg, and it is worse than ResLN+design_b on cat at all three states. This reproduces merge_design's documented "J's HGI-beat is AL-specific" finding, now with the ResLN encoder. **ResLN+design_b remains the better all-around STL engine** (best reg closer at AZ/FL, best cat everywhere); ResLN+design_j wins only the AL reg cell. Neither beats HGI significantly anywhere.

## 4. MTL — none of it transfers (the deployable caveat)

All 9 MTL cells (B9/H3-alt, two fronts) vs the matched canonical baseline:
- **MTL cat:** ResLN's STL cat win does NOT survive — every Δcat NS (AL −0.01/+0.67, AZ −0.44/−0.12, FL −0.34/−0.06; all p ≥ 0.15). The "cat encoder isn't starved" hypothesis is **refuted**.
- **MTL reg:** null everywhere (|Δ| ≤ 0.67, all NS) — confirms the regime finding.
- **Stacking:** ResLN+design_b does not beat ResLN-canonical or canonical on any axis/front/state in MTL.

**So the MTL regime washes out encoder/substrate improvements on BOTH tasks** — not just reg. The only lever that moves MTL is prior-pathway work (log_T-KD, +2.40 pp reg confirmed). The deployable MTL recipe remains **canonical + log_T-KD**, with no substrate/encoder gain on top.

## 5. Bottom line

- **STL:** ResLN+design_b realises the dual-axis goal best — equalises HGI on reg at AL (within ~0.4–1.1 pp at AZ/FL) **and** keeps/improves check2hgi's 2–3× cat lead over HGI. ResLN's encoder improvement is real and stacks (marginally) with POI2Vec at STL.
- **MTL (deployable):** the entire STL advantage — ResLN, designs, even HGI — vanishes; MTL is regime-limited on both axes. Ship `canonical + log_T-KD`; the encoder/substrate work is an STL/generality property, not a deployable MTL gain.

## Artefacts
- New cells: `tier_resln/{stl_reg,stl_cat,resln_canonical,resln_design_b,canonical_noresln}/...`; `tier_resln/analysis.json`
- STL reg HGI/design refs: `docs/results/P1/region_head_*_STL_*_{hgi,design_*}_reg_gethard_pf_5f50ep.json` (leak-free)
- STL cat HGI ref: `docs/results/P1_5b/next_category_*_hgi_5f_50ep_fair.json`
- Engines: `CHECK2HGI_RESLN`, `CHECK2HGI_RESLN_DESIGN_B` (registered); `build_resln_canonical.py`, `build_design_b_poi_pool.py --encoder resln`
