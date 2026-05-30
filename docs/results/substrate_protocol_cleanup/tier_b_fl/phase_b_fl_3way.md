# Tier B FL — THREE-WAY substrate comparison (canonical c2hgi vs designs B/J/M/L vs HGI), framed as gap-closure

**Date**: 2026-05-29
**Scope**: Florida, seed=42, 5 folds. STL three-way from existing P1 JSONs; MTL three-way from the FL B9 Tier B cells (`tier_b_fl/`); the STL-α=0 vs MTL-α=0 ISOLATION cell run this session.
**Method**: RAW per-fold `top10_acc` (disjoint = each head's own best-val epoch), paired Wilcoxon one-sided (scipy exact, n=5). Gap-closure % = (Δdesign−canon) / (HGI−canon).
**Why three-way**: per `merge_design/STATE.md`, the designs exist to *close the canonical→HGI next-region gap while preserving cat*. A designs-vs-canonical-only comparison (the prior FL extension) cannot answer "how much of the gap is closed" or "where HGI sits." This doc fixes that by anchoring every comparison to HGI as the reference ceiling.

> **UPDATE 2026-05-29 — the HGI MTL+F1 leg IS now evaluated** (user-approved follow-up; a dedicated agent built HGI FL embeddings + ran HGI MTL under B9). Full numbers in [`hgi_mtl_fl.md`](hgi_mtl_fl.md), integrated into §2 below. **Ceiling answer: HGI does NOT beat canonical in MTL reg at FL** (disjoint 64.49 vs 63.98, Δ+0.51, p=0.41 NS). The earlier "un-evaluated leg" caveat is resolved; the concurrent HGI build flagged below was that sanctioned follow-up.
>
> ⚠ (Historical note, now resolved) During this agent's run a concurrent session was observed building HGI FL embeddings (`build_hgi_fl_train.py`). That was the user-sanctioned HGI-ceiling agent, correctly left untouched; its result is integrated here.

---

## 1. STL three-way (FL, reg_gethard = next_stan_flow WITH α·log_T prior, alpha_init=0.1)

Per-fold `top10_acc` at the top10-best epoch. Canonical = `check2hgi`; designs = `_leakfree` builds (the non-`_leakfree` P1 files are the LEAKY ~82 % runs and are NOT used); HGI = `hgi`.

| Substrate | reg Acc@10 (%) | Δ vs canonical | p_gt vs canon | Δ vs HGI | HGI>x p | gap-closure |
|---|---:|---:|---:|---:|---:|---:|
| canonical c2hgi | 69.22 | — | — | −2.12 | — | — |
| **B** (POI2Vec @ pool) | 69.93 | +0.71 | 0.0625 (ns) | −1.41 | 0.0312 | 34 % |
| **J** (H + λ-anchor) | **70.34** | **+1.12** | **0.0312 ✓** | −0.99 | 0.0312 | **53 %** |
| **M** (B + cosine distill) | 70.11 | +0.89 | 0.0625 (ns) | −1.23 | 0.0312 | 42 % |
| **HGI** | **71.34** | +2.12 | 0.0312 ✓ | — | — | — |

Per-fold (gethard): canonical `[69.66,69.57,69.56,68.56,68.75]`; J `[69.79,70.87,70.60,69.80,70.66]`; HGI `[71.82,71.93,71.63,70.51,70.79]`.

**STL reads:** canonical→HGI gap = **+2.12 pp**. Only **J** is Wilcoxon-strict over canonical (+1.12 pp, p=0.0312), closing **53 %** of the gap; B/M are positive but not significant (p=0.0625). Every design remains **strictly below HGI** (HGI>design p=0.0312 for all). The design STL advantage at FL is **real but modest, and only partial gap-closure**.

### 1b. STL no-prior cross-check (next_gru, NO log_T) — merge_design Test 2

| Substrate | reg Acc@10 (%) | Δ vs canonical | Δ vs HGI |
|---|---:|---:|---:|
| canonical | 68.36 | — | −2.50 |
| **J** | 69.22 | +0.86 (p=0.0312 ✓) | −1.64 (p=0.0312) |
| HGI | 70.86 | +2.50 | — |

Removing the prior (next_gru) does NOT erase the J advantage (+0.86 pp, still strict) and *widens* HGI−J to +1.64 pp — the embedding-quality gap to HGI is real, not a Markov-prior artefact. (Reconciles exactly with `merge_design/T2_FINDINGS.md`.)

---

## 2. MTL three-way (FL, B9 recipe, α·log_T trainable alpha_init=0.1), TWO FRONTS

Disjoint = each head's own best epoch (oracle); joint = single epoch maximising `sqrt(cat_f1·reg_top10)` (deployable). reg one-sided design>canon.

| Substrate | reg DISJOINT (Δ, p) | reg JOINT (Δ) | cat DISJOINT (Δ) | cat JOINT (Δ) |
|---|---:|---:|---:|---:|
| canonical c2hgi | 63.98 | 61.14 | 70.49 | 66.98 |
| **B** | 63.82 (−0.16, p=0.875) | 57.65 (−3.49) | 68.61 (−1.88) | 67.33 (+0.35) |
| **J** | 64.06 (+0.08, p=0.312) | 57.52 (−3.63) | 68.80 (−1.69) | 68.01 (+1.03) |
| **L** (Lever-5 KL distill) | 63.97 (−0.01, p=0.500) | 57.67 (−3.48) | 68.71 (−1.78) | 67.68 (+0.70) |
| **HGI** | **64.49 ± 0.55 (+0.51, p=0.41 NS)** | 63.16 (+2.02, p=0.0625) | **34.84 (−35.6)** | 33.99 (−32.99) |

**MTL reads:** no design moves disjoint MTL-reg over canonical (all |Δ|≤0.16 pp, none significant; best is J +0.08 pp p=0.31). On the deployable joint front all designs sit ~−3.5 pp below canonical — an artefact of their slightly lower cat dragging the geom-selected epoch to a worse reg point, NOT a reg-quality loss (disjoint reg is flat). The cat disjoint −1.7 to −1.9 pp is the **build-scope CheckinEncoder re-init confound** (design builds re-train a fresh CheckinEncoder), consistent with AL D3 — NOT a substrate cost (and J/L don't even touch the cat path).

**THE CEILING (HGI MTL, evaluated 2026-05-29):** HGI — which wins STL reg by +2.12 pp — does **NOT** beat canonical in MTL disjoint reg (64.49 vs 63.98, Δ+0.51, Wilcoxon **p=0.41 NS**). Its STL→MTL drop is ~6.4 pp (70.9→64.5), landing it on top of canonical (63.98) and the designs (63.8–64.1). The marginal joint-reg "edge" (+2.02, p=0.0625) is an artefact: HGI's cat collapses to 34.84 % F1 (−35.6 pp — HGI is non-viable as an MTL substrate, as expected), so its geom-selected epoch ≈ its reg-best epoch (no cat to trade off), unlike canonical/designs which sacrifice reg for cat at the joint point.

**Designs close 0 % of any MTL canonical→HGI gap — because there is NO MTL gap to close.** The strongest reading now available: **MTL flattens EVERYONE.** HGI's clear STL reg advantage vanishes under B9 joint training (64.49 ≈ canonical 63.98 ≈ designs 63.8–64.1). Even the substrate that wins at STL cannot carry its advantage into MTL. So the designs are not "failing to carry HGI's advantage" — HGI itself has no MTL advantage to carry. **The FL reg bottleneck is the joint-training regime, not the substrate.**

---

## 3. THE ISOLATION CELL — STL-stan-α=0 vs MTL-stan-α=0 (regime or head?)

The single clean experiment the AL/AZ re-audit lacked: take the EXACT reg head (`next_stan_flow`) with α frozen to 0 (`freeze_alpha=true, alpha_init=0.0`, so `final_logits = stan_logits` alone — log_T contributes literally nothing) and run it in STL (single-task next_region) vs the existing MTL-α=0 cell.

FL reg chance = 10/4703 ≈ **0.213 %** (top10).

| Config | reg Acc@10 | per-fold | vs chance |
|---|---:|---|---|
| **STL stan-α=0 canonical** | **72.74 %** | [73.25,72.01,72.57,72.58,73.31] | **341× chance** |
| **STL stan-α=0 design_b** | **73.12 %** | [73.89,72.60,72.89,72.72,73.49] | 343× chance (Δ vs canon +0.37 pp, p=0.0312) |
| MTL stan-α=0 canonical | ~0.03 % all-epoch-mean (best-ep 0.13 %) | — | **0.1× chance (floor)** |
| MTL stan-α=0 design_b | ~0.03 % all-epoch-mean (best-ep 0.14 %) | — | 0.1× chance; Δ vs canon +0.003 pp, p=0.50 |

### Verdict: it is the REGIME, not the head.

The **same head**, the **same α=0 config**, the **same embeddings**:
- under **STL** the stan encoder branch LEARNS region at **~73 % Acc@10** (341× chance) — fully, without any log_T prior;
- under **MTL** (B9 joint, 50 ep) the identical branch sits **at/below chance (~0.03 %)**.

So the MTL-α=0 floor is **definitively a property of the joint-training regime**, not of the next_stan_flow head and not of the substrate. Under the B9 joint config the α·log_T anchor supplies essentially ALL usable MTL-reg signal; the substrate-carrying encoder branch is driven to floor — and removing the anchor uncovers no hidden substrate gain at either STL→MTL boundary.

**Bonus observation:** STL-α=0 (72.74 %) is *higher* than STL-with-prior gethard (69.22 %). At STL the converged encoder doesn't need the prior — consistent with Test 2's finding that the prior is not what carries the STL signal.

**Hedge (carry the AL verification rigor):** α=0 is an OUT-OF-TRAINING config. The defensible claim is regime-and-config-scoped: *"under the B9 50-epoch joint regime the reg encoder branch is driven to floor; the same branch trains to ~73 % when the joint objective is removed."* This is NOT an absolute architectural law that "MTL can never learn region in the encoder branch" — it is specific to this recipe/epoch budget. But the STL↔MTL contrast is now a clean SINGLE-CELL apples-to-apples (same head, same config, same state, same embeddings), which the AL re-audit's cross-state/cross-design contrast was not.

---

## 4. Corrected FL headline

The prior FL-extension headline — *"the AL/AZ STL→MTL substrate collapse REPEATS at FL"* — over-claims, because it implies a large STL design advantage that then collapses. The accurate, hedged FL story:

1. **The FL STL design advantage over canonical is already SMALL** (J +1.12 pp with prior / +0.86 pp no-prior; B/M not even Wilcoxon-strict) and only **partially closes** the canonical→HGI gap (J: 53 % with prior; the remaining ~1 pp to HGI is real embedding-quality, per Test 2/2½).
2. **In MTL, designs give NO gain over canonical on either front** (disjoint |Δ|≤0.16 pp ns; joint −3.5 pp is a cat-driven geom-selection artefact, not reg loss).
3. **The MTL reg encoder is anchor/regime-limited, NOT substrate-limited.** The isolation cell shows the encoder branch CAN learn region (~73 % at STL-α=0) but is driven to floor under the B9 joint regime — so even a *better* substrate cannot express itself in MTL-reg under this recipe.
4. **The HGI ceiling confirms it independently: MTL flattens everyone.** HGI (the STL winner, +2.12 pp) drops to 64.49 ≈ canonical 63.98 in MTL reg (p=0.41 NS). No substrate — designs OR HGI — beats canonical in MTL reg at FL. Three converging lines now agree: (a) STL-α=0 73 % vs MTL-α=0 0.03 % (regime kills the encoder); (b) designs null in MTL; (c) even HGI null in MTL. The bottleneck is unambiguously the joint-training regime, and the fix belongs to `mtl_improvement` (make the reg encoder learn region under joint training), NOT to substrate engineering or more priors.

**Precise corrected statement:** *"At FL the small STL design reg advantage (J +0.86–1.12 pp, partial 53 % gap-closure to HGI) does not survive MTL — designs are indistinguishable from canonical in MTL-reg on both fronts. The cause is the joint-training regime, not the substrate: the isolation cell shows the identical α=0 reg encoder branch learns region at ~73 % under STL but floors (~0.03 %) under MTL. FL's ~10× data does NOT rescue the encoder branch under the 50-epoch joint regime. This DISTINGUISHES anchor/regime-dominance (the cause) from any FL-specific Markov-saturation explanation."*

---

## 5. Cat axis (non-inferiority framing)

Designs show a uniform ~−1.7 to −1.9 pp disjoint cat drop in MTL. This is **NOT a substrate property** — it is the build-scope confound (every design build re-initialises and re-trains the CheckinEncoder for 500 ep, rewriting the cat input wholesale; quantified at AL D3 = 100 % of cat-input cells changed, meanabs ≈ 1.2). J and L don't even touch the cat path, yet show the same drop. A region-only build (reuse canonical `embeddings.parquet` byte-identical, swap only `region_embeddings.parquet`) would be expected to recover "cat flat"; the current builds cannot test that. TOST-style non-inferiority cannot be claimed on the confounded comparison — but the drop is attributable to the build harness, not the merge mechanism.

---

## Artefacts

- STL gethard three-way: `docs/results/P1/region_head_florida_region_5f_50ep_STL_FLORIDA_{check2hgi,design_{b,j,m}…_leakfree,hgi}_reg_gethard_pf_5f50ep.json`
- STL no-prior (Test 2): `docs/results/P1/region_head_florida_region_5f_50ep_T2_FL_{check2hgi,design_j,hgi}_reg_nogru_nolog_t_pf_5f50ep.json`
- MTL B9 cells: `docs/results/substrate_protocol_cleanup/tier_b_fl/{mtl_canonical,mtl_design_b,mtl_design_j,mtl_design_l}/florida/seed42/mtlnet_*/`
- MTL α=0 cells: `tier_b_fl/{a0_canonical,a0_design_b}/...`
- **ISOLATION cells (this session):** `tier_b_fl/stl_a0_{canonical,design_b}/region_head_florida_region_5f_50ep_stl_a0_{canonical,design_b}_florida.json`
- Isolation launcher: `/tmp/fl_logs/run_stl_a0.sh` (detached, marker `stl_a0.DONE`)
- MTL analyzer: `scripts/substrate_protocol_cleanup/analyze_tier_b_fl.py`
