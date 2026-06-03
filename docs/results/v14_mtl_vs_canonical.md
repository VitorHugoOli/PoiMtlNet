# v14 MTL vs Matched Canonical — Full Multi-Seed Result (2026-06-03)

**What this is.** The first paper-grade MTL evaluation of the **v14** substrate
(`check2hgi_design_k_resln_mae_l0_1`) against a **matched canonical** baseline
(`check2hgi`, frozen GCN v11 substrate), run in one harness. It upgrades the prior
**2-fold seed42 FL pilot** (`embedding_eval/FINAL_SYNTHESIS.md` §"Option (b) step 1") to
**5-fold × 4-seed {0,1,7,100}, three states, leak-free**.

**Headline.** Under MTL, **v14 ≈ matched canonical** (tie at FL on both tasks; mixed within
noise at the small states). The strong v14 *STL* dual-axis gains (next-cat 67.36 ≫ HGI; next-reg
0.7024, closing ~69% of the canon→HGI gap) **do NOT survive MTL deployment** — confirming the
documented **regime finding** ("substrate gains wash out under the cross-attention MTL regime;
v14 is STL-only, NO MTL benefit"). This is a **confirmation** of the prior studies, not a
contradiction (see §Reconciliation).

---

## Protocol
- **Engines:** v14 `check2hgi_design_k_resln_mae_l0_1` vs canonical `check2hgi` (frozen v11 GCN).
- **Task / recipe:** `--task mtl --task-set check2hgi_next_region`, **B9** (FL) / **H3-alt** (AL/AZ)
  per NORTH_STAR, `mtlnet_crossattn`, `static_weight --category-weight 0.75`,
  `--cat-head next_gru --reg-head next_getnext_hard`, `--task-a-input-type checkin
  --task-b-input-type region`.
- **Seeds:** {0, 1, 7, 100} (reporting set; NOT dev-seed 42). **5 folds, 50 epochs.**
- **Leak-free:** seeded per-fold `region_transition_log_seed{S}_fold{N}.pt` (verified loaded).
- **log_T-KD OFF** (`--log-t-kd-weight 0.0`) — deliberate, to isolate the **substrate swap**
  (GCN→v14) from the v12 KD axis. (v12 default KD ON would lift small-state reg further; §0.8.)
- **Why matched canonical, not frozen §0.1:** the same canonical substrate yields 53.7–64.5 reg@10
  in this harness depending on selector/basis vs §0.1's 63.27 — a documented harness/protocol
  offset. The matched in-harness baseline is the only valid Δ; §0.1 is shown for reference only.
- **Metrics:** reg = `top10_acc_indist` (Acc@10), cat = macro-F1.
- **Selector note:** results reported on three checkpoint-selection bases (below). The **corrected
  joint selector** `joint_geom_simple = sqrt(cat_F1 · reg_Acc@10)` is now the **code default**
  (C21 fix promoted 2026-06-03 — see `CANONICAL_VERSIONS.md` §selector, `CONCERNS.md` §C21).
- **Raw artefacts:** `scripts/_v14_run/` (drivers, manifests, per-run logs); per-run dirs under
  `results/check2hgi_design_k_resln_mae_l0_1/<state>/` and `results/check2hgi/<state>/`.

## Results (v14 vs matched canonical; Δ>0 ⇒ v14 better)

### JOINT-GEOM-SIMPLE — the deployable single-checkpoint selector (corrected default)
| state | v14 reg@10 | canon reg@10 | Δreg | v14 catF1 | canon catF1 | Δcat |
|---|---|---|---|---|---|---|
| FL | 61.21±0.45 | 61.54±0.13 | **−0.33** | 66.73±0.32 | 66.77±0.15 | **−0.04** |
| AL | 50.14±0.83 | 48.00±0.33 | **+2.14** | 46.50±0.22 | 45.29±0.13 | **+1.21** |
| AZ | 37.78±0.29 | 38.79±0.22 | **−1.01** | 48.52±0.21 | 47.81±0.27 | **+0.72** |

### DIAGNOSTIC-BEST — per-task own-best epoch (the §0.1 reporting convention)
| state | v14 reg@10 | canon reg@10 | Δreg | v14 catF1 | canon catF1 | Δcat | §0.1 ref (reg/cat) |
|---|---|---|---|---|---|---|---|
| FL | 61.28±0.05 | 61.49±0.27 | −0.21 | 70.26±0.07 | 70.34±0.06 | −0.09 | 63.3 / 68.6 |
| AL | 47.23±0.32 | 49.46±0.47 | −2.23 | 46.78±0.14 | 45.96±0.21 | +0.82 | 50.2 / 40.6 |
| AZ | 38.27±0.33 | 40.29±0.13 | −2.02 | 48.75±0.16 | 48.86±0.26 | −0.11 | 40.8 / 45.1 |

### JOINT-LEGACY — the v11 broken selector `0.5*(cat_f1+reg_f1)` (for reference; do NOT use)
| state | v14 reg@10 | canon reg@10 | Δreg | v14 catF1 | canon catF1 | Δcat |
|---|---|---|---|---|---|---|
| FL | 55.99±2.27 | 55.92±2.88 | +0.08 | 69.45±0.22 | 69.33±0.31 | +0.12 |
| AL | 49.49±0.94 | 48.00±0.33 | +1.48 | 46.68±0.21 | 45.29±0.13 | +1.39 |
| AZ | 37.42±0.28 | 38.79±0.22 | −1.38 | 48.58±0.26 | 47.81±0.27 | +0.77 |

## Interpretation
- **FL (large state): clean tie** on both axes and both selection bases → **no MTL benefit** from v14.
  This is the headline.
- **Small states (AL/AZ):** the per-task *ceiling* (diagnostic-best) gives canonical a ~2pp reg edge,
  but on the *deployable* geom_simple checkpoint v14 edges canonical at AL (+2.1 reg / +1.2 cat) and
  is within noise at AZ. Net: v14 is not worse where it counts (deployment), modestly better at AL.
- **No decisive v14 MTL reg win anywhere.** Substrate gains wash out under cross-attn MTL — exactly
  the regime finding.
- **Selector observation (independent value):** the corrected geom_simple selector recovers the
  deployable reg from the broken-selector's noisy 56.0±2.3 to 61.2±0.5 at FL (≈ the diagnostic-best
  ceiling) — concrete evidence the C21 fix matters.

## Reconciliation with prior studies (audit, 2026-06-03)
An independent audit reconciled these MTL numbers against `embedding_eval` and
`substrate-protocol-cleanup`:
- The v14 "improves next-cat AND next-reg, closes ~69% of the HGI gap" headline is an **STL**
  (Part-1 substrate) result (`embedding_eval/FINAL_SYNTHESIS.md` §FINAL, lines ~464–478). It was
  **never an MTL claim**. Every doc carrying it also carries the explicit **"STL-only, NO MTL
  benefit"** qualifier (FINAL_SYNTHESIS banner; `CANONICAL_VERSIONS.md` §v14: *"Do not cite v14 as
  an MTL improvement"*; NORTH_STAR §regime finding).
- Therefore our MTL tie is a **confirmation**, not a failure to replicate. Our full multi-seed run is
  the stronger version of the docs' own hedged 2-fold seed42 pilot (cat −0.21 / reg +0.03 there;
  ≈ tie here) — same verdict, more power.
- **Attribution note:** `substrate-protocol-cleanup` (closed 2026-05-29) predates v14 (blessed
  2026-06-02) and never names it; v14 is an `embedding_eval` product. Both studies state the
  identical STL-only law, so the conclusion is unchanged.
- **What our MTL run does NOT test:** the **STL HGI-gap** claim (v14 0.7024 vs canon 0.6943 vs HGI
  0.7060). That is a different regime and our MTL comparison had no HGI region arm. It is tested by
  the STL verification sweep below.

## STL verification — prior-study claims REPLICATE (2026-06-03, FL, seeds {0,1,7,100}, 5-fold)
Re-ran the prior studies' STL protocol to assure their numbers reproduce. Drivers:
`scripts/_v14_run/stl_driver.sh` (v11/v14) + `stl_hgi_driver.sh` (HGI); reg via
`p1_region_head_ablation.py --heads next_stan_flow --input-type region --region-emb-source <eng>
--per-fold-transition-dir output/check2hgi/florida` (seeded per-fold log_T, leak-free); cat via
`train.py --task next --cat-head next_gru`. Multi-seed mean ± SD over {0,1,7,100}.

### STL next-reg Acc@10 — the HGI-gap claim, replicated EXACTLY
| engine | this run | prior (FINAL_SYNTHESIS) | Δ |
|---|---|---|---|
| v11 (canonical) | 69.43 ± 0.05 | 69.43 | **+0.00** |
| **v14** | **70.24 ± 0.10** | **70.24** | **+0.00** |
| HGI | 70.62 ± 0.12 | 70.60 | +0.02 |

- Ordering **canon (69.43) < v14 (70.24) < HGI (70.62)** confirmed. v14 closes
  **(70.24−69.43)/(70.62−69.43) = ~68%** of the canon→HGI gap (prior claim ~69%); **HGI retains a
  −0.38pp edge** (prior −0.36pp). The STL gap claim reproduces to the second decimal.

### STL next-cat macro-F1 — relative claim holds (v14 ≈ canon ≫ HGI)
| engine | this run (multi-seed) | prior (seed42) | Δ |
|---|---|---|---|
| v11 (canonical) | 65.79 ± 0.11 | 67.32 | −1.53 |
| v14 | 65.88 ± 0.14 | 67.36 | −1.48 |
| HGI | 33.74 ± 0.28 | 34.29 | −0.55 |

- **v14 ≈ v11 (65.88 vs 65.79, within noise) and both ≫ HGI (~33.7)** — the relative claim holds.
- The ~1.5pp absolute shortfall vs the prior 67.3 is a **seed offset, not a regression**: the prior
  next-cat table was **seed42** (the development seed, which overshoots per CLAUDE.md §dev-vs-reporting
  seed); our **multi-seed {0,1,7,100}** lands ~65.8. v14's "67.36 ≈ frozen-canon ≫ HGI" claim is
  therefore confirmed in *relative* terms; the absolute headline number was seed42-specific.

### STL verdict
**The prior STL claims fully replicate.** v14's STL next-reg gain (+0.81pp over canonical, closing
~68% of the HGI gap, HGI keeping a small edge) and its cat dominance over HGI both reproduce. This,
together with the MTL tables above, gives the complete picture: **v14's substrate gains are real at
STL but do NOT transfer to MTL** — exactly the documented regime finding. The user's "not replicable"
was a STL-vs-MTL regime distinction, now resolved on both sides with paper-grade evidence.
Raw artefacts: `scripts/_v14_run/stl_manifest.tsv`, `docs/results/P1/region_head_florida_region_5f_50ep_<eng>_s<seed>.json` (reg),
`results/<eng>/florida/next_*ep50*/` (cat).

## See also
- `CANONICAL_VERSIONS.md` §v14 (engine identity, build, STL-only grade) + §selector (C21 default flip).
- `embedding_eval/FINAL_SYNTHESIS.md` (the v14 STL authority + the 2-fold MTL pilot this supersedes).
- `CONCERNS.md` §C21 (joint-selector fix), `future_works/joint_selection_and_loss_combination.md`.
- `NORTH_STAR.md` (the regime finding), `RESULTS_TABLE.md §0.1` (v11 paper canon, per-task basis).
