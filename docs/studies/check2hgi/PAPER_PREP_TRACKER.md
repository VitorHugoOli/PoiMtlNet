# Paper Prep Tracker — Check2HGI Study

**Created:** 2026-04-27 (post-F49). **Owner:** consolidates remaining work *to paper submission*. Distinct from `FOLLOWUPS_TRACKER.md` (per-experiment status) — this file is **paper-deliverable** focused: what claims need to land, what experiments still need to run, what doc rewrites are needed, and what the camera-ready timeline looks like.

For the per-experiment scientific work, see `FOLLOWUPS_TRACKER.md`. For the F49-specific status, see `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §13-14.

---

## 1 · Paper-grade claims — committable now

These are paper-grade and can be written into Methods + Results + Discussion immediately:

| Claim | Evidence | Doc |
|---|---|---|
| **CH16 — Check2HGI > HGI on next-cat, head-invariant at AL+AZ** | Phase-1 8/8 head-state probes positive at p=0.0312 (linear/next_gru/next_single/next_lstm × AL/AZ); Δ +11.58 to +15.50 pp | `research/SUBSTRATE_COMPARISON_FINDINGS.md` §1, §2.1, §5 + `results/probe/` + `results/phase1_perfold/` |
| **CH15 — reframed as head-coupled (was "HGI > C2HGI on reg under STAN")** | Phase-1 Leg II.2: under matched MTL reg head `next_getnext_hard`, AL TOST non-inf at δ=2pp + AZ +2.34 pp p=0.0312 | `research/SUBSTRATE_COMPARISON_FINDINGS.md` §2.2 + `results/phase1_perfold/{AL,AZ}_*_reg_gethard_5f50ep.json` |
| **CH17 — Check2HGI > POI-RGNN external** | POI-RGNN protocol audit + 28-32 pp delta | `docs/baselines/POI_RGNN_AUDIT.md` + `baselines/next_category/poi_rgnn.md` + `baselines/next_category/results/<state>.json` |
| **CH18 — MTL B3 is substrate-specific (Tier A) + H3-alt closes the matched-head reg gap** | Phase-1 Leg III: HGI substitution breaks reg by 30 pp at AL+AZ; F48-H3-alt 5f on AL+AZ+FL closes/exceeds F21c gap | `research/SUBSTRATE_COMPARISON_FINDINGS.md` §3 + `research/F48_H3_PER_HEAD_LR_FINDINGS.md` + `results/hgi/{alabama,arizona}/mtlnet_*_20260427_*` |
| **CH19 — Per-visit context = ~72% of CH16 cat substrate gap (mechanism)** | Phase-1 C4: POI-pooled C2HGI counterfactual at AL — linear probe + matched-head STL agree | `research/SUBSTRATE_COMPARISON_FINDINGS.md` §4 + `results/probe/alabama_check2hgi_pooled_last.json` + `results/check2hgi_pooled/alabama/...` |
| **CH20 Layer 1 — Cat-supervision transfer ≤ |0.75| pp on AL/AZ/FL n=5** | F49 3-way decomp + F49c FL n=5 | `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` |
| **CH20 Layer 2 — Loss-side `task_weight=0` is unsound under cross-attn MTL (methodological)** | Gradient-flow argument + 4 passing tests | `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` + `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py` |
| **CH21 — TOP-LINE: MTL win is interactional architecture × substrate, not transfer** | Synthesis of CH18 + CH19 (substrate) + CH20 (architecture) | `CLAIMS_AND_HYPOTHESES.md §CH21` + `SESSION_HANDOFF_2026-04-27.md §0.3` + `README.md` headline |
| **AL architectural = +6.48 ± 2.4 pp at ~2.7σ** | F49 5f | `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §10 |
| **AZ architectural = −6.02 ± 1.6 pp at ~3.7σ** | F49 5f | same |
| **F49b reproduction gate PASSED** | AL static_weight λ=0 + max_lr=3e-3 OneCycleLR + next_gru, 53.18 ± 4.56 vs 52.27 ± 5.03 (~0.13σ) | `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §5 |

## 2 · Open paper-relevant work — ranked by paper-block importance

### 2.1 — Headline-blocking (must land before submission)

| # | Item | Why it blocks the paper | Cost | Owner |
|---|---|---|---|---|
| **P1** | **F37 STL `next_gru` cat 5f per state** | Required for the cat-side matched-head STL baseline post-F27. Without it, the cat-side MTL-over-STL claim has an asterisk. | ~3h on 4050 | 4050 (user-assigned) |
| **P2** | **F37 STL `next_getnext_hard` reg 5f on FL** | Required for the FL absolute architectural Δ vs STL ceiling (CH19 Layer 3). AL+AZ already have this; FL pending. Without it the FL architectural row of the F49 decomposition table reads "TBD". | ~2h on 4050 (separate from cat F37) | 4050 |
| **P3** | **CA + TX upstream pipelines + 5f H3-alt** | Headline paper covers FL+CA+TX. CA/TX have no Check2HGI embeddings yet (F22/F23 retired into F34/F35). | F34/F35 1f ~12h on Colab T4; F24/F25 5f ~25h | Colab |

### 2.2 — Sharpens claims (nice-to-have for v1, may slip to camera-ready)

| # | Item | Why | Cost | Owner |
|---|---|---|---|---|
| **P4** | **Paired Wilcoxon p-values on F49 cells per fold** | Tightens the AL/AZ "transfer ≈ 0" + "co-adapt ≈ 0" claims with formal p-values. Same-fold-split assumption holds (`--no-folds-cache`, seed 42). | ~30 min on existing JSONs | m4_pro / any |
| **P5** | **Paired Wilcoxon: H3-alt vs predecessor B3 across folds (AL+AZ)** | Formal stat strength for "H3-alt > B3 on reg" — already-tracked open item from `MTL_ARCHITECTURE_JOURNEY.md §9`. | ~30 min | m4_pro / any |
| **P6** | **Seed sweep H3-alt on AL+AZ {0, 7, 100}** | Cross-seed σ for the architectural claim. Already in `FOLLOWUPS_TRACKER` post-H3-alt list. | ~3h MPS | m4_pro |
| **P7** | **`MTL_ARCHITECTURE_JOURNEY.md` post-F49 narrative pass** | Adds the F49 attribution paragraph + clarifies that the F45 α-growth mechanism is *operational* while F49's architecture-dominant finding is *causal*. (Already added §10/§11 prose; could expand §11 if reviewers ask.) | ~30 min | this branch |

### 2.3 — Camera-ready / post-submission

| # | Item | Why deferred | Cost | Owner |
|---|---|---|---|---|
| **P8** | **`POSITIONING_VS_HMT_GRN.md` rewrite for F49** | Document currently cites legacy +14.2 pp transfer + 25 pp architectural overhead — both refuted. Deprecation note added 2026-04-27 to keep readers from citing wrong numbers; full rewrite for camera-ready. | ~2h | this branch |
| **P9** | **Block-internal `ffn_a` ablation** | F49 plan §"B-side processing" notes `_CrossAttnBlock.ffn_a/ln_a*` are NOT frozen in either F49 variant. A "totally-frozen-cat-side-block" variant would isolate purely-architectural-with-no-cat-side-FFN-training — but breaks the reg pipeline (b reads a outputs as K/V via residuals). May require an autograd-detach variant rather than freeze. | ~1-2h dev + ~1h compute | camera-ready |
| **P10** | **α instrumentation per epoch per fold per F49 variant** | Quantitative confirmation of "α growth same regardless of cat freeze" — tightens the architecture-vs-optimizer attribution. Already-tracked in `MTL_ARCHITECTURE_JOURNEY.md §9`. | ~1h dev + re-run on /tmp data | camera-ready |
| **P11** | **FL frozen-cat reg-path instability investigation** | F49c showed frozen-cat reg path on FL has Acc@10 σ = 12 vs 1.4 on loss-side — per-fold reg-best epochs {2, 14, 9, 4, 2} suggest α-growth doesn't engage when cat features are random at FL scale. Worth a paragraph in the paper limitations + a follow-up reproduction at different seeds. | ~30 min seed sweep | camera-ready |
| **P12** | **Extend `experiments/check2hgi_up` to H3-alt (currently B3-only)** | The check2hgi-up embedding-variant study (`experiments/check2hgi_up/run_mtl_b3.py`) uses the predecessor B3 MTL recipe as a fair-comparison harness for embedding variants. Question: does the H3-alt regime change which embedding variant wins? If H3-alt's reg lift is architecture-dominant (per F49), embedding-variant rankings should be stable across B3 and H3-alt — but worth verifying with a small AL spot-check before camera-ready. **Decision needed by the user:** (a) extend run_mtl_b3.py with H3-alt flags + new script run_mtl_h3alt.py, (b) keep B3 as the stable harness and just disclaim in the paper, or (c) defer entirely as out-of-scope. | ~15 min dev + ~30 min compute on AL | user-decision |

---

## 3 · Doc rewrites needed

### 3.1 — Already done (this sweep, 2026-04-27)

- `README.md` — F49 status added at top of "where to start" navigation
- `AGENT_CONTEXT.md` — era updated to "Post-H3-alt + F49 era"; nav table reordered
- `CLAIMS_AND_HYPOTHESES.md` — CH19 added; CH18 attribution refinement; dashboard updated
- `MTL_ARCHITECTURE_JOURNEY.md` — §10 four-phase trajectory now includes F49; cross-references updated
- `NORTH_STAR.md` — F49 attribution refinement footnote (architecture-dominant on AL, etc.)
- `OBJECTIVES_STATUS_TABLE.md` — v5 with F49; mechanism attribution sharpened block
- `PAPER_STRUCTURE.md` — §1.1 paper-reshaping findings now lists F49; FL row of validation status now lists F49 archives
- `FOLLOWUPS_TRACKER.md` — F49 / F49b / F49c rows all done
- `CONCERNS.md §C12` — fully resolved 2026-04-27
- `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` — §13/§14 final n=5 outcome
- `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` — review-pass mods integrated
- `research/POSITIONING_VS_HMT_GRN.md` — deprecation note pointing to F49 (full rewrite deferred to camera-ready, P8 above)
- `SESSION_HANDOFF_2026-04-27.md` — final state captured

### 3.2 — Pending rewrites

- ~~`results/RESULTS_TABLE.md` — needs F49 cells per state.~~ ✅ Done 2026-04-27 (post-audit): Phase-1 substrate cells + F49 3-way decomposition cells appended.
- **`docs/studies/check2hgi/research/POSITIONING_VS_HMT_GRN.md`** — full rewrite for camera-ready (P8 above). Quantitative numbers updated (F49 + Phase-1); HMT-GRN contrast preserved.
- **A new `research/F49_METHODOLOGICAL_NOTE.md`** — Layer 2 paper-prep document. Standalone short note (~3 pp) on "loss-side ablation is unsound under cross-attn MTL" with the gradient-flow argument + the encoder-frozen alternative + applicability to MulT/InvPT. Could be a stand-alone methods note OR a paragraph + appendix in the main paper. Decide closer to write-up time.
- **A new `research/CH21_JOINT_CLAIM.md`** — TOP-LINE paper claim doc. Synthesise CH18 (substrate-specific MTL) + CH19 (per-visit context mechanism) + CH20 (architecture-dominant + Layer 2). Frames the paper's main contribution as "MTL win is interactional architecture × substrate, not transfer." Currently lives only in `CLAIMS_AND_HYPOTHESES.md §CH21` + `SESSION_HANDOFF_2026-04-27.md §0.3` + `README.md` headline; needs its own paper-prep doc with figure suggestions + drafting hooks.

---

## 4 · Risk register (paper-relevant)

| Risk | Mitigation | Status |
|---|---|---|
| F37 FL F21c lands and shows STL > MTL by a wide margin → CH18 claim weakens at FL | Frame as "AL exceeds, FL ties, AZ closes 75%" rather than "MTL beats STL universally". Paper section already uses this framing. | monitored |
| Reviewer asks for paired Wilcoxon p-values on F49 transfer ≈ 0 claim | Run P4 above (~30 min). | accept |
| Reviewer asks for FL n=5 frozen-cat instability explanation | Per-fold reg-best epoch table is in F49 results doc §13. Mention in limitations + offer P11 as future work. | accept |
| Reviewer challenges the H3-alt mechanism (architecture vs α growth conflict) | Both are true at different levels; the operational story is α growth, the causal story is architecture. F49 + MTL_ARCHITECTURE_JOURNEY §10 lay this out clearly. | accept |
| CA + TX upstream pipeline takes longer than expected (data quality issues, Colab quota) | Prepare AL+AZ+FL-only paper as a fallback. AL+AZ+FL three-state coverage is already paper-grade for CH18+CH19. | monitor C13 |
| `POSITIONING_VS_HMT_GRN.md` legacy numbers leak into write-up by accident | Deprecation note added 2026-04-27 prevents quoting wrong numbers; F49 is the canonical source. | mitigated |

---

## 5 · Submission-readiness checklist

For BRACIS 2026 (or whatever venue), the headline submission needs:

**Substrate-side (Phase 1):**
- [x] CH16 (substrate): Check2HGI > HGI on cat F1 head-invariant — Phase-1 confirms AL+AZ at p=0.0312, 8/8 head-state probes
- [x] CH15 (substrate): reframed as head-coupled — under matched MTL reg head C2HGI ≥ HGI everywhere
- [x] CH17 (external baselines): POI-RGNN audit — landed; new ports of STAN, MHA+PE, REHDM in `baselines/` tree
- [x] CH18 (substrate-specific MTL): MTL+HGI breaks reg by 30 pp at AL+AZ — landed (Tier A)
- [x] CH19 (mechanism): per-visit context = ~72% of substrate gap at AL — landed
- [ ] **F36 FL Phase-2 substrate grid** (probe + cat STL × 2 + reg STL × 2 + MTL counterfactual) — `PHASE2_TRACKER.md` queued, FL data on disk
- [ ] **P2-CA-grid** (CA Phase-2 grid, gated on P2-CA-up CA upstream pipeline)
- [ ] **P2-TX-grid** (TX Phase-2 grid, gated on P2-TX-up TX upstream pipeline)

**Architecture-side (F49):**
- [x] CH18 (resolved gap): H3-alt closes the F21c reg gap — landed AL+AZ+FL (Tier A)
- [x] CH20 Layer 1+2 (transfer null + methodological note): F49 — landed AL+AZ+FL n=5
- [ ] CH20 Layer 3 (FL absolute architectural Δ vs STL F21c): blocked on F37 (4050)

**Joint claim:**
- [x] CH21 (top-line): MTL win is interactional architecture × substrate, not transfer — synthesised from CH18 + CH19 + CH20
- [ ] **Headline paper section** drafting CH21 as the top-line story (not "MTL transfers signal")

**Statistical strengthening (cheap):**
- [ ] Paired Wilcoxon p-values (P4 + P5): ~1h total on existing JSONs
- [ ] CH19 cross-state replication (Phase-2 C4 at FL — optional)

**Paper drafting:**
- [ ] Methods section drafted with F27 cat-head + B3 + H3-alt + F49 attribution + Phase-1 substrate framework
- [ ] Results section with 3-state × 4-cell F49 decomposition table + 8-cell Phase-1 substrate-Δ table + MTL counterfactual table
- [ ] Methodological appendix with Layer 2 (loss-side ablation under cross-attn) + matched-head STL policy revision (Phase-1 §5)
- [ ] Limitations: AL is dev state; FL frozen-cat instability (F49 §13); CA/TX gated on Check2HGI training; CH15 was head-coupled (recorded but reframed)
- [ ] Reviewer-rebuttal ammo: per-fold JSONs in `results/{phase1_perfold,probe,paired_tests}/` + F49 reproduction-gate result + 4 passing regression tests + faithful baseline ports in `baselines/`

---

## 6 · How to use this doc

When picking up the branch:
1. Read `SESSION_HANDOFF_2026-04-27.md` (most-recent state).
2. Read `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` (the F49 final picture).
3. Then read this file (PAPER_PREP_TRACKER) for what's left for the paper.
4. Pick a P1 item from §2.1 (or P4 from §2.2 if you want a cheap morale-win statistical-strength pass).
5. Update both this file and `FOLLOWUPS_TRACKER.md` when work lands.

Cross-references:
- Per-experiment open items: `FOLLOWUPS_TRACKER.md`
- Per-claim catalog: `CLAIMS_AND_HYPOTHESES.md`
- Risks + concerns: `CONCERNS.md`
- Mechanism narrative: `MTL_ARCHITECTURE_JOURNEY.md`
- Final F49 numbers: `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`
