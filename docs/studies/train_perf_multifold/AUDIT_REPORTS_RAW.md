# Raw audit reports (workflow wf_5f410c3d-ef8, 9 agents)

## Investigation: fold-variance

Confirmed. The committed board JSON (`docs/results/closing_data/a40/al_champG_a40_s0.json`) shows `cat_per_fold=[63.344, 64.632, 64.733, 64.194, 59.339]` and `reg_per_fold=[71.845, 68.807, 73.174, 70.738, 63.665]` — fold 5 is the low outlier on both, and also has the earliest best-epochs (`cat_best_epochs[4]=14`, `reg_best_epochs[4]=20`), consistent with a lower, earlier-plateauing val ceiling. I have everything needed.

---

# Fold-5 outlier diagnosis — AL champion-G seed-0 (StratifiedGroupKFold)

## 1. How the 5 folds are built (`src/data/folds.py`)
- `_create_check2hgi_mtl_folds` builds **one** split shared by both tasks. The splitter is `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)` (**folds.py:1381–1383**), and the split call is `sgkf.split(X, y_cat, groups=userids)` (**folds.py:1393**).
- `y_cat` (the stratification label) = `_map_categories(df['next_category'])`, an int 0–7 from `load_next_data` (**folds.py:1226**, def **731–769**, mapping **339–341**, `CATEGORIES_MAP` has 8 classes incl. `None`).
- `groups = userids` → **user-disjoint** folds (a user's rows never straddle train/val).
- **Region labels are loaded but NOT used by the splitter.** `y_region = region_df["region_idx"]` (**folds.py:1250**) and `last_region_idx` (**1298**, consumed only by the STAN-Flow dual-tower aux gate, **1284–1289**) feed the model, not the partition. So the split balances `next_category` only; region support, user transition-predictability, and per-user category concentration are left to chance.

Reproduction matched the board exactly (96 326 rows, 1 101 users, 1 052 regions; no NaN-drop), so this analysis is on the true frozen seed-0 partition.

## 2. Per-fold VAL metrics (reproduced split)

| fold | n_val | val users | distinct reg | cold reg (OOD) | cold-sample % | tail(≤5) sample % | mean tgt reg global-freq | **stay-rate** (reg==last) | per-user cat-entropy | **cat F1** | **reg@10** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 19265 | 222 | 742 | 37 | 0.68 | 0.82 | 721 | 0.330 | 1.409 | 63.38 | 72.11 |
| 2 | 19264 | 218 | 711 | 34 | 0.40 | 0.78 | 675 | 0.318 | 1.412 | 64.40 | 68.95 |
| 3 | 19265 | 221 | 670 | 27 | 0.47 | 0.60 | 775 | **0.356** | 1.416 | 64.55 | **73.13** |
| 4 | 19267 | 218 | 720 | 40 | 0.64 | 0.83 | 700 | 0.345 | 1.405 | 64.00 | 70.62 |
| **5** | 19265 | 222 | 727 | **46** | 0.55 | **0.846** | **644** | **0.296** | **1.380** | **59.60** | **63.85** |

Sizes are balanced (n_val ≈ 19 265, ~220 users each), so it is **not** a sample-count artifact. **Next_category class proportions are bit-identical across all 5 folds** (`[.179,.065,.342,.037,.055,.271,.051,0]`) — stratification works perfectly, so **category class imbalance is ruled out** as the cause of the cat deficit.

## 3. Hypothesis — backed by the numbers
Fold 5's deficit is a **user-composition draw**, not noise. The held-out unit is the *user* (only ~220 users/fold from 1 101 total), and fold 5 drew a jointly harder cohort. Two mechanisms, both quantified:

**Reg deficit (−6 to −9 pp) ← self-transition (stay) rate.** STAN-Flow exploits `last_region_idx`. Fold 5's val users self-loop only **29.6 %** of the time vs the global 32.9 % and fold 3's 35.6 % — the weakest "gimme" signal of any fold. Across the 5 folds, **stay-rate vs reg@10 Pearson r = +0.907**; fold 5 is lowest on both. Fold 5 also has the **most cold/OOD regions (46)**, **least-popular targets** (mean global region-freq 644, lowest), and **highest tail-region mass (0.846 %)** — all pushing reg@10 down. Note cold-sample mass is only ~0.5 %, so **cold/OOD regions are a secondary, not primary, driver**; the dominant lever is the transition (stay) predictability.

**Cat deficit (−5 pp) ← per-user category concentration (a macro-F1 effect).** The metric is `cat_macro_f1`. Fold 5's users are the **most category-concentrated** (lowest per-user next-cat entropy, **1.380** vs 1.40–1.42). Concentrated users let the model lean on the dominant `Food` class (boosts accuracy but starves minority-class recall → lower macro-F1). Across folds, **per-user cat-entropy vs cat-F1 r = +0.973**.

**Joint drop is a single underlying factor.** **cat-F1 vs reg@10 across folds r = +0.866**, and stay-rate predicts *both* (r = +0.907 reg, +0.800 cat). Fold 5's users are simply less repetitive/more dispersed, so both heads underperform together. Corroborating: fold 5 has the earliest best-epochs (cat 14, reg 20 vs 15–18 / 28–38) — its val ceiling is genuinely lower and plateaus sooner, not an under-training fluke.

## 4. Recommendation — report it (expected CV variance), don't refit the split
This is **genuine, structural CV variance from user heterogeneity**, deterministic for seed-0, not a bug. The single-label SGKF *can only* stratify on `next_category` while grouping on `userid`; it has no mechanism to also balance the ~1 052-class region support or user-level stay-rate, and with only 1 101 users a 5-way user split is inherently lumpy (one fold lands the dispersed cohort).

- **Report as-is with multi-seed CIs.** Averaging over the **n=20** grid (seeds {0,1,7,100} × 5 folds) re-draws the user partition four ways, so the fold-5-style draw is diluted and the std/CI tightens. The per-fold spread is *signal about user heterogeneity*, worth a one-line robustness note, not a defect to hide.
- **A "fix" exists but is out of scope and must not touch the frozen split.** A composite-key stratification — bucket *users* by (stay-rate band × modal category) before SGKF — would balance region/transition difficulty across folds and shrink fold-to-fold variance. But it changes the partition and would break frozen-§0.1 reproduction. Park it as a future-works robustness ablation (report both old and new), never as a silent swap.

Net: expected CV variance, driven by user-grouped held-out cohorts; tighten with the existing n=20 multi-seed plan, leave the frozen split untouched.

(Scratch scripts: `/tmp/claude-1010/-home-vitor-oliveira-PoiMtlNet/97304ca3-5ef6-49aa-871b-f5fb71b9fee9/scratchpad/repro_split.py` and `repro_split2.py`.)

---

## Investigation: stan-head-correctness

# STAN Reg-Head Faithfulness Audit

## Verdict (TL;DR)
Our reg head is a **STAN-*inspired* head over a learned substrate, not a realization of raw-input STAN** — and it is **internally correct**. Every major divergence from `research/baselines/stan/model.py` is a **documented, intentional deviation** dictated by the fact that ours consumes a pre-trained Check2HGI/HGI sequence (which already encodes per-check-in ST context) rather than raw POI tokens + Δt/Δd. I found **no genuine correctness BUGS** in the champion path (`mtlnet_crossattn_dualtower` → `next_stan_flow_dualtower` → `NextHeadSTAN`). The codebase's own audit doc (`docs/baselines/next_region/stan.md`) already states the in-house `next_stan` is "**not literature-faithful by themselves; the substrate-as-input version of STAN**." I surface 3 low-severity UNCLEAR/latent-fragility items below, none of which fire on the actual data pipeline.

---

## 1. Faithful STAN components (`research/baselines/stan/model.py`)

| Component | Where | Form |
|---|---|---|
| Multi-modal input emb | `model.py:275` (`+ model.py:233-235`) | `e_loc(poi) + e_time(hour_of_week)`; user-emb **dropped** (cold-user CV, documented `model.py:8-12`) |
| Pairwise ST bias | `_PairwiseBias model.py:91-119` | **scalar** `E_t[Δt] + E_d[Δd]`, learned 1-D interval tables, **linear interpolation** (`_interp_scalar model.py:67-88`); Δt minutes, Δd haversine-km |
| Trajectory attn (Layer 1) | `_SelfAttn model.py:122-151` | **single-head**, **bare** (no LN, no residual, no FFN); bias added to raw QKᵀ, **NO `1/√d`** (audit fix #3, `model.py:144-146`); key-pad masked + `nan_to_num` (`model.py:149`) |
| Matching layer (Layer 2) | `_MatchingLayer model.py:154-215` | `content = S·E_R` (`model.py:207`); **multiplicative** Δd gate `content * bias_match` (`model.py:210`); pad→0; **learned `Linear(seq_len,1)` collapse over ALL positions**, **NO softmax** (`model.py:184,214-215`); candidates = region emb table |
| Output | `model.py:302-304` | one logit per region (`[B, n_regions]`) |
| Loss | `train.py:144` | CrossEntropy (deviation from BPR, documented) |
| Transition prior | — | **NONE** |
| Sequence handling | `etl.py:54-60,196-227` | **prefix-expansion**, every position supervised, `CONTEXT_LEN=50` |
| No causal mask | (bidirectional) | all 9 inputs are past obs of a later target |

---

## 2. Our heads → faithful mapping

**`NextHeadSTAN`** (`src/models/next/next_stan/head.py`) — the shared core:
- **Input**: `input_proj: Linear(embed_dim→128) + LN + Dropout` on the **substrate** sequence (`head.py:200-202,260-263`). No separate `e_loc`/`e_time` — subsumed by Check2HGI (documented `head.py:18-32`).
- **Trajectory block** `_STANBlock` (`head.py:138-164`): **pre-norm self-attn + FFN with residuals** (LN/FFN/residual that faithful omits), **multi-head** (`num_heads=4/8`), **`1/√head_dim` scaling** (`head.py:111-112`).
- **Pairwise bias** `_STANAttention.pair_bias` (`head.py:82-95`): **fully-learnable `[heads,S,S]`** matrix, default **`bias_init="alibi"`** (recency-decay `-slope·|i−j|`), not Δt/Δd interval tables.
- **Matching** (`head.py:205-206,274-276`): a second `_STANAttention` with **`last_query_only=True`** → returns the **last valid step's** attention output (`head.py:128-134`), then `classifier` (`head.py:208-212`). The `Linear(d_model,num_classes)` rows play the role of STAN's region-candidate embeddings (content dot product), minus the Δd gate.
- **Masked softmax** (`head.py:115-120`): masks **keys** (`padding_mask[:,None,None,:]`), correct; fully-pad rows get last step un-masked to avoid all-`-inf` NaN (`head.py:257-258`) — faithful equivalent of `nan_to_num`.

**`NextHeadStanFlow`** (`next_stan_flow/head.py`): `NextHeadSTAN` + **`α·log_T[last_region_idx]`** prior (`head.py:121-147`); `last_region_idx` = region of the **observed** last POI (hard gather), read via the aux thread-local (`aux_side_channel.py`). Not in faithful STAN — explicitly "**not a faithful reproduction**" (`head.py:8-12`, GETNext-inspired).

**`NextHeadStanFlowDualTower`** (champion, `next_stan_flow_dualtower/head.py`):
- **private tower** = `NextHeadSTAN` on RAW `[B,9,64]` region seq (priv 4-head/0.3 dropout, `head.py:187-201`) — the STL reg pathway;
- **shared tower** = `NextHeadSTAN` on cross-attn `[B,9,256]` (8-head/0.1, `head.py:217-227`);
- sub-STAN classifiers → `Identity`, used **only via `forward_features`** (`head.py:201,227,464-473`);
- **`fusion_mode='aux'`** (champion): `feat = priv + β·aux_proj(shared)`, β init 0.1 (`head.py:237-239,411-412`);
- one fused classifier (`head.py:321-326`) → `_apply_prior` (`head.py:414-432`).
- **Plumbing** (`mtlnet_crossattn_dualtower/model.py:45-130`): `raw_region_seq=next_input` = the **post-pad-mask** raw region seq (`model.py:52-53,129`); param-partition stays in `reg_specific_parameters` (`model.py:19-24`). Correct.

---

## 3. Discrepancy classification

| # | Component | Faithful ref | Ours (file:line) | Class |
|---|---|---|---|---|
| D1 | Input emb | `e_loc+e_time` additive | `input_proj(substrate)` | **DEVIATION** — substrate carries ST context (`head.py:18-32`; audit doc "substrate-as-input") |
| D2 | Pairwise bias | Δt/Δd interval tables, scalar, interp | learnable `[heads,S,S]` ALiBi/gaussian | **DEVIATION** (interval/spatial info intentionally dropped, `head.py:55-63,18-32`) |
| D3 | Traj block | bare attn, no LN/FFN/resid, single-head, no `/√d` | pre-norm **+FFN+residual**, multi-head, **`/√head_dim`** | **DEVIATION** — note `/√d` aligns with **paper Eq.7**, the *reference repo* is the one that omits it (faithful audit fix #3). Ours is arguably *more* paper-correct here. |
| D4 | Matching/pooling | mult. Δd gate + `Linear(seq,1)` over **all** positions, **no softmax** | **last-position softmax-attn readout** + classifier | **DEVIATION** (documented `head.py:8-17`). See §4 skeptical note — this is *not* the v4 collapse bug. |
| D5 | `α·log_T` prior | none | added, on final logits (`flow:121-147`, `dualtower:414-432`) | **DEVIATION** (GETNext-inspired, documented `flow:8-27`); placement correct (final region logits); `last_region_idx` correctly last-valid (`next_region.py:137-163`) |
| D6 | Dual-tower | single tower | private raw ⊕ shared cross-attn, `aux` fusion | **DEVIATION/extension** (MTL-only; `T2.1_DUALTOWER_DESIGN.md`). Plumbing correct. |
| D7 | `freeze_alpha`+`alpha_init=0.0` | n/a | α→buffer, prior term ≡0 → pure STAN logits (`flow:101-104`, `dualtower:329-332`) | **DEVIATION** (champion G runs prior-OFF, NORTH_STAR L3); semantics **correct** |
| D8 | GRM read gate | none | `h=h_pre+γ⊙(blk−h_pre)`, off by default (`head.py:238-242,264-270`) | **DEVIATION** (R10 P4, default-off → champion bit-identical); math correct |
| D9 | Loss / seq construction | CE / prefix-expansion-50 | CE / substrate 9-window (stride-1 overlap on board) | **DEVIATION** (data-level, documented; faithful flags stride-9 starvation for the *raw baseline*, not this head) |
| U1 | last-step pooling index | learned collapse (alignment-agnostic) | `last_idx = num_valid−1` (`head.py:131-132`) | **UNCLEAR→OK**: correct **only** under left-packed (pad-at-end) convention — **confirmed** `core.py:262-263` (`history + [pad]*…`) and folds reshape `folds.py:365`. Latent fragility if a right-aligned/interior-pad seq is ever fed. |
| U2 | fully-pad row readout | n/a | softmax un-masks idx 8 (`head.py:257-258`) but `last_query_only` reads idx 0 (`head.py:131-132`) | **UNCLEAR, benign**: all-pad seqs are skipped (`core.py:285`) and zeroed pre-head (`next_gru/head.py:80-82`); degenerate, no real-data effect |
| U3 | `aux`/`last_region_idx` row-alignment | n/a | thread-local publish (`aux_side_channel.py:89-101`) | **UNCLEAR→OK**: requires `num_workers=0` same-thread (documented caveat `aux_side_channel.py:26-38`); MTL loop satisfies it |

---

## 4. Skeptical deep-dive: is the matching layer (D4) the v4 bug?

The faithful audit's **#1 MAJOR bug** was that v4's matching "**additive Δd bias + softmax over positions + reuse scores as values**" collapsed to a **proximity prior**, because raw embeddings init at std-0.02 `/√d` left the content channel **~25× below the distance bias → born dead** (`stan.md` audit table; `model.py:236-242` fix #2).

Our `next_stan` matching is **NOT** that bug, for three independent reasons:
1. **No distance bias to collapse onto** — the bias is a learnable relpos/ALiBi matrix on the **same scale** as content (`attn = (q@k)*scale + pair_bias`, `head.py:112-113`), not a dominant Δd term.
2. **Proper `attn @ v`** (V = projected values, `head.py:123`), not "reuse scores as values."
3. **Learned substrate input** (LN-normalized, `head.py:261`), not std-0.02 raw embeddings — no dead-content-channel regime.

So D4 is a legitimate **DEVIATION** (last-position softmax readout, documented `head.py:8-17`), not a latent collapse. The one *substantive* consequence worth stating in the paper: ours reads out **only the last position** (reinforced by the hard `α·log_T[last_region]` prior in the flow variants), whereas STAN aggregates **all** positions — i.e., ours leans harder on the most-recent check-in. This is a modeling choice, not an error.

---

## 5. Likely BUGS → fixes → eval

**None rise to BUG.** The honest output of this audit is that the champion path has no correctness defect vs the faithful reference; the divergences are intentional. The only items worth hardening (defensive, not behavior-changing on current data):

- **U1 (latent fragility, recommend a guard, not a fix):** `last_idx = (num_valid-1)` is silently wrong if a sequence is ever right-aligned or has interior pads. Today's pipeline is left-packed (`core.py:262-263`), so it is correct. **Proposed hardening:** replace the count-based index with the explicit last-True position used in `next_region.py:137-141` (`S-1 - valid[:,::-1].argmax`), making pooling alignment-robust. **Eval:** bit-identical on current data, so run **AL champion-G reg cell** (`scripts/mtl_improvement/c25_g_multistate.sh`, AL) and assert **byte-identical** Acc@10 (64.47±0.11) — a non-zero delta would itself reveal a hidden interior-pad case.
- **U2 (cosmetic):** for fully-pad rows the un-masked softmax index (8) and the readout index (0) disagree. **Fix:** none needed (rows skipped upstream); if desired, set readout to the same un-masked index. **Eval:** none — unreachable on real data.

If the intent is to also ship a **faithful-on-substrate** STAN (closer to D2/D4), that is a *new variant*, not a bug fix — it would feed Δt/Δd (absent from `next_region.parquet`, per `head.py:27-29`) and swap the readout for a multiplicative-gate `Linear(seq,1)` collapse; expected direction unknown (the audit itself notes coarse regions erode STAN's edge).

---

## Key references
- Faithful: `research/baselines/stan/model.py:91-215` (bias/attn/matching), `:262-304` (forward), `etl.py:54-60,196-227` (prefix-expansion), `train.py:140-144` (const-LR/CE), `README_FAITHFUL_STAN.md`, `docs/baselines/next_region/stan.md` (the self-audit that already declares the in-house head non-faithful).
- Ours: `src/models/next/next_stan/head.py:55-291`, `next_stan_flow/head.py:50-147`, `next_stan_flow_dualtower/head.py:104-498`, `src/models/mtl/mtlnet_crossattn_dualtower/model.py:45-157`, `src/data/inputs/next_region.py:129-172`, `src/data/aux_side_channel.py`, `src/data/inputs/core.py:262-263` (left-pack), `docs/NORTH_STAR.md:3` (champion G = dualtower/`aux`/prior-OFF).

---

## File audit: scripts/train.py

I have read the full file and verified the cross-references. Here is the audit report.

# Audit: `scripts/train.py` (2291 lines) — slimming + best-practices

Verified against `src/configs/canon.py`, `src/tasks/__init__.py`, `src/configs/experiment.py`, and the test/usage sites. Every canon bundle passes `--log-t-kd-weight` explicitly (`canon.py:46,52,58,76`), so the v12-default-ON KD block is a **live but narrow `--canon none` path**, not dead — preserved below.

## TOP items (ranked by value / risk)

| # | What | file:line | Category | Risk |
|---|------|-----------|----------|------|
| 1 | Merge `_run_category` + `_run_next` into one `_run_stl(..., task)` helper | `260-301` vs `304-345` | DUPLICATION | MED |
| 2 | Strip 84 dated/codename narration lines → docs; keep invariants | see §Comments | STALE COMMENTS | SAFE (mostly) |
| 3 | Extract KD-override block to `_apply_kd_overrides(config,args)` | `1653-1776` | OVER-LONG FN | MED (A/B) |
| 4 | Extract `_select_fold_subset(fold_results,args,max_folds)` | `2199-2229` | OVER-LONG FN | MED |
| 5 | Extract `_resolve_check2hgi_task_set(...)` (n_regions + resolve) | `2160-2197` | OVER-LONG FN | MED (A/B) |
| 6 | Delete unused `Dict` import + redundant local `LEGACY_CATEGORY_NEXT` re-import | `35`, `135` | DEAD CODE | SAFE |
| 7 | Factor the 19× `if config.task_type != "mtl": raise` + 3× per-head-LR guard | throughout `_apply_cli_overrides` | DUPLICATION | SAFE |
| 8 | Extract loss-calibration dict builder `_build_loss_calibration(args)` | `1394-1409` | OVER-LONG FN | SAFE |

---

## DEAD CODE / unused

- **`from typing import Dict, Optional` (`35`)** — `Dict` is never used (only `Optional`, at `1850`/`2087`). **SAFE:** change to `from typing import Optional`.
- **`from tasks.presets import LEGACY_CATEGORY_NEXT` (`135`, inside `_run_mtl`)** — redundant: `LEGACY_CATEGORY_NEXT` is already module-level imported from `tasks` (`50`), and `tasks/__init__.py:14-15` re-exports the *same object* from `tasks.presets`. **SAFE:** delete the local import; the module-level name is in scope.
- **`import os` is function-local in two spots** (`91` in `_make_run_dir`, `1934` `import os as _os` in `_preflight_canon_guards`). Not dead, but inconsistent. **SAFE (optional):** hoist a single module-level `import os`.
- **`--next-target` / `next_target` (`589-595`, applied `1376-1377`)** — dormant "reserved for future next_poi" marker (`experiment.py:41`); validated but has **no functional effect on training today**. Not safe to delete (it is plumbed + `__post_init__`-validated at `experiment.py:339`); flag as dormant, not dead.

## STALE / NARRATION COMMENTS (move to docs; PRESERVE invariants)

84 lines carry dated/codename narration (`grep` count). Distinguish two classes:

- **SAFE to move to docs (pure historical narration, no runtime contract):**
  - `132-134` `# AUDIT-C2: legacy preset both default to F1, so this is a no-op today…`
  - `206-211` `# AUDIT-C2 fix — wire each task's primary_metric…~3.5 pp on FL MTL runs`
  - `1342-1344`, `1389`, `1410`, `1599`, `1808` (`Tier C1 modality-bug fix, 2026-05-28`), `2097-2102` (the Designs-B/J/L paragraph), `2257-2258` (`cagrad + aligned_mtl added 2026-06-08`).
  - The big `1653-1667` / `1734-1735` / `1753` study-codename preambles (`v12 DEFAULT FLIP`, `R1/R3/R5 (mtl_frontier)`) — condense each to one line; the *mechanism* belongs in `CANONICAL_VERSIONS.md`/`mtl_frontier/FINDINGS.md` (already referenced).
- **PRESERVE (load-bearing invariants — do NOT move):**
  - Module docstring `10-23` (the "3 silently-wrong defaults" + `--folds` semantics contract).
  - `_preflight_canon_guards` body `1949-1971` (the `MTL_STRICT` env contract + torch-2.11.0 TopK tie-break invariant).
  - `1804-1813` modality-persist note (downstream `route_task_best.py` contract — invariant, though the trailing date can drop).
  - The `--per-fold-transition-dir` leak-guard help (`815-828`) and the `2055-2058` "post-seed torch globals" note.
  - `2254-2258` `_BACKWARD_ONLY_LOSSES` rationale (correctness guard).

Net SAFE comment reduction realistically ~40-50 lines.

## OVER-LONG FUNCTIONS → concrete extractions

`_apply_cli_overrides` is ~500 lines (`1315-1815`) of flat `if`-chains; `main` is ~315 lines (`1974-2287`). Behavior-preserving cuts (verbatim moves):

- **`_build_loss_calibration(args) -> dict`** ← `1396-1409` (the `_lc` assembly). Returns the `_lc` dict; caller does the single `dataclasses.replace(config, loss_calibration=_lc)`. Pure, no config dependence. **SAFE.**
- **`_apply_kd_overrides(config, args, logger) -> ExperimentConfig`** ← `1653-1776` (log_t_kd default-flip + log_c_kd + R3 arms + cat_kd). Returns the updated `config`. Self-contained but writes hot loss-path fields and embeds the v12-default-ON logic — **MED, needs a metric-parity A/B** on one `--canon none` check2hgi_next_region run (the only path that exercises the default-ON branch `1697-1719`).
- **`_select_fold_subset(fold_results, args, max_folds) -> dict`** ← `2199-2229` (the `--only-fold` / `--only-folds` / `max_folds` slicing with the 5-split guards). Returns the filtered dict. **MED** (controls which folds execute — A/B that fold keys are preserved).
- **`_resolve_check2hgi_task_set(task_set, fold_results, args, config) -> (task_set, config)`** ← `2160-2197` (n_regions union scan + `resolve_task_set` + model_params injection). **MED, A/B** — the `max_b` scan over `fr.next.train/val.y` is numerically load-bearing (sizes the reg head).
- **`_guard_mtl_only(config, flag_name)`** helper for the **19×** `if config.task_type != "mtl": raise ValueError(f"--{x} requires --task mtl")` (e.g. `1428,1465,1485,1502,1527,1542,1556,1566,1577,1602,1607,1617,1627,1642,1678,1724,1739,1767,1781,1791,1801`). And **`_require_per_head_lr(config, flag_name)`** for the 3× identical `all(getattr(config,k) ... ("cat_lr","reg_lr","shared_lr"))` guard (`1558,1568,1609`). **SAFE** (pure validation, no numeric effect).

## DUPLICATION

- **`_run_category` (`260-301`) ≈ `_run_next` (`304-345`)** — structurally identical; differ only in: runner import (`category_cv`/`next_cv`), `model_name` (`"Category"`/`"Next"`), `tasks` arg, `fold_results[i].category`/`.next`, the `DatasetHistory` (`get_category`/`get_next` + description), and `_make_run_dir` task label. Collapse to `_run_stl(config, results_path, fold_results, *, task: str)` parameterized by a small table. **MED** (constructs equivalent objects; assert byte-identical run dirs/monitors via a smoke run).
- **`_run_mtl` (`129-181`) vs `_run_mtl_check2hgi` (`184-257`)** — ~80% shared `MLHistory(...)` scaffold + checkpoint wiring; differ on `tasks`, the two `DatasetHistory` entries, and the monitor string (`val_f1_category` vs `val_joint_geom_lift`). A shared `_build_mtl_history(...)` is possible but lower value (the two diverge on dataset semantics). **MED — defer.**
- **reg/cat head-param parsing repeated 3×**: `_parse_key_value_overrides(args.reg_head_param ...)` at `2124-2129` (early) and `2176-2181` (late), plus `args.cat_head_param`. Parse once near the top of `main` and reuse. **SAFE.**
- **`canon_active` recomputed** independently in `_parse_args` (`387-389`) and `_preflight_canon_guards` (`1945-1947`) instead of reading the stamped `args._canon_active` (`1279`). Use the stamped flag in preflight. **SAFE.**
- **`min_epoch=int(getattr(config,"min_best_epoch",0) or 0)`** duplicated at `162` and `236`. Trivial.

## BEST-PRACTICE fixes

- **Magic monitor strings** (`val_f1_category` `170`, `val_joint_geom_lift` `245`, `val_f1` `292,336`) are implicit contracts with runner emit-keys (the docstring at `108-113` even warns a typo is silently dropped). Promote to named constants (or import from the runners) to make the coupling checkable. **SAFE, low value.**
- **`_DIM_KEYS` (`2005-2009`) defined inline in `main`** — hoist to module scope beside `_TASK_TYPES`/`_DEFAULT_FACTORIES` for symmetry. **SAFE.**
- **Inconsistent error exits**: `_apply_cli_overrides` raises `ValueError`→caught→`sys.exit(2)` (`1997-2001`), while `main`/`_resolve_folds` raise `SystemExit` directly (`1890,2024,2033,2207`). Pick one convention. **SAFE (cosmetic).**
- **Mutable-default safety**: `_parse_key_value_overrides(items, ...)` guards `items or []` (`1305`) — fine; the `action="append", default=[]` argparse defaults are not mutated. No bug, just confirm.
- **`_pre = argparse.ArgumentParser` pre-parse (`381-390`)** duplicates three flags (`--canon/--task/--config`) that the main parser re-declares — unavoidable with the prepend-bundle design, but add a one-line comment that the duplication is intentional (the main `--canon` decl at `396-406` is the source of truth for help/choices).

---

## Verdict

**Realistic slim: ~10-15% net (≈230-330 lines) without touching `help=` text** — driven by the runner dedup (#1), narration trimming (#2), and the guard-helper factoring (#7); up to **~25%** if the multi-paragraph `help=` epics (`574-588`, `923-945`, `1099-1114`) are condensed with their load-bearing invariants relocated to the already-referenced docs. **SAFE-to-apply-now:** #6 (dead imports), #7/#8 (guard + loss-calib extraction — pure validation/dict assembly), and the comment trims in the SAFE list. **Need a metric-parity A/B (one `--canon none` check2hgi_next_region + one default `--canon v16` run, compare per-task diagnostic_best epochs):** #1 (runner merge), #3 (KD block — exercises the v12-default-ON branch), #4 (fold-subset slicing), #5 (n_regions scan). The KD-default block (`1697-1719`) and the n_regions union scan (`2168-2175`) are the two genuinely numeric hot spots — extract verbatim only, never refactor their logic.

---

## File audit: src/training/runners/mtl_cv.py

# Audit: `src/training/runners/mtl_cv.py` (2010 lines) — Best-Practices + Slimming Plan

**Shape of the file:** 2 functions carry it all — `train_model` (L221–1332, ~1110 lines) and `train_with_cross_validation` (L1336–2010, ~675 lines) — plus 8 module helpers (L16–217). **487 of 2010 lines are comment-only (24%)**, the dominant slimming lever. The numeric hot path (per-batch loss/KD/step) is small; most bulk is narration + guard scaffolding + diagnostics that can move out of the two mega-functions without touching numbers.

---

## 1. DEAD CODE / unused params & imports

| # | Site | Category | Risk | Action |
|---|------|----------|------|--------|
| D1 | L4 `import numpy as np` | dead import | **SAFE** | `np` is never referenced (`grep "np\."` = 0 hits). Delete. |
| D2 | L249 `task_best_save_dir` param of `train_model` | dead param | **SAFE** | Accepted but **never read** inside `train_model` body (L263–1332 has zero refs; confirmed). It's threaded in at the call site L1925 but only ever used by the *caller* `train_with_cross_validation` (L1875–1960). Remove from `train_model` signature **and** drop the `task_best_save_dir=...` kwarg at L1925. |
| D3 | L40 `from torch.nn import CrossEntropyLoss` placed *after* `guard_finite_step` (L16–38) | import-after-code | **SAFE** | Move up to the import block (L1–11). Mid-file import is a lint/style smell only. |
| D4 | L347 `import os as _os`, L429 `import os as _os_ng`, L525 `import os as _os_s1`, L1610 `import os as _os` | redundant inline imports (4 distinct aliases for `os`) | **SAFE** | `os` is a stdlib top-level import — hoist one `import os` to L1 and delete all four inline aliases (rename `_os`/`_os_ng`/`_os_s1` → `os`). |
| D5 | L580 & L655 `import math as _math` | redundant inline import | **SAFE** | `math` already imported at L3. Delete both inline `import math as _math`, use the module-level `math`. |

> Note: the F50/R-program gated levers (`freeze_cat_after_epoch`, `reg_freeze_at_epoch`, `alpha_frozen_until_epoch`, `loss_scale_norm`, `log_c_kd_*`, `cat_kd_weight`) are **NOT dead** — they are live `mtl_frontier` study branches wired from CLI via `getattr(config,…)` at L1910–1935, default no-op. `alternating_optimizer_step` is in the champion-G B9 large-state recipe (CLAUDE.md). Do **not** delete these; only their narration comments are movable (§2).

---

## 2. STALE / NARRATION COMMENTS (move to docs; keep invariants)

The file embeds a lab notebook. Dated/codename narration that belongs in `docs/findings/` or `docs/CHANGELOG.md`, **not** the hot path:

| Site | What it is | Risk | Action |
|------|-----------|------|--------|
| L386–392 | "F50 D5 — encoder weight-trajectory diagnostic… hypothesis (F50 T3 §5.5)" | **SAFE** | Compress to one line: "Encoder drift diagnostic; see docs/findings F50-D5." |
| L575–578, L588–600, L675–685, L730–732 | T4.0a / Tier A1 / Phase 3 §4.5 / R1/R3 derivation prose for KD terms | **SAFE** | Keep the 1-line math contract (KL direction, τ² scaling); move the multi-paragraph derivation + doc-path citations to a docstring on the extracted helper (§3, KD1). |
| L1015–1022 | "2026-06-12 (HANDOFF_AUDIT X3 / CODE_AUDIT P1-C) — β trajectory logging" | **SAFE** | One-liner. |
| L1143–1151, L1164–1179 | 2026-04-15 review-agent + C21 monitor evolution prose | **SAFE** | Keep "default = geom_simple = sqrt(cat_f1·reg_top10)"; move the history to docs. |
| L1387–1392, L1445–1460, L1498–1514 | F51/C22/C29 leak-discovery blast-radius narration | **KEEP-TRIMMED** | These wrap **load-bearing leak guards** — keep the *invariant* sentence (why the guard exists + the rebuild command) but cut the multi-paragraph forensics + doc cross-refs. |

**PRESERVE verbatim (load-bearing invariants — do NOT move to docs):**
- L339–359 autocast env-var contract (`MTL_DISABLE_AMP`, `MTL_AUTOCAST_BF16`) — behavioral switch.
- L516–529 `MTL_STREAM_TRAIN_METRIC` byte-identical contract.
- L601–673 log_T-KD padding/leak-exclusion + the "requires a log_T-aware reg head" contract.
- L1427–1537 the C22 stale-mtime / n_splits / C29 engine-mismatch leak guards (the seed-tagged `region_transition_log_seed{S}_fold{N}.pt` invariant is the repo's reproducibility backbone).
- L16–38 `guard_finite_step` docstring (NaN-poison rationale) — **and the function name is imported by `tests/test_training/test_mtl_nonfinite_guard.py:27`, so the symbol must not be renamed/moved.**

---

## 3. OVER-LONG FUNCTIONS → pure-helper extractions

Both functions far exceed a maintainable size. Concrete extractions, ranked by value:

### Extraction A — per-fold prior resolution (HIGHEST VALUE, low numeric risk)
- **Site:** L1393–1598 (the entire `if per_fold_dir is not None:` block inside `train_with_cross_validation`), ~205 lines.
- **Category:** over-long function / IO+validation island in a loop body.
- **Risk:** **SAFE** (no numeric/tensor math — it's file existence, `stat().st_mtime`, `torch.load` payload inspection, and `dataclasses.replace`; all the leak `raise`s are preserved verbatim).
- **Action:** Extract `def _resolve_per_fold_priors(config, ts, i_fold, seed) -> dict` returning `per_fold_model_params` (the `dict(config.model_params)` with the swapped `task_set`). Cut L1393–1598 verbatim into it; the guard `raise`/`logger` calls move unchanged. Net: removes ~200 lines from the CV loop, makes the 5 leak guards independently testable.

### Extraction B — cross-task KD terms (HIGH VALUE, **RISKY** — hot numeric)
- **Site:** L588–760 (log_T-KD L601–673, log_C-KD L675–728, reverse cat-KD L730–760), ~170 lines inside the autocast.
- **Category:** over-long function / hot numeric block.
- **Risk:** **RISKY** — this directly forms `task_b_loss`/`task_a_loss`; any reordering changes fp results. Also has two side effects to preserve: `model._r5_gate_std` (L667) and the `globals()["_LOGC_FIRED"]`/`_CATKD_FIRED` one-shot log flags (L718/L752).
- **Action:** Extract `def _apply_cross_task_kd(model, pred_task_b, pred_task_a, truth_task_b, truth_task_a, task_b_loss, task_a_loss, *, log_t_kd_*, log_c_kd_*, cat_kd_*, epoch_idx, aux) -> (task_b_loss, task_a_loss)`. **Must be cut byte-identical and gated behind an A/B parity run** (the `--log-t-kd-weight 0.0` default fast-path is what §0.1 uses, so a v11-canon run is trivially unaffected, but any non-zero-KD frontier run needs a bit-compare). Replace `globals()` flags with a module-level `_KDFired` dataclass/closure flag — code-smell fix, but behavior-preserving.

### Extraction C — per-epoch diagnostic payload (HIGH VALUE, low risk)
- **Site:** L982–1086, ~105 lines building `diagnostic_payload`.
- **Category:** over-long function / diagnostics.
- **Risk:** **SAFE** — feeds `fold_history.log_diagnostic` only; **never** drives selection/early-stop/checkpointing (those key off val metrics, L1217–1276). The only subtlety: it carries loop state `next_enc_prev`/`cat_enc_prev` (L1073/L1084) — pass in and return them.
- **Action:** `def _collect_epoch_diagnostics(model, *, epoch_grad_*, enc_init/prev tensors, gate_stats, …) -> (payload: dict, next_enc_prev, cat_enc_prev)`. Cut verbatim.

### Extraction D — joint-selector arithmetic (MED value, selection-relevant)
- **Site:** L1132–1191, ~60 lines computing `joint_score`/`joint_acc1`/`joint_arith_lift`/`joint_geom_lift`/`joint_geom_simple`/`joint_selector_value`.
- **Category:** over-long function.
- **Risk:** **MED** — pure arithmetic but it *selects the checkpoint* (`joint_improved`, L1224). Cut verbatim it's deterministic; verify with a checkpoint-epoch A/B on one fold.
- **Action:** `def _compute_joint_selectors(val_metrics_task_a, val_metrics_task_b, f1_val_*, majority_fractions, checkpoint_selector) -> dict`. Returns all six scalars (they're re-used in the callback metrics dict L1295–1302).

### Extraction E — torch.compile setup (MED value, SAFE)
- **Site:** L1602–1631, ~30 lines (dynamo cache-limit raise + dynamic/mode env handling).
- **Risk:** **SAFE** — compile config only; default path is eager-equivalent.
- **Action:** `def _maybe_compile_model(model, config) -> model`. Note the **bare `except Exception: pass` at L1623** swallows all dynamo errors silently — at minimum log at debug level (best-practice fix, behavior-neutral).

### Extraction F — hoist nested closure
- **Site:** L393–399 `_flatten_encoder` (nested in `train_model`).
- **Risk:** **SAFE** — pure function of an `nn.Module`. Hoist to module level next to the other `_flatten_*` helpers (L71).

---

## 4. DUPLICATION (within file + with `mtl_eval.py`)

| # | Sites | Risk | Action |
|---|-------|------|--------|
| DUP1 | **L900–948** (`mtl_cv` S1 streaming TRAIN metric) ≈ **`mtl_eval.py:216–272`** (S2 chunked VAL metric) | **MED** | Near-identical accumulator pattern: per-batch `argmax`/`_rank_of_target`/`topk`-hit lists → reassemble via `_handrolled_cls_metrics`+`_mrr_from_rank`+`_ndcg_from_rank`. Extract a shared `StreamingClsMetric` accumulator into `tracking/metrics.py` (where those helpers already live). Both paths are documented byte-identical; an A/B is cheap (pure, no model state). High DRY payoff. |
| DUP2 | **L347–359** (`mtl_cv` autocast ctx) ≈ **`mtl_eval.py:171–179`** | **SAFE** | Both build `_autocast_ctx` from `MTL_DISABLE_AMP`/dtype. Extract `_build_autocast_ctx(device, *, disable, dtype)` into a shared util. Minor. |
| DUP3 | **L969–973** vs **L1281–1285** progress `set_postfix` blocks (train-only vs train+val) | **SAFE** | Fold into a tiny `_postfix(f1_b, f1_a, best_b, best_a, val=…)` helper. |
| DUP4 | **L453–465** (`freeze_cat_after_epoch` in loop) ≈ **L1642–1647** (`freeze_cat_stream` at fold start) ≈ **L482–489 / L1654–1659** (reg-side freeze) | **SAFE** | Same "freeze `category_encoder`+`category_poi`, `.eval()`" pattern repeated 4×. Extract `_freeze_stream(model, side: Literal["cat","reg"])`. |

---

## 5. BEST-PRACTICE fixes (typing / structure / smells)

- **B1 (SAFE):** `globals()["_LOGC_FIRED"]` / `_CATKD_FIRED` (L718–719, L752–753) — module-global mutation for one-shot logging. Replace with a module-level mutable flag object or `functools`-style guard; never `globals()`.
- **B2 (SAFE):** Bare `except Exception: pass` at **L1623** (dynamo cache cfg) and **L1986** (`_prof.record_quality`) silently swallow everything. Narrow to expected exceptions or log at debug.
- **B3 (SAFE):** Untyped public params on `train_model` — `optimizer`, `scheduler`, `next_criterion`, `category_criterion`, `mtl_criterion`, `num_epochs`, `num_classes`, `task_best_tracker` have no annotations (L222–262). Add types; the function already partially annotates, so this is consistency.
- **B4 (SAFE):** Magic thresholds repeated as literals: `> 256` (cardinality gate for handrolled path) at L528, L1099, L1103 (and `mtl_eval.py:149,254`); `1e-6`/`1e-8`/`1e-12` clamps scattered (L1158, L1162, L662, L703…). Promote `_HANDROLLED_CARD_THRESHOLD = 256` and the KD/lift epsilons to named module constants.
- **B5 (SAFE):** `_S1_TOPK = (3,5)` (L530) and `mtl_eval._S2_KS=(1,3,5,10)` (L159) are stringly-coupled to `compute_classification_metrics`'s default `top_k` — if that helper's default changes, the streaming path silently diverges. Add an assertion or import the default from `tracking.metrics`.
- **B6 (SAFE):** `train_model` has **44 parameters**. Beyond the dead `task_best_save_dir` (D2), the KD knobs (L250–258, 9 params) and freeze knobs (L241–247) are natural `@dataclass` config bundles (`KDConfig`, `FreezeSchedule`) — collapses the signature and the L1926–1935 call site. **MED** (it's a public-ish signature; `scripts/train.py` calls it indirectly via `train_with_cross_validation`, so internal — but verify no other caller).

---

## TOP items by value/risk

1. **§2 narration trim** (487 comment-only lines, 24%) — SAFE, highest raw line savings, zero numeric risk. Keep the env-var/leak invariants (the explicit KEEP list above).
2. **Extraction A** (per-fold prior, ~205 lines) — SAFE, biggest single-block removal, makes 5 leak guards testable.
3. **D2 dead `task_best_save_dir` + D1 `numpy` + D3–D5 inline imports** — SAFE, free wins.
4. **Extraction C** (diagnostics, ~105 lines) — SAFE, non-load-bearing.
5. **DUP1** (S1/S2 streaming metric) — MED, real DRY across `mtl_cv`↔`mtl_eval`, cheap A/B.
6. **Extraction B** (KD terms, ~170 lines) — RISKY, gate behind metric-parity A/B (default `--log-t-kd-weight 0.0` §0.1 path is auto-safe; non-zero frontier runs need bit-compare).
7. **Extraction D** (joint selectors) — MED, selection-relevant; verify checkpoint-epoch parity.

## Verdict

**Realistic slim: ~25–35%** (2010 → ~1300–1500 lines) if narration is trimmed to invariants **and** Extractions A/C/E + the per-fold-prior and KD blocks move to siblings/helpers; **~15% (→~1700) is the SAFE-now floor** (D1–D5 + narration trim + Extractions A, C, E, F + DUP2/3/4) with **no possibility of numeric change**. **Need a metric-parity A/B before applying:** Extraction B (KD, RISKY), Extraction D (joint selectors, MED), and DUP1 (streaming-metric consolidation) — all three sit on or feed the scored/selection path even though each is documented byte-identical. `guard_finite_step` must keep its name (test-imported).

---

## File audit: src/training/runners/mtl_eval.py

I have everything needed. Compiling the audit.

---

# Audit: `src/training/runners/mtl_eval.py` (320 lines)

**Scope confirmed:** one public fn `evaluate_model` (L71–320, ~250 lines) + one helper `_ood_restricted_topk` (L19–67). Sole internal caller: `mtl_cv.py:1108`. `_ood_restricted_topk` is *also imported by 4 external baseline scripts* (`scripts/baselines/{poi2vec_e2e,b3_hmt_grn,flashback_e2e,ctle_e2e}.py`) — so its name/signature is a **public contract; do not rename/move it.** All 4 module imports are live (`contextlib` L178, `logging` L152, `os`, `torch`); no dead imports.

## TOP items (ranked by value / risk)

### 1. Stale/narration comments → docs. Category: STALE-COMMENT. Risk: **SAFE**. Highest value.
~40–50 comment lines are dated audit-codename narration whose findings *already live in docs*. Trim to the load-bearing contract only.
- **L163–170** `2026-06-12 (HANDOFF_AUDIT X4 / CODE_AUDIT P1-D)` + the `Δreg −0.09…−0.31pp` story → the finding is in `docs/results/mtl_improvement/X_SERIES_FINDINGS.md` and `CHANGELOG.md:56`. **PRESERVE only:** "MTL_DISABLE_AMP_EVAL=1 (or MTL_DISABLE_AMP=1) forces fp32 eval; default keeps fp16."
- **L181–189** `HANDOFF_AUDIT X1 / CODE_AUDIT P0-A — cross-attn pairing roll probe` → finding recorded in `X_SERIES_FINDINGS.md:29` + `log.md:2324` ("X1 roll probe NULL"). **PRESERVE only:** "MTL_ROLL_TASKB_EVAL=1 rolls task-b by 1 at eval (diagnostic; read cat-F1 only). Default off."
- **L119–126** "S2 (perf-audit)" + **L131–138** "FOOTGUN GUARD (2026-06-20)" with the "~20 GB for CA (586k × 8501 × 4 B)" anecdote → mechanism is in `docs/studies/pre_freeze_gates/OOM_MEMORY_FIX.md`. **PRESERVE the invariants:** "chunked path is byte-identical (streamed per-row reductions, on-GPU fp16 tie-break), reg-only (C>256), behind MTL_CHUNK_VAL_METRIC / auto-enabled above MTL_S2_AUTO_BUDGET_GB."
- **L31–34** (in `_ood_restricted_topk`) the "~31K syncs × 50 epochs" perf anecdote → compress to one line; keep "vectorised `torch.isin` membership."

These env-var names (`MTL_DISABLE_AMP_EVAL`, `MTL_ROLL_TASKB_EVAL`, `MTL_CHUNK_VAL_METRIC`, `MTL_S2_AUTO_BUDGET_GB`) are the load-bearing contracts — keep every one; only the prose/dates/Δ-numbers go.

### 2. Duplicated streamed-metric reconstruction with `mtl_cv.py`. Category: DUPLICATION. Risk: **MED (A/B required)**.
The chunked reg-metric assembly at **L261–272** is near-byte-identical to the S1 train-metric assembly at **`mtl_cv.py:935–948`** (same `_handrolled_cls_metrics` → dict → `_mrr_from_rank`/`_ndcg_from_rank`/`top{k}_acc` over `(3,5)`; only diff is S1 `.cpu()` vs S2 on-GPU tensors).
- **Action:** extract a shared `def _streamed_cls_metrics(preds, tgts, rank, hit, num_classes, top_k=(3,5)) -> dict` into `src/tracking/metrics.py` (next to `_handrolled_cls_metrics`). Returns the metrics_next dict. Both S1 and S2 call it. Removes ~12 dup lines here + ~10 in `mtl_cv.py`.
- **Risk MED:** it is a SCORED hot-numeric block on the frozen path. The cut is byte-identical *only if* device-placement of inputs is preserved (S1 passes CPU tensors, S2 GPU tensors — reductions are per-row/additive so identical, but `.item()` on cuda-vs-cpu must be A/B-verified). Gate behind a metric-parity A/B (one FL + one CA seed, assert dict equality vs full-logit path).

### 3. Duplicated OOD-from-accumulators block. Category: DUPLICATION. Risk: **MED (A/B required)**.
**L292–312** (chunked OOD) reimplements `_ood_restricted_topk` (L19–67) from streamed `_HIT`/`_R` instead of logits.
- **Action:** extract `_ood_from_streamed(tgts, rank, hit, train_labels, ks=(1,5,10)) -> dict` beside `_ood_restricted_topk`; have the full path keep calling `_ood_restricted_topk`. Returns the `ood_b` dict. Removes ~20 lines.
- **Risk MED:** the comment at L294–296 flags a real numeric subtlety ("`.float().mean()` NOT a running int count — drifts at the last ULP"). Preserve that invariant verbatim in the helper; A/B against current output before landing.

### 4. Extract the S2 chunk-decision into a pure helper. Category: OVER-LONG-FN. Risk: **SAFE**.
**L127–158** (gate resolution: `_nc_b_gate`, `_full_logit_gb`, `_S2_BUDGET_GB`, `_auto_chunk`, `_chunk_val`, the auto-WARN) is a self-contained decision.
- **Action:** `def _decide_chunk_val(dataloaders, nc_b_gate) -> bool` returning `_chunk_val` (does the WARN internally via module `logger`). Leaves only the `sv_*` accumulator init (L159–161) inline. Shrinks the function body by ~30 lines into a 1-line call.
- **SAFE:** it selects between two paths the repo asserts are byte-identical; extraction does not change *which* path is chosen, so numeric output is untouched. Apply now.

### 5. Duplicated autocast-hatch with `mtl_validation.py`. Category: DUPLICATION. Risk: **MED**.
**L171–179** is verbatim-equal to **`mtl_validation.py:24–32`** (same `MTL_DISABLE_AMP_EVAL`/`MTL_DISABLE_AMP` read + `torch.autocast(...float16)` else `nullcontext`).
- **Action:** `def eval_autocast_ctx(device) -> ContextManager` in a shared eval util (e.g. `src/training/shared_evaluate.py` or `src/training/helpers.py`); both sites import it. Centralises the fp32-eval env contract.
- **Risk MED:** behavior-affecting context; mechanically identical but verify the `device.type`/`DEVICE.type` distinction (mtl_eval uses the passed `device`, mtl_validation uses global `DEVICE`) — pass the device in explicitly to avoid a silent change.

## Secondary best-practice fixes
- **DEAD PARAM `mtl_criterion` (L75):** unused in the body (only L75 signature + L99 docstring "kept for API compat"). Only one caller passes it positionally (`mtl_cv.py:1113`). Category: DEAD-CODE. Risk **MED** (signature churn for 1 call site) — low value; optional.
- **Missing return type + incomplete docstring (L71–110):** no `-> tuple[dict, dict, float]` annotation; the Args block stops at `device` and omits `num_classes`, `task_a/b_num_classes`, `train_labels_a/b`. Category: BEST-PRACTICE/typing. Risk **SAFE**.
- **Cryptic single-letter locals in the scored chunked path:** `_P/_T/_R/_HIT` (L261–264), `_am/_aM/_fm/_fw` (L265), `_ni/_noo` (L299–300). Rename to `preds/tgts/rank/hit/...`. Category: NAMING. Risk **SAFE** (pure local renames, byte-identical) — but do it in the same commit as the E2/E3 extraction to avoid touching the frozen block twice.
- **Magic number `256` (L149)** duplicated across `mtl_eval.py:149`, `mtl_cv.py:528,1099,1103`; it is the named `_CARDINALITY_HAND_ROLLED_THRESHOLD` but only *function-local* in `metrics.py:247`. Promote to a module-level `CARDINALITY_HAND_ROLLED_THRESHOLD` in `metrics.py` and import at all 4 sites. Category: MAGIC-NUMBER/DUP. Risk **SAFE**.
- **Broad `except Exception: pass` (L143–144)** swallowing the `_full_logit_gb` probe — narrow to `(TypeError, AttributeError, IndexError)`. The downstream `locals().get("_n_val", -1)` (L157) is then dead defensiveness (`_n_val` is always bound when `_auto_chunk` is True) → simplify to `_n_val`. Category: ERROR-HANDLING. Risk **SAFE**.
- **Minor dup:** `_nc_b_gate` resolution (L127–129) repeats the `nc_b` resolution (L254) except for the extra `getattr(model,"num_classes",0)` fallback — unify into one local computed once. Risk **SAFE**, low value.

## Verdict
Realistic slim: **~25–30%** (320 → ~225–240 lines). **Apply-now (SAFE, no A/B):** item 1 (comment trim, biggest single win), item 4 (chunk-decision helper), local renames, return-type/docstring, shared `256` constant, narrowed except. **Needs a metric-parity A/B before landing (frozen-§0.1 scored paths):** items 2 and 3 (streamed-metric + OOD dedup into `metrics.py`) and item 5 (shared autocast ctx) — run one FL + one CA/TX seed and assert the returned `metrics_next`/`metrics_category`/`ood_*` dicts are bit-identical to current output. Do not touch the `MTL_*` env-var reads, the on-GPU fp16 tie-break note, or the `.float().mean()`-not-int-count OOD invariant (L294–296).

---

## File audit: src/data/folds.py

Audit complete. I read all 1558 lines of `src/data/folds.py` and verified every liveness claim with repo-wide greps (callers in `scripts/train.py`, `src/training/runners/mtl_cv.py`, `scripts/study/freeze_folds.py`, `scripts/eval_hard_vs_soft_region_idx.py`, sibling `src/data/dataset.py`).

# Audit: `src/data/folds.py` (1558 lines) — slimming + best-practices

## Liveness facts established (anchor every recommendation)
- `_create_mtl_folds` (TaskType.MTL legacy user-isolation) **and** `_create_check2hgi_mtl_folds` (TaskType.MTL_CHECK2HGI) are **both live**. Champion-G runs the check2hgi path: `train.py:2143` sets `fold_resolve_key="mtl_check2hgi"` when `--task mtl --task-set check2hgi_next_region`; bare `--task mtl` (LEGACY preset) hits `_create_mtl_folds`.
- `rebuild_dataloaders`/`load_folds`/`save_folds` are the **hot frozen-fold path** for paper reproduction (`train.py:1892-1912`). RISKY to touch.
- `aligned_pairing`/`AlignedJointLoader`/`_create_aligned_joint_loader`/`joint_train_loader` are **live but opt-in** (`train.py:961,1789,1880` → `mtl_cv.py:316-319,1938`). Default off; do not delete.
- `_get_num_workers()` (folds.py:267-284) **always returns 0** — every `num_workers > 0` branch is statically dead.
- `save_split_manifests` (1507-1552) has **zero callers anywhere** in the repo (`grep -rn save_split_manifests` → only the def). Its private feeder `self._fold_manifests` and serializer `_json_default` (371-379) are reachable **only** through it.
- FoldData.x=None on the check2hgi path is safe: the one script that comments about `.next.val.x` (`eval_hard_vs_soft_region_idx.py:192`) actually reads `creator._fold_indices` instead, never `.x`.

---

## TOP items (ranked by value / risk)

### 1. Dead `save_split_manifests` + `_json_default` + manifest plumbing — MED
`save_split_manifests` (1507-1552) is uncalled; it transitively kills `_json_default` (371-379) and makes the per-fold manifest building dead: `_create_mtl_folds` lines **1142-1164** (`self._fold_manifests.append({...})`) and `_create_check2hgi_mtl_folds` lines **1407-1413**. **Also a latent bug:** lines 1546-1547 read `getattr(self, 'state', None)` / `getattr(self, '_engine_value', None)` which are **never assigned** anywhere (`grep` confirms) → the digest JSON always emits `state:null, engine:null`. **Action:** confirm no notebook/external caller, then delete the method + `_json_default` + both manifest-build blocks (~75 lines). If kept as public API, fix the null-attr bug by assigning `self.state`/`self._engine_value` in `create_folds`. Category: DEAD CODE.

### 2. Dead DataLoader-worker branches + `_worker_init_fn` — SAFE (behaviorally) / MED (removes escape hatch)
Because `_get_num_workers()≡0`, these ternaries are constant: `persistent_workers=num_workers>0` (always False), `prefetch_factor=2 if num_workers>0 else None` (always None), `worker_init_fn=... else None` (always None) — repeated at **456-458, 500-502, 570-572**; `_worker_init_fn` (334-336) is only referenced by the always-false ternary. **Action:** drop the three worker kwargs to their constant values (or remove) and delete `_worker_init_fn` (~12 lines). Byte-identical. **Preserve the load-bearing `_get_num_workers` comment 268-283** (it documents the measured workers≠quality-neutral finding — that is an invariant, not narration). Category: DEAD CODE.

### 3. Triplicated DataLoader-kwargs block — SAFE
`_create_dataloader` (426-459), `_create_aux_dataloader` (462-504), `_create_aligned_joint_loader` (540-574) repeat the identical kwargs (`num_workers, pin_memory=torch.cuda.is_available() and dataset_device is None, persistent_workers, prefetch_factor, worker_init_fn`). **Action:** extract `_loader_kwargs(num_workers, dataset_device) -> dict` and splat it; the aux builder additionally wraps in `AuxPublishingLoader`, the joint builder adds `generator=g` and omits `collate_fn` — all expressible as overrides. ~20 lines saved, byte-identical (same literal kwargs). Category: DUPLICATION.

### 4. `_create_check2hgi_mtl_folds` is ~298 lines (1196-1493) — extract two pure helpers — SAFE
- **Extract region-label loading + alignment guard + aux gate**, lines **1228-1302**, into `_load_region_labels(state, engine, X, userids, task_set) -> (y_region, y_last_region|None, use_aux)`. Pure numpy/validation, no model math; returns the same arrays → byte-identical if cut verbatim. This also relocates the C1 alignment guard (1235-1249) and the aux-gate set `_HEADS_REQUIRING_AUX_MTL` (1284-1288) — **keep the env/leak invariants, move only the dated narration** (see §6).
- **Promote nested `_resolve_x`** (1334-1359) to module-level `_resolve_task_input(input_type, state, embedding_engine, x_checkin)`. Pure; SAFE.
Net: the method body drops to the split loop + `_build_fold`, ~120 lines shorter and far more readable. Category: OVER-LONG FUNCTION.

### 5. `_create_mtl_folds` POI-classification block — extract, but A/B-gate — RISKY
Lines **1079-1121** (the train/val/ambiguous POI partition + the str-coercion `np.isin` at 1116-1121) **determine exact frozen fold membership** for legacy-MTL. Extracting to `_classify_pois(poi_users, train_users, val_users, cat_placeids) -> (train_cat_idx, val_cat_idx, train_excl, val_excl, ambiguous)` is a clean ~43-line cut, but because it is the frozen split logic it must be A/B-verified (compare `fold_set_digest` / `_fold_indices` before vs after on one state). Verbatim cut should be identical; flag RISKY only because of the §0.1-freeze. **Keep the str-coercion comment 1105-1118** (load-bearing: explains the empty-fold/`num_samples=0` crash). Category: OVER-LONG FUNCTION.

### 6. Stale dated/codename narration that belongs in docs — SAFE (comments only)
These are pure post-mortem narration (move to `docs/`, leave a one-line pointer):
- folds.py **1276-1283** — the `2026-06-12 (HANDOFF_AUDIT X2 / CODE_AUDIT P0-B)` dual-tower paragraph. Keep only "`next_stan_flow_dualtower` must be in this set so the aux side-channel reaches the head."
- **661-671** — the `_guard_mtl_check2hgi_ram` coefficient-calibration story (CA "49 GB RSS → coefficient 3.9; use 4.0"). Keep the `4.0` and one line; the derivation is doc material.
- **125-139** — `_LazyFoldMapping` CA-stride "113 GB OOM" anecdote; trim to the invariant ("built on demand, not cached; byte-identical split/loaders/seeds").
- **201-209 / 248-253** — `__getitems__` "perf fix 2026-06-24, ca-mtl speed workflow" → keep the one-line "batched index_select; rows/order from sampler ⇒ byte-identical", drop the codename.
- **1212-1218** — `SUBSTRATE_COMPARISON_PLAN §5` / `substrate-protocol-cleanup Tier B (2026-05-28)` provenance prose.

**Do NOT touch (load-bearing invariants, keep verbatim):** the `_get_num_workers` workers≠quality finding (268-283); `_dataset_device` env contracts + "byte-identical" claim (296-326); the C1 userid-content alignment guard rationale (1235-1239); `_warn_if_ungated_overlap` `emit_tail`/`MTL_STRICT` contract (692-728); the str-coercion crash note (1105-1118); the `FoldData.x is None` rationale (96-101, 1460-1462); the log_T/aux-gate sync note (1262-1274). Category: STALE/NARRATION.

### 7. Redundant local imports — SAFE
`os` is module-level (line 23) yet re-imported as `import os as _os` inside `_dataset_device` (**310**) and inside `_resolve_x` (**1349**); `StratifiedKFold` is imported at top (line 36) yet re-imported locally in `_create_mtl_folds` (**1058**). **Action:** delete all three, use the module-level names. Byte-identical. Category: DEAD CODE / BEST-PRACTICE.

### 8. Duplicated RAM-guard boilerplate — SAFE
`_guard_cpu_resident_ram` (613-637) and `_guard_mtl_check2hgi_ram` (640-689) repeat: `import psutil`/`except: return`, `float(os.environ.get("MTL_RAM_HEADROOM_GB","16"))`, `psutil.virtual_memory().available/1024**3`. **Action:** extract `_avail_and_headroom_gb() -> (avail_gb, headroom_gb)`; keep the two distinct estimate formulas + messages. Also note `load_next_data:757-762` re-documents the guard the guard already docstrings — trim that comment to one line. Category: DUPLICATION.

### 9. `POIDataset` / `POIDatasetWithAux` device-move duplication — SAFE
The `t.to(device) if t.device != device else t` / `t if t.device.type=='cpu' else t.cpu()` ternary pair is written 2× in `POIDataset.__init__` (189-193), 3× in `POIDatasetWithAux.__init__` (229-235), and `__getitems__` is near-identical (208-209 vs 250-253). **Action:** add a module helper `_place(t, device)` and have `POIDatasetWithAux` subclass/reuse it. ~10 lines. Note the sibling `src/data/dataset.py:POIDataset` is a *different, minimal* class (numpy-in, no device/`__getitems__`) used only by `scripts/p1_poi_head_ablation.py` — the cross-file name clash is already documented there; not a merge target. Category: DUPLICATION.

### 10. Typing / minor best-practice — SAFE
- `_LazyFoldMapping.__init__(self, n_splits, build)` (142): `build` untyped → `Callable[[int], FoldResult]`.
- `_create_aligned_joint_loader(x_b, y_b, x_a, y_a, aux, ...)` (540): all tensor params untyped.
- `FoldCreator.__init__ task_set: Optional[object]` (879) — the `Optional[object]` is a deliberate import-decoupling choice (documented 894-899); leave but the comment justifies it.
- Magic default `"16"` for headroom appears 3× (325, 626, 659) → a module constant `_DEFAULT_HEADROOM_GB = 16`.
- `_create_dataloader` positional-bool call sites (e.g. 992, 996, 1169-1188) pass `True/False/False` un-named — readability only; cosmetic.
Category: BEST-PRACTICE.

---

## Verdict
Realistic behavior-preserving slim: **~10-15%** (~150-230 lines), almost all from narration-comment relocation (§6), dead worker branches (§2), the dead manifest path (§1), and the three SAFE de-duplications (§3, §8, §9).

**SAFE to apply now (no metric A/B):** §1 (after confirming no external caller), §2, §3, §6, §7, §8, §9, §10 — none alter tensor values, split indices, RNG draw order, or loader iteration order.

**Needs a metric-parity A/B (fold-digest compare):** §4's region-label extraction (verbatim, very likely identical but it's the frozen check2hgi data path) and especially §5's `_classify_pois` extraction (it computes the frozen legacy-MTL fold membership). Verify via `fold_set_digest` equality before/after on one state per affected path. Do **not** touch `rebuild_dataloaders`/`load_folds` (775-865) or any RNG/`generator=`/shuffle ordering — those are the §0.1-reproducibility surface.

---

## File audit: src/tracking/storage.py

Both code paths are test-covered (full `save()` → `test_ml_history.py:696-870`; partial → `test_fold_partial_persist.py`), which gives a clean bidirectional A/B gate for the dedup. I have everything needed.

---

# Audit: `src/tracking/storage.py` (834 lines) — slimming + best-practices

Scope: `HistoryStorage` (`_folder_name`, `save`, `save_fold_partial`, `_save_metrics`/`_save_reports` and their per-fold twins) + `SummaryGenerator`. Frozen-§0.1 repo: this file writes *artifacts read by downstream aggregation*, so JSON keys/values and CSV float formats are the load-bearing surface — anything that changes them is RISKY.

## TOP ITEMS (ranked by value/risk)

### 1. `_save_reports` ⇄ `_save_fold_report` are byte-identical bodies — DUPLICATION (~110 lines)
- **`_save_reports` 519-634** (116 lines) and **`_save_fold_report` 348-465** (118 lines).
- I diffed `360-465` vs `522-634`: the **only** differences are (a) one extra indentation level (`_save_reports` wraps the body in `for _pos, fold in enumerate(...)`) and (b) comment wording. The logic — joint-epoch extraction, `fold_info` dict, per-task report JSON/CSV, `diagnostic_best_epochs`, the `CANONICAL_BEST_METRICS` `per_metric_best` block, `primary_checkpoint.task_metrics`, the `best_epochs` legacy alias — is identical.
- **Category:** DUPLICATION. **Risk: SAFE** (extraction is byte-identical-if-cut-verbatim, and both ends are covered: `test_ml_history.py:794-870` pins `fold_info.json` schema/values for the full path, `test_fold_partial_persist.py:54-163` pins it for the partial path).
- **Action:** extract `_write_fold_report(self, path: Path, fold: FoldHistory, i: int) -> None` containing lines **522-634 dedented one level** (returns nothing; writes the 3 files). Then:
  - `_save_reports` → `for _pos, fold in enumerate(self.history.folds): self._write_fold_report(path, fold, self.history.fold_label(_pos))`
  - `_save_fold_report` → `self._write_fold_report(path, self.history.folds[fold_idx], self.history.fold_label(fold_idx))`
  - Delete the now-false rationale docstring at **348-356** ("Keeping the logic in one place would be cleaner; for now the duplication is acceptable…") — the refactor *is* that single place.
- **Caveat:** the body assembles the `per_metric_best` numeric sub-dict, so although the cut is verbatim, gate on running both test files (they assert `pytest.approx` values) as the metric-parity A/B before considering it done.

### 2. `CANONICAL_BEST_METRICS` defined twice, inline — DUPLICATION / magic constant
- Identical 11-tuple at **418-423** and **583-588**.
- **Category:** DUPLICATION + best-practice (a frozen metric whitelist hidden inside a function body). **Risk: SAFE.**
- **Action:** hoist to a module-level constant `CANONICAL_BEST_METRICS = (...)` near the top (after imports). Both sites reference it. Subsumed automatically by item 1, but worth doing even if item 1 is deferred. Keep the tuple contents exactly (order is irrelevant — it's iterated, not indexed).

### 3. `_save_metrics` ⇄ `_save_fold_metrics` — DUPLICATION (~12 lines)
- **`_save_fold_metrics` 335-346** and the inner body of **`_save_metrics` 500-517** are the same train/val-CSV-per-task logic (one operates on `folds[fold_idx]`, the other loops). Confirmed: both have the two `save_csv` calls with identical filenames/format.
- **Category:** DUPLICATION. **Risk: SAFE** (CSV writes, no numeric transform; covered by `test_ml_history.py:719-733` + `test_fold_partial_persist.py:145-163`).
- **Action:** extract `_write_fold_metrics(self, path: Path, fold, i: int) -> None`; loop it in `_save_metrics`, call it directly in `_save_fold_metrics`.

### 4. Stale / narration comments that belong in docs
- **322-325** (`save_fold_partial`): narrates a *rejected* design ("The simplest way without touching those methods is to operate on a single-element proxy. Instead, inline the fold-level subset here for clarity."). Pure dead narration. **Risk: SAFE — delete.**
- **411-417** and **577-582**: dated `F50 T3 fix (2026-04-29)` narration with research-doc references (`research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §5.5`, `MTL_FLAWS_AND_FIXES.md §2.10`). The *behavior* (per-metric best epoch) is load-bearing; the dated story is not. **Risk: SAFE** — trim each to one line, e.g. `# per-metric best epoch (F1-best ≠ top10-best); additive per_metric_best sub-dict`. Item 1 collapses these to a single site anyway.
- **87-92** (C7 `aggregation_basis` block): this one is **partly load-bearing** — it documents the three-basis contract (`joint_best` / `per_task_f1_best` / `per_metric_best`) that downstream readers depend on. **PRESERVE the basis-name list**; the "C7 fix" codename can drop. **Risk: SAFE** for trimming the codename only.
- **628-631** (`best_epochs` legacy key): load-bearing back-compat note (a downstream consumer reads `fold_info['best_epochs']`; `test_ml_history.py:797` asserts its absence under joint selection). **PRESERVE** — keep as a one-liner.
- **560-563** ("Export every metric… no hardcoded list"): explanatory, low value, harmless — optionally trim.

### 5. `map_category` is duplicated across files with **divergent** fallback strings
- `storage.py:62-70` (`map_category`, fallback `f"Unknown-{key}"`, `except Exception`) vs `display.py:111-119` (`_map_category`, fallback `f"Class {key}"`, `except (ValueError, TypeError)`).
- **Category:** DUPLICATION-with-divergence. **Risk: MED/RISKY** — `storage.map_category` output becomes **JSON dict keys** in `fold{i}_{task}_report.json` (line 552/390) and category-summary CSV rows; changing the `Unknown-{key}` fallback or broadening/narrowing the `except` would alter on-disk artifacts. **Do NOT blindly merge.** If consolidating, extract a shared `map_category(key, label_map, *, fallback="Unknown-{key}")` and pass each call site its current fallback verbatim — only then is it behavior-preserving. Recommend leaving as-is unless a shared util is wanted.

## OTHER FINDINGS

### Over-long functions → SAFE pure extractions
- **`SummaryGenerator.generate` 80-149** (~70 lines) does three things. Two clean extractions, both byte-identical-if-verbatim:
  - `_assemble_stats(perf, diagnostic_perf, per_metric_perf) -> dict` = lines **93-124** (pure dict construction over `self._stats`). **Risk: SAFE.**
  - `_write_category_summaries(out, cat_metrics) -> None` = lines **127-149** (the two `pd.DataFrame` builds + `save_csv`). **Risk: SAFE** — but note it carries the `±` formatting (`f"{mean*100:.2f} ± {stdev*100:.2f}"`, lines 138-139) and `float_format='%.4f'`; cut verbatim, no change.
- **`_save_plots` 636-695** (~60 lines): two near-identical plot loops (per-task 651-671, model-level 676-695). Extract `_plot_metric(folds_iter, metric, ...)`. **Risk: SAFE for §0.1** (plots are not numeric outputs), low value.
- **`_save_diagnostics` 697-751** (~55 lines): mixes epoch-series CSV/plots + confusion-matrix + attention + generic-artifact fallback. Could split `_save_diagnostic_series` / `_save_artifacts`. **Risk: SAFE for §0.1**, low value.

### Best-practice / correctness (not pure slimming)
- **Lambda variable shadowing:** `429` and `594` use `key=lambda i: values[i]` where `i` is already the fold label in scope. Lambda-local so **no behavior bug**, but a readability foot-gun. **Risk: SAFE** — rewrite as `max(range(len(values)), key=values.__getitem__)` (also drops the lambda).
- **`fold_label` inconsistency (latent fan-out bug):** `_save_metrics` (502-503) and `_save_reports` (520-521) name files via `self.history.fold_label(_pos)`, but **`_save_plots` (651,676)** and **`_save_diagnostics` (699)** use raw `enumerate(..., start=1)` positional `i`. For a full run `fold_label(pos)==pos+1`, so §0.1 is unaffected; under a `fold_ids` fan-out subset the plot/diagnostic filenames would use position, not the real fold id. **Risk: MED to "fix"** (changes fan-out filenames; zero change for full runs). Flag as a known inconsistency, not a slimming action.
- **Import ordering / `logger` placement:** `logger` is created at **line 7**, *between* the stdlib imports (1-5) and the third-party imports (`numpy`/`pandas`/`matplotlib`, 9-14). Best-practice: move all imports to the top, define `logger` after. **Risk: SAFE**, cosmetic.
- **`_stats` 151-160** returns int `0` for empty (`{'mean': 0, ...}`) but floats otherwise — harmless JSON inconsistency. Leave (changing risks artifact diff on empty-series edge cases).
- **`_save_params` 497** uses bare `open(..., 'w')` while the rest of the file uses the `save_text`/`save_json` helpers. **Risk: SAFE** — swap to `save_text(self.history.model_arch, path / 'arch.txt')`. Tiny.
- **No dead imports/params found:** all of `Any/Dict/List/Optional/Union/TYPE_CHECKING`, `mean/stdev`, `numpy/pandas/matplotlib`, and `_SafeEncoder`/`save_json`/`save_csv`/`save_text`/`ensure_dir`/`map_category` are used in-file (the module-level helpers are **not** imported anywhere else — grep across `src/scripts/pipelines/research/experiments/tests` returns nothing — so they are effectively private to this file and could even lose the implicit "public util" status, but keep them: cheap and self-documenting).

## VERDICT
Realistic slimming **~18-22% (≈150-180 of 834 lines)**, almost entirely from the `_save_reports`/`_save_fold_report` dedup (item 1, ~110 lines) plus the `CANONICAL_BEST_METRICS` hoist (item 2), `_save_metrics` dedup (item 3, ~12 lines), and narration-comment trims (item 4, ~30 lines).

- **SAFE to apply now (no metric A/B needed):** items 2, 4 (delete 322-325; trim dated F50/C7 codenames, preserve the basis-name list + `best_epochs` note), the lambda-shadow rewrite (429/594), import/`logger` reordering, `_save_params` `open`→`save_text`, and the `SummaryGenerator.generate` extractions (`_assemble_stats`, `_write_category_summaries`).
- **SAFE but gate on the metric-parity A/B** (run `tests/test_tracking/test_ml_history.py` + `tests/test_tracking/test_fold_partial_persist.py`, which pin `fold_info.json`/`full_summary.json` values): items **1** and **3** — byte-identical extraction touching the `per_metric_best`/`primary_checkpoint` numeric assembly.
- **Leave / MED-RISKY:** `map_category` consolidation with `display.py` (item 5 — divergent fallback strings land in JSON keys) and the `fold_label` fan-out inconsistency (changes only fan-out filenames). Plot/diagnostic extractions are SAFE for §0.1 but low value.

Citations: target file `/home/vitor.oliveira/PoiMtlNet/src/tracking/storage.py`; data model in `/home/vitor.oliveira/PoiMtlNet/src/tracking/{fold.py,metric_store.py,experiment.py}`; duplication sibling `/home/vitor.oliveira/PoiMtlNet/src/tracking/display.py:111-119`; coverage `/home/vitor.oliveira/PoiMtlNet/tests/test_tracking/{test_ml_history.py:696-870,test_fold_partial_persist.py}`.

---

## File audit: src/tracking/experiment.py

## Audit: `src/tracking/experiment.py` (MLHistory) — 308 lines

Verified the full file plus every cross-module reader (`storage.py`, `display.py`, `records.py`, `fold.py`, `metric_store.py`, `time_history.py`, the three CV runners, `scripts/train.py`, `adapters.py`). All call-sites and attribute readers were grepped; findings below cite the reader site where relevant.

### Context that bounds risk
- The adapter feature (`self._adapter`, default `None`; set only by `adapters.py:15`) is a W&B/MLflow side-channel. **It is never active in §0.1 reproduction runs**, so any change confined to `_adapter is not None` blocks cannot move a numeric result — those are SAFE for repro but may change external-tracking output (flagged MED where output-visible).
- `storage.save()` / `save_fold_partial()` / `compare_records` calls are the on-disk artefact path and must stay byte-for-byte; no finding touches their arguments.

---

### TOP items (ranked by value / risk)

**1. Dead write-only state: `self.monitor`, `self.mode`** — `experiment.py:83-84` — DEAD CODE — **SAFE (low value)**
Confirmed zero readers repo-wide: `history.monitor` / `history.mode` / `.h.monitor` / `.h.mode` all return nothing (grep empty). The live monitor/mode flow uses the **local** params into `FoldHistory(...)` at lines 97-98, not `self.*`. The two assignments are write-only. Action: delete both lines (FoldHistory construction already uses the locals). Keep only if you want them as public introspection attrs — but nothing reads them today.

**2. Redundant local `import logging` inside `step()`** — `experiment.py:218` — DEAD CODE / DUPLICATION — **SAFE**
Module already imports `logging` at line 4. Action: delete line 218; the `logging.getLogger(__name__).warning(...)` at 219-221 works off the top-level import.

**3. Decompose over-long `step()` (38 lines, 4 concerns)** — `experiment.py:192-229` — OVER-LONG FUNCTION — **SAFE (verbatim cuts)**
`step()` interleaves: end-fold (194), verbose display (195-196), adapter fold-metric assembly (197-205), per-fold partial save try/except (210-222), advance+start-next (223-229). Two byte-identical extractions, both pure relocation:
- `_emit_adapter_fold_end(self, fold) -> None` ← lines **197-205** (guarded by `_adapter is not None`; default-off → no §0.1 effect).
- `_save_fold_partial_safe(self) -> None` ← lines **210-222** (the `try/except storage.save_fold_partial`). Relocation only; the `storage.save_fold_partial(...)` call args are unchanged.
Result: `step()` shrinks to its control flow (end → emit → save → advance). No numeric block is touched.

**4. Dead defensive branch: `hasattr(self.timer, 'get_duration')`** — `experiment.py:306` — DEAD BRANCH — **SAFE**
`TimeHistory.get_duration` always exists (`time_history.py:34`) and after `self.timer.stop()` at line 298 `duration` is set, so it cannot raise. The `else 0` fallback is unreachable. Inside `_adapter is not None` only → no §0.1 effect. Action: replace `self.timer.get_duration() if hasattr(...) else 0` with `self.timer.get_duration()`.

**5. Codename narration comments → move substance to docs, keep invariants** — STALE/NARRATION — **SAFE (comment-only)**
- `experiment.py:85-86` `# AUDIT-C2 — per-task monitor overrides; see fold.FoldHistory.` — drop the codename; keep the load-bearing half: *"Default None preserves legacy single-metric (F1) behaviour."*
- `experiment.py:90` `# F50 B1 — selector min-epoch gate (skip init artifacts).` — drop `F50 B1`; keep *"min-epoch gate for best-selection."*
- **PRESERVE (load-bearing invariants, do NOT trim substance):** `110-114` (run_id/fold_ids fan-out collision contract — this is the on-disk-naming guarantee that `fold_label` enforces), `257-261` (run_id rundir-leaf contract + the PID-collision rationale behind the `os.getpid()` fallback), `206-209` (why the partial save lives in `step()`: OOM SIGKILL / SSD SIGBUS mid-run), `125` (`# Lazy init to avoid circular imports`). None are dated "fixed 2026" lines; the only removable narration is the two audit codenames.

**6. Module-level logger** — `experiment.py:218-221, 290-292` — DUPLICATION / BEST-PRACTICE — **SAFE**
`logging.getLogger(__name__).warning(...)` is spelled inline twice. Sibling `records.py:16` already uses the idiomatic `logger = logging.getLogger(__name__)` module global. Action: define `logger = logging.getLogger(__name__)` once at module top, use `logger.warning(...)` at both sites.

**7. `get_curr_fold()` ⇄ `fold` property duplicate body** — `experiment.py:170-177` — DUPLICATION — **SAFE**
Both return `self.folds[self.curr_i_fold]`. Both names are used externally (`fold` property: `next_cv.py:195/199/204`, `category_cv.py:97`; `get_curr_fold()`: `mtl_cv.py:1878/1902`, `category_cv.py:88`, `next_cv.py:185`), so neither can be deleted. Action: collapse the index expression to one site — make `get_curr_fold(self) -> FoldHistory: return self.fold`. Removes the duplicated indexing, keeps both public names.

---

### Best-practice / typing fixes (all SAFE, low value)

- **`FlopsMetrics.__init__(self, flops, params)` untyped** — `experiment.py:19` — add `flops: float, params: int`. Callers pass numerics (`next_cv.py:174`, `mtl_cv.py:1862`, `category_cv.py:76`) or `0` for the no-profile path; types are stable.
- **Missing `-> None` on metadata setters** — `experiment.py:233,236,239` (`set_model_parms`, `set_model_arch`, `set_flops`) — annotate. `set_adapter` already has it (244).
- **`set_adapter(self, adapter)` untyped param** — `experiment.py:244` — type as `Optional[object]` (or a small `TrackingAdapter` Protocol matching `adapters.py` callbacks `on_run_start/on_fold_end/on_run_end/close`).

### Correctness smell (NOT for §0.1, flag MED — do not auto-apply)

- **Hardcoded `"f1"` in the adapter fold-metric block** — `experiment.py:202-203` (`th.val.best("f1")`, keys `{task}_best_f1`) — ignores the per-task monitor (`task_monitors`) and `mode` (uses `MetricStore.best` default `'max'`). For a reg task whose `BestModelTracker.monitor` is `accuracy`, the emitted `best_f1` is mislabelled/possibly-absent (`"f1" in th.val` → `(-1, 0)`). This only affects external-tracking payloads (adapter default `None`), so it is repro-neutral but a latent bug. Suggested: use `th.best.monitor` instead of the literal. **MED** because it changes adapter output; leave to a follow-up, not part of the slimming pass.

### Considered and rejected (NOT dead)
`fold_label` (used by `storage.py:338/358/503/521`), `fold_ids` (set by `mtl_cv.py:1358`), `run_id`/`min_epoch` (set by `scripts/train.py:162-163,236-237`), `datasets` (read `storage.py:475-476`), `model_parms`/`model_arch`/`model_type` (read by storage+display), `FlopsMetrics` + `to_dict` (read `storage.py:486`), `_compare_and_save_records` and its broad `except` (intentional: never abort the on-disk save) — all live. `start()`'s `on_run_start` block (266-272) could also extract to `_emit_adapter_run_start()` for symmetry but is low value.

---

### Verdict
Realistic **pure-deletion slimming ≈ 8-12%** (~25-35 lines: dead `monitor`/`mode` state, redundant `import logging`, the dead `hasattr` branch, two codename comments, logger dedup). The larger win is **clarity, not line count**: extracting items 3 + 7 cuts `step()` from 38 → ~15 lines and removes the duplicate fold-index/logger spellings without relocating any artefact-writing or numeric logic. **All TOP-7 items are SAFE-to-apply-now and repro-neutral** (every adapter-scoped change is behind the default-`None` guard; no `storage.*` call signature is touched). **No item requires a metric-parity A/B** — there is no hot numeric block in this file (loss/metric math lives in `metrics.py`/`metric_store.py`/`fold.py`). The only non-now item is the MED `"f1"` adapter literal (item under Correctness smell), which is a separate bug-fix, not a slimming move.

---

## File audit: src/training/helpers.py

# Audit — `src/training/helpers.py` (387 lines): slimming + best-practices

Scope: 4 public helpers (`compute_class_weights`, `setup_optimizer`, `setup_per_head_optimizer`, `setup_scheduler`) + 1 private (`_build_reg_head_warmup_decay_lambda`). Consumers: `category_cv.py:14/37/45/46`, `next_cv.py:10/115/143/144`, `mtl_cv.py:57-60/1689/1702/1717`, and tests `tests/test_training/test_helpers.py` (locks the public API + `_build_reg_head_warmup_decay_lambda` + `setup_scheduler` LambdaLR per-group behavior). No behavior change is proposed to any numeric path; everything below is byte-preserving or comment-only.

---

## TOP items (ranked by value / risk)

### 1. DEAD IMPORT — `DataLoader` never used — **SAFE**
`helpers.py:22` `from torch.utils.data import DataLoader`. Grep of the whole file shows `DataLoader` appears only on the import line (no body use). Delete the line. Zero risk, removes an import.

### 2. DUPLICATE `typing` import line — merge — **SAFE**
`helpers.py:8` (`from typing import Optional, Union`) and `helpers.py:23` (`from typing import Iterable`) are two separate imports from the same module, with `numpy/torch/sklearn/torch.optim` imports wedged between them (line 23 sits *below* the `torch.utils.data` import — out of PEP-8 import grouping). Merge into one `from typing import Iterable, Optional, Union` at line 8; drop line 23. SAFE.

### 3. DUPLICATION inside `setup_scheduler` — base-LR overwrite repeated 3× — **SAFE/MED**
The identical block
```python
if not multi_group_per_head:
    for pg in optimizer.param_groups:
        pg["lr"] = max_lr
```
appears verbatim at `helpers.py:304-306` (constant), `312-314` (cosine), `328-330` (warmup_constant). Extract a private `_overwrite_base_lr(optimizer, max_lr, multi_group_per_head)` and call it in the 3 sites. Logic is identical → behavior-preserving. Category: numeric-adjacent (it seeds the optimizer base LR that the scheduler reads), but the extracted statements are byte-identical, so confidence is high — mark **SAFE**, optionally smoke-verify one constant + one cosine run. Saves ~6 lines and de-risks future edits (a 4th scheduler branch that forgets this guard would silently flatten per-head LRs — see the load-bearing note at `helpers.py:292-295`).

### 4. OVER-LONG FUNCTION — `setup_scheduler` (`helpers.py:253-387`, ~135 lines) — extract per-branch builders — **MED**
It is a 5-way `if scheduler_type == …` ladder. The two **complex, exploratory** branches dominate the length and only run in archived sweeps (see §"reachability" below):
- `warmup_constant` body `helpers.py:320-349` → extract `_build_warmup_constant(optimizer, max_lr, epochs, steps_per_epoch, pct_start, multi_group_per_head)` returning a `SequentialLR`. Pure over its args; cut verbatim.
- `reg_head_warmup_decay` body `helpers.py:350-382` → extract `_build_reg_head_warmup_decay(optimizer, epochs, steps_per_epoch, peak_mult, warmup_epochs, plateau_epochs)` returning a `LambdaLR`. Cut verbatim; it already delegates the math to `_build_reg_head_warmup_decay_lambda`.

After extraction `setup_scheduler` becomes a thin dispatcher (~40 lines). **Risk MED, not SAFE**: these mutate `optimizer.param_groups` in-place and build schedulers — moving identical statements into an immediately-called helper is behavior-preserving, but because they touch the LR trajectory I'd gate on a one-fold A/B (assert per-step LR list is bit-identical before/after) for `warmup_constant` and `reg_head_warmup_decay`. The `onecycle`/`constant`/`cosine` branches (the only ones on the frozen-§0.1 path) are short — leave inline.

### 5. STALE / NARRATION COMMENTS that belong in docs — **SAFE (comment-only), preserve invariants**
These are dated audit-codename narration; trim to a 1-liner that keeps the *invariant*, move the story to `docs/findings/`. Each is comment-only → SAFE, but the **bolded invariant must survive**:
- Module docstring `helpers.py:1` "extracted in Phase 4a" — pure history; trim.
- `compute_class_weights` docstring `helpers.py:35-41` — "PR #8's on-device-tensor optimization" / `can't convert mps:0…` is narration. **KEEP the invariant**: "tensors routed through `.cpu().numpy()` once at this boundary" (test `test_helpers.py:58-67` locks MPS acceptance). Drop the PR attribution.
- `setup_per_head_optimizer` docstring `helpers.py:113-128` — "F48-H3" codename. **KEEP**: "requires `cat_specific_parameters`/`reg_specific_parameters`/`shared_parameters`; only `MTLnetCrossAttn`" (this matches the runtime `raise` at 129-136).
- α-peel comment `helpers.py:149-155` ("F50 B9 … STL at ep 17-20 hits α~2.0 … F50_T3_HYPERPARAM_BRAINSTORM.md") — **KEEP**: "peel α into a zero-WD group so AdamW WD doesn't pull it to 0". Drop the F50 story.
- β-peel comment `helpers.py:164-170` ("2026-06-12 HANDOFF_AUDIT X3 / CODE_AUDIT P1-C … β decay ≈0 by epoch 25") — **KEEP the env-var contract**: "`MTL_BETA_NO_WD=1` peels β into the zero-WD group; default unset → no-op". This is a load-bearing env-var contract, not just narration.
- D3/D6 comment `helpers.py:178-184` — **KEEP**: "split reg into encoder vs head when either `reg_encoder_lr`/`reg_head_lr` is set; unset one → defaults to `reg_lr`". Drop the codenames.
- `_build_reg_head_warmup_decay_lambda` `helpers.py:222` "F50 F64/B2", and `setup_scheduler` branch comments `helpers.py:320-327` (F48-H2) and `350-357` (F64/B2) — trim codenames, keep the ramp/hold/decay shape description.

**Do NOT touch (load-bearing invariants):** the sklearn absent-class block `helpers.py:54-60` (numeric: absent classes default to 1.0 to preserve CE normalisation; test `test_helpers.py:84` locks it), the `requires_grad` frozen-param filter rationale `helpers.py:137-142` (prevents AdamW WD shrinking frozen weights; tests `test_mtlnet_crossattn_lambda0_gradflow.py` lock it), and the `multi_group_per_head` guard comment `helpers.py:292-295`.

### 6. BEST-PRACTICE: inconsistent union-typing style — **SAFE**
Mixed PEP-604 and `typing`: `Iterable[...] | None` at `helpers.py:74,108` vs `Optional[float]` at `helpers.py:259` and `Union[np.ndarray, torch.Tensor]` at `helpers.py:27`. Pick one (repo already uses `|` in two sites). Cosmetic, SAFE — but `compute_class_weights` `Union` is referenced by the docstring, keep readable.

### 7. BEST-PRACTICE: missing return annotations + E731 lambda — **SAFE**
- `setup_scheduler` (`helpers.py:253-264`) has no `->` return type (siblings annotate `-> AdamW`). Add `-> torch.optim.lr_scheduler.LRScheduler` (or a Union). `_build_reg_head_warmup_decay_lambda` (`helpers.py:216-221`) also lacks `-> Callable[[int], float]`.
- `identity_fn = lambda _: 1.0` at `helpers.py:372` is a flake8 E731 (assign-lambda). Replace with `def identity_fn(_): return 1.0` or `(lambda _: 1.0)` inline. Cosmetic.

---

## NOT dead (do not delete) — reachability notes
- **`warmup_constant` / `reg_head_warmup_decay` scheduler branches are NOT dead**: exposed via `scripts/train.py:621-624` `--scheduler` choices and exercised by archived sweeps (`scripts/canonical_improvement/t61–t64*.sh`, `scripts/run_f48_h2_warmup_constant.sh`, `scripts/run_f50_b2_f52_f65_fl.sh`) and by `test_helpers.py:228` (`test_setup_scheduler_lambda_lr_per_group`). They are *exploratory* (the frozen §0.1 canon uses only `cosine` (B9) / `constant` (H3-alt) per CLAUDE.md), but live + test-locked. Treat as **RISKY** to remove — at most relocate to an `exploratory_schedulers.py` module, which I do not recommend now (CLI + test coupling).
- **`MTL_BETA_NO_WD`, `reg_encoder_lr`/`reg_head_lr` (D3/D6), `alpha_no_weight_decay`** are all wired from `experiment.py:53/112-128` ↔ `mtl_cv.py:1687-1699`/`train.py` — live config, not dead params.

## Known cross-file duplication (informational, low priority)
`compute_class_weights`' sklearn-balanced formula `N/(C·n_c)` is intentionally re-implemented in `src/losses/calibrated.py:100` (comment there explicitly says "Matches `compute_class_weights`"). This is a deliberate reproduction for the cat-ceiling, documented at `calibrated.py:18`; not a refactor target (coupling them risks the calibrated baseline). Note only.

---

## Verdict
Realistic slim: **~20-25% fewer lines** (387 → ~290-300) achievable with **zero numeric change** — almost entirely from trimming dated audit-codename narration (item 5) plus the two dead/duplicate imports (items 1-2) and the 3× base-LR dedup (item 3). **SAFE-to-apply-now:** items 1, 2, 3, 5 (comment trims preserving the bolded invariants), 6, 7. **Needs a metric-parity A/B (MED):** item 4 — extracting the `warmup_constant` and `reg_head_warmup_decay` builders (verify per-step LR trajectory is bit-identical; these run only in archived sweeps, not the frozen canon). The numeric core — `compute_class_weights` absent-class fill (54-66), the `requires_grad`/α/β WD-group peeling (137-213), and the `onecycle/constant/cosine` branches — should stay logically untouched.

---

