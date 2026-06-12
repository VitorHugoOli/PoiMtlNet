# Deep code-flow audit + "why doesn't MTL reg beat STL" analysis (2026-06-12)

> User-requested pre-closure deep audit: (1) hunt the next C25-class silent flaw in the MTL code
> flow; (2) analyze why MTL reg only *matches* the STL ceiling and whether anything can change that
> (with literature). Method: 3 hostile code auditors (data path / training loop / model forward) +
> 1 literature analyst, all findings **independently re-verified at the code level** by the design
> agent before entering this record. Every claim below carries file:line evidence.

> ## ✅ RESOLUTION (X-series executed 2026-06-12 — all probes NULL; full detail: `docs/results/mtl_improvement/X_SERIES_FINDINGS.md`)
> The three "MTL-only levers" that were structurally disabled (P0-A aligned cross-attn, P0-B
> log_T-KD-on-G, P1-C β-no-WD) were fixed and exercised; the two stress-tested claims (mixing-dead,
> "matches"-precision) were checked. **Every result is null or negligible** — the "matches, can't beat"
> verdict is now earned at a strictly higher standard, and champion G is unchanged:
> - **P0-A (X1):** roll probe Δcat-F1 **−0.004** → cross-attn mixing is genuinely dead (not a noise-pair
>   artifact); aligned-training run justified-skipped.
> - **P0-B (X2):** aux gate **FIXED** (`folds.py` + `p1_region_head_ablation.py`); first real KD-on-G test
>   **NULL** (FL reg +0.05 / AL reg −0.13, FL cat −0.57); gate fix empirically inert on G (Δ −0.022 vs R0).
> - **P1-C (X3):** β **decays to ≈0 by gradient even with WD removed** (mean final −0.0001) → β→0 is not a
>   WD artifact; β-no-WD run NULL (FL reg +0.015 / cat +0.139).
> - **P1-D (X4):** fp32-eval vs fp16-eval Δreg **−0.005** → the "matches" −0.31pp gap is precision-clean.
> - **P1-F** (CLI bool coercion) was fixed in-session; **P1-E** (dead `category-weight` under alt-opt) and
>   **P1-G** items are documentation-only (no rerun). Mechanisms 1-6 → future work (`closing-data`).

---

## PART 1 — Code-flow findings

### P0-A. Cross-attention trains on RANDOMLY-PAIRED samples, evaluates on ALIGNED pairs
**Found independently by all 3 code auditors. Verified.**

- For `check2hgi_next_region`, task_a (check-in modality) and task_b (region modality) are the
  **same N windows**, row-aligned by construction (`src/data/folds.py:856-858` "only the label
  column differs").
- But `_create_check2hgi_mtl_folds` builds **two independent `shuffle=True` DataLoaders** over the
  same `train_idx` (`folds.py:1054-1061` task_b, `1076-1080` task_a) with **no `generator=`
  argument** (`_create_dataloader`, `folds.py:285-318`) → each `iter()` draws its own permutation
  from the global RNG (the first `iter()` advances the RNG before the second draws → permutations
  always differ).
- The train loop zips them per batch (`mtl_cv.py:463`; `src/utils/progress.py:36-42`) and the
  cross-attn block mixes **row i ↔ row i** (`mtlnet_crossattn/model.py:127-148`). So at **train**
  time every window cross-attends a *random unrelated* window's other-modality stream. At **val**
  both loaders are `shuffle=False` over the same `val_idx` → **aligned**.
- **What it does NOT do:** corrupt losses/labels (each task scores its own batch) or invalidate
  G's reported numbers (eval is aligned and consistent across every arm; selection consistent).
- **What it DOES do:** (1) the cross-attn pathway can only learn to treat the other stream as
  batch-level noise → per-sample cross-task transfer was **never trainable**. The study's
  "K/V mixing is dead" (F52 P5 identity-attn ≈ baseline) and "cat gain is architecture, not
  region-transfer" conclusions are **mechanistically consistent with this wiring** — they may be
  consequences of noise-pair training, not intrinsic facts about the task pair. (2) The +3pp cat
  gain was measured under an eval-time aligned channel training could never exploit — the
  cat-transfer decomposition inherits the same property.
- **Why it matters for closure:** this is the first candidate lever found since C25 that is
  **MTL-only — it cannot lift the STL ceiling** (STL has no second stream). A fix (one shared
  permutation for both loaders) is exempt from the rising-tide magnitude rule.
- **Affected:** every `mtlnet_crossattn*` MTL run — v11/B9 through v16/G, all states, all tiers.
- **Probes (→ HANDOFF_AUDIT X1):** (i) zero-training falsification: evaluate a trained G twice,
  normal vs task-b val batch rolled by 1 (`torch.roll(x_b,1,0)`) — Δ≈0 ⇒ the model ignores
  pairing (mixing truly dead, claims safe); Δ>0 ⇒ eval depends on alignment training never saw;
  (ii) the real test: ONE G run at AL+FL with a single shared permutation feeding both loaders —
  any lift = the misalignment was suppressing cross-modal transfer all along.

### P0-B. `next_stan_flow_dualtower` missing from the aux gate — prior AND log_T-KD structurally DEAD on the dual-tower
**Verified.** `_HEADS_REQUIRING_AUX_MTL = {next_getnext_hard, next_getnext_hard_hsm,
next_stan_flow, next_stan_flow_hsm}` (`src/data/folds.py:933-937`) omits the dualtower head →
`use_aux=False` → plain loaders → `get_current_aux()` returns `None` in every forward. Two
consequences:
1. `_apply_prior` always takes the defensive branch `logits + α·0.0`
   (`next_stan_flow_dualtower/head.py:281-287`) — **every "prior-ON" dual-tower arm ever run was
   actually prior-OFF** (the T2.1 prior-ON/OFF deltas were noise between identical configs).
2. The trainer's KD branch requires `_aux is not None` (`mtl_cv.py:514-519`) → the `c25_gv2.sh`
   arms `g_kd0.1`/`g_kd0.2` (lines 40-41) were **no-ops**. CHAMPION §5's "log_T-KD was tested on
   the dual-tower — adds nothing" is a **dead-codepath artifact: it never ran.** KD was the one
   confirmed reg lever pre-G (v12 default W=0.2) — whether it helps G is **genuinely untested**,
   and it too is exempt from the rising-tide rule (KD distills log_T into the MTL head; the STL
   ceiling froze at α=0 without KD).
- G's own numbers unaffected (G pins prior-OFF + KD 0.0 → numerically identical either way).
- **Fix + probe (→ X2):** add the head (and `_hsm` variants) to both gate sets
  (`folds.py:933-937`, `scripts/p1_region_head_ablation.py:104`) or gate on
  `hasattr(head,"log_T")`; assert `get_current_aux() is not None` in a smoke batch; then ONE
  real `G + --log-t-kd-weight 0.2` run at AL+FL vs G.

### P1-C. The `aux` fusion scalar β is weight-decayed at wd=0.05 and never logged
**Verified.** `self.beta = nn.Parameter(torch.tensor(0.1))` (`head.py:221`) sits in the reg
param group; `setup_per_head_optimizer` peels only `next_poi.alpha` into the zero-WD group
(`src/training/helpers.py:148-161`). AdamW pulls β→0 every step — the **exact mechanism the study
already diagnosed for α (F50 B9)**, now acting on what CHAMPION §2 calls "the key lever" (the
shared-pathway coefficient). Per-epoch diagnostics log `head_alpha` only (`mtl_cv.py:743-746`) —
β drift is invisible. Could partially manufacture the "shared pathway adds little" reading.
**Probe (→ X3):** log β per epoch on one G run; if it decays materially, one run with β in a
0-WD group.

### P1-D. Eval-precision asymmetry inside the "matches" verdict
**Verified.** `evaluate_model` autocasts fp16 unconditionally on CUDA (`mtl_eval.py:110-126`;
`MTL_DISABLE_AMP` gates only the **training** autocast, `mtl_cv.py:289-299`), and training runs
fp16 **without GradScaler**. The STL p1 ceiling harness never autocasts (fp32). So the R0 bar
compares fp16-eval MTL metrics to fp32-eval ceiling metrics, and `_rank_of_target`
(`src/tracking/metrics.py:114-147`) counts strictly-higher logits → fp16 quantization over
~5-9k region logits creates more exact ties → **tie-optimistic** ranks on the MTL side only.
The headline Δreg is decided at −0.09…−0.31pp — within plausible precision delta.
**Probe (→ X4):** extend the AMP escape hatch to eval, retrain ONE G seed, score the same
weights fp16-eval vs fp32-eval; same for one ceiling fold if needed.

### P1-E. v11/B9: `--category-weight 0.75` is a DEAD FLAG under `--alternating-optimizer-step`
**Verified.** With alternating, `loss = task_a_loss` (even batches) / `task_b_loss` (odd) raw
and unweighted — `_get_weighted_loss`/static_weight is never called (`mtl_cv.py:596-612`).
The v11/B9 canon (FL/CA/TX) therefore trained as 50/50 alternating single-task steps; the 0.75
weighting only ever existed at H3-alt small states (no alt-opt). v11 numbers stand
(self-consistent), but every doc describing B9's mechanism as "static_weight 0.75" describes a
weighting that never entered the objective; any past `category-weight` sweep under B9 was a no-op.
**Action:** documentation correction (CANONICAL_VERSIONS §v11 note) — no rerun needed.

### P1-F. CLI bool coercion: `KEY=False` silently means True for head/model/loss params
**Verified + FIXED this session.** `_coerce_cli_value` used `json.loads`, which rejects
Python-style `True/False` → they stayed strings, and `bool("False") == True`. G's own
`freeze_alpha=True` worked by truthiness coincidence; any ablation passing `freeze_alpha=False`
silently inverted. Fixed in `scripts/train.py` (accepts `True/False/None`), guarded by
`tests/test_configs/test_cli_param_coercion.py`.

### P1-G. Smaller, conditional (documented for the record; no closure action)
- **Dualtower head without dualtower model silently cripples reg** (`head.py:264-270` zero-fills
  the private feature with no warning under `--model mtlnet_crossattn`). Canon pins both; a
  warn-on-train guard is cheap future hygiene.
- **Cat-ceiling comparand asymmetry**: the STL cat ceiling is `next_gru` 2-layer/dropout 0.3
  **+ logit-adjust τ=0.5**; G's cat head is built 4-layer/dropout 0.1, plain CE
  (`experiment.py:355-356` + `mtlnet/model.py:292-305` inject over head defaults). The +3pp cat
  comparison spans head depth/dropout/calibration differences, consistent with (and absorbed by)
  the "architecture-dominated" decomposition — but disclose in the paper.
- **`per_metric_best` takes `max()` for every key incl. `loss`/`ood_fraction`**
  (`storage.py:216-238`) — diagnostic JSON only; headline bases unaffected.
- **Legacy-preset only**: `evaluate_model` cycles the shorter val loader → duplicated samples in
  legacy `{category,next}` val metrics (`mtl_eval.py:116`); `itertools.cycle` replays cached
  batches with stale aux for unequal-length task pairs (`progress.py:36-42`) — both INERT for
  check2hgi (equal loader lengths) and G (prior-OFF).
- **Region branch lacks a row-count/freshness assert** between `sequences_next.parquet` and
  `next_region.parquet` at fold-build (`folds.py:973-986`) — same stale-artifact class as the
  log_T case, currently unguarded (the log_T case itself IS fenced now).
- **PAD coupling wart**: pad masking is really "abs-sum == 0" and works only because
  `PAD_VALUE == 0`; pads are PROVEN exact zero vectors end-to-end (verified empirically: 0 false /
  0 missed masks over 20k AL rows) — but a future `PAD_VALUE` change breaks every mask silently.
  Cheap guard: assert `InputsConfig.PAD_VALUE == 0` at model init.
- **`min_size_truncate` desyncs the scheduler** (`mtl_cv.py:1237-1244` uses max-length
  unconditionally) — non-default ablation path only.
- **Per-head LR flags dead under OneCycleLR** — already documented (CHAMPION §2), reproduced
  empirically (all groups ride 1.2e-4→3e-3); flagged here because v16 carries three cosmetic
  flags and a future per-head-LR sweep under onecycle would be a silent no-op. A guard or a
  `max_lr` list is the fix-shape.

### Audited-clean (so nobody re-chases)
Pad semantics end-to-end (zero vectors proven at the parquet level); target/input alignment +
no cross-user bleed; **user-disjoint folds** (`StratifiedGroupKFold` on userids) identical
between MTL and the p1 STL harness → fold-paired G-vs-ceiling comparisons; OOD/indist definition
(train-fold label set; cannot drop seen classes); region embeddings frozen-by-materialization,
target step never in the input window; class weights/calibration train-fold-only; KD strictly
gated at W=0; log_T finite under prior-OFF (no 0·−inf); BestTracker clones (no mutation bug);
scheduler stepping exact for the default strategy; eval under `no_grad`+`eval()`; seeding incl.
CUDA determinism; canon v16 ≡ CHAMPION §3 field-by-field (two deltas: §3 adds `--no-checkpoints`;
bare `--canon v16` runs dev seed 42 — pass `--seed` for reporting runs).

---

## PART 2 — Why doesn't MTL reg beat STL? (and what could change it)

### The literature verdict: parity is the EXPECTED outcome here
- **Theory**: at cos≈0 the aux gradient contributes no first-order progress on the main loss in
  expectation (Du et al., arXiv:1812.02224 — their gradient-gating rule reduces to "neutral" at
  orthogonality); MTL gains concentrate where the main task is data-starved/under-fit (Bingel &
  Søgaard, EACL 2017; AuxiLearn, ICLR 2021); at k=2 with a tuned scalarization baseline no MTO
  method helps (Kurin NeurIPS'22; Xin NeurIPS'22). Our main task is data-rich and tuned → zero
  expected loss-side gain.
- **Information budget**: the aux task carries ~log₂7 ≈ 2.8 bits vs the ~10-13-bit region target.
  Every positive category-aux result in the next-POI literature uses **180-300+ category
  vocabularies** (iMTL: 184-251; GETNext: fine-grained Foursquare; MCARNN: full category set as
  latent topics). Our 7-class aux is an outlier on the weak side → frame the negative as the
  **weak-auxiliary regime**, not "category auxiliaries don't work."
- **Domain papers dissolve on inspection**: iMTL (IJCAI'20) has NO single-task-location ablation —
  its gain comes from *conditional coupling* (predicted activity scores concatenated into the
  location head, Eq. 8), and its "parallel MTL without interaction" arm LOSES to full iMTL.
  GETNext (SIGIR'22) uses category mainly **input-side** (time-aware category embeddings + a
  transition map added to logits); its aux heads are unused at inference and the ablation removes
  two decoders at once, single-seed. Mobility Tree (AAAI'24) is the strongest loss-side-only
  claim (−4.5% rel. without MTL) but single-seed with deltas at seed-noise scale. **MCARNN
  (IJCAI'18) is the one genuine same-architecture MTL>STL ablation** (NYC Acc@10 0.4601 vs
  0.4268) — pre-empt it: 2018 low-capacity RNN, batch 16, no seeds/variance, rich latent-topic
  aux, and its own λ-sweep shows the location task improving as the aux weight SHRINKS
  (a low-capacity-regularization story, not a transfer story).
- **One framing caveat for the paper (reviewer-proofing)**: Fifty et al. (NeurIPS'21) argue raw
  gradient cosine is a poor transfer predictor — their *lookahead inter-task affinity* (does a
  step on task A reduce task B's loss?) can be nonzero at cos≈0 (second-order/subspace effects).
  Phrase the orthogonality claim as a **first-order, average statement** ("no average first-order
  conflict or cooperation on the shared trunk"), not "nothing to exploit"; optional hardening =
  one lookahead-affinity measurement.

### BUT: three study-internal levers were never actually exercised (audit Part 1)
The honest status of "reg can't beat STL" after this audit: **not yet fully earned.** Three
mechanisms that move G *without* moving the STL ceiling (exempt from the rising-tide magnitude
rule) were structurally disabled:
1. **Aligned cross-attn training** (P0-A) — per-sample cross-modal transfer was untrainable;
2. **log_T-KD on the dual-tower** (P0-B) — the one confirmed pre-G reg lever, never actually ran
   under G;
3. **β free of weight decay** (P1-C) — the shared→reg pathway coefficient under constant decay.
These are the X-series probes (HANDOFF_AUDIT). If all three are null, the "matches, can't beat"
claim is closed at a strictly higher standard; if any lifts, the study has a real post-G lever.

### Literature mechanisms beyond this study (→ future work / closing-data / next paper)
Ranked by (plausibility × cost), all with citable precedent:
1. **Conditional coupling** — feed cat-head output (7-dim softmax or penultimate features) into
   the reg head (concat/FiLM), train+inference (iMTL pattern). Cheap; modest prior (their aux
   space was ~200-class).
2. **Category-conditioned logit prior** — `logits_reg += W·p(cat)` with a learned/empirical
   region|category matrix (CatDM WWW'20 / LBPR IJCAI'17 two-stage pattern). Cheap; with 7 classes
   a soft prior, not a filter.
3. **Coarse-to-fine factorization of the REGION vocabulary itself** (semantic IDs / learned
   hierarchies — arXiv:2506.01375; BiGSL arXiv:2411.01169): attacks the large-softmax estimation
   problem, orthogonal to the MTL wall (note T5.3 falsified *hierarchical softmax* on a
   spatial-cluster tree at FL — the semantic-ID/codebook family is a different mechanism).
   Medium.
4. **Category-transition input feature** (GETNext pattern): 7×7 transition row of the current
   check-in's category fused into the reg head input. Cheap.
5. **Region→category consistency loss** — KL between the region posterior marginalized through
   the region→category map and the cat head's posterior (MobTCast NeurIPS'21 analogue; same
   family as our log_T-KD win). Medium; would be a contribution, not a replication.
6. **Gradient-gated aux weighting** (Du'18/AuxiLearn): predicted null at cos≈0 — run only as a
   falsification checkbox if ever needed.

### Disposition (user decision 2026-06-12 pending; recommendation)
- **X-series (X1-X4) → THIS study** (HANDOFF_AUDIT, gates closure): they stress-test the study's
  own central claims (mixing-dead, KD-dead-end, β lever, the "matches" verb) and are cheap.
- **Doc corrections → done this session** (CHAMPION §5 KD row, CANONICAL_VERSIONS v11 B9 note,
  PAPER_UPDATE literature-positioning).
- **Mechanisms 1-6 + Fifty-style affinity hardening → future work** (carry into `closing-data` /
  the next study; recorded here + INDEX Tier 7 card).
