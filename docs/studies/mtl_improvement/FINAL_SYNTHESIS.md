# FINAL SYNTHESIS — `mtl_improvement` (CLOSED 2026-06-12)

> **The one document to read about this study.** Launched 2026-05-16, closed 2026-06-12, branch
> `mtl-improve`. Everything below is the distilled, audit-hardened outcome; the full trail lives in
> the reading map (§9). Successors: the **`closing-data`** study (`docs/studies/closing_data/`)
> inherits the pre-freeze gate + the CA/TX majors; the BRACIS paper restatement (T6.2) is
> author-side per `PAPER_UPDATE.md`.

---

## 1. Outcome in one paragraph

The study set out to close the −7…−17 pp "MTL sacrifices region-prediction" gap. It found the gap
was **not architectural at all**: a single silent confound (C25 — the MTL heads trained on
class-WEIGHTED CrossEntropy while the metric and the STL ceilings are unweighted) produced the
entire narrative. With the confound fixed and a reg-private dual-tower architecture (**champion G**),
a **single jointly-trained MTL model MATCHES the single-task STL region ceiling (matched metric,
Δ −0.09…−0.31 pp) and BEATS the single-task category ceiling by +2.6…+4.1 pp at all 4 available
states × 4 seeds** — inverting the original tradeoff. The mechanism is **gradient orthogonality**
(cos(∇cat,∇reg) ≈ 0 on the shared trunk, tested intrinsic): the tasks neither fight nor cooperate at
the loss level, so no MTL balancer helps, more parameter-sharing hurts, and the winning design
*exploits* orthogonality (protect reg in a private tower, let cat harvest the shared encoder). The
result survived three hostile audit passes, one data-integrity fix, and an X-series that exercised
every structurally-disabled MTL-only lever — all null.

## 2. The headline numbers (R0 matched-metric bar — THE citable table)

4 states × 4 seeds {0,1,7,100}, FULL `top10_acc` both sides, fold-paired with the STL harness.
Source: `docs/results/mtl_improvement/R0_matched_metric_bar.json` (every value independently
recomputed during the closure audit).

| state | G reg (full) | STL reg ceiling | Δreg | G cat-F1 | STL cat | Δcat |
|---|---|---|---|---|---|---|
| AL | 62.57 ± 0.10 | 62.67 ± 0.13 | **−0.09** (matches) | 52.91 ± 0.27 | 50.35 | **+2.56** |
| AZ | 54.68 ± 0.24 | 54.80 ± 0.22 | **−0.12** (matches) | 54.48 ± 0.74 | 50.39 | **+4.08** |
| GE | 58.35 ± 0.04 | 58.44 ± 0.06 | **−0.09** (matches) | 61.43 ± 0.26 | 57.50 | **+3.93** |
| FL | 72.97 ± 0.06 | 73.27 ± 0.06 | **−0.31** (matches) | 73.16 ± 0.04 | 69.96 | **+3.20** |

Composite (2-model STL deploy): its old +7–12 pp reg edge collapses to **+0.53 pp at FL** while G
wins cat +3.2 at ~half the deploy footprint → dominated on the **joint** reading (not "strictly").
The −0.31 FL gap is fp32-precision-clean (X4) and pairing-safe (X1).

**Champion G** (= canon **v16**, the `scripts/train.py --task mtl` default): `mtlnet_crossattn_dualtower`
+ reg head `next_stan_flow_dualtower` (`raw_embed_dim=64 fusion_mode=aux freeze_alpha=True
alpha_init=0.0` — prior-OFF), cat head `next_gru`, `static_weight cw=0.75`, both heads UNWEIGHTED CE,
onecycle max-lr 3e-3, KD off, v14 substrate, 50ep/5f/bs2048. Exact reproduce command + prerequisites:
`CHAMPION.md §3`; version pin + traceback (`--canon v11..v16`): `docs/results/CANONICAL_VERSIONS.md`.

## 3. The six findings (what a future agent must know)

1. **C25 — the class-weighting confound WAS the "MTL→STL reg gap."** `default_mtl` silently trained
   MTL heads on class-weighted CE while the reported Acc@10 + the STL ceilings are unweighted;
   class-balancing optimizes *away from* top-K, depressing MTL reg ~10–14 pp (scales with class
   count) and cat ~3–5 pp. Fix: per-task `use_class_weights_{reg,cat}=False` defaults. Everything
   pre-C25 that measured "MTL sacrifices reg" measured the confound. (`docs/CONCERNS.md §C25`.)
2. **The dual-tower is the right architecture; capacity is not the lever.** Reg needs a **private,
   un-diluted pathway** (raw region seq → private STAN tower) fused **additively** (`aux`; `gated`
   competition dilutes) with the biased α·log_T prior OFF. The private STAN is load-bearing (lighter
   towers lose 1.8–3.4 pp). Architecture *capacity/quality* was falsified 5 independent ways
   (MoE/CGC, SwiGLU, MulT, x-stitch, more blocks — all null on reg). The cat-private variant **G′ is
   a closed FL-only dead-end** (craters small-state cat by up to −15 pp; head↔task-cardinality
   mismatch — `CONCERNS §C26`). Do-not-retry table: `CHAMPION.md §5`.
3. **The tasks are gradient-orthogonal — first-order, tested intrinsic.** cos(∇cat,∇reg) ≈ 0 on the
   shared trunk (pooled +0.0008 over 16 runs, 4 states × 4 seeds, n=3,797 epoch-fold points), and it
   persists in a fully-shared model where reg's gradient dominates → intrinsic to the task pair, not
   manufactured by G. This is the study's unifying mechanism: nothing for balancers to resolve;
   forcing sharing induces conflict that isn't there; exploit (protect + harvest), don't fight.
   Phrase it as a *first-order average* statement (Fifty'21 lookahead-affinity is the reviewer
   counter). (`WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`.)
4. **No modern MTL optimizer helps — a convergent-evidence negative.** The full `src/losses`
   registry (~19 arms), loss-scale normalization (falsified — starves the high-cardinality reg
   head), a static-weight fairness sweep (0.75 on the Pareto front), and the RLW litmus all fail to
   Pareto-beat tuned static weighting — the expected k=2 result (Kurin/Xin NeurIPS'22). Wording
   discipline: it is NOT "every method individually tuned" (see the evidence-strength banner in
   `docs/results/mtl_improvement/T4_audit_and_verdict.md`); gradient-surgery methods are additionally
   inapplicable to the dual-tower as wired (collapse to equal-weight — `CONCERNS §C27`).
5. **The +3 pp cat gain is architecture-dominated, not task transfer.** Decomposition (reg-weight-0
   ablation, 4-seed, integrity-fixed): cross-attn trunk alone +2.27 (FL) / +3.22 (AL) over the STL
   cat ceiling; genuine region→category transfer only **+0.93 (FL) / −0.67 (AL)**. Paper framing:
   "the joint cross-attn architecture is a better category encoder, with a small region transfer at
   scale" — NOT "region teaches category." (`docs/results/mtl_improvement/cat_transfer_and_T53.md`.)
6. **The rising-tide rule.** Every reg-INPUT lever (dense/overlap supervision, a better substrate,
   richer priors) lifts the STL ceiling as much as it lifts G — score every probe as **G − ceiling
   on the MATCHED metric**, never as raw lift. Two clean rising-tide nulls (R1 overlap, R2 HGI
   routing) plus the magnitude rule are why the reg-input axis is closed.

## 4. Why MTL reg matches but cannot beat STL (the earned answer)

- **Theory + literature say parity is the expected outcome here**: at cos≈0 the auxiliary
  contributes no first-order progress (Du'18); MTL gains concentrate in data-starved regimes
  (Bingel & Søgaard'17); k=2 tuned scalarization is unbeatable (Kurin/Xin'22); and the 7-class
  (~2.8-bit) category auxiliary is far below the 180–300+ class vocabularies behind every positive
  category-aux result in the next-POI literature. Frame the negative as the **weak-auxiliary
  regime**. Domain "MTL wins" dissolve on inspection (iMTL = conditional coupling, no STL ablation;
  GETNext = input-side category; MCARNN = 2018 low-capacity single-run). Citations + the
  pre-emption of MCARNN: `CODE_AUDIT_2026-06-12.md` Part 2.
- **The claim is earned, not assumed**: the X-series exercised every structurally-disabled MTL-only
  lever (real KD-on-G: null; β freed of weight decay: β→0 is gradient-driven; eval precision:
  clean) and the published numbers are pairing-safe. **One counterfactual remains open by
  construction** — whether *aligned-pairing training* could activate cross-modal mixing (the roll
  probe is circular against it) — inherited by `closing-data` as a **pre-freeze gate** (§8).
- **Mechanisms that could go beyond parity** (literature-backed, NOT tested here — future work):
  conditional coupling (cat output → reg head input), category-conditioned logit prior,
  semantic-ID/coarse-to-fine region-vocabulary factorization, category-transition input features,
  region→category consistency loss. Ranked with citations: `INDEX.html #T7-FW`.

## 5. Corrections & retractions registry (cite the RIGHT claim)

| superseded claim | corrected claim | where |
|---|---|---|
| G "beats BOTH STL ceilings" (+Δreg +0.26…+1.59) | reg **"matches"** (indist-vs-full artifact; matched Δ −0.09…−0.31) | B-A2 + R0; `CHAMPION.md §1` |
| "composite strictly dominated, ½ params" | dominated on the **joint** reading (+0.53 reg-only FL edge remains; G = base+4.9% params) | R0; `PAPER_UPDATE.md` |
| "log_T-KD tested on the dual-tower — adds nothing" | was a **dead codepath** (aux-gate bug, C28); the REAL test (X2) is null — cite X2, not the old arms | `X_SERIES_FINDINGS.md §X2` |
| Tier 4 "per-method-tuned + arch-wired" | **convergent-evidence negative** (defaults screen + targeted retunes + RLW + cos≈0 + literature) | `T4_audit_and_verdict.md` banner |
| FL region-transfer +1.08 | **+0.93** (manifest race had pointed 3 seeds at the wrong run — C28) | `cat_transfer_and_T53.md` |
| X1 "mixing genuinely dead, not a noise-pair artifact" | numbers **pairing-safe**; deployed model performs no per-sample mixing; the aligned-TRAINING counterfactual is untested (circular probe) → closing-data pre-freeze gate | `X_SERIES_FINDINGS.md §X1` banner |
| B9/v11 "static_weight cw=0.75" mechanism | the flag is **DEAD under `--alternating-optimizer-step`** — B9 trained 50/50 alternating (numbers stand; mechanism description corrected) | `CANONICAL_VERSIONS.md §v11` note |
| G′ (cat-private) FL cat 74.77 | **DO NOT CITE** — FL-only dead-end, craters small states, rescue screen closed | `CHAMPION.md §G′`; `CONCERNS §C26` |
| Pre-C25 Tier-2 "dual-tower loses / irreducibly architectural" + the composite headline (CH25) + regime finding (CH28) | all measured under the C25 confound — overturned/dissolved | `CONCERNS §C25`; INDEX Tier-2 banner |

## 6. Process lessons (why this study needed three audit passes)

1. **The biggest "architecture problem" was a loss-objective mismatch** found only by hostile
   re-reading of defaults (C25). Audit the objective↔metric pairing before the architecture.
2. **Dead codepaths produce confident nulls** (C28: the aux gate made KD/prior arms silent no-ops
   that *looked* like clean negative results). Assert the mechanism fired (aux non-None, α
   trajectory) — never trust a null whose lever you didn't see move.
3. **Rundir-capture races contaminate manifests** (C28: `ls -dt|head` under concurrency pointed 3
   "seeds" at a different architecture's run). PID-suffixed rundirs + per-run seed echo are
   mandatory; re-verify any multi-seed cell whose rows share a timestamp.
4. **Probes can be circular**: a model trained under a degenerate regime (noise pairing) is forced
   into the invariance an eval-time probe then "discovers." Ask what the probe has power against.
5. **Matched metric or it didn't happen** (B-A2: indist-vs-full inflated Δreg by ~0.6 pp; the X4
   fp16/fp32 check closed the precision side). Comparisons must match metric, seeds, folds, AND
   eval precision.
6. **Development seed ≠ reporting seeds** (seed 42 develops; {0,1,7,100} report) and **paper-grade
   claims need the multi-seed + the falsification attempt** (G′ was over-promoted for hours until
   the multi-state confirm cratered it — test-don't-assume cuts both ways).

## 7. What this study shipped in code (all on `mtl-improve`)

- **`--canon` version selector** (`src/configs/canon.py` + `scripts/train.py`): v16 (=G) is the MTL
  default; `--canon v11|v12|v15|none` for traceback; explicit flags override; guarded by
  `tests/test_configs/test_canon.py`. Contract: **pin `--canon` in every script**.
- **Per-task class-weight controls** (`use_class_weights_{reg,cat}`, CLI `--[no-]reg/cat-class-weights`)
  — the C25 fix; unweighted is the default.
- **Models/heads**: `mtlnet_crossattn_dualtower` (+ `_catpriv`, `_swiglu`, `mult`, `xstitch`) and
  `next_stan_flow_dualtower` (aux/gated/private_only fusion, freezable α).
- **Fixes**: aux-gate (`folds.py` — dualtower head now reaches KD/prior), CLI `KEY=False` bool
  inversion (`_coerce_cli_value` + `test_cli_param_coercion.py`), `_BACKWARD_ONLY_LOSSES` preflight,
  reg checkpoint monitor Acc@1→Acc@10, `geom_simple` checkpoint selector.
- **Env-gated probes** (defaults unchanged): `MTL_DISABLE_AMP`, `MTL_DISABLE_AMP_EVAL`,
  `MTL_ROLL_TASKB_EVAL`, `MTL_BETA_NO_WD`; gated `--loss-scale-norm` flag (falsified, default off).
- **Regression guards**: `tests/test_regression/test_mtl_param_partition.py` (dualtower family +
  PCGrad private-tower gradient coverage, 33 tests).

## 8. Open / inherited items (NOT part of this study anymore)

| item | owner | spec |
|---|---|---|
| **Pre-freeze gate: aligned-pairing training test** (could change the recipe) | `closing-data` Phase 0 | `X_SERIES_FINDINGS.md §X1` correction banner |
| **CA/TX majors** (v14 builds + G + ceilings at the two largest states; HSM-at-8.5k and conv_attn FL-only lever folded in) | `closing-data` | INDEX `#T6-1` card |
| **BRACIS paper restatement** (§0.1 + CH25/CH28 verbs + the orthogonality/limitations sections) | author (T6.2) | `PAPER_UPDATE.md` (read top-down; layers are dated) |
| Beyond-parity mechanisms (conditional coupling, cat-conditioned prior, semantic-ID factorization, consistency loss) + Fifty-style affinity hardening | future work | INDEX `#T7-FW`; `CODE_AUDIT_2026-06-12.md` Part 2 |

## 9. Reading map

- **This file** → the distilled outcome. Then, by need:
- `CHAMPION.md` — exact config, reproduce command, prerequisites, do-not-retry table.
- `docs/results/CANONICAL_VERSIONS.md` — v16 pin + `--canon` traceback to v11/v12/v15.
- `docs/results/mtl_improvement/` — all result artifacts (R0 bar JSON, X_SERIES_FINDINGS, T4 verdict,
  cat-transfer decomposition, T5.2 sweep, orthogonality intrinsic test).
- `WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md` — the mechanism narrative + figures (paper-facing).
- `PAPER_UPDATE.md` — what the paper must change (dated layers, newest first).
- `CODE_AUDIT_2026-06-12.md` — the deep audit (code findings + literature analysis).
- `INDEX.html` — every experiment card with results (Tiers 0–7); `log.md` — full chronology;
  `HANDOFF.md` — the historical "you are here" stack.
- `docs/CONCERNS.md §C25–C28` — the four concerns this study minted.
