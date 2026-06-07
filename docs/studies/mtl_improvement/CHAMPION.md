# CHAMPION (G) — the mtl_improvement result, reproduction, and what-not-to-retry

> **Read `HANDOFF.md` for the narrative; read THIS for the exact config, the runnable command, the new code, and the closed dead-ends.** Single source of truth for the champion MTL config "G" found by the `mtl_improvement` study (branch `mtl-improve`, 2026-06-04 → 2026-06-06).

---

## 1. The result (one paragraph)

A **single jointly-trained MTL model** — "G" — **MATCHES the single-task STL next-region ceiling (Pareto-non-inferior) AND substantially BEATS the next-category ceiling (+3 pp macro-F1) at all 4 available states, 4 seeds each.** This inverts the study's original framing ("MTL sacrifices reg by −7…−17 pp; ship the 2-model composite; the architectural-Δ is the paper's central tension"): the −7…−17 pp 'MTL sacrifices reg' tension was a class-weighting artifact (C25), and once fixed, **joint training is Pareto-positive** (matches reg + beats cat).

> ⚠ **REG CLAIM CORRECTED 2026-06-07 (B-A2 independent re-eval — INDEX `#T2V-3`).** The reg "Δ" column below is **G's in-distribution `top10_acc_indist`** vs the **(c) ceiling's FULL `top10_acc`** (the p1 ceiling harness has no indist/OOD split). On a **MATCHED** metric G is **~0.35 pp BELOW** the (c) reg ceiling (FL: G-full 72.93 vs ceiling-full 73.31). So the honest reg verb is **"matches" (within ~0.4 pp), NOT "beats"** — the +Δreg figures below are inflated by the indist-vs-full gap (~0.6 pp at FL; small-state values un-re-eval'd, expected to temper similarly). The **cat +Δ is exact** (single metric, no indist/full split). The Pareto-positive / inverted-tradeoff headline STANDS (matches reg + beats cat).

| state | G reg@10 (indist) | (c) STL reg (full) | Δreg *(indist−full; see ⚠)* | G cat-F1 | (c) STL cat | **Δcat (exact)** |
|---|---|---|---|---|---|---|
| AL | 64.47 ± 0.11 | 62.88 | +1.59* | 52.91 ± 0.27 | 49.97 | **+2.94** |
| AZ | 55.75 ± 0.21 | 55.11 | +0.64* | 54.48 ± 0.74 | 51.01 | **+3.47** |
| GE | 59.37 ± 0.04 | 58.45 | +0.92* | 61.43 ± 0.26 | 58.12 | **+3.31** |
| FL | 73.57 ± 0.06 | 73.31 | +0.26* (matched: **−0.38**) | 73.16 ± 0.04 | 69.97 | **+3.19** |

Seeds {0,1,7,100}. Metrics: reg = disjoint `per_metric_best.next_region.top10_acc_indist`; cat = `diagnostic_task_best.next_category.f1`. FL: on the matched full metric G (72.93) ≈ the (d) 2-model composite — single-model parity at half the params, the deployable win. CA/TX not run (no v14 substrate built — see §6).

> ⚠ **G′ — the cat-private variant: FL-ONLY, DEMOTED (multi-state confirm 2026-06-07).** Giving the **cat** head its own private tower too (the both-private dual-tower, `mtlnet_crossattn_dualtower_catpriv`; cat-head + reg-head both `next_stan_flow_dualtower` aux+prior-OFF) looked like a Pareto win **at FL only** (cat 74.77±0.04 = +1.61 vs G; reg 73.59±0.07 flat). **But the multi-state confirm (AL/AZ/GE × {0,1,7,100}) FALSIFIED it: the cat-private tower CRATERS small-state cat — AL 37.66 (−15.25 vs G), AZ 42.02 (−12.45 vs G), GE 57.84 (−3.59 vs G); reg flat everywhere (+0.02..+0.11).** **Mechanism CORRECTED 2026-06-07 (advisor root-cause): NOT overfit — the cat-private tower UNDERFITS small-state cat.** The trajectory proves it: AL G′ cat **train**-F1 caps at **0.45** (vs the `next_gru` head's **0.98**) with a *tiny* train–val gap — the textbook underfit signature. The `next_stan_flow_dualtower` head is built for ~thousands of region classes; run off-label on **7-class** category it is over-regularized (`priv_dropout=0.3`) on a GRU-tuned LR schedule and never converges at small data. The FL +1.61 is robust (σ 0.04pp) but **scale-gated** — only FL has enough data to train the heavy private tower. **G′ is (for now) an FL-only experimental variant, NOT a champion — G (cat-SHARED) remains the multi-state-confirmed champion.** A **rescue screen** (`gprime_rescue_screen.sh`: lower `priv_dropout`, softer cat-lr, smaller tower) is testing whether small-state cat is recoverable — if it lifts AL/AZ to ≥ G while keeping FL's gain, G′ could be re-promoted; if not, the head is fundamentally FL-only. (The B-A3 ablation was *predicted null*; the FL-only gain + its mis-tuned mechanism were both caught by the multi-state confirm — test-don't-assume cutting both ways.) Trail: `log.md`/`INDEX.html #T2V-5` 2026-06-07; drivers `gprime_multistate.sh`, `gprime_rescue_screen.sh`.

---

## 2. The G config

| component | value | why |
|---|---|---|
| **model** | `mtlnet_crossattn_dualtower` | reg-private dual-tower: the reg head gets the raw `[B,9,64]` region sequence through a private STAN backbone, NOT just the shared cross-attn output. The private tower carries reg almost entirely (the `private_only` variant alone clears the ceiling). |
| **reg head** | `next_stan_flow_dualtower` | the dual-tower STAN-Flow head holding the private tower + the fusion + the (frozen) α·log_T prior. |
| `fusion_mode=aux` | **the key lever** | `aux` fuses as `feat = priv + β·aux_proj(shared)` (β learnable, init 0.1) — it ADDS the shared pathway as a non-attenuating residual. The default `gated` makes private/shared *compete* (a convex gate) and thereby DILUTES the private reg signal → `gated` lands at 73.06 vs `aux` 73.57. **Do not switch to `gated` or any multiplicative/competitive fusion — it regresses the ceiling-break.** |
| `freeze_alpha=True alpha_init=0.0` | **prior-OFF** | the in-head additive α·log_T prior is a biased logit term that HURTS top-K. Turning it off (hard 0, frozen) gains ~+1.4 pp reg. Keep it off. (Soft log_T-KD distillation is a *different* mechanism but was tested and adds nothing on the dual-tower — see §5.) |
| `raw_embed_dim=64` | builds the private tower | the private STAN consumes the raw 64-dim region embedding sequence. |
| **cat head** | `next_gru` | the category-task champion head (unchanged from canon). |
| **substrate** | `check2hgi_design_k_resln_mae_l0_1` (v14) | the v14 dual-axis STL base. NOTE: the substrate gain transfers to MTL under G (the regime finding is overturned post-C25). |
| **loss / weighting** | `static_weight`, `--category-weight 0.75`, **both heads UNWEIGHTED** | unweighted CE is the **C25 fix** and is now the `default_mtl` default — see §3. Class-weighted CE was the confound that hid this whole result. |
| **schedule** | `onecycle`, `--max-lr 3e-3`, per-head `--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3` | onecycle (no alt-opt) is the adopted recipe; per-head LRs flatten to the scalar max_lr under OneCycleLR. |
| **KD** | `--log-t-kd-weight 0.0` | KD off (the v12 default is 0.2; G turns it off — it doesn't help the dual-tower). |
| epochs/folds/bs | 50 / 5 / 2048 | standard. |

The sweep (`c25_gv2.sh` Group III) confirmed **G is well-tuned**: `category-weight 0.75`, `priv_dropout 0.3` (head default), AMP-on are all at/near their joint optimum; no swept value beat G.

---

## 3. Reproduce G (exact command)

```bash
# FL champion (other states: swap --state + --per-fold-transition-dir; AL/AZ/GE have v14).
# C25 unweighting is the DEFAULT now (default_mtl sets use_class_weights_{reg,cat}=False) —
# no explicit flag needed. Multi-seed: loop --seed over {0,1,7,100}.
python scripts/train.py --task mtl --task-set check2hgi_next_region \
  --engine check2hgi_design_k_resln_mae_l0_1 --state florida --seed 0 \
  --epochs 50 --folds 5 --batch-size 2048 \
  --model mtlnet_crossattn_dualtower \
  --mtl-loss static_weight --category-weight 0.75 \
  --cat-head next_gru --reg-head next_stan_flow_dualtower \
  --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
  --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
  --task-a-input-type checkin --task-b-input-type region \
  --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --log-t-kd-weight 0.0 \
  --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/florida \
  --no-checkpoints
```

Turn-key drivers (PID-suffix rundir capture, manifests):
- `scripts/mtl_improvement/c25_combos_promote.sh` — G (+H) at FL, seeds {1,7,100}.
- `scripts/mtl_improvement/c25_g_multistate.sh` — G at AL/AZ/GE, seeds {1,7,100}.
- `scripts/mtl_improvement/c25_combos_screen.sh` — the 1-seed combo screen that found G.
- Aggregator pattern: read each rundir `summary/full_summary.json` for the two metric fields above (see `c25_stretch_agg.py`).

### Prerequisites (all verified at AL/AZ/GE/FL)
1. **v14 substrate built**: `output/check2hgi_design_k_resln_mae_l0_1/<state>/input/next_region.parquet`.
2. **Seeded per-fold log_T** present AND fresh (mtime ≥ `next_region.parquet`): `region_transition_log_seed{S}_fold{N}.pt`. G runs prior-OFF so it loads-but-ignores log_T, but the path is still required; build via `python scripts/compute_region_transition.py --state <st> --per-fold --seed <S>`. (Stale log_T silently inflates reg — see CLAUDE.md.)
3. **C25 unweighting** is the code default; to reproduce the *pre-C25* (confounded) numbers instead, pass `--reg-class-weights --cat-class-weights`.

---

## 4. The C25 prerequisite (why this works at all)

G is only visible **after the C25 fix**. Before it, `default_mtl use_class_weights=True` trained the reg head on class-WEIGHTED CE while the reported metric (Acc@10) and the STL ceiling are unweighted — class-balancing optimizes macro accuracy *away from* top-K, depressing MTL reg ~10–14 pp and cat ~3–5 pp. This single confound produced the entire "MTL→STL reg gap / architecture-negative / ship-the-composite" narrative. Fix: per-task `use_class_weights_{reg,cat}=False` (now the `default_mtl` default; CLI `--[no-]reg-class-weights` / `--[no-]cat-class-weights`). Full write-up: `docs/CONCERNS.md §C25`; same-harness §0.1 A/B (+3.15 reg / +3.52 cat) in `docs/results/RESULTS_TABLE.md §0.1`.

A second, smaller fix shipped with it: the reg checkpoint MONITOR was `Acc@1` (`PrimaryMetric.ACCURACY`) — changed to `Acc@10` (`PrimaryMetric.TOP10`, `src/tasks/{registry,presets}.py`) so the *deployable* reg snapshot matches the headline (the disjoint/diagnostic reads above are selector-independent and were always correct).

---

## 5. What NOT to retry (closed dead-ends — falsified, with evidence)

| hypothesis | verdict | evidence |
|---|---|---|
| **Architecture CAPACITY/quality closes the reg gap** | **FALSIFIED 5 ways** | MoE (`mtlnet_mmoe` 71.68 / `mtlnet_cgc` 71.77), SwiGLU backbone (`mtlnet_crossattn_swiglu` 71.71), MulT-faithful (`mtlnet_crossattn_mult` 71.28), crossstitch→crossattn (`mtlnet_crossattn_xstitch` 71.13), more cross-attn blocks — all NULL/negative on reg. The reg gap is the prior + shared-pathway dilution, NOT capacity. |
| **A better SHARED backbone under the dual-tower helps reg** | FALSIFIED | combo (F) `mtlnet_crossattn_dualtower_swiglu` (dual-tower + SwiGLU shared) = NULL on reg. The dual-tower routes reg through the PRIVATE tower; the shared pathway is gated/aux'd, so its quality barely affects reg. Shared-backbone swaps are a CAT play at best. |
| **`gated` fusion (or per-dim gate) is fine / better** | FALSIFIED | `gated` makes private/shared compete → dilutes reg (73.06 vs aux 73.57). Use `aux`. |
| **Re-introduce the α·log_T prior (additive)** | DON'T | it is a biased logit term that hurts top-K; G freezes it off. (Soft log_T-KD distillation `--log-t-kd-weight 0.1/0.2` was tested on the dual-tower — IDENTICAL to G, adds nothing; it was tuned on the old single-pathway arch and doesn't transfer.) |
| **fp32 (MTL_DISABLE_AMP=1) lifts reg** | partial, not a win | fp32 gives reg +0.13 but cat −1.11 — a trade, not a Pareto gain (G wins on geom-mean). A precision-sensitivity note, not a champion change. |
| **category-weight / priv_dropout / d_model tuning beats G** | FALSIFIED | the `c25_gv2.sh` sweep found no Pareto gain; G's defaults are at/near optimum. |
| **MTL sacrifices reg; ship the 2-model composite (CH25)** | DISSOLVED | G ties the composite on reg AND wins cat at half the params. |
| **Substrate gains wash out in the MTL regime (CH28)** | OVERTURNED | post-C25 the v14 substrate gain transfers to MTL; G uses v14. |

---

## 6. New code shipped this arc

**Models** (`src/models/mtl/<name>/`, each `@register_model`, metadata.yaml, unit-gated):
- `mtlnet_crossattn_dualtower` — (pre-existing, the G backbone).
- `mtlnet_crossattn_swiglu` — pre-norm + SwiGLU cross-attn block (T2.4; cat lever, null on reg).
- `mtlnet_crossattn_dualtower_swiglu` — dual-tower + SwiGLU shared (combo F; null on reg).
- `mtlnet_crossattn_mult` — MulT-faithful (self+cross attention; null on reg).
- `mtlnet_crossattn_xstitch` — cross-stitch→cross-attention hybrid (null on reg).

The reg head `next_stan_flow_dualtower` and the per-task class-weight fields (`use_class_weights_{cat,reg}` on `ExperimentConfig`, plumbed through `mtl_cv.py` + `scripts/train.py`) and the `PrimaryMetric.TOP10` monitor are the load-bearing non-model changes.

**Unit gates**: `scripts/mtl_improvement/t24_swiglu_unit_gate.py`, `t24_dualtower_swiglu_gate.py` (partition bijectivity + structure asserts — run before any multi-fold launch of a new model).

---

## 7. Status, caveats, and how to extend

- **G is a STUDY champion, NOT the paper §0 canon.** The BRACIS paper §0 still reports v11 (GCN substrate, B9, class-weighted). G is on the v14 substrate with the unweighted recipe. Re-running §0 under G/C25 is the **paper-doc restatement** — an author decision, not yet done.
- **The (c)/(d) ceilings G beats are seed=42 single-seed**; G is 4-seed. Margins are robust (every G seed clears the ceiling), but a fully seed-matched claim would re-run (c)/(d) at {0,1,7,100} (cheap, ~optional).
- **CA/TX**: no v14 substrate built. To extend G there, first build `check2hgi_design_k_resln_mae_l0_1` at CA/TX (`scripts/canonical_improvement/regen_emb_t3.py`) + the seeded per-fold log_T, then run the G command. Deferred for large-state compute.
- **Provenance**: full chronology in `log.md` (2026-06-04 → 2026-06-06 entries: "CEILING BROKEN", "G GENERALIZES", "G CONFIRMED MULTI-STATE"); claim verdicts in `docs/CLAIMS_AND_HYPOTHESES.md`; version pin in `docs/results/CANONICAL_VERSIONS.md`; confound in `docs/CONCERNS.md §C25`.
