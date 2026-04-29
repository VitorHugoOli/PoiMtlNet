# F50 T3 — Hyperparameter Brainstorm + Sweep Plan (2026-04-29)

**Trigger:** F50 T3 finding (`F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md`) revealed the FL gap is temporal not architectural. Per the user's request, an independent research agent produced a comprehensive prioritized list of hyperparameter / training-config changes to test the now-revised mechanism hypotheses.

**This doc captures the agent's full Tier-A/B/C list + the screening + validation plan.** Updates are recorded as `## Run log` at the bottom as runs land.

---

## Approach: FL-only screening (revised from user's GA proposal)

User proposed: GA screening → FL validation. Reasons we pivoted to **FL 1f×25ep screening → FL 5f×50ep validation**:

1. Georgia data was deleted in the disk cleanup (~30s to re-fetch but lost in-flight time).
2. **GA's 100-region reg head won't exhibit the same α-growth dynamics as FL's 4702-region head.** A scheduler change that helps reg at GA might not help at FL.
3. AL is closer to FL's reg-head scale (1109 regions) but still requires a fetch.
4. **FL 1f×25ep is ~7 min** — short enough to screen 10+ configs in ~70 min total. Eliminates state-generalisation risk.

For Tier-A entries (all CLI-only, no dev cost), we run direct **FL 5f×50ep** since they're cheap enough (~19 min each).

For Tier-B entries that require dev work, we'll smoke-test on FL 1f×25ep before committing to FL 5f×50ep.

---

## Tier A — CLI-only, single flag change. RUN FIRST.

| # | Name | Hypothesis | Expected | CLI delta vs H3-alt |
|---|---|---|---|---|
| **A1** | `onecycle50` ⭐ | Scheduler IS the cause: STL won under OneCycle (peak at ep 15 = where α grows). | **+3 to +5 pp likely** | replace `--scheduler constant` with `--scheduler onecycle --max-lr 3e-3` |
| **A2** | `cosine50` | Pure decay (no warmup) — isolates "decay from peak" vs "warmup ramp" of A1. | +1 to +3 pp | `--scheduler cosine --max-lr 3e-3` |
| **A3** | `alpha_init_2.0` ⭐ | α growth is the rate-limiting step. Skip it: initialise α at STL's converged value (~2.0). If MTL just needs the prior magnitude, ≈ STL. | **+3 to +6 pp** if mechanism is α-magnitude | append `--reg-head-param alpha_init=2.0` |
| **A4** | `epochs100_constant` | MTL reg-best is ep 5 — but maybe joint pipeline reaches its α-equivalent later under the constant schedule. | Probably tied (D8 says ep 5 is structural) | `--epochs 100` |
| **A5** | `onecycle_pct0.4_alpha2` ⭐ | Stack A1+A3: OneCycle pct_start=0.4 (peak ep 20 = STL window) + α already grown. | **+5 to +8 pp** if A1/A3 are partially additive | `--scheduler onecycle --max-lr 3e-3 --pct-start 0.4 --reg-head-param alpha_init=2.0` |
| **A6** | `cw_low_onecycle` | cw=0.25 gave +0.94 under constant. Re-run cw=0.25 under OneCycle — combined "less cat dominance" + "STL scheduler". | +2 to +4 pp | `--category-weight 0.25 --scheduler onecycle --max-lr 3e-3` |
| **A7** | `seed_replication` | Sanity check: rerun H3-alt with seeds {0,1,2,3,4}. Noise-floor probe. If σ_seed ≥ 1.5 pp, change strategy. | Noise floor | rerun H3-alt 5× with `--seed {0..4}` |

---

## Tier B — Small dev, then run

| # | Name | Hypothesis | Expected | Dev | Implementation |
|---|---|---|---|---|---|
| **B1** | `delayed_min_selector` ⭐ | Per-task best (`mtl_cv.py:632`) commits at ep 5 greedily. Add `min_epoch_for_best=15` window. | +0 to +6 pp | ~30 min | `mtl_cv.py:632` gate; `--min-best-epoch 15` flag |
| **B2** | `warmup_decay_reghead_lr` | D6 with constant 3e-2 destabilizes; warmup-decay variant. | +3 to +6 pp | ~1h | LambdaLR on reg_head group only |
| **B3** | `two_phase_50_50` ⭐ | Train reg STL-only 50ep (full α growth), then 50 ep MTL fine-tune. **Decisive**. | +5 to +8 pp | ~2h | STL phase + MTL warm-start |
| **B4** | `freeze_alpha_warmup_then_unfreeze` | Cat stabilizes for 5 ep, then unfreeze α (avoid co-adaptation). | +2 to +4 pp | ~1h | `--alpha-frozen-until-epoch 5` |
| **B5** | `cycle_cat_not_reg` | Force cycling cat dataloader regardless of length (currently cycles whichever is shorter). | +0 to +2 pp | ~30 min | `progress.py:zip_longest_cycle` flag |
| **B6** | `head_only_lr_ramp` | 4× linear ramp on reg_head_lr only. | +2 to +4 pp | ~30 min | LambdaLR on reg_head |
| **B7** | `monitor_reg_only` | Switch model_task best monitor to reg-only F1. | +0 to +2 pp | trivial | `--monitor val_f1_next` |
| **B8** | `gradient_accum_4` | Larger effective batch for stable α gradient. | +1 to +3 pp | CLI | `--gradient-accumulation-steps 4` |
| **B9** | `weight_decay_zero_alpha` | AdamW WD=0.05 may suppress single-scalar α growth. Exempt α. | +1 to +3 pp | ~30 min | helpers.py separate group |
| **B10** | `bs1024_double_steps` | Smaller batches → 2× α update steps per epoch. | +1 to +3 pp | CLI | `--batch-size 1024` |

---

## Tier C — Speculative

| # | Name | Hypothesis | Expected | Dev |
|---|---|---|---|---|
| **C1** | `alpha_init_from_stl_ckpt` | Per-fold α loaded from F37 STL ckpt. Decisive direct test. | +5 to +8 pp | ~3h |
| **C2** | `pretrained_reg_encoder` | Variant of B3 with selective freeze. | +4 to +7 pp | ~3h |
| **C3** | `alternating_epoch_phase` | Per-EPOCH alternating: 10 cat / 10 reg / 10 joint. | +2 to +5 pp | ~4h |
| **C4** | `scheduled_cat_weight` | cw 1.0 → 0.0 ramp over ep 0-25. | +2 to +4 pp | ~1h (existing `scheduled_static`) |
| **C5** | `stl_alpha_distill` | KD loss on α-decoded logits using STL teacher. | +3 to +6 pp | ~6h |
| **C6** | `seed_ensemble_average` | 5 seeds × 5 folds. Probably noise reduction only. | +0.5 to +1 pp | CLI |
| **C7** | `reg_only_pretrain_5ep` | B3 compromise: just 5 ep STL pretrain. | +2 to +4 pp | ~2h |
| **C8** | `freeze_shared_in_phase2` | After warm-start, freeze cross-attn backbone. | +2 to +5 pp | ~1h |

---

## Top-5 priority (run order)

1. **A1 `onecycle50`** — direct test of the load-bearing T3 hypothesis. CLI-only.
2. **A3 `alpha_init_2.0`** — orthogonal direct test of α-magnitude hypothesis. CLI-only.
3. **A5 `onecycle_pct0.4_alpha2`** — stacks A1+A3. If both are independently positive, A5 should give the largest single-shot gain.
4. **B1 `delayed_min_selector`** — cheap selector test. If reg has a real later peak that the greedy selector misses, this reveals it.
5. **B3 `two_phase_50_50`** — decisive mechanism test. If a STL-pretrained reg path doesn't recover the gap when wrapped in MTL fine-tuning, the gap is fundamentally architectural.

---

## Decision rule after Tier A

- **A1 alone hits +3 pp** → ship `--scheduler onecycle` as new champion. Mechanism = scheduler.
- **A3 alone hits +3 pp** → α-magnitude is the issue; minimal-change paper claim "we propose alpha_init=2.0".
- **A5 hits +5 pp** → both contribute; combined recipe is the new champion.
- **None of A1-A5 hits +3 pp** → run B1 (selector) + B3 (two-phase). If those also fail, the gap is genuinely structural; lock H3-alt + ship paper with the temporal-dynamics framing.

---

## Run log

| date | config | run dir | reg top10 | reg-best ep | Δ vs H3-alt | verdict |
|---|---|---|---|---|---|---|
| 2026-04-29 | H3-alt CUDA REF | `_0153` | 73.61 ± 0.83 | {4-6} | — | reference |
| (running...) | A1 `onecycle50` | TBD | TBD | TBD | TBD | TBD |
