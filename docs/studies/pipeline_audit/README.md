# pipeline_audit — full input-builder + MTL-train pipeline audit (2026-07-01)

**Status: CLOSED 2026-07-01** (single-day audit, branch `audit/pipeline-correctness-perf`).
**Scope** (user request): (1) correctness of the input-builder + training flows — do they do
what they claim, with no data leaks; (2) training performance on the A40 — GPU levers that do
not cost quality (tested at AL 1 fold; 5-fold gate for adopted changes); (3) the cross-task
input-pairing question (should the same user's cat + reg sequences be fed together?); (4)
reconcile with the documented improvement backlog (`train_perf_multifold`, `future_works`).

**Method**: 10-reader map workflow over the whole pipeline → 14-verifier adversarial
workflow (every finding refute-first, incl. 2 fresh-eyes leak hunters) → endorsed fixes
landed behind parity gates (eager byte-identity + compiled 5-fold quality gate).

---

## 1 · Headline verdicts

1. **The champion v17 path is leak-clean.** Two independent fresh-eyes hunts (data-flow lens,
   metric/selection lens) found **no unreported val→train leak** and **no metric
   mis-computation** on the champion path. Specifically verified clean: one shared
   user-grouped StratifiedGroupKFold for both tasks (no cross-task split contamination),
   train-only class weights / OOD label sets / selectors, no scaler fit anywhere, target
   embedding never enters the input window, log_T provably inert + skipped, S1/S2 streaming
   metrics mathematically identical to the full-logit paths (no per-chunk averaging bias),
   CPU-cloned best-state snapshots. The known select-on-val optimism (no held-out test split)
   and the substrate's transductive training remain the two *documented* protocol properties.
2. **The pairing question is answered** — see §2. Short: the suspicion is mechanically
   correct (train batches are randomly paired across tasks; cross-attention mixes row i with
   row i), but the paired-input fix already exists (`--aligned-pairing`, G0.1) and measured
   **null at FL / harmful at AL (cat −4.77)**. Misalignment does **not** explain the
   orthogonal transfer; deployment-style (aligned) input is exactly what validation measures.
3. **One real latent training bug found + fixed**: per-batch `optimizer.zero_grad` silently
   broke gradient accumulation for any `ga>1` run (champion at ga=1 unaffected — provably
   byte-identical, eager-parity verified). Plus 10 confirmed guard/latent-path defects fixed
   and 11 doc contradictions corrected (§3).
4. **Perf**: the pipeline is already near-optimal for the A40 after `train_perf_multifold`
   (bs8192, S1/S2, P1/P3/P6, compile). The one adopted new lever is **P4
   `MTL_NO_TRAIN_DIAGNOSTICS=1`** — ~**9% wall** at AL warm-cache, byte-identical in eager
   (6/6 artifacts), bit-equal compiled in-session (§4). Everything else measured is ≤1%.

## 2 · The cross-task pairing question (user question, answered)

**Question**: for a sequence that predicts next-region, should the equivalent same-user
sequence predicting next-category be passed in the same step? Are we doing that today?

**What the pipeline actually does** (all verified first-hand, file:line in §5):

| Level | Behavior | Verdict |
|---|---|---|
| Rows | `next.parquet` row i == `next_region.parquet` row i == same (user, window). Enforced at build (count + per-row userid) and at load; verified empirically at AL (12,709 rows, identical userid arrays). | ALIGNED |
| Split | ONE `StratifiedGroupKFold(groups=userid, y=next_category)` reused verbatim by BOTH tasks → identical train/val indices. A window can never be train for one task and val for the other. | ALIGNED (leak-clean) |
| Train batches | The two task train loaders shuffle **independently** (global torch RNG, no shared generator) → cat batch k and reg batch k are **different random windows**. | **RANDOMLY PAIRED** |
| Val batches | Both val loaders are unshuffled over the same `val_idx` → row i pairs with itself. | ALIGNED |
| Model | `mtlnet_crossattn_dualtower` mixes activations **per row**: `MHA(Q=cat_i, K/V=reg_i)` bidirectionally → under independent shuffles the cross-read content is an unrelated user at train time. | PAIRING MATTERS mechanically |

So the suspicion is **factually right**: training cross-attends unrelated users while
validation (= the deployment scenario, same user both inputs) is aligned. **But** the
proposed fix already exists and was tested (pre_freeze_gates G0.1, `--aligned-pairing`:
one joint loader, shared permutation, seeded `seed+fold`): **FL cat +0.17 / reg ±0.00
(null); AL cat −4.77 / reg −0.84 (hurts)** — random pairing acts as a beneficial
augmentation for the shared encoder at small states. Consistent with the X1 roll probe
(`MTL_ROLL_TASKB_EVAL=1`: rolling the reg stream by 1 at eval moves FL cat-F1 by −0.004 —
the trained model **ignores** per-sample cross-stream content). The champion's reg head is
additionally dominated by its **private tower** (`priv + β·aux_proj(shared)`, β init 0.1 and
driven toward 0 by gradient), so the misalignment-tainted pathway is nearly severed.

**Conclusion**: the orthogonal transfer is real but is a *regime* property, not a pairing
artifact — forcing pairing does not unlock cross-task transfer. Reported numbers ARE
representative of the customer scenario (validation is aligned).

> **RESOLVED 2026-07-02 — [`PAIRING_BATTERY.md`](PAIRING_BATTERY.md)** (user follow-up: "why
> does aligned get WORSE at AL? this is weird"): a 5-arm × 4-seed decomposition battery with a
> new exactly-matched **deranged control** proved the aligned deficit (−3.03 cat / −0.60 reg
> under v17/overlap, 4/4 seeds) is **100% pairing semantics** — cross-reading your OWN
> window is an overfit shortcut, cross-reading a random other window is beneficial noise;
> the per-step-diversity and machinery hypotheses are REFUTED (derange ≡ base). The
> cond_coupling × aligned cell was tested: conditioning is only usable when semantically
> paired (reg +0.47 over aligned, 4/4 seeds) but never beats the champion — the R-CC closure
> survives de-confounding. Champion random pairing validated as the correct default;
> aligned pairing REFUTED at binding grade at AL. The three follow-ups below are updated
> accordingly in the memo:
- **Binding G0.1** (frozen base, seeds {0,1,7,100}) — pre-registered, still pending; the
  `--aligned-pairing` CLI crash that forced the advisory onto `lane1_run.sh` is **fixed** here.
- **cond_coupling × aligned-pairing** — the ONE mechanism that provably requires pairing
  (per-sample cat→reg conditioning). Every historical R-CC run predates `--aligned-pairing`
  → the "coupling is dead" verdict is mechanistically confounded; `b4_cascade.py` also pins
  `cond_coupling=posterior` with cross-attn severed and no alignment (its only cat→reg edge
  trains on garbage). A guard now WARNs (hard-fails under `MTL_STRICT=1`).
- **Batch-level pairing middle ground** (group batches by user/region-neighborhood, not row
  alignment) — unexplored; keeps the augmentation effect while making K/V content relevant.

## 3 · Correctness findings (14 adversarial verdicts: 12 CONFIRMED, 1 PARTIAL, 1 REFUTED=clean)

**Fixed on this branch** (all default-preserving; eager parity + 5-fold gate in §6):

| # | Finding (severity) | Fix |
|---|---|---|
| V1 | Per-batch `zero_grad` (`mtl_cv.py`) broke ga>1: each step applied only the LAST micro-batch ×1/ga; the step-count test couldn't catch it (**high**, latent — champion ga=1 unaffected) | Removed (byte-identical at ga=1, all loss paths audited); `ExperimentConfig` field default 2→1; new gradient-CONTENT test pins the closed form |
| — | `--aligned-pairing` crashed (`ExperimentConfig` lacked the field; FoldCreator wiring silently read False) — known bug since 2026-06-19 | Field added (default False) + regression test; smoke-verified end-to-end |
| — | `--only-fold/--only-folds` unusable under ANY `--canon` (bundle injects `--folds 5` → mutual-exclusion trip) | `--folds 5` tolerated (≡ the forced canonical 5-split) |
| V4 | Canon guards (dev-seed-42 + wrong-substrate) silently skipped when `--task` omitted — injection resolves task→mtl, guard didn't (**medium**) | Guard now reads the parse-time `args._canon_active` |
| V3 | `cond_coupling≠none` trains on garbage conditioning under default shuffles; no guard (**medium**; confounds R-CC / b4_cascade) | WARN after task-set resolution; `MTL_STRICT=1` hard-fails |
| V5 | `--only-folds`/`--folds N` eagerly materialized ALL selected folds (defeats `_LazyFoldMapping`; ≥3 folds at TX overlap → host OOM past the RAM-guard model) (**medium**) | `_RestrictedFoldMapping` lazy id-preserving subset view (3 call sites) |
| V6 | Auto-fp32 (large-C) was train-only → bare CA/TX runs scored val under fp16 (rank-tie optimism; fp32-trained logits can saturate fp16). Also `p3_board.sh` set NO precision env despite the fp32 board invariant (**medium**) | Auto-fp32 now `setdefault`s `MTL_DISABLE_AMP_EVAL=1` (explicit envs win); `p3_board.sh` exports `MTL_DISABLE_AMP=1` (overridable) |
| V7 | `next_region.parquet` userid guard silently degraded to row-count-only when the column was missing; dk_ovl builder asserted count only (**medium**) | Hard-fail on missing userid (18/18 on-disk artifacts have it); builder asserts per-row userid equality |
| V2 | Task-b region tensor had NO row-alignment guard vs `sequences_next.parquet` (stale seq file with equal N would silently mis-pair every window); concat branch also hardcoded CHECK2HGI windowing (**medium**) | `expect_userids` threaded from the fold creator into both builders (per-row guard); concat branch now threads `seq_engine` |
| V8 | Stale-log_T mtime guard silently skipped when the transition dir lacks `input/next_region.parquet` — for an ACTIVE prior that is the +8..+12pp leak class (**medium**; champion inert-skip unreachable) | WARN (hard-fail under `MTL_STRICT=1`) when prior active and reference parquet missing |
| V9 | OneCycle budget used `max(len)` under `min_size_truncate` (LR never fully anneals; opt-in strategy only); val-loader cycling double-counts the shorter task's samples in SCORED metrics (legacy pairs only — champion immune: equal-length loaders) (**medium**) | Strategy-aware `steps_per_epoch` (max branch verbatim); loud WARN on val-length mismatch in both eval paths |
| V10 | `validation_best_model` ran TWO identical full val passes (same state both slots). PARTIAL: real but small (~5 s/fold TX, not minutes) | `is`-guard single-pass fast path (byte-identical; two-pass kept for future distinct-state callers) |
| V13 | Freeze flags (`freeze_cat_after_epoch`, `reg_freeze_at_epoch`, stream freezes) set `.eval()` once but `model.train()` re-enabled dropout in the frozen stream every later epoch — "frozen-at-init" probes (W6, F49 λ0) were dropout-stochastic (**medium**, non-champion) | Re-assert `.eval()` after `model.train()` (gated on the flags); W6 doc annotated (directional conclusions stand; n=20 extensions must not mix regimes) |
| V14 | 11 doc contradictions: DEFAULT_CANON v16-vs-v17 (flows/SYSTEM_REFERENCE), garbled v17 invocation (`--canon v17` + `--canon none` in ONE command; no doc had the working v17-on-dk_ovl command), `MTL_STAN_LEGACY_MASK` "bit-exact repro" claim (Phase-5c disproved), CLAUDE.md `MTL_SKIP_INERT_LOGT` "default OFF" + "ga: 2 steps" staleness, `overlapping_windows.md` "NOT adopted" (adopted 2026-06-19/21), CLOSURE.md n_regions stale-open, train.py help/docstrings | All corrected (exact texts from the verifier; the working v17-on-dk_ovl command now in CANONICAL_VERSIONS §v17) |

**Confirmed, documented-only** (no code change warranted):
- **V11 (leak hunt) = REFUTED**: no new leak. Latent hazards recorded: the frozen-fold-cache
  route is seed-blind + signature-blind for `mtl_check2hgi` (`_signatures_match` vacuous) and
  `rebuild_dataloaders` drops the aux channel — currently unreachable (no such cache can be
  produced); harden before anyone extends `freeze_folds.py`.
- **V12 (metric hunt) = clean**: S1/S2/rank/selector/scorers verified correct; one comment
  overclaiming k=10 topk device-independence softened.
- fp16-eval tie-optimism at small states remains the **documented canonical** choice
  (measured Δ−0.005pp at FL); A1 (true-fp32 board-wide) is the standing QUALITY_IMPROVEMENTS
  #1 ADOPT lever — unchanged by this audit.
- `min_best_epoch=0` in v16/v17 (B9 pinned 5): empirically NOT biting (AL fold-0 joint epoch
  31, diag 26/32) — monitor; do not edit the frozen bundle.
- Legacy-`TaskType.MTL` path defects (independent category-split fallback, ambiguous-POI
  policy, eval cycling) are real but the path is unreachable from the current CLI without
  `--canon none` + no task-set; `pipelines/train/mtl.pipe.py` is broken-loud under
  DEFAULT_CANON=v17 (engine guard exit 2) — flagged for cleanup, not fixed here.
- `nash_mtl` CLI ga-blocklist is conservative/stale (NashMTL has a real `get_weighted_loss`);
  harmless direction — left as-is.

## 4 · Performance audit

**Baseline (AL v17 fold-0, fp32 board protocol, `--profile`)**: wall 162 s (≈46 s cold
compile; ~125 s warm), 3.09 batch/s, peak 12.4 GB, GPU util p50 98% / mean 73.6%, graph
breaks 2 (post-P1 state). Sections: forward 32 s, eval 19.6 s, backward 3 s, train-metric
1.3 s, data ≈0. Quality: cat 64.40 / reg 71.66 (diag-best; sane vs board v17 AL n=20 64.54/69.80).

**Adopted: P4 `MTL_NO_TRAIN_DIAGNOSTICS=1`** (proposed in `train_perf_multifold` PLAN §Tier-2,
never implemented). Gates the batch-0 grad-cosine diagnostic (2 extra full backwards/epoch
with `retain_graph` + 3 host syncs) and — for `static_weight` runs only — leaves inductor
`donated_buffer` at its default under `--compile` (no retain_graph users remain).
- **Timing (AL fold-0, warm cache)**: baseline 137 s → P4 125 s (**−9%**).
- **Quality**: eager 8-ep parity — **6/6 artifacts byte-identical**; compiled 50-ep —
  bit-equal to baseline within the same inductor-cache session (cat 64.4029 / reg 71.6636,
  identical across warm-base / P4-cold / P4-warm).
- Default **OFF** (bare runs byte-identical, repo convention). Recommended ON for drivers
  that don't consume `grad_cosine_shared` (n=20 sweeps, the A40.md H2/H3 cells).
- **Default FLIPPED same day (user decision)**: diagnostics now ride with the profiler —
  bare runs skip them (`MTL_TRAIN_DIAGNOSTICS` auto-set to `1` on `--profile`/`MTL_PROFILE=1`,
  else `0`; explicit env wins; legacy `MTL_NO_TRAIN_DIAGNOSTICS=1` forces off). Bare runs get
  the ~9% for free; profiled runs keep the diagnostic (opt out with
  `MTL_TRAIN_DIAGNOSTICS=0` for production-exact profiled timing). Training numerics
  byte-identical either way (the parity pair proved diagnostics-on == diagnostics-off).

**Measured/assessed and NOT worth it** (do not re-propose): `validation_best_model` dedup
(landed anyway as trivial byte-identical cleanup, but it is ~5 s/fold at TX, ~0.1% — the
verifier disproved the "minutes" premise); per-step `guard_finite_step` host sync (~1 s/fold
AL, semantics-bearing); eval-every-epoch reduction and larger eval batch (selection/tie-break
risk). The `train_perf_multifold` DONE/DECLINED inventory (P7 compile-once, P2/P2a, P5,
fused AdamW, SDPA-in-STAN, bf16-default, num_workers>0, pack_padded_sequence) was
re-confirmed and stands — the remaining big-ticket axis is per-epoch eval at large states
(7.6% of TX fold wall), which is quality-coupled (canonical tie-break) and deliberately left.

## 5 · Key file:line index (verified first-hand)

- Independent train shuffles: `src/data/folds.py:1421-1424` (reg), `:1464-1467` (cat); no generator: `:447-458`; deliberate (`:275-283`).
- Shared split both tasks: `src/data/folds.py:1371-1396`. Row-alignment guards: `next_region.py:87-103`, `folds.py` (`_load_and_validate_check2hgi_data`).
- Cross-attention per-row mixing: `src/models/mtl/mtlnet_crossattn/model.py:168-199`; dualtower raw-seq private tower: `src/models/next/next_stan_flow_dualtower/head.py:187-239`.
- Aligned pairing (G0.1): `src/data/folds.py:506-573,1436-1448`; verdict `docs/studies/pre_freeze_gates/LANE1_G01_VERDICT.md`; roll probe `src/training/runners/mtl_eval.py` (`MTL_ROLL_TASKB_EVAL`).
- ga bug (fixed): `src/training/runners/mtl_cv.py` (comment block where the per-batch zero_grad lived; boundary zero in `_optimizer_micro_step`).

## 6 · Gates run for this audit's changes

1. Unit: full `tests/test_training + test_configs + test_data` — 360 passed (incl. 2 new
   regression tests: `aligned_pairing` field; ga=2 gradient content, fails on pre-fix code).
2. Eager parity: AL fold-0 8-ep fp32 eager, post-fix vs pre-fix artifacts — **PASS,
   byte-identical 6/6** (val+train metric CSVs + both fold reports; covers the zero_grad
   removal, the lazy fold-subset view, and the validation_best_model single-pass path)
   (`runs/postfix_parity_eager` vs `runs/al_parity_eager_off`).
3. Compiled 5-fold AL v17 champion run post-fix — **PASS**: cat 64.53 ± 1.87 / reg
   69.69 ± 3.13 (seed 0, n=5) vs board v17 AL n=20 64.54 / 69.80; fold-0 bit-equal to the
   pre-fix baseline (64.4029/71.6636, same inductor-cache session); fold-5 is the documented
   structural AL outlier (`runs/postfix_5fold`, wall 614 s incl. per-epoch val ×5 folds).
4. P4 A/B: `runs/al_v17_f0_{warmbase,p4cold,p4warm}` + `runs/al_parity_eager_{off,on}`.

Artifacts: `docs/studies/pipeline_audit/runs/` (profiles, logs, scores), drivers
`run_al_v17_audit_1fold.sh`, `run_p4_ab.sh`, `run_postfix_gates.sh`.
