# Phase 6 — Convergence & Canonical MTL Benefit Claims

**Goal:** test our modern MTL configuration (DSelectK + fusion + gradient surgery) against the canonical MTL benefit claims from Caruana (1997), Ruder (2017), and Crawshaw (2020). Closes the loop on CBIC 2025's convergence finding (MTL = 4× slower) and provides the "modern MTL benefits" chapter of the paper.

**Duration:** ~4–6 h (mostly post-hoc analysis of logs from earlier phases; one inference-only run; one cross-probe run).

**Runs in parallel with:** P5 (mechanistic / diagnostic analyses). Both require the champion from P4 to be locked.

**Embedded claims:** C22 (epochs-to-target) · C23 (wall-clock ≤ 2×) · C24 (train-val gap) · C25 (noise robustness) · C27 (representation transfer) · C28 (no negative transfer). C26 (sample efficiency) is **deferred to journal** unless time permits.

---

## Preconditions

- P1–P4 complete. Champion config (arch, optimizer, heads, hparams) locked.
- P2 single-task baselines exist on AL (category-only, next-only) and their per-epoch `MetricStore` logs are archived.
- P2 champion MTL run exists on AL with per-epoch logs.
- Frozen fold indices from P0.8 (`output/{engine}/{state}/folds/fold_indices.pt`) — so single-task and MTL runs share byte-identical val splits (required for C28 paired test).

---

## Steps

### P6.1 — Epochs-to-target (C22)

**Goal:** show MTL reaches target F1 in ≤ cumulative single-task epochs.

**No new training.** Post-hoc analysis of existing logs.

**Method:**
1. For each model (single-task-cat, single-task-next, MTL-cat, MTL-next), load per-fold per-epoch val F1 from `MetricStore`.
2. Define per-model target = 0.90 × (best val F1 reached across training). Intrinsic target per model — avoids the CBIC pitfall of a shared absolute threshold.
3. Epochs-to-target = first epoch where val F1 ≥ target, averaged over 5 folds.
4. Report `E_MTL_cat + E_MTL_next` vs `E_ST_cat + E_ST_next`. Wait — both MTL tasks train in a single run, so use `max(E_MTL_cat, E_MTL_next)` as the effective MTL epoch count.

**Output:** `docs/studies/results/P6/C22_epochs_to_target.json` + a small table for the paper.

**Pass:** `max(E_MTL_cat, E_MTL_next) ≤ E_ST_cat + E_ST_next`.

---

### P6.2 — Wall-clock ratio (C23, sharpens C21)

**Goal:** `wall_MTL / (wall_single_cat + wall_single_next) < 2`.

**No new training.** Read `wall_clock_seconds` from each run's `metadata.json` (archived via `/study import`).

**Method:** simple arithmetic on 3 numbers. Report ratio + absolute seconds for each.

**Output:** one row in `docs/studies/results/P6/C23_wallclock.json`. One sentence in the paper.

**If ratio > 2:** report honestly; frame as "still better than CBIC's 4× but MTL does carry a training cost" — this is a legitimate finding too.

---

### P6.3 — Train-val generalization gap (C24)

**Goal:** show MTL's train-val F1 gap is smaller than single-task's for at least one head.

**No new training.**

**Method:**
1. From `FoldHistory` logs, extract final-epoch train F1 and val F1 per fold per model.
2. Gap = mean over folds of `(F1_train − F1_val)`.
3. Compare gaps: MTL-cat vs ST-cat, MTL-next vs ST-next.

**Output:** `docs/studies/results/P6/C24_train_val_gap.json` + bar chart.

**Pass:** at least one task shows `gap_MTL < gap_ST` by > 1 percentage point.

**Notes:** if `MetricStore` doesn't log train F1 on every epoch (to save time), we may only have val. In that case, skip C24 or re-run one fold with full logging — decide in P0.

---

### P6.4 — Noise robustness (C25)

**Goal:** MTL degrades more gracefully than single-task under embedding noise.

**Inference-only.** ~1 h compute.

**Method:**
1. Load best MTL checkpoint and best single-task checkpoints (all from AL champion run).
2. For σ ∈ {0.0, 0.05, 0.1, 0.2}:
   - On each val fold: add `torch.randn_like(x) * σ` to the fused embedding input.
   - Compute F1 per task per model.
3. Plot F1 vs σ. Report "F1 drop at σ=0.1" as headline number.

**Output:** `docs/studies/results/P6/C25_noise_robustness.json` + `figure_noise.png`.

**Implementation:** small wrapper script `scripts/study/eval_noise.py` that loads a checkpoint, wraps the fusion input with a noise transform, and re-runs evaluation. ~100 LoC.

**Pass:** MTL's F1 drop at σ=0.1 is < single-task's F1 drop for at least one task.

---

### P6.5 — Sample efficiency (C26) — **deferred to journal**

Skip unless P6.1–P6.4 + P6.6 + P6.7 finish ahead of schedule with ≥ 2 days before BRACIS deadline.

If attempted (BRACIS scope): AL only, fractions {25%, 50%, 100%}, 5 folds × 3 models × 3 fractions = 45 runs (~12 h).
If attempted (journal scope): AL + AZ, fractions {25%, 50%, 75%, 100%}, 5 folds × 3 models × 4 fractions × 2 states = 120 runs.

---

### P6.6 — Representation transfer (C27)

**Goal:** show the MTL-trained shared backbone is a better frozen feature extractor than single-task backbones.

**~2 h compute.**

**Method, cross-task variant:**
1. Load MTL champion checkpoint on AL. Freeze shared backbone (encoder + shared layers).
2. Attach a fresh linear classifier on top. Train only the linear layer on AL next-task data. 5 folds × 20 epochs (fast — only ~5K linear params).
3. Repeat with single-task-next backbone frozen.
4. Compare linear-probe F1.

**Method, cross-state variant (if time):**
1. Take MTL backbone trained on AL.
2. Freeze, train linear head on AZ data only.
3. Compare to AZ-trained-from-scratch and to ST-next-AL-backbone linear probe.

**Output:** `docs/studies/results/P6/C27_repr_transfer.json` + bar chart.

**Implementation:** `scripts/study/linear_probe.py` — loads checkpoint, freezes backbone, replaces head with `nn.Linear`, re-runs training with frozen base. ~150 LoC.

**Pass:** MTL-backbone linear-probe F1 > ST-backbone linear-probe F1 (Wilcoxon signed-rank across folds).

---

### P6.7 — No negative transfer (C28)

**Goal:** **mandatory reviewer shield.** Show per-task MTL F1 ≥ best single-task F1 at the fold level.

**No new training.** Paired-fold analysis on P2 data.

**Method:**
1. For each fold i ∈ {1..5}: pull MTL's cat_F1_i and ST-cat's F1_i.
2. Wilcoxon signed-rank test (one-sided, alternative: MTL ≥ ST).
3. Report p-value + Cohen's d effect size + per-fold pairs.
4. Same for next-task.

**Output:** `docs/studies/results/P6/C28_negative_transfer.json` — 2 tests (cat, next), each with p-value, mean delta, Cohen's d, per-fold pairs.

**Pass:** p > 0.05 (i.e., we cannot reject the null that MTL ≥ ST) for both tasks — ideally with visibly positive deltas. If p < 0.05 *and* delta is negative on either task, we have negative transfer on that task and must discuss it in the paper.

**Implementation:** `scripts/study/paired_test.py` — reads two run archives, aligns by fold index, runs scipy `wilcoxon`. ~80 LoC.

**Notes:** **This is the single highest-priority P6 step.** If the paired test passes (MTL ≥ ST per task), we have a concrete refutation of CBIC 2025's "MTL ≈ ST" finding, framed in stronger per-task form. If it fails, the paper's framing must change — better to know now.

---

## Compute budget

| Step | Compute | Output |
|------|---------|--------|
| P6.1 C22 | 0 (post-hoc) | table |
| P6.2 C23 | 0 (post-hoc) | 1 ratio |
| P6.3 C24 | 0 (post-hoc) | bar chart |
| P6.4 C25 | ~1 h (inference) | figure |
| P6.5 C26 | *deferred* | — |
| P6.6 C27 | ~2 h | bar chart |
| P6.7 C28 | 0 (post-hoc) | 2 stat tests |
| **Total** | **~3 h compute + ~2 h analysis/writing** | |

---

## Outputs

- `docs/studies/results/P6/SUMMARY.md` rolling up all 6 claims with verdict and pointer to evidence.
- Per-claim JSON artifacts under `docs/studies/results/P6/`.
- 2 figures (noise robustness curve; representation-transfer bar chart).
- Paper paragraph(s) for the MTL-benefits section.
- Updated `CLAIMS_AND_HYPOTHESES.md` — each C22–C28 has `status` and `evidence` pointer.

---

## Exit criteria

- [ ] C22, C23, C24, C27, C28 have status ∈ {confirmed, refuted, partial}.
- [ ] C25 has status ∈ {confirmed, refuted, partial} if P6.4 was run; otherwise `deferred` with reason.
- [ ] C26 status = `deferred_to_journal` unless attempted.
- [ ] `docs/studies/results/P6/SUMMARY.md` exists and is complete.

---

## Relation to other phases

- **Depends on P2** for single-task baselines. If P2 is skipped, C22/C24/C27/C28 cannot be tested.
- **Depends on P4** for the locked champion (checkpoints used in C25/C27 probes).
- **Parallel to P5** — P5 covers mechanistic claims (gradient cosine, wall-clock scaling); P6 covers MTL-benefit claims. No overlap.
- **Feeds synthesis** — results go directly into the paper's "MTL is worth it when configured right" section.

---

## Risk notes

1. **P2 may not log training F1 per epoch.** Would hurt C24. Mitigation: add `--log-train-f1` flag to training loop in P0; re-enable when archiving P2 data.
2. **Single-task runs may use different folds than MTL runs** if seed/fold-freezing not handled properly. Mitigation: **P0.8 fold-freezing is a hard prerequisite for C28 paired test to be valid.**
3. **C26 deferred but reviewers may ask.** Mitigation: include a short "sample-efficiency experiments are left for journal extension" sentence in paper's limitations.
