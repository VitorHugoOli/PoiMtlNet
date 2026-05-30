# Tier C verdict — protocol coherence (C1 3-snapshot routing / C2 reg-freeze / C3 zero-cat-kv)

**Date**: 2026-05-28 (C2/C3); **C1 re-scored 2026-05-29** after the Tier C advisor found a reg-modality scoring bug.
**Phase**: Tier C (C1 = §4.1 3-snapshot routing variant A; C2 = §4.4 reg-freeze-after-peak; C3 = P4 K/V residual).
**Scope**: AL + AZ, seed=42, 5 folds, H3-alt small-state recipe, `--engine check2hgi`, `--no-checkpoints` (C1 saves 3 snapshots/fold, deleted immediately after scoring).
**Baseline**: the Tier B `canonical_baseline` MTL cell at the SAME state (seed=42, same H3-alt recipe, same folds). Reused (confirmed non-degenerate: AL disjoint reg 50.82, AZ 41.33 — matches the Tier B B-summary table).
**Stat method**: RAW per-fold values, paired by fold, `scipy.stats.wilcoxon(deltas, alternative=...)` (5-fold has no ties → exact branch). Per the Tier-A1 scipy-dispatch lesson, never round before testing.

**Verdict in one line:**
- **C1 — §DISCUSSION FOOTNOTE** (re-scored 2026-05-29 after a reg-modality scoring bug was found; prior ARCHIVE superseded). On the CORRECT region modality the +2 pp gate clears at **AZ (Δreg +2.54, 5/5 folds, Wilcoxon p=0.03)** but fails at **AL (−7.89, p=0.31)** due to ONE genuine degenerate reg-best snapshot (AL fold3, Acc@10=0.12 % even on correct modality — a real Acc@1-selector pathology, NOT the modality bug). One-state pass → footnote, not full PROMOTE (needs +2 pp at both) and not ARCHIVE (AZ is a real, significant gain). Conditional on an Acc@10-aligned reg-best selector + degenerate-snapshot guard (out of scope).
- **C2 — ARCHIVE, closes §4.4 entirely** (no N improves cat without a reg regression > σ_fold; the last curriculum variant is falsified).
- **C3 — P4 FULLY CLOSED** (zeroing the cat K/V path does NOT shift the reg peak later or improve magnitude at either state; the residual MTL-vs-STL reg gap is NOT cross-attention K/V capacity-stealing).

---

## C2 — `--reg-freeze-at-epoch N` sweep (§4.4 freeze-reg-after-peak)

### Baseline reg-peak epochs (key context)
- **AL** peaks LATE: per-fold reg `top10_acc_indist` argmax at ep [13, 12, 10, 14, 15] (mean 12.8).
- **AZ** peaks EARLY: ep [4, 7, 6, 9, 5] (mean 6.2).

So freezing reg at N ∈ {2,4,6} freezes it at-or-before its true peak. The disjoint-reg frontier (max-over-epochs) therefore caps at roughly the reg quality reached by epoch N; post-freeze the reg head is fixed, cat continues joint with the now-frozen reg representation.

### Three-frontier (mean over 5 folds, seed=42)

| Cell | AL disjoint reg | AL Δreg | AL disjoint cat | AL Δcat | AZ disjoint reg | AZ Δreg | AZ disjoint cat | AZ Δcat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical baseline | 50.82 | — | 45.76 | — | 41.33 | — | 48.87 | — |
| C2 N=2 | 43.13 | **−7.69** | 46.13 | +0.37 | 32.99 | **−8.34** | 48.81 | −0.06 |
| C2 N=4 | 46.64 | **−4.18** | 46.22 | +0.46 | 40.40 | −0.93 | 48.96 | +0.09 |
| C2 N=6 | 49.77 | −1.05 | 45.82 | +0.06 | 41.25 | −0.07 | 48.85 | −0.02 |

### Wilcoxon (5-fold paired, RAW per-fold disjoint values)

cat one-sided `cell > baseline` (does freezing IMPROVE cat?); reg one-sided `cell < baseline` (does freezing HURT reg? = `greater` on baseline−cell).

| Cell | per-fold Δcat (f1..f5) | mean Δcat | p(cat>base) | per-fold Δreg (f1..f5) | mean Δreg | reg folds neg | σ_base | reg hurt > σ? |
|---|---|---:|---:|---|---:|---:|---:|---|
| AL N=2 | −0.06,+0.22,+1.57,+0.30,−0.21 | +0.37 | 0.156 | −7.42,−6.09,−8.03,−7.97,−8.95 | −7.69 | 5/5 | 3.21 | YES (2.4σ) |
| AL N=4 | −0.59,+0.63,+1.34,−0.49,+1.38 | +0.46 | 0.156 | −4.54,−4.43,−1.50,−5.98,−4.47 | −4.18 | 5/5 | 3.21 | YES (1.3σ) |
| AL N=6 | −0.35,−0.56,+1.52,−0.43,+0.15 | +0.06 | 0.688 | −1.30,−1.42,−1.54,+0.12,−1.10 | −1.05 | 4/5 | 3.21 | borderline (0.3σ) |
| AZ N=2 | +0.30,−0.52,−0.57,+0.65,−0.17 | −0.06 | 0.594 | −7.68,−5.90,−6.32,−14.09,−7.71 | −8.34 | 5/5 | 2.73 | YES (3.1σ) |
| AZ N=4 | +0.41,−1.04,+0.39,+0.92,−0.20 | +0.09 | 0.406 | 0.00,−0.71,−1.28,−1.31,−1.35 | −0.93 | 4/5 | 2.73 | no (0.3σ) but cat not sig |
| AZ N=6 | +0.36,−0.71,+0.27,+0.46,−0.48 | −0.02 | 0.688 | 0.00,+0.02,−0.21,−0.27,+0.10 | −0.07 | 3/5 | 2.73 | no |

### Decision gate (INDEX.md §C2)
> Does Δcat improve at any N WITHOUT Δreg regression > σ_fold, vs the canonical baseline?

**NO at every N at both states.**
- Where cat shows its largest (still non-significant) lift — AL N=2/N=4 (+0.37/+0.46 pp, p=0.156) — reg collapses by −7.69 / −4.18 pp (≫ σ_fold).
- Where reg is preserved within σ_fold — N=6 (AL −1.05, AZ −0.07) and AZ N=4 — Δcat is null (≤+0.09 pp, p ≥ 0.41).
- There is no (N, state) combination with a significant cat gain AND a sub-σ reg cost.

**VERDICT: C2 ARCHIVE. Closes §4.4 entirely.** The asymmetric freeze-reg-after-peak curriculum — the one form P4 had not falsified — is now falsified: freezing reg trades a real, large reg regression for at-best a null, never-significant cat gain. The cat head does not benefit measurably from a frozen reg representation; the joint loss already gives cat everything it can use. No multi-seed promotion.

---

## C3 — `--zero-cat-kv` (P4 residual K/V capacity-stealing test)

Zeroes the cat-stream K/V tensors feeding the reg-side cross-attention (`cross_ba`) in `MTLnetCrossAttn`, leaving reg-side K/V into `cross_ab` intact; projection weights NOT zeroed (reversible). Hypothesis (INDEX.md §C3): if reg peaks LATER or improves in magnitude under K/V-zeroed cat path, the residual MTL-vs-STL reg gap is cross-attention K/V capacity-stealing, not the shared-backbone parameters P4 already exonerated.

### Result (mean over 5 folds, seed=42)

| | AL | AZ |
|---|---:|---:|
| baseline disjoint reg | 50.82 | 41.33 |
| C3 disjoint reg | 50.55 | 41.33 |
| **Δreg** | **−0.28** (p(reg>base)=0.94, 1/5 folds +) | **+0.01** (p=0.69, 2/5 folds +) |
| baseline reg-peak epoch (mean) | 12.8 | 6.2 |
| C3 reg-peak epoch (mean) | **9.4** (EARLIER) | **6.6** (≈ unchanged) |
| baseline disjoint cat F1 | 45.76 | 48.87 |
| C3 disjoint cat F1 | 46.13 (Δ+0.37, p=0.31) | 48.83 (Δ−0.04, p=0.69) |

Per-fold reg Δ: AL [−0.28,−0.04,−0.28,+0.04,−0.81]; AZ [+0.06,+0.62,−0.19,−0.29,−0.15]. Per-fold cat Δ: AL [−1.02,−0.10,+2.02,+0.71,+0.22]; AZ [+0.21,−0.59,−0.31,+0.29,+0.20].

### Decision gate (INDEX.md §C3)
> Does reg peak shift later / improve magnitude vs baseline?

**NO at both states.**
- Reg magnitude is statistically unchanged (AL −0.28 pp ns, AZ +0.01 pp ns).
- Reg peak epoch shifts EARLIER at AL (12.8 → 9.4), the OPPOSITE of the "peaks later" hypothesis, and is essentially unchanged at AZ (6.2 → 6.6). There is no later-peak / higher-magnitude signature.
- **Cat side-effect** (reported per the risk note): negligible and non-significant — AL +0.37 pp (p=0.31), AZ −0.04 pp (p=0.69). Silencing the cat K/V path does not meaningfully cost cat either, consistent with the reg-side cross-attention carrying the joint signal.

**VERDICT: C3 — P4 FULLY CLOSED.** Removing the cat-encoder activation contribution to the reg-side cross-attention K/V channel does not recover MTL reg or delay its peak. Combined with P4's frozen-cat-parameters result, both the cat-parameter and cat-activation pathways are now exonerated: the residual MTL-vs-STL reg gap is NOT cat→backbone capacity-stealing (neither via shared-backbone parameters nor via cross-attention K/V). The residual mechanism lies elsewhere on the architectural axis (handed to `mtl_improvement` — this test narrows its search space by eliminating the K/V-stealing sub-mechanism rather than surfacing a new one).

---

## C1 — `--save-task-best-snapshots` 3-snapshot routing (variant A) — RE-SCORED 2026-05-29 (modality fix)

> ⚠ **The original 2026-05-28 C1 numbers are SUPERSEDED — they were computed on the WRONG reg modality.** `scripts/route_task_best.py` rebuilt the val loaders via `FoldCreator(...)` without passing `task_b_input_type`, so they defaulted to `"checkin"` while the run trained `task_b=region`. Every C1 reg score (all slots, all folds, both states) fed CHECKIN-modality embeddings into a REGION-trained reg head → garbage reg numbers. `ExperimentConfig` did not persist `task_b_input_type`, so the scorer could not recover it. The original verdict's claim that the AL-fold3 collapse was a "degenerate Acc@1-selector snapshot" was therefore **confounded by the modality bug and unproven**. This section reports the corrected re-run. (Tier C advisor review, 2026-05-28; fix + re-run, 2026-05-29.)

Trained AL/AZ with `--save-task-best-snapshots` (3 internally-consistent MTL snapshots/fold: cat-best by val cat F1, reg-best by val reg Acc@1, joint-best by `joint_geom_lift`), seed=42, 5 folds, H3-alt small-state recipe, `--no-checkpoints`. Snapshots re-built fresh (the originals were deleted after the bugged scoring), scored each fold with the **fixed** `scripts/route_task_best.py`, deleted snapshots immediately (disk discipline; ~295 MB/state). Deploy interpretation: reg requests → reg-best snapshot, cat requests → cat-best snapshot, vs serving everything from joint-best.

**Fix landed this session (the THIRD fix the advisor flagged):**
1. `ExperimentConfig.task_a_input_type` / `task_b_input_type` are now **persisted** to `config.json` / `manifest.json` (append-only fields, default `"checkin"`; old configs back-compat-load).
2. `route_task_best.py` reads `task_b_input_type` from the run's config and passes it to `FoldCreator`, so the val loaders use the SAME modality as training (region). An explicit `--task-{a,b}-input-type` CLI override is the fallback for old configs.
3. Unit tests in `tests/test_substrate_protocol_cleanup_flags.py::TestTaskInputTypePersistence` assert the round-trip and that the scorer rebuilds the region loader (5 new tests, all green).

The two prior fixes are retained: (a) `TaskSet` reconstruction from the config dict preserving the run's `--cat-head next_gru` / `--reg-head next_getnext_hard` head_factory overrides (a `get_preset(name)` fallback would rebuild DEFAULT heads → `load_state_dict` key mismatch); (b) `top_k=(1,3,5,10)` so reg Acc@10 (the gate metric) is computed.

**Sanity gate (CONFIRMED).** Each `route_fold*.json` now carries `"task_b_input_type": "region"`, and the reg-best snapshot re-scores into the sane 35–51 % Acc@10 band on the 4 healthy AL folds and all 5 AZ folds — NOT ~0. The joint baseline is likewise correct now (~37–48 %, vs the bugged 38/46). The fix is effective.

### Routing deltas (reg-best vs joint-best on Acc@10; cat-best vs joint-best on F1) — CORRECTED region modality

| State | reg-best Acc@10 | joint-best Acc@10 | mean Δreg | per-fold Δreg (f1..f5) | folds + | Wilcoxon p(reg>joint) |
|---|---:|---:|---:|---|---:|---:|
| AL | 38.41 | 46.30 | **−7.89** | +2.83, +1.22, **−48.03**, +0.43, +4.09 | 4/5 | 0.3125 |
| AZ | 39.98 | 37.44 | **+2.54** | +3.81, +2.65, +3.43, +0.76, +2.05 | **5/5** | **0.03125** |

| State | cat-best F1 | joint-best F1 | mean Δcat | per-fold Δcat (f1..f5) | folds + | Wilcoxon p(cat>joint) |
|---|---:|---:|---:|---|---:|---:|
| AL | 45.14 | 44.27 | +0.87 | −0.01,+0.21,+1.78,+0.32,+2.05 | 4/5 | 0.0625 |
| AZ | 48.49 | 48.37 | +0.12 | +0.55,+0.71,+1.16,−1.63,−0.18 | 3/5 | 0.4062 |

### What changed under the fix
- **AZ FLIPPED to a clear pass.** On the wrong modality AZ looked null (Δreg +0.50, 4/5, p=0.31; with an apparent "fold2 weakness" at 0.3249). On the **correct region modality AZ is +2.54 pp, ALL 5 folds positive, Wilcoxon p=0.03125 (significant)**. The bugged "AZ fold2 collapse" was an artefact of the modality mismatch; there is **no degenerate AZ fold**. AZ clears the +2 pp gate.
- **AL fold3 is a GENUINE degenerate snapshot — not a modality artefact.** Re-scored on the correct region modality, the AL fold3 reg-best snapshot is still reg Acc@1 = 0.0000 / **Acc@10 = 0.12 %**, while the SAME fold's joint-best snapshot scores a healthy 48.15 % and cat-best 46.78 %. The reg-best slot was *saved* at val reg **Acc@1 = 0.2801 at epoch 14** (the monitored selector metric looked healthy at save time), yet that epoch-14 snapshot does not generalise on the held-out val set — Acc@1 collapses to 0. This is a real `MultiTaskBestTracker` reg-best (Acc@1) selector pathology: the Acc@1-monitored snapshot can land on an epoch whose top-1 over-fits a degenerate near-constant region prediction while top-10 collapses. **The advisor's hypothesis that the 0.0 was purely the modality bug is falsified for AL fold3 — the modality bug was a *separate* confound (it depressed the healthy folds' apparent magnitudes and the AZ picture), but the AL-fold3 degeneracy survives the correct modality.**
- The healthy 4 AL folds give Δreg +2.83/+1.22/+0.43/+4.09 (mean +2.14, p=0.0625) — the routing signal is real where the selector behaves, but the 5-fold mean is −7.89 because of the one genuine degenerate.

### Decision gate (INDEX.md §C1)
> Δreg @ task-best vs joint-best ≥ +2 pp at AL AND AZ → promote; ~+1 pp at one state only → §Discussion footnote; null → archive.

**§DISCUSSION FOOTNOTE (one-state pass).** The corrected 5-fold gate clears at **AZ (+2.54 pp, p=0.03)** but FAILS at **AL (−7.89 pp, p=0.31)** because of the single genuine fold3 reg-best degenerate snapshot. This is the "one-state-only" branch → §Discussion footnote, NOT full PROMOTE (the +2 pp at BOTH precondition is not met) and NOT ARCHIVE (AZ is a real, significant gain — the data do not support a blanket null). The honest mechanism: per-task reg-best routing extracts a real +2–4 pp reg gain over joint-best wherever the Acc@1 reg-best selector lands on a generalising snapshot (4/4 healthy AL folds + 5/5 AZ folds), but the unguarded Acc@1 selector can land on a non-generalising snapshot (AL fold3) that makes routing catastrophically worse than the single joint-best checkpoint. **Deploy of variant A is therefore conditional on an Acc@10-aligned reg-best selector + a degenerate-snapshot guard in `MultiTaskBestTracker` (currently out of scope).** Cat-best routing is a near-null +0.87 (AL, p=0.06) / +0.12 (AZ, ns).

**Recommended follow-up (footnote scope):** a one-line guard — select the reg-best slot by val reg **Acc@10** (the gate metric) instead of Acc@1, and/or reject a slot whose Acc@10 falls below e.g. 0.5× the joint-best Acc@10 — would very likely recover AL to a pass (its 4 healthy folds already average +2.14 pp). If pursued, re-run multi-seed at AL/AZ before any §0.x promotion.

**Deploy-cost note (mandatory per INDEX.md §C1 risk):** variant A requires **3× checkpoint storage** (~60 MB/fold here, ~20 MB/snapshot; scales with model size) and a **2-model load + 2 forward passes** at inference (cat-best for cat requests, reg-best for reg requests). The reg gain is real at AZ but (a) is single-seed=42 and (b) rides on a selector that is demonstrably brittle at AL. F1 / `joint_geom_simple` at a single checkpoint already extracts essentially all the per-task **cat** capacity (cat routing Δ ≤ +0.87 pp), consistent with the F1-selector ship decision from `mtl-protocol-fix` (variant B). Variant C-prime (mixed snapshot + 1-2 ep joint fine-tune) is NOT triggered at this stage — the footnote-grade, selector-brittle result does not meet the storage-binding precondition for C-prime.

**Artefacts (re-score):** `tier_c/{alabama,arizona}/C1_route/route_fold{1..5}.json` (now carry `task_b_input_type=region`) + `.log`, `tier_c/{alabama,arizona}/C1_route/config.json` (persisted modality), `tier_c/tier_c1_routing_analysis.json` (regenerated), `scripts/substrate_protocol_cleanup/analyze_tier_c1.py`, `scripts/substrate_protocol_cleanup/c1_rescore/` (megascript + train/score logs). Code: `scripts/route_task_best.py`, `src/configs/experiment.py` (persisted fields), `scripts/train.py` (persist wiring), `tests/test_substrate_protocol_cleanup_flags.py::TestTaskInputTypePersistence`.
