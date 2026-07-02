# pipeline_audit quality follow-ups (2026-07-01)

> **UPDATE 2026-07-02 — items 1–3 RESOLVED/DEMOTED by the pairing battery**
> ([`../studies/pipeline_audit/PAIRING_BATTERY.md`](../studies/pipeline_audit/PAIRING_BATTERY.md),
> 5 arms × 4 seeds × 5 folds at AL on the v17 base):
> **#1 cond_coupling × aligned — TESTED, verdict de-confounded null vs champion**
> (conditioning IS only usable when semantically paired: reg +0.47 over the aligned arm,
> 4/4 seeds — a real mechanism finding — but it never beats base on either head; the R-CC
> closure stands). **#2 binding G0.1 — effectively ANSWERED at AL** (aligned −3.03 cat /
> −0.60 reg, 4/4 seeds, paired; REFUTED as champion candidate; the deficit is pure
> self-pairing semantics — the deranged control recovers base exactly; FL remains
> advisory-null, mechanism-consistent). **#3 batch-level pairing — DEMOTIVATED** (any
> random-other partner ≡ base; no cat gap for a smarter partner to close). Items 4–6 remain
> open; #4's β-trajectory read is now best piggy-backed on any future aligned-arm run.

**Date drafted**: 2026-07-01, from the `pipeline_audit` study
([`docs/studies/pipeline_audit/README.md`](../studies/pipeline_audit/README.md)).
**What's deferred and why**: model-quality experiment candidates surfaced by the full
input-builder + train-pipeline audit. None are perf items; each needs an n=20-grade A/B and
was out of the audit's quality-neutral scope. Items already covered by
`train_perf_multifold/QUALITY_IMPROVEMENTS.md` (27-lever catalog) or refuted there are NOT
repeated here.

## 1 · cond_coupling × aligned-pairing — the untested cell (highest interest)

Per-sample cat→reg conditioning (`cond_coupling≠none`) is the ONE cross-task mechanism whose
value provably requires row-aligned pairing (conditioning on the SAME sample's category
posterior). Every historical R-CC/R-CC+ run (mtl_frontier) predates `--aligned-pairing`
(the flag is one day younger than the R-CC closure), so "the +0.235 cap is the regime" was
concluded from arms whose conditioning input was an unrelated sample at train time while val
was aligned — mechanistically confounded. Also `scripts/baselines/b4_cascade.py` pins
`cond_coupling=posterior` + `disable_cross_attn=True` with no alignment: the cond edge is its
ONLY cat→reg coupling and it trains on garbage (a `train.py` guard now WARNs on this combo).
- **Experiment**: champion v17 base + `--aligned-pairing` + `cond_coupling=posterior`
  (and/or `cc_e2e`) vs champion, AL/FL, seeds {0,1,7,100}.
- **Acceptance**: reg or cat moves ≥0.3 pp n=20 → re-open the coupling direction; else the
  R-CC verdict is de-confounded and final.
- **Unblocked by**: the `--aligned-pairing` CLI fix (ExperimentConfig field) in this audit.

## 2 · Binding G0.1 (aligned pairing, pre-registered)

The advisory G0.1 (seed-0) verdict was NULL@FL / NEGATIVE@AL; the pre-registered BINDING run
(frozen base, seeds {0,1,7,100}, 0.3 pp gate, incl. the stride-1/overlap interaction) is
still pending (`docs/studies/pre_freeze_gates/LANE1_G01_VERDICT.md:39-44`). Cheap now that
the CLI path works. Value: closes the pairing question at n=20 grade for the paper.

## 3 · Batch-level pairing middle ground (novel, unexplored)

Between "row-aligned" (hurts small states) and "random" (content-free cross-reads): group
BOTH loaders' batches by user or region-neighborhood so cross-attention reads
related-but-not-identical contexts. Keeps the augmentation benefit, makes K/V semantically
relevant. No code exists (only aligned vs independent paths in `folds.py`).
- **Acceptance**: beats BOTH random and aligned pairing at AL n=20 to justify the loader work.

## 4 · β (dual-tower fusion scalar) weight-decay exemption + trajectory read

β (init 0.1) sits in the reg AdamW group at wd=0.05 and is documented as able to drift to 0
(severing the shared→reg pathway); `MTL_BETA_NO_WD` exists but is unexercised, and β's
trajectory has only been read under MISALIGNED pairing (where the gradient is expected to
kill it). Read β per epoch under `--aligned-pairing`; if it grows, the shared pathway carries
signal only when pairing is semantic — a cheap mechanism probe piggy-backing on #2.

## 5 · Input-builder hygiene levers (small, quality-adjacent)

- **OOV embedding ≠ pad zero-vector** (`core.py:523`): a dedicated OOV vector distinguishes
  "unknown POI" from "no observation" for the `x.abs().sum()==0` masks. Currently latent
  (0 'None' labels on disk) — land together with any embedding regen.
- **Per-window minimum real-context filter**: `MIN_SEQUENCE_LENGTH` gates a user's TOTAL
  count, not per-window context; 1-4-real-step tail windows still enter training. Ablate a
  `≥k` non-pad-steps filter at AL.
- **Emit raw (unrestricted) `top10_acc`** alongside `top10_acc_indist` in the S2 val path —
  one extra key from the already-accumulated hit vector; makes the OOD discount visible.
- **Tail-target category from the target's own row** (`core.py:630-634` uses
  first-occurrence-by-POI-ID) — exact per-event category, immune to per-POI conflicts.

## 6 · Region-support fold stratification (variance, not bias)

The check2hgi split stratifies on the 7-class cat label only; region support (C up to 8.5k)
varies freely per fold under the user-group constraint → per-fold reg Acc@10 variance partly
reflects split luck (the documented AL fold-5 outlier). Composite-key stratification is
already catalogued (QUALITY_IMPROVEMENTS C4, future-works-only, redraws the frozen split) —
this entry just records the audit's confirmation that region labels are entirely
unstratified today.

## Pointers

`docs/studies/pipeline_audit/README.md` (findings + file:line index),
`docs/studies/pre_freeze_gates/LANE1_G01_VERDICT.md` (G0.1),
`docs/studies/train_perf_multifold/QUALITY_IMPROVEMENTS.md` (the 27-lever catalog this memo
deliberately does not duplicate), `docs/research/mtl_frontier.md` (R-CC context).
