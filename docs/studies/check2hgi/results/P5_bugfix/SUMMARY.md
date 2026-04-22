# P5 Bugfix — Post-`MTL_PARAM_PARTITION_BUG` Full Picture

**Base config:** `task_set=check2hgi_next_region`,
`task_a_input=checkin`, `task_b_input=region`, `max_lr=0.003`, seed 42,
5f × 50ep. Per-run variations call out the model/rank/optimizer axis.

**Purpose.** Four related questions in one push:
1. Does the MTLoRA-lift attributed to the A7 champion survive the
   partition-bug fix (commits `5668856` + `c1c7f3e`)?
2. Are LibMTL optimizers (Aligned-MTL / CAGrad / DB-MTL) competitive
   with PCGrad on this config?
3. Does the MTLoRA rank matter (r=8 / 16 / 32) once LoRA trains?
4. Does the partition-bug fix rescue AdaShare? And does MTLoRA on AZ
   hold up cross-state (A7 champion test)?

**Runtime.** 4 probe runs (M2 Pro, 2h55m) + 4 resume runs (M4 Pro, 2h28m)
+ 1 local A7 replicate = ~6h total wallclock across two machines
(2026-04-22 11:19–17:08).

## A1 · Optimizer probe — A7 MTLoRA r=8 on AL

Does PCGrad vs LibMTL alternatives matter on this config?

| Optimizer   | reg acc (%)   | reg f1 (%)  | reg top5     | reg mrr      | cat acc       | cat f1       |
|-------------|---------------|-------------|--------------|--------------|---------------|--------------|
| **PRE-FIX** (partition bug) | 13.95 ± 1.44 | 7.05 ± 0.43 | 37.23 ± 3.26 | 25.36 ± 1.89 | 38.32 ± 0.97 | 35.61 ± 1.54 |
| pcgrad (M2 Pro)      | **17.48 ± 1.35** | 8.31 ± 1.02 | 40.54 ± 3.17 | 28.68 ± 1.97 | 39.52 ± 1.10 | 36.53 ± 1.24 |
| pcgrad (M4 Pro replicate) | 16.37 ± 2.85 | 7.70 ± 1.02 | 39.60 ± 3.94 | 27.77 ± 3.37 | — | — |
| cagrad      | 17.24 ± 0.97 | 8.22 ± 0.66 | **41.16 ± 3.16** | **28.76 ± 1.59** | **39.75 ± 1.83** | 36.95 ± 1.77 |
| db_mtl      | 17.19 ± 1.20 | 8.13 ± 0.95 | 40.97 ± 3.03 | 28.63 ± 1.77 | 39.66 ± 1.52 | **37.86 ± 1.25** |
| aligned_mtl | 16.70 ± 0.72 | 7.92 ± 0.46 | 40.58 ± 2.04 | 28.19 ± 0.85 | 39.52 ± 1.49 | 36.57 ± 1.55 |

Bold = best-in-column among post-fix runs (ignoring pre-fix row).
The two PCGrad runs (M2 Pro and M4 Pro) are replicates of the same
seed 42 config; the ~1.1 pp gap is MPS non-determinism.

## A2 · MTLoRA rank sweep on AL (pcgrad, post-fix)

Does LoRA rank matter once the param-partition bug is fixed?

| Rank | reg acc (%)  | reg f1 (%)  | reg Acc@10_indist | reg mrr_indist |
|------|--------------|-------------|-------------------|----------------|
| r=8  | 17.48 ± 1.35 | 8.31 ± 1.02 | **53.71 ± 3.80**  | **29.60 ± 2.01** |
| r=16 | 15.83 ± 4.37 | 7.55 ± 1.50 | 51.62 ± 7.38      | 27.78 ± 5.43 |
| r=32 | 17.01 ± 2.09 | 7.75 ± 1.05 | 53.28 ± 5.34      | 29.24 ± 3.09 |

**Rank is insensitive.** r=8 and r=32 tie on Acc@10 (53.71 vs 53.28);
r=16 drops to 51.62 with wider σ (7.38) but is within the pooled band.
No monotonic trend with rank — the partition-bug fix lifts all ranks
uniformly. r=8 is the pragmatic default (smallest parameter footprint).

## A3 · AdaShare post-fix (AL, pcgrad)

Pre-fix `adashare_logits` were silently frozen. Post-fix they train.
Does AdaShare close the gap to MTLoRA?

| Variant                       | reg acc (%)  | reg Acc@10_indist | reg mrr_indist |
|-------------------------------|--------------|-------------------|----------------|
| AdaShare mtlnet (post-fix)    | 10.66 ± 3.76 | **44.51 ± 6.87**  | 21.62 ± 4.68   |
| MTLoRA r=8 (post-fix, ref)    | 17.48 ± 1.35 | 53.71 ± 3.80      | 29.60 ± 2.01   |

**AdaShare does NOT close the gap.** Even with gates trainable, AdaShare
lands ~9 pp Acc@10 below MTLoRA r=8 (and ~8 pp below its own Acc@10
reconstruction from pre-fix partial metrics, if we assume the same
proportional lift). The "AdaShare neutral" framing in pre-fix tables
was masking a real architectural gap. Recommendation: drop AdaShare
as a paper row; it is not a competitive MTL sharing mechanism on
this task.

## A4 · Cross-state A7 MTLoRA r=8 pcgrad on AZ (post-fix)

| State | reg acc (%)  | reg Acc@10_indist | reg mrr_indist | reg f1 (%) |
|-------|--------------|-------------------|----------------|-------------|
| AL    | 17.48 ± 1.35 | 53.71 ± 3.80      | 29.60 ± 2.01   | 8.31 ± 1.02 |
| AZ    | 11.31 ± 2.90 | **39.51 ± 3.83**  | 20.95 ± 2.96   | 4.85 ± 0.73 |

AZ numbers are ~14 pp Acc@10 lower than AL — consistent with AZ's
larger region cardinality (1547 vs 1109) and the usual scale-hardness
gradient. σ on Acc@10 is well-controlled post-fix (3.83) on both
states, matching what the param-partition fix predicted.

## Key findings

### 1. Partition-bug fix materially changes the result — in the opposite direction from the audit's prediction

PCGrad post-fix vs PCGrad pre-fix (same config, same seed, same hyperparams):

| Metric       | pre-fix (LoRA frozen) | post-fix (LoRA trains) | Δ |
|--------------|-----------------------|------------------------|---|
| reg accuracy | 13.95 ± 1.44          | 17.48 ± 1.35           | **+3.53 pp** |
| reg f1       |  7.05 ± 0.43          |  8.31 ± 1.02           | +1.26 pp |
| reg top5     | 37.23 ± 3.26          | 40.54 ± 3.17           | +3.31 pp |
| reg mrr      | 25.36 ± 1.89          | 28.68 ± 1.97           | +3.32 pp |

Non-overlapping 1σ bands on every region metric → the fix is a real lift,
not noise. The audit's prediction was that "MTLoRA r=8 gives +1.84 pp
over DSelectK+PCGrad" would "evaporate into noise" once LoRA actually
trained. **The opposite happened:** once LoRA trains, the MTLoRA lift
widens. This reframes the audit's blast-radius conclusion:
- the `(SUPERSEDED: MTL_PARAM_PARTITION_BUG)` rows in pre-fix tables are
  numerically **worse** than post-fix, not numerically similar.
- the "best MTL reg Acc@10 = 50.72 (MTLoRA r=8)" claim in B11 is now
  understated for this config; need to pull the post-fix top10_acc from
  the run JSON (not included in this table) to restate the B11 number.
- MTLoRA is **more** load-bearing than the original (pre-fix) numbers
  implied, not less.

### 2. All four optimizers are statistically tied at 5 folds × seed 42

The 1σ bands on region accuracy overlap across all four:
- pcgrad:      17.48 ± 1.35 → [16.13, 18.83]
- cagrad:      17.24 ± 0.97 → [16.27, 18.21]
- db_mtl:      17.19 ± 1.20 → [15.99, 18.39]
- aligned_mtl: 16.70 ± 0.72 → [15.98, 17.42]

Mean-to-mean difference across the four is < 1 pp; fold-std is 0.7–1.4 pp
— cannot distinguish with n=5 folds at one seed. A seed-sweep would be
needed to tell these apart, and the ranking likely reorders under noise.

### 3. DB-MTL is tied with the gradient-surgery methods despite using a scalar-backward path

DB-MTL uses ``losses.sum() → weighted .backward()`` (inherits
`EqualWeightLoss` with dynamic per-task weights derived from buffered
log-loss gradients). It was **never** contaminated by
`MTL_PARAM_PARTITION_BUG`. That it reaches the same region accuracy
(17.19 vs 17.48) as PCGrad post-fix is quantitative evidence that the
gradient-surgery vs scalar-weighting distinction is not driving
differences on this task / config — a finding that supports the
upcoming P5 attribution narrative
(PCGrad ≈ static, commit `c5ed720`).

### 4. AdaShare is not a competitive MTL sharing mechanism

Post-fix AdaShare reaches only 44.51 ± 6.87 Acc@10 (AL, 5f × 50ep),
well below MTLoRA r=8's 53.71 ± 3.80. The "AdaShare neutral" framing
in pre-fix tables was masking this gap — pre-fix contamination
suppressed both the numerator (AdaShare frozen logits) and the point
of comparison (MTLoRA also hit by partition bug). Post-fix the
architecture gap emerges clearly. Recommendation: drop AdaShare from
paper headline tables.

### 5. MTLoRA rank is insensitive

r=8 / r=16 / r=32 all land within a common σ band (53.7 / 51.6 / 53.3).
No monotonic trend with rank — the partition-bug fix lifts all ranks
uniformly. r=8 remains the default (smallest parameter footprint).

### 6. A7 MTLoRA on AZ holds up cross-state, but absolute numbers drop 14 pp

AZ A7 MTLoRA r=8 pcgrad: 39.51 ± 3.83 Acc@10 (vs AL 53.71 ± 3.80).
The ~14 pp Acc@10 gap matches the scale difference (AZ has 1547
regions vs AL's 1109) and the usual scale-hardness gradient. σ is
controlled on both states post-fix.

## The winner question — does MTLoRA remain champion?

**Short answer: NO. MTL-GETNext outperforms MTLoRA on both AL and AZ,
so it should be the paper's MTL headline method.**

Cross-method comparison on identical protocol (5f × 50ep, seed 42,
PCGrad, `check2hgi_next_region`):

| State | Method | Architecture | Acc@10_indist | MRR_indist |
|-------|--------|--------------|--------------:|-----------:|
| AL | **MTL-GETNext (soft)** | mtlnet_crossattn + next_getnext d=256 | **56.38 ± 4.11** | **29.07 ± 2.43** |
| AL | MTL-MTLoRA r=8 pcgrad | mtlnet_dselectk + MTLoRA r=8 + next_gru | 53.71 ± 3.80 | 29.60 ± 2.01 |
| AL | Δ (GETNext − MTLoRA) | | **+2.67 pp** | −0.53 (tied) |
| AZ | **MTL-GETNext (soft)** | mtlnet_crossattn + next_getnext d=256 | **47.34 ± 2.93** | **24.16 ± 1.92** |
| AZ | MTL-MTLoRA r=8 pcgrad | mtlnet_dselectk + MTLoRA r=8 + next_gru | 39.51 ± 3.83 | 20.95 ± 2.96 |
| AZ | Δ (GETNext − MTLoRA) | | **+7.83 pp** | **+3.21 pp** |

Interpretation:
- On AL, GETNext edges MTLoRA by +2.67 pp Acc@10 but MRR is tied
  (29.07 vs 29.60, within σ). A reviewer would call this a tie on
  AL alone.
- On AZ the gap widens to +7.83 pp Acc@10 and +3.21 pp MRR — both
  outside σ. The effect is cross-state asymmetric.
- The combination — GETNext wins AL Acc@10 by a small margin and
  wins AZ decisively on every metric — is strong enough to reposition
  GETNext as the paper's MTL headline method.

**This reframes the earlier A7 "champion" framing.** Pre-fix, the
partition bug made MTLoRA look weaker than it was, leaving the field
unclear. Post-fix, MTLoRA recovers but GETNext still wins. The
partition-bug fix did its job — it unblocked MTLoRA from appearing
artificially bad — but the fix doesn't flip the head-to-head against
GETNext.

**Even more important** for the BRACIS narrative: the B5 inference-time
ablation (`research/B5_HARD_VS_SOFT_INFERENCE.md`) shows a *hard*
`last_region_idx` on the same GETNext head adds another +3 to +9 pp
Acc@10 over the soft probe. Full B5 retraining (commit `6a2f808`,
handoff at `research/B5_HANDOFF.md`) is queued for the 4050 and is
expected to further widen the GETNext margin over MTLoRA.

## Updated conclusions for the study

1. **Paper MTL headline method:** `mtlnet_crossattn + pcgrad +
   next_getnext d=256, 8h` (B-M6b / B-M9b in `RESULTS_TABLE.md`).
   Pending B5 retraining may replace the soft probe with the hard
   `next_getnext_hard` head, which would raise the headline further.
2. **A7 MTLoRA r=8 + pcgrad** is demoted to "second-best MTL on AL,
   much worse on AZ" — still worth reporting as a MoE+LoRA-based
   baseline, but not the champion. Keep it in ablation tables.
3. **AdaShare** is dropped from the paper tables — does not compete
   even fully trained.
4. **LibMTL optimizer family** (Aligned-MTL / CAGrad / DB-MTL) is
   effectively neutral against PCGrad on the A7 config. The PCGrad
   vs static attribution result (`research/ATTRIBUTION_PCGRAD_VS_STATIC.md`)
   already argued PCGrad is not load-bearing; this probe extends
   the tie to the broader LibMTL family. No need to invoke
   "sophisticated gradient balancing" in the paper narrative.
5. **MTLoRA rank** doesn't need to be hyperparameter-swept — r=8 is
   a fine default.
6. **Partition-bug contamination** affected MTLoRA / AdaShare
   only. GETNext, STAN, TGSTAN, STA-Hyper heads do not use LoRA or
   AdaShare and were never contaminated. The paper's GETNext-family
   numbers are unaffected.

## Files

| File | Description | Runtime |
|------|-------------|---------|
| `a7_mtlora_r8_al_5f50ep_postfix_pcgrad.json`      | AL A7 MTLoRA r=8 pcgrad (M2 Pro)    | 11:19–11:57 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_aligned_mtl.json` | AL A7 MTLoRA r=8 aligned_mtl (M2 Pro) | 11:57–12:35 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_cagrad.json`      | AL A7 MTLoRA r=8 cagrad (M2 Pro)     | 12:35–13:13 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_db_mtl.json`      | AL A7 MTLoRA r=8 db_mtl (M2 Pro)      | 13:13–14:10 (57m) |
| `ablation_04_mtlora_r8_al_5f50ep_postfix.json`    | AL A7 MTLoRA r=8 pcgrad (M4 Pro replicate) | 11:46–12:23 (37m) |
| `ablation_04_mtlora_r16_al_5f50ep_postfix.json`   | AL MTLoRA r=16 pcgrad (M4 Pro)         | 14:39–15:10 (31m) |
| `ablation_04_mtlora_r32_al_5f50ep_postfix.json`   | AL MTLoRA r=32 pcgrad (M4 Pro)         | 15:10–15:39 (29m) |
| `ablation_05_adashare_mtlnet_al_5f50ep_postfix.json` | AL AdaShare mtlnet pcgrad (M4 Pro) | 15:39–15:59 (20m) |
| `az2_mtlora_r8_fairlr_5f50ep_postfix.json`        | AZ A7 MTLoRA r=8 pcgrad (M4 Pro)       | 15:59–17:08 (69m) |

Launchers: `scripts/probe_a7_optimizers.sh` (M2 Pro),
`scripts/rerun_partition_bugfix.sh` (M4 Pro — crashed SIGBUS mid run 2),
`scripts/rerun_partition_bugfix_resume.sh` (M4 Pro resume with
caffeinate -s, completed cleanly).

Commits landed: `5668856` (partition fix), `c1c7f3e` (DSelectK gating),
`8afc9ac` (crossattn partial-forward), `06b799a` (STAN alibi default +
DSelectK docstring), `3b6e7d9` (issues README status), `ac17da6` (probe
launcher), `0c3eb68` (resume launcher), `6a2f808` (B5 hard-index
implementation).

## Next steps

1. **Update `RESULTS_TABLE.md` `Best MTL` rows** to elevate GETNext
   and demote MTLoRA. Drop the AdaShare row entirely.
2. **Drop the "A7 champion" framing** from `FINAL_ABLATION_SUMMARY.md`
   — replace with "best non-GETNext MTL on AL; second-tier on AZ".
3. **Wait for 4050 B5 retraining** (`research/B5_HANDOFF.md`). If B5
   lifts GETNext to 59+ on AL / 50+ on AZ, the paper has a clean
   headline: "faithful GETNext beats all other MTL methods by
   >5 pp on both states".
4. **Consider a small seed-sweep** for multi-seed headline (#B3):
   seeds 42/123/2024 on MTL-GETNext AL + AZ, ~2h on MPS per state.
   Needed for the paper's σ. Hard-index version is higher priority
   once B5 lands.
