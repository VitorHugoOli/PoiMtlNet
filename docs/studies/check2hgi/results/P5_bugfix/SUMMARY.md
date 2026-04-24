# P5 Bugfix — Post-`MTL_PARAM_PARTITION_BUG` Full Picture

**Base config:** `task_set=check2hgi_next_region`,
`task_a_input=checkin`, `task_b_input=region`, `max_lr=0.003`, seed 42,
5f × 50ep. Per-run variations call out the model/rank/optimizer axis.

**Purpose.** Five related questions in one push:
1. Does the MTLoRA-lift attributed to the A7 champion survive the
   partition-bug fix (commits `5668856` + `c1c7f3e`)?
2. Are LibMTL optimizers (Aligned-MTL / CAGrad / DB-MTL) competitive
   with PCGrad on this config?
3. Does the MTLoRA rank matter (r=8 / 16 / 32) once LoRA trains?
4. Does the partition-bug fix rescue AdaShare?
5. Does the A1 optimizer picture replicate on AZ (cross-state test)?

**Runtime.** 4 AL probe runs (M2 Pro, 2h55m) + 4 resume runs
(M4 Pro, 2h28m) + 1 local A7 replicate + 4 AZ probe runs
(M2 Pro, ~20h spread 2026-04-22 15:56 → 2026-04-23 11:44, CAGrad AZ
was a 14h outlier due to MPS memory pressure across folds)
= ~25h total wallclock across two machines.

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
lands ~9 pp Acc@10 below MTLoRA r=8. The "AdaShare neutral" framing in
pre-fix tables was masking a real architectural gap. Recommendation:
drop AdaShare as a paper row; it is not a competitive MTL sharing
mechanism on this task.

## A4 · Cross-state optimizer probe — A7 MTLoRA r=8 on AZ

The AL A1 probe was replicated on AZ with the same 4 optimizers.

| Optimizer   | reg acc (%)      | reg f1 (%)     | reg top5         | reg mrr          | cat f1          |
|-------------|------------------|----------------|------------------|------------------|-----------------|
| pcgrad      | **13.38 ± 1.63** | 5.22 ± 0.63    | **31.02 ± 3.03** | **22.27 ± 1.97** | **42.13 ± 0.91** |
| cagrad      | 12.77 ± 2.21     | 5.24 ± 0.72    | 30.93 ± 3.55     | 21.96 ± 2.83     | 41.24 ± 0.53    |
| db_mtl      | 12.53 ± 2.26     | **5.33 ± 0.76**| 30.77 ± 3.65     | 21.72 ± 2.86     | **42.44 ± 0.47** |
| aligned_mtl | 12.35 ± 1.64     | 5.26 ± 0.62    | 29.82 ± 3.07     | 21.19 ± 2.16     | 41.29 ± 0.59    |

(Second-machine M4 Pro replicate of AZ pcgrad at `az2_mtlora_r8_fairlr_5f50ep_postfix.json`:
reg acc 11.31 ± 2.90, Acc@10_indist 39.51 ± 3.83 — sits slightly below
the M2 Pro result, same ordering-noise story as AL pcgrad.)

**Cross-state consistency.** All four optimizers statistically tied on
both AL and AZ. 1σ bands:
- AL pcgrad [16.13, 18.83] · cagrad [16.27, 18.21] · db_mtl [15.99, 18.39] · aligned_mtl [15.98, 17.42]
- AZ pcgrad [11.75, 15.01] · cagrad [10.56, 14.98] · db_mtl [10.27, 14.79] · aligned_mtl [10.71, 13.99]

Mean-to-mean differences are within fold-level noise (0.7–2.3 pp);
relative ordering (pcgrad ≥ cagrad ≈ db_mtl ≥ aligned_mtl) is
consistent across both states but not statistically separable at
n=5 folds × 1 seed. A seed sweep would be needed to break the tie.

AZ region accuracy sits ~4 pp below AL — AZ has 1540 regions (vs AL
1109), harder classification problem. AZ category F1 sits ~5 pp above
AL (42 vs 36) — AZ has ~2× more check-ins so the 7-class category task
has more data. These scale effects are consistent across all four
optimizers.

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
   effectively neutral against PCGrad on the A7 config — on AL **and**
   on AZ. The PCGrad vs static attribution result
   (`research/ATTRIBUTION_PCGRAD_VS_STATIC.md`) already argued PCGrad
   is not load-bearing; this probe extends the tie to the broader
   LibMTL family across two states. No need to invoke "sophisticated
   gradient balancing" in the paper narrative.
5. **DB-MTL ties gradient-surgery on both states despite using a
   scalar-backward path.** Quantitative evidence that gradient-surgery
   vs scalar-weighting is not the distinguishing axis here.
6. **MTLoRA rank** doesn't need to be hyperparameter-swept — r=8 is
   a fine default.
7. **Partition-bug contamination** affected MTLoRA / AdaShare only.
   GETNext, STAN, TGSTAN, STA-Hyper heads do not use LoRA or AdaShare
   and were never contaminated. The paper's GETNext-family numbers are
   unaffected.

## Files

| File | Description | Runtime |
|------|-------------|---------|
| `a7_mtlora_r8_al_5f50ep_postfix_pcgrad.json`      | AL A7 MTLoRA r=8 pcgrad (M2 Pro)          | 11:19–11:57 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_aligned_mtl.json` | AL A7 MTLoRA r=8 aligned_mtl (M2 Pro)     | 11:57–12:35 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_cagrad.json`      | AL A7 MTLoRA r=8 cagrad (M2 Pro)          | 12:35–13:13 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_db_mtl.json`      | AL A7 MTLoRA r=8 db_mtl (M2 Pro)           | 13:13–14:10 (57m) |
| `ablation_04_mtlora_r8_al_5f50ep_postfix.json`    | AL A7 MTLoRA r=8 pcgrad (M4 Pro replicate)| 11:46–12:23 (37m) |
| `ablation_04_mtlora_r16_al_5f50ep_postfix.json`   | AL MTLoRA r=16 pcgrad (M4 Pro)            | 14:39–15:10 (31m) |
| `ablation_04_mtlora_r32_al_5f50ep_postfix.json`   | AL MTLoRA r=32 pcgrad (M4 Pro)            | 15:10–15:39 (29m) |
| `ablation_05_adashare_mtlnet_al_5f50ep_postfix.json` | AL AdaShare mtlnet pcgrad (M4 Pro)     | 15:39–15:59 (20m) |
| `az2_mtlora_r8_fairlr_5f50ep_postfix.json`        | AZ A7 MTLoRA r=8 pcgrad (M4 Pro)          | 15:59–17:08 (69m) |
| `a7_mtlora_r8_az_5f50ep_postfix_pcgrad.json`      | AZ A7 MTLoRA r=8 pcgrad (M2 Pro)          | 15:56–18:09 (2h12m, 2026-04-22) |
| `a7_mtlora_r8_az_5f50ep_postfix_aligned_mtl.json` | AZ A7 MTLoRA r=8 aligned_mtl (M2 Pro)     | 18:09–19:58 (1h49m) |
| `a7_mtlora_r8_az_5f50ep_postfix_cagrad.json`      | AZ A7 MTLoRA r=8 cagrad (M2 Pro)          | 19:58–09:53+1d (13h55m, MPS slowdown) |
| `a7_mtlora_r8_az_5f50ep_postfix_db_mtl.json`      | AZ A7 MTLoRA r=8 db_mtl (M2 Pro)           | 09:53–11:44 (1h51m, 2026-04-23) |

Launchers:
- `scripts/probe_a7_optimizers.sh` (M2 Pro — accepts `STATE=alabama|arizona`)
- `scripts/rerun_partition_bugfix.sh` (M4 Pro — crashed SIGBUS mid run 2)
- `scripts/rerun_partition_bugfix_resume.sh` (M4 Pro resume with
  caffeinate -s, completed cleanly).

Commits landed on `worktree-check2hgi-mtl`:
- `5668856` partition fix (MTLnet + MTLnetDSelectK `task_specific_parameters`)
- `c1c7f3e` DSelectK gating on legacy task_set
- `8afc9ac` crossattn partial-forward override
- `06b799a` STAN alibi default + DSelectK docstring
- `3b6e7d9` issues README status update
- `ac17da6` AL probe launcher
- `754394c` STATE parameterization
- `2fb55bb` AL probe results + v1 SUMMARY
- `0c3eb68` resume launcher
- `6a2f808` B5 hard-index implementation

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
