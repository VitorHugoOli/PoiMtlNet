# P2-screen Leaderboard — AL, 1f × 15ep, per-task modality, default next_mtl head

**Config:** 5 archs × 5 losses = 25 configs planned; 21 completed (gradnorm on CGC/MMoE/DSelectK/PLE skipped after CGC+gradnorm deadlocked in disk syscall for 25 min; gradnorm on base FiLM completed but collapsed to 0.5% region Acc@10).

**Single-task baselines (for Δm):**
- AL next-category F1 (Check2HGI, 5f × 50ep): **39.16%** (from P1.5b)
- AL next-region Acc@10 (Check2HGI TCN head, 5f × 50ep): **56.11%** (from P1)
- Note: P2 MTL uses `next_mtl` transformer head for region, which caps region numbers below TCN.

## Leaderboard (21 configs, sorted by framework joint_score)

| Rank | Arch | Loss | Cat F1 | Reg A@1 | Reg A@10 | Reg MRR | Joint | Δm |
|:---:|---|---|---:|---:|---:|---:|---:|---:|
| **1** | mtlnet_dselectk | pcgrad | 36.32% | 5.40% | 32.49% | 13.60% | **18.70%** | −24.67% |
| **2** | mtlnet_mmoe | nash_mtl | 35.89% | 6.81% | 33.74% | 15.45% | **18.61%** | −24.10% |
| **3** | mtlnet_dselectk | cagrad | 36.23% | 4.83% | 30.54% | 12.67% | 18.56% | −26.53% |
| 4 | mtlnet_dselectk | equal_weight | 36.10% | 4.86% | 32.25% | 13.21% | 18.56% | −25.17% |
| 5 | mtlnet_mmoe | pcgrad | 35.98% | 6.23% | 31.95% | 14.66% | 18.52% | −25.59% |
| 6 | mtlnet | cagrad | 36.22% | 5.66% | 27.39% | 12.68% | 18.52% | −29.34% |
| 7 | mtlnet_mmoe | equal_weight | 35.79% | 6.39% | 32.25% | 14.81% | 18.46% | −25.56% |
| 8 | mtlnet_mmoe | cagrad | 35.92% | 5.57% | 31.62% | 13.80% | 18.36% | −25.95% |
| 9 | mtlnet | nash_mtl | 35.43% | 6.09% | 27.87% | 12.96% | 18.23% | −29.93% |
| 10 | mtlnet_dselectk | nash_mtl | 35.86% | 4.44% | 28.00% | 11.84% | 18.22% | −29.25% |
| 11 | mtlnet | pcgrad | 35.82% | 4.77% | 21.93% | 10.38% | 18.21% | −34.72% |
| 12 | mtlnet | equal_weight | 35.46% | 7.00% | 28.23% | 13.62% | 18.16% | −29.57% |
| 13 | mtlnet_cgc | equal_weight | 34.58% | 7.51% | **34.67%** | 15.80% | 18.02% | −24.96% |
| 14 | mtlnet_cgc | cagrad | 34.50% | **7.63%** | **34.68%** | 15.64% | 17.93% | −25.05% |
| 15 | mtlnet_cgc | pcgrad | 34.47% | 7.54% | 34.53% | 15.73% | 17.93% | −25.22% |
| 16 | mtlnet_cgc | nash_mtl | 34.41% | **8.04%** | **34.93%** | **16.40%** | 17.91% | −24.94% |
| 17 | mtlnet_ple | equal_weight | 34.11% | 4.41% | 24.43% | 10.66% | 17.25% | −34.68% |
| 18 | mtlnet_ple | cagrad | 33.48% | 4.59% | 24.35% | 10.98% | 16.94% | −35.55% |
| 19 | mtlnet_ple | nash_mtl | 33.33% | 4.97% | 25.47% | 11.52% | 16.92% | −34.75% |
| 20 | mtlnet_ple | pcgrad | 33.29% | 3.30% | 25.02% | 10.43% | 16.82% | −35.20% |
| 21 | mtlnet | gradnorm | **25.77%** | **0.03%** | **0.51%** | 0.97% | 12.89% | −66.64% |

Skipped (predicted to deadlock): mtlnet_cgc+gradnorm (confirmed hung), mtlnet_mmoe+gradnorm, mtlnet_dselectk+gradnorm, mtlnet_ple+gradnorm.

## Findings

1. **Expert-gating beats FiLM.** MMoE and DSelectK sweep the top-5. Base FiLM (`mtlnet`) sits 6-12th. Consistent with the fusion study's prior finding.

2. **Arch is the dominant factor, not optimizer.** Within-arch loss spread is small (~0.5 pp joint across DSelectK's 4 losses; ~0.2 pp across MMoE's 4). Picking the arch matters; picking the optimizer is a tweak.

3. **CGC is a region specialist.** Best region Acc@10 (34.68–34.93% across all 4 losses) and lowest category F1 (34.41–34.58%). Interesting asymmetry — CGC's task-specialist experts favor the region side disproportionately. Not ideal for bidirectional headline; worth reporting as a design finding.

4. **Category F1 is saturated at ~35–37% across the grid**, below the STL baseline of 39.16%. At 15 epochs and with MTL dilution, category F1 is consistently *lower* than single-task on AL. This is evidence that **AL MTL does not improve category** — the dev state's category task has no headroom for MTL to fill. CA/TX may differ (less data per class → more room).

5. **gradnorm is unreliable on this task.** Even on base FiLM (where it didn't deadlock), gradnorm collapsed to 0.5% region Acc@10 — a complete learning failure. Plus the hang on CGC. Drop gradnorm from P2-promote/confirm.

6. **PLE is the weakest gating arch** (17.25–16.82% joint). Its progressive-routing expert hierarchy is overkill for 2 tasks + small dataset. Drop.

## Recommendations for P2-promote (next stage)

Promote the top 5–6 configs to 2-fold × 15-epoch confirmation:

| Priority | Arch | Loss | Rationale |
|---|---|---|---|
| 1 | mtlnet_dselectk | pcgrad | Screen #1 (top joint); DSelectK is dominant |
| 2 | mtlnet_mmoe | nash_mtl | Screen #2; MMoE+nash is strongest MMoE |
| 3 | mtlnet_dselectk | equal_weight | Test whether DSelectK needs a balancing optim |
| 4 | mtlnet_mmoe | pcgrad | Cross-check: does pcgrad also help MMoE? |
| 5 | mtlnet_cgc | nash_mtl | Region-specialist — understand if its cat F1 gap closes with more epochs |
| 6 (control) | mtlnet | nash_mtl | FiLM baseline for the paper's ablation table |

## Known caveats (for paper write-up)

- **Head:** these screen numbers use the default `next_mtl` transformer head for both task slots. P1 showed the transformer collapses standalone on region. The MTL's shared backbone + task-b encoder lifts it to ~30–35% region Acc@10 here, but this is still capped vs the TCN/GRU head's 56%. Final P3 headline must swap to GRU region head (~50-LOC refactor needed).
- **Epochs:** 15 epochs is under-trained for convergence. The inversion (cat F1 35–37% < STL 39.16%) is partly a training-duration artifact. P2-confirm at 50 epochs should close this gap.
- **Single fold × seed:** screen is single-fold, seed 42. Variance not characterized; promote+confirm will handle it.
