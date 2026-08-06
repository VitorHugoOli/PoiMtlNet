# Probe: is the 0.75/0.25 split a leak artifact, and are the heads competing?

> **Asked 2026-08-06 (user).** The joint loss is `0.75·L_cat + 0.25·L_reg`, tuned on the **leaked**
> check2hgi. If the leak is what made the category head productive, that split may no longer be
> right. And separately: are the two heads' losses actually competing?
>
> **Answer: no on both counts.** The heads' gradients are orthogonal, not competing, and neither
> rebalancing the split nor applying gradient surgery recovers anything. The category collapse is
> **not** a multi-task optimization problem.

## Setup

Alabama, seed 0, 5 folds, 50 epochs, v18 engine (forward-only + elapsed time), fp32, `--compile`,
`--tf32`, per-head cat-lr 1e-3 — the frozen v17 recipe in every respect except the loss line. All
three arms run `MTL_TRAIN_DIAGNOSTICS=1`, enabling `mtl_cv._compute_gradient_cosine`: once per epoch,
on batch 0, `cos(∇L_cat, ∇L_reg)` over the **shared trunk parameters**, plus each task's gradient
norm → `<rundir>/diagnostics/fold*_diagnostics.csv`. 250 measurements per arm (50 epochs × 5 folds).

## Results

| arm | loss | cat | reg | cos mean | % epochs cos<0 | ‖∇cat‖ | ‖∇reg‖ | wall |
|---|---|---|---|---:|---:|---:|---:|---:|
| **A** | `static_weight` 0.75/0.25 | 27.3836 ± 1.691 | 69.6831 ± 3.002 | +0.00115 | 50% | 0.1507 | 0.1220 | 1340 s |
| **B** | `static_weight` 0.50/0.50 | 27.6771 ± 1.725 | 69.6873 ± 2.996 | +0.00148 | 46% | 0.1324 | 0.1238 | 1340 s |
| **C** | `pcgrad` (surgery) | 27.5797 ± 1.633 | 69.6063 ± 3.015 | +0.00159 | 46% | 0.1358 | 0.1249 | 1244 s |

Paired per-fold against A (same seed, same folds, n=5):

| contrast | Δcat | p | Δreg | p |
|---|---:|---:|---:|---:|
| B − A (equal split) | **+0.294** | 0.42 | +0.004 | 0.95 |
| C − A (surgery) | **+0.196** | 0.26 | −0.077 | 0.57 |

## 1 · The heads are orthogonal, not competing

Across all three arms and 750 measurements the shared-trunk cosine sits at **+0.001**, sd ≈ 0.012,
and never exceeds **|0.067|** in either direction. Roughly half the epochs are marginally negative
and half marginally positive — symmetric noise around zero, which is the signature of *orthogonal*
task gradients, not competing ones.

**This reproduces the pre-leak finding.** `scripts/mtl_improvement/plot_grad_cosine.py` records
"cosine ≈ 0 → the two tasks' gradients are orthogonal → there is no conflict", measured on the
**leaked** substrate. Removing the leak left the gradient geometry unchanged. The leak was therefore
never creating, nor masking, a gradient conflict.

## 2 · Rebalancing the split does nothing

Moving to 0.50/0.50 buys **+0.29 pp** category (p = 0.42) — about a fifth of the fold-to-fold sd
(1.69), and driven almost entirely by a single fold (the five per-fold diffs are +1.53, −0.26, +0.36,
+0.10, −0.26). Region does not move at all (+0.004, p = 0.95).

The weight *is* doing its mechanical job: lowering the category weight drops its shared-trunk
gradient norm from 0.151 to 0.132, pulling the cat:reg norm ratio from 1.24 to 1.07. The optimization
responds; the metric does not. So the split is not where the ~8.4 pp of lost category advantage went.

## 3 · Gradient surgery does nothing, for the reason the cosine predicts

PCGrad projects a task's gradient onto the normal plane of the other **only when their cosine is
negative**. Here the negative excursions are ~0.01–0.03 in magnitude, so there is essentially nothing
to project — and indeed C lands on A (+0.196 cat, p = 0.26; −0.077 reg, p = 0.57). This was recorded
as a prediction *before* the arm finished, and the arm confirmed it.

## 4 · A reproducibility result worth keeping

Probe arm A reproduces the wave-1 Alabama joint cell **exactly** — 27.3836 / 69.6831 to four decimals
— despite diagnostics being on, which disables inductor's donated-buffer optimization under
`--compile`. Two consequences: the diagnostics are numerically inert (so they can be enabled freely
in mechanism studies), and this configuration is bit-reproducible under `--compile` at this state,
which the canonical docs treat as not guaranteed.

## What this rules out, and what it doesn't

**Ruled out:** the category collapse is not a loss-balance artifact, not a gradient-conflict problem,
and not something gradient surgery addresses. Three independent interventions on that axis are all
null at n=5, and the mechanism measurement (cosine ≈ 0) explains *why* they are null rather than
leaving it as an empirical accident.

**Not ruled out:** n = 5, one seed, **one state, and the smallest one**. A +0.29 effect is unresolved
at this n, not zero. Alabama also has the shortest per-user histories of the six, and both the loss
split and any conflict could behave differently where the category signal is stronger (FL/CA/TX).

The reading these support is that the category signal is simply much weaker once a visit cannot see
its successor — the MTL machinery is dividing a smaller pie, not dividing it badly.

## Reproducing

```bash
bash docs/studies/closing_data/v18/run_lossweight_probe.sh   # resumable; per-arm sidecars
# sidecars: docs/results/closing_data/v18_probe/{A_cw075,B_cw050,C_pcgrad}.json
```
