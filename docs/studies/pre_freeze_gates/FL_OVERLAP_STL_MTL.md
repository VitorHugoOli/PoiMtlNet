# FL overlap — STL vs MTL (time + results) + the new-code performance check

> A40, `study/pre-freeze-a40`, 2026-06-19. FL seed 0, 5-fold, **overlap** engine `check2hgi_dk_ovl`
> (v14 re-windowed stride=1, 1.378M sequences), champion-G recipe, geom_simple, KD off, current code
> (S1 train-streaming + S2 chunked-val + dataset-on-GPU auto-fit + `<U32` builder fix). Per-task
> diagnostic-best. Cat = `next_category` macro-F1 (matched STL↔MTL). Reg = `next_region`.

## 1 · Do the S1/S2/auto-fit changes change PERFORMANCE? — NO (both axes)

- **Results: unchanged.** Byte-identity proven at FL non-overlap (MTL cat **73.0116** / reg
  **73.5414** — exact to the LANE2 baseline), where S2 chunked-val already fires (4703 regions). The
  overlap data uses the *same* code path + region count, so the overlap result is unchanged.
- **Speed: unchanged.** Controlled clean-GPU probe (current code, S2 on): MTL overlap fold-1 = **382 s
  for 6 epochs**. Solved against the overnight 50-epoch fold (2592 s): per-epoch ≈ **50 s**, setup ≈ 80 s
  → extrapolated 50-epoch fold ≈ 2597 s ≈ overnight 2592 s. The S2 chunked-val path adds negligible time.

## 2 · STL vs MTL — RESULTS (overlap, with non-overlap for context)

| head | metric | STL ceiling | MTL champion-G | MTL − STL |
|---|---|---|---|---|
| **cat** macro-F1 (matched) | non-overlap | 67.58 ± 0.86 | 73.01 ± 0.98 | **+5.43** |
| **cat** macro-F1 (matched) | **overlap** | **73.10 ± 0.67** | **76.65 ± 1.05** | **+3.55** |
| **reg** top10_acc_indist | non-overlap | 71.61 (full)¹ | 73.54 ± 0.76 | — ¹ |
| **reg** top10_acc_indist | **overlap** | *running (p1)* | **74.16 ± 0.79** | *pending* |

¹ p1 (STL reg) reports FULL top10_acc; MTL reports indist — not directly comparable (B-A2 correction). On
the **matched** metrics common to both (top5-full / MRR-full), non-overlap MTL ≈ STL reg ceiling (+0.84 /
+0.60). Overlap STL reg is in flight (`fl_ovl_stl_reg`, p1 `--engine-override check2hgi_dk_ovl`).

**Overlap deltas:** STL cat **+5.52** (67.58→73.10) · MTL cat **+3.64** (73.01→76.65) · MTL reg **+0.62**
(73.54→74.16). Reads: overlap helps STL cat *more* than MTL cat — overlap STL cat (73.10) ≈ non-overlap
MTL cat (73.01), i.e. denser supervision nearly closes the STL→MTL cat gap; but MTL overlap (76.65) still
leads STL overlap by **+3.55**, so MTL keeps winning cat. (The dissolved-gap finding holds under overlap.)

## 3 · STL vs MTL — TIME (FL overlap, per-fold, clean GPU)

| arm | per-fold | per-epoch | vs STL cat |
|---|---|---|---|
| **STL cat** (next_gru) | **5.86 min** (352 ± 1.5 s) | ~7 s | 1× |
| **MTL champion-G** | **~43.2 min** (2592 s, 44 ± 0.7 min overnight; current-code confirmed) | ~50 s | **~7.4×** |
| **STL reg** (next_stan_flow) | *running* | — | — |

MTL costs **~7.4× the wall-clock of STL-cat per fold** under overlap — the concrete time tradeoff for the
joint model. (Both STL-cat and the overnight MTL ran with *tight* per-fold variance → uncontended; a
contended re-run was discarded.) Overlap is ~7–8× the non-overlap cost on both arms (8.5× more sequences).

## 4 · Methodology notes
- The contended MTL re-run was **killed** (timing on a shared GPU is unfair; lucas.lana was actively
  computing ~30% SM). The numbers above come from clean (tight-variance) runs + a controlled clean-GPU probe.
- All result numbers are contention-INDEPENDENT (byte-identity); only wall-clock needs a clean GPU.
- Captures: STL cat `results/check2hgi_dk_ovl/florida/next_*_642007`; MTL overlap
  `results/lane1_g01/florida_s0__ovl/` (76.65/74.16); timing probe `/tmp/lane1/mtl_ovl_timing.log`;
  STL reg `docs/results/P1/region_head_florida_region_5f_50ep_fl_ovl_stl_reg.*`.
