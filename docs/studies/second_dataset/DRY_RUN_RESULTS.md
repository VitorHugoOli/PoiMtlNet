# second_dataset — Mac dry-run results (pipeline shakeout, NOT paper numbers)

**Status:** NYC complete · Istanbul running (2026-06-15) · **Machine:** Mac M2 Pro (MPS)

> ⚠ **These are DRY-RUN numbers, not paper results.** Per the methodology advisor + the
> PRE_FREEZE_PROGRAM DAG, the real validation (champion G + STL ceilings + Markov floor,
> 4 seeds) is **Level-4, post-freeze, on CUDA**. This Mac run is a **pipeline shakeout**: its
> purpose is (a) exercise the new corpus end-to-end (substrate → inputs → STL/MTL → floors),
> (b) verify the bit-parity / per-fold-`log_T` invariant, (c) get a *directional* read on
> whether the champion's behaviour transfers. Substrate = ResLN, 80 epochs (not the frozen
> 500-ep GCN/v14); trainings = 1 seed (42) × 5 folds × 50 ep. Absolute numbers undershoot and
> are NOT comparable across corpora or to paper cells. Compare **gap-to-ceiling / lift-over-floor**.

## NYC (regions = TIGER tracts, 1,912)

| Task | Floor | STL ceiling | MTL champion | MTL − STL |
|---|---|---|---|---|
| **next-category** (macro-F1) | majority ≈ 27.8% acc | 44.2% ±0.9 | **54.0% ±0.9** | **+9.8 pp** |
| **next-region** (Acc@10, indist) | Markov-1 **24.6%** | 30.3% ±0.5 | **29.8% ±0.4** | **−0.5 pp** |

(next-region also: STL Acc@1 10.5% / MRR 17.2%; MTL Acc@1 10.2% / MRR 16.9%.)

**Both defining champion behaviours replicate on NYC:**
- ✅ **MTL beats the STL category ceiling** (+9.8 pp) — the known MTL cat gain.
- ✅ **MTL matches the STL region ceiling** (−0.5 pp, within noise) **and clears the Markov-1
  floor** (+5.2 pp) — joint training does not sacrifice region. This is champion G's central claim,
  reproduced on a different LBSN source.

**Validation outcomes (the shakeout's purpose):** end-to-end pipeline runs on the new corpus;
folds align with structurally-built sequences (bit-parity holds, row count 30,155); per-fold
priors load (`[C4 per-fold log_T] ... seed42_fold*.pt`); STL next-region clears the Markov floor.

## ⚠ Recipe lesson (a real catch — read before Phase V)

An early MTL run gave next-region Acc@10 = **14.6%** (below the Markov floor) while STL-region got
30.3% on the *same* substrate. Root cause: **`--canon v16` (champion G) prepends
`--reg-head-param freeze_alpha=True alpha_init=0.0`** (intended for its `next_stan_flow_dualtower`
head). When the command overrode `--model`/`--reg-head` back to `mtlnet_crossattn` /
`next_getnext_hard`, that **`freeze_alpha` leaked onto `next_getnext_hard` and froze its log_T-prior
coefficient α at 0** — disabling the exact prior that makes region prediction work.

**Fix / rule:** for an MTL run with the `next_getnext_hard` (prior-based) region head on a custom
substrate, use **`--canon none` + the full explicit recipe** (crossattn, onecycle `--max-lr 3e-3`,
`--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`, unweighted, `--log-t-kd-weight 0.0`, active prior).
Do **not** mix `--canon v16` with manual `--reg-head` overrides. The exact v16 champion-G (dual-tower
on the v14 substrate) needs that substrate built first (Phase V).

The corrected MTL command (verified on NYC):
```bash
python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region \
  --state <city> --engine check2hgi --seed 42 --epochs 50 --folds 5 --batch-size 2048 \
  --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
  --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --cat-head next_gru --reg-head next_getnext_hard \
  --task-a-input-type checkin --task-b-input-type region \
  --no-reg-class-weights --no-cat-class-weights \
  --per-fold-transition-dir output/check2hgi/<city> --log-t-kd-weight 0.0
```

## Istanbul (regions = mahalle, 520 — PRIMARY, non-US)

| Task | Floor | STL ceiling | MTL champion | MTL − STL |
|---|---|---|---|---|
| **next-category** (macro-F1) | majority ≈ 27.0% acc | 50.4% ±0.8 | **59.4% ±0.5** | **+9.0 pp** |
| **next-region** (Acc@10, indist) | Markov-1 **52.5%** | 68.6% ±0.8 | **69.6% ±0.4** | **+1.0 pp** |

(next-region also: STL Acc@1 31.4% / MRR 44.1%; MTL Acc@1 31.5% / MRR 44.4%. Region Acc@10 is far
higher than NYC's because mahalle = 520 coarse regions vs NYC's 1,912 fine tracts — compare
gap-to-ceiling, not absolute. Substrate converged to best=1.26 at epoch 80, i.e. still descending —
slightly under-trained, fine for a dry run.)

## Cross-city verdict — the champion transfers (directional)

| | MTL beats STL **cat** ceiling | MTL matches STL **reg** ceiling | MTL beats Markov-1 floor |
|---|---|---|---|
| **NYC** (US, 1,912 tracts) | **+9.8 pp** | **−0.5 pp** | +5.2 pp |
| **Istanbul** (non-US, 520 mahalle) | **+9.0 pp** | **+1.0 pp** | +17.0 pp |

Both defining champion-G behaviours — the **MTL category gain** and **MTL region parity (no
sacrifice) above the floor** — replicate on a different LBSN source and on both a US and a non-US
city. Strong directional external-validity signal; paper numbers still require the frozen-substrate
CUDA Phase V.

## Caveats for Phase V (do NOT carry these numbers forward)
- Substrate is ResLN-80ep, not the frozen recipe → absolute numbers will move; only the relative
  pattern (MTL≈STL reg, MTL>STL cat, both>floor) is the takeaway.
- Single seed (42, the dev seed) — paper numbers use {0,1,7,100}.
- MPS, reduced epochs. Reproduce on CUDA with the frozen substrate for the actual Level-4 validation.
