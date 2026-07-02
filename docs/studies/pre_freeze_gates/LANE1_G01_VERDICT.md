# Lane 1 — G0.1 aligned-pairing advisory: VERDICT = NULL → v16 stands

> A40, `study/pre-freeze-a40`, 2026-06-18. The lone recipe-CHANGING closing_data P0 gate (inherited from
> mtl_improvement X1). Same-code A/B vs champion G at AL+FL seed0, v14 substrate, geom_simple, KD off.
> Lever = `--aligned-pairing` (NEW this session): one shared per-epoch permutation drives both MTL train
> loaders via a single joint loader, so cat-window k trains paired with reg-window k (same window) instead
> of the default independent shuffles (random cross-task pairing). Val was already aligned.

## Result (per-task diagnostic-best; cat=next_category macro-F1, reg=next_region Acc@10-indist)

| state | arm | cat-F1 | Δcat | reg@10 | Δreg |
|---|---|---|---|---|---|
| AL s0 | baseline | 52.38 | — | 64.34 | — |
| AL s0 | aligned  | 47.60 | **−4.77** | 63.51 | −0.84 |
| FL s0 | baseline | 73.01 | — | 73.54 | — |
| FL s0 | aligned  | 73.19 | **+0.17** | 73.54 | **±0.00** |

Both runs genuine (5/5 folds; AL aligned reg converges normally over epochs 20-29 — not degenerate).
FL baseline cat 73.01 == documented deterministic champion-G FL seed0 (73.012) → harness reproduces G.

## Verdict: NULL — no positive promotion. Recipe stays v16 / champion G.

- **FL (the scale state): NULL** (cat +0.17, reg ±0.00). Aligned training does **not** activate any
  cross-stream mixing benefit — exactly what X1's roll probe predicted, now confirmed on the *aligned-training
  counterfactual* the X1 banner flagged as the missing, non-circular test.
- **AL (small state): aligned HURTS cat (−4.77)**, reg ≈ flat. So random pairing is **not** a confound —
  at small data it acts as a **beneficial augmentation** for the shared cross-attn cat encoder (each cat
  window sees varied reg contexts → regularisation); forcing alignment removes that diversity and the
  encoder underperforms. The deployed champion-G (random pairing) is therefore *better* than aligned, not
  merely invariant to it.

## Disposition

- **NOT a v17 candidate.** No head improves under aligned pairing; FL null, AL negative.
- **"Champion G wins WITHOUT per-sample mixing" is fully earned** — and strengthened: random pairing is a
  mild *augmentation* at small states, not a confound. The X1 circularity (a model trained under random
  pairing is forced invariant, so the roll probe had no power against "mixing is learnable under aligned
  pairing") is now **closed**: aligned training was tested directly and unlocks nothing.
- **Binding-run note:** this is the **advisory** (current base, seed0). The formal closure is the
  pre-registered **binding** G0.1 on the **frozen** base, full {0,1,7,100}, 0.3 pp gate. The advisory is
  clearly null/negative (not borderline), so the binding run is expected null; only the binding run can
  re-pin recipe, and it should be run on the frozen (possibly-overlap) base to also clear the stride-1
  interaction caveat (denser supervision could in principle change the picture — but the AL-negative makes
  a positive surprise unlikely).

## Captures
`results/lane1_g01/{alabama,florida}_s0__{baseline,aligned}/`. Scorer: `scripts/pre_freeze_gates/lane1_score.py`.
Driver: `scripts/pre_freeze_gates/lane1_run.sh`. Implementation: `--aligned-pairing` →
`FoldCreator(aligned_pairing=)` → `_create_aligned_joint_loader`/`AlignedJointLoader` (`src/data/folds.py`) →
`FoldResult.joint_train_loader` → `mtl_cv.train_model(joint_train_loader=)`. Alignment validated by an
in-session A/B + a 5-fold smoke (both pass); a **landed regression test is P3** (the flag is default-inert,
so champion numbers are unaffected). No folds-cache path needed (champion G builds folds on the fly).

---

## Addendum 2026-07-02 — binding-grade resolution + mechanism (pipeline_audit)

The pre-registered binding run is **superseded** by the `pipeline_audit` pairing battery
([`../pipeline_audit/PAIRING_BATTERY.md`](../pipeline_audit/PAIRING_BATTERY.md)): AL, seeds
{0,1,7,100} × 5 folds, paired, on the **v17 board base** (`check2hgi_dk_ovl`, bs8192,
per-head LR, compiled) — the base any adoption decision would target today. Verdict:
**aligned − base = cat −3.025 ± 0.079 / reg −0.597 ± 0.075 (4/4 seeds negative both heads)
→ aligned pairing REFUTED as a champion candidate at AL**; the advisory's stride-1
interaction caveat is cleared (the deficit replicates ON the overlap base). Mechanism
resolved by an exactly-matched **deranged control** (same joint machinery/permutations/inits,
task-b rolled by 1): derange ≡ base (cat −0.04, reg +0.04) → the deficit is 100%
self-pairing SEMANTICS (own-window cross-read = overfit shortcut; random-other = beneficial
noise), 0% loader machinery / per-step diversity. Prior-session notes: the
`--aligned-pairing` CLI crash (missing `ExperimentConfig` field, this doc's P3 concern) was
fixed 2026-07-01 (branch `audit/pipeline-correctness-perf`, PR #57) with a regression test.
