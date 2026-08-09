# v18 — event log

One line per event, newest at the bottom. Charter §8 deliverable.

- `2026-08-07` — v18 substrate built for all six states (forward-only graph + 4 elapsed-time node columns, `in_channels` 15). Readout equivalence proved: `prefix_forward_only` is an identity on a forward-only graph, verified to float32 epsilon over every window at 4 states.
- `2026-08-07` — first wave cells run under the *class-weighted* recipe inherited from v17. Later superseded.
- `2026-08-07` — AMP precision leak found and fixed (`PRECISION_CAVEAT.md`): a bare `export` in `cell_joint` persisted into later cells, so cat cells ran fp16-or-fp32 depending on resume state. Two results VOID. All cells now pin precision per-command via `env`.
- `2026-08-08` — retuning sweep, 103 arms. Both arms retuned so Δcat is not biased against MTL. Δcat pooled −0.363 → −0.105.
- `2026-08-08` — class weighting found to be **size-graded and sign-flipping**: alabama −1.203 (p=0.004) vs texas +1.556 (p=0.030).
- `2026-08-08` — **logit adjustment** (Menon ICLR'21, τ=0.5) beats both tuned class-weight endpoints: dedicated +2.86, MTL +3.18, all p ≤ 0.0014. τ=1.0 a flat null.
- `2026-08-09` — Fable review of `SWEEP_FINDINGS.md`. Verified every load-bearing number against the raw JSONs; caught two wording overreaches ("axis is closed", "confirmed at a large state"), both scoped.
- `2026-08-09` — §11 added to `SWEEP_FINDINGS.md`: 16 arms had been run but never written down (steps A/B/C, row 3b, reproducibility check).
- `2026-08-09` — **texas** dedicated, fold 0: logit adjustment transfers to a large state — 36.6226 vs 33.6127 (weights ON) and 35.1696 (weights OFF). +3.010 / +1.453. The size-tiered class-weight rule is dissolved.
- `2026-08-09` — **step D** finally run (planned 2026-08-08, never executed): florida MTL batch size is a **null** (geom span 0.078 over bs 8192/16384/32768). bs8192 kept everywhere.
- `2026-08-09` — **region + logit adjustment settled at two states**: alabama Acc@10 −1.841 (p=0.0002), istanbul −2.749 (p<0.0001), while macro-F1 *rises* (+0.377 / +0.671). Bayes-consistency for balanced error trading against the reported Acc@10. **τ=0 for region everywhere.** Confirmed the MTL region criterion never receives the offset (`mtl_cv.py:500`).
- `2026-08-09` — `FINAL_SETTINGS.md` **approved by the author**. 19 cat+joint sidecars moved to `v18_superseded_oldrecipe/`; the 10 region sidecars kept (recipe unchanged, and both fresh τ=0 arms reproduced the stored values *exactly*: AL 69.9956, IST 75.1563).
- `2026-08-09 06:14` — **regeneration launched** (`run_regen.sh`, seeds 0 then 1, n=10) at commit `e351d4b0`.
- `2026-08-09 06:20` — alabama s0 cat = **30.7654** (322 s), vs 28.0334 under the old recipe: **+2.73**, consistent with the +2.86 logit-adjustment gain measured in the sweep.
- `2026-08-09 06:28` — istanbul s0 cat = **35.3539** (802 s).
- `2026-08-09` — Opus audit of the live run. Two **blockers** found and fixed in `make_results.py`: (1) the end-of-wave aggregation called `make_results.py` without `score_all.py --write`, so it would have republished the **superseded** 2026-08-07 numbers under a fresh timestamp and today's SHA, silently and without crashing; (2) `PROVENANCE.md` hardcoded `cw 0.75` and no logit adjustment — a factual error about what produced the numbers. Also fixed a `None` crash that would have destroyed the whole report on any partially-failed wave.
- `2026-08-09 06:39` — alabama s0 joint cat=**30.6985** reg=**69.5928** (1167 s) — **reproduces the sweep arm `mla_alabama_tau0.5_cw0.50` to 0.0000 on both heads**, despite a different inductor cache directory.
- `2026-08-09 06:59` — istanbul s0 joint cat=**35.4227** reg=**75.2177** (1863 s) — vs the same sweep arm, +0.053 cat / +0.047 reg. Expected: `CLAUDE.md` records that under `--compile` the number is governed by the inductor cache/compile session (autotuning + reduction-order nondeterminism) and should be treated as within-fold-σ, not bit-reproducible. 0.05 pp against a fold-σ of ~1.7 (cat) / ~3.0 (reg) is far inside that. **Useful reproducibility statement for the dissertation: two independent runs of the identical recipe agree to ≤0.05 pp.**
