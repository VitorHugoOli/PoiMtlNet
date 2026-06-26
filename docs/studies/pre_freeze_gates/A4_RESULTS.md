# A4 — transductivity bound · running results (M4 Pro)

> Branch `study/pre-freeze-gates`. Disclosure gate (`evaluation_protocol_review.md §4.1`): the
> Check2HGI/v14 substrate trains on ALL check-ins (incl. validation-fold users), so val embeddings
> are shaped by val data. A4 quantifies the resulting downstream inflation by rebuilding the
> substrate per fold on **train users only** and re-evaluating.

## Method
- For each fold (StratifiedGroupKFold(seed) — the SAME split the reg eval uses), rebuild the v14
  substrate on train-users-only check-ins: `scripts/pre_freeze_gates/a4_build.py` →
  preprocess_check2hgi (train-only check-in graph = the transductive surface, rebuilt) → design_k
  (v14). The borrowed HGI Delaunay/POI2Vec **spatial** scaffolding is held fixed (POI-spatial priors,
  not a check-in-transductive channel — scoping noted). Train-only region embeddings are GEOID-remapped
  into the full region index space (zeros for regions absent from train — the inductive gap).
- Eval (`a4_eval.py`): per fold, train `next_stan_flow` on the train split, eval Acc@10 on the val
  split, using train-only vs full-corpus region embeddings — **same fold, same per-fold train-only
  log_T, same device (CPU)**. Inflation = full-corpus − train-only.
- **Scope: reg only.** Cat hits the inductive wall (val-user check-ins at train-unseen POIs have no
  embedding); deferred as the inductive-Check2HGI future-work the handoff anticipates. Reg is clean —
  regions are train-covered (AL inductive gap **2.6–2.8%** of regions absent from train).

## Validation
The full-corpus arm computed here (CPU) reproduces the A2 v14 reg cell (MPS): **AL full = 61.89 vs
A2 v14 AL reg 61.90** — confirms the bespoke eval matches the harness and that CPU≡MPS for this metric.
(A column-name bug that silently produced 0-dim embeddings was caught by this very check — fixed; raw
train-only embeddings + region maps are now persisted so a remap change never forces a rebuild.)

## Result
| state | n folds | full-corpus v14 Acc@10 | train-only v14 Acc@10 | **inflation** | inductive gap (regions absent from train) |
|---|---|---|---|---|---|
| AL (seed0) | 5 | 61.89 | 62.22 | **−0.33 pp** (within fold noise; per-fold Δ mixed sign) | 2.6–2.8% |
| AZ (seed0) | 5 | 53.08 | 53.06 | **+0.01 pp** (within fold noise; per-fold Δ mixed sign) | — |
| FL (seed0) | 5 | 69.97 | 70.08 | **−0.12 pp** (within fold noise) | — |

### Verdict — both axes measured; transductive inflation ≈ 0 (with scoped caveats)

**Reg (measured): inflation ≈ 0 (AL −0.33pp, within fold noise).** Note this is the **low-sensitivity
axis** — both arms share the same per-fold train-only log_T, and the α·log_T prior **dominates** reg
Acc@10 (shown here: the 0-dim-embedding bug scored 44.3 from log_T alone vs ~62 with embeddings; and
`embedding_eval/FINAL_SYNTHESIS.md`). So the reg null alone wouldn't generalize — which is why the
cat axis was measured directly.

**Cat (the substrate-driven axis where the lift lives) — MEASURED via in-coverage POI proxy: inflation
≈ 0 (AL +0.29pp, within noise; per-fold Δ ranges −2.97…+2.73).** Method: val sequences whose input
POIs are all train-covered (AL **66.8%** of val rows; 33% cold-POI remainder excluded), POI-level v14
embeddings, next_gru macro-F1, full-corpus vs train-only-fold POI embeddings, same rows/device.
Full 29.07 vs train-only 28.78. So the substrate's **POI representations are not transductively
inflated on cat** — the axis that carries Check2HGI's category lift shows no meaningful leakage on the
measurable subset.

⇒ **Disclosure-gate status: ON NULL on BOTH axes (reg measured ≈0; cat measured ≈0 via POI proxy).
One-paragraph defusal; headline numbers do NOT need re-anchoring.** Caveats to state in the paper:
(1) the cat measurement is a **POI-level proxy** on the **~67% in-coverage subset**, not the exact
check-in-level §0.1 setup; (2) the residual — the contextual per-visit component and the cold-POI
remainder — is not directly measurable on a transductive substrate (inductive wall) and is bounded by
the **inductive-Check2HGI future-work** (`research/future_work.md §2`). The evidence available points
to negligible inflation; the inductive build would close the residual.

| state | n folds | reg: full / train-only / **Δ** | cat (POI proxy, in-cov): full / train-only / **Δ** | cat in-cov frac |
|---|---|---|---|---|
| AL (seed0) | 5 | 61.89 / 62.22 / **−0.33 pp** | 29.07 / 28.78 / **+0.29 pp** | 66.8% |
| AZ (seed0) | 5 | 53.08 / 53.06 / **+0.01 pp** | 31.09 / 30.83 / **+0.27 pp** | 71.9% |
| FL (seed0) | 5 reg / 4 cat | 69.97 / 70.08 / **−0.12 pp** | 36.20 / 36.19 / **+0.00 pp** | 86.9% |

> **AZ added 2026-06-26** (`results/pre_freeze_gates/a4/a4_result_arizona_s0.json` + `a4_cat_result_arizona_s0.json`):
> third state confirms the null on both axes — reg +0.01 pp, cat +0.27 pp (in-cov 71.9%), both within fold noise,
> matching AL/FL. Per-fold cat Δ ranges −0.62…+0.91. The disclosure gate stays ON NULL across AL, AZ, FL.

**FL confirms AL on both axes**: reg −0.12pp (full 69.97 ≈ A2 v14 FL reg 70.09 → validated), cat
+0.00pp at **86.9%** in-coverage (denser state → the POI proxy covers most val sequences, so the cat
null is on the large majority of FL's data; FL cat n=4 folds — f0 lacked preserved POI embeddings).
Across AL+FL, transductive inflation is ≈0 on both heads → the disclosure gate is firmly ON NULL.

## Reproducibility artifacts (raw result JSONs)

The raw A4 score JSONs are now committed under [`docs/results/pre_freeze_gates/a4/`](../../results/pre_freeze_gates/a4/)
(closing the gap where only this writeup was tracked; `results/pre_freeze_gates/a4/*` is gitignored):

- **AZ** — `a4_result_arizona_s0.json` (region) + `a4_cat_result_arizona_s0.json` (category), from the
  2026-06-26 run on this box (the exact run that produced the AZ rows above).
- **AL / FL** — `a4_result_{alabama,florida}_s0.json` + `a4_cat_result_*` regenerated on this box and
  **verified to reproduce the committed numbers** (CPU≡MPS for this metric, as noted above). The large
  per-fold intermediates (`*_trainonly_*.parquet`, `*_maps.pkl`) stay gitignored — only the small score
  JSONs are tracked.
