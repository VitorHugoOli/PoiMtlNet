# Merge-design audit — index

Reorganized 2026-05-05 after FL leak-confound was confirmed; reorganized
again 2026-05-06 after K (Delaunay) was empirically falsified. **Start
at [`STATE.md`](STATE.md)** for the current big-picture verdict — this
INDEX is just a navigation map.

`MERGE_DESIGN_NOTES.md` and `SUMMARY.md` (older narrative docs) live
in this folder for archaeology only; they include leak-affected numbers
and pre-K hypotheses. Do not quote them as authoritative.

## What changed in the cleanup

- The Apr-25 `region_transition_log.pt` (single, leaky) inflated FL
  `next_getnext_hard` Acc@10 by ~13 pp. All FL design rows that used it are
  invalid as comparisons.
- Per-fold leak-free `region_transition_log_seed42_fold{1..5}.pt` were built
  for AL/AZ/FL.
- AL/AZ design results were already on per-fold leak-free logs — verified
  and retained.
- FL leak-free reruns are in progress; each per-design doc carries its own
  FL status block.

## Design ladder (cost-ordered)

| Design | Aim | Doc |
|---|---|---|
| A | Late-fusion (concat c2hgi+HGI inputs) | [DESIGN_A.md](DESIGN_A.md) |
| E | Per-task projector heads with reg stop-grad on POI2Vec-augmented c2hgi | [DESIGN_E.md](DESIGN_E.md) |
| B | POI2Vec injected at the POI pool boundary of c2hgi (frozen prior) | [DESIGN_B.md](DESIGN_B.md) |
| H | Learnable POI table at the pool boundary (no POI2Vec dependency) | [DESIGN_H.md](DESIGN_H.md) |
| I | LoRA r=8 on B (parameter-efficient variant) | [DESIGN_I.md](DESIGN_I.md) |
| J | H + anchor regulariser λ=0.1 toward POI2Vec | [DESIGN_J.md](DESIGN_J.md) |
| M | B + POI distillation loss toward POI2Vec | [DESIGN_M.md](DESIGN_M.md) |
| D | Heterograph (POIs + check-ins as first-class node types) | [DESIGN_D.md](DESIGN_D.md) |
| K | J + Delaunay POI-POI GCN (✗ falsified — K = J empirically) | [DESIGN_K.md](DESIGN_K.md) |

## Live levers — REFRAMED 2026-05-06

After the advisor review ([ADVISOR_REFRAMING.md](ADVISOR_REFRAMING.md)),
the previous lever order was discarded. Six merge variants converging
within ±0.1 pp on FL reg means feature-side interventions are
saturated. **Diagnose the consumer (POI2Region) and the head, not the
input.**

**Active ordered tests** (in execution order):

1. **Next-POI probe** — AL+AZ × {canonical, HGI, J}. May resolve half
   the research goal already.
2. **Reg-head ablation** — J FL with `next_gru` (no `log_T`). Real-vs-
   masked diagnostic.
3. **POI2Region hyperparam sweep** — J at AL × `num_heads × GCN-layers`.
   The user's exact hypothesis.
4. **Boundary-weight sweep** — J at AL × `alpha_{c2p,p2r}`. Secondary tuning.

**Deprioritised proposals** (kept for archaeology / fall-back):

| Lever | Status | Doc |
|---|---|---|
| 4 — POI2Vec at p2r | deprioritised (advisor: feature-side saturation) | [LEVER_4_POI2VEC_P2R.md](LEVER_4_POI2VEC_P2R.md) |
| 5 — distribution-level distill | deprioritised (advisor: feature-side saturation) | [LEVER_5_DIST_DISTILL.md](LEVER_5_DIST_DISTILL.md) |
| 6 — POI↔POI contrastive boundary | ✗ FALSIFIED 2026-05-07 (α-sweep complete; no α overcomes HGI) | proposal [LEVER_6_TWO_OUTPUT.md](LEVER_6_TWO_OUTPUT.md), result [LEVER_6_FINDINGS.md](LEVER_6_FINDINGS.md) |

## Cross-cutting docs

- [STATE.md](STATE.md) — big-picture status (start here)
- [HISTORY.md](HISTORY.md) — chronological log of every phase, hypothesis, and result (Phase 1-10 closed, **Phase 11 reopened 2026-05-07** at the substrate level)
- [PHASE_11_PLAN.md](PHASE_11_PLAN.md) — substrate-audit experiment plan (S1-S4, decision rules, outputs)
- [ADVISOR_REFRAMING.md](ADVISOR_REFRAMING.md) — 2026-05-06 advisor critique that produced the post-K test plan
- [T1_T2_T2quarter_FINDINGS.md](T1_T2_T2quarter_FINDINGS.md) — Tests 1, 2, 2½, 2¾, 3 results
- [T2_FINDINGS.md](T2_FINDINGS.md) — Tests 2 + 2½ standalone summary (kept for archaeology)
- [LEVER_6_FINDINGS.md](LEVER_6_FINDINGS.md) — Lever 6 α-sweep + final 6-falsification table
- [AUDIT_HGI_GAP.md](AUDIT_HGI_GAP.md) — diagnosis of what HGI has that
  c2hgi+merges lack; updated with K falsification
- [SPEEDUP_AUDIT.md](SPEEDUP_AUDIT.md) — orthogonal: how to make builds
  faster without changing quality (~15-25 % MPS via Tier 1 wins)

## AL/AZ leak-free dominance verdict

| Design | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 | Verdict |
|---|---:|---:|---:|---:|---|
| canonical c2hgi | 40.76 | 59.15 | 43.21 | 50.24 | baseline |
| HGI | 25.26 | 61.86 | 28.69 | 53.37 | reg upper-bound, cat catastrophic |
| A | 32.27 | 40.43 | 31.69 | 41.08 | ✗ FAIL both axes |
| E | 31.22 | 61.24 | 33.83 | 52.62 | ✗ FAIL cat (−9 pp) |
| B | 41.51 (+0.76) | 61.49 (+2.34) | 43.91 (+0.70) | 52.59 (+2.35) | ✓ DOMINANCE |
| H | 40.97 (+0.21) | 62.35 (+3.20) | 44.14 (+0.94) | 52.30 (+2.06) | ✓ DOMINANCE |
| I | 41.62 (+0.87) | 61.35 (+2.20) | 43.71 (+0.50) | 52.55 (+2.31) | ✓ DOMINANCE |
| J | 41.81 (+1.05) | 61.95 (+2.80) | 43.74 (+0.53) | 52.16 (+1.91) | ✓ DOMINANCE (AL reg strict, beats HGI) |
| M | 41.31 (+0.55) | 61.56 (+2.41) | 43.67 (+0.46) | 52.45 (+2.21) | ✓ DOMINANCE (cat strict 5/5 both states) |
| D | 72.88 (+32.12)* | 62.23 (+3.08) | 74.73 (+31.52)* | 52.95 (+2.71) | ⚠ cat lift is leak artifact |

*Source: `../../results/paired_tests/design_audit_al_az.json` (regenerated
2026-05-05 via `scripts/probe/finalize_design_ijm.py`).*

Wilcoxon one-sided greater at p=0.0312 (n=5 floor) for cat strict wins.
TOST non-inferiority at δ=2 pp for cat dominance gate.

## FL status (in progress)

FL reg numbers in the original notes were obtained with the leaky single
`region_transition_log.pt`. The fresh canonical (82.45) and design-B/I/J/M
(82.2-82.4) are all from the same leaky path, so the **relative**
ordering is unaffected, but the absolute level should not be quoted.

Leak-free FL canonical: **a10 = 0.6960 (3-fold checkpoint)**, matching the
historical leak-free canonical. Leak-free FL B, I, J, M reg reruns are
launched against per-fold log_T; results land in
`docs/studies/check2hgi/results/P1/region_head_florida_*_leakfree.json`
and propagate into the per-design docs.

FL cat (`next_gru` 5f×50ep) is not run yet; per AL/AZ the cat axis is
preserved by B/H/I/J/M within ±1 pp of canonical, so the dominance
prediction at FL is non-inferiority pending verification.

## How to regenerate

```bash
# AL/AZ leak-free paired tests
python scripts/probe/finalize_design_ijm.py \
  --designs b d e h i j m --states alabama arizona \
  --out design_audit_al_az.json

# FL leak-free reg rerun (one design at a time, MPS)
python scripts/p1_region_head_ablation.py \
  --state florida --heads next_getnext_hard \
  --folds 5 --epochs 50 --seed 42 --input-type region \
  --region-emb-source check2hgi_design_<x> \
  --override-hparams d_model=256 num_heads=8 \
    transition_path=output/check2hgi/florida/region_transition_log_seed42_fold1.pt \
  --per-fold-transition-dir output/check2hgi/florida \
  --tag STL_FLORIDA_design_<x>_reg_gethard_pf_5f50ep_leakfree
```
