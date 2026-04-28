# STAN on HGI substrate — substrate ablation for the region head

**Date:** 2026-04-25. **Tracker:** F36. **Scope:** AL + AZ (FL/CA/TX deferred per user). **Status:** ✅ replicated across both states and both heads.

> **Phase-1 reframing (2026-04-27).** This doc's headline ("HGI > Check2HGI for the region task") is based on the **STAN** sequence head. Phase 1 added the **matched MTL reg head** (`next_getnext_hard` = STAN + α·log_T graph prior) as a new probe. Under that head, the substrate preference **reverses** at AL+AZ (AL tied within σ + TOST non-inferior at δ=2 pp; AZ +2.34 pp Acc@10, p=0.0312, 5/5 folds positive). The STAN-head finding is therefore **head-coupled to STAN's POI-stable preference**, not pure substrate quality. CH15 was reframed accordingly — see `baselines/check2hgi_v_hgi/phase1_verdict.md §2.2` and `CLAIMS_AND_HYPOTHESES.md §CH15` reframing. The STAN-head data in this doc is preserved as a head-sensitivity probe, not refuted.

## Summary

Substrate-swap ablation: same heads (`next_stan`, `next_gru`), same fold protocol (StratifiedGroupKFold(5), seed 42, 50 epochs, OneCycleLR max_lr=3e-3, AdamW lr=1e-4 wd=0.01, batch 2048), same row alignment — only the region embeddings fed into the 9-step input window change (Check2HGI → HGI).

**HGI substrate is universally better than Check2HGI for the region task** by +2.6 to +4.8 pp Acc@10. The pattern holds on both AL and AZ, and for both STAN and GRU heads.

This **reverses CH16's direction on the region task**: CH16 established Check2HGI > HGI on next-category macro-F1, but for next-region the substrate preference flips.

## Results — 5-fold × 50 epoch

### Alabama (10 K rows, 1 109 regions)

| Head | Substrate | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| `next_stan` | Check2HGI (prior) | 24.64 ± 1.38 | — | 59.20 ± 3.62 | 36.10 ± 1.96 | 6.34 |
| **`next_stan`** | **HGI (this study)** | **27.40 ± 2.14** | **51.87** | **62.88 ± 3.90** | **39.02 ± 2.66** | 8.77 ± 0.94 |
| Δ | | **+2.76** | — | **+3.68** | **+2.92** | +2.43 |
| `next_gru` | Check2HGI (prior) | 23.60 ± 1.86 | — | 56.94 ± 4.01 | 34.57 ± 2.34 | — |
| **`next_gru`** | **HGI (this study)** | **26.41 ± 2.46** | **51.18** | **61.70 ± 3.46** | **38.06 ± 2.65** | 9.23 ± 1.28 |
| Δ | | **+2.81** | — | **+4.76** | **+3.49** | — |

### Arizona (26 K rows, 1 547 regions)

| Head | Substrate | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| `next_stan` | Check2HGI (prior) | 24.48 ± 2.29 | 43.07 | 52.24 ± 2.38 | 33.70 ± 2.36 | — |
| **`next_stan`** | **HGI (this study)** | **26.26 ± 2.38** | **45.76** | **54.86 ± 2.84** | **35.87 ± 2.47** | 7.76 ± 0.85 |
| Δ | | **+1.78** | +2.69 | **+2.62** | **+2.17** | — |
| `next_gru` | Check2HGI (prior) | 23.63 ± 2.04 | 40.57 ± 2.39 | 48.88 ± 2.48 | 32.13 ± 2.21 | — |
| **`next_gru`** | **HGI (this study)** | **25.20 ± 1.86** | **44.45** | **52.84 ± 2.65** | **34.61 ± 2.11** | 7.89 ± 0.33 |
| Δ | | **+1.57** | +3.88 | **+3.96** | **+2.48** | — |

### Cross-state, cross-head pattern

| State | Head | Δ Acc@10 (HGI − Check2HGI) | Δ MRR |
|---|---|---:|---:|
| AL | STAN | +3.68 | +2.92 |
| AL | GRU | +4.76 | +3.49 |
| AZ | STAN | +2.62 | +2.17 |
| AZ | GRU | +3.96 | +2.48 |
| **FL (added 2026-04-26)** | **STAN** | **+0.96** | **+0.75** |

Direction is uniform (HGI wins all 5 cells); magnitude is **monotone-decreasing in data scale** — the HGI advantage shrinks from +3–5 pp at AL to ~+1 pp at FL (159 K rows). FL still favours HGI but the gap is narrow enough that a multi-seed n=3 confirmation would tighten the conclusion. See `STAN_THREE_WAY_COMPARISON.md` for the full cross-state pattern including substrate-vs-faithful-vs-Markov.

**Source JSONs:**
- `results/P1/region_head_alabama_region_5f_50ep_STAN_HGI_al_5f50ep.json`
- `results/P1/region_head_arizona_region_5f_50ep_STAN_HGI_az_5f50ep.json`
- `results/P1/region_head_florida_region_5f_50ep_STAN_HGI_fl_5f50ep.json`
- `results/P1/region_head_florida_region_5f_50ep_STAN_CHECK2HGI_fl_5f50ep.json` (FL Check2HGI side, added 2026-04-26 since prior STAN-on-Check2HGI had only AL/AZ)

## Interpretation

CH16's substrate finding (Check2HGI > HGI) is **task-specific to next-category, not universal across heads**:

- **Next-category** consumes the check-in's embedding directly — Check2HGI's per-checkin contextual encoding (which absorbs neighbour POIs in the local trajectory window) provides discriminative signal that HGI's POI-only embedding lacks.
- **Next-region** is a coarser target (~1.1–1.5 K vs 7 classes) consumed via a sequence of region embeddings. HGI's region embeddings — built from POI-level graph signal and aggregated to the region — appear to retain more of the region-discriminative geometry than Check2HGI's region embeddings, which were built atop check-in-contextual representations.

Plausible mechanism: Check2HGI's contextual aggregation **smooths** the region embedding space (because it averages multiple check-ins per POI/region), while HGI preserves a sharper region-level representation suited to a region-prediction head. This is consistent with the +3–5 pp gain being uniform across heads — it's a substrate property, not a head-substrate-interaction.

## Implications

1. **Paper baseline table.** The STL STAN ceiling for AL/AZ should be reported on **HGI** substrate, not Check2HGI. New numbers:
   - AL STL STAN: 59.20 (Check2HGI) → **62.88 (HGI)** Acc@10.
   - AZ STL STAN: 52.24 (Check2HGI) → **54.86 (HGI)** Acc@10.
2. **F21c gap widens.** Matched-head STL `next_getnext_hard` already beats MTL-B3 by 8.77 pp on AL Check2HGI (`F21C_FINDINGS.md`). Adding HGI substrate to that head likely lifts the STL further, widening the STL>MTL gap and reinforcing the paper-reframing pressure (F21c §Interpretation A/B/C).
3. **CH16 is task-conditional.** The "Check2HGI > HGI" claim must be qualified: holds for next-category, **reverses for next-region**. Update `CLAIMS_AND_HYPOTHESES.md` CH16 wording.
4. **Two follow-ups become attractive:**
   - **F36b** — `next_getnext_hard` on HGI substrate, AL + AZ. Closes the matched-head STL story on both substrates.
   - **F38** — MTL+STAN with HGI region-emb input, AL + AZ. Tests whether the substrate gain transfers into MTL.

## Caveats

- **17-row label-noise floor.** HGI's `next.parquet` and Check2HGI's `next.parquet` disagree on `next_category` for 17/12709 AL rows (0.13%). Userid order is row-aligned and `poi_8` lookups match exactly (verified spot-check). The region label is sourced via `sequences_next.parquet`'s `target_poi`, so the next_category mismatch does not affect this experiment's region target. Documenting for transparency.
- **AL+AZ only.** FL/CA/TX deferred per scoping decision. The +3–5 pp gain is consistent enough across both states that we expect replication, but treat the headline-state numbers as projected until measured.
- **Same fold ordering, same input pipeline.** This is the cleanest possible substrate-swap — only the embedding values differ; row count, fold split, target labels, optimiser are bit-identical.

## Reproduction

```bash
PY=/Users/vitor/Desktop/mestrado/ingred/.venv/bin/python
caffeinate -i env \
  PYTHONPATH=src DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred/data \
  OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  "$PY" -u scripts/p1_region_head_ablation.py \
    --state {alabama,arizona} --heads next_stan next_gru \
    --folds 5 --epochs 50 --input-type region \
    --region-emb-source hgi \
    --tag STAN_HGI_{al,az}_5f50ep
```

Wall time: AL 2.5 min, AZ 5.5 min on M4 Pro MPS.
