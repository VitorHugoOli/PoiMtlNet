# Gap A — closure (2026-04-30)

Final closure of the next-region external-baselines comparison for the check2hgi study, scoped to AL/AZ/FL after a feasibility analysis showed CA/TX faithful runs were not justifiable on the compute budget.

Supersedes the mid-flight `GAP_A_AUDIT_SNAPSHOT_20260430.md` snapshot (kept for historical reference).

## Final scope

| Axis | States covered | Rationale |
| --- | --- | --- |
| **Substrate comparison** (HGI vs Check2HGI) | AL/AZ/FL/CA/TX (5 states) | Already closed via STAN-STL across all 5 states. This is the paper's primary contribution. |
| **Faithful published-architecture comparison** | AL/AZ/FL (3 states) | STAN-faithful + REHDM-faithful both closed at these 3 states. CA/TX faithful runs were not feasible: STAN-faithful at R=8501 regions is ~5–7 h/fold sequential; REHDM-faithful at CA/TX scale projects to ~75–120 h/state. |

The paper's "5-state external-baseline comparison" claim therefore covers the substrate axis (5 states, controlled architecture). The protocol-purity faithful axis is reported at 3 states (AL/AZ/FL) — sufficient for trend confirmation.

## Final next-region table — Acc@10 (mean ± σ)

| Baseline | Variant | AL | AZ | FL | CA | TX |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Markov-1-region (floor) | — | 47.01 ± 3.55 | 42.96 ± 2.05 | 65.05 ± 0.93 | 52.09 ± 0.80 | 54.94 ± 0.46 |
| STAN | `faithful` | 34.46 ± 3.88 | 38.96 ± 3.41 | 65.36 ± 0.69 | ⚪ | ⚪ |
| STAN | `stl_check2hgi` | 59.20 ± 3.62 | 52.24 ± 2.38 | 72.62 ± 0.52 | 58.82 ± 1.04 | 61.35 ± 0.36 |
| **STAN** | `stl_hgi` | **62.88 ± 3.90** | **54.86 ± 2.84** | **73.58 ± 0.43** | **60.45 ± 0.97** | **62.70 ± 0.37** |
| **ReHDM** † | `faithful` | **66.06 ± 0.98** | 54.65 ± 0.77 | 65.68 ± 0.26 § | ⚪ | ⚪ |
| ReHDM | `stl_check2hgi` | 26.22 ± 1.58 | 23.24 ± 1.27 | 38.74 ± 0.49 | ⚪ | ⚪ |
| ReHDM | `stl_hgi` | 42.78 ± 2.82 | 34.00 ± 3.02 | 54.49 ± 0.32 | ⚪ | ⚪ |

⚪ = intentionally out of scope (compute-infeasible / coverage already adequate via substrate axis).
† REHDM faithful uses paper protocol (chronological 80/10/10 + 24h sessions + 5 seeds). σ is inter-seed.
§ FL REHDM faithful uses `batch_size=128 + lr/max_lr scaled 4×` (linear scaling rule from paper b=32). Validated on AL/AZ — see "Optimization journey" below.

## Cells closed in this campaign (2026-04-30)

| Cell | Result | Tag |
| --- | --- | --- |
| CA + TX floors (majority, top-K, markov-1, markov-K-cat) | ✅ | `compute_simple_baselines` |
| CA + TX MHA+PE faithful 5f×11ep | ✅ | `FAITHFUL_MHAPE_{state}_5f11ep` |
| CA + TX POI-RGNN faithful 5f×35ep | ✅ | `FAITHFUL_POIRGNN_{state}_5f35ep` |
| CA + TX STAN-STL (`stl_check2hgi`, `stl_hgi`) | ✅ | `STL_{CA,TX}_{check2hgi,hgi}_stan_5f50ep` |
| AZ REHDM `stl_check2hgi` 5f×50ep | ✅ | `REHDM_STL_STUDY_v3_az_check2hgi_5f50ep` |
| FL REHDM `stl_check2hgi` 5f×50ep | ✅ | `REHDM_STL_STUDY_v3_fl_check2hgi_5f50ep` |
| FL REHDM `stl_hgi` 5f×50ep | ✅ | `REHDM_STL_STUDY_v3_fl_hgi_5f50ep` |
| FL REHDM `faithful` 5seeds×50ep (b=128) | ✅ | `REHDM_BS128_fl_5seeds_50ep` |

## Optimization journey

Three rounds of optimization were applied during the campaign, each validated for quality before scaling.

### 1. STAN faithful — abandoned at CA/TX scale

Smoke on AL OPT2 (TF32 + bf16 + GPU-resident tensors + cheap-metric per epoch + full-metric at best epoch): fold 0 Acc@10 = 34.78% vs ref 34.46% mean (within 1σ). Speed unchanged at AL (~8 min/fold) — AL is overhead-bound by Python kernel-launch overhead at small workload, not GPU-bound.

Sequential CA OPT2 launched: at 1h09m on fold 0, no fold completion emitted. Extrapolated 5-fold ETA ~5–7h for CA, similar for TX. Compared with the substrate-axis coverage already closed at 5 states via STAN-STL, the marginal value was insufficient to justify further compute. Killed and noted as ⚪ scope-out.

Code-level optimizations preserved on `worktree-check2hgi-mtl` for future use:
- `research/baselines/stan/train.py` — TF32, bf16 autocast, GPU-resident fold tensors, cheap epoch metrics.

### 2. REHDM `stl_*` — already efficient at study protocol

Smoke on AL `stl_check2hgi` 5f×2ep: ~1.7s/epoch steady, 5f×50ep extrapolates to ~7 min total. No optimization needed — REHDM-STL is already efficient at study protocol because the precomputed-collaborator-pool collate is cheap and the model is small.

### 3. REHDM `faithful` — linear LR scaling, validated on AL/AZ

**Bottleneck.** At paper batch_size=32 with 12 DataLoader workers, GPU sat at 35% utilization. Diagnosis: per-batch GPU work was ~6 ms while per-batch host overhead (launch + H2D + scheduler step + grad-finite checks across 2.54M params) was ~10 ms. Adding more workers did not help — the GPU was starving on small kernel launches, not waiting on data.

**Resolution.** Increase batch_size to amortize per-batch overhead. Linear LR scaling rule applied: `b=32, lr=5e-5, max_lr=5e-4` → `b=128, lr=2e-4, max_lr=2e-3` (4× scaling).

**Validation on AL + AZ (2 seeds × 50ep) before launching FL:**

| State | b=128 (2 runs) | Reference b=32 (5 runs, paper-protocol) | Δ Acc@10 | Δ MRR |
| --- | --- | --- | --- | --- |
| AL | Acc@10 65.85 ± 1.53, MRR 37.38 ± 0.74 | 66.06 ± 0.98, 37.83 ± 1.17 | −0.21 (within 1σ) | −0.45 (within 1σ) |
| AZ | Acc@10 54.94 ± 0.12, MRR 31.72 ± 0.84 | 54.65 ± 0.77, 30.96 ± 0.36 | +0.29 (within 1σ) | +0.76 (within 1σ) |

Both within 1σ of paper-batch references. Quality preserved.

**FL launched with validated b=128 + 4× LR scaling.** Per-epoch dropped from 67 s (b=32, 12 workers) to ~22 s (b=128). 5 seeds × 50 epochs completed in ~92 min vs the original ~4.5h projection.

## Provenance — JSONs that back these numbers

```
docs/studies/check2hgi/results/baselines/
├── REHDM_BS128_al_2runs_50ep_run{0,1}.json + summary  (validation, AL)
├── REHDM_BS128_az_2runs_50ep_run{0,1}.json + summary  (validation, AZ)
├── REHDM_BS128_fl_5seeds_50ep_run{0..4}.json + summary  (campaign, FL)
├── REHDM_STL_STUDY_v3_az_check2hgi_5f50ep_fold{0..4}.json + summary
├── REHDM_STL_STUDY_v3_fl_check2hgi_5f50ep_fold{0..4}.json + summary
├── REHDM_STL_STUDY_v3_fl_hgi_5f50ep_fold{0..4}.json + summary
├── faithful_mha_pe_{california,texas}_5f_11ep_*.json
├── faithful_poi_rgnn_{california,texas}_5f_35ep_*.json
└── (STAN STL via P1 region_head_ablation, archived under results/P1/)
```

State-aggregated views (built by `scripts/_gap_a_finalize.py`):
```
docs/studies/check2hgi/baselines/{next_category,next_region}/results/{california,texas}.json
docs/studies/check2hgi/baselines/next_region/results/florida.json (manually augmented with rehdm.faithful block)
```

Tarball of gitignored artifacts (preprocessed REHDM/STAN ETL outputs + full training logs) at `/teamspace/studios/this_studio/gap_a_artifacts_20260430.tar.gz` (initial 99 MB version; refreshed below).

## Run environment

- Lightning Studio H100 80 GB (single GPU, dedicated)
- scikit-learn 1.8.0 (PR #32540 fix in `StratifiedGroupKFold(shuffle=True)`)
- PyTorch + bf16 autocast on CUDA, TF32 enabled (`torch.set_float32_matmul_precision('high')`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Raw Gowalla checkins via gdrive folder (gdown), TIGER 2022 tract shapefiles for STAN spatial-join
- 5-fold StratifiedGroupKFold protocol for STL rows (`stratify=target_category`, `groups=userid`, `seed=42`); chronological 80/10/10 + 5 seeds for REHDM faithful per paper §5.1

## Headline patterns

1. **STAN-STL `stl_hgi` is the strongest substrate-axis baseline at all 5 states.** HGI > Check2HGI on next_region under STAN's head, consistent across AL/AZ/FL/CA/TX. The within-baseline `stl_hgi` − `stl_check2hgi` gap is +3.68 pp (AL) / +2.62 pp (AZ) / +0.96 pp (FL) / +1.63 pp (CA) / +1.35 pp (TX) — magnitude shrinks with scale.
2. **REHDM `faithful` matches its paper-published trend at FL** (Acc@10 65.68 vs AL 66.06 vs AZ 54.65). FL Acc@10 is essentially tied with STAN-faithful FL (65.36) — reasonable since both are trained on raw inputs under their respective protocols. FL is the only state where faithful crosses the Markov-1-region floor (+0.31 pp for STAN, +0.63 pp for REHDM).
3. **REHDM `stl_*` underperforms STAN `stl_*` by 20–35 pp under study protocol** (cold-user 5-fold). The full hypergraph is operative but loses to STAN's matching-attention readout when the input is a 9-step embedding sequence rather than 6-ID raw inputs at warm-user windows.
4. **AL/AZ from-scratch faithful (STAN) sits below Markov-1-region** (4–13 pp below). The architecture needs either pre-trained substrate or much more data.

Full deep-dive interpretation: `../research/STAN_THREE_WAY_COMPARISON.md` and `../research/SUBSTRATE_COMPARISON_FINDINGS.md`.

## What changed on disk

- `docs/studies/check2hgi/baselines/README.md` — status board updated; STAN/REHDM faithful CA/TX marked ⚪ (skip).
- `docs/studies/check2hgi/baselines/next_region/comparison.md` — ReHDM faithful FL filled (line 13 + faithful detail table § footnote on b=128 protocol). ReHDM stl rows filled for AZ/FL.
- `docs/studies/check2hgi/baselines/next_region/results/{arizona,florida}.json` — `baselines.rehdm.{faithful,stl_check2hgi,stl_hgi}` blocks added.
- `research/baselines/stan/train.py` — TF32 + bf16 + GPU-resident tensors + cheap-metric loop (uncommitted, optional).
- `research/baselines/rehdm/train.py` + `train_stl_study.py` — TF32 + bf16 + 12 workers + prefetch_factor=4 (uncommitted, optional).

## Outstanding (intentionally not addressed)

- STAN/REHDM faithful at CA/TX (⚪): documented as out of scope.
- GA: not in study (no checkins, no substrate, no inputs); estimated 30–40 h wall-clock to add from scratch — not pursued.
- next_category at REHDM: REHDM is a region baseline; we did not evaluate it on next_category. STAN-STL and floor coverage at next_category is already 5-state.
