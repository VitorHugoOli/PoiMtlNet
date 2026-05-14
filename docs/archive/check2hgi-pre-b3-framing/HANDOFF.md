# Check2HGI Study — Session Handoff (2026-04-16, updated end-of-day)

## Status at a glance

| Phase | Status |
|---|---|
| **P0** | ✅ complete — integrity, simple baselines (region Markov 1/2/3-step), CH14 audit |
| **P1** | ✅ complete — 5 heads × {checkin, region, concat}; GRU confirmed on AL 5f×50ep (56.94 ± 4.01) + FL 5f×50ep (68.33 ± 0.58); `next_mtl` scrapped as region-head candidate |
| **P1.5** | ✅ complete — Check2HGI vs HGI on AL region single-task. Tied (57.02 vs 56.11 Acc@10). Expected — pooling to region erases check-in-level variance. See CH15. |
| **P1.5b** | ✅ complete — Check2HGI vs HGI on AL next-category. **Check2HGI wins by +15.68 pp F1 (39.16 vs 23.48)** with non-overlapping std envelopes on all 6 metrics. CH16 confirmed robustly. |
| **P2-prep** | ✅ CGC/MMoE/DSelectK/PLE TaskSet-aware |
| **P4-prep** | ✅ per-task input pipeline shipped (`FoldCreator.task_a_input_type` / `task_b_input_type`, CLI flags, `region_sequence.py`) |
| **P2** | ready — arch × optim grid with TCN region head (compute-efficient); GRU sanity check on top-3 at end |
| **P3** | ready after P2 champion — headline multi-seed on **FL + CA + TX** (not AL alone; AL stays as ablation state) |
| **P4** | pipeline ready; development on AL can run in parallel with P2; headline replication on CA/TX after P3 champion |
| **P5** | gated on P4 outcome |
| **P6** | research track — no blocker |

**Concerns log:** `CONCERNS.md` now tracks 9 open issues with resolutions/fallbacks (dataset breadth, FL saturation, joint metric, head swap framing, null-result backup, GRU vs TCN, embedding choice, CH04 reframe, SSD reliability).

## Paper contribution (revised 2026-04-16 evening after user clarification)

Three intertwined claims — all three must land for the paper to be strong; any two still gives a workshop-grade result:

1. **[CH16] Check2HGI improves next-category prediction over HGI.** Check2HGI was designed as an HGI modification that adds check-in-level contextual variation. The PRIMARY substrate claim is that this design lifts next-category F1. Region-side tie (CH15, confirmed) is expected because pooling to region erases per-visit variance.
2. **[CH17] Check2HGI surpasses POI-RGNN (published) and a prior HGI-based next-category article.** External published comparisons anchoring the paper's contribution.
3. **[CH03] Per-task input modality is the Pareto-bidirectional MTL design.** Check2HGI uniquely enables it because it supplies both check-in and region modalities simultaneously; HGI architecturally cannot. Confirmed directionally on AL (P4-dev).

Integration test: **[CH01/CH02] Bidirectional MTL** — with (1)+(2)+(3), does MTL preserve / improve both heads? Tested in P3 on FL/CA/TX headline.

**Out of scope — Future work (P6):** region-side encoder enrichment (temporal / spatial / graph / loss) is the planned research track to improve next-region. Check2HGI's current region numbers are deliberately not optimized; improving them is a separate contribution.

## Task pair

**{next_category (7 classes), next_region (~1K classes)}** on Check2HGI check-in-level embeddings.

## Key numbers (2026-04-16)

| Metric | AL | FL |
|---|---|---|
| Check2HGI single-task next-cat macro-F1 | **38.67%** (5f×50ep) | TBD |
| Check2HGI single-task next-region Acc@10 (region-emb, `next_gru`, **5f×50ep**) | **56.94% ± 4.01** | **68.33% ± 0.58** |
| Check2HGI single-task next-region Acc@10 (region-emb, `next_tcn_residual`, 5f×50ep) | 56.11% ± 4.02 | — (ties GRU on AL, 20× faster) |
| Check2HGI single-task next-region Acc@10 (region-emb, `next_gru` scaled hd=384, 1f×50ep) | 54.68% (single-fold) | — |
| Check2HGI single-task next-region Acc@10 (check-in-emb, `next_gru` default, 1f×30ep) | 20.11% (below Markov!) | — |
| Check2HGI single-task next-region Acc@10 (concat input, `next_gru`, 1f×30ep) | 49.57% (worse than region-only) | — |
| **Single-task factor vs Markov-1-region** | **1.21× (AL)** | **1.050× (FL)** |
| POI-RGNN published next-cat (FL/CA/TX) | 31.8–34.5% | 34.5% |
| Simple floor: majority next-cat | 34.2% | 24.7% |
| Simple floor: Markov next-cat | 31.7% | 37.2% |
| **Simple floor: Markov-1-region next-region Acc@10** | **47.01%** | **65.05%** |
| Simple floor: Markov-2-region / Markov-3-region Acc@10 | 37.9% / 35.2% | 59.2% / 56.7% |

**Note:** the old `markov_1step` (POI-level) baseline (21.3% AL / 45.9% FL) had a ~50% fallback rate to top-k-popular — it was a degenerate baseline. The corrected `markov_1step_region` is the paper-reported floor. The POI-level entry is kept for continuity but not used as a floor.

## 2026-04-17 mid-morning — P2 deep-dive results

**Three important findings (in order of discovery):**

### C11 fold leakage fixed; CH16 sharpened

STL `next` single-task used `StratifiedKFold` (not user-disjoint), unlike MTL's grouped scheme. Fix applied (`5217095`). After refair on AL 5f × 50ep: Check2HGI F1 38.58 ± 1.23 (was 39.16), HGI F1 20.29 ± 1.34 (was 23.48). **CH16 Δ grew from +15.67 to +18.30 pp** — Check2HGI is *more* advantaged on fair folds because HGI leaks through POI-level taste memorisation. See `issues/FOLD_LEAKAGE_AUDIT.md`.

### REGION_HEAD_MISMATCH fixed, reveals deeper issue

GRU region head wired in + pad-mask re-zero in MTL forward (`b92fc62`). Lifted MTL reg Acc@10 by only +1.26 pp (47.62 Transformer → 48.88 GRU). The remaining gap to STL-GRU-standalone's 56.94% is **backbone dilution**, not head choice. See `issues/REGION_HEAD_MISMATCH.md` + `issues/BACKBONE_DILUTION.md`.

### CH01 bidirectional FAILS on AL

MTL dselectk+pcgrad 5f × 50ep GRU head:
- cat F1: 36.08 ± 1.96 vs STL 38.58 ± 1.23 (Δ −2.50, σ-overlap)
- reg Acc@10: 48.88 ± 6.26 vs STL 56.94 ± 4.01 (Δ −8.06, σ-overlap)
- Δm = −14.12%; Pareto gate fails on both r_A and r_B.

**Mechanistic finding (paper-worthy on its own):** MTL lift is **inversely related to task-B head's standalone strength**. Weak head (Transformer, 7.4% standalone) → MTL lifts to 47.62% (+40 pp). Strong head (GRU, 56.94% standalone) → MTL dilutes to 48.88% (−8 pp). Shared backbone's fixed capacity cannot exceed a strong head's own ceiling.

### Next experiments launched

1. **FL 1f × 50ep validate** (bg `bpnvxd3b4`) — does 127K samples close the dilution gap?
2. **Research subagent** (agent `abe5733efde94d6d4`) — SOTA MTL papers for backbone-dilution fixes.

### Three-path decision matrix post-FL

- FL succeeds → CH01 survives on headline states; AL framed as small-data caveat.
- FL partially succeeds → loosen CH01 to "no regression on either head."
- FL fails → retire CH01; lead paper with CH16 + CH03 + mechanistic insight.

### FL 1f × 50ep VERDICT (2026-04-17) — ASYMMETRIC MTL EFFECT

**With FL STL cat fair baseline now complete:**

| Task | FL MTL (1f×50ep) | FL STL fair | Δ | Verdict |
|---|---|---|---|---|
| cat F1 | 64.78 | 63.17 (1f×50ep) | **+1.61 pp** | **✅ MTL lifts category** |
| reg Acc@10 | 57.05 | 68.33 ± 0.58 (5f) | −11.28 pp | ❌ MTL dilutes region |
| reg MRR | 27.49 | 52.74 (5f) | −25.25 pp | ❌ |

**Δm = −14.82%; Pareto gate FAILS but ASYMMETRICALLY (cat helps, reg hurts).**

**This refines the story substantially:**

| | AL (10K) | FL (127K) |
|---|---|---|
| Cat F1 Δ | −2.50 (tied) | **+1.61 (clear lift)** |
| Reg Acc@10 Δ | −8.06 | −11.28 |

- **Category benefits from MTL at scale**, not at small-data. Data-quantity matters for the weaker task. AL's 10K is under-trained.
- **Region dilutes regardless of scale** because the standalone GRU head already saturates the signal its input carries. Shared-backbone capacity caps below the standalone ceiling.

**Updated narrative:** not "MTL doesn't work" — "MTL is a **task-asymmetric tradeoff**; per-task routing (MTLoRA / AdaShare) is needed to preserve the cat lift while recovering region." The ablation is the paper's main contribution.

**This elevates the ablation protocol (`research/MTL_ABLATION_PROTOCOL.md`) from a contingency to the paper's main experimental contribution.** Starting cheapest-first (RLW sanity → gradient-scaling asymmetric transfer → curriculum warmup → MTLoRA → AdaShare).

**Perf fix deployed mid-run (commit `37cbca1`):** vectorised `NextHeadGRU`'s last-valid-timestep extraction. The Python for-loop version made FL 1f × 50ep take 2h34m; future ablation runs should be ~50% faster.

---

## P1.5 — embedding comparison (2026-04-16)

Check2HGI vs HGI region embeddings on AL, TCN head, 5f × 50ep, region input, seed 42:

| Substrate | Acc@1 | Acc@10 | MRR |
|-----------|-------|--------|-----|
| Check2HGI | 21.76 ± 1.8 | **56.11 ± 4.02** | 33.4 ± 2.4 |
| HGI | 21.82 ± 1.5 | **57.02 ± 2.92** | 33.14 ± 1.87 |
| Δ (HGI − C2HGI) | +0.06 | +0.91 | −0.26 |

**Tied within noise.** Region-level embeddings converge to similar quality regardless of the upstream POI representation (check-in-level vs POI-level). The pooling to region smooths out contextual variation.

**Framing pivot (CH15 / CONCERNS §C07):** Check2HGI does *not* outperform HGI on the region task. The paper's contribution is not "Check2HGI > HGI." The *meaningful* Check2HGI advantage is at the **check-in-level input**, where HGI architecturally cannot compete (HGI has only per-POI vectors). This feeds the next claim.

## P4-dev — per-task input modality on AL (1f × 20ep, FiLM MTL, default head — directional)

| Variant | Cat F1 | Reg Acc@10 | Reg MRR | Pareto |
|---------|--------|------------|---------|--------|
| **per_task** (cat=checkin, reg=region) | 36.66 | **33.19** | **16.38** | ✅ only bidirectionally strong |
| concat (both=[checkin⊕region]) | 35.10 | 12.16 | 5.53 | ❌ dominated |
| shared_checkin (both=checkin) | 36.78 | 2.30 | 1.57 | ✅ cat-max, kills region |
| shared_region (both=region) | 20.19 | 34.44 | 16.38 | ✅ reg-max, kills category |

**Per-task is the only Pareto-bidirectional variant.** Concat is strictly dominated; shared modalities each collapse the opposite head. CH03 directionally confirmed on AL. Final P4 headline on CA/TX after P2 champion + GRU region head is pending.

Key implication: this makes the paper's per-task modality claim (CH03) the strongest *empirical* result in the study to date — the ordering is robust and the effect is large (regions fall from 33% to 2% when the region head loses its modality).

## P1 findings (frozen)

1. **Head champions:** `next_gru` (AL 56.94 ± 4.01, FL 68.33 ± 0.58) and `next_tcn_residual` (AL 56.11 ± 4.02) tie statistically. TCN is ~20× faster per fold — preferred when compute is binding in P2.
2. **`next_mtl` (Transformer) scrapped as region head:** catastrophic failure on region task (AL 7.40 ± 1.72 vs GRU 56.94). Architectural mismatch — head was designed for 7-class next-category; its dropout=0.35 × 4 transformer layers × 9-position sequence × 1109/4702-class softmax don't converge in 50ep. Remains a valid candidate for the category (Task-A) slot where it's the legacy champion.
3. **Input modality dominates head choice.** Check-in-emb input caps all heads at ~20% Acc@10 (AL, at Markov floor). Region-emb input lifts `next_gru` to 53–57% Acc@10 on AL, 68% on FL.
4. **Concat input is worse than region-only** (49.57% vs 53.33% on AL 1f×30ep). Check-in context = noise for region target. → motivates per-task input modality (P4).
5. **CH04 retired as a gate.** Best neural head on AL is 1.21× Markov-1-region; on FL 1.050×. Does not meet the original 2× target. Demoted to a reported comparison, not a go/no-go.
6. **FL is a dense-data regime.** FL Markov-1-region already sits at 65.05% Acc@10 because (r_t, r_{t+1}) transitions have 85%+ coverage on 159K training rows. Neural generalisation over 9-step context adds ~3.3 pp. Implication for P3: the bidirectional MTL lift on FL is the binding constraint — AL has more headroom (~10 pp).
7. **Scaling `next_gru` (hidden_dim 256→384, num_layers 2→3, label_smoothing=0.1, 30→50ep)** adds ~+1.4 pp. Returns diminishing.

## Ready for P2

All P2 pre-reqs are met:

1. ✅ **TaskSet-aware MTL variants** (CGC/MMoE/DSelectK/PLE + FiLM) — verified legacy-bit-exact.
2. ✅ **Per-task input pipeline** — `FoldCreator.task_a_input_type` / `task_b_input_type`, `scripts/train.py --task-a-input-type --task-b-input-type`, `src/data/inputs/region_sequence.py`. All 121 prior tests pass.
3. ✅ **P1 baselines frozen** — region-task single-task benchmark locked for paired comparison against MTL.

## Open decisions for P2 launch

1. **Region head champion.** GRU (safe, well-characterised) vs TCN-residual (statistically equivalent, 20× faster). Recommend TCN for P2 grid, GRU for P3 headline.
2. **P2 grid scope.** arch ∈ {FiLM, CGC, MMoE, DSelectK, PLE} × optim ∈ {NashMTL, PCGrad, GradNorm, naive} = 20 configs × 2 states × 5f×50ep. AL alone ~10 h on M2 Pro (per fold ~6 min with TCN region head). Full matrix = ~20 h.
3. **P3 multi-seed count.** 3 seeds × 5 folds = 15 paired samples per comparison.
4. **P4 config matrix.** 4 input variants × multi-seed × 2 states = 12 runs at 5f×50ep. Budget ~8 h. Confirm all 4 or drop `shared_region` (would be bad for category).
