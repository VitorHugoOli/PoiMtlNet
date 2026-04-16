# Check2HGI Study — Master Plan

**Goal:** Validate that next-region as an auxiliary task improves next-category prediction on Check2HGI check-in-level embeddings, with a champion MTL configuration found through systematic ablation.

## Phase overview

| Phase | What | Claims | Duration | Requires |
|---|---|---|---|---|
| **P0** | ✅ Integrity + simple baselines + audits | CH04 floor | done | — |
| **P1** | Region-head validation + head ablation (single-task) | CH04, CH05 | ~2h | P0 |
| **P2** | TaskSet-parameterise all MTL archs + full arch × optim ablation | CH06 | ~1 day | P1 (head winner) |
| **P3** | MTL headline: champion config, multi-seed n=15 | **CH01**, CH02, CH07 | ~4h | P2 (champion) |
| **P4** | Dual-stream: region_embedding as parallel input | CH03, CH08 | ~3h | P3 |
| **P5** | Cross-attention (gated on P4 ≥ 2pp on FL) | CH09 | ~6h | P4 |

| **P6** | Check2HGI encoder enrichment (literature research + implementation + ablation) | CH12, CH13 | ~2 days | P3 baseline needed |

**Total:** ~18h for P0–P5 sequential (excluding P5 if gated). P6 is an independent research track that can run in parallel with P4/P5 once P3 baseline exists.

## Execution order rationale

**P1 (heads) before P2 (MTL ablation):**
- The region head is **untested** — it's the next-category head repurposed for ~1K classes. We need to know it works before layering MTL.
- The head winner feeds into the MTL ablation as a **fixed choice**. Wrong head → wrong MTL conclusions.
- HMT-GRN uses a GRU for their region head (not a transformer). If GRU wins in P1, that changes which MTL backbone benefits most in P2.
- P1 is cheaper (~5 runs) than P2 (~15+ runs).

**P2 (MTL ablation) before P3 (headline):**
- The headline must use the champion (arch, optim) pair, not an arbitrary default.
- Running the headline first with NashMTL+base-MTLnet and then discovering CGC+equal_weight is better would waste the multi-seed n=15 budget.

## Phase details

### P1 — Region-head validation + head ablation

**Single-task next_region on Check2HGI, Alabama only.**

5 head variants, screen at 1-fold × 10-epoch, top-2 confirmed at 5-fold × 50-epoch:

| Head | Architecture | Why |
|---|---|---|
| `next_mtl` | 4-layer transformer + causal + attn pool | Default (next-category champion) |
| `next_gru` | Bi-GRU | HMT-GRN's region-head approach |
| `next_lstm` | LSTM | Classic recurrent |
| `next_tcn_residual` | TCN + residual | Was standalone next-category winner in prior fusion work |
| `next_temporal_cnn` | Shallow temporal CNN | Simple CNN baseline |

**Also run:** single-task next_category with `next_mtl` (reference for P3 pairing — already done: 38.67% F1 on AL).

**Gate:** at least one head achieves region Acc@10 ≥ 2× Markov floor (21.3% → ≥ 42.6%). If none do, investigate before P2.

### P2 — Full MTL architecture × optimiser ablation

**Pre-requisite (code work, ~half day):** Parameterise `MTLnetCGC`, `MTLnetMMoE`, `MTLnetDSelectK`, `MTLnetPLE` with `TaskSet` — same pattern as base `MTLnet`. All hardcode `category_x, next_x` in their `_mix()` internals and need the slot rename.

**Also:** research whether alternative MTL architectures better suit {next_category, next_region} (both tasks are sequential from the same input, unlike fusion's flat+sequential pair). Consider:
- Asymmetric sharing (region → category but not vice versa)
- Cross-stitch / sluice networks (learn which layers to share per task)
- Simpler approaches (FiLM may suffice since both tasks share input structure)

**Ablation grid:**

5 architectures × all available optimisers (NashMTL, equal_weight, CAGrad, Aligned-MTL, PCGrad, DWA, FAMO, etc.):

| Stage | Runs | Config | Time/run | Total |
|---|---|---|---|---|
| Screen | 5 × N_optim | 1-fold × 10-epoch, AL | ~1 min | ~1h |
| Promote top-10 | 10 | 2-fold × 15-epoch, AL | ~3 min | ~30 min |
| Confirm top-5 | 5 | 5-fold × 50-epoch, AL | ~22 min | ~2h |
| FL replication of top-2 | 2 | 5-fold × 50-epoch, FL | ~80 min | ~3h |

**Head config:** region head = P1 winner (slot B); category head = `next_mtl` (slot A, proven).

**Gate:** champion identified with sensible metrics (next-category F1 > 30%, next-region Acc@10 > 15% on AL).

### P3 — MTL headline (multi-seed n=15)

Champion arch + optim + heads from P2.

| Run | State | Seeds | Folds | Purpose |
|---|---|---|---|---|
| P3.1.AL | AL | {42, 123, 2024} | 5 each | CH01 + CH02 |
| P3.1.FL | FL | {42, 123, 2024} | 5 each | CH01 + CH02 replication |

Compare against P1 single-task references (same folds, same seeds).

FL runs with `--use-class-weights` to mitigate 22.5% majority next-region class.

### P4 — Dual-stream region input

Feed `[B, 9, 128]` (check-in ⊕ region embedding per timestep) instead of `[B, 9, 64]`.

Same champion config as P3. Multi-seed on AL + FL.

### P5 — Cross-attention (gated on P4)

Only runs if P4 shows ≥ 2pp next-category F1 lift on FL. New `MTLnetCrossAttn` architecture with bidirectional cross-attention between check-in and region streams.

## Datasets

| State | Check-ins | POIs | Regions | Sequence rows |
|---|---|---|---|---|
| Alabama (primary) | 113,846 | 11,848 | 1,109 | 12,709 |
| Florida (replication) | 1,407,034 | 76,544 | 4,703 | 159,175 |
| Arizona (triangulation) | ~120K | ~10K | 1,547 | 26,396 |

### P6 — Check2HGI encoder enrichment (research + implementation + ablation)

**This is a research phase, not a pure execution phase.** Requires literature review and evaluation before implementation.

**Source:** `docs/issues/CHECK2HGI_ENRICHMENT_PROPOSAL.md` proposes four tracks:

| Track | What | Literature to review | Expected impact |
|---|---|---|---|
| **Temporal** | Replace fixed 4D sin/cos with learnable multi-frequency time embedding (Time2Vec-like). Add time-gap, dwell-time proxy, recency decay. | Time2Vec (Kazemi et al. 2019), TiSASRec (Li et al. 2020), ImNext (He et al. 2024) | Medium — temporal patterns drive next-category prediction |
| **Spatial** | Add continuous geospatial positional encoding from (lat,lon). Add distance-to-previous-POI, distance-to-user-centroid, distance-to-region-centroid. | Sphere2Vec (Mai et al. 2023), Space2Vec (Mai et al. 2020), PE-GNN (Zhang et al. 2023) | Medium — spatial context currently enters only via graph edges |
| **Graph** | New edge families: KNN spatial, temporal-window co-occurrence, revisit-strength. | LSTPM (Sun et al. 2020), Graph-Flashback (Rao et al. 2022), STAN (Luo et al. 2021) | Small-medium — edge enrichment is a known but modest boost |
| **Loss** | Multi-view contrastive, hard negatives, auxiliary pretext tasks (masked check-in reconstruction). | MGCL (Zhu et al. 2024), SelfMove (2023), GraphCL (You et al. 2020) | Unknown — research-grade |

**Execution plan:**

1. **Literature survey** (~4h): for each track, read the 2–3 papers listed, extract the mechanism, assess implementation cost, and predict whether it helps on our data characteristics (Gowalla state-level, 7 categories, ~1K regions, user-sequence-dominant graph).
2. **Prioritise:** rank tracks by expected-lift / implementation-cost ratio. Start with the cheapest high-impact track.
3. **Implement Phase 1 enrichment** (the winner): modify `research/embeddings/check2hgi/preprocess.py::_build_node_features` with a config flag (`temporal_mode={basic, enriched}`, `spatial_mode={none, geo_basis}`). Keep backward compatibility — `basic` reproduces the current 11-feature vector.
4. **Regenerate embeddings** for AL with the enriched config. ~20 min.
5. **Ablation:** compare enriched vs vanilla Check2HGI on the P3 champion MTL config. If enriched wins by ≥ 2pp next-category F1 → adopt as the new default.
6. **If Phase 1 enrichment works**, consider stacking Phase 2 (second track).

**Gate:** P3 must exist as the vanilla-Check2HGI baseline before any enrichment is tested — otherwise the lift is unattributable ("was it the enrichment or the new MTL config?").

**Key question for the literature review:** does temporal enrichment (Time2Vec-like learnable frequencies) help more than spatial enrichment (geo positional encoding) for **next-category** prediction specifically? The fusion study's Sphere2Vec + Time2Vec results on the category task may give a prior — but on POI-level embeddings, not check-in-level. The check-in-level context might absorb temporal info naturally from the user-sequence edges (which carry exponential time-decay weights), making spatial enrichment the higher-value addition.

## Exit criteria

Branch merges when:
1. CH01 + CH02 resolved with evidence (P3).
2. CH06 resolved (P2 champion identified).
3. CH03 resolved (P4 dual-stream tested).
4. P1 head ablation documented.
5. All legacy tests green.
6. Paper findings section drafted.
