# Check2HGI Study — Master Plan

**Paper contribution (revised 2026-04-16 evening):** three intertwined claims, summarized in `HANDOFF.md` §Paper contribution:

1. **[CH16] Check2HGI improves next-category F1 over HGI** (primary substrate claim; Check2HGI was designed for exactly this).
2. **[CH17] Check2HGI surpasses POI-RGNN and a prior HGI-based next-category article** (external published comparisons).
3. **[CH03] Per-task input modality is the Pareto-bidirectional MTL design** (architectural claim; confirmed directionally on AL).

Integration test: **[CH01/CH02]** bidirectional MTL preserves / improves both heads on FL+CA+TX.

**Development vs headline states:**
- **Development / ablation (cheap iteration):** Alabama (AL). Small, sparse, ~30 min per 5f×50ep run.
- **Headline paper table:** Florida (FL) + California (CA) + Texas (TX) — the three large states Check2HGI is trained on. See `CONCERNS.md §C01–C02`.

## Phase overview

| Phase | What | Claims | Duration | Status / Requires |
|---|---|---|---|---|
| **P0** | Integrity + simple baselines (region Markov, 1/2/3-step) + audits | floor | done | ✅ |
| **P1** | Region-head ablation × input type {checkin, region, concat} + hparam sweep | CH04 (reframed), CH05 | done | ✅ |
| **P1.5** | Embedding comparison region-side: Check2HGI vs HGI (AL, region single-task) | CH15 | done | ✅ tied — expected |
| **P1.5b** | Embedding comparison category-side: Check2HGI vs HGI (AL, next-category single-task) | **CH16** | ~20 min | P1 done; running 2026-04-16 evening |
| **P2** | Full arch × optim ablation (AL only, TCN head for compute) | CH06 | ~1 day | P1.5b (primary substrate claim) |
| **P3** | MTL headline (bidirectional, Δm): champion config, multi-seed n=15, on FL + CA + TX | **CH01**, **CH02**, CH07 | ~15h | P2 champion |
| **P4** | Per-task input modality: 4-way ablation on AL, top-2 replicated on CA/TX | **CH03**, CH08 | ~4h AL + ~6h CA/TX | P4 pipeline ✅; runs in parallel with P2 or after |
| **P5** | Cross-attention between task-specific encoders (gated on P4 outcome) | CH09 | ~6h | P4 |
| **P6** | Check2HGI encoder enrichment (literature + implementation + ablation) | CH12, CH13 | ~2 days | P3 baseline |

**Total:** ~40 h for the sequential path through P3 on all three headline states; ~50 h including P4 headline. P5 gated; P6 independent.

**Head champion (see `CONCERNS.md §C06`):** `next_gru` is the reported champion (HMT-GRN-aligned, statistically best on AL 5f×50ep). `next_tcn_residual` is the compute-efficient substitute used for grids (P2 arch×optim, P4 on AL) — ~20× faster with statistically equivalent region-task Acc@10 on AL. At the end of P2, the top-3 arch×optim configs get re-run with `next_gru` on AL as a sanity check before P3 commits.

## Execution order rationale

**P1 (heads) before P1.5 (embedding):** confirmed the region task is learnable with a GRU head. Needed a working head before asking "which embedding is better at driving that head?"

**P1.5 (embedding) before P2 (arch × optim):** if POI2HGI wins on the region task, the whole paper substrate shifts and we'd be wasting a 20-run arch×optim grid on the wrong embedding. Cheap gate (~1 h, 1 state), high-value decision.

**P2 before P3 (headline):** the headline must use the champion (arch, optim) pair, not an arbitrary default. Running the n=15 headline with NashMTL+FiLM and then discovering CGC+GradNorm is better would waste the multi-seed budget on the three headline states.

**P4 (per-task modality) parallel with P2:** the per-task input pipeline is already shipped. P4's AL development phase (4 variants × 5f × 50ep × 1 seed) can run in parallel with P2's grid — they use the same compute and don't depend on each other. Only the P4 CA/TX replication depends on the P3 champion arch.

**P3 (headline) before P5 / P6:** both P5 (cross-attention) and P6 (encoder enrichment) need a P3 vanilla-baseline to compare against; otherwise any lift is unattributable.

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

### P3 — MTL headline (multi-seed n=15, bidirectional)

Champion arch + optim + heads from P2. **Tests CH01 (bidirectional) + CH02 (no negative transfer).**

| Run | State | Seeds | Folds | Purpose |
|---|---|---|---|---|
| P3.1.AL | AL | {42, 123, 2024} | 5 each | CH01 + CH02 on BOTH heads |
| P3.1.FL | FL | {42, 123, 2024} | 5 each | CH01 + CH02 on BOTH heads (replication) |

**Comparison:** paired MTL-vs-single-task on **each head** (cat F1 on AL/FL, region Acc@10 on AL/FL) across 15 paired samples. All four comparisons must show MTL ≥ single-task with α=0.05 for CH01 to pass.

FL runs with `--use-class-weights` to mitigate 22.5% majority next-region class.

### P4 — Per-task input modality (CH03)

The MTL architecture has two independent task-specific encoders. Rather than forcing both to see the same input, feed each its task-appropriate modality and let the shared backbone bridge them.

**4-way comparison** at the P3 champion MTL arch + optim:

| Variant | Task A (`category_encoder`) input | Task B (`next_encoder`) input | Feature dim |
|---|---|---|---|
| **per_task** (proposed) | check-in emb `[B, 9, 64]` | region emb `[B, 9, 64]` | 64 each |
| concat | `[checkin ⊕ region]` `[B, 9, 128]` | `[checkin ⊕ region]` `[B, 9, 128]` | 128 each |
| shared_checkin | check-in emb | check-in emb | 64 each |
| shared_region | region emb | region emb | 64 each |

All variants at the champion arch + multi-seed n=15 on AL; top-2 replicated on FL (CH08 state-dependence).

**Implementation note:** requires a small extension to the MTL fold creator (`FoldCreator` path for check2HGI) to produce per-task X tensors, plus CLI flags `--task-a-input-type` / `--task-b-input-type`. ~80 LOC in the data pipeline, zero in the model (MTLnet already unpacks `(category_input, next_input)` separately).

### P5 — Cross-attention between task-specific encoders (gated on P4)

Only runs if P4's per-task-modality champion does not saturate the single-task ceilings (i.e., further arch capacity has room to add value). `MTLnetCrossAttn` with bidirectional cross-attention between `category_encoder`'s output and `next_encoder`'s output, applied before the shared backbone.

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
1. CH01 + CH02 resolved with evidence on FL + CA + TX (P3).
2. CH03 resolved (P4 per-task modality 4-way comparison).
3. CH06 resolved (P2 champion identified with GRU sanity check).
4. CH15 resolved (P1.5 embedding choice justified).
5. All `CONCERNS.md` entries at status `resolved`, `monitored`, or explicit `deferred` — no `open`.
6. All legacy tests green.
7. Paper findings section drafted (both primary and backup narrative paths pre-written — see `CONCERNS.md §C05`).
