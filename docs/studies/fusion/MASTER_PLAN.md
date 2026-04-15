# Master Plan

This is the overall strategy. Each phase is detailed in `phases/Pk_*.md`. Claims tested in each phase are enumerated in `CLAIMS_AND_HYPOTHESES.md`.

---

## Guiding principles

1. **Claim-driven, not experiment-driven.** Every test exists to validate or refute a claim. If a test doesn't map to a claim, we don't run it.
2. **Progressive narrowing.** Fast screens → promotion → full confirmation. Don't run 5f × 50ep when 1f × 10ep will answer the question.
3. **Two-state replication by default.** Alabama + Arizona (similar size, cheap). Florida only for final validation and claims that require it.
4. **Archive everything.** JSON summaries go under `docs/studies/results/`, versioned in git, keyed by test_id. A year from now we should be able to re-analyze.
5. **Critical coordinator.** After each test, the coordinator validates integrity and analyzes. Surprising results pause the pipeline and trigger new hypothesis generation.
6. **No shortcuts on controls.** Every claim of the form "X helps because Y" needs both a positive evidence (X with Y works) and a negative control (X without Y, or Y without X).

---

## Dataset plan

| State | Role | Size | Typical run cost (5f × 50ep) |
|-------|------|------|------------------------------|
| **Alabama (AL)** | Fast-path primary | small | ~22 min |
| **Arizona (AZ)** | Fast-path replication | small-medium | ~25 min (est.) |
| **Florida (FL)** | Slow heavy validation | 7× AL | ~4 h |
| (optional) California (CA) | Additional baseline comparison | ~5× AL | ~3 h |
| (optional) Texas (TX) | Additional baseline comparison | ~6× AL | ~3 h |
| (optional) Georgia (GA) | Additional baseline comparison | ~3× AL | ~1.5 h |

**Default per phase:** AL + AZ for screening and confirmation. FL only for Phase 3 cross-embedding and final champion validation.

---

## Phase overview

```
                 ┌────────────────────────────────────────┐
                 │  P0 — Preparation                       │
                 │  Embeddings regenerated, integrity      │
                 │  checks, tooling, baseline verification │
                 └───────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────┐
                 │  P1 — Architecture × Optimizer         │
                 │  5 × 20 grid on fusion                  │
                 │  Embedded claims: C01-C05               │
                 │  Output: best (arch, optim)             │
                 └───────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────┐
                 │  P2 — Heads + MTL vs Single-task       │
                 │  9 cat × 10 next, + single-task        │
                 │  Embedded claims: C06-C10               │
                 │  Output: best heads, MTL benefit size   │
                 └───────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────┐
                 │  P3 — Cross-embedding validation       │
                 │  Best config × {DGI, HGI, Fusion}      │
                 │  + CBIC config × {DGI, HGI, Fusion}    │
                 │  Embedded claims: C11-C14               │
                 └───────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────┐
                 │  P4 — Hyperparameter sensitivity        │
                 │  Champion config: DSelectK hparams,     │
                 │  backbone size, LR, batch, dropout      │
                 │  Embedded claims: C15-C18               │
                 └───────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────┐
                 │  P5 — Mechanistic / diagnostic         │
                 │  Gradient cosine, per-source gradients  │
                 │  Per-category F1, statistical package   │
                 │  Embedded claims: C18-C21               │
                 └───────────────┬────────────────────────┘
                                 │  (runs parallel to P5)
                 ┌───────────────▼────────────────────────┐
                 │  P6 — Convergence & MTL benefit claims │
                 │  Revisit Caruana/Ruder canonical claims │
                 │  with modern config (DSelectK+fusion)   │
                 │  Embedded claims: C22-C28               │
                 └───────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────┐
                 │  Synthesis → new PAPER_FINDINGS.md     │
                 │  Paper writing                          │
                 └────────────────────────────────────────┘
```

**Note on P5 vs P6:** both run after P4 locks the champion. P5 = *mechanism* (why does it work? gradient diagnostics, per-category breakdowns). P6 = *benefits* (does it deliver what MTL promises? convergence, no negative transfer, transferable representations). They can run in parallel — P6 is mostly post-hoc analysis of P2 data.

---

## Parallelism strategy

### Hardware: M4 Pro 24GB > M2 Pro 32GB for this workload

Decision recorded here so future reviewers don't second-guess the choice.

| Factor | M4 Pro 24GB | M2 Pro 32GB |
|---|---|---|
| Memory bandwidth | 273 GB/s | 200 GB/s |
| MPS matmul/attention (FP16) | ~30-40% faster | baseline |
| Unified RAM | 24 GB | 32 GB |
| macOS 15 kernel optimizations | yes | partial |

**Memory is not the bottleneck.** Peak footprint for the worst case (Florida MTL, 5f × 50ep):
  MTLnet + optimizer + activations ≈ 2 GB · both fold dataloaders ≈ 4 GB · PyTorch + MPS cache + Python + OS ≈ 8 GB → total ~15 GB. 24 GB leaves ~8 GB headroom; the extra 8 GB on M2 Pro is dead weight.

**Throughput is the bottleneck.** P1 alone is 100 runs × 1 min on AL plus 100 × 25 min on AZ. A 30-40% faster GPU knocks ~10 h off the BRACIS path.

**When M2 Pro 32GB would actually be better:** running two training processes in parallel on the same box (24 GB gets tight with two × ~10 GB), or keeping Florida + a Jupyter notebook with the full raw checkin parquet resident (~5 GB) during training. Neither is the plan here — Colab handles parallelism.

**MPS tuning before long runs** (either box):

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # prevent OOM from aggressive MPS caching on FL
export PYTORCH_ENABLE_MPS_FALLBACK=1           # quietly CPU-fall back on unimplemented ops
```

### This machine (MPS, always sequential for training)

- Queue Phase 1 screening on AL (100 runs × ~1 min)
- Queue Phase 1 screening on AZ (100 runs × ~1 min)
- These are 100 × 1 = ~1.7 h total back-to-back

### Other machines (user-managed)

Can run in parallel with this machine on:

- **FL experiments** (longest). Once we know the champion from AL+AZ, FL runs in background for 3-4 h per config.
- **Additional states (CA, TX, GA)** if needed for baseline comparison with HAVANA/POI-RGNN numbers that are only reported per state.
- **Embedding regeneration for additional states** (if we decide to expand beyond AL+AZ+FL).

### What CAN'T be parallelized

- Phase dependencies: P2 needs P1 winner. P3 needs P2 winner. P4 needs P3 winner.
- Coordinator decisions: human-in-the-loop reviews between phases.

### Communication protocol

User runs experiments on other machine → copies the run directory back to `results/fusion/<state>/...` on this machine → coordinator's `/study import` action archives them to `docs/studies/results/` with correct metadata.

---

## Timing budget (aspirational)

| Phase | AL | AZ | FL | Wall-clock (best case, parallel) |
|-------|----|----|----|-----------------------------------|
| P0 | — | — | — | 2-4 h (embedding regen + tooling + fold freezing) |
| P1 | ~4 h | ~4 h | 1 run, 4 h | **6-8 h** (AZ in parallel on other box) |
| P2 | ~6 h | ~6 h | 1 run, 4 h | **8-10 h** |
| P3 | ~1.5 h | ~1.5 h | 2 runs, 8 h | **8-10 h** |
| P4 | ~7 h | — | — | **7-8 h** |
| P5 | 2-4 h | 1-2 h | 1-2 runs | **4-8 h** |
| P6 | ~3 h (mostly post-hoc) | — | — | **~3-5 h** (runs parallel to P5) |
| **Total** | ~24 h | ~13 h | ~25 h | **~36-48 h wall-clock** over ~5-7 days |

This is tight for BRACIS deadline (2026-04-20). If needed, we ship after P3 + partial P4.

---

## Decision gates between phases

After P1: **Is fusion competitive with HGI-only at the best optimizer class?**
  → If yes: proceed to P2 testing new heads
  → If no: go straight to P3, frame paper as "HGI is still the winner, fusion not justified"

After P2: **Does MTL actually beat single-task on fusion?**
  → If yes: the "MTL when configured right" thesis holds. Proceed to P3 as planned.
  → If no: reframe paper as "task-specific fusion improves POI prediction; MTL vs single-task is orthogonal"

After P3: **Does the best config beat CBIC baseline on ALL three embeddings?**
  → If yes: we have a clean "new framework > old framework" story. Proceed to P4.
  → If no: some embeddings may not benefit; identify which and why.

After P4: **Is the champion robust to hyperparameter choices?**
  → If yes: ship the paper with the found values.
  → If no: investigate, possibly re-run P3 with better tuning.

After P6: **Does the champion clear the canonical MTL claims (no negative transfer, comparable convergence, transferable representations)?**
  → If yes: the paper's "MTL delivers when configured right" chapter is complete.
  → If **C28 fails (negative transfer on a task)**: reframe paper honestly — "MTL improves joint score but costs task X per-task F1 — trade-off discussion."

---

## Success criteria (how we know we're done)

- All claims in `CLAIMS_AND_HYPOTHESES.md` have status ∈ {confirmed, refuted, partial}. None left as `pending`.
- All results under `docs/studies/results/` have summary.json + metadata.json.
- `docs/PAPER_FINDINGS.md` is rewritten from the catalog.
- Paper draft ready for BRACIS submission in `articles/BRACIS_2026/`.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| New embeddings reveal old claims were wrong | High | This is *why* we're redoing the study. Expect findings to change. |
| Single-task-fusion matches MTL-fusion (C06 refuted) | Medium | Reframe paper around fusion+optimizer, drop MTL as hook. |
| Gradient-surgery doesn't help at matched batch (C02 refuted) | **Already seen in T0.2** | Reframe: gradient surgery *accelerates* but doesn't *improve* convergence. |
| 5×20 grid takes too long | Low | Screen at 1f×10ep first; only promote top-10 to longer runs. |
| BRACIS deadline misses | Medium | Prioritize P1+P2+P3 for paper; P4+P5 can go to journal extension. |
| Other machine unavailable | Medium | AL+AZ on this machine are enough for BRACIS. FL becomes nice-to-have. |

---

## When to re-plan

Re-visit this master plan if:
- Any decision gate (above) flips the planned direction
- A new hypothesis emerges that requires a phase we didn't plan for
- Compute budget changes materially
- The BRACIS deadline changes
