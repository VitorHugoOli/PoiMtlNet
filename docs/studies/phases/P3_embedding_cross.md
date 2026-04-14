# Phase 3 — Cross-embedding validation

**Goal:** (a) establish that the champion config outperforms the CBIC config on *all three* embeddings (not just fusion), and (b) quantify the role of the embedding itself in final performance.

**Duration:** ~6-10 h (mostly AL + AZ on this machine; FL goes to another machine for ~8 h).

**Embedded claims:**
- C01 — fusion improves over single-source HGI (core thesis)
- C03 — equal weighting suffices on single-source (completion of the test from P1)
- C04 — architecture rankings depend on embedding (completion)
- C11 — embedding quality is the dominant factor
- C12 — CBIC config fails on all embeddings, not just DGI
- C13 — fusion gain is from complementarity, not dimension
- C14 — Time2Vec specifically helps next-POI

---

## Preconditions

- P1 + P2 complete.
- Champion identified: (arch*, optim*, cat_head*, next_head*).
- Embeddings available for DGI, HGI, Fusion on AL + AZ (FL optional).
- If C13 is in scope: HGI at 128D embedding trained (may require upstream work, see below).

---

## Experimental design

### Core matrix

We need a 2 × 3 matrix of configurations, run on AL + AZ (+ FL for the final):

|                    | DGI      | HGI      | Fusion   |
|--------------------|----------|----------|----------|
| **CBIC config**    | baseline | test     | test     |
| **Champion config**| test     | test     | champion |

That's **6 configurations** per state.

- **CBIC config:** `mtlnet` (FiLM + hard sharing) + `nash_mtl` optimizer + default heads. This is the exact configuration of the CBIC 2025 paper.
- **Champion config:** arch*, optim*, cat_head*, next_head* from P1/P2.

All at 5f × 50ep. On AL: ~22 min × 6 = ~2.2 h. On AZ: ~same. On FL: ~4 h × 6 = ~24 h (so FL is expensive; we may run only the 2-3 most important cells there).

### Minimum subset for FL

If FL budget is tight, run only:
- Champion on fusion (already have from P1 confirmation potentially)
- CBIC config on DGI (for the CBIC vs us comparison — the story's foundation)
- CBIC config on fusion (to confirm CBIC config fails on fusion too)

That's 3 FL runs ≈ 12 h.

---

## Steps

### P3.1 — Run CBIC config on DGI (all 3 states)

This replicates the CBIC 2025 result on the new (fixed-label) data.

**Expected:**
- If results roughly match CBIC paper (AL: Cat 46%, FL: Cat 48%): great, data regeneration preserves the baseline.
- If results deviate much: either the label fix changed things, or our pipeline drifted. Investigate before proceeding.

### P3.2 — Run CBIC config on HGI (all 3 states)

Same CBIC architecture, but with HGI embeddings. Tests whether the issue was just DGI.

**Expected outcome** (from C12):
- CBIC config on HGI should still underperform champion config on HGI, because the problem is the arch+optim combo, not the embedding.
- If CBIC config on HGI matches champion on HGI: CBIC's bottleneck was actually the embedding; reframe C12.

### P3.3 — Run CBIC config on fusion (all 3 states)

Tests whether fusion alone rescues CBIC's architecture. If so, the architecture choice matters less; if not, fusion needs the new architecture too.

### P3.4 — Run champion config on DGI (all 3 states)

The interesting control. Does the champion arch+optim+heads rescue DGI embeddings?

**Expected:** champion on DGI should be much better than CBIC on DGI, but still worse than champion on HGI (embedding quality matters).

### P3.5 — Run champion config on HGI (all 3 states)

The cleanest test of C01 — does fusion beat HGI specifically?

### P3.6 — Run champion config on fusion (all 3 states)

Already have AL + AZ from P1 confirmation; run FL here (or confirm we've already done it).

### P3.7 (optional) — HGI at 128D for C13

Requires upstream work: train HGI with embedding_dim=128 (modify HGI training config), regenerate inputs.

If done: run champion config on HGI-128D. Compare to champion on fusion-128D. If HGI-128D ≈ fusion-128D, the gain was from dimension not complementarity.

**Assessment:** likely too expensive to do before BRACIS. Move to journal extension.

### P3.8 (optional) — Time2Vec ablation for C14

Modify fusion to be HGI(64) + HGI(64) instead of HGI + Time2Vec for the next task. Run champion. If this matches fusion-champion, Time2Vec contributes no signal; if it's worse, Time2Vec helps.

Requires generating a "HGI-only-doubled" fusion input. ~2 h of upstream work. Do only if scientifically important.

---

## Test IDs

- `P3_AL_CBIC_DGI`, `P3_AL_CBIC_HGI`, `P3_AL_CBIC_Fusion`
- `P3_AL_champ_DGI`, `P3_AL_champ_HGI`, `P3_AL_champ_Fusion`
- Same for AZ, FL

---

## Analysis

### C01 (fusion > HGI)

Compare:
- Champion on HGI (P3.5)
- Champion on fusion (P3.6)

On each state, delta = champion-fusion − champion-HGI.

**Confirm** if delta > 0.02 joint consistently across AL, AZ, FL.

### C11 (embedding dominates)

For the champion config, compute:
- Range across {DGI, HGI, Fusion} = max_embedding - min_embedding joint F1
- Range across {archs × optims} at best embedding = from P1

If embedding-range > config-range: **C11 confirmed**.

### C12 (CBIC config fails across embeddings)

For each embedding, compute:
- delta_emb = champion_on_emb − CBIC_on_emb

If delta_emb > 0.05 on all three: **C12 confirmed**.
If delta_emb is small on any embedding (e.g., CBIC is fine on HGI): **C12 partial** — CBIC was not just failing on DGI; it was failing on DGI specifically.

### C04 (arch rankings embedding-dependent)

Using P1 results on fusion + selective P3 runs on DGI/HGI with the same architecture grid:
- For each embedding, rank architectures by joint F1
- If the ranking is stable: C04 refuted
- If ranking changes: C04 confirmed

For P3 we cannot re-run the full 5×20 grid on DGI/HGI (too expensive). A cheaper option: run the top-3 P1 architectures on DGI/HGI with the best optimizer from P1, and confirm the ranking is preserved or flipped.

---

## Compute budget

| Step | AL | AZ | FL |
|------|----|----|----|
| P3.1-P3.6 core | 6 × 22min = ~2h | 6 × 25min = ~2.5h | 6 × 4h = ~24h (minimum 3 cells = ~12h) |
| P3.7 HGI-128D | +2h upstream + 3×22m | +3×25m | +3×4h |
| P3.8 Time2Vec ablation | +1h upstream + 2×22m | +2×25m | +2×4h |
| **Minimum (AL+AZ, core)** | **~2h** | **~2.5h** | — |
| **Realistic (AL+AZ+FL core)** | 2h | 2.5h | 12h (on other machine) |
| **Stretch (+ P3.7 + P3.8)** | +4h | +4h | +20h (won't fit for BRACIS) |

---

## Surprises to watch for

| Symptom | Interpretation |
|---------|----------------|
| CBIC config on HGI ≈ Champion on HGI | CBIC's failure was about DGI embedding quality, not arch/optim. **Major reframe.** |
| Champion on DGI > CBIC on DGI by a lot | New config rescues DGI too — strong evidence for framework claim |
| Fusion < HGI on champion config | The fusion story collapses. Either label bug still exists or our fusion strategy was wrong. |
| Florida champion-fusion < Alabama champion-fusion | Concerning but not fatal; may reflect state-dependent data characteristics |
| HGI-128D ≈ fusion-128D | C13 refuted; the gain was dimensional. Pivotal for the paper. |

---

## Outputs

- `docs/studies/results/P3/` populated
- `docs/studies/results/P3/SUMMARY.md`:
  - The 2×3 matrix table with deltas
  - Per-state replication
  - Updated claim statuses (C01, C04, C11, C12, C13?, C14?)
- Paper Table 1 (three-way evolution: CBIC → HGI-only → Fusion) filled with new numbers

---

## Phase gate for P4

Proceed to P4 if:
1. At least AL core (6 cells) done.
2. Champion config confirms it improves over CBIC config on at least 2 of 3 embeddings.
3. At least one state confirms C01 (fusion > HGI on champion config).

If champion ≈ CBIC on any embedding: investigate before proceeding — may signal a code bug.
