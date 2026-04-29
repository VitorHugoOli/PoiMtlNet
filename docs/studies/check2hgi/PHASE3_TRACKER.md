# Phase 3 Tracker — MTL CH18 closure on CA + TX

**Created 2026-04-29** after Phase 2 STL closed at all 5 states. Phase 2 MTL CH18 is confirmed at **AL+AZ+FL** (3 states); CA+TX deferred because Lightning T4 (15 GB VRAM) cannot fit the cross-attn MTLnet at canonical hparams for those state sizes (8497 / 6553 regions).

This tracker is the live work queue for **Phase 3 — MTL CH18 cross-state closure on CA + TX, on Lightning A100**. Numbers from this phase land in the paper's MTL substrate-specificity table.

> **Phase 3 launch authorisation:** ✅ implicit — gated only on bigger GPU. Phase 2 STL closure already confirms CH16 + reframes CH15; CH18 is the only remaining cross-state pillar.

> **🔒 Protocol parity with Phase 2 — DO NOT CHANGE.** Phase 3 deliberately uses the **same full-dataset transition matrix** as Phase 2 (`output/check2hgi/<state>/region_transition_log.pt`), including its known F44 leakage. This is *required* so CA + TX MTL numbers are directly comparable to AL + AZ + FL MTL numbers. Any change to the transition-matrix construction (per-fold rebuild, leakage-safe variant, etc.) breaks the cross-state CH18 paired comparison and would have to be a separate Phase 4 that re-runs **all 5 states × all task types** for protocol parity.
>
> **Pinned hparams (do not deviate):** lr=1e-4, bs=2048, NashMTL static_weight α_cat=0.75, model=mtlnet_crossattn, cat_head=next_gru, reg_head=next_getnext_hard with `d_model=256 num_heads=8`, transition matrix = full-dataset c2hgi, 5-fold StratifiedGroupKFold on userids, seed=42, 50 epochs. Same as Phase 2.

---

## Status board

🟢 = 5-fold complete · 🟡 = 1-fold or partial · 🔴 = pending · ⚫ = blocked.

### MTL B3 counterfactual grid (2 states × 2 substrates = 4 cells)

| Test | C2HGI | HGI | Combined paired test |
|---|:-:|:-:|:-:|
| **CA × C2HGI** (B3 north_star) | 🔴 pending | n/a | — |
| **CA × HGI** (counterfactual) | n/a | 🔴 pending | 🔴 paired test pending |
| **TX × C2HGI** (B3 north_star) | 🔴 pending | n/a | — |
| **TX × HGI** (counterfactual) | n/a | 🔴 pending | 🔴 paired test pending |

### Reference rows (already closed in Phase 1 / Phase 2)

| State | C2HGI cat F1 | HGI cat F1 | C2HGI reg Acc@10 | HGI reg Acc@10 |
|---|---:|---:|---:|---:|
| AL | 42.71 ± 1.37 | 25.96 ± 1.61 | 59.60 ± 4.09 | 29.95 ± 1.89 |
| AZ | 45.81 ± 1.30 | 28.70 ± 0.51 | 53.82 ± 3.11 | 22.10 ± 1.63 |
| FL | (1f reference, ~67%) | 34.74 ± 0.76 | — | 58.27 ± 3.37 |
| **CA** | **pending** | **pending** | **pending** | **pending** |
| **TX** | **pending** | **pending** | **pending** | **pending** |

---

## 1 · Acceptance criteria (when does Phase 3 close?)

Per state, both MTL B3 cells (C2HGI substrate + HGI counterfactual) must land all 5 folds. Then aggregate paired tests:

- **CH18 cross-state confirmation** — MTL+C2HGI > MTL+HGI on cat F1 at p < 0.05 AND on reg Acc@10_indist at p < 0.05, **per state**.
- Pass acceptance: ≥ 2 of {AL, AZ, FL, CA, TX} significant. AL+AZ+FL already pass; CA+TX would push to **5/5 states** for the paper's strongest CH18 claim.

When CA + TX both close: paper's MTL substrate-specificity table fills with full cross-state data; finalize `PAPER_DRAFT.md` MTL section.

---

## 2 · GPU requirements + parallel strategy

### Why T4 (15 GB) was not enough

CA's reg head has 8497 regions, TX's has 6553. The `next_getnext_hard` head computes `stan_logits + α·transition_prior` where:
- `transition_prior` is `[num_regions, num_regions]` float32 = **274 MB for CA, 164 MB for TX**
- `stan_logits` is `[batch, num_regions]` float32 = **66 MB at bs=2048 for CA**
- Backprop intermediates roughly double the activation footprint

Combined with the cross-attn MTLnet body + AdamW optimizer state + STAN attention buffers, the working set hits ~14.4 GB on T4. We OOM'd at bs={2048, 1024, 512} all reaching ~14.3 GB peak.

### Recommended GPU

**A100 (40 GB)** — 25 GB headroom over the T4 ceiling. Canonical bs=2048 fits with ~10-12 GB margin. ~3× faster than T4. ~$1.10-1.30/hr on Lightning.

A100 (80 GB) and H100 (80 GB) both work but are over-provisioned for this workload (memory-bandwidth bound, not compute-bound). H100's tensor-core advantage doesn't translate proportionally here. **A100 40 GB is the cost-effective sweet spot.**

### Parallel execution + cell packing

The 4 cells are independent — no shared state, no shared dataloaders.

**Per-cell VRAM (canonical hparams, measured on T4 + estimated for A100):**

| Cell type | VRAM @ bs=2048 | Cells per 40 GB A100 | Cells per 80 GB A100 |
|---|---|---|---|
| MTL B3 cross-attn | ~22-25 GB | **1 (safe)** / 2 (tight, ~35 GB) | 3 (comfortable) |
| Cat STL `next_gru` (Phase 2 only — done) | ~6 GB | 5-6 | 11-12 |
| Reg STL `next_getnext_hard` (Phase 2 only — done) | ~10 GB | 3-4 | 7-8 |

For Phase 3 (MTL only):

| Layout | Wall-clock | Total cost (rough) | Setup |
|---|---|---|---|
| 1× A100 40 GB sequential | ~3-5 h | ~$5 | minimal |
| **1 pod × 4 A100 (40 GB) — 1 cell/GPU** ⭐ | **~50 min** | **~$5-6** | minimal |
| 1 pod × 2 A100 (40 GB) — 2 waves | ~2 h | ~$5 | minimal |
| 1 pod × 1 A100 (80 GB) — pack 2 cells/GPU, 2 waves | ~1.5 h | ~$3-4 | minimal |
| 1 pod × 2 A100 (80 GB) — pack 2 cells/GPU, 1 wave | ~50 min | ~$3-4 | minimal |

**Recommended: `1 pod × 4 A100 (40 GB)` if available** — same total compute cost as sequential, ~4-5× faster wall-clock, single bootstrap. The orchestrator `scripts/run_phase3_mtl_parallel.sh` auto-detects GPU count and dispatches accordingly.

If only 1-2 GPUs available, the orchestrator falls back gracefully (2-wave on 2 GPUs, or sequential on 1).

**Cell packing on 80 GB A100:** 2 MTL cells per GPU is feasible but the 4-cell parallel layout on 4× 40 GB is cleaner (no risk of cross-process contention on the same CUDA stream). Only consider packing if 80 GB hardware is what's available.

---

## 3 · Pre-flight checklist

Before launching, confirm in the new pod:

```bash
# Repo
cd /teamspace/studios/this_studio
git clone https://github.com/VitorHugoOli/PoiMtlNet.git
cd PoiMtlNet
git checkout worktree-check2hgi-mtl

# GPU + driver
nvidia-smi  # expect 4× A100 with ≥40 GB each, CUDA 12.x

# Deps (see scripts/setup_lightning_pod.sh — automates this)
bash scripts/setup_lightning_pod.sh
```

After bootstrap, verify upstream data is on disk (the bootstrap downloads it via gdown):

```bash
for state in california texas; do
  for engine in check2hgi hgi; do
    ls output/$engine/$state/embeddings.parquet \
       output/$engine/$state/region_embeddings.parquet \
       output/$engine/$state/input/next.parquet \
       output/$engine/$state/input/next_region.parquet \
       2>/dev/null && echo "  ✓ $engine/$state" || echo "  ✗ $engine/$state INCOMPLETE"
  done
  ls output/check2hgi/$state/region_transition_log.pt 2>/dev/null \
     && echo "  ✓ transition $state" || echo "  ✗ transition $state MISSING"
done
```

Need 5 files per (engine, state) = 8 cells, plus 2 transition matrices. ~12 GB total on disk.

---

## 4 · Launch templates

### 4.1 · Parallel (4× A100 in one pod)

```bash
bash scripts/run_phase3_mtl_parallel.sh
# Launches 4 cells, one per GPU, via CUDA_VISIBLE_DEVICES.
# Logs land in logs/phase3/MTL_B3_{CALIFORNIA,TEXAS}_{check2hgi,hgi}_5f50ep.log
```

### 4.2 · Sequential (1× A100)

```bash
bash scripts/run_phase3_mtl_grid.sh
# Runs CA c2hgi → CA hgi → TX c2hgi → TX hgi sequentially.
# ~5-6 h wall-clock total on a single A100.
```

### 4.3 · Single-cell (manual, e.g. for testing)

```bash
bash scripts/run_phase3_mtl_cell.sh california check2hgi 0
#                                     ^state       ^engine    ^GPU id (CUDA_VISIBLE_DEVICES)
```

All scripts use canonical `bs=2048` (override via `MTL_BATCH_SIZE` env var if needed).

---

## 5 · Post-run extraction + paired tests

```bash
python3 scripts/finalize_phase3.py
```

Extracts per-fold metrics from `results/<engine>/<state>/mtlnet_*/folds/foldN_info.json`, writes to:

- `docs/studies/check2hgi/results/phase1_perfold/{CA,TX}_{check2hgi,hgi}_mtl_{cat,reg}.json`

Then runs paired tests:

- `docs/studies/check2hgi/results/paired_tests/{california,texas}_mtl_{cat_f1,reg_acc10,reg_mrr}.json`

Acceptance criteria evaluation:
- Per state: cat F1 paired Wilcoxon p < 0.05 AND reg Acc@10 paired Wilcoxon p < 0.05.

---

## 6 · Don't

- **Don't push to `main`.** Stay on `worktree-check2hgi-mtl`.
- **Don't change canonical hparams** (lr=1e-4, bs=2048, NashMTL static_weight cat=0.75, d_model=256, num_heads=8). These are pinned by `NORTH_STAR.md` and Phase 2 evidence — different settings would invalidate cross-state CH18 comparison.
- **Don't run on T4 again** — proven insufficient. Even bs=512 OOMs at CA scale.
- **Don't skip the gdown step** — the upstream `output/` data is gitignored, must be re-downloaded into each fresh pod.

## 7 · Cross-references

- [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md) — Phase 2 state + STL closure summary.
- [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md) — Phase 2 cross-state CH16/CH15 verdicts; Phase 3 will append the CH18 closure.
- [`PHASE3_LIGHTNING_HANDOFF.md`](PHASE3_LIGHTNING_HANDOFF.md) — concrete pod-setup step-by-step for the next agent.
- [`NORTH_STAR.md`](NORTH_STAR.md) — pinned MTL B3 hparams (do NOT change).
