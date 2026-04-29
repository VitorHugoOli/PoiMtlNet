# Phase 3 Tracker — Scope D: leakage-free reg STL + MTL CH18 across all 5 states

**Created 2026-04-29.** Phase 2 closed STL (cat) at 5/5 states with CH16 + CH15 reframed. Phase 2's reg STL and MTL data carry the **F44 transition-matrix leakage** (full-dataset transition prior includes val edges; AUDIT-C4 finding). This phase **re-runs all reg STL + MTL cells with per-fold transitions** so the entire reg/MTL evidence chain becomes leakage-free, in one consistent protocol across all 5 states.

> **Why Scope D and not "just CA+TX MTL"**: a leakage-free CA+TX would not be comparable with leaky AL+AZ+FL. To preserve a clean cross-state paired protocol, all reg STL + MTL cells must use the same transition-matrix construction. Scope D fixes everything in one pass.

> **What does NOT need re-running**: cat STL `next_gru` (no transition matrix → no F44 leakage) and the substrate linear probe (head-free, CPU-only). Phase 2 evidence for those stands.

---

## Status board

🟢 = 5-fold complete (leakage-free, `_pf` suffix) · 🟡 = partial · 🔴 = pending.

### Reg STL `next_getnext_hard` (5 states × 2 engines = 10 cells)

| State | C2HGI | HGI | Combined paired |
|---|:-:|:-:|:-:|
| AL | 🔴 | 🔴 | 🔴 |
| AZ | 🔴 | 🔴 | 🔴 |
| FL | 🔴 | 🔴 | 🔴 |
| CA | 🔴 | 🔴 | 🔴 |
| TX | 🔴 | 🔴 | 🔴 |

### MTL B3 cross-attn (5 states × 2 engines = 10 cells)

| State | C2HGI | HGI | Combined paired (cat F1 + reg Acc@10) |
|---|:-:|:-:|:-:|
| AL | 🔴 | 🔴 | 🔴 |
| AZ | 🔴 | 🔴 | 🔴 |
| FL | 🔴 | 🔴 | 🔴 |
| CA | 🔴 | 🔴 | 🔴 |
| TX | 🔴 | 🔴 | 🔴 |

### Reference rows (Phase 2 leaky data — preserved as historical, NOT to be overwritten)

Phase 2 reg STL + MTL data lives at `phase1_perfold/<STATE>_<engine>_{reg_gethard,mtl_{cat,reg}}_5f50ep.json` (no `_pf` suffix). Phase 3 outputs land alongside with `_pf` suffix.

---

## 1 · Acceptance criteria

Per state, both reg STL cells (C2HGI + HGI) and both MTL cells (C2HGI + HGI) must land all 5 folds with per-fold transitions. Then aggregate paired tests:

- **CH18 leakage-free cross-state confirmation** — MTL+C2HGI > MTL+HGI on cat F1 at p<0.05 AND on reg Acc@10_indist at p<0.05, **per state**, at ≥ 2 of {AL, AZ, FL, CA, TX}.
- **CH15 reframing leakage-free** — TOST δ=2pp non-inferiority on reg STL Acc@10 + MRR per state at ≥ 4 of 5 states.
- **Comparison with leaky Phase 2 data** — quantify the absolute Acc@10/MRR shift from removing the F44 leakage; expected to be small (< ~2 pp) and substrate-symmetric (Δ̄ unchanged).

When CA + TX both close: paper's CH18 + CH15 reframing tables fill with leakage-free numbers across all 5 states; Phase 2 leaky tables become the "F44-leaky reference" appendix.

---

## 2 · GPU requirements + parallel strategy

### Per-cell VRAM (canonical hparams, T4-measured + A100-extrapolated)

| Cell type | VRAM (~bs=2048) | Cells per 40 GB A100 | Cells per 80 GB A100 |
|---|---|---|---|
| Reg STL `next_getnext_hard` | ~10 GB | **2-3 in parallel** | 6-7 |
| MTL B3 cross-attn | ~22-25 GB | **1 (safe)** | 3 (comfortable) |

T4 (15 GB) is **not enough** — proven during Phase 2 closure attempt (CA MTL OOM at bs={2048,1024,512}).

### Parallel layout — wall-clock estimates

20 GPU cells (10 reg STL + 10 MTL) on 4× A100 40 GB:

| Layout | Reg STL phase | MTL phase | **Total wall-clock** | Cost (~$1.20/A100·hr) |
|---|---|---|---|---|
| 1× A100 40 GB sequential | ~225 min | ~350 min | **~9.5 h** | ~$11 |
| 2× A100 40 GB | ~110 min (3 waves) | ~175 min (5 waves) | ~4.7 h | ~$11 |
| **4× A100 40 GB** ⭐ | **~50 min (2 waves)** | **~110 min (3 waves)** | **~2.7 h** | **~$13** |
| 4× A100 80 GB (pack reg 2/GPU, MTL 2/GPU) | ~25 min | ~90 min | ~2.0 h | ~$18 |

**Recommended: 4× A100 (40 GB).** Best speed-vs-cost. Orchestrator `scripts/run_phase3_parallel.sh` auto-detects GPU count and dispatches accordingly.

### Smart scheduling option

Run reg STL on GPUs 0+1 simultaneously with MTL on GPUs 2+3. Bottleneck becomes MTL (~110 min) since reg STL finishes early. Net: ~2 h end-to-end on 4× A100. Implementation TBD (current orchestrator does step 2 → step 3 sequentially, which is simpler and only ~50 min slower).

---

## 3 · Pre-flight checklist

Before launching, in the new pod:

```bash
# 1. Repo on the right branch
cd /teamspace/studios/this_studio
git clone https://github.com/VitorHugoOli/PoiMtlNet.git
cd PoiMtlNet
git checkout worktree-check2hgi-mtl

# 2. GPU
nvidia-smi   # expect ≥1× A100 with ≥40 GB

# 3. Bootstrap (deps + gdown all 5 states)
#    Default downloads CA+TX. For full Scope D, also pass AL/AZ/FL gdrive IDs:
STATES="alabama arizona florida california texas" \
  AL_C2HGI_GDID=<id> AL_HGI_GDID=<id> \
  AZ_C2HGI_GDID=<id> AZ_HGI_GDID=<id> \
  FL_C2HGI_GDID=<id> FL_HGI_GDID=<id> \
  bash scripts/setup_lightning_pod.sh
```

Bootstrap is idempotent — re-running skips already-installed deps and already-downloaded data.

After bootstrap, the script verifies all 10 upstream parquets + 5 transition matrices are on disk and prints the next-step CLI.

---

## 4 · Launch templates

### 4.1 · Step 1 — per-fold transitions (CPU, ~5 min per state)

```bash
bash scripts/build_phase3_per_fold_transitions.sh
# Or just a subset:
STATES="california texas" bash scripts/build_phase3_per_fold_transitions.sh
```

Builds `output/check2hgi/<state>/region_transition_log_fold{1..5}.pt` from `StratifiedGroupKFold(seed=42)` train-only edges. Idempotent.

### 4.2 · Step 2 — parallel grid (auto-dispatch)

```bash
nohup bash scripts/run_phase3_parallel.sh > logs/phase3/orchestrator.log 2>&1 &
echo $! > logs/phase3/orchestrator.pid
```

Auto-detects GPU count:
- ≥2 GPUs: dispatches reg STL waves first, then MTL waves
- 1 GPU: falls back to `scripts/run_phase3_grid.sh` (sequential)

### 4.3 · Step 2 alternative — sequential grid (1 GPU)

```bash
nohup bash scripts/run_phase3_grid.sh > logs/phase3/orchestrator.log 2>&1 &
```

Runs all 20 cells sequentially: reg STL (10) then MTL (10). ~9-10 h on A100.

### 4.4 · Step 3 — finalize

```bash
python3 scripts/finalize_phase3.py
```

Extracts per-fold metrics, runs paired tests with `_pf` suffix (preserves leaky Phase 2 data), prints cross-state CH15 + CH18 status board.

### 4.5 · Single-cell (manual, e.g. for testing)

```bash
# scripts/run_phase3_reg_stl_cell.sh STATE ENGINE GPU_ID
bash scripts/run_phase3_reg_stl_cell.sh california check2hgi 0

# scripts/run_phase3_mtl_cell.sh STATE ENGINE GPU_ID
bash scripts/run_phase3_mtl_cell.sh california check2hgi 0
```

---

## 5 · Output naming convention

Phase 3 outputs use a **`_pf` suffix** to clearly distinguish from leaky Phase 2 data. Both coexist in the repo so reviewers can compare:

| Leaky Phase 2 | Leakage-free Phase 3 |
|---|---|
| `phase1_perfold/<STATE>_<engine>_reg_gethard_5f50ep.json` | `phase1_perfold/<STATE>_<engine>_reg_gethard_pf_5f50ep.json` |
| `phase1_perfold/<STATE>_<engine>_mtl_cat.json` | `phase1_perfold/<STATE>_<engine>_mtl_cat_pf.json` |
| `phase1_perfold/<STATE>_<engine>_mtl_reg.json` | `phase1_perfold/<STATE>_<engine>_mtl_reg_pf.json` |
| `paired_tests/<state>_reg_acc10.json` | `paired_tests/<state>_reg_acc10_pf.json` |
| `paired_tests/<state>_mtl_cat_f1.json` | `paired_tests/<state>_mtl_cat_f1_pf.json` |
| `P1/region_head_<state>_*_STL_<STATE>_<engine>_reg_gethard_5f50ep.json` | `P1/region_head_<state>_*_STL_<STATE>_<engine>_reg_gethard_pf_5f50ep.json` |

Run dir names also include `_pf` in the tag (e.g. `STL_TEXAS_check2hgi_reg_gethard_pf_5f50ep` and `MTL_B3_TEXAS_check2hgi_pf_5f50ep`).

---

## 6 · Don't

- **Don't change canonical hparams** (lr=1e-4, bs=2048, NashMTL static_weight α_cat=0.75, model=mtlnet_crossattn, cat_head=next_gru, reg_head=next_getnext_hard with d_model=256 num_heads=8, 5-fold StratifiedGroupKFold(userids, seed=42), 50 epochs). Same as Phase 2 — only the transition matrix changes.
- **Don't overwrite Phase 2 leaky data.** Use `_pf` suffix; both must coexist.
- **Don't run on T4** — proven OOM at any batch size for CA-scale MTL.
- **Don't push to `main`.** Stay on `worktree-check2hgi-mtl`.
- **Don't re-run cat STL or probe** — they don't use a transition matrix and are already leakage-free per the F44 audit.

## 7 · Drive backup after closure

After Phase 3 completes, bundle gitignored artefacts (run dirs + logs) for Drive:

```bash
BUNDLE=/teamspace/studios/this_studio/phase3_drive_bundle_$(date +%Y-%m-%d)
mkdir -p $BUNDLE/{results,logs,output_per_fold_transitions}
# Run dirs
cp -r results/{check2hgi,hgi}/{alabama,arizona,florida,california,texas}/{mtlnet_*_pf*,$(echo {alabama,arizona,florida,california,texas} | xargs -n1 -I{} echo "next_*")} $BUNDLE/results/ 2>/dev/null
# Logs
cp -r logs/phase3 $BUNDLE/logs/
# Per-fold transition matrices (gitignored under output/)
for state in alabama arizona florida california texas; do
    [ -f "output/check2hgi/$state/region_transition_log_fold1.pt" ] || continue
    mkdir -p $BUNDLE/output_per_fold_transitions/$state
    cp output/check2hgi/$state/region_transition_log_fold*.pt $BUNDLE/output_per_fold_transitions/$state/
done
cd /teamspace/studios/this_studio
tar czf phase3_drive_bundle_$(date +%Y-%m-%d).tar.gz $(basename $BUNDLE)/
```

Then upload that single tarball to Drive `mestrado_data/PoiMtlNet/phase3_archives/`.

Per-fold and paired-test JSONs are git-tracked (no Drive needed):
```bash
git add docs/studies/check2hgi/results/{phase1_perfold/*_pf*,paired_tests/*_pf.json,P1/*_pf*} \
        docs/studies/check2hgi/{PHASE3_TRACKER.md,research/SUBSTRATE_COMPARISON_FINDINGS.md}
git commit -m "study(check2hgi): Phase 3 Scope D — leakage-free reg STL + MTL CH18 closed"
git push origin worktree-check2hgi-mtl
```

## 8 · Cross-references

- [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md) — Phase 2 STL closure (cat + probe, leakage-free already).
- [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md) — Phase 2 cross-state CH16/CH15 verdicts; Phase 3 will append the leakage-free CH15+CH18 closure section.
- [`PHASE3_LIGHTNING_HANDOFF.md`](PHASE3_LIGHTNING_HANDOFF.md) — concrete pod-setup step-by-step.
- [`NORTH_STAR.md`](NORTH_STAR.md) — pinned MTL B3 hparams (do NOT change).
- F44 in `FOLLOWUPS_TRACKER.md` — the audit finding this phase resolves.
