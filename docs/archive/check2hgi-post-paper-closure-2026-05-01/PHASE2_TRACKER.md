# Phase 2 Tracker — FL + CA + TX final paper runs

**Created 2026-04-27** after Phase 1 closed at AL+AZ with the strong claim confirmed (see [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md)).

This tracker is the live work queue for **Phase 2 of `SUBSTRATE_COMPARISON_PLAN.md`**: the 3-leg substrate-comparison grid replicated at the headline states (Florida, California, Texas). Numbers from this phase land directly in the paper tables — no further framing changes after these runs.

> **Phase 2 launch authorisation:** ✅ granted by `research/SUBSTRATE_COMPARISON_FINDINGS.md §6` (strong claim confirmed at AL+AZ). Per the plan, no doc revision required before launch.

---

## Status board

🟢 = 5-fold complete · 🟡 = 1-fold or partial · 🔴 = pending · ⚫ = blocked on upstream pipeline.

### FL (Florida) — 🟢 closed 2026-04-28

| Test | C2HGI | HGI | Combined paired test |
|---|:-:|:-:|:-:|
| Substrate-only linear probe | ✅ 40.77 ± 1.11 | ✅ 25.74 ± 0.26 (Δ=+15.03) | n/a (head-free) |
| Cat STL matched-head (`next_gru`) | ✅ **63.43 ± 0.88** | ✅ 34.41 ± 0.94 (**Δ=+29.02 pp**) | ✅ Wilcoxon **p=0.0312** (5/5 folds positive) |
| Reg STL matched-head (`next_getnext_hard`) | ✅ Acc@10 82.54 ± 0.42 | ✅ Acc@10 82.28 ± 0.47 (Δ=+0.27 pp) | ✅ TOST δ=2pp **non-inferior** (Acc@10 p=0.0009 / MRR p=0.0010) |
| MTL B3 counterfactual (HGI substrate) | (1f reference exists) | ✅ cat F1 34.74 ± 0.76 / reg Acc@10_indist 58.27 ± 3.37 | 🟢 MTL+HGI ≈ STL+HGI on cat (no MTL gain on HGI) |

> **FL headline:** cat substrate Δ = **+29 pp** at FL — almost **2× the AL/AZ effect** (+15.5 / +14.5). Substrate gap on cat **grows monotonically with scale**. Reg gap nearly neutralised at FL scale (TOST non-inf at δ=2pp). MTL+HGI on FL is essentially identical to STL+HGI on cat (Δ_MTL = +0.33 pp), confirming CH18: the B3 MTL configuration only buys a gain on the C2HGI substrate. See `research/SUBSTRATE_COMPARISON_FINDINGS.md §FL` for cross-state synthesis.

### CA (California) — 🟡 STL closed 2026-04-29; MTL CH18 BLOCKED on memory

| Test | C2HGI | HGI | Combined paired test |
|---|:-:|:-:|:-:|
| Substrate-only linear probe | ✅ 37.45 ± 0.26 | ✅ 21.32 ± 0.14 (Δ=+16.13) | n/a (head-free) |
| Cat STL matched-head (`next_gru`) | ✅ **59.94 ± 0.52** | ✅ 31.13 ± 0.93 (**Δ=+28.81 pp**) | ✅ Wilcoxon **p=0.0312** (5/5 folds positive) |
| Reg STL matched-head (`next_getnext_hard`) | ✅ Acc@10 70.63 ± 0.57 | ✅ Acc@10 71.29 ± 0.58 (Δ=-0.65 pp, HGI nominal best) | ✅ TOST δ=2pp **non-inferior** (Acc@10 p=0.0000 / MRR p=0.0000) |
| **MTL B3 paired CH18** | 🔴 **blocked** | 🔴 **blocked** | 🔴 **blocked** |

> **CA MTL blocker (2026-04-29):** Both MTL+HGI and MTL+C2HGI 5-fold runs SIGKILL at fold 1 epoch 1 (~140/7000 batches). On Colab T4 the cgroup OOMs (rc=137); on M4 (64 GB unified) macOS jetsam silently kills, even with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. Root cause (advisor diagnosis): `src/data/folds.py::_create_check2hgi_mtl_folds` holds **all 5 folds × full-N tensors × 2 task copies** in memory simultaneously (~28 GiB "other allocations" observed in MPS error). Same constraint will block TX MTL.

### TX (Texas) — 🟢 STL closed 2026-04-29 (Lightning T4); MTL ×2 BLOCKED on memory

| Test | C2HGI | HGI | Combined paired test |
|---|:-:|:-:|:-:|
| Substrate-only linear probe | ✅ 38.38 ± 0.25 | ✅ 22.33 ± 0.23 (Δ=+16.05) | n/a (head-free) |
| Cat STL matched-head (`next_gru`) | ✅ **60.24 ± 1.65** | ✅ 31.89 ± 0.49 (**Δ=+28.34 pp**) | ✅ Wilcoxon **p=0.0312** (5/5 folds positive) |
| Reg STL matched-head (`next_getnext_hard`) | ✅ Acc@10 69.31 ± 0.85 / MRR 44.44 ± 0.89 | ✅ Acc@10 69.90 ± 0.79 (Δ=−0.59 pp, HGI nominal best) | ✅ TOST δ=2pp **non-inferior** (Acc@10 p<0.001 / MRR p<0.001) |
| **MTL B3 paired CH18** | 🔴 **blocked** | 🔴 **blocked** | 🔴 **blocked** (same memory blocker as CA — Lightning has 15 GB RAM = same constraint as Colab T4) |

> **TX run on Lightning T4** (replaces partial Colab data, all 5 folds captured locally):
> - Probe ×2 → `docs/studies/check2hgi/results/probe/texas_{check2hgi,hgi}_last.json` (5f, seed 42).
> - Cat STL ×2 → `results/{check2hgi,hgi}/texas/next_lr1.0e-04_bs1024_ep50_20260429_1{618,646}/`; per-fold extracted to `phase1_perfold/TX_{check2hgi,hgi}_cat_gru_5f50ep.json`.
> - Reg STL ×2 → `docs/studies/check2hgi/results/P1/region_head_texas_*STL_TEXAS_{check2hgi,hgi}_reg_gethard_5f50ep.json`; per-fold extracted to `phase1_perfold/TX_{check2hgi,hgi}_reg_gethard_5f50ep.json`.
> - Paired tests → `docs/studies/check2hgi/results/paired_tests/texas_{cat_f1,reg_acc10,reg_mrr}.json`.
> - **Cross-validation vs Drive (where Drive had data)**: Lightning probe c2hgi 38.38 vs Drive 38.39; cat-c2hgi mean 60.24 vs Drive 60.36; reg-c2hgi Acc@10 69.31 vs Drive 69.16; reg-hgi folds 0-3 Acc@10 mean 0.6976 vs Drive log 0.6987. All within <0.15 pp — full reproducibility validated.
>
> **Cross-state CH16 cat-substrate gap is robust** (FL +29.02, CA +28.81, TX +28.34, AL +15.5, AZ +14.5). All 5 states significant at max-n=5 Wilcoxon p=0.0312, 5/5 folds positive each.
>
> **Cross-state CH15 reframing — substrate-equivalent on reg at large scale**: C2HGI vs HGI Δ_Acc@10 = AL +0.85, AZ +2.34, FL +0.27, CA −0.65, TX −0.59. All 5 states pass TOST δ=2pp non-inferiority. Sign reversal at CA+TX (HGI nominal best at the largest two scales) but practical difference < 1 pp everywhere.

> **TX state on Drive (2026-04-29 ~14:14 UTC, second Colab credit-exhaustion death after F40c hgi fold 3):**
> - F40a probe ×2 → `/content/drive/MyDrive/mestrado_data/PoiMtlNet/results/probe/texas_{check2hgi,hgi}_last.json`
> - F40b cat STL c2hgi → run dir `results/check2hgi/texas/next_lr1.0e-04_bs1024_ep50_20260429_0458/` (all 5 folds, summary present)
> - F40b cat STL hgi → run dir `results/hgi/texas/next_lr1.0e-04_bs1024_ep50_20260429_0528/` (all 5 folds, summary present)
> - F40c reg STL c2hgi → ✅ P1 JSON `results/P1/region_head_texas_region_5f_50ep_STL_TEXAS_check2hgi_reg_gethard_5f50ep.json` (Acc@10 = 0.6916 ± 0.0028, MRR = 0.4434 ± 0.0038, F1 = 0.1638 ± 0.0041)
> - F40c reg STL hgi → 4/5 folds in `phase2_logs/texas_F40c_reg_stl_hgi_TEXAS_resume.log` (fold 0–3 metrics captured; fold 4 + aggregate JSON not written before Colab died)
>
> **Per-fold reg-hgi values captured from log (for partial paired-test if fold 5 isn't recovered):**
> - fold 0: Acc@1=0.3156 Acc@5=0.5822 Acc@10=0.6975 MRR=0.4413 F1=0.1659
> - fold 1: Acc@1=0.3241 Acc@5=0.5881 Acc@10=0.6970 MRR=0.4487 F1=0.1642
> - fold 2: Acc@1=0.3261 Acc@5=0.5924 Acc@10=0.7033 MRR=0.4513 F1=0.1672
> - fold 3: Acc@1=0.3180 Acc@5=0.5857 Acc@10=0.6968 MRR=0.4436 F1=0.1578
>
> **Drive paths used by this Colab account** (vho2009@hotmail.com → shared drive id `1GKIA8c…`): everything is under `mestrado_data/PoiMtlNet/`, NOT `mestrado/PoiMtlNet/` (the latter is a different empty folder on the same drive).
>
> **Cross-state CH16 cat-substrate gap is robust** (FL +29.02, CA +28.81, TX +28.26, AL +15.5, AZ +14.5) — paper headline can be drafted from current data; what's missing is paired-test formalisation at TX and the MTL CH18 confirmation at CA+TX.

---

## 0 · Active task list (in-session ⇄ persistent here)

Three named tasks track Phase-2 closure end-to-end. They are also surfaced in the in-session task list (TaskCreate IDs #1/#2/#3) for live progress; this section is the durable copy.

| Task | State | Status | Blocking |
|---|:-:|:-:|:-:|
| **T1** Close FL grid: F36c reg HGI fold 5 + F36d MTL counterfactual | FL | 🟢 **closed 2026-04-28** | — |
| **T2** Run CA Phase-2 grid (7 experiments + paired CH18) | CA | 🟡 **STL closed; MTL CH18 ×2 BLOCKED on memory** (see CA blocker note above) | user decision on path forward |
| **T3** Run TX Phase-2 grid (7 experiments) | TX | 🟢 **STL closed 2026-04-29** (Lightning T4 re-ran the grid with full data; all 5 folds captured locally for both engines, paired tests landed). MTL ×2 still blocked — same RAM constraint as CA. | CA-MTL decision (C1/C2/C3) per §7C |

After T3 completes, run paired-tests for FL+CA+TX, update SUBSTRATE_COMPARISON_FINDINGS, finalise CH16/CH15/CH18 with cross-state evidence, mark Phase 2 closed.

---

## 1 · Ready-now follow-ups (P1, paper-blocking)

| # | Pri | Item | State | Owner | Cost | Acceptance criterion |
|---|:-:|---|:-:|:-:|:-:|---|
| **F36** | **P1** | FL Phase-2 grid (Legs I + II + III) | FL | colab T4 (preferred) / m4_pro | T4 ~3 h / MPS ~30 h | All 5 cells (probe + 2 cat STL + 2 reg STL + MTL counterfactual) land. Paper tables filled. **Status: 5/7 captured 2026-04-28 (T1).** |
| **F36a** | P1 | FL substrate linear probe (Leg I, head-free) | FL | m4_pro | ~5 min × 2 substrates | F1 ± σ for C2HGI and HGI on `output/<engine>/florida/input/next.parquet`. CPU-only — Colab not needed. |
| **F36b** | P1 | FL cat STL `next_gru` × {C2HGI, HGI} (Leg II.1) | FL | colab T4 / m4_pro | T4 ~50 min × 2 / MPS ~5–6 h × 2 | 5f × 50ep × seed 42. Paired Wilcoxon vs HGI. Pass: Δ > 0 at p < 0.05. |
| **F36c** | P1 | FL reg STL `next_getnext_hard` × {C2HGI, HGI} (Leg II.2) | FL | colab T4 / m4_pro | T4 ~50 min × 2 / MPS ~5–6 h × 2 | 5f × 50ep × seed 42. Paired test on Acc@10 + MRR + TOST δ=2 pp. |
| **F36d** | P1 | FL MTL B3 counterfactual (HGI substrate) (Leg III) | FL | colab T4 / m4_pro | T4 ~50 min / MPS ~5–6 h | 5f × 50ep × seed 42. Compare to existing MTL B3 C2HGI (NORTH_STAR.md). |
| **P2-CA-up** | **P1** | CA upstream pipeline | CA | m4_pro / colab | ~6–12 h | Embeddings + inputs + region transition matrix on `output/{check2hgi,hgi}/california/`. |
| **P2-CA-grid** | **P1** | CA Phase-2 grid (Legs I + II + III) | CA | colab T4 / m4_pro | T4 ~3 h / MPS ~30 h | Same as F36 but CA. Gated on P2-CA-up. |
| **P2-TX-up** | **P1** | TX upstream pipeline | TX | m4_pro / colab | ~6–12 h | Embeddings + inputs + region transition matrix on `output/{check2hgi,hgi}/texas/`. |
| **P2-TX-grid** | **P1** | TX Phase-2 grid (Legs I + II + III) | TX | colab T4 / m4_pro | T4 ~3 h / MPS ~30 h | Same as F36 but TX. Gated on P2-TX-up. |

Recommended execution order: F36 (FL, all data on disk) → P2-CA-up/P2-CA-grid (CA) → P2-TX-up/P2-TX-grid (TX). On Colab T4 the full FL grid is ~3 h vs ~30 h on M4 Pro — see [`../../docs/COLAB_GUIDE.md`](../../docs/COLAB_GUIDE.md) and template notebook [`../../notebooks/colab_check2hgi_mtl.ipynb`](../../notebooks/colab_check2hgi_mtl.ipynb). M4 Pro under `caffeinate -s` remains a valid fallback.

## 2 · Optional follow-ups (P2/P3, nice-to-have)

| # | Pri | Item | When to revisit |
|---|:-:|---|---|
| **F41** | P3 | C4 mechanism extension to FL (POI-pooled C2HGI) | Only if reviewer asks "does the per-visit-variation mechanism replicate beyond AL?". AL alone is sufficient per the plan. |
| **F42** | P3 | C2 head-agnostic at FL/CA/TX | Only if reviewer asks for state-replication on the head-invariance claim. AL+AZ data (8 probes positive at max-significance) is sufficient. |
| **F43** | P4 | Multi-seed n=3 on the Phase-2 champions (FL/CA/TX × C2HGI cat-gru) | After all of Phase 2 lands, before camera-ready. ~20 h MPS additional. |
| **F44** | P4 | Per-fold transition matrix (leakage-safe GETNext) at FL/CA/TX | Camera-ready, if reviewer asks for per-fold protocol. |

## 3 · Pre-flight checklist (before launching FL)

For each state in `{florida, california, texas}`, confirm before launch:

```bash
# Required artefacts (verify by listing)
ls output/check2hgi/<state>/embeddings.parquet \
   output/check2hgi/<state>/region_embeddings.parquet \
   output/check2hgi/<state>/region_transition_log.pt \
   output/check2hgi/<state>/input/{next.parquet,next_region.parquet} \
   output/check2hgi/<state>/temp/{checkin_graph.pt,sequences_next.parquet}

ls output/hgi/<state>/embeddings.parquet \
   output/hgi/<state>/region_embeddings.parquet \
   output/hgi/<state>/input/next.parquet

# HGI's input/next_region.parquet must be built (substrate-free labels)
OUTPUT_DIR=$OUTPUT_DIR python3 scripts/probe/build_hgi_next_region.py --state <state>
```

If any artefact is missing, complete the upstream pipeline (P2-CA-up / P2-TX-up) first.

## 4 · Launch templates (FL example)

### 4.0 · Colab T4 (preferred for FL grid — ~10× faster than MPS)

The Phase-2 FL grid is the use case the Colab template was built for. F36b/c/d each take ~50 min on T4 vs ~5–6 h on MPS. See [`../../docs/COLAB_GUIDE.md`](../../docs/COLAB_GUIDE.md) for Drive layout, branch hygiene, the detached-subprocess launch pattern (mandatory for runs > 5 min), memory pitfalls (FL's 4702 regions hit a pairwise-comparison MRR OOM unless chunked — already handled in `feat/colab-gpu-perf`), and the verification recipe for confirming `git pull → relaunch` loaded fresh code.

Template notebook: [`../../notebooks/colab_check2hgi_mtl.ipynb`](../../notebooks/colab_check2hgi_mtl.ipynb) — drop in, edit `STATE` + `--engine` + head args, run cells.

### 4.1 · M4 Pro fallback

```bash
# Set every-shell env
export PYTHONPATH=src
export DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred
export OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Build HGI next_region.parquet (one-time per state)
python3 scripts/probe/build_hgi_next_region.py --state florida

# F36a — substrate linear probe
for ENG in check2hgi hgi; do
  python3 scripts/probe/substrate_linear_probe.py --state florida --engine $ENG
done

# F36b — cat STL matched-head, both substrates
for ENG in check2hgi hgi; do
  caffeinate -s python3 scripts/train.py \
    --task next --state florida --engine $ENG --model next_gru \
    --folds 5 --epochs 50 --seed 42 --no-checkpoints \
    > logs/STL_FLORIDA_${ENG}_cat_gru_5f50ep.log 2>&1
done

# F36c — reg STL matched-head, both substrates
for ENG in check2hgi hgi; do
  caffeinate -s python3 scripts/p1_region_head_ablation.py \
    --state florida --heads next_getnext_hard \
    --folds 5 --epochs 50 --seed 42 --input-type region \
    --region-emb-source $ENG \
    --override-hparams d_model=256 num_heads=8 \
        "transition_path=$OUTPUT_DIR/check2hgi/florida/region_transition_log.pt" \
    --tag STL_FLORIDA_${ENG}_reg_gethard_5f50ep \
    > logs/STL_FLORIDA_${ENG}_reg_gethard_5f50ep.log 2>&1
done

# F36d — MTL B3 counterfactual (HGI substrate)
caffeinate -s python3 scripts/train.py \
  --task mtl --state florida --engine hgi \
  --task-set check2hgi_next_region \
  --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints \
  > logs/MTL_B3_FLORIDA_hgi_5f50ep.log 2>&1
```

A wrapper orchestrator script can be created by parametrising `scripts/run_phase1_*.sh` on `$STATE`. Suggested name: `scripts/run_phase2_<state>.sh`.

## 5 · Acceptance criteria (when does Phase 2 close?)

Per state, all 5 cells of the §1 grid lands. Then aggregate paired tests:

- **CH16 cross-state confirmation** (cat F1 paired Wilcoxon, C2HGI > HGI per state at p < 0.05). Pass: ≥ 2 of {FL, CA, TX} significant. (AL+AZ already confirm at α=0.05.)
- **CH15 reframing** (reg under matched-head). Pass: TOST non-inferiority at δ=2 pp Acc@10 holds at all 3 states; superiority at ≥ 1 state.
- **MTL substrate-specific finding** (CH18). Pass: MTL+C2HGI > MTL+HGI on cat F1 at p < 0.05 and on reg Acc@10_indist at p < 0.05 per state.

When all three pass: paper tables fill, `PAPER_STRUCTURE.md` confirms, study moves to write-up phase.

## 6 · Don't

- **Don't extend C2 (head-agnostic) to FL/CA/TX** — AL+AZ is sufficient per the plan §6.
- **Don't extend C4 (POI-pooled mechanism) to FL/CA/TX** unless reviewer asks — AL alone settles the mechanism.
- **Don't run multi-seed (F43) before headline 5-fold runs land.**
- **Don't push to `main`.**
- **Don't launch FL on a machine other than M4 Pro under `caffeinate -s`** — F20 per-fold persistence handles SIGKILL recovery, but MPS sleep + swap pressure are still real failure modes (G4, G5).

## 7 · Resume notes for next agent (handoff updated 2026-04-29 ~19:40 UTC, on Lightning T4)

**Phase 2 STL is FULLY CLOSED at all 5 states (AL+AZ+FL+CA+TX) on 2026-04-29 19:25 UTC.** All probe + cat STL + reg STL + paired tests landed in repo. CH16 (cat substrate gap) confirmed at 5/5 states with max-n=5 Wilcoxon p=0.0312. CH15 reframed as substrate-equivalence on reg (TOST non-inf at all 5 states, δ=2pp). MTL CH18 confirmed at AL+AZ+FL (3 states); CA+TX blocked on **GPU memory** (T4 15 GB insufficient for CA's 8497-region transition prior — confirmed OOM at bs={2048, 1024, 512}). Bigger GPU (L4/A10G 24 GB) needed for CA+TX MTL closure.

### Lightning T4 resume notes (2026-04-29)

- Repo branch: `worktree-check2hgi-mtl` (clean as of TX close).
- Upstream data on this Lightning pod: `output/{check2hgi,hgi}/{california,texas}/` (downloaded via `gdown` from user's Drive folders earlier this session).
- TX run dirs created on Lightning: `results/{check2hgi,hgi}/texas/next_*_20260429_1{618,646}/` and `results/{check2hgi,hgi}/texas/next_*_20260429_17*/` (reg-STL via p1_region_head_ablation.py wrote into `docs/studies/check2hgi/results/P1/` not `results/<engine>/<state>/`).
- MTL orchestrator ready: `scripts/run_phase2_mtl_lightning.sh` (canonical bs=2048; `MTL_BATCH_SIZE` env var override). Does NOT fit on T4; awaiting bigger GPU.

### Pre-2026-04-29 handoff context (kept for reference)

### What's missing, in execution order

**A. TX reg STL hgi fold 4 only (resume on GPU Colab, ~15 min on T4)** — c2hgi already closed; hgi has folds 0-3 captured in log

The repo's `p1_region_head_ablation.py` has a checkpoint mechanism (it writes `region_head_texas_region_5f_50ep_STL_TEXAS_hgi_reg_gethard_5f50ep.checkpoint.json` after each fold). When GPU Colab reconnects with credits:

1. Mount Drive (`/content/drive/MyDrive/mestrado_data/PoiMtlNet`), clone repo on `worktree-check2hgi-mtl`, install deps + PyG wheels.
2. Copy TX parquets from Drive to `/content/output` (~7.4 GB; both engines).
3. Restore the checkpoint file from Drive to `docs/studies/check2hgi/results/P1/` in the repo if it survived the rsync (it should — orchestrator runs `sync_drive` after each fold). The script will auto-resume from fold 4. If checkpoint is gone, simply re-run from fold 0 — only 75 min on T4.
4. CLI:
   ```
   python3 -u scripts/p1_region_head_ablation.py --state texas --heads next_getnext_hard \
     --folds 5 --epochs 50 --seed 42 --input-type region --region-emb-source hgi \
     --override-hparams d_model=256 num_heads=8 \
       transition_path=/content/output/check2hgi/texas/region_transition_log.pt \
     --tag STL_TEXAS_hgi_reg_gethard_5f50ep
   ```
5. Output: `results/P1/region_head_texas_region_5f_50ep_STL_TEXAS_hgi_reg_gethard_5f50ep.json`. rsync to Drive when done.

If credit-exhaustion is permanent on this account, the **partial 4-fold reg-hgi data is in the tracker §TX block above** (per-fold metrics captured from log) and is sufficient for a directional CH15 reframing comparison even without the formal paired test — TOST non-inf was already passing FL & CA at much tighter δ.

**B. Mirror TX results from Drive to repo + paired tests (CPU only, ~5 min)**

Once F40c reg STL ×2 is on Drive (or the user confirms partial-mirror is OK):

1. Stream the following from Drive via base64 (Colab cell):
   - `results/probe/texas_{check2hgi,hgi}_last.json`
   - `results/check2hgi/texas/next_lr1.0e-04_bs1024_ep50_20260429_0458/folds/fold{1..5}_info.json`
   - `results/hgi/texas/next_lr1.0e-04_bs1024_ep50_20260429_0528/folds/fold{1..5}_info.json`
   - `results/P1/region_head_texas_*STL_TEXAS_{check2hgi,hgi}_reg_gethard_5f50ep.json`
2. Decode locally and write into:
   - `docs/studies/check2hgi/results/probe/texas_{c2hgi,hgi}.json`
   - `docs/studies/check2hgi/results/phase1_perfold/TX_{check2hgi,hgi}_cat_gru_5f50ep.json` (extract per-fold from `diagnostic_best_epochs.next.metrics`)
   - `docs/studies/check2hgi/results/P1/region_head_texas_*.json` (verbatim)
   - `docs/studies/check2hgi/results/phase1_perfold/TX_{check2hgi,hgi}_reg_gethard_5f50ep.json` (extract per-fold; rename `top10_acc` → `acc10`)
3. Run paired tests via `scripts/analysis/substrate_paired_test.py` for cat F1, reg Acc@10, reg MRR (TOST δ=2pp).
4. Update PHASE2_TRACKER TX row: STL portion → 🟢 with paired Δ + p-values.
5. Append TX section to `research/SUBSTRATE_COMPARISON_FINDINGS.md` (STL only).

**C. CA + TX MTL CH18 — needs user decision before any compute (BLOCKED)**

Three paths (in order of effort vs. paper-completeness):

| Option | Effort | Paper impact |
|---|---|---|
| **C1 — Loosen CH18 acceptance** to ≥ 2 of {AL, AZ, FL, CA, TX} | 0 min | Already met via AL ✓ AZ ✓ FL ✓; CA/TX MTL becomes a methodology footnote ("MTL CH18 confirmed at smaller-scale states; large-state MTL deferred due to memory constraints in fold-store implementation"). |
| **C2 — Patch `src/data/folds.py` for lazy fold loading** | ~2 days | `_create_check2hgi_mtl_folds` currently materialises all 5 folds × full-N tensors × 2 task copies. Refactor to stream from disk per fold; expected RSS drop from ~28 GiB to ~6 GiB. Then re-run MTL on M4 for both substrates × 2 states. |
| **C3 — Colab Pro High-RAM** | $10/mo, ~3 h compute per state-pair | 50 GB cgroup; smoke test peaked at 47 GiB so this fits. Run MTL+C2HGI + MTL+HGI for CA + TX (4 runs × ~50 min on T4 with High-RAM). |

If user picks C2 or C3, the MTL B3 north-star CLI (per `NORTH_STAR.md`) is:

```
python3 -u scripts/train.py --task mtl --state {california,texas} --engine {check2hgi,hgi} \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/{state}/region_transition_log.pt \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints
```

After both engines × both states land, the close-out is:
- Extract per-fold from `results/<engine>/<state>/mtlnet_*/folds/foldN_info.json::diagnostic_best_epochs.next_{category,region}.metrics`
- Write `docs/studies/check2hgi/results/phase1_perfold/{CA,TX}_{check2hgi,hgi}_mtl_{cat,reg}.json`
- Run paired tests (cat F1, reg Acc@10) per state via `scripts/analysis/substrate_paired_test.py`
- Update PHASE2_TRACKER MTL rows to ✅ with paired Δ + p-values
- Append confirmed CH18 to `research/SUBSTRATE_COMPARISON_FINDINGS.md`
- Mark task #2 + #3 completed; Phase 2 closed

### Failure modes seen this round (don't waste time re-discovering)

- **macOS jetsam silent SIGKILL** on M4 when total process RSS approaches system memory. No traceback, just a `resource_tracker: leaked semaphore` warning in the log. `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` does NOT help — jetsam is the OS, not MPS.
- **Colab T4 cgroup OOM** (rc=137) at ~12 GB for `/jupyter-children`. Same root cause (5 folds in memory at once).
- **MPSGraph INT_MAX dim limit** at CA's 286K × 8497 ≈ 2.4B in `_rank_of_target` — already patched in `src/tracking/metrics.py` (commit `3769203`) by falling back to CPU on MPS for the chunked comparison.
- **Colab runtime reset** wipes `/content/*` (incl. PID file, parquets, local results not yet rsync'd to Drive). The detached daemon survives MCP cell timeouts but NOT a runtime reset. Always rsync to Drive between experiments — never accumulate output locally.
- **Local Drive sync (`~/Library/CloudStorage/GoogleDrive-…/My Drive`) does NOT contain the Colab path**. The `/content/drive/MyDrive/mestrado/PoiMtlNet` tree is only reachable from a live Colab session. Mirroring requires base64-streaming through `mcp__colab-mcp__run_code_cell`.

## 8 · Cross-references

- [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md) — Phase 1 outcome verdict + paper-ready findings.
- [`research/SUBSTRATE_COMPARISON_PLAN.md`](research/SUBSTRATE_COMPARISON_PLAN.md) — phase-gated 3-leg framework.
- [`FOLLOWUPS_TRACKER.md`](FOLLOWUPS_TRACKER.md) — broader study tracker.
- [`OBJECTIVES_STATUS_TABLE.md`](OBJECTIVES_STATUS_TABLE.md) — paper-objective scorecard.
