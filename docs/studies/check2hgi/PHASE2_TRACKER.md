# Phase 2 Tracker — FL + CA + TX final paper runs

**Created 2026-04-27** after Phase 1 closed at AL+AZ with the strong claim confirmed (see [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md)).

This tracker is the live work queue for **Phase 2 of `SUBSTRATE_COMPARISON_PLAN.md`**: the 3-leg substrate-comparison grid replicated at the headline states (Florida, California, Texas). Numbers from this phase land directly in the paper tables — no further framing changes after these runs.

> **Phase 2 launch authorisation:** ✅ granted by `research/SUBSTRATE_COMPARISON_FINDINGS.md §6` (strong claim confirmed at AL+AZ). Per the plan, no doc revision required before launch.

---

## Status board

🟢 = 5-fold complete · 🟡 = 1-fold or partial · 🔴 = pending · ⚫ = blocked on upstream pipeline.

### FL (Florida) — 5/7 captured 2026-04-28, 2 pending

| Test | C2HGI | HGI | Combined paired test |
|---|:-:|:-:|:-:|
| Substrate-only linear probe | ✅ 40.77 ± 1.11 | ✅ 25.74 ± 0.26 (Δ=+15.03) | n/a (head-free) |
| Cat STL matched-head (`next_gru`) | ✅ **63.43 ± 0.88** | ✅ 34.41 ± 0.94 (**Δ=+29.02 pp**) | 🟡 ready to compute |
| Reg STL matched-head (`next_getnext_hard`) | ✅ Acc@10 82.54 ± 0.42 | 🟡 4/5 folds (Acc@10 82.22 ± 0.50) | 🔴 needs fold 5 |
| MTL B3 counterfactual | (existing data) | 🔴 never ran | 🔴 |

> **FL headline (post-harvest):** cat substrate Δ = +29 pp, ~2× the AL/AZ effect — substrate gap **grows** with scale on cat. Reg gap nearly neutralised at FL scale (+0.27 pp). See `PHASE2_FL_STATUS.md` for full state + retry instructions (daemon-launcher pattern to avoid Colab cell-timeout disconnect that caused the F36c-fold5/F36d losses).

### CA (California) — upstream ready on Drive, grid pending

| Test | C2HGI | HGI | Combined paired test |
|---|:-:|:-:|:-:|
| Upstream pipeline (embeddings + inputs + transition matrix) | ✅ on Drive | ✅ on Drive | — |
| All Phase-2 tests | 🔴 | 🔴 | 🔴 |

### TX (Texas) — upstream ready on Drive, grid pending

| Test | C2HGI | HGI | Combined paired test |
|---|:-:|:-:|:-:|
| Upstream pipeline | ✅ on Drive | ✅ on Drive | — |
| All Phase-2 tests | 🔴 | 🔴 | 🔴 |

---

## 0 · Active task list (in-session ⇄ persistent here)

Three named tasks track Phase-2 closure end-to-end. They are also surfaced in the in-session task list (TaskCreate IDs #1/#2/#3) for live progress; this section is the durable copy.

| Task | State | Status | Blocking |
|---|:-:|:-:|:-:|
| **T1** Close FL grid: F36c reg HGI fold 5 + F36d MTL counterfactual | FL | 🟡 5/7 done | — |
| **T2** Run CA Phase-2 grid (7 experiments) | CA | 🔴 pending | T1 (validate daemon pattern) |
| **T3** Run TX Phase-2 grid (7 experiments) | TX | 🔴 pending | T2 |

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

## 7 · Cross-references

- [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md) — Phase 1 outcome verdict + paper-ready findings.
- [`research/SUBSTRATE_COMPARISON_PLAN.md`](research/SUBSTRATE_COMPARISON_PLAN.md) — phase-gated 3-leg framework.
- [`FOLLOWUPS_TRACKER.md`](FOLLOWUPS_TRACKER.md) — broader study tracker.
- [`OBJECTIVES_STATUS_TABLE.md`](OBJECTIVES_STATUS_TABLE.md) — paper-objective scorecard.
