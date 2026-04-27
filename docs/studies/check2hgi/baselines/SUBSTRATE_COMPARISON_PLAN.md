# Substrate Comparison Plan — Check2HGI vs HGI on next_category & next_region

**Created 2026-04-27.** Owner: this doc. Triggered by user ask: *validate Check2HGI ≥ HGI for next-cat and next-reg under the matched-head STL of the MTL north_star, then critically review whether STL heads alone are scientifically sufficient.*

> **Headline (after critique).** Matched-head STL is **necessary but not sufficient**. The plan now hangs the substrate claim on three legs that must agree: **(I) substrate-only diagnostics** (linear probe), **(II) matched-head STL** (the user's original ask), and **(III) MTL counterfactual** (MTL north_star with HGI substituted for Check2HGI). Plus a pre-registered counterfactual for the *mechanism* claim (POI-pooled Check2HGI). Without all three, "Check2HGI > HGI" can be attributed to head choice, optimisation, or coupling artefacts.

> **Execution policy.** All experiments run on **this machine (M4 Pro, MPS)**. Work is gated into two phases:
>
> - **Phase 1 — Validation (AL + AZ).** Cheap states. Land all three legs + critique remediation. **Decisions and plan changes happen here**: if Phase 1 produces an asymmetric / null / mechanism-refuted verdict (§9), revisit framing *before* committing FL/CA/TX compute.
> - **Phase 2 — Final paper runs (FL + CA + TX).** Headline states. Launched only after Phase 1 closes with a stable interpretation. Numbers go straight into the paper tables.
>
> FL is feasible on M4 Pro under `caffeinate -s` + the F20 per-fold persistence (commit landed 2026-04-23; survives mid-run SIGKILL). CA + TX require their upstream pipelines first (§7).

---

## 1 · Goal grid + pre-registered claim

The substrate hypothesis lives on a 2 × 2 axis. The plan must populate every cell, not just the diagonal.

|   | **STL** | **MTL (north_star)** |
|---|---|---|
| **Check2HGI** | partial — cat n=1 head, reg via STAN (3 states) + F21c on 2 states | ✅ committed (B3, all states) |
| **HGI** | partial — cat n=1 (AL only), reg via STAN (3 states) | 🔴 **missing entirely** |

Plus a substrate-only diagnostic plane (no head, no coupling) — currently empty.

### 1.1 Pre-registered outcome paths

Two falsifiable claims; the data picks. The plan does not assume either lands.

- **Strong claim**: Check2HGI > HGI on **both** tasks (matched-head STL + matched MTL + linear probe all favor Check2HGI).
- **Weak claim** (paper-positioned, given existing CH15 evidence): Check2HGI > HGI on **cat** (significant, paired test); Check2HGI **≥** HGI on **reg** under non-inferiority margin δ (TOST).

The reg-side weak-claim TOST equivalence margin is **pre-registered as δ = 2 pp Acc@10** (≈ ½ of the existing CH15 STAN-substrate gap on FL). We commit to δ in this doc *before* the runs land — see §6 / C3.

### 1.2 What "matched-head STL of the MTL north_star" actually means

Per `NORTH_STAR.md` (post-F27, committed 2026-04-24):

```
task_a head    : next_gru                       # cat
task_b head    : next_getnext_hard              # reg (STAN + α·log_T[last_region_idx])
task_a input   : check-in embeddings (9-step window)
task_b input   : region embeddings (9-step window)
```

Matched-head STL keeps head + input + hparams identical and varies only the substrate engine.

---

## 2 · Head provenance — what was actually run with which head

| Result family | Cat head used | Reg head used | Source verified |
|---|---|---|---|
| `P1_5b/next_category_*_5f_50ep_fair.json` (CH16 AL) | **`next_single`** (Transformer + attn-pool + temporal bias) — `default_next` `model_name="next_single"` | n/a | `src/configs/experiment.py::default_next` at commit 5217095 |
| `P1/region_head_*_STAN_*.json` (CH15 STAN STL) | n/a | **`next_stan`** | `scripts/p1_region_head_ablation.py --heads next_stan` at commit 82e4607 |
| `B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json` (F21c) | n/a | **`next_getnext_hard`** | F21C_FINDINGS.md + commit 4ffecd3 |
| MTL north_star (post-F27 B3) | **`next_gru`** | **`next_getnext_hard`** | `NORTH_STAR.md` |

→ **Cat: `next_single` ≠ `next_gru`** (existing CH16 is mismatched-head). **Reg: F21c on `next_getnext_hard` is matched-head**, but only Check2HGI side, only AL+AZ. The previous version of this doc misnamed `next_single` as `next_mtl` — corrected here.

### 2.1 Existing data interpreted under matched-head policy

| Cell | Status | Note |
|---|---|---|
| STL HGI cat — `next_single` | AL ✅ (20.29 ± 1.34) | Existing P1_5b. Head-mismatched but a valid head-sensitivity probe (C2). |
| STL Check2HGI cat — `next_single` | AL ✅ (38.58 ± 1.23) · AZ ✅ (42.08 ± 0.89) · FL ✅ 1f only | Existing P1_5b. |
| STL HGI cat — `next_gru` (matched) | 🔴 all states | New runs §4.1. |
| STL Check2HGI cat — `next_gru` (matched) | 🔴 all states (F27 sweep is *MTL*, not STL) | New runs §4.1. |
| STL HGI reg — STAN | AL ✅ · AZ ✅ · FL ✅ | Existing. Head-mismatched probe (C2). |
| STL Check2HGI reg — STAN | AL ✅ · AZ ✅ · FL ✅ | Existing. Head-mismatched probe (C2). |
| STL HGI reg — `next_getnext_hard` (matched) | 🔴 all states | New runs §4.2. |
| STL Check2HGI reg — `next_getnext_hard` (matched) | AL ✅ (F21c) · AZ ✅ (F21c) · FL 🔴 | New FL run §4.2. |

---

## 3 · Substrate-only diagnostics (Leg I) — the missing first step

Before the head-coupled runs, run **head-free probes** of the embeddings themselves. These directly test substrate quality and decouple it from any head's interaction with per-visit context.

### 3.1 Linear probe (Alain & Bengio 2017 standard)

Train a single linear classifier on raw embeddings → 5-fold StratifiedGroupKFold(seed=42) macro-F1. **No sequence model, no attention, no graph prior.** If Check2HGI's linear-probe F1 > HGI's, the substrate carries the lift by construction. If not, the lift is head-dependent and the paper's framing must shift.

**Reusable infrastructure:** `experiments/check2hgi_up/run_variant.py::linear_probe_cv` already implements this — currently called from `eval_anchors.py` on raw graph features. Wrap it for substrate parquets:

```python
# scripts/probe/substrate_linear_probe.py (NEW, ~60 LOC)
# For each {state, engine, task}:
#   1. Load embeddings parquet (output/<engine>/<state>/embeddings.parquet for cat,
#      region_embeddings.parquet for reg).
#   2. Build (X, y, groups) the same way builders.py does for STL training.
#   3. Call linear_probe_cv(X, y, groups, k=5, seed=42).
#   4. Save f1_mean / f1_std / per-fold deltas for paired test.
```

Cost: ~5 min × 6 cells (3 states × 2 substrates) per task = **~30 min total**. Lands first, before any §4/§5 launches.

### 3.2 Class-separability metrics (supplementary)

Cheap diagnostics that don't require training:

- **k-NN classifier (k=10)** on the embeddings, same fold split — classification F1 as a non-parametric probe.
- **Silhouette coefficient** of the 7-class structure in cat-embedding space.
- **Effective rank** + **anisotropy** of each substrate (gives texture for the discussion section: e.g., HGI may be more anisotropic / lower-rank because it pools per-POI).

All three computable in a single ~10 min script per state-engine cell. Useful for §10 paper interpretation, not stat-test bound.

### 3.3 What this leg cannot do

- It does not test the **per-visit variation** mechanism — the linear probe will still see *every check-in's vector* for Check2HGI and *the POI's single vector duplicated* for HGI. To isolate the per-visit mechanism, see C4 (§6) — POI-pooled Check2HGI as a counterfactual.
- It does not measure *sequential* / context-window utility. That's what §4 (matched-head STL) tests.

---

## 4 · Matched-head STL grid (Leg II) — the user's original ask

10 STL cells to land, split by phase. All runs on M4 Pro under `caffeinate -s`.

### 4.1 next_category — `next_gru` head, check-in input

**Phase 1 (validation):**

| State | Substrate | Status | Cost (MPS) |
|---|---|---|---:|
| AL | check2hgi | 🔴 | ~30 min |
| AL | hgi | 🔴 | ~30 min |
| AZ | check2hgi | 🔴 | ~1.5 h |
| AZ | hgi | 🔴 | ~1.5 h |

**Phase 2 (paper):**

| State | Substrate | Status | Cost (MPS) |
|---|---|---|---:|
| FL | check2hgi | 🔴 | ~5–6 h |
| FL | hgi | 🔴 | ~5–6 h |
| CA | check2hgi | 🔴 (after F22 pipeline) | ~6–8 h |
| CA | hgi | 🔴 (after F22 pipeline) | ~6–8 h |
| TX | check2hgi | 🔴 (after F23 pipeline) | ~6–8 h |
| TX | hgi | 🔴 (after F23 pipeline) | ~6–8 h |

**Launch template** (requires I1 — see §8):

```bash
PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
caffeinate -s python scripts/train.py \
  --task next --state $STATE --engine $ENGINE --head next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints \
  --tag STL_${STATE^^}_${ENGINE}_cat_gru_5f50ep
```

### 4.2 next_region — `next_getnext_hard` head, region input

**Phase 1 (validation):**

| State | Substrate | Status | Cost (MPS) |
|---|---|---|---:|
| AL | check2hgi | ✅ F21c (68.37 ± 2.66 Acc@10) | done |
| AL | hgi | 🔴 | ~45 min |
| AZ | check2hgi | ✅ F21c (66.74 ± 2.11) | done |
| AZ | hgi | 🔴 | ~2 h |

**Phase 2 (paper):**

| State | Substrate | Status | Cost (MPS) |
|---|---|---|---:|
| FL | check2hgi | 🔴 | ~5–6 h |
| FL | hgi | 🔴 | ~5–6 h |
| CA | check2hgi | 🔴 (after F22 pipeline) | ~6–8 h |
| CA | hgi | 🔴 (after F22 pipeline) | ~6–8 h |
| TX | check2hgi | 🔴 (after F23 pipeline) | ~6–8 h |
| TX | hgi | 🔴 (after F23 pipeline) | ~6–8 h |

**Launch template** (requires I2 — see §8):

```bash
PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
caffeinate -s python scripts/p1_region_head_ablation.py \
  --state $STATE --engine $ENGINE --heads next_getnext_hard \
  --folds 5 --epochs 50 --seed 42 --input-type region \
  --override-hparams \
      d_model=256 num_heads=8 \
      "transition_path=$OUTPUT_DIR/check2hgi/${STATE}/region_transition_log.pt" \
  --tag STL_${STATE^^}_${ENGINE}_reg_gethard_5f50ep
```

> **Transition matrix is substrate-independent.** Verified: `scripts/compute_region_transition.py` reads `sequences_next.parquet` (region→region IDs only), not embeddings. The same `output/check2hgi/<state>/region_transition_log.pt` is reused for both substrates' gethard runs. **No I3 patch needed.** (Earlier draft said otherwise — corrected.)

---

## 5 · MTL counterfactual (Leg III) — MTL north_star with HGI substituted

The paper's deployment is MTL. "STL Check2HGI > STL HGI" + "MTL Check2HGI > all-baselines" does not close the loop without showing that **substituting HGI into the same MTL configuration degrades it** (or fails to lift it). One run per state.

**Phase 1 (validation):**

| State | Status | Tag | Cost (MPS) |
|---|---|---|---:|
| AL | 🔴 | `MTL_B3_AL_hgi_5f50ep` | ~1 h |
| AZ | 🔴 | `MTL_B3_AZ_hgi_5f50ep` | ~1.5 h |

**Phase 2 (paper):**

| State | Status | Tag | Cost (MPS) |
|---|---|---|---:|
| FL | 🔴 | `MTL_B3_FL_hgi_5f50ep` | ~5–6 h |
| CA | 🔴 (after F22 pipeline) | `MTL_B3_CA_hgi_5f50ep` | ~6–8 h |
| TX | 🔴 (after F23 pipeline) | `MTL_B3_TX_hgi_5f50ep` | ~6–8 h |

```bash
PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
caffeinate -s python scripts/train.py \
  --task mtl --state $STATE --engine hgi \
  --task-set check2hgi_next_region \
  --architecture mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints \
  --tag MTL_B3_${STATE^^}_hgi_5f50ep
```

→ archive under `results/mtl_hgi_counterfactual/<state>_5f50ep_b3.json`.

> **Engine resolution wrinkle**: `CHECK2HGI_NEXT_REGION` preset is hard-coded to read check2hgi region embeddings + check-in inputs. With `--engine hgi` the IoPaths routing should pick up HGI's `embeddings.parquet` + `region_embeddings.parquet`, but task_b's `transition_path` override stays pointed at the (substrate-free) check2hgi transition log per §4.2 note. **Verify with a 1-fold smoke before the 5-fold launch** to catch any preset-binding issue.

---

## 6 · Critique remediation (revised)

| # | Lacuna | Action | Notes |
|---|---|---|---|
| **C1** | n=1 state (AL only) for CH16 | Phase 1 closes the n≥2 minimum (AL+AZ); Phase 2 extends to FL/CA/TX for the paper. | — |
| **C2** | n=1 head probe | **Head-agnostic STL probe at AL + AZ** (not just AL): for cat, run STL at both states with 3 heads {`next_gru`, `next_single`, `next_lstm`}; report sign(Δ_substrate) per head. Reg head-agnostic is *already done* via the existing STAN runs (mismatched-head reg evidence at 3 states) + the new gethard runs (matched). Cost: ~3 h MPS for AL cat sweep + ~6 h for AZ cat sweep. | One-state head invariance is weak; two-state is review-defensible. |
| **C3** | n=1 seed × 5 folds — informal σ-overlap | Phase 1 (AL+AZ): **3 seeds × 5 folds = 15 paired samples** for the matched-head pairs. Phase 2 (FL/CA/TX): **5×1** (single seed) to keep wall-clock manageable on M4 Pro — 5×1 is the publication standard for headline runs at FL scale. Use **paired-t on per-fold deltas** if Shapiro-Wilk passes (n=15 is enough), Wilcoxon otherwise. For reg under the **weak claim**, run **TOST with δ=2 pp Acc@10** (pre-registered §1.1) — non-inferiority must be tested explicitly, not inferred from "no significant difference". | 5×1 → smallest one-sided Wilcoxon p = 0.0312; the 15-sample Phase-1 resolution gives finer evidence at the validation states where decisions get made. |
| **C4** | No mechanism counterfactual ("per-visit variation drives the F1 lift") | **Mandatory at AL** (FL only if AL survives). Generate **POI-pooled Check2HGI** (mean-pool check-ins per `placeid` → one vector per POI). Run **linear probe (§3) and matched-head STL cat (§4.1) at AL with `--engine check2hgi_pooled`**. Predicted ranking *under the mechanism*: Check2HGI > Check2HGI-pooled ≈ HGI. Any other ordering refutes per-visit-variation as the mechanism. | Without C4, the paper's mechanism story is narrative, not evidence. |

**Execution order — phase-gated:**

**Phase 1 (validation, AL + AZ).** Allows decisions/changes before committing FL/CA/TX compute.

1. **I1 + I2 + I5 + I6 patches** (§8) — infrastructure must land first.
2. **§3 substrate-only probes** at AL + AZ (linear probe + k-NN + sep metrics) — ~1 h total. Cheapest discriminator; shapes what's worth running next.
3. **§4 matched-head STL grid** at AL + AZ, both tasks (3 seeds × 5 folds = 15 samples for headline pairs).
4. **§5 MTL counterfactual** at AL + AZ (1 seed × 5 folds).
5. **C2 head-agnostic cat sweep** at AL + AZ.
6. **C4 POI-pooled probe at AL** (substrate-only linear probe + matched-head STL cat).
7. **C3 paired-test + TOST analysis** on Phase-1 outputs.
8. **Phase-1 verdict checkpoint** — slot the result into the §9 outcome matrix. If asymmetric / null / mechanism-refuted: **stop and revise framing with the user before launching Phase 2.** If strong claim holds: proceed.

**Phase 2 (paper headline, FL → CA → TX).** Final numbers for the paper. Launched only after Phase 1 closes.

9. **Pre-Phase-2 smoke**: 1-fold smoke of every Phase-2 launch on FL to catch regressions (the F20 per-fold persistence guarantees fold-1 lands even if subsequent folds SIGKILL).
10. **§4 matched-head STL grid** at **FL** (5×1 each cell, 4 cells: cat × {C2HGI, HGI} + reg × {C2HGI, HGI}).
11. **§5 MTL counterfactual** at **FL**.
12. **C4 POI-pooled extension to FL** — only if Phase-1 AL retained the lift; otherwise skip.
13. **F22 / F23 upstream pipelines** for CA and TX (per `FOLLOWUPS_TRACKER.md`).
14. **§4 + §5 grid at CA** then **TX**, mirror of FL.
15. **Final paired-test + TOST aggregation** across all 5 states.

**Wall-clock estimate (M4 Pro):**

- Phase 1 ≈ ~12 h compute (infra + AL + AZ everything) — fits in a long weekend or 2 night runs under `caffeinate -s`.
- Phase 2 (FL alone) ≈ ~25 h compute — 1–2 days; parallelisable with afternoon AZ jobs only if disjoint substrates.
- Phase 2 (CA) ≈ ~30 h + upstream pipeline. Phase 2 (TX) ≈ ~30 h + upstream pipeline.
- End-to-end ≈ 5–7 days of M4 Pro wall-clock if runs are sequential; less with overlap.

---

## 7 · Phase 2 — FL + CA + TX final paper runs

These runs land the headline paper numbers. **Do not launch any of them until Phase 1 (§6 step 8) closes with a stable verdict.**

FL is ready to launch (embeddings + inputs already exist on disk under `output/{check2hgi,hgi}/florida/`). CA + TX require their upstream pipelines first (`FOLLOWUPS_TRACKER.md §F22–F25`). C2 (head-agnostic) is **not** repeated at FL/CA/TX — Phase-1 AL+AZ is sufficient. C4 (POI-pooled) extends to FL only if Phase-1 AL retained the lift.

### 7.1 Florida (ready now)

```bash
# Substrate-only probe (Leg I)
python scripts/probe/substrate_linear_probe.py --state florida

# Matched-head STL cat (§4.1)
for ENGINE in check2hgi hgi; do
  PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  caffeinate -s python scripts/train.py \
    --task next --state florida --engine $ENGINE --head next_gru \
    --folds 5 --epochs 50 --seed 42 --no-checkpoints \
    --tag STL_FL_${ENGINE}_cat_gru_5f50ep
done

# Matched-head STL reg (§4.2)
for ENGINE in check2hgi hgi; do
  PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  caffeinate -s python scripts/p1_region_head_ablation.py \
    --state florida --engine $ENGINE --heads next_getnext_hard \
    --folds 5 --epochs 50 --seed 42 --input-type region \
    --override-hparams d_model=256 num_heads=8 \
        "transition_path=$OUTPUT_DIR/check2hgi/florida/region_transition_log.pt" \
    --tag STL_FL_${ENGINE}_reg_gethard_5f50ep
done

# MTL counterfactual (§5)
PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
caffeinate -s python scripts/train.py \
  --task mtl --state florida --engine hgi \
  --task-set check2hgi_next_region --architecture mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints \
  --tag MTL_B3_FL_hgi_5f50ep
```

**FL operational notes:**

- Wall-clock per cell ≈ 5–6 h. 5 cells = ~25–30 h sequential.
- `caffeinate -s` is mandatory (G4: MPS sleep-induced SIGBUS).
- F20 per-fold persistence (`src/tracking/storage.py::save_fold_partial`) means a SIGKILL mid-fold-N still preserves folds 0..N-1.
- Watch macOS swap pressure; kill Spotlight + idle apps before launches (G5).
- Run a 1-fold smoke first to validate the §5 MTL+HGI engine-resolution wrinkle.

### 7.2 California + Texas (after upstream pipelines)

CA + TX upstream pipelines (`F22 / F23` in tracker) must land first. Once `output/{check2hgi,hgi}/california/` and `.../texas/` exist with embeddings + region labels + transition matrix, the run grid is identical to §7.1 with `--state california` / `--state texas`.

```bash
# Per state ∈ {california, texas}:
# (a) Train Check2HGI + HGI embeddings
python pipelines/embedding/check2hgi.pipe.py --state $STATE
python pipelines/embedding/hgi.pipe.py --state $STATE
# (b) One transition matrix (substrate-independent), one next_region label set
python scripts/compute_region_transition.py --state $STATE
python scripts/regenerate_next_region.py --state $STATE
# (c) Inputs for both substrates
python pipelines/create_inputs.pipe.py --state $STATE --engine check2hgi
python pipelines/create_inputs.pipe.py --state $STATE --engine hgi
# (d) Substrate-only probes
python scripts/probe/substrate_linear_probe.py --state $STATE
# (e) Matched-head STL grid (cat + reg, both substrates)
for ENGINE in check2hgi hgi; do
  python scripts/train.py --task next --state $STATE --engine $ENGINE \
    --head next_gru --folds 5 --epochs 50 --seed 42 --no-checkpoints \
    --tag STL_${STATE^^}_${ENGINE}_cat_gru_5f50ep
  python scripts/p1_region_head_ablation.py --state $STATE --engine $ENGINE \
    --heads next_getnext_hard --folds 5 --epochs 50 --seed 42 --input-type region \
    --override-hparams d_model=256 num_heads=8 \
        "transition_path=$OUTPUT_DIR/check2hgi/${STATE}/region_transition_log.pt" \
    --tag STL_${STATE^^}_${ENGINE}_reg_gethard_5f50ep
done
# (f) MTL counterfactual (HGI substituted)
python scripts/train.py --task mtl --state $STATE --engine hgi \
  --task-set check2hgi_next_region --architecture mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints \
  --tag MTL_B3_${STATE^^}_hgi_5f50ep
```

---

## 8 · Infrastructure prerequisites

Small patches the runs depend on. Each ≤ 60 LOC.

| # | Patch | File | Required for | Status |
|---|---|---|---|---|
| **I1** | ~~Wire `--head` CLI through `scripts/train.py`~~ — **already exists** as `--model` (line 372). Use `--model next_gru` for matched-head STL cat. | — | §4.1 launches | ✅ already wired |
| **I2** | ~~Add `--engine` axis to `scripts/p1_region_head_ablation.py`~~ — **already exists** as `--region-emb-source {check2hgi,hgi}` (line 632). Labels come from check2hgi sequences (substrate-free); embedding lookup switches per substrate. | — | §4.2 HGI rows | ✅ already wired |
| **I3** | ~~Per-engine transition matrix~~ — **not needed**. Substrate-independent. | — | — | ❌ dropped |
| **I4** | `CHECK2HGI_POOLED` engine: a new `pipelines/embedding/check2hgi_poi_pooled.pipe.py` that mean-pools `output/check2hgi/<state>/embeddings.parquet` per `placeid` into `output/check2hgi_pooled/<state>/embeddings.parquet`; register engine in `src/configs/paths.py::EmbeddingEngine` + `IoPaths`. | new pipeline + paths config | C4 | 🔴 |
| **I5** | Linear-probe wrapper: load substrate parquets → call `experiments/check2hgi_up/run_variant.py::linear_probe_cv`. | new `scripts/probe/substrate_linear_probe.py` | §3.1 | 🔴 |
| **I6** | Paired-test + TOST analyser: load 3-seed × 5-fold per-fold metrics → paired-t / Wilcoxon + TOST(δ). | new `scripts/analysis/substrate_paired_test.py` | C3 | 🔴 |

I1, I2, I3 already resolved. I5 lands before §3; I4 + I6 can land in parallel with §4 runs.

---

## 9 · Outcome interpretation matrix (pre-registered)

After all three legs land, classify into one of these end-states. Each requires a different paper framing.

| Linear probe (Leg I) | Matched-head STL (Leg II) | MTL counterfactual (Leg III) | C4 mechanism | Verdict |
|---|---|---|---|---|
| C2HGI > HGI | C2HGI > HGI cat; C2HGI ≥ HGI reg (TOST) | MTL+C2HGI > MTL+HGI | C2HGI > pooled ≈ HGI | **Strong claim holds.** Substrate causes lift; per-visit variation is the mechanism; weak-claim margin satisfied on reg. |
| C2HGI > HGI | C2HGI > HGI cat; HGI > C2HGI reg outside δ | MTL+C2HGI > MTL+HGI on cat; ≤ on reg | C2HGI > pooled ≈ HGI | **Asymmetric finding.** Substrate wins on cat; reg cap is real — paper reframes around CH15 honestly (HGI > C2HGI on reg is a finding, not a failure). |
| C2HGI ≈ HGI | C2HGI > HGI cat | MTL+C2HGI > MTL+HGI | unchanged | **Head/coupling-coupled lift.** Substrate alone doesn't carry the gain — the per-visit context matters only when paired with a sequence-aware head. Paper framing: "Check2HGI's contribution is *interactional*, not standalone." |
| C2HGI > HGI | C2HGI ≈ HGI both tasks | both tied | C2HGI ≈ pooled ≈ HGI | **Embedding-quality null.** Substrate looks better in linear probe but downstream heads absorb the difference. Investigate optimisation / capacity issues. |
| any | any | any | C2HGI ≈ pooled | **Mechanism refuted.** Per-visit variation is *not* what drives the gap. Paper has to find a new story for *why* Check2HGI is better (training signal? gradient flow? regularisation effect?). |

The plan does not pre-decide which path lands — that's the data's job.

---

## 10 · Don't

- **Don't launch Phase 2 (FL/CA/TX) before Phase 1 (AL+AZ) closes** with a verdict in §9. Phase 1 is the decision-making phase; Phase 2 is paper-final.
- **Don't launch FL without `caffeinate -s`** — MPS sleep-induced SIGBUS is the dominant FL failure mode (G4). The F20 per-fold persistence partially mitigates it but doesn't eliminate the wasted compute.
- **Don't quote existing P1_5b / B3_baselines numbers as matched-head evidence** for cat — the head was `next_single`, not `next_gru`. Quote them as head-sensitivity probe (C2) only.
- **Don't drop the existing `next_single` / STAN substrate numbers** — they remain valid head-sensitivity ablations.
- **Don't repeat C2 / C4 at FL / CA / TX.** AL+AZ is sufficient for head-invariance; AL is sufficient for the mechanism counterfactual (extend C4 to FL only if Phase-1 AL kept the lift).
- **Don't rebuild the region transition matrix per-engine** — it's substrate-independent.
- **Don't infer non-inferiority on reg from "no significant difference"** — run TOST with the pre-registered δ = 2 pp Acc@10.
- **Don't push to `main`.**
