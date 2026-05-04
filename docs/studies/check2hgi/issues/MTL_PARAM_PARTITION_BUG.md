# MTL Parameter-Partition Bug — Gradient-Surgery Losses Silently Freeze LoRA / AdaShare — Audit

**Severity:** HIGH — contaminates every MTLoRA and AdaShare result that used PCGrad / CAGrad / Aligned-MTL as the MTL loss. The affected params stay at initialisation for the entire run, so the measured "MTLoRA lift" and "AdaShare neutrality" reflect the **baseline DSelectK / baseline MTLnet** numbers plus seed noise, not the adapter/gate contribution.

**Detected:** 2026-04-22 during the pre-P5 model/optimizer critical review (this issue).

**Status:** OPEN — fix is small (extend substring filters in 2 classes), re-runs are bounded (6 JSON files across AL + AZ, all 5f × 50ep or 1f × 50ep — see [Re-run triage](#re-run-triage) below).

---

## TL;DR

`PCGrad`, `CAGrad`, and `Aligned-MTL` all implement MTL balancing by **assigning `p.grad = g` only for params in `shared_parameters ∪ task_specific_parameters`** (src/losses/pcgrad/loss.py:134-145, src/losses/cagrad/loss.py:143-156, src/losses/aligned_mtl/loss.py:108-121). Anything outside that union is skipped: `optimizer.zero_grad(set_to_none=True)` already nulled `.grad`, no surgery step restores it, AdamW sees `p.grad is None` and does nothing.

Two parameter groups on this branch sit **outside** the union for the affected classes:

| Class | Param group | Where |
|---|---|---|
| `MTLnet` (baseline) | `adashare_logits` | src/models/mtl/mtlnet/model.py:214 |
| `MTLnetDSelectK` | `lora_A_cat`, `lora_B_cat`, `lora_A_next`, `lora_B_next`, `skip_alpha_cat`, `skip_alpha_next` | src/models/mtl/mtlnet_dselectk/model.py:85-105 |

Downstream:
- `lora_B_*.weight` starts at **zero** (src/models/mtl/mtlnet_dselectk/model.py:104-105) → `lora_B(lora_A(enc)) = 0` at step 0 → never updated → **still 0 at step N**. The LoRA branch contributes **nothing** under PCGrad/CAGrad/Aligned-MTL.
- `skip_alpha_*` starts at **zero** (src/models/mtl/mtlnet_dselectk/model.py:85-86) → `α · enc = 0` → stays 0 under surgery losses. The α-skip branch also contributes **nothing**.
- `adashare_logits` starts at **2.0** → `sigmoid(2.0) ≈ 0.88` → gates sample ~1 at step 0 → never updated → AdaShare behaves like **always-on** (i.e. the base FiLM path) for the entire run.

`NashMTL` is **unaffected** — it builds a weighted scalar loss and calls plain `.backward()`, so autograd reaches every parameter that touches the loss (src/losses/nash_mtl/loss.py:312, 336). The FiLM-only variants (MTLnet-without-AdaShare, MTLnetCGC, MTLnetMMoE, MTLnetPLE, MTLnetCrossAttn) are also unaffected — their partitions cover every submodule.

---

## Evidence

### 1. Param-partition keyword mismatch (static)

`MTLnet.task_specific_parameters` (src/models/mtl/mtlnet/model.py:525-538) matches only names containing:
```
category_encoder | next_encoder | category_poi | next_poi
```
`MTLnet.shared_parameters` (src/models/mtl/mtlnet/model.py:516-523) matches only:
```
shared_layers | task_embedding | film
```
`adashare_logits` is a top-level `nn.Parameter` whose name is literally `"adashare_logits"` — it matches neither list.

`MTLnetDSelectK.task_specific_parameters` (src/models/mtl/mtlnet_dselectk/model.py:192-209) matches:
```
category_encoder | next_encoder | category_poi | next_poi
| dselect.category_selector | dselect.next_selector
| dselect.category_selector_weights | dselect.next_selector_weights
```
The LoRA adapters are created as `self.lora_A_cat = nn.Linear(...)` etc. → their qualified names are `lora_A_cat.weight` etc. — match no substring. Same story for `skip_alpha_{cat,next}`.

### 2. Gradient-surgery loss behaviour (static, confirmed by reading)

`PCGrad._set_pc_grads` (src/losses/pcgrad/loss.py:123-145):
```python
for p, g in zip(shared_parameters, non_conflict_shared_grads):
    p.grad = g
...
for p, g in zip(task_specific_parameters, task_specific_grads):
    p.grad = g
```
No fallback for params outside the two lists. `CAGrad` and `Aligned-MTL` use the same pattern (same file line-refs above).

### 3. Empirical alignment with past observations

| Observation (past) | Prediction under this bug |
|---|---|
| "AdaShare NEUTRAL (−0.31)" on the baseline MTLnet at fair LR (commit `27f7c5c`, 2026-04-18) | AdaShare gates never train → behaviour identical to base MTLnet within σ → **neutral result is the expected symptom**, not a genuine null. |
| "MTLoRA r=8 marginal (+1.84 pp)" vs DSelectK+PCGrad champion (commit `c6a4ab4`, 2026-04-17) | LoRA A/B + α-skip contribute 0 under PCGrad → MTLoRA run ≡ DSelectK+PCGrad baseline + different seed variance → **a sub-2 pp difference is consistent with seed noise**, not a real lift. |
| "MTLoRA r=16 / r=32" produced no monotonic improvement over r=8 | If LoRA never trains, rank is irrelevant → **flat scaling curve is predicted**. |

None of these observations were diagnostic of the bug at the time because each had an alternative explanation. Together, in light of the static finding, they are predicted by the bug with high specificity.

### 4. A one-line invariant catches this forever

`tests/test_regression/test_regression.py:288-301` asserts the invariant `shared_ids ∪ task_ids == all_ids` for the baseline `MTLnet`. The test currently passes only because AdaShare is gated off by default (`use_adashare=False`) and LoRA lives only in `MTLnetDSelectK`. The same invariant is **not** asserted for `MTLnetDSelectK`, `MTLnetCGC`, `MTLnetMMoE`, `MTLnetPLE`, `MTLnetCrossAttn` — only the base class. Extending the test to all MTL variants (and to `MTLnet(use_adashare=True)`) would have caught this at PR time.

---

## Affected combinations

| Model | MTL loss | Affected? | Reason |
|---|---|---|---|
| `MTLnet` (no AdaShare) | any | **No** | Partition complete |
| `MTLnet(use_adashare=True)` | `nash_mtl` | **No** | Plain `.backward()` reaches `adashare_logits` |
| `MTLnet(use_adashare=True)` | `pcgrad` / `cagrad` / `aligned_mtl` | **Yes** | `adashare_logits` never trains |
| `MTLnetCGC` / `MTLnetMMoE` / `MTLnetPLE` | any | **No** | Partitions cover every submodule |
| `MTLnetCrossAttn` | any | **No** | Partitions cover every submodule |
| `MTLnetDSelectK` (after MTLoRA commit `a9309cb`, 2026-04-17) | `nash_mtl` | **No** | Plain `.backward()` trains LoRA + α |
| `MTLnetDSelectK` (after MTLoRA commit) | `pcgrad` / `cagrad` / `aligned_mtl` | **Yes** | LoRA A/B + α-skip never train → equivalent to pre-MTLoRA DSelectK |
| `MTLnetDSelectK` (before MTLoRA commit) | any | **No** | LoRA params did not yet exist |

---

## Re-run triage

**The data pipeline is unaffected.** Check-in embeddings, region embeddings, next_region labels, fold splits, Markov baselines, and the GETNext transition matrix (`region_transition_log.pt`) are all computed **outside** the MTL training loop — this bug cannot have touched them. Only the MTL training runs listed below are contaminated, and only their JSON metrics need to be regenerated.

### Must re-run (after fix)

| # | Run file | Config | Scope | Cited in |
|---|---|---|---|---|
| 1 | `docs/studies/check2hgi/results/P2/ablation_04_mtlora_r8_al_5f50ep.json` | `dselectk + pcgrad + MTLoRA r=8`, AL 5f × 50ep | A7 / B11 (claimed "best MTL reg Acc@10 = 50.72") | RESULTS_TABLE A-M2/B-M2, BASELINES_AND_BEST_MTL A7/B11, FINAL_ABLATION_SUMMARY R4 |
| 2 | `docs/studies/check2hgi/results/P2/ablation_04_mtlora_r16_al_5f50ep.json` | `dselectk + pcgrad + MTLoRA r=16`, AL 5f × 50ep | MTLoRA rank-sweep entry | FINAL_ABLATION_SUMMARY (implicit) |
| 3 | `docs/studies/check2hgi/results/P2/ablation_04_mtlora_r32_al_5f50ep.json` | `dselectk + pcgrad + MTLoRA r=32`, AL 5f × 50ep | MTLoRA rank-sweep entry | FINAL_ABLATION_SUMMARY (implicit) |
| 4 | `docs/studies/check2hgi/results/P2/ablation_05_adashare_mtlnet_al_5f50ep.json` | `mtlnet(use_adashare=True) + pcgrad`, AL 5f × 50ep | "AdaShare NEUTRAL" claim (commit `27f7c5c`) | BASELINES_AND_BEST_MTL, FINAL_ABLATION_SUMMARY |
| 5 | `docs/studies/check2hgi/results/P2/az2_mtlora_r8_fairlr_5f50ep.json` | `dselectk + pcgrad + MTLoRA r=8`, AZ 5f × 50ep | AZ replication of MTLoRA | STATUS_REPORT_2026-04-20 (P3 / AZ2) |
| 6 | `docs/studies/check2hgi/results/P2/rerun_R4_mtlora_r8_fairlr_al_5f50ep.json` | `dselectk + pcgrad + MTLoRA r=8` (fair-LR rerun), AL 5f × 50ep | R4 in fair-LR leaderboard | FINAL_ABLATION_SUMMARY R4, RESULTS_TABLE |

**Total compute:** 5f × 50ep × 4 AL configs + 5f × 50ep × 1 AZ config ≈ 5 training runs × (AL scale) + 1 × (AZ scale). With the P2 cost of ~25-40 min/run on the workstation used for the prior ablations, **~3-4 hours of machine time total**. Can be parallelised across the two machines in `scripts/p7_launcher.sh`.

### Valid — no re-run needed

All CrossAttention runs are **safe** — `MTLnetCrossAttn` has no AdaShare or LoRA attached, and its `shared_parameters` / `task_specific_parameters` iterate over full submodules (src/models/mtl/mtlnet_crossattn/model.py:238-252). The union covers every param. This includes:

- `ablation_06_crossattn_al_5f50ep.json` (A-M3 / B-M4) — the paper champion
- `ablation_08_nashmtl_crossattn_az_5f50ep.json` (AZ cross-attn + nash-MTL)
- `al_lambda0_crossattn_fairlr_5f50ep.json`
- `az1_crossattn_fairlr_5f50ep.json` (A-M6 / B-M7)
- `fl_crossattn_fairlr_1f50ep.json` (A-M10 / B-M8)
- `rerun_R3_crossattn_fairlr_al_5f50ep.json`
- All P8 MTL-STAN / MTL-GETNext runs (`crossattn + pcgrad + STAN/GETNext`) — A-M4..A-M11, B-M5..B-M6d, B-M9..B-M12. Cross-attn partition is complete; STAN/GETNext heads are in `next_poi` which is in `task_specific_parameters`.

Baseline MTLnet (no AdaShare), CGC, MMoE, PLE variants are also unaffected:

- `ablation_05_mtlnet_baseline_al_5f50ep.json` (the fair baseline for AdaShare comparison) — **valid**
- `rerun_R2_mtlnet_fairlr_al_5f50ep.json` — **valid**
- `rerun_R1_lambda0_fairlr_al_5f50ep.json` — **valid** (no task-loss interaction = no MTL-loss involvement)
- All `screen_leaderboard.md` entries using `{mtlnet, mtlnet_cgc, mtlnet_mmoe, mtlnet_ple}` crossed with `{pcgrad, cagrad, nash_mtl, equal_weight}` — **valid**
- All `budget_test_dselectk_pcgrad_*` runs that pre-date commit `a9309cb` (2026-04-17) — **valid** (LoRA params did not exist yet). `budget_test_dselectk_pcgrad_al_5f_50ep.json` is pre-date, **valid**.
- `validate_dselectk_pcgrad_gru_al_5f_50ep.json` — was this run pre-MTLoRA? The surrounding commit activity on 2026-04-17 (validate, then MTLoRA added) strongly suggests pre-MTLoRA → **valid**, but **verify** by checking the run's timestamp relative to commit `a9309cb` before publishing.
- `validate_fl_dselectk_pcgrad_gru_1f_50ep.json` — FL validate, same question → verify timestamp → probably **valid** (pre-MTLoRA).

### Claims to restate in the paper draft after re-runs

| Current claim | Status after re-runs |
|---|---|
| "MTLoRA r=8 gives +1.84 pp over DSelectK+PCGrad" | Likely to **evaporate into noise**. Keep MTLoRA as a *null-or-small-positive* row; do not lead with it. |
| "AdaShare NEUTRAL" | Status **unknown** — the prior result was a silent no-op (gates never trained). The re-run is the first real test of AdaShare on this task. Could be positive, negative, or genuinely null. |
| "Best MTL reg Acc@10 = 50.72 (MTLoRA r=8)" (B11) | Will be replaced. The honest best MTL reg Acc@10 from safe runs is **52.41 ± 4.70** (cross-attn R3, FINAL_ABLATION_SUMMARY.md:19) — which was already the de-facto champion. **Science does not change.** |
| "MTLoRA closes architectural overhead" (ablation_architectural_overhead.md:49) | Needs to be re-examined. If MTLoRA genuinely trains now, does it close the gap? If not, the paper's "architectural fixes required" argument still holds but the specific recipe is different. |

---

## Fix

Two edits, both <10 LOC. Fix lives in the same commit as the test extension so the invariant cannot regress.

**1. `src/models/mtl/mtlnet/model.py` — extend `task_specific_parameters`:**
```python
def task_specific_parameters(self) -> Iterator[nn.Parameter]:
    return (
        p
        for name, p in self.named_parameters()
        if any(
            key in name
            for key in (
                "category_encoder",
                "next_encoder",
                "category_poi",
                "next_poi",
                "adashare_logits",   # NEW
            )
        )
    )
```
Rationale: AdaShare gates are per-task (shape `[2, num_blocks]`) and parameterise how the shared pipeline specialises per task. Semantically task-specific, so `task_specific_parameters` is the right bucket; gradient-surgery losses will then `.grad`-set them via `task_specific_grads`.

**2. `src/models/mtl/mtlnet_dselectk/model.py` — extend `task_specific_parameters`:**
```python
def task_specific_parameters(self) -> Iterator[nn.Parameter]:
    return (
        p
        for name, p in self.named_parameters()
        if any(
            key in name
            for key in (
                "category_encoder",
                "next_encoder",
                "category_poi",
                "next_poi",
                "dselect.category_selector",
                "dselect.next_selector",
                "dselect.category_selector_weights",
                "dselect.next_selector_weights",
                "lora_A_cat", "lora_B_cat",    # NEW
                "lora_A_next", "lora_B_next",  # NEW
                "skip_alpha_cat",               # NEW
                "skip_alpha_next",              # NEW
            )
        )
    )
```

**3. `tests/test_regression/test_regression.py` — generalise the partition invariant** so this class of bug cannot return silently:
- Parameterise `test_mtl_shared_vs_task_params` over `{mtlnet, mtlnet(use_adashare=True), mtlnet_cgc, mtlnet_mmoe, mtlnet_dselectk, mtlnet_ple, mtlnet_crossattn}`.
- For each, assert `shared_ids & task_ids == set()` and `shared_ids | task_ids == all_ids`.
- Both assertions must hold with **non-default** flags enabled (AdaShare on the baseline, MTLoRA on DSelectK — which is always-on post-2026-04-17).

---

## Verification plan

1. **Apply the two fixes + extended test.** The test should go red on `main` (before fix) for DSelectK and AdaShare-on baseline, green after the fix.
2. **Micro-verification: gradient presence.** Run a 1-epoch training step with `pcgrad` on `mtlnet_dselectk` post-fix and assert `model.lora_B_cat.weight.grad is not None` and `model.lora_B_cat.weight.grad.abs().sum() > 0`. Same for `skip_alpha_cat.grad`. Same for `adashare_logits.grad` on the AdaShare path. Add this as a regression test.
3. **Sanity-run (1 fold, 5 epochs) of contaminated config A7** (`dselectk + pcgrad + MTLoRA r=8` on AL). Compare to pre-fix run:
   - If metrics are within σ → confirms the old runs were "LoRA-frozen equivalent" (no real MTLoRA contribution in old runs).
   - If metrics differ substantially → MTLoRA does something real once it trains; the direction (better or worse) determines whether the prior claim survives.
4. **Full re-run** of the 6 runs in [Re-run triage](#re-run-triage) at 5f × 50ep (AL) / 5f × 50ep (AZ), seed 42, per the commit-level hyperparameters of the original runs (check `scripts/p7_launcher.sh` and the per-run command logged in each JSON).
5. **Update RESULTS_TABLE.md, BASELINES_AND_BEST_MTL.md, FINAL_ABLATION_SUMMARY.md** with the re-run numbers. Flag the pre-fix rows with `(SUPERSEDED: MTL_PARAM_PARTITION_BUG)` to preserve the audit trail.

---

## Blast radius on paper claims

**Paper-level risk: LOW-MEDIUM, contained.** The headline claim — "cross-attention is the best MTL architecture for this task, closes the cat gap, narrows the reg gap" — is **not affected** (cross-attn was never contaminated). The secondary claim — "MTLoRA is the best recipe for the region head" — will likely be retracted or heavily softened. The Δm numbers in FINAL_ABLATION_SUMMARY.md row "R4 MTLoRA r=8" need to be replaced; the adjacent rows (R1, R2, R3, R5) are all safe.

The bug does **not** invalidate:
- Any STL baselines (no MTL loss involved)
- The fair-LR finding (max_lr=0.003 >>> max_lr=0.001)
- The architectural-overhead claim (λ=0.0 isolation was on the CrossAttn variant, safe)
- The capacity-ceiling / BACKBONE_DILUTION characterisation (safe variants only)
- The scale-curve and FL validation results (safe variants only)

---

## References

- Critical review that surfaced this: conversation 2026-04-22, triggered by the pre-P5 model/optimizer review.
- `src/losses/pcgrad/loss.py:123-145` — offending pattern in PCGrad
- `src/losses/cagrad/loss.py:141-156` — same pattern in CAGrad
- `src/losses/aligned_mtl/loss.py:95-121` — same pattern in Aligned-MTL
- `src/losses/nash_mtl/loss.py:312, 336` — safe pattern (weighted scalar + plain `.backward()`)
- `tests/test_regression/test_regression.py:288-301` — the one-invariant test that needs to be generalised across all MTL variants
