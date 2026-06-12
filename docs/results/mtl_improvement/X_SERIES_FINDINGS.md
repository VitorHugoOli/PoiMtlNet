# X-series — deep code-audit probe results (2026-06-12, HANDOFF_AUDIT)

The four+one probes that gated study closure (`CODE_AUDIT_2026-06-12.md` PART 1 → HANDOFF_AUDIT
X1–X4 + the β follow-up). All are **MTL-only levers** (they cannot lift the STL ceiling, so they are
exempt from the rising-tide magnitude rule) and/or stress-tests of three published claims
(mixing-dead, KD-dead-end, "matches"). Scored vs the R0 matched bar; promote gate = ≥0.3pp either head.
Raw: `x_series_results.json`. seed0, FL+AL, post-aux-gate-fix code.

## Bottom line — ALL NULL. The "matches, can't beat" verdict is now earned at a strictly higher standard.

Every structurally-disabled MTL lever the audit surfaced is, when actually exercised, null or
negligible; and the two stress-tested claims hold under proper testing.

| probe | what it tests | result | verdict |
|---|---|---|---|
| **X1** roll probe | does cat depend on per-sample cross-attn pairing? | FL cat-F1 aligned 73.012 vs task-b-rolled 73.008 → **Δ −0.004** | **numbers pairing-safe; deployed model performs no per-sample mixing.** ⚠ The aligned-TRAINING counterfactual remains untested (see §X1 correction) → closing-data pre-freeze gate |
| **X2** KD-on-G, FL | does log_T-KD help G's reg (first real test)? | FL reg KD0.2 72.976 vs KD-off 72.929 → **+0.05**; FL cat −0.57 | **NULL** (≪0.3pp; trades cat) |
| **X2** KD-on-G, AL | " | AL reg KD0.2 62.506 vs R0 seed0 62.64 → **−0.13** | **NULL** (KD slightly hurts) |
| **X2** G-unchanged | is the aux-gate fix inert on G? | FL KD-off post-fix 72.929 vs R0 seed0 G 72.951 → **Δ −0.022** | **inert ✓** (bit-identity confirmed empirically) |
| **X4** fp32 eval | is the −0.31pp "matches" gap a fp16 artifact? | FL reg fp16 72.929 vs fp32 72.924 → **Δ −0.005** | **immaterial** — "matches" is precision-clean |
| **X3** β no-WD | did wd=0.05 (not gradient) drive β→0 and suppress the shared→reg pathway? | β still → ≈0 by ep~20 with WD off (mean final −0.0001); FL reg +0.015 / cat +0.139 | **NULL** — β→0 is GRADIENT-driven, not a WD artifact |

## Per-probe detail

### X1 — cross-attn pairing roll probe (CODE_AUDIT P0-A)
The two MTL train loaders draw independent shuffles → the cross-attn block mixes row i (cat) ↔ row i
(reg) of **randomly-paired** windows at train, while val is aligned (`folds.py:1054-1080`,
`mtl_cv.py:463`). Probe (zero new training): re-train G with the task-b/reg stream **rolled by 1 along
the batch dim at eval only** (`MTL_ROLL_TASKB_EVAL=1`, `mtl_eval.py`) so cat row i cross-attends reg row
i−1. **cat-F1 is unchanged (Δ −0.004)** → the deployed model ignores per-sample pairing → every
published number (the +3pp cat, the decomposition, the R0 bar) is **pairing-safe**, and the deployed G
performs no per-sample cross-modal mixing.

> ⚠ **VERDICT WORDING CORRECTED (2026-06-12 design-agent review).** The original write-up concluded
> "mixing is a clean intrinsic fact, NOT a noise-pair-training artifact" and skipped the
> aligned-training run on that basis. **That inference is circular:** a model trained under random
> pairing is *forced* to become invariant to the cross-stream (its per-sample content is noise w.r.t.
> the row), and the roll probe measures exactly that trained-in invariance — it would read Δ≈0
> whether or not aligned training could activate mixing. The probe has **no power against the
> counterfactual** "mixing is learnable under aligned pairing." The same conditioning applies to
> F52 P5 (identity-attn ≈ baseline, measured under misaligned training) and to X3's β→0-by-gradient
> (under misaligned training the shared cross-attended feature is noise-mixed w.r.t. reg's row — the
> gradient *should* kill β). **What is earned:** numbers safe; deployed-model mixing absent.
> **What is NOT earned:** "intrinsic / not an artifact." **Disposition:** the aligned-training test
> (one shared-permutation change + G at AL+FL seed0) is inherited by the **`closing-data` study as a
> PRE-FREEZE gate** — it must run BEFORE the final recipe freezes for the CA/TX majors, because a
> positive would change the recipe. Paper wording: claim "the architecture wins *without*
> per-sample cross-modal mixing (pairing-invariance verified)", not "cross-modal mixing is
> intrinsically useless for this task pair."

### X2 — aux-gate fix + the FIRST REAL KD-on-G test (CODE_AUDIT P0-B)
`next_stan_flow_dualtower` was missing from `_HEADS_REQUIRING_AUX_MTL` (`folds.py:933-937`) →
`get_current_aux()` returned None → every "prior-ON" dualtower arm was prior-OFF AND the `c25_gv2.sh`
`g_kd0.1/0.2` arms were **no-ops** (so CHAMPION §5's old "KD adds nothing on the dual-tower" was a
dead-codepath artifact). **Fixed** (head added to both gate sets; smoke: optimizer/forward see a real
aux; G's own metrics bit-identical by construction — α=0 + KD 0.0 — and confirmed empirically: post-fix
KD-off FL G 72.929 ≈ R0 seed0 72.951, Δ −0.022). **Real test:** G + `--log-t-kd-weight 0.2` (τ=1.0) at
FL+AL seed0. **NULL:** FL reg +0.05 / AL reg −0.13 (both ≪ the 0.3pp gate), FL cat −0.57. KD was the one
confirmed *pre-G* reg lever (v12 single-pathway default W=0.2); it does **not** carry to the dual-tower
champion. KD stays off; the "adds nothing" verdict now stands on a real test.

### X4 — eval-precision parity for the "matches" verdict (CODE_AUDIT P1-D)
MTL eval autocast fp16 on CUDA (`mtl_eval.py`) while the STL p1 ceiling is fp32; fp16 ties are scored
target-favorably (`metrics.py` strictly-higher rank count), and the headline Δreg is −0.09…−0.31pp.
Added an eval escape hatch (`MTL_DISABLE_AMP_EVAL=1`, also honored in `mtl_validation.py`) and scored
the same G training fp16-eval vs fp32-eval (identical weights by determinism, training autocast
untouched). **FL reg fp16 72.929 vs fp32 72.924, Δ −0.005** → the fp16 tie-optimism is immaterial; the
"matches" verb (and the −0.31pp number) is precision-clean. No R0 re-score needed.

### X3 — β weight-decay probe (CODE_AUDIT P1-C)
The dual-tower fusion scalar β (`priv + β·aux_proj(shared)`, init 0.1) sits in the reg group at wd=0.05
(only α is peeled into the zero-WD group). **β logged per epoch (new):** it decays **0.108 → ≈0 by
~epoch 25** — the exact AdamW pull-toward-zero F50 diagnosed for α, acting on what CHAMPION §2 calls "the
key lever" (the shared→reg pathway coefficient). Probe: re-run G at FL seed0 with β peeled into the
zero-WD group (`MTL_BETA_NO_WD=1`, `helpers.py`) to test whether **WD** (not the model's own gradient)
was driving β→0 and suppressing the shared pathway. **Result — NULL, and decisively so:** with WD
removed, β **still decays 0.108 → ≈0 by epoch ~17-25 in all 5 folds** (mean final −0.0001) → **β→0 is
GRADIENT-driven, not a weight-decay artifact.** The model *chooses* to zero the shared→reg fusion
coefficient. Metrics confirm: FL reg 72.944 / cat 73.151 vs the KD-off baseline 72.929 / 73.012 →
**Δreg +0.015 / Δcat +0.139** (both ≪ 0.3pp gate). This is the strongest form of the "shared pathway
adds little to reg" claim: reg genuinely lives in the private tower because the model *learns* to gate
the shared pathway off, not because WD forces it. (Nuances CHAMPION §2's "β is the key lever": β is the
coefficient the model elects to disable.) Rundir `…20260612_202300_800572`; gate code `MTL_BETA_NO_WD`
in `helpers.py` (env-gated, G default unchanged).
⚠ *Conditioning caveat (see the §X1 correction banner):* β→0-by-gradient was measured under
**misaligned pairing** — the shared cross-attended feature is noise-mixed w.r.t. reg's row, so the
gradient is *expected* to kill β in that regime. "The model elects to gate the shared pathway off" is
earned **under the current training distribution**; whether β survives under aligned pairing is part
of the same `closing-data` pre-freeze gate.
