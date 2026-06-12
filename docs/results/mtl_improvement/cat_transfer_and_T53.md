# Cat-transfer ablation (a) + T5.3 HSM reg head (b) — 2026-06-08

Two follow-up probes after the Tier-3/4 closes; reg matched-metric (R0 method). The cat-transfer
decomposition (a) is **multi-seed {0,1,7,100}**; the T5.3 HSM mechanism test (b) is seed0. (P0 fix
2026-06-12: the FL multi-seed was re-run — see the dated note under the table.)

## (a) Cat-transfer ablation — is MTL-cat's +3pp gain region-driven or architecture-driven?

**Question.** G's MTL category head beats the STL cat ceiling by +3pp. Is that *positive transfer*
from the region task, or just the *cross-attn shared architecture* being a better cat encoder than
the STL head (`next_gru` on raw check-in embeddings)?

**Method.** Run G's exact recipe but `--category-weight 1.0` → reg loss weight = 0 → reg contributes
**zero gradient** → the shared cross-attn trunk + cat head train cat-ONLY (with the architecture, no
region co-training). Driver `scripts/mtl_improvement/cat_transfer_ablation.sh`. Reg confirmed OFF
(FL reg cratered to 0.12% — the trunk is cat-only-trained). Decompose:
`total = architecture (STL→cat+trunk) + region-transfer (cat+trunk→G)`.

| state | STL cat (no trunk) | cat+trunk, reg OFF (4-seed) | G cat (reg ON, 4-seed) | **architecture** | **region-transfer** | total |
|---|---|---|---|---|---|---|
| AL | 50.35 | 53.57 ± 0.24 | 52.91 ± 0.27 | **+3.22** | **−0.67** | +2.56 |
| FL | 69.96 | 72.24 ± 0.03 | 73.16 ± 0.04 | **+2.27** | **+0.93** | +3.20 |

(Multi-seed {0,1,7,100}; FL re-run 2026-06-12 — see the P0 note below; AL is the original 4 distinct
seeds. Both signs hold. The reg-OFF isolation isn't perfectly clean: the cat stream still attends to the
reg stream's K/V in the bidirectional cross-attn even at reg-weight 0, so a little region structure
remains in the "architecture" term — see `orthogonality_intrinsic_test.md`. Source JSON:
`cat_transfer_decomposition_4seed.json`.)

> **⚠ P0 data-integrity fix (2026-06-12, HANDOFF_AUDIT).** The published FL row was **72.09 ± 0.08 →
> architecture +2.13 / transfer +1.08**. A re-audit found the `cat_transfer_manifest.tsv` FL rows
> `s1/s7/s100` ALL pointed to one rundir (`…20260610_031405`) — and that rundir is **not a cat-transfer
> run at all** but the FL *fully-shared* intrinsic-test run (reg ON, `NextHeadStanFlow`, reg top10≈0.73,
> grad-ratio 1.78). FL cat-transfer at {1,7,100} had never run; the `ls -dt|head` capture race mis-mapped
> it. Re-ran the genuine reg-OFF ablation at {1,7,100} (reg cratered ≈0 ✓, distinct PID-suffixed rundirs):
> true FL cat+trunk = **72.24 ± 0.03** (tighter σ). Net effect on the published numbers: architecture
> **+2.13 → +2.27**, transfer **+1.08 → +0.93** — **transfer sign unchanged (positive), moved −0.15pp
> (< the 0.3pp flag threshold)**, and now *closer* to the original seed0 estimate (+0.89): the
> contamination had mildly inflated transfer. The qualitative verdict is unchanged.

**Verdict — the cat gain is ARCHITECTURE-DOMINATED; region transfer is modest and scale-dependent.**
The cross-attn shared trunk accounts for +2.3 to +3.2 pp; genuine region co-training adds only **+0.93
at FL** (large state) and slightly **HURTS at AL (−0.67)** — at small data the region signal mildly
distracts the cat head. This is exactly what the gradient-orthogonality finding predicts: orthogonal
gradients ⇒ little *direct* cross-task transfer ⇒ the category improvement is mostly a better encoder,
with a small representation-level transfer that only materializes where there's enough data (FL).

**Caveat (makes the conclusion conservative).** The STL cat ceiling used `next_gru` + logit-adjust
τ=0.5, while cat+trunk and G use plain CE. T2V.7/B-A4 showed logit-adjust *helps* STL cat (+2.7) but
*hurts* MTL cat — so the STL ceiling here is, if anything, an *inflated* comparand. Against a
plain-CE STL baseline the architecture component would be even larger → the architecture-dominance
conclusion is robust.

**Paper implication.** Re-state the cat result precisely: MTL category beats single-task **mostly
because the joint cross-attn architecture is a better category encoder**, with a small genuine
region→category transfer at large scale (+0.89 FL). It is NOT primarily "the region task teaches
category." This is the honest, orthogonality-consistent framing.

## (b) T5.3 — hierarchical-softmax reg head (the last live Tier-5 lever)

**Question.** Does hierarchical softmax (HSM) beat flat softmax for the high-cardinality reg head
(FL 4.7k regions)? The HSM head (`next_stan_flow_hsm`) is **single-pathway** (no dual-tower), so test
the *mechanism* at the STL/ceiling level first (cheap) before any dual-tower-HSM build.

**Method.** Built the FL region hierarchy (`build_region_hierarchy.py`, 69 clusters). p1 at FL,
prior-OFF, 5f seed0: `next_stan_flow_hsm` (hierarchy) vs `next_stan_flow` (flat). Driver
`scripts/mtl_improvement/t53_hsm_stl_test.sh`.

| reg head | FL reg Acc@10 (full) |
|---|---|
| `next_stan_flow` (flat softmax) | 73.22 ± 0.77 |
| `next_stan_flow_hsm` (hierarchical) | 73.21 ± 0.80 |

**Verdict — FALSIFIED.** HSM = flat (−0.01, within σ 0.8). Hierarchical softmax gives **no accuracy
gain** at 4,700 classes — it is a speed/memory technique, not an accuracy improver, and flat softmax
on a GPU is fine at this cardinality. **No dual-tower-HSM build is motivated** for G (saved the code
+ compute). Tier 5's last residual is closed: flat softmax is sufficient. (Larger states CA 8.5k /
TX 6.5k were not tested, but the FL null at 4.7k + the speed-not-accuracy nature of HSM make a gain
there unlikely; flag as untested if a reviewer asks.)

## Net
- (a) The cat-beats-STL story is **architecture-dominated** (cross-attn trunk), region-transfer modest
  (+0.89 FL / −0.71 AL) — consistent with orthogonal gradients. Refines the paper claim.
- (b) HSM falsified — flat softmax sufficient; Tier 5 closed. **Champion G unchanged.**
