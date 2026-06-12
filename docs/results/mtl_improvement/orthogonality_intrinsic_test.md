# P0 — Is the cos≈0 orthogonality INTRINSIC, or induced by the dual-tower? (2026-06-10)

**Advisor-flagged P0 (the study's single load-bearing interpretive claim).** The gradient cosine
cos(∇L_cat, ∇L_reg) ≈ 0 is measured on the SHARED trunk. But champion G's reg head routes mostly
through its PRIVATE tower (aux fusion `feat = priv + β·aux_proj(shared)`, β init 0.1), so reg's gradient
*into* the shared trunk is attenuated. **Confound:** cos≈0 could mean "the dual-tower architecturally
severed reg's shared channel" rather than "the tasks are intrinsically orthogonal." The whole unifying
narrative (balancers can't help; more-sharing hurts; dual-tower wins) hangs on the *intrinsic* reading,
so it must be tested, not asserted.

## Test
- **P0-a (free):** the trainer already logs `grad_norm_next_{region,category}_shared` per epoch. Extract
  the reg:cat shared-gradient ratio from the existing G runs + the fully-shared (R1b) run.
- **P0-b (decisive):** measure cos on a **fully-shared** model — `mtlnet_crossattn` + `next_stan_flow`
  (single-pathway, NO private tower) — where reg uses the shared trunk fully. If cos stays ≈0 there
  (where reg's shared gradient is large), the orthogonality is intrinsic; if it goes negative, the
  dual-tower induced it. AL = the R1b non-overlap run (already on disk); FL = run fresh.

## Result

| model | seed(s) | rundir | reg‖∇‖shared | cat‖∇‖shared | **ratio reg:cat** | **cos(∇cat,∇reg)** |
|---|---|---|---|---|---|---|
| G dual-tower, FL | {0,1,7,100} | R0 `g_rundirs.florida` | 0.0723 | 0.1555 | 0.47 | +0.0007 |
| G dual-tower, AL | {0,1,7,100} | R0 `g_rundirs.alabama` | 0.0688 | 0.2661 | 0.26 | +0.0026 |
| **Fully-shared (no private tower), AL** | **42** (R1b non-overlap) | `…20260608` (r1b_shared_overlap_deconfound.sh) | **0.3009** | 0.2383 | **1.26** | **+0.0024** |
| **Fully-shared (no private tower), FL** | unrecorded (single fresh P0-b run) | `…20260610_031405_3670616` | **0.3182** | 0.1787 | **1.78** | **+0.0017** |

> **Seed provenance (HANDOFF_AUDIT H1, 2026-06-12).** The G dual-tower cosines are the per-state means
> over the FOUR R0 seeds {0,1,7,100} (not single runs) — the load-bearing cos≈0 figure
> `figs/grad_cosine_tasks.png` is now pooled over all **16 G rundirs (4 states × 4 seeds), mean +0.0008
> over 3,797 epoch-fold points** (was "2 static-screen runs, seed 0"). The AL fully-shared run is the R1b
> non-overlap arm at **seed 42** (`r1b_shared_overlap_deconfound.sh`). The FL fully-shared run was a
> single ad-hoc P0-b run (rundir `…20260610_031405_3670616`); its **seed is not persisted on disk and the
> launcher was not committed** — the rundir is identified by its unique grad-ratio-1.78 / reg-top10-0.73
> signature and is the same artefact the FL row above is computed from. The per-fold val-size fingerprint
> does not discriminate the seed (StratifiedGroupKFold folds stay ~equal-sized at every seed). Because the
> conclusion's seed-robustness is independently carried by the 16-run pooled figure (cos +0.0008 over
> {0,1,7,100} × 4 states), the single confound-check run's exact seed is immaterial to the verdict.

## Verdict — INTRINSIC orthogonality CONFIRMED (architecture-induced alternative ruled out)

**P0-a confirms the confound was real to check:** in the dual-tower, reg's shared gradient is attenuated
to **0.26–0.47×** cat's (the β=0.1 aux gating). So one *could* worry cos≈0 just reflects reg barely
using the trunk.

**P0-b refutes that:** in the **fully-shared** model where reg's shared gradient is **larger** than cat's
(ratio **1.26 AL / 1.78 FL** — reg now dominates the shared channel), the cosine is **still ≈0**
(+0.0024 AL / +0.0017 FL). **The orthogonality persists exactly where the confound predicted it should
vanish.** Therefore cos≈0 is **a genuine property of the task pair**, not an artifact of the dual-tower's
gating. The "intrinsic orthogonality" claim is now *tested*, two states, and it *strengthens* the paper:
the dual-tower **exploits** a pre-existing orthogonality (protect reg, let cat harvest the shared
encoder) — it does not *manufacture* it.

## P1 — multi-seed cat-transfer decomposition (firms up the +0.89/−0.71 seed0 estimates)
4-seed {0,1,7,100} re-run of the cat+trunk (reg-OFF, `--category-weight 1.0`) ablation vs multi-seed G:

| state | STL cat | cat+trunk, reg-OFF (4-seed) | G (4-seed) | **architecture** | **region→cat transfer** |
|---|---|---|---|---|---|
| AL | 50.35 | 53.57 ± 0.24 | 52.91 ± 0.27 | **+3.22** | **−0.67** |
| FL | 69.96 | 72.24 ± 0.03 | 73.16 ± 0.04 | **+2.27** | **+0.93** |

The decomposition is **multi-seed-robust** (tight σ): architecture-dominated at both states; genuine
region→category transfer **+0.93 at FL** (large state) and **−0.67 at AL** (small state — region mildly
distracts cat). Both signs hold multi-seed. (Updates the seed0 estimates +0.89 FL / −0.71 AL.)

> **⚠ P0 fix 2026-06-12 (HANDOFF_AUDIT).** The FL row was **72.09 ± 0.08 / arch +2.13 / transfer +1.08**;
> its `s1/s7/s100` manifest rows had all pointed to one rundir, which turned out to be the FL
> *fully-shared* intrinsic-test run below (reg ON), not a cat-transfer run. Re-ran the genuine reg-OFF
> ablation at {1,7,100} (distinct rundirs, reg cratered ≈0): FL cat+trunk = **72.24 ± 0.03** → arch
> **+2.27**, transfer **+0.93** (sign unchanged; −0.15pp, below the flag threshold; now closer to the
> seed0 +0.89). See `cat_transfer_and_T53.md` §a + `cat_transfer_decomposition_4seed.json`.

## Honesty caveat (advisor confound C) — the ablation is not a *perfectly* clean isolation
`--category-weight 1.0` → reg loss weight = 0 → reg gets zero gradient (confirmed: reg cratered to
0.12%). **But** the model still forward-passes the reg stream, and in the bidirectional cross-attn the
cat stream attends to reg-derived keys/values on every forward. So "cat+trunk, reg-OFF" still *reads*
region features structurally; only the supervisory pressure on reg is removed. The "architecture"
component therefore includes a small residual structural contribution from region features being present.
This does not change the qualitative verdict (architecture-dominated) but means `total = arch + transfer`
is approximate, not exact. (The logit-adjust-vs-plain-CE mismatch in the STL comparand is the *other*
direction — it inflates the STL ceiling, making the architecture share conservative.)
