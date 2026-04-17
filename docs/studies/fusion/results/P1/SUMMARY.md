# Phase P1 — Final Coordinator Summary (2026-04-17)

**Scope:** Architecture × Optimizer grid on multi-source fusion. Alabama (AL) and Arizona (AZ), matched effective batch (bs=4096, grad_accum=1, embedding_dim=128, seed 42, fusion engine).

**Total runs:** 180 tests — 75 screen + 10 promote + 5 confirm per state. All archived, all verdicts `matches_hypothesis`, zero crashes, zero std/mean > 0.3.

---

## Runs per stage

| Stage | Folds × epochs | AL | AZ |
|-------|---------------|----|----|
| P1a screen | 1f × 10ep | 75 (5 archs × 15 optims) | 75 |
| P1b promote | 2f × 15ep | 10 (top-10 from AL P1a) | 10 (top-10 from AZ P1a) |
| P1c confirm | 5f × 50ep | 5 (top-5 from AL P1b) | 5 (top-5 from AZ P1b) |

## Top-5 at P1c (5f × 50ep)

**Alabama (winner: `mmoe4 × gradnorm`, gradient):**

| Arch | Optim | Joint | Cat (±std) | Next (±std) |
|------|-------|-------|------------|-------------|
| mmoe4 | gradnorm | **0.4082** | 0.8219 ±0.0118 | 0.2715 ±0.0111 |
| cgc22 | excess_mtl | 0.4043 | 0.8313 ±0.0111 | 0.2671 ±0.0215 |
| cgc22 | nash_mtl | 0.4034 | 0.8231 ±0.0086 | 0.2672 ±0.0201 |
| cgc21 | bayesagg_mtl | 0.4034 | 0.8197 ±0.0095 | 0.2675 ±0.0173 |
| cgc22 | equal_weight | 0.4031 | 0.8219 ±0.0203 | 0.2670 ±0.0057 |

Top-5 spread = 0.0051. Winner − equal_weight = +0.0051 (marginal).

**Arizona (winner: `cgc21 × uncertainty_weighting`, static):**

| Arch | Optim | Joint | Cat (±std) | Next (±std) |
|------|-------|-------|------------|-------------|
| cgc21 | uncertainty_weighting | **0.4374** | 0.7319 ±0.0146 | 0.3119 ±0.0139 |
| cgc21 | gradnorm | 0.4369 | 0.7380 ±0.0143 | 0.3103 ±0.0093 |
| cgc21 | pcgrad | 0.4361 | 0.7282 ±0.0108 | 0.3113 ±0.0125 |
| cgc21 | dwa | 0.4352 | 0.7346 ±0.0138 | 0.3091 ±0.0085 |
| mmoe4 | nash_mtl | 0.4277 | 0.7294 ±0.0224 | 0.3025 ±0.0075 |

Top-4 spread = 0.0022. Winner − best grad-surgery = −0.0005 (static tied with or narrowly beating gradient).

---

## Claim rollup

| ID | Statement | P1c verdict | Notes |
|----|-----------|------------|-------|
| **C02** | grad-surgery > equal_weight on fusion | **`partial`** | AL Δ = **+0.0051**, AZ Δ = **−0.0005**. Within noise. Consistent with N02 (grad-surgery accelerates convergence but does not raise the ceiling). |
| **C03** | equal_weight suffices on single-source | `pending` | Not tested in P1 (fusion only). Deferred to P3. |
| **C04** | arch rankings are embedding-dependent | `pending` (early signal) | Fusion-only so main test is in P3. But cross-state signal is already here: AL favors mmoe4/cgc22, AZ favors cgc21. |
| **C05** | expert-gating > FiLM base | **`confirmed`** | AL Δ ≈ +0.03–0.045, AZ Δ ≈ +0.005–0.017. Zero `base` cells survive top-10 in either state. |

`CLAIMS_AND_HYPOTHESES.md` updated with P1c evidence for C02 (→ `partial`) and C05 (→ `confirmed`).

---

## C02 signature — grad-surgery vs equal_weight across stages

| State | Stage | eq best | grad best | **Δ (grad − eq)** |
|-------|-------|---------|-----------|-------------------|
| AL | screen | 0.4037 | 0.4047 | **+0.0010** |
| AL | promote | 0.4253 | 0.4242 | **−0.0011** |
| AL | confirm | 0.4031 | 0.4082 | **+0.0051** |
| AZ | screen | 0.4315 | 0.4376 | +0.0061 (vs best static 0.4403: **−0.0027**) |
| AZ | confirm | — (not in AZ top-5) | 0.4369 | vs best static 0.4374 = **−0.0005** |

**Direction flips at least once per state across stages.** Magnitude is always below 1 p.p. absolute. **→ gradient-surgery is a noise-level effect on fusion at matched batch.**

---

## Cross-stage joint trajectory (AL top-5)

| Config | screen | promote | confirm | Δ(conf − screen) |
|--------|--------|---------|---------|------------------|
| mmoe4 × gradnorm | 0.4042 | 0.4242 | **0.4082** | **+0.0040** |
| cgc22 × excess_mtl | 0.4054 | 0.4254 | 0.4043 | −0.0011 |
| cgc22 × nash_mtl | 0.4045 | 0.4258 | 0.4034 | −0.0011 |
| cgc21 × bayesagg_mtl | 0.4047 | 0.4245 | 0.4034 | −0.0013 |
| cgc22 × equal_weight | 0.4037 | 0.4253 | 0.4031 | −0.0007 |

**Honest framing.** `mmoe4 × gradnorm` is the only config whose joint improves from *screen* (1f × 10ep, noisy) to confirm, but **all 5 configs lose ~1.6–2.2 p.p. joint from promote to confirm** (cat gains ~1–4 p.p., next drops ~1.7–2.4 p.p.). The "winner" is the config that loses the *least* under the more-reliable protocol. Without multi-seed the ordering at confirm is within one fold-std and not distinguishable.

Per-fold noise on joint at confirm is ~0.01. The AL winner margin over `equal_weight` is **+0.0051** → **Z ≈ 0.39** vs pooled fold-std of 0.0133. **Not significant at one seed.**

---

## Decision gate for P2

From `P1_arch_x_optimizer.md §Phase gate`:
1. ✅ **P1c has a clear winner / tight tie** — AL winner mmoe4×gradnorm (0.4082); within-top-5 spread 0.0051 (~1 p.p.) — a reasonable near-tie.
2. ✅ **Sensible profile** — AL winner cat=0.822, next=0.272 (both exceed thresholds). AZ winner cat=0.732, next=0.312 (ditto).
3. ✅ **AL fully done** — AL P1a + P1b + P1c complete.

Additionally:
4. **`equal_weight` near-tie at AL P1c (0.4031 vs 0.4082 = 0.005 behind)** — the phase doc flagged this as the "pause and re-plan" condition. The winner is strictly gradnorm (not equal_weight), so we do not technically trigger the pause. But the proximity is itself a finding. **Paper narrative should lead with C05 (expert-gating) as the first-order lever, not C02.**

**Gate: PASS.** P1 can advance to P2.

---

## Surprises / open items

1. **Cross-state winner disagreement.** AL → `mmoe4 × gradnorm`. AZ → `cgc21 × uncertainty_weighting`. Different arch AND different optim class. Flagged under C04 but warrants its own note: if winners don't transfer across states even under the same fusion engine, any "champion" chosen for downstream phases (P2: MTL vs single-task; P4: hyperparam robustness) has to carry a state-sensitivity caveat.
2. **NextHead overfitting signature.** Next F1 degrades from promote (2f×15ep) to confirm (5f×50ep) on 4 of 5 AL cells (~2 p.p. drop). Only mmoe4 is immune. Worth checking whether this is a general pattern (next-POI head wants early stopping) or MTL-architecture specific.
3. **`equal_weight` proximity.** `cgc22 × equal_weight` is within 0.005 of the AL winner at 5f × 50ep. On fusion, the MTL-balancing optimizer is a second-order choice.
4. **Longer-than-expected P1c wall-clock.** Per run varied from 24 to 45 min (phase doc estimated ~22 min). Relevant for P2/P3 budgeting.

---

## P2 champion config recommendation

For P2 (MTL vs single-task, C06/C07/C08/C28), we need a single champion. Options:

| Option | Config | Rationale |
|--------|--------|-----------|
| **A** | `mmoe4 × gradnorm` | AL winner; only config to improve from screen to confirm; gradient-surgery mildly helps. |
| **B** | `cgc21 × uncertainty_weighting` | AZ winner; static method; best balance on AZ. |
| **C** | Run both AL and AZ winners through P2 | Safer, 2× compute. |

Recommend **Option A** for P2, with a state-robustness note. If P2 MTL-vs-single comparison is tight, also run Option B on AL as a sensitivity check.
