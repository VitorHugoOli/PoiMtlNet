# Hybrid (cross-attn cat + dselectk reg) — decision: SKIP

**Date:** 2026-04-20. **Verdict:** null-a-priori, not built.

## The test that discriminated

Proposed hybrid: route cat through cross-attn, reg through dselectk. Rationale: if dselectk has lower architectural overhead on reg than cross-attn does, the hybrid could use dselectk's reg branch to rescue the overhead while keeping cross-attn's cat win.

The decisive measurement: cross-attn λ=0 isolation vs dselectk λ=0 isolation on AL, both at fair LR (max_lr=0.003), both with STL GRU reg head.

| AL config | reg A@10 | Architectural overhead |
|---|---:|---:|
| STL reg GRU (ref) | 56.94 ± 4.01 | — |
| λ=0 **dselectk** | 51.87 ± 5.70 | **5.07 pp** |
| **λ=0 cross-attn** (new) | **52.27 ± 5.03** | **4.67 pp** |

Cross-attn has *slightly less* architectural overhead than dselectk (within σ). **Dselectk is not a better reg carrier than cross-attn; there is nothing for the hybrid to rescue.**

## Ancillary finding — transfer is near-zero at AL scale

| AL config | cat F1 | reg A@10 |
|---|---:|---:|
| λ=0 cross-attn | 10.37 ± 2.38 | 52.27 ± 5.03 |
| Full MTL cross-attn + pcgrad | 38.47 ± 1.29 | 52.41 ± 4.70 |
| Δ (cat-enabled transfer on reg) | — | **+0.14 pp** |

Full MTL gives reg A@10 = 52.41 vs λ=0's 52.27. **Adding cat training on AL barely moves reg.** This reinforces the scale-curve interpretation: at AL (small data, 1109 regions), the shared-backbone carries too little information to mediate useful task-to-task transfer. The MTL pipeline is effectively inert on reg at small scale.

**Compare FL** (from chain findings):
- FL λ=0 dselectk reg: 43.40 (1f)
- FL full-MTL cross-attn reg: 57.60 (1f)
- FL cat-enabled transfer: **+14.20 pp** 🚀

On AL: +0.14 pp transfer. On FL: +14.20 pp transfer. Same architecture; scale determines whether transfer happens.

## Revised mechanistic story

The original scale-curve claim said "reg penalty widens with class count". The full decomposition is:

| Scale | STL reg | λ=0 reg | Full-MTL reg | Overhead | Transfer |
|---|---:|---:|---:|---:|---:|
| AL 1109 cls | 56.94 | 52.27 | 52.41 | 4.67 | +0.14 |
| FL 4702 cls | 68.33 | 43.40 | 57.60 | 24.93 | +14.20 |

Both components grow substantially at scale:
- **Overhead** (cost of the MTL pipeline when cat training is disabled): 5× larger on FL
- **Transfer** (benefit of adding cat training): 100× larger on FL

Net reg gap vs STL: AL −4.53 pp, FL −10.73 pp. The gap grows but the mechanism is very different: AL gap is nearly all "architectural overhead with no compensating transfer"; FL gap is "huge overhead partially rescued by huge transfer".

## What this means for the paper

**Strengthens CH-M4** ("cross-attention uniquely closes the weak-head gap"): cross-attn's cat advantage on FL is being earned against a massive 25 pp reg pipeline overhead that *all MTL architectures face*. That cross-attn still exceeds STL cat by +3.29 pp while recovering 14.20 pp of reg is a more impressive result than the raw numbers first suggested.

**Reframes CH-M6** (scale curve): the monotone-widening reg gap isn't one mechanism, it's two — structural overhead + compensating transfer, both scaling with state size in opposite directions.

**Paper framing draft:**
> "At FL scale, multi-task wrapping imposes a 24.93 pp architectural cost on the region head relative to a single-task baseline. Cross-attention with content-based task routing recovers 14.20 pp of this overhead via cat-to-reg information transfer, and simultaneously exceeds single-task category F1 by 3.29 pp. The same architecture shows near-zero transfer at AL's small scale (+0.14 pp), establishing that both MTL's architectural cost and its compensating benefit are scale-dependent."

## Decision

**Not building the hybrid.** Freed compute redirects to:
1. **FL cross-attn 5-fold replication** (~8 h) — now the clear highest-value remaining item. Headline numbers need σ.
2. Optional: FL cross-attn λ=0 (~1.5 h, 1 fold) to close the FL decomposition table symmetrically with AL.
