# Backbone Dilution in MTL — Audit

**Severity:** HIGH — likely root cause of CH01 failure on AL. May or may not persist on FL/CA/TX.

**Detected:** 2026-04-17 during P2-validate run comparing Transformer-head vs GRU-head MTL at matched compute on AL.

**Status:** CHARACTERISED, not yet fixed. Candidate fixes being researched in parallel with FL data-richness test.

---

## TL;DR

MTL with the current shared-backbone architecture (FiLM/CGC/MMoE/DSelectK/PLE + 4-layer residual shared block) **dilutes** the region task when its head is strong standalone, and **lifts** it when its head is weak standalone. The net is that MTL cannot exceed the strongest single-task baseline under this architecture on AL's small-data regime (10 K train samples).

Quantitatively (AL, 5f × 50ep, mtlnet_dselectk+pcgrad, per-task modality, fair folds):

| Task-B head | Standalone (P1) | MTL (this study) | MTL lift |
|---|---|---|---|
| Transformer (`next_mtl`) | **7.40%** Acc@10 | 47.62% Acc@10 (budget test) | **+40.22 pp** 🎉 |
| GRU (`next_gru`) | **56.94%** Acc@10 | 48.88% Acc@10 (validate) | **−8.06 pp** 😬 |

And on category (task A, both MTL runs the same head — Transformer-style `next_mtl`):
- STL fair: **38.58%** F1
- MTL (either task-B head): 36.08–36.67% F1 (dilution ~2 pp regardless of task-B head)

So category is dilution-bound at roughly the same level whichever task-B head we choose. Region dilution-or-lift depends entirely on the standalone ceiling of the task-B head.

---

## Mechanism (hypothesis)

The shared backbone (4 residual blocks × `shared_layer_size=256`) is a **capacity bottleneck** that must represent signal for both tasks simultaneously. Its capacity is:

- Fixed regardless of how strong each head is downstream.
- Split between the two tasks (via FiLM modulation / expert routing / etc.).
- Trained jointly on a joint loss (MTL optimizer).

Consequences:

1. **Weak-head regime** (Transformer on region): the head alone cannot extract the signal from the region input. The shared backbone — which DOES see transferable category context via its joint training — compensates, lifting region performance far above what the broken head could achieve alone. Shared backbone is a *help*, not a cap, because the head can't saturate the signal alone.

2. **Strong-head regime** (GRU on region): the head alone already extracts near-all the signal from its 9-step region input. The shared backbone's *fixed* capacity cannot add value beyond what GRU-alone already captures; meanwhile, FiLM modulation + shared-layer transformations introduce representation-space rotations optimized for *both* tasks, which are *suboptimal* for region alone. Net: dilution.

3. **Category head (Transformer)** sits in a middle regime: its 7-class output task doesn't fully saturate its own capacity, so the shared backbone can nominally help. But on AL's small data, the backbone's capacity split slightly penalises category too — not as much as region (−2 pp) because the category task has less signal to extract from 9-step check-in input anyway.

---

## Why this matters for CH01

If backbone dilution is the mechanism behind CH01's failure on AL:

- **CH01 might still succeed on FL/CA/TX** because larger datasets let the shared backbone learn task-specific subspaces within its 256-dim capacity more effectively. With 10 K samples, every parameter is under-fit; the backbone can't specialise well. With 127 K samples (FL), it might.

- **CH01 might fail on all states**, in which case our MTL architecture is fundamentally wrong for bidirectional improvement. The paper pivots to the nuanced "MTL lift inversely correlates with standalone head strength" insight and de-emphasises CH01 as a headline claim.

- **CH03 (per-task modality) and CH16 (Check2HGI > HGI)** are both unaffected by backbone dilution. They both target different comparisons.

---

## Candidate fixes (to be researched + tested)

1. **Per-task skip connections** — add a direct path from task encoder → task head that bypasses the shared backbone. If the shared backbone can dilute, give each head an escape hatch. Expected: eliminates dilution (head falls back to standalone when backbone doesn't help), preserves lift (when backbone does help, signal combines additively).

2. **Larger shared backbone** — double `shared_layer_size` from 256 to 512 or add more blocks. Expensive. Hypothesis: dilution is a capacity issue; more capacity reduces it.

3. **Asymmetric task scheduling** — train task B (stronger head) alone for N epochs, then jointly for M epochs. Curriculum approach. Hypothesis: the backbone specialises on task B first, then learns to also serve task A.

4. **Different loss balancing** — pcgrad/nashmtl project gradients but don't prevent dilution of representations. Try losses that explicitly encourage representation divergence (e.g., uncertainty weighting with strong prior on task-B dominance).

5. **Cross-attention between task encoders** instead of a shared backbone — lets each task encoder keep its full capacity, cross-attention enables information transfer on demand. Higher-risk, higher-reward.

6. **LoRA-style adapters on a frozen strong-task backbone** — train each task's STL-optimal backbone, then fine-tune with LoRA adapters that allow task-A signal to enter task-B's representations without rewriting them.

---

## How to decide which fix

1. **FL test (1f × 50ep)** answers "is dilution data-dependent or architectural?" Result in ~90 min.
2. **Research subagent** collects 2024–2026 SOTA MTL papers addressing similar failure modes.
3. Based on (1) + (2), pick the cheapest plausible fix and implement + test.

---

## Paper narrative implications

Three scenarios:

- **Best case**: FL succeeds (r_A > 0 AND r_B > 0). AL failure is framed as "small-data regime; MTL's shared-backbone benefit requires enough samples to avoid capacity split starvation." Paper reports both AL (fails) and FL/CA/TX (succeeds) — the contrast itself is a useful finding.

- **Middle case**: FL partially succeeds (one head positive, the other within σ). Frame as "MTL shows task-asymmetric lift; quantify the gain on the helped head." CH01 loosened from "both improve" to "no regression on either, positive lift on one."

- **Worst case**: FL fully fails. CH01 retired. Paper leads with CH16 (substrate) + CH03 (architectural) + this mechanistic-insight finding as its three contributions. The insight — "MTL lift correlates inversely with task-specific head's standalone strength" — is genuine and publishable on its own.
