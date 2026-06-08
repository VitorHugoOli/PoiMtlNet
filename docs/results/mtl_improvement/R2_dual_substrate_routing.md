# R2 — Dual-substrate routing HGI→reg under champion G (FL pilot, 2026-06-08)

**Tier 3, FL, multi-seed {0,1,7,100}.** Driver `scripts/mtl_improvement/r2_dual_substrate_routing.sh`;
manifest `scripts/mtl_improvement/r2_routing_manifest.tsv`; data `R2_routing_bar.json`.

The part-2 routing memo was shelved on the **"substrate washes out in MTL"** premise — which C25
falsified (the substrate now transfers; Δreg(v14−canon) +0.81 at FL). HGI carries a documented
**+0.36pp STL reg edge** over v14. R2 routes HGI's region embeddings into champion G's **private reg
tower** (via the `REGION_EMB_ENGINE=hgi` env-var hook, `src/data/folds.py:980`) while cat (task_a) +
the shared aux pathway stay on v14 — an inference-time swap, no rebuild (HGI `region_embeddings.parquet`
on disk at FL, 64-dim, matches the dual-tower `raw_embed_dim=64`; routing verified real: HGI vs v14
region-emb mean |Δ| = 0.30). Reg scored matched-metric (R0 method). Control = G-v14 FL from R0.

## Result (FL, 4 seeds)

| arm | reg-full | cat F1 | reg-indist |
|---|---|---|---|
| **G-HGI-routed** | **73.22 ± 0.08** | 73.05 ± 0.15 | 73.83 ± 0.08 |
| G-v14 (control, R0) | 72.97 ± 0.06 | 73.16 ± 0.04 | 73.57 |

Anchors: (c) v14 STL reg ceiling 73.27; **(d) HGI STL reg ceiling 73.49** (the matched bar for HGI
region-emb). Per-seed G-HGI reg-full: [73.14, 73.20, 73.36, 73.18].

| comparison | Δ reg |
|---|---|
| G-HGI − G-v14 (champion) | **+0.25** (cat −0.11) |
| G-HGI − **(d) HGI ceiling** (matched substrate) | **−0.27** |
| G-HGI − (c) v14 ceiling | −0.05 |
| [ref] G-v14 − (c) v14 ceiling (R0 matched gap) | −0.30 |

## Verdict — NULL on the matched bar; but the substrate edge TRANSFERS (premise falsified)

**Two findings:**
1. **POSITIVE mechanism — HGI's STL reg edge survives the joint dynamics under G.** G-HGI beats G-v14
   by **+0.25pp reg** (outside σ 0.08), capturing ~⅔ of HGI's +0.36pp STL edge. This **directly
   refutes the "substrate washes out in MTL" premise** that shelved the part-2 routing memo — exactly
   the C25 reframe (substrate now transfers to MTL). A clean confirmation, not a number chase.
2. **NULL on the deployable bar — it's a rising tide, not a gap-closer.** HGI lifts BOTH G (+0.25) and
   its own STL ceiling (+0.22 = 73.49 − 73.27) by the same amount, so the **matched G−ceiling gap is
   unchanged**: −0.27 (HGI substrate) ≈ −0.30 (v14 substrate, R0). Per the gate ("promote iff routing
   moves G−ceiling >0"), this is a **NULL** → v14 is sufficient under G. And the small absolute +0.25
   reg gain costs a **second region-embedding substrate at inference** (HGI alongside v14) — not
   justified for +0.25pp at flat cat (the original memo's cost/benefit, now quantified).

**Same shape as R1:** the reg lever (denser supervision in R1, a better substrate here) lifts G and the
ceiling equally → the matched bar holds. The mechanism story is the value: post-C25, both the data-scale
lever AND the substrate lever transfer into G's private reg tower (they did not under the old shared
backbone) — but neither beats the achievable ceiling, because the ceiling moves with them.

## Scope / caveats
- FL only (HGI region-emb on disk only at FL; AL/AZ/GE HGI margin ≤0.7pp + substrate absent — not
  worth a build for a confirmed-rising-tide lever). Multi-seed {0,1,7,100}, tight σ (0.08).
- Folds generated on-the-fly per run (G uses its own MTL fold creator); the control G-v14 is the R0
  multi-seed value (same recipe, same seeds, only the reg-tower region-emb source differs).
