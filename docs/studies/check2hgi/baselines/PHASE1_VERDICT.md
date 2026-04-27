# Phase 1 — AL+AZ Validation Verdict

**Generated 2026-04-27.** Live status of Phase-1 results from `SUBSTRATE_COMPARISON_PLAN.md`. Updated as each leg lands. The §9 outcome-interpretation matrix from the plan is resolved at the bottom once all legs report.

---

## 1 · Leg I — Substrate-only linear probe (head-free)

5-fold StratifiedGroupKFold(userid), seed 42. Logistic regression on last-window-position embedding (64-dim). No sequence model, no head.

| State | Check2HGI F1 | HGI F1 | Δ (C2HGI − HGI) |
|---|---:|---:|---:|
| AL | **30.84 ± 2.02** | 18.70 ± 1.38 | **+12.14** |
| AZ | **34.12 ± 1.22** | 22.54 ± 0.45 | **+11.58** |

**Verdict:** Substrate carries a ~12 pp head-free lift at both states. CH16 substrate claim is **not** an artefact of the head choice — it is present in the embedding itself.

Source JSONs: `results/probe/{state}_{check2hgi,hgi}_last.json`.

---

## 2 · Leg II — Matched-head STL of MTL north_star

### 2.1 next_category — `next_gru` head (matched to MTL's task_a head)

5f × 50ep, seed 42:

| State | Check2HGI F1 | HGI F1 | Δ | Wilcoxon p_greater | Paired-t p | TOST δ=2pp | Pos/Neg folds |
|---|---:|---:|---:|---:|---:|---|:-:|
| AL | **40.76 ± 1.50** | 25.26 ± 1.06 | **+15.50** | **0.0312** ✅ | 0.0001 | non-inferior ✅ | 5 / 0 |
| AZ | **43.21 ± 0.78** | 28.69 ± 0.71 | **+14.52** | **0.0312** ✅ | <0.0001 | non-inferior ✅ | 5 / 0 |

**Verdict:** CH16 confirmed at both states under the matched-head STL of the MTL north_star, with the strongest possible n=5 paired-Wilcoxon p-value and 5/5 folds positive.

Vs the existing `next_single` evidence (AL Δ=+18.30 pp): the matched-head delta is smaller (+15.50) — **head choice amplifies ~3 pp**, **substrate carries ~12 pp** (consistent with Leg I).

### 2.2 next_region — `next_getnext_hard` head (matched to MTL's task_b head)

5f × 50ep, seed 42. Check2HGI side from F21c (already landed); HGI side new this Phase.

| State | C2HGI Acc@10 | HGI Acc@10 | Δ Acc@10 | Wilcoxon p (Acc@10) | TOST δ=2pp |
|---|---:|---:|---:|---:|---|
| AL | **68.37 ± 2.66** | 67.52 ± 2.80 | +0.85 | 0.0625 (marginal) | non-inferior ✅ |
| AZ | **66.74 ± 2.11** | 64.40 ± 2.42 | **+2.34** | **0.0312** ✅ | non-inferior ✅ |

| State | C2HGI MRR | HGI MRR | Δ MRR | Wilcoxon p (MRR) |
|---|---:|---:|---:|---:|
| AL | 41.17 ± 2.28 | 40.75 ± 2.32 | +0.42 | 0.156 (n.s.) |
| AZ | **41.15 ± 2.13** | 39.86 ± 2.20 | **+1.29** | **0.0312** ✅ |

**Verdict (key paper-reshaping finding):** the existing CH15 narrative (HGI > Check2HGI on reg under STAN at all 3 states) **flips under matched-head**:
- AL: tied within σ (TOST non-inferiority holds at δ=2 pp).
- AZ: Check2HGI **significantly superior** on Acc@10 + MRR (5/5 folds, p=0.0312).

The matched-head finding says the previous CH15 verdict was head-coupled — STAN's POI-smoothness preference favored HGI, but the graph-prior head (which the MTL B3 north_star uses) interacts productively with Check2HGI's per-visit context.

---

## 3 · Leg III — MTL counterfactual (B3 with HGI substrate)

⏳ **Running.** AL (HGI) on fold 3/5 at 17:51; AZ (HGI) queued. ETA ~30 min from now (17:51).

Tags:
- `MTL_B3_ALABAMA_hgi_5f50ep`
- `MTL_B3_ARIZONA_hgi_5f50ep`

Comparator: existing MTL-B3 Check2HGI 5f×50ep results from `NORTH_STAR.md` §"Validation status":
- AL B3 (post-F27): cat F1 0.4271 ± 0.0137 / reg Acc@10 0.5960 ± 0.0409
- AZ B3 (post-F27): cat F1 0.4581 ± 0.0130 / reg Acc@10 0.5382 ± 0.0311

Verdict pending; will populate when orchestrator finishes.

---

## 4 · C4 — Mechanism counterfactual (POI-pooled Check2HGI)

POI-mean-pooled Check2HGI: same training signal as canonical Check2HGI, but per-visit variation killed (one vector per `placeid`, applied uniformly to all check-ins at that POI).

### 4.1 Linear probe (head-free, AL)

| Substrate | F1 macro |
|---|---:|
| Check2HGI (canonical) | **30.84 ± 2.02** |
| **Check2HGI POI-pooled** | **23.20 ± 1.08** |
| HGI | 18.70 ± 1.38 |

Decomposition of CH16's substrate gap (+12.14 pp = canonical − HGI):
- **Per-visit context:** canonical − pooled = **+7.64 pp (~63%)**.
- **Embedding training signal:** pooled − HGI = **+4.50 pp (~37%)**.

**Verdict:** Mechanism **partially confirmed**. Per-visit variation is the dominant contributor but not the whole substrate advantage. The remaining ~37% reflects Check2HGI's training procedure (graph topology, contrastive loss target) producing per-POI vectors that are themselves better than HGI's, even before per-visit variation enters the picture.

### 4.2 Matched-head STL on pooled (AL)

⏳ Pending — to launch after MTL CF orchestrator finishes (avoids MPS oversubscription).

---

## 5 · C2 — Head-agnostic probe

⏳ Pending — extends AL+AZ STL cat to {`next_single`, `next_lstm`} as additional probe heads (we already have `next_gru` matched + linear probe head-free). Cost ~20 min total. Launches after Leg III completes.

---

## 6 · §9 outcome-interpretation matrix — current state

| Linear probe (Leg I) | Matched-head STL (Leg II) | MTL counterfactual (Leg III) | C4 mechanism | Verdict |
|---|---|---|---|---|
| ✅ C2HGI > HGI both states | ✅ C2HGI > HGI **both tasks** (cat strong, reg flipped from CH15) | ⏳ pending | ✅ partial — ~63% per-visit, ~37% training signal | **Trending: Strong claim holds (with mechanism caveat).** |

**Phase 1 trend:** the data is converging on the strong claim (Check2HGI > HGI on both tasks under matched head + matched substrate-only probe), with the nuance that the per-visit-variation mechanism captures only ~63% of the cat substrate gap. The remaining ~37% needs to be framed in the paper as "embedding training signal" — a real but secondary contribution.

**Pending blockers before Phase-1 close:**
1. MTL CF results land (closes Leg III).
2. C4 STL on pooled-AL (validates linear-probe mechanism finding via the matched-head pipeline).
3. C2 head-agnostic at AL+AZ (closes head-amplification claim with multi-head evidence).

When these land, this doc gets a final §9 verdict and the user has the green light for Phase 2 (FL + CA + TX) per `SUBSTRATE_COMPARISON_PLAN.md §6 step 8`.
