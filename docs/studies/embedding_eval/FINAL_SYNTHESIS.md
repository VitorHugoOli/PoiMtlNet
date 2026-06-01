# embedding_eval — Final Synthesis (macro)

**Study goal (recap).** Build a better *evaluation* of the Check2HGI substrate so we can (a) compare proposed embedding improvements rigorously and (b) **measure them against HGI on next-reg**, with improvements layered on **v13** (`check2hgi_resln_design_b` = ResLN + Design-B POI2Vec). MTL is **Part 2** — the plan is to first strengthen the embeddings / close the next-reg HGI gap at STL, then a separate study works the MTL on top of the chosen substrate.

---

## 1. What this study delivered: the evaluation ladder
A 4-level, leak-aware, controlled substrate evaluation (`docs/studies/embedding_eval/README.md`):
- **L0** train-free geometry (kNN-LOO, silhouette, centroid-sep, **adjacency-coherence** for region, CKA), **L1** linear probe, **L2** STL sequence task, **L3** MTL (Part-2 ground truth).
- Each task probed on the **artifact it consumes** (next-cat→final embedding; next-reg→**region embeddings**) — caught a real wrong-artifact bug.
- **Same-protocol control**, **placeid-isolated folds**, **leak gates** (per-step + autocorrelation-ceiling + L2-vs-control), multi-state (FL/AL/AZ), two adversarial advisor passes.
- Found + fixed 3 real bugs: wrong region artifact; baseline-mismatch v3c "gain"; **MTL region modality hard-coding check2hgi region embeddings** (`region_sequence.py`).

This is the study's primary product: a metric stack that the next phase can reuse.

---

## 2. Macro comparison — canonical (v11/v12) vs v13 vs our improvements vs HGI

### next-cat — Check2HGI family DECISIVELY beats HGI (no gap to close)
STL `next_gru` macro-F1 (RESULTS_TABLE §0.3) — Check2HGI 1.5–1.9× HGI at all 5 states:
| state | Check2HGI | HGI | v13 (resln+B) |
|---|---|---|---|
| FL | 63.4 | 34.4 | **70.6** (widens) |
| AL | 41.4 | 25.3 | **51.3** |
| AZ | 43.9 | 28.7 | **52.5** |
Our L0/L1 confirm: probe acc check2hgi-family ~0.98 vs HGI ~0.68; v13/resln top. **Category is a Check2HGI win; HGI is far behind.**

### next-reg — HGI ahead; the gap is NOT closed at the substrate level (incl. v13)
⚠ **Provenance correction (2nd advisor).** Two different number sources must not be conflated:
- **Borrowed (prior tier_resln study, NOT re-verified here):** v13 STL `next_stan_flow` Acc@10 — AL 61.99 (ties HGI 61.86), AZ 52.98 (80% of gap), FL 70.21 (30%). Possibly stale-log_T-era / different protocol.
- **This study's OWN controlled L2** (`nextreg_stl.md`, same-protocol, FL): check2hgi 0.7274, **resln 0.7275 (≈ canonical — TIES, does NOT close the gap)**, HGI **0.7362**. ⇒ under our controlled ladder, **v13/resln is region-NEUTRAL (HGI still ~0.9pp ahead at FL); the gap is NOT reproduced as closed.**

So the honest position: **the next-reg HGI gap (≈1–3pp) remains OPEN; no substrate we tested — canonical, v13, or any re-screen candidate — closes it robustly under our controlled evaluation.** (The prior "v13 ties HGI at AL" is unverified here and may be small-state noise.)

**Our re-screen (L0–L2, vs same-protocol control) — none improves on v13:**
| candidate | next-cat | next-reg | verdict |
|---|---|---|---|
| v3c (WD 5e-2) | no-op | no-op (below ctrl) | falsified (3 states) |
| T2.4 DropEdge | no-op | no-op | falsified |
| T4.3 sidefeat | ≤ ctrl (hurts MTL cat −1.6pp) | adj_coh +0.05–0.07 but L2 +0.3pp NS, redundant w/ log_T | not robust |
| T6.1 p2p | no-op | no-op | falsified |
| GATv2 | **leak** (fwd-temporal cat copy) | — | disqualified |
| R-GCN | **leak** | adj_coh high (leak-inflated) | disqualified |
| GCN² region head | (n/a) | +3.0pp under next_gru but **+0.56pp NS under stan_flow** (redundant w/ log_T); AL/AZ residual within 1 SD | not robust |

**adj_coh detected HGI's region edge (3/3 states) and a small real sidefeat structure — but the deployed `next_stan_flow` log_T prior already captures that spatial signal, so neither a higher-adj_coh substrate nor an adjacency-aware head adds robustly.**

---

## 3. Where we are (corrected after 2nd advisor — honest status)
- **Evaluation goal: ACHIEVED, with one standing limit.** The ladder is rigorous, leak-aware, HGI-anchored, self-corrected 3 bugs — a reusable harness. **Limit (the study's own founding worry): L0–L2 screen STL; they cannot certify an MTL-only effect.** v13 itself (STL-best, zero-MTL-benefit) is the proof. So "no candidate survives STL" ≠ "no candidate helps MTL."
- **next-cat: solved in Check2HGI's favour** (1.5–1.9× HGI; v13 widens). Robust, multi-level.
- **next-reg gap vs HGI: OPEN.** Under our controlled ladder, no substrate (canonical, v13, or any re-screen lever) closes it robustly. v13 is region-**neutral** (ties canonical), not gap-closing, in our runs.
- **"Structural / not closeable" — DOWNGRADED to "no robust substrate-level lift detected."** The GCN²/adjacency lever is redundant-with-log_T at FL (+0.56pp NS) and within ~1 SD at AL/AZ — BUT the sign is **positive 3/3 (mean ≈+0.9pp)**, a sub-noise-floor effect that is **untested at adequate power**, not proven absent. Most of this study's MTL/L2 evidence is **single-seed (dev seed 42)** — the weakest link.

**⇒ Carry v13 to Part 2 — but for the honest reason:** it is the **strongest category substrate and region-neutral** (does not hurt reg), so it is a safe base. It is **NOT** carried because "it closes the region gap" (unverified here) or "nothing beats it in MTL" (single-seed-42 only). Carrying v13 to MTL is a **bet** that its STL representational edge eventually pays off under a better joint regime — reasonable, but a bet.

---

## 4. How to proceed — the ONE thing before Part 2, then Part 2
**Before committing to Part 2 — run the deferred paired multi-seed test** (the study's biggest gap): seeds {0,1,7,100}, paired per-fold, FL + one small state, of the **two surviving real-geometry levers** against the GCN+log_T baseline IN MTL:
1. **+sidefeat substrate** (the only positive, base-independent, leak-free geometry signal: adj_coh +0.055 on GCN & ResLN, best silhouette, CKA 0.90 = real change). Its MTL behaviour on a multi-seed basis is genuinely unknown (single-seed-42 showed cat −1.6pp, but dev-seed-42 overshoots).
2. **Adjacency-aware region head** (the strongest *positive* finding: +2.7–3.0pp under next_gru, ~7 SD). The lever is the **head, not the substrate** — but it must be designed **orthogonal to log_T** (GCN² re-derived log_T's observed-transition signal and collapsed; a head encoding what log_T can't — multi-hop reachability / hierarchy / cold-start regions — is the real lead).

This replaces dev-seed-42 with reporting seeds, tests in the **MTL regime the ladder cannot certify**, and resolves the sign-consistent ~0.9pp sub-noise effect. If null at power → "carry v13, region gap is a substrate dead-end" is *earned*. If positive → Part 2 has its target.

**Then Part 2 (MTL on v13):**
- **Primary hypothesis: the adjacency-aware / region-graph reg head** (orthogonal to log_T) — the most actionable positive lead.
- **Secondary: dual-substrate routing** (Check2HGI final emb for cat ⊕ HGI region emb for reg) — Check2HGI owns cat, HGI owns region. Lower priority (Design-A late-fusion failed in MTL; task-routed variant untested).
- Prior-pathway work (log_T-KD already gives +2–5pp small-state MTL-reg) remains the only *confirmed* MTL-reg lever.
- **Reuse this ladder** as the Part-2 screening harness — but always confirm survivors in **MTL multi-seed**, never STL-only.

**Bottom line (corrected).** The study **achieved its evaluation goal** (a rigorous, leak-aware, HGI-anchored ladder) and cleanly **falsified 5 of the proposed improvements + 2 leaks**. But the **next-reg HGI gap is OPEN** — *no* substrate (v13 included) closes it robustly under controlled evaluation; v13 is the best **category** engine and **region-neutral**, so it is the safe Part-2 base. The two real-geometry survivors (**sidefeat**, **adjacency-aware head**) were screened-out at STL but **never tested in MTL at power** — and since the whole study exists because STL≠MTL, that multi-seed MTL test is the mandatory first step before declaring the region gap a substrate dead-end.
