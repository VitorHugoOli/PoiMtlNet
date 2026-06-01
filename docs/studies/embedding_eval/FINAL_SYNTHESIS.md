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

### next-reg — HGI ahead; v13 best closes the gap; nothing we re-screened improves on v13
STL `next_stan_flow` Acc@10 (RESULTS_TABLE §0.3 + tier_resln). HGI ahead by 1.6–3.1pp (canonical); **v13 closes most of it**:
| state | v11/canonical (Check2HGI) | **v13 (resln+B)** | HGI | gap v13→HGI |
|---|---|---|---|---|
| AL | 59.15 | **61.99** | 61.86 | **+0.13 (TIES HGI)** |
| AZ | 50.24 | 52.98 | 53.37 | −0.39 (80% closed) |
| FL | 69.22 | 70.21 | 71.34 | −1.13 (30% closed) |

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

## 3. Where we are
- **Evaluation goal: ACHIEVED.** The ladder gives the rigorous, leak-aware, HGI-anchored comparison the study set out to build — and it self-corrected 3 real bugs.
- **next-cat: solved in Check2HGI's favour.** v13 widens the already-decisive lead over HGI.
- **next-reg gap vs HGI: best closed by v13** (ties at AL, 80% at AZ, 30% at FL) — established by tier_resln and **confirmed here as the frontier**: none of the 6 re-screened levers (v3c/dropedge/sidefeat/p2p/gat/rgcn/gprop) closes the **residual** FL gap (~1.1pp); two are leaks, the rest are no-ops or log_T-redundant.
- **The residual next-reg gap is structural** (HGI's hierarchical region-graph) and **largely redundant with the log_T transition prior** at deploy — so it is not closeable by check-in-substrate tweaks.

**⇒ The substrate to carry to Part 2 is v13** (`check2hgi_resln_design_b`): best dual-axis STL engine, decisive on cat, closes most of the region gap. Our extensive re-screen says nothing beats it at the substrate level.

---

## 4. How to proceed (Part 2 = MTL)
- **Carry v13 as the Part-2 substrate.** (Caveat from §0.9 / our L3 spot-checks: substrate/encoder gains are **regime-limited in MTL** — v13 ≈ canonical in MTL; the only lever that moved MTL-reg is the **log_T-KD prior (v12, +2–5pp small states)**. So Part 2's leverage is the **joint-training architecture + prior pathway**, not the substrate alone.)
- **To close the residual next-reg gap vs HGI** (the part substrate tweaks can't reach): the eval points at **HGI-style region-graph structure**. Options for a future study: (a) a region-graph-aware reg head whose spatial signal is *orthogonal* to log_T (our GCN² proxy was redundant with log_T — needs a design that adds beyond it); (b) a **dual-substrate fusion** (Check2HGI final embedding for cat ⊕ HGI region embeddings for reg) — Check2HGI owns cat, HGI owns region; Design-A late-fusion failed in MTL, but a task-routed fusion is untested with this evaluation.
- **Reuse this ladder** as the Part-2 screening harness (L0 leak gates + adj_coh + L2/L3), now hardened.

**Bottom line:** the study succeeded at its evaluation goal and identified **v13 as the strongest substrate** (decisive cat, best region-gap closure). No proposed improvement beats v13; the residual HGI region edge is structural and log_T-redundant, so closing it further is an architecture/fusion problem for Part 2, not a substrate-tweak problem.
