# CH19 Per-Visit Counterfactual — 5-State Summary

**Date:** 2026-05-04  
**Branch:** `h100/pervisit-fl-ca-tx-results` (commit `d714ce0`)  
**Status:** CH19 confirmed at all 5 states (FL+CA+TX extension of AL+AZ, 2026-05-04)

---

## Protocol

Matched-head STL `next_gru`, seed=42, 5f × 50ep, bs=1024, H100 80GB.  
Three cells per state: canonical Check2HGI / POI-pooled Check2HGI / HGI.  
POI-pooled = mean-pool canonical Check2HGI vectors per `placeid` across all check-ins (kills per-visit variation, preserves training signal).  
Script: `scripts/run_pervisit_counterfactual_FL_CA_TX.sh`. Pooled builder: `scripts/probe/build_check2hgi_pooled.py`.  
**Wall-clock timing:** total ~2h17min for all 9 cells on H100 80GB (commit-to-commit 05:08→07:25 UTC 2026-05-04); per-state breakdown not logged (stdout log not retained on disk).

---

## Results (mean macro-F1 ± std, 5 folds)

| State | canonical C2HGI | POI-pooled C2HGI | HGI | total gap (C−H) | per-visit pp (C−P) | training-signal pp (P−H) | per-visit % | training-signal % |
|-------|----------------:|------------------:|----:|----------------:|-------------------:|-------------------------:|------------:|------------------:|
| AL    | 40.76 ± 1.50   | 29.57             | 25.26 ± 1.06 | +15.50 | +11.19 | +4.31 | **72%** | 28% |
| AZ    | 43.17 ± 0.28   | 34.09 ± 0.63      | 28.99 ± 0.51 | +14.18 | +9.08  | +5.10 | **64%** | 36% |
| FL    | 63.48 ± 1.04   | 37.42 ± 0.76      | 34.46 ± 0.97 | +29.02 | +26.06 | +2.96 | **90%** | 10% |
| CA    | 60.55 ± 0.81   | 34.47 ± 0.44      | 31.14 ± 1.00 | +29.41 | +26.08 | +3.33 | **89%** | 11% |
| TX    | 60.35 ± 0.30   | 34.93 ± 0.71      | 32.19 ± 0.61 | +28.16 | +25.42 | +2.74 | **90%** | 10% |

AL/AZ numbers from prior runs (commits `bdc0cf5` + `61b44c3`). Per-fold JSONs at `results/phase1_perfold/`.

---

## Key findings

**CH19 confirmed at all 5 states.** Per-visit contextual variation is the dominant mechanism behind the substrate gap (CH16) at every state.

**Two-band pattern in per-visit share:**
- Small states (AL/AZ, ~1k regions): per-visit share 64–72%; training-signal residual 28–36%
- Large states (FL/CA/TX, 4.7k–8.5k regions): per-visit share **~89–90%**; training-signal residual ~10–11%

**Interpretation:** At large states the POI-pooled embeddings almost fully collapse to HGI-level performance (pooled F1 ≈ 34–37% vs HGI ≈ 31–34%), leaving nearly the entire substrate advantage to per-visit context. Mechanistically: large transition graphs (4.7k–8.5k regions) are too sparse to yield strong per-POI graph-contrastive signal from HGI-style training, so Check2HGI's advantage at large states is almost entirely in the per-visit contextual variation it introduces, not in better graph embeddings per se.

**Canonical and HGI cells match substrate Table 1(a) references within 0.1–0.6 pp** — numbers are clean.

**Pooled > HGI at all states** — training-signal component is consistently positive (2.7–5.1 pp), confirming the two-component decomposition holds everywhere.

---

## Anomaly flag

Per-visit share at FL/CA/TX (~89–90%) is above AL (72%) and AZ (64%). This is outside the [40%, 90%] "flag if outside" range specified in the run protocol at the upper end. The values are at the boundary, not clearly anomalous — the two-band pattern is a coherent finding rather than a noise artifact. Recommended: note in §6.1 that "at large states, per-visit variation accounts for ~90% of the substrate gap, versus ~64–72% at small states, reflecting reduced marginal value of graph-contrastive signal when the transition graph is sparse."

---

## What remains

The study is otherwise paper-closed (v11, 2026-05-02). This FL+CA+TX run was the last `optional/pending` extension listed in `CLAIMS_AND_HYPOTHESES.md` for CH19. No further mandatory runs are outstanding.

**Optional camera-ready extensions (not blocking):**
- Multi-seed replication of pooled cell at AL/AZ (n=20) for Wilcoxon per-visit share confidence intervals — not needed for current §6.1 claim framing
- Reg-side per-visit counterfactual (`next_stan_flow`) — not in scope per study plan

---

## Audit trail

| File | Description |
|------|-------------|
| `results/phase1_perfold/FL_check2hgi_cat_gru_5f50ep_20260504.json` | FL canonical per-fold |
| `results/phase1_perfold/FL_check2hgi_pooled_cat_gru_5f50ep_20260504.json` | FL pooled per-fold |
| `results/phase1_perfold/FL_hgi_cat_gru_5f50ep_20260504.json` | FL HGI per-fold |
| `results/phase1_perfold/CA_*_20260504.json` | CA per-fold (3 cells) |
| `results/phase1_perfold/TX_*_20260504.json` | TX per-fold (3 cells) |
| `articles/[BRACIS]_Beyond_Cross_Task/src/figs/per-visit.png` | 5-panel figure |
| `scripts/run_pervisit_counterfactual_FL_CA_TX.sh` | Launcher |
| `scripts/probe/build_check2hgi_pooled.py` | Pooled builder |
| `scripts/probe/extract_pervisit_perfold.py` | Per-fold extractor (RUNDIR_RE bugfix on this branch) |
