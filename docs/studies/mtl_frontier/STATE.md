# mtl_frontier — STATE

**Status:** SCAFFOLDED, not launched · **Machine:** A40 · **Created:** 2026-06-14
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Level / blocking
- Level 0 (exploration). Blocks: `closing_data` P2 FREEZE (a promoted lever → v17 → re-pin before freeze).
- Runs in parallel with `second_dataset` (Mac) and `closing_data` P1a (reading).

## First-wave queue
| ID | Lever | State | Verdict |
|---|---|---|---|
| R1 | log_C co-location prior + probability-chain | **CLOSED 2026-06-15** | **NULL** — real but sub-threshold (AL multi-seed Δreg **+0.207±0.196**, p=0.008 15/20 pairs, gate ≥0.3 FAIL; FL seed0 +0.171 / cat −0.27; W non-monotonic, peaks at 0.2). Not v17. See `FINDINGS.md §R1`. |
| R2 | STEM-AFTB gating sweep | **CLOSED 2026-06-15** | **NULL** (not v17) — reg clean multi-seed null at all states; cat lift is AL-only & **decays with scale** (AL +0.636 / AZ +0.173 / GE +0.158 / **FL −0.026**); user-approved multi-state confirm → does NOT generalize. Citable STEM-AFTB dose-response. See `FINDINGS.md §R2`. |
| R3 | live cross-task distillation | **CLOSED 2026-06-15** | **NULL** — CrossDistil (warm-up+error-correct+reverse) doesn't beat G+log_T-KD. Fwd refinements don't rescue R1 (AL cat −0.18); reverse reg→cat AL cat +0.45 seed0 → multi-seed **+0.100±0.282** (p=0.31, one seed −0.34); FL null. log_T-KD saturates the output-prior family. See `FINDINGS.md §R3`. |

Later waves (R4–R9) gated on first-wave outcomes — see `AGENT_PROMPT.md`.

**R10 (★ user-requested) — Memory-Caching / GRM gating at the layer level** (arXiv:2602.24281, no code).
Second-wave architectural lever adjacent to R2: GRM-gated / SSC-routed read between the dual towers
(primary), and GRM/Memory-Soup fusion across Check2HGI hierarchy levels (speculative, STL-first).
"On the layers, not the transformers." Run R2 first; promote ≥0.3 pp over G, multi-seed. See `AGENT_PROMPT.md §R10`.

## Promote-gate convention
≥0.3 pp either head, multi-seed {0,1,7,100} → STOP for user (recipe → v17) → register in `closing_data` G0.2.
Null → log here + `../log.md` row; do not silently fold into the freeze.

## Decisions log
- 2026-06-14 — scaffolded from `docs/research/mtl_frontier.md` §4 (R1–R9). Optimizer aisle declared closed
  (19-arm null + Kurin/Xin/Mueller); only R9 residual sanity arms remain citable-cheap.
- 2026-06-15 — **R1 launched + CLOSED NULL.** Built train-only per-fold/seed `P(region|cat)`
  (`compute_region_colocation.py`) + ESMM KD coupling on top of log_T-KD (`--log-c-kd-weight`,
  default off; G unchanged). Screen AL+FL seed0 → AL +0.331 (promote-eligible) / FL +0.171 null →
  multi-seed AL {0,1,7,100} = **+0.207±0.196 (gate ≥0.3 FAIL)**, Wilcoxon p=0.008; weight sweep
  non-monotonic (peaks W=0.2, craters at 0.6). Real-but-small incremental signal over log_T-KD;
  weak-7-class-auxiliary + spatial overlap with log_T. Proceeding to **R2 (STEM-AFTB gating sweep)**.
- 2026-06-15 — **R2 launched + CLOSED NULL.** Built directional per-layer AFTB gates
  (`detach_ab`/`detach_ba` + `aftb_spec`, champion G unchanged). AL seed0 → all 5 configs cross gate;
  AL multi-seed → reg null (best +0.173 p=0.009 sub-threshold), cat AL-only high-var lift. User-approved
  multi-state confirm (aftb_late, AZ/GE seed0 + FL {1,7,100}): **cat decays with scale, AL-only**
  (AL +0.64 / AZ +0.17 / GE +0.16 / FL −0.03) → does NOT generalize → NOT v17. Inverse-G′. Citable
  STEM-AFTB dose-response (cross-task gradient is small-state harmful noise; reg unaffected — sharing
  topology doesn't move the reg gap). Proceeding to **R3 (live cross-task distillation)**.
- 2026-06-15 — **R3 launched + CLOSED NULL.** Built CrossDistil: warm-up gating + error-correction on
  the fwd cat→reg co-loc KD + a new reverse reg→cat arm (`log_C_rev`=P(cat\|region), `--cat-kd-weight`).
  Screen vs G+log_T-KD: fwd refinements don't rescue R1 (AL cat −0.18); reverse AL cat +0.45 seed0 →
  multi-seed +0.100±0.282 (p=0.31, seed1 −0.34) → noise; FL null. log_T-KD saturates the output-prior
  family. **First wave (R1/R2/R3) = three nulls, all the same regime** (small-state-only, FL-null,
  reg-immovable). User decision "full R3 then R10" → proceeding to **R10 (GRM/SSC gated read)**.
