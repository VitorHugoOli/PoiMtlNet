# docs/studies/ — Cross-study outcomes log

One line per study per closure or major direction shift. **Outcomes only.** Process narrative lives in each study's own `log.md`. Append-only.

| Date | Study | Outcome |
|---|---|---|
| 2026-05-04 | `hgi_category_injection` | **CLOSED** — AZ falsified; FL/CA/TX re-open standby. |
| 2026-05-06 | `merge_design` | **ACTIVE-CLOSING** — Designs A-M and Levers 1-6 saturated/falsified; Lever 5 orphaned (rescued by `substrate-protocol-cleanup` Tier B4 on 2026-05-28). |
| 2026-05-19 | `canonical_improvement` | **CLOSED** — Tier 1-6 substrate axis exhausted at ±0.8 pp; no further single-substrate ceiling. |
| 2026-05-24 | `mtl-protocol-fix` | **CLOSED v6 final** — F1 selector recovers ~95% substrate capacity at deploy (+5.6 pp FL); P4 identifies residual gap as architectural (NOT cat-vs-reg, NOT long-tail, NOT substrate). |
| 2026-05-24 | `mtl-protocol-fix` Phase 3 | **3 follow-ups** — log_T-KD §4.5 PROMOTED (+2-5 pp Wilcoxon-strict); class-balanced sampler §4.6 FALSIFIED (−18 to −30 pp); composite STL c2hgi+HGI §4.2 ESTABLISHED as current project headline (+7 to +12 pp). |
| 2026-05-16 | `mtl_improvement` | **LAUNCHED** — T0-T8 chain on branch `mtl-improve` (backbones, loss, batch, LR, α, heads, multi-seed champion); execution pending. |
| 2026-05-28 | `substrate-protocol-cleanup` | **LAUNCHED** — Tier A-D (log_T-KD multi-seed, Designs B/J/Lever 4/Lever 5 MTL+F1, 3-snapshot routing, freeze-reg-after-peak, K/V capacity-stealing pilot, window/mask audit); small-state only; ~40-45 GPU-h budget. |
| 2026-05-31 | `embedding_eval` | **LAUNCHED** — 4-level (L0 geometry → L3 MTL) leak-aware substrate evaluation ladder; re-screen dropped improvements on the correct (region-emb) artifact. |
| 2026-06-02 | `embedding_eval` | **CLOSED (Part-1 substrate)** — champion **v14 = `check2hgi_design_k_resln_mae_l0_1`** (resln+mae cat ⊕ Delaunay reg, orthogonal stack); design_k RE-OPENED + re-validated at FL (overturned prior AL/AZ-only K≡J); leak-free multi-seed: reg +0.9-1.1pp over canonical (closes 54-78% of HGI gap, HGI keeps small edge), cat ≈ frozen-canon ≫ HGI. NO MTL benefit (v14 or routing) → regime is the wall (Part-2). See CANONICAL_VERSIONS §v14. |
| 2026-06-12 | `mtl_improvement` | **CLOSED** — the C25 class-weighting confound WAS the "MTL sacrifices reg" gap; champion **G (= canon v16, the train.py MTL default)** MATCHES the STL reg ceiling (matched Δ −0.09…−0.31) + BEATS the STL cat ceiling (+2.6…+4.1) at 4 states × 4 seeds; mechanism = gradient orthogonality (tested intrinsic); no balancer/optimizer helps; cat gain architecture-dominated (transfer +0.93 FL / −0.67 AL). Survived 3 audit passes + the X-series. Read `mtl_improvement/FINAL_SYNTHESIS.md`. CA/TX + the aligned-pairing pre-freeze gate → `closing_data`. |
| 2026-06-12 | `closing_data` | **SCAFFOLDED (not launched)** — the final study: cross-study re-eval → pre-freeze gates (G0.1 aligned-pairing) → recipe FREEZE → CA/TX majors once → final 6-state tables. Launch pending user sign-off on `closing_data/PLAN.md`. |

## How to append

One row per closure or major direction shift. Format:

```
| YYYY-MM-DD | <study-name> | <Outcome in one sentence, lead with verb (CLOSED / PROMOTED / FALSIFIED / LAUNCHED / RE-OPENED)>. |
```

**Outcomes only.** Save the why/how for the study's own `log.md`. If you can't summarise the outcome in one sentence, the outcome isn't crisp enough to be logged yet.
