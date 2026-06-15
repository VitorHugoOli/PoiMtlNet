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
| 2026-06-12 | `mtl_improvement` | **CLOSED** — the C25 class-weighting confound WAS the "MTL sacrifices reg" gap; champion **G (= canon v16, the train.py MTL default)** MATCHES the STL reg ceiling (matched Δ −0.09…−0.31) + BEATS the STL cat ceiling (+2.6…+4.1) at 4 states × 4 seeds; mechanism = gradient orthogonality (tested intrinsic); no balancer/optimizer helps; cat gain architecture-dominated (transfer +0.93 FL / −0.67 AL). Survived 3 audit passes + the X-series. Read `archive/mtl_improvement/FINAL_SYNTHESIS.md`. CA/TX + the aligned-pairing pre-freeze gate → `closing_data`. |
| 2026-06-12 | `closing_data` | **SCAFFOLDED (not launched)** — the final study: cross-study re-eval → pre-freeze gates (G0.1 aligned-pairing) → recipe FREEZE → CA/TX majors once → final 6-state tables. Launch pending user sign-off on `closing_data/PLAN.md`. |
| 2026-06-12 | `closing_data` | **RE-SCOPED (pre-launch)** — now the experimental engine for the NEW paper: STL baselines RE-RUN + champion + the BRACIS suite (T1–T5 / §0.1–0.6 via a RUN_MATRIX inventory), ALL states × 4 seeds × 5 folds under the frozen recipe; substrate identity = a freeze decision. Story chosen in a follow-up effort. |
| 2026-06-14 | pre-freeze program | **SCAFFOLDED** — `PRE_FREEZE_PROGRAM.md` orchestrator + 3 upstream studies feeding `closing_data` G0.2: `mtl_frontier` (A40, R1–R3 exploration, FIRST), `pre_freeze_gates` (A40, A2/A4/overlapping-windows), `second_dataset` (Mac, Massive-STEPS NYC ETL ∥ + validation bridge). Hierarchy: explore → gates → FREEZE → regen → validate. Launch pending user sign-off. |
| 2026-06-14 | `mtl_frontier` | **SCAFFOLDED (not launched)** — output-level/asymmetric-sharing exploration (R1 log_C co-location prior, R2 STEM-AFTB gating, R3 cross-task distillation); optimizer aisle declared closed; ≥0.3 pp lever → v17 pre-freeze gate. A40. |
| 2026-06-14 | `mtl_frontier` | **R10 ADDED (user)** — Memory-Caching / GRM gating at the LAYER level (arXiv:2602.24281, no code): GRM/SSC gated read between dual towers (primary) + GRM-soup fusion across Check2HGI hierarchy levels (speculative). "On the layers, not the transformers"; second-wave, run after R2. |
| 2026-06-14 | `pre_freeze_gates` | **SCAFFOLDED (not launched)** — A2 feature-concat control (interpretation), A4 transductivity bound (disclosure), overlapping-windows adopt/keep (base change; effect already validated AL). Must resolve before `closing_data` freeze. A40. |
| 2026-06-14 | `second_dataset` | **SCAFFOLDED (not launched)** — Massive-STEPS NYC (recommended; user-confirm): Mac ETL in parallel (cat→7-root map, coords→tracts, shipped temporal split), validation phase = temporal-split bridge. Off the freeze critical path. |
| 2026-06-14 | `baseline_gap` | **SCAFFOLDED (not launched)** — closes the audit gap: owns the net-new external baselines (B1 CTLE, B2 POI2Vec/skip-gram, B3 HMT-GRN-style MTL, B4 cascade, B5 Flashback/DeepMove) from `baseline_gap_analysis.md`. Triage → RUN_MATRIX rows/columns decision pre-freeze (P1b); runs fold into P3. A40 + Mac. |
| 2026-06-15 | `mtl_frontier` R1 | **NULL** — log_C co-location ESMM-KD prior (`prior(reg)=Σ_c P(reg\|c)·P̂(c)`, on top of log_T-KD): AL multi-seed Δreg **+0.207±0.196** (gate ≥0.3 FAIL; Wilcoxon p=0.008, 15/20 fold-seed pairs +, no cat regression), FL seed0 +0.171/cat−0.27; weight non-monotonic (peaks W=0.2, craters at 0.6). Real-but-small increment over log_T-KD (weak 7-class aux + spatial overlap). Not v17. Code behind `--log-c-kd-weight` (default off). → R2. |
| 2026-06-15 | `mtl_frontier` R2 | **NULL** (not v17) — STEM-AFTB per-layer/direction stop-grad on G's cross-attn: reg **clean multi-seed null** all states; cat lift is **AL-only & decays with scale** (AL +0.636 / AZ +0.173 / GE +0.158 / **FL −0.026**) — user-approved multi-state confirm shows no generalization (inverse-G′). Citable STEM-AFTB dose-response: cross-task gradient is small-state harmful noise; sharing topology doesn't move the reg gap (cos≈0). Directional `aftb_spec` code, G default unchanged. → R3. |
| 2026-06-15 | `mtl_frontier` R3 | **NULL** — CrossDistil (live-teacher generalization of log_T-KD): warm-up + error-correction on fwd cat→reg + new reverse reg→cat arm. Fwd refinements don't rescue R1 (AL cat −0.18); reverse AL cat +0.45 seed0 → multi-seed **+0.100±0.282** (p=0.31, seed1 −0.34); FL null. **log_T-KD saturates the output-prior family.** Code behind `--cat-kd-weight`/`--log-c-kd-{warmup-epochs,ec-lambda}`, G default unchanged. First wave R1/R2/R3 = 3 nulls, same regime. → R10. |

## How to append

One row per closure or major direction shift. Format:

```
| YYYY-MM-DD | <study-name> | <Outcome in one sentence, lead with verb (CLOSED / PROMOTED / FALSIFIED / LAUNCHED / RE-OPENED)>. |
```

**Outcomes only.** Save the why/how for the study's own `log.md`. If you can't summarise the outcome in one sentence, the outcome isn't crisp enough to be logged yet.
