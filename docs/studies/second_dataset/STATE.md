# second_dataset — STATE

**Status:** **Phase E (ETL) COMPLETE — 2 cities** (2026-06-15) · Phase V BLOCKED (CUDA + frozen champion) · **Machine:** Mac (ETL) + CUDA box (Phase V) · **Created:** 2026-06-14
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Hand-off:** [`PHASE_E_REPORT.md`](PHASE_E_REPORT.md) · **Stats:** [`STATS_T1.md`](STATS_T1.md) · **Category map:** [`category_map.md`](category_map.md) · **Dry run:** [`DRY_RUN_RESULTS.md`](DRY_RUN_RESULTS.md) · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## ✅ Mac dry-run (pipeline shakeout, 2026-06-15) — champion behaviour transfers
Directional (NOT paper numbers; ResLN-80ep substrate, 1 seed × 5 folds × 50 ep — paper run = frozen substrate + CUDA, Phase V):
- **NYC**: MTL cat **+9.8 pp** over STL ceiling; MTL reg **−0.5 pp** (matches ceiling) + **+5.2 pp** over Markov-1 floor.
- **Istanbul**: MTL cat **+9.0 pp**; MTL reg **+1.0 pp** (matches ceiling) + **+17.0 pp** over Markov-1 floor.
- Both champion-G behaviours (cat gain + reg parity-above-floor) replicate on a different LBSN source, US + non-US. Pipeline validated end-to-end; bit-parity + per-fold priors confirmed. Recipe lesson: never mix `--canon v16` with manual `--reg-head` (freeze_alpha leak zeroed the log_T prior → use `--canon none` + explicit recipe). Detail: [`DRY_RUN_RESULTS.md`](DRY_RUN_RESULTS.md).

## Corpora (both built)
- **`istanbul` — PRIMARY** (non-US → stronger external validity; Gowalla & NYC are both US). 544k ck / 23.7k users. **TWO region defs built; mahalle = PRIMARY** (advisor 2026-06-15): mahalle 520 real-admin at top-level `output/check2hgi/istanbul/` (region_mode=admin; OSM `admin_level=8`, no clean gov source) + H3 res-9 2,585 SECONDARY at `…/istanbul/h3/`.
- **`nyc` — secondary** (US source-diversity). Regions = TIGER tracts (1,912). 272k ck / 6.9k users.

## Input-window comparison (within-user w9; full table in STATS_T1.md)
AL 12,709 · AZ 26,396 · **NYC 30,155** · FL 159,175 · **Istanbul 58,075 (mahalle, primary) / 60,091 (H3)** · CA 358,302 · TX 460,976. → Istanbul ≈ between AZ and FL; NYC ≈ AZ (both small-to-mid → H3-alt recipe).

## ⚠ Key findings (detail in PHASE_E_REPORT.md §F1–F3, verified vs source + paper)
- **F1 — shipped split is NOT temporal.** User-stratified RANDOM split over trails (seed 42, 7:1:2; users in all 3 splits by design; only 14% of train∩test users have test-after-train). The "free temporal-split bridge" rationale in `future_work.md §8` is **falsified**. The fix: build our OWN **chronological per-user split** from the per-check-in timestamps — the scoped **Phase E2** item (Mac, no CUDA, no freeze dep); doing it on Massive-STEPS is now the **recommended** route to A5 (modern non-US+US corpus), with a Gowalla-side chronological re-split as the fallback.
- **F2 — shipped split ⊥ repo window=9** (trails median=2). **F3 — Massive-STEPS native windowing is within-trail; it has NO category/region task** (both are our Gowalla-ported additions).
- **Decision (user): build BOTH** protocols; **lead with set (a)** within-user + user-grouped CV (controlled Gowalla-parity = the honest external-validity comparison); set (b) within-trail = footnote. A **3rd native-shape set** (within-trail, window≈5, overlapping) is recommended, not yet built.
- **Phase-V non-negotiables:** freeze each city's substrate to the *bit-identical Gowalla recipe* (transductivity must be matched); report Markov-1 floor + STL ceilings in-table & compare gap-to-ceiling (not absolute Acc@k); drop temporal-bridge language.

## Queue
| Phase | Item | State |
|---|---|---|
| E | acquire + parse (both cities) | ✅ done — Nyc.parquet (272k), Istanbul.parquet (544k) |
| E | FSQ-v1→7-root category map (versioned) | ✅ done — shared map; NYC 585/585, Istanbul 580/580 covered |
| E | regions (NYC TIGER tracts / Istanbul H3 res-9) + cardinality | ✅ done — NYC 1,912 tracts; Istanbul 2,585 H3 cells |
| E | folds + both protocols + per-fold priors | ✅ done — (a) NYC 30,155 / IST 58,075 (mahalle primary; 60,091 H3) seq + 25 priors each {0,1,7,100,42}; (b) NYC 9,509 / IST 6,419 (mahalle) + shipped-train prior |
| E | T1-style stats | ✅ done — `STATS_T1.md` |
| E | 3rd native-shape set (within-trail w5 overlapping) | ⬜ recommended, not built (offered to user) |
| **E2** | **chronological per-user re-split (timestamp-ordered 80/10/10-ish) + matching per-fold train-only priors** — Mac, parallel, **no CUDA / no freeze dep**; this is the **temporal bridge (roadmap A5)**, corrective to F1 | ⬜ **not built** — scoped (the shipped split is RANDOM, not temporal; build our own from timestamps) |
| V | substrate embeddings (frozen recipe) → next/category parquets | ⛔ blocked on freeze + CUDA |
| V | champion G + STL ceilings + Markov floor, 4 seeds (within-user CV = Gowalla-parity) | ⛔ blocked on freeze + CUDA |
| V | **A5 temporal bridge**: re-run headline cells on the Phase E2 chronological split | ⛔ blocked on freeze + CUDA (split built in E2) |

## Conventions
- Mac = ETL only; no heavy CUDA training. Phase V waits for a CUDA box + the frozen champion/substrate.
- Both cities are small-to-mid by region count → expect the **H3-alt** recipe in Phase V.
- Scope = validation phase, headline cells — NOT the full closing_data matrix.

## Decisions log
- 2026-06-14 — scaffolded from `future_work.md §8`. Massive-STEPS NYC recommended pending user confirm.
- 2026-06-15 — NYC Phase E run. Findings F1/F2 surfaced. User: **build BOTH** protocols.
- 2026-06-15 — independent reviews (ETL-correctness + user-sequence advisor) vs source+paper: confirmed F1–F3; tz bug fixed (timestamps are LOCAL); lead with set (a).
- 2026-06-15 — **user steer: add Istanbul as PRIMARY** (non-US; Gowalla+NYC both US) at **H3 res 9**. Built + verified, mirroring NYC. NYC kept as secondary.
- 2026-06-15 — Phase E COMPLETE (both cities). NO Gowalla data overwritten (verified). Scripts city-generic (`scripts/second_dataset/`).
- 2026-06-15 — TIGER-for-Istanbul search: no clean *gov* mahalle polygon (TÜİK/TKGM/HGM gate/fragment; İBB request closed). Best real-admin = OSM `admin_level=8` mahalle (~972; 520 populated). **User: build BOTH region defs** for Istanbul → mahalle (520, real-admin) + H3 res-9 (2,585), both leak-free + verified.
- 2026-06-15 — input-window comparison table built (AL/AZ/FL/CA/TX vs NYC/Istanbul) — see STATS_T1.md.
- 2026-06-15 — **methodology advisor: mahalle = PRIMARY Istanbul region**, H3 = secondary robustness. Rationale: external validity = transferring the SAME task; real-admin (mahalle) keeps task identity vs Gowalla/NYC tracts; gap-to-ceiling absorbs granularity (Gowalla already 1.1k→8.5k); H3 changing task-kind is the unprotected/attackable axis. Rebuilt: mahalle → top-level (region_mode=admin), H3 → `h3/` (reproducible via `build_region_variant --variant h3 --source h3`). Both verified leak-free.
