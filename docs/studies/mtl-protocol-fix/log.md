# mtl-protocol-fix — Progress Log

Append-only chronological log. Every agent working on this track adds entries here.
Dates are absolute (e.g. `2026-05-20`), never relative ("today", "yesterday").

Sections at the bottom of each entry:
- **Decision** if you changed direction.
- **Blocker** if you got stuck (and what unblocked you, in a later entry).
- **Next** what the next agent should pick up.

---

## 2026-05-20 — Study launched

**Phase**: Design complete; no experiments run yet.

**What happened**

- Folder `docs/studies/mtl-protocol-fix/` created alongside `canonical_improvement/` and `merge_design/`.
- `AGENT_PROMPT.md`, `considerations.md`, `INDEX.html` written. The study is direct successor to the closed `canonical_improvement` (Tier 1-6, 26 mechanism families, ceiling ±0.8 pp on substrate axis) and supersedes the F1/F2/F3 split of `mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`.
- 5 future-work memos created under `docs/future_works/` for items deliberately scoped OUT of this study (paper-canon re-evaluation, substrate-adaptive MTL balancing, MTL architecture revisit, head/window/batch audit, reg-head architecture sweep).
- Doc cross-references updated: `docs/CHANGELOG.md`, `docs/README.md`, `docs/CONCERNS.md` C21, `docs/NORTH_STAR.md`, `docs/AGENT_CONTEXT.md`, `docs/future_works/README.md`, `docs/studies/canonical_improvement/log.md`, `docs/studies/mtl-exploration/README.md` — all now point to this study as the active protocol-axis track.
- User-directed scope clarified (2026-05-20): IN = Rank 1 (F1 selector fix) + Rank 3 (Tier 5/6 candidate re-eval under F1) + three-frontier MTL evaluation protocol. OUT = Rank 2 / Rank 4 / Rank 6 / Rank 7 / §4.3 — all documented in `docs/future_works/`.

**Decision** — Branch `mtl-protocol-fix` will be the dedicated worktree for execution. Do not contaminate `check2hgi-canonical-improve` or any other branch.

**Decision** — Three-frontier evaluation (best joint + best disjoint + STL ceiling) is the study's primary methodological deliverable; replaces the implicit single-selector reporting that obscured C21.

**Decision** — Phase-1 single-seed n=5 first; multi-seed only in Phase 3 if residual gap characterisation requires statistical power. Full §0.1 n=20 multi-seed re-evaluation is deferred to `docs/future_works/paper_canon_reevaluation.md` (sequenced after `mtl_architecture_revisit.md`).

**Next**

1. Implementing agent must read in order: this log, `INDEX.html` (top-down), `considerations.md`, `docs/CONCERNS.md` C21, `docs/studies/canonical_improvement/log.md` 2026-05-19 final entry, `docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`.
2. Create the dedicated worktree before running anything.
3. Start with Phase 0 (pre-flight gate: reproduce `T6_4_dual_selector_final.json` shipping arm under F1-fixed code path).
4. Use `TaskCreate` to break down every Phase into validate → code-change → unit-test → re-evaluate → analyze sub-tasks.
5. After each Phase completes, call advisor with the Phase's results before proceeding.
6. After the full study completes, run a final advisor pass on the whole INDEX.html + log + the three-frontier table before declaring done.

---

## 2026-05-20 — Phase 0–2.P1 executed; F2 revised after STL-on-shipping retrain

**Phase**: Phase 0 (PASS) → Phase 1 (v1→v4 iterations) → Phase 2 P1 (STL on shipping, DONE).

**What happened**

- **Phase 0** (pre-flight reproduction gate): PASS. FL @ B9 per-task disjoint reg = 76.59 ± 0.63 matched matched-protocol gate 76.12 ± 0.33 within σ. Verdict at `docs/results/mtl_protocol_fix/phase0_2_three_frontier_verdict.md`.
- **Phase 1 v1** (2026-05-20 01:50–03:09 UTC): launched 5-state MTL with `--mtl-joint-selector geom_simple`. Discovered FL substrate was T6.x drift (May 19 15:37 mtime ≈ T6.2/T6.3 sweep timing); CA/TX used pre-T3.2 canonical (May 14 mtime). Numbers misleading.
- **Phase 1.5 substrate regen** (03:16–05:27 UTC): regenerated v3c+T3.2 ResLN shipping embeddings at all 5 states via `regen_emb_t3.py --encoder resln --weight-decay 5e-2 --epoch 500 --seed 42`. Confirmed shipping = canonical + v3c + T3.2 ResLN; no Tier 5/6 winners missing (independent audit confirmed). Disk hygiene: ~46 GB stale checkpoints deleted; FL/CA/TX shapefiles re-downloaded from US Census after accidental purge.
- **Phase 1 v2** (03:52–06:05 UTC): retrained MTL at all 5 on consistent shipping. Captured the C21 bug fully: FL bug Δ +24.18 → +11.42 (using corrected `selector_bug = geom_simple − b9` labeling); CA +8.30 → +6.93; TX +10.43 → +8.90; AL/AZ ~+1 pp.
- **Phase 1 v3** (13:08–13:13 UTC, post advisor review): fixed protocol error — AL/AZ re-run with **H3-alt** (small-state recipe: scheduler=constant, no alpha-no-WD, no alternating-step) instead of B9. Cat F1 lifts: AL +3.71 pp (42.05 → 45.76), AZ +1.89 pp (46.98 → 48.87). Reg essentially unchanged.
- **Phase 2 P1** (13:18–14:46 UTC): retrained STL on shipping at all 5 states × 2 heads (cat via `train.py --task next --model next_gru`, reg via `p1_region_head_ablation.py --heads next_getnext_hard`). **STL substrate lift asymmetric**: FL +9.69 pp (69.22 → 78.91), AZ +3.36, AL +2.95, CA +1.27, TX +0.92.
- **F2 FALSIFIED**: the v1-v3 "MTL@FL beats STL by +7.25 pp" headline was a STL substrate-cohort artifact. On consistent shipping substrate, MTL trails STL at ALL 5 states (FL: −2.44, CA: −6.58, TX: −8.98, AL: −11.28, AZ: −12.27).
- **Mechanism diagnosis (read-only investigation 14:50 UTC)**: STL `best_epoch` per state — FL: ~24, CA: ~11, TX: ~10; MTL @ disjoint best_epoch — FL: 4.4, CA: 2.0, TX: 1.0. MTL reg val peaks early (ep 1-4) then degrades from cat-task interference (negative transfer). At FL with abundant data (~34 train_seq/region), substrate is learnable in ≤4 epochs and MTL near-recovers STL ceiling. At CA/TX/AL/AZ (~7-12 train_seq/region), substrate would need ep 10-24 of reg-only learning that MTL can't deliver before negative transfer dominates.

**Decision** — Use revised F2 framing: "MTL approaches STL on reg only when substrate is learnable within MTL's effective horizon (limited by negative transfer onset). Gap shrinks with n_regions × n_train_seq density."

**Findings (Phase 1 v4 final)**:
- C21 selector bug (geom − b9): FL +11.42, TX +8.90, CA +6.93, AL +0.95, AZ +1.16. Scale-conditional, threshold ~2-3k regions.
- F1 fix recovers most substrate capacity at large states: capacity gap (disjoint − geom) 1.4-3.6 pp at FL/CA/TX.
- Residual MTL-vs-STL on shipping at substrate capacity: FL −2.44, CA −6.58, TX −8.98, AL −11.28, AZ −12.27.
- b9 selector at FL is BIMODAL per fold (3/5 land in ~70% mode, 2/5 in ~49% crashed mode); CA/TX MTL reg peaks at ep 1-2 then crashes monomodally (no bimodality at CA/TX).
- **Verdict**: F1, F3, F4 PASS as paper-grade findings. F2 REVISED (no MTL@FL "win", but gap-shrinks-with-density story is cleaner). F5 holds with corrected residual numbers.

**Next**:
- Phase 2 P2 — multi-seed FL n=20 (seeds 0,1,7,100 already partially run at canonical; need shipping multi-seed). Stabilizes F4 bimodality estimate.
- Phase 2 P3 — Tier 5/6 §Discussion candidates (T5.2b, T5.3) re-eval under F1 selector.
- Phase 2 P4 (NEW, motivated by Phase 1 v4 mechanism diagnosis) — test horizon hypothesis at CA/TX with `--freeze-cat-after-epoch 1` for first 10 epochs.

---

## 2026-05-20 — Phase 2 P5 + P6 — STALE log_T bug discovery + study closure

**Phase**: Phase 2 P5 (stale log_T audit), P6 (CA/TX preliminary), final synthesis.

**What happened — the critical discovery**

User flagged that 4-seed STL FL multi-seed gave 70.8 ± 0.1 while my seed=42 STL FL gave 78.91. 8 pp seed-variance on a supervised problem is too large; audit found smoking gun:

- **FL `region_transition_log_seed42_fold*.pt` mtime was 2026-05-06** — TWO WEEKS old, NEVER rebuilt during this session. Hash differs from a fresh rebuild.
- AL/AZ seed=42 log_T mtime 2026-05-15 (stale but next_region.parquet didn't change at AL/AZ — verified bit-identical, so no impact).
- CA/TX seed=42 log_T mtime 2026-05-20 01:39 (built at session start — clean).

Empirical confirmation: re-ran FL seed=42 STL + MTL with freshly-rebuilt log_T:
- STL Acc@10: STALE 78.91 ± 0.27 → FRESH **70.89 ± 0.52** (Δ = −8.02 pp)
- MTL @ disjoint: STALE 76.47 → FRESH ~64 (Δ ≈ −12 pp at substrate capacity)
- MTL @ geom_simple: STALE 72.88 ± 1.49 → FRESH 61.14 ± 0.95 (Δ = −11.74 pp)
- MTL @ b9: STALE 61.47 ± 11.48 → FRESH 53.73 ± 9.22 (Δ = −7.74 pp)

FRESH FL seed=42 values match multi-seed {0,1,7,100} mean within σ. **The seed=42 outperformance at FL was 100% stale log_T artifact.** No development-seed bias at FL.

**Independent advisor audit (verdict: BOUNDED severity)**

- Tier 5 + small-state Tier 6: CLEAN (sandboxed sweeps rebuild log_T per variant).
- **Tier 6 FL-MTL sweeps (T6.1, T6.2, T6.4): stale log_T used** — but BOTH shipping baseline AND every variant used SAME stale log_T → **relative falsifications HOLD**; absolute Acc@10 biased by unknown sign-and-magnitude.
- Region-idx layout STABLE across regens (poi_to_region map cached via checkin_graph.pt force_preprocess=False); only train-split entries of log_T differ between old and new.
- No Tier 5/6 winner missed: closest "almost-winner" T6.2 a2.0-0.3 was within +0.18 pp on stale log_T; cannot flip an 8 pp gap.

**Phase 2 P6 — CA/TX preliminary**

Single-seed=42 CA/TX disjoint vs §0.1 v11 multi-seed n=20:
- CA: 50.61 (seed=42) vs 47.35 ± 0.11 (§0.1) — **+3.26 pp dev-seed overshoot**
- TX: 50.83 (seed=42) vs 42.84 ± 0.14 (§0.1) — **+7.99 pp dev-seed overshoot**

Verified CA log_T fresh (hash-identical on rebuild). next_region.parquet at CA also content-identical to a fresh rebuild (mtime drift only, not stale content). **Therefore CA/TX overshoot is genuine development-seed bias**, not a data-leak issue. Multi-seed at CA + TX deferred to future work; §0.1 v11 already establishes the multi-seed numbers.

**Phase 2 closure decisions**

User chose pragmatically to:
1. NOT run full multi-seed CA + TX (5-6 GPU hours; §0.1 v11 already published)
2. Run CA + TX seed=0 preliminary as anchor data points
3. SKIP Phase 2 Tier 5/6 §Discussion candidate re-eval (per-epoch CSVs were lost in disk cleanup; would need 5-6 hours to reconstruct)
4. Move directly to Phase 3 residual-gap characterisation → brief for next-tier study

**Findings (Phase 1 v5 final, post-stale-log_T fix)**

Multi-seed FL (4 seeds, fresh log_T) — these are the trustworthy numbers:
- STL Acc@10 = **70.92 ± 0.10** ✓ matches §0.1 v11 (70.62 ± 0.09)
- MTL @ disjoint = **63.91 ± 0.16** ✓ matches §0.1 v11 (63.27 ± 0.10)
- MTL @ geom_simple = **61.54 ± 4.54**
- MTL @ b9 = **55.92 ± 3.40**
- **Selector bug (geom_simple − b9) = +5.62 pp**: real, deployable-axis gain
- **Capacity gap (disjoint − geom_simple) = +2.37 pp**: F1 fix recovers ~95% of substrate capacity at FL

Findings reaffirmed:
- **F1** (scale-conditional selector bug): real at FL on multi-seed (+5.6 pp); needs CA/TX multi-seed verification.
- **F2 (REVISED)**: MTL trails STL on reg at all 5 states. FL gap = STL 70.92 − MTL@disjoint 63.91 = **−7.01 pp** (matches §0.1 v11's −7.34). No "MTL@FL beats STL" — that was the stale-log_T outlier.
- **F3** (F1 fix recovers most capacity): UPHELD, ~95% recovery at FL.
- **F4** (bimodal b9 at FL): seed-dependent; tight σ at seed=1/100, wide σ=~10 at seed=0/7/42.
- **F5 (REVISED)**: Phase 2 P4 frozen-cat horizon test FALSIFIES negative-transfer mechanism. MTL reg peaks at ep 2 even with cat fully frozen → the gap is **MTL backbone architecture**, NOT cat-task interference. Direct region-emb access (like STL) is needed.

**Decisions (study closure)**

- Phase 2 Tier 5/6 §Discussion candidate re-eval is DEFERRED to future work (per-epoch CSVs lost; advisor verdict on impact: bounded, no missed winners).
- Multi-seed CA + TX is DEFERRED to future work (§0.1 v11 already has n=20 multi-seed; preliminary seed=0 is enough anchor).
- Next-tier study (MTL architecture / loss balancing) takes over per the F5 mechanism finding.

**Next**

1. Final verdict promotion: `docs/results/mtl_protocol_fix/phase1_verdict.md` → v6 final with stale-log_T audit + Phase 2 closure.
2. Cross-reference notes added to:
   - `docs/studies/canonical_improvement/log.md` — Tier 6 FL-MTL stale log_T caveat
   - `docs/CONCERNS.md` — new C-entry for stale log_T silent corruption
   - `CLAUDE.md` — new preflight guard documented (this session)
3. Hand off to next study (substrate_adaptive_mtl_balancing or mtl_architecture_revisit) per the F5 mechanism finding.

**Blocker** (none — study CLOSED with caveats documented)

---

## (template — copy and date for next entry)

## YYYY-MM-DD — <Short title>

**Phase**: <Phase 0 / 1 / 2 / 3 / Final synthesis>

**What happened**

- <bullet>
- <bullet>

**Decision** (only if changed direction):
- <what changed and why>

**Blocker** (only if stuck):
- <what's blocked>
- <what you tried>

**Findings** (only if results landed):
- per state, three-frontier numbers (best joint / best disjoint / STL)
- statistical significance / Wilcoxon p / fold-by-fold deltas
- **Verdict**: <"fix validated" | "partial" | "regression" | "inconclusive at n=5">
- Updated INDEX.html `#PhaseN-*` Results placeholder: yes/no

**Next**:
- <experiment ID> next, or <decision needed>

---
