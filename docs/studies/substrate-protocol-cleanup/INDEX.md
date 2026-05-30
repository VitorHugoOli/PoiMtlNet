# substrate-protocol-cleanup — Study Index

**Status:** Tier D CLEAN · Tier A1 PROMOTED (log_T-KD) · Tier B all-NULL · **Tier C: C1 §Discussion-FOOTNOTE (re-scored 2026-05-29 after a reg-modality scoring bug; AZ +2.54 pp p=0.03 passes, AL fails on one genuine degenerate fold), C2 closes §4.4, C3 closes P4**.
**Drafted:** 2026-05-28
**Predecessor:** [`docs/studies/mtl-protocol-fix/`](../mtl-protocol-fix/) (CLOSED 2026-05-24, v6 final)
**Parallel (independent branch `mtl-improve`):** [`docs/studies/mtl_improvement/`](../mtl_improvement/)
**Owns:** §4.1, §4.4 (partial), §4.5 multi-seed, §4.7 (cross-study with merge_design), §B window/mask audit
**Does NOT own:** anything on the MTL backbone / loss / batch / LR / α / head / multi-seed champion ship axes — those belong to `mtl_improvement`.

---

## Why this study exists

The closure of `mtl-protocol-fix` left 7 items genuinely deferred (see [`../mtl-protocol-fix/DEFERRED_WORK.md`](../mtl-protocol-fix/DEFERRED_WORK.md)). They split cleanly into two groups:

1. **Architectural** (residual −7 to −12 pp MTL-vs-STL reg gap is mechanism-identified as backbone-related per P4 frozen-cat test). → owned by `mtl_improvement` on branch `mtl-improve`.
2. **Substrate + protocol** (substrate variants, checkpoint selection, supervision signals, no-GPU audits). → owned by **this study**.

Splitting prevents merge collisions on `src/training/runners/mtl_cv.py` and `src/configs/experiment.py`.

## Scope discipline

- **States:** AL + AZ for main sweeps. FL/CA/TX only as 1-fold pilots to confirm sign-and-magnitude of a hypothesis already validated at AL/AZ.
- **Selector:** F1 (`joint_geom_simple`) is the deployable axis everywhere. The three-frontier table (best joint / best disjoint / STL ceiling) is mandatory for every experiment.
- **log_T:** per-fold seed-tagged (post-2026-05-15). C22 stale-log_T preflight gate before every run.
- **Seeds:** dev seed=42 for single-seed comparisons; {0, 1, 7, 100} for any paper-grade promotion (C23).

---

## Tier A — Multi-seed promotion (paper-grade, small states only)

### A1 — log_T-KD §4.5 multi-seed n=20 at AL/AZ

- **Source verdict:** Phase 3 Rank 1 PROMOTED at single-seed n=5 ([`phase3_rank1_findings.md`](../../results/mtl_protocol_fix/phase3_rank1_findings.md)).
- **Action:** Re-run the log_T-KD sweep at AL/AZ with 4 seeds × 5 folds = 20 cells, W=0.2 (the strongest single-seed cell).
- **Tooling:** `scripts/mtl_protocol_fix/run_log_t_kd_sweep.sh` + `summarize_log_t_kd.py` (re-use; add `--seeds 0,1,7,100`).
- **Cost:** ~3-4 GPU-h per state × 2 states = ~6-8 GPU-h.
- **Decision gate:** if Wilcoxon n=20 paired p ≤ 0.05 at AL AND AZ on disjoint reg → promote to paper §0.4 / §0.6; if only one passes → §Discussion footnote; if neither → file as null-at-n=20.
- **Pre-flight:** C22 stale-log_T check at each of the 4 seeds × 2 states.

---

## Tier B — Substrate cross-study under F1 (MTL re-eval; small states only)

### Common framing

Designs B and J have **STL** numbers published in `docs/studies/merge_design/` showing dominance vs canonical c2hgi at AL/AZ. The MTL evaluation under the F1 selector was never run. The hypothesis: if these substrates already improve STL reg at AL/AZ, MTL+F1 should preserve that lift while keeping cat non-inferior.

**Important framing decision:** Tier B is **substrate-axis re-evaluation**, NOT a substrate competition with canonical Check2HGI for shipping. Even a positive result here does NOT supersede `mtl_improvement`'s architectural axis. A winner in Tier B becomes a "free upgrade" to whatever the architectural champion is.

### B1 — Design B (POI2Vec at pool boundary) MTL under F1 at AL/AZ

- **Substrate build:** `scripts/probe/build_design_b_poi_pool.py` (exists, used in merge_design STL runs).
- **MTL train:** canonical B9-ish recipe + `--engine check2hgi_design_b` (or equivalent build flag); `--mtl-joint-selector geom_simple`.
- **Cost:** 4 GPU-h at AL + 4 GPU-h at AZ = 8 GPU-h.
- **Pre-flight:** confirm Design B substrate parquet exists at AL/AZ post-leak-fix.
- **Decision gate:** Wilcoxon disjoint reg p ≤ 0.05 vs canonical (5-fold, seed=42) AND Δcat ≥ −0.5 pp → promote to multi-seed (move to A-tier); else null/falsified.

### B2 — Design J (H + anchor regulariser λ=0.1) MTL under F1 at AL/AZ

- **Substrate build:** `scripts/probe/build_design_j_anchor.py` (exists).
- **MTL train + selector:** same as B1, swap engine flag.
- **Cost:** 8 GPU-h.
- **Decision gate:** identical to B1.

### B3 — Lever 4 (POI2Vec at p2r boundary, additive)

- **Mechanism:** add region-prior term to `L_p2r` (see [`LEVER_4_POI2VEC_P2R.md`](../merge_design/LEVER_4_POI2VEC_P2R.md)). Independent additive lever.
- **Pre-flight:** must be applied on top of a substrate already validated. Apply first on canonical c2hgi (control), then on whichever of B1/B2 promoted (if any).
- **Cost:** 4 GPU-h on canonical + 4 GPU-h on B/J winner = up to 8 GPU-h.
- **Decision gate:** lift of +0.3 pp disjoint reg with no cat regression → promote to multi-seed.
- **Order:** run AFTER B1 and B2 verdicts are in.

### B4 — Lever 5 (KL distill on top-k neighbour softmax, orphan rescue)

- **Why here:** `merge_design` is closed (STATE.md 2026-05-06); Lever 5 has no other live owner and `LEVER_5_DIST_DISTILL.md` estimates ~3 GPU-h at AL+AZ on MPS, which is cheaper than Tier C2. Absorbed here per the post-launch advisor pass (2026-05-28).
- **Mechanism:** add a distribution-level distillation loss to the c2hgi `L_p2r`, matching the discriminator's top-k softmax to a target softmax computed over POI2Vec / HGI neighbour ranking. Substrate-build-script change only; no MTL backbone change.
- **Pre-flight:** confirm `build_design_l_distkl.py` (or its successor name) exists; if not, draft mirroring `build_design_b_poi_pool.py`.
- **Cost:** ~3 GPU-h AL+AZ. Single-seed=42 first.
- **Decision gate:** Wilcoxon disjoint reg p ≤ 0.05 vs canonical AND Δcat ≥ −0.5 pp → multi-seed promote; else null.
- **Sequencing:** run alongside B1/B2 (independent substrate variant).
- **Architectural-axis independence:** Lever 5 modifies only the substrate-build script; it does not depend on the MTL backbone champion landing in `mtl_improvement`. Safe to run now.

### B-summary table (to fill in as runs land)

Filled 2026-05-28 (Tier B Wave 1 + B3). Numbers are seed=42, 5-fold, H3-alt, disjoint reg `top10_acc_indist` / geom_simple reg, Δ vs `canonical_baseline` cell at same state, RAW-value Wilcoxon (one-sided design > canonical). AL row / AZ row per cell.

| Substrate variant | MTL reg @ disjoint (AL / AZ) | MTL reg @ geom_simple (AL / AZ) | Δ vs canonical reg (AL / AZ) | Δ cat F1 (AL / AZ) | Wilcoxon p (AL / AZ) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| canonical c2hgi (control) | 50.82 / 41.33 | 48.56 / 39.60 | — | — | — | BASELINE |
| Design B (B1) | 50.44 / 41.35 | 48.68 / 39.11 | −0.38 / +0.03 | −2.17 / −2.41 | 0.91 / 0.44 | **NOT PROMOTED** (FALSIFIED AL / NULL AZ) |
| Design J (B2) | 50.60 / 41.30 | 47.34 / 39.43 | −0.22 / −0.02 | −2.05 / −2.66 | 0.78 / 0.69 | **NOT PROMOTED** (FALSIFIED both) |
| Lever 5 (B4) | 50.54 / 41.34 | 47.83 / 40.23 | −0.28 / +0.01 | −2.49 / −2.41 | 0.81 / 0.69 | **NOT PROMOTED** (FALSIFIED AL / NULL AZ) |
| canonical + Lever 4 (B3) | 50.58 / 41.25 | 48.43 / 39.58 | −0.24 / −0.08 | −2.68 / −2.54 | 0.78 / 0.84 | **NOT PROMOTED** (FALSIFIED both) |
| (winner) + Lever 4 (B3b) | n/a | n/a | n/a | n/a | n/a | SKIPPED (no Wave-1 winner) |

Wave 1 verdict doc: [`../../results/substrate_protocol_cleanup/tier_b/phase_b1b2b4_verdict.md`](../../results/substrate_protocol_cleanup/tier_b/phase_b1b2b4_verdict.md). All three designs fail BOTH the disjoint-reg significance gate (every p ≥ 0.44) AND the Δcat ≥ −0.5 pp non-inferiority gate (uniform ~−2.4 pp cat regression). No leak signature (reg flat, cat regresses — opposite of a label-shortcut leak). B3 runs canonical+Lever4 control only.

---

## Tier C — Protocol coherence

### C1 — §4.1 per-task 3-snapshot routing (variant A)

- **Mechanism:** during training, save 3 independent full MTL checkpoints — one at the epoch maximising val cat F1, one at the epoch maximising val reg Acc@10, one at the epoch maximising `joint_geom_simple`. At inference, route cat requests to the cat-best snapshot, reg requests to the reg-best snapshot.
- **NOT variant C:** variant C (DEFERRED_WORK as-written) — mixing backbone from epoch X with heads from epochs Y and Z — is **incoherent** (heads never trained against that backbone). We explicitly reject variant C.
- **Implementation:**
  - Extend `BestTracker` in `src/tracking/best_tracker.py` to maintain three separate best-state dicts.
  - Add `--save-task-best-snapshots` CLI flag to `scripts/train.py`.
  - Add `scripts/route_task_best.py` that loads 3 snapshots and scores per-task on the held-out fold.
- **Cost:** ~1-2 days code + unit test + 4 GPU-h scoring at AL/AZ (zero retraining once the flag is in for the next sweep; can also re-score Phase 3 Rank 1 W=0.2 runs if their per-epoch state was preserved — likely was not, so plan on a fresh single-seed=42 5-fold at AL/AZ).
- **Decision gate:** Δreg @ task-best vs joint-best ≥ +2 pp at AL AND AZ → multi-seed promote (move to A-tier); ~+1 pp at one state only → §Discussion footnote; null → archive.
- **Risk:** the routing overhead (loading two checkpoints at inference) is non-trivial for deploy; verdict must include a deploy-cost note.
- **VERDICT (re-scored 2026-05-29): §DISCUSSION FOOTNOTE — one-state pass.** ⚠ The 2026-05-28 ARCHIVE was based on a **reg-modality scoring bug** (`route_task_best.py` rebuilt val loaders on `task_b=checkin` while the run trained `region`; `ExperimentConfig` did not persist the modality). Fixed (persist `task_*_input_type`; scorer reads it; 5 new unit tests) and re-run with fresh snapshots. **Corrected:** on the region modality the +2 pp gate clears at **AZ (Δreg +2.54 pp, 5/5 folds positive, Wilcoxon p=0.03125)** but fails at **AL (−7.89, p=0.31)**. The AL failure is ONE **genuine degenerate reg-best snapshot** (fold3: reg-best Acc@10=**0.12 %** even on the correct region modality — saved at val reg Acc@1=0.2801 @ ep14 but does not generalise; the same fold's joint-best scores a healthy 48 %). This is a real `MultiTaskBestTracker` Acc@1-selector pathology — **NOT the modality bug** (the modality bug was a separate confound that previously depressed the healthy folds and the AZ picture; the advisor's "0.0 is purely modality" hypothesis is falsified for AL fold3). AZ has **no** degenerate fold (the prior "fold2 collapse" was a modality artefact). One-state pass → footnote, not full PROMOTE (needs +2 pp at both) and not ARCHIVE (AZ is a real, significant gain). Healthy 4 AL folds avg +2.14 pp. Cat-best routing near-null (+0.87 AL p=0.06 / +0.12 AZ ns). Deploy cost (3× storage + 2-model load) + selector brittleness. **Conditional follow-up:** swap reg-best selector to Acc@10 + add a degenerate-snapshot guard, then multi-seed AL/AZ before any §0.x promotion. See `phase_c_verdict.md` §C1.

### C2 — §4.4 freeze-reg-after-peak (asymmetric reg-stop pilot)

- **Mechanism:** P4 frozen-cat falsified cat-vs-reg interference. The mirror variant — let reg train to its ep ≈ 2-4 peak, freeze reg head + its specific encoder, then continue training cat-only — is the one curriculum form NOT falsified.
- **Implementation:**
  - Add `--reg-freeze-at-epoch N` flag to `scripts/train.py`. When `epoch == N`, freeze `task_b_encoder.*` + `next_head.*` + zero the reg loss.
  - No new selector logic; the existing geom_simple still applies.
- **Cost:** ~4 hours code + 2 GPU-h at AL + 2 GPU-h at AZ = 4 GPU-h experimental.
- **Sweep:** `N ∈ {2, 4, 6}` × {AL, AZ} × seed=42 × 5 folds.
- **Decision gate:** does Δcat improve at any N without Δreg regression > σ_fold? If yes → multi-seed promote. If no → archive as falsifying the last curriculum variant; closes §4.4 entirely.
- **Code-adjacency to `mtl_improvement` T3.5 cat-only warmup:** both touch freezing logic in `runners/mtl_cv.py`. Coordinate via the rebase cadence in §"Branch-coordination protocol" below.
- **VERDICT (2026-05-28): ARCHIVE — CLOSES §4.4 ENTIRELY.** No N improves cat without a reg regression > σ_fold at either state (seed=42 5-fold). Where cat lifts most (AL N=2/4: +0.37/+0.46 pp, p=0.16, never significant) reg collapses −7.69 / −4.18 pp (≫ σ_fold 3.21); where reg is preserved (N=6: AL −1.05 / AZ −0.07) cat is null (≤ +0.06 pp). The asymmetric freeze-reg-after-peak curriculum — the one form P4 had not falsified — is now falsified. See `phase_c_verdict.md` §C2.

### C3 — P4 residual hole: cross-attention K/V capacity-stealing test (near-zero compute)

- **Why this exists:** P4 frozen-cat (`phase1_phase2_verdict_v6_final.md` §P4) froze the cat encoder *parameters* at ep 0+ and zeroed `cat_weight`. But under cross-attention MTL (CH20, C20), the cat encoder *output* still flows through K/V into the shared backbone — even with frozen weights, the cat-encoder activations participate in shared-backbone learning. P4 therefore did NOT fully isolate "cat-pathway contributions to backbone capacity."
- **Mechanism:** zero the cat K/V projection (i.e., set the cross-attention's contribution from the cat encoder to zero) during forward, leaving reg-side cross-attention intact. If MTL reg still peaks at ep 2 and degrades, P4's architectural conclusion is fully closed. If MTL reg improves under K/V-zeroed cat path, the residual mechanism is **capacity-stealing through the cross-attention K/V channel**, not the shared-backbone parameters.
- **Implementation:** ~1 day code (add `--zero-cat-kv` flag to `MtlnetCrossattn.forward`) + 2 GPU-h × {AL, AZ} × seed=42 × 5 folds.
- **Decision gate:** if reg peak shifts later or magnitude improves at any AL/AZ → file new finding (P4-residual-K/V), strengthens `mtl_improvement`'s arch-axis case. If no change → P4 conclusion fully closed.
- **Risk:** silenced K/V may change cat F1 — report it as a side effect; the test target is reg dynamics, not cat performance.
- **Sequencing:** run anytime; independent of A/B/C1/C2/D.
- **VERDICT (2026-05-28): P4 FULLY CLOSED — no P4-residual-K/V finding filed.** Zeroing the cat→reg cross-attention K/V channel does NOT recover MTL reg or delay its peak: AL reg Δ−0.28 pp (ns), peak ep 12.8→9.4 (EARLIER, opposite of the hypothesis); AZ reg Δ+0.01 pp (ns), peak 6.2→6.6 (≈unchanged). Cat side-effect negligible (AL +0.37 ns, AZ −0.04 ns). Combined with P4 frozen-cat-params, both cat-parameter AND cat-activation pathways are exonerated — the residual MTL-vs-STL reg gap is NOT cat→backbone capacity-stealing. Narrows `mtl_improvement`'s arch-axis search by elimination. See `phase_c_verdict.md` §C3.

---

## Tier D — No-GPU audit

### D1 — Window / causal-mask audit (head_window_batch_audit §B)

- **Scope:**
  - Verify `src/data/inputs/core.py:generate_sequences` produces no leak — target check-in never appears in the input window of size 9.
  - Verify `NextHeadMTL` causal mask in `src/models/heads/next.py` strictly blocks position i from seeing position j > i.
  - Verify the per-fold log_T regenerator (`scripts/compute_region_transition.py`) uses train-only transitions for the matching seed+fold.
  - Verify `task_a_input_type=checkin` and `task_b_input_type=region` apply the same masking discipline (no cross-modality mask leak).
- **Prior art (mandatory citations in the audit doc):**
  - `docs/findings/F50_T4_C4_LEAK_DIAGNOSIS.md` — last leak audit that produced a fix in this code path.
  - `docs/findings/F50_T4_BROADER_LEAKAGE_AUDIT.md` — broader leak inventory.
  - `docs/CONCERNS.md` C19 (n_splits guard, RESOLVED 2026-05-15) — most recent leak-class fix; D1 must confirm the guard still holds across the post-fix code paths.
- **Output:** `docs/studies/substrate-protocol-cleanup/window_mask_audit.md` with either (a) "no leak found, with code-line citations" or (b) "leak found at <location>, fix proposed, before/after numbers needed".
- **Cost:** ~1 day calendar, **zero GPU**.
- **Decision gate:** audit clean → archive; audit finds leak → file a CONCERN, halt Tier B/C, fix, then re-run any affected experiments.
- **Mandatory closure table** — at the end of the audit doc, include this canonical-vs-current comparison so a future reader sees in one glance how the study moved the numbers:

  | Metric (FL multi-seed, disjoint reg unless noted) | Canonical c2hgi MTL (Phase 2 fresh-log_T baseline) | Best from this study | Δ pp | Source |
  |---|---:|---:|---:|---|
  | reg Acc@10 @ disjoint | 63.91 ± 0.16 | _fill_ | _fill_ | _Tier A / B / C cell_ |
  | reg Acc@10 @ geom_simple | 61.54 ± 4.54 | _fill_ | _fill_ | _Tier A / B / C cell_ |
  | reg Acc@10 @ STL ceiling | 70.92 ± 0.10 | (unchanged baseline) | — | §0.1 v11 |
  | cat F1 @ joint best | _baseline_ | _fill_ | _fill_ | _Tier A / B / C cell_ |
  | AL reg Acc@10 @ disjoint | _baseline_ | _fill_ | _fill_ | _Tier A / B / C cell_ |
  | AZ reg Acc@10 @ disjoint | _baseline_ | _fill_ | _fill_ | _Tier A / B / C cell_ |

  Same table replicated in each promoted Tier finding doc, scoped to the relevant state(s). Empty / unfilled rows are acceptable for items that were FALSIFIED — show the null result.

### D1 ↔ `mtl_improvement` T0.2 mask-audit handoff (mandatory before launching D1)

`mtl_improvement` T0.2 (mask audit) is gating its Tier-1 launch on the same artefact. The two studies are on independent branches and MUST NOT duplicate this audit.

**Protocol:**

1. The first agent to reach D1 / T0.2 posts an issue-style entry in BOTH `log.md` files (here and in `mtl_improvement/log.md`) declaring "TAKING D1+T0.2 audit; ETA <date>; output at `docs/studies/substrate-protocol-cleanup/window_mask_audit.md`".
2. The other agent treats that file as the shared artefact and cites it. No re-audit.
3. If the audit finds a leak, BOTH studies pause their dependent Tiers (B/C here; T1 there) until the fix lands on `main` and both branches rebase.

This avoids duplication and prevents the two studies' fix histories from diverging.

---

## Dependency map between Tiers

```
                          Tier D (no GPU, anytime)
                                │
                                ▼ (gates if leak found)
   Tier A ──────────────────► Tier B ──────────────────► Tier C
   (cheap, no new             (real new code,            (real new code,
    code, multi-seed           single-seed,               single-seed,
    promotion)                 then multi if promoted)    then multi if promoted)
```

- **D before B/C** — if D finds a leak, B/C results may be invalidated. Run D first or in parallel with A.
- **A independent of B/C** — A re-uses Phase 3 code; no new substrate or training surgery.
- **B before B3 — Lever 4** — B3 needs B1/B2 outcomes to pick the substrate to apply itself on.

---

## Falsified history (do not re-execute under any reframing)

| Item | Phase 3 verdict | Source |
|---|---|---|
| Class-balanced sampler at reg head (§4.6) | FALSIFIED −18 to −30 pp | `phase3_rank2_findings.md` |
| Frozen-cat → MTL reg recovers (P4) | FALSIFIED interference hypothesis | `phase1_phase2_verdict_v6_final.md` §P4 |
| Cat-first curriculum (§4.4 first form) | FALSIFIED via P4 | same |
| T6.2 a2.0_0.3 substrate variant under F1 | FALSIFIED −3.50 pp | `phase2p6_salvage_T6_2_T5_3_vs_shipping.json` |
| T5.3 multi-view substrate under F1 | sub-Bonferroni, no flip | same |

If you find yourself proposing something in this table reframed — STOP and write a justification of why your reframe is mechanistically distinct, in `log.md`, before touching code.

---

## Open concerns this study touches

- **C15** (MTL coupling vs matched-head STL on reg, scale-conditional, FL flips) — Tier B substrate variants are the natural test bed; if any Design B/J flip at FL pilot, C15 evolves.
- **C18** (encoder-swap leak-probe directional drift, MONITORED) — Tier B substrate swaps to Designs B/J and Lever 5 inject non-canonical encoder topologies; the cumulative leak-budget rule from C18 applies and Tier B must run the leak-probe diff per design before promotion.
- **C21** (joint_canonical_b9 selector destroys ~+11 pp of reg capacity) — already resolved by F1; this study cites the resolution and uses geom_simple as the deployable axis throughout.
- **C22** (stale log_T silent corruption) — pre-flight gate at every run.
- **C23** (dev-seed contamination at large states) — small states OK at seed=42; FL pilot (if invoked) must use {0,1,7,100}.

## Branch-coordination protocol with `mtl_improvement`

| Code path touched here | Conflict risk with `mtl_improvement` Tier | Mitigation |
|---|---|---|
| `src/tracking/best_tracker.py` (C1) | T8 multi-seed ship | **Weekly rebase** of this study's branch from `main`; if T8 lands BestTracker changes first, this study adopts the new base API and ports the 3-snapshot-dict extension onto it. |
| `src/training/runners/mtl_cv.py` freeze logic (C2) | T3.5 cat-only warmup (same file region) | Both flags coexist (`--reg-freeze-at-epoch` and T3.5's flag are orthogonal). Coordinate via PR review on `main`; whichever lands first reviews the second. |
| `src/configs/experiment.py` (any new flag) | most `mtl_improvement` Tiers add flags | Append-only; merge conflicts trivial. |
| `src/models/mtl/mtlnet_crossattn.py` (`--zero-cat-kv`, C3) | T2 backbone alternatives | Low risk — T2 builds NEW backbones; C3 modifies the cross-attn champion. If T2 produces a winner, C3 may need to port `--zero-cat-kv` to the new champion. Defer port unless requested. |
| Window/mask audit doc (D1 / T0.2) | T0.2 (same audit) | See §"D1 ↔ T0.2 handoff" above. |

**Rebase cadence:** the implementing agent for this study rebases on `main` at the start of every working session and after each Tier closure. `mtl_improvement` lives on `mtl-improve`; expected sync point is when `mtl-improve` merges to `main`.

---

## Live docs this study would touch on closure

- `docs/CHANGELOG.md` — closure entry
- `docs/CLAIMS_AND_HYPOTHESES.md` — new CH entries for any promoted Tier A/B/C variant
- `docs/CONCERNS.md` — closure of C15 if substrate variant explains scale-conditional flip; new concern if D1 finds a leak
- `docs/findings/` — new finding files per promoted variant
- `docs/results/substrate_protocol_cleanup/` — JSONs + summaries (mirror Phase 3 layout)
- `docs/NORTH_STAR.md` — possible recipe update only if Tier A/B/C promotes AND `mtl_improvement` does not produce a contradicting champion
- `docs/results/RESULTS_TABLE.md` — possible new sub-section for substrate variants under F1

---

## What this study does NOT touch (recap)

- Architectural axis (`mtl_improvement` owns)
- Large-state production runs (`composite_two_substrate_engine.md` owns FL/CA/TX, held)
- Paper canon n=20 re-aval (`paper_canon_reevaluation.md` owns, standby)
- POI decoder distill (`poi_decoder_hgi_distill.md` owns, standby)
- **`hgi_category_injection` FL/CA/TX re-open** ([`STATUS.md`](../hgi_category_injection/STATUS.md) flags this as the re-open criterion; it is **NOT** in this study's scope even though Tier B touches substrate variants — different mechanism, different supervision target. Future agents must NOT absorb HGI-category-injection re-open by mistake just because Tier B looks substrate-adjacent.)

## §4.2 Composite cross-reference (project-headline context)

`mtl-protocol-fix` Phase 3 Rank 4 established the deploy composite (STL c2hgi cat + STL HGI reg) as the **current project headline** on the reg axis: **+7 to +12 pp** disjoint reg lift vs MTL at every state ([`phase3_rank4_composite_analysis.md`](../../results/mtl_protocol_fix/phase3_rank4_composite_analysis.md)). This study's Tier A log_T-KD lift (+2 to +5 pp at AL/AZ/FL) is **smaller in magnitude** and addresses a different question (can a single MTL artefact match composite without two-model deploy cost?). Document this framing in any finding doc — Tier A's deliverable should not be confused for the project's strongest reg lift.

---

## Pointers

- Sibling parallel study: [`../mtl_improvement/`](../mtl_improvement/) (branch `mtl-improve`)
- Predecessor closed study: [`../mtl-protocol-fix/`](../mtl-protocol-fix/)
- Cross-referenced substrate study: [`../merge_design/`](../merge_design/) (Designs B, J + Lever 4)
- Future-works re-routing table: [`../../future_works/README.md`](../../future_works/README.md) §"2026-05-28 re-routing"
