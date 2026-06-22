# Execution Plan — full-scope "complete the picture, then write" (overlapping-windows ADOPTED)

> Created 2026-06-18. The user's strategic decision: **adopt overlapping windows** (a base change) and
> **execute every open experiment — the overlapping-windows rebuild, Massive-STEPS Phase V, and the new
> external baselines — BEFORE writing**, so the paper rests on the complete picture. This doc is the
> critique-hardened master ordering. It supersedes the optimistic first-pass; it was adversarially
> stress-tested and corrects two material holes (see §2).
>
> **Relationship to existing docs:** the gate ledger lives in [`PRE_FREEZE_PROGRAM.md`](PRE_FREEZE_PROGRAM.md);
> the freeze prerequisites in [`closing_data/FREEZE_READINESS.md`](closing_data/FREEZE_READINESS.md); the P3
> ops inventory in [`closing_data/M0_P3_PLAN.md`](closing_data/M0_P3_PLAN.md); the run ledger draft in
> [`closing_data/RUN_MATRIX.md`](closing_data/RUN_MATRIX.md). This plan re-sequences them under the ADOPT
> decision and flags where that decision changes them.

## 0 · The discipline (why "do everything" does not mean "a moving target")

"Complete the picture before writing" is the right call for a strong, defensible paper — **but only if the
base is frozen ONCE and the full board runs on that frozen base.** The failure mode is letting a board
result tempt a base change (p-hacking / moving target). The non-negotiable rule:

> Resolve the base-**defining** forks (windowing, recipe via G0.1, baseline set) → **FREEZE one base** →
> run everything on it (champion + STL ceilings + all baselines + Phase V) → **then** write final numbers.
> No P3 training before the freeze. The "full picture" comes from a **complete board on a frozen base**,
> not from iterating the base after seeing results.

The v0 article draft ([`articles/netcore/PAPER_DRAFT_v0.md`](../../articles/netcore/PAPER_DRAFT_v0.md)) locks
**story + structure** now; **all numbers are provisional** and regenerate at the board re-score (M4).

## 1 · The two material corrections (from the adversarial stress-test) — read these first

### 1a · The second-dataset windowing inconsistency is REAL and must be a freeze sub-axis
Massive-STEPS set-(a) **and** the E2 chronological bridge are built on **non-overlapping** window=9
(`second_dataset/STATE.md`, STATS_T1). If the main board adopts overlap, a clean external-validity claim
**cannot pair stride-1 Gowalla against stride-9 Massive-STEPS** — the cat "rising-tide" effect is itself
windowing-driven, so mismatched windowing confounds the transfer claim. The earlier "second_dataset DONE,
off critical path, idle" classification is **wrong under ADOPT.** Two honest options (a freeze decision):
- **(i) Mirror overlap on Massive-STEPS:** rebuild set-(a) sequences + the 25 per-fold priors/city + the E2
  chrono split at stride-1, and **re-prove the E2 leak checks** — note this is a *per-user chronological*
  split, a **more dangerous leak surface** than the user-grouped-CV the overlap memo proved safe (a stride-1
  window near the 80/10/10 boundary can straddle train/val/test). Cost: 2 cities × 2 protocols rebuild + a
  fresh E2 leak re-audit on the CUDA Phase-V box.
- **(ii) Run the external-validity table at NON-overlap on BOTH arms** — keep a non-overlap Gowalla comparand
  specifically for that table, and footnote it. Cheaper; the main-board overlap claim and the
  external-validity claim then live on different (but each internally-matched) windowing.

### 1b · "ADOPT" must be a GATED decision, not a pre-commitment
Overlap is validated at **AL / single-seed / KD-off only**, and the memo itself warns the cat lift is
**scale-dependent** (AL +9.8 pp but FL +1.3 pp at data saturation). Freezing a 6-state board on it before the
evidence exists is a moving-target on the *central* narrative ("overlap strengthens the regime") and a
p-hacking surface on G0.1's binding base. **Honoring the user's ADOPT steer the disciplined way = run the
cheap validation FIRST:** reproduce the overlap effect at **FL + one small/mid state, multi-seed**, and
**re-run the leak audit on the overlapping surface**, *then* sign the base change. This is not
re-litigating the decision — it is de-risking it so the base you freeze is the validated one. **If the
FL/small-state reproduction comes back weak, that is itself a finding to weigh before committing the full
board.**

## 2 · Parallel lanes that are SAFE to start NOW (windowing-independent)

These are the only pre-freeze spends that survive the overlap rebuild:

| Lane | Where | Work | Windowing note |
|---|---|---|---|
| **A — substrate builds** | H100 (metered) | CA + TX v14 builds (measure CA first; overflow TX → A40). **Same machine + fixed seed as the canonical-v14 regeneration** so the build IS the hash anchor (else it's a second non-identical artifact). | Embedding is per-check-in → independent ✓ |
| **B — recipe + base validation** | A40 (free post-R4–R9) | **G0.1-advisory** (AL+FL seed0, current base, fast signal) + the **overlap FL/small-state multi-seed reproduction + leak re-audit** (§1b) | the validation runs are the gate that decides ADOPT |
| **C — baseline CODE** | A40/Mac dev | implement + smoke-test B1 CTLE, B2a POI2Vec, B2b skip-gram, B3 HMT-GRN-style, B4 cascade, B5 Flashback | **code** independent; but see ⚠ below |
| **D — reading lane** | no GPU | B1–B5 triage final sign-off + RUN_MATRIX reconciliation (drafts landed) | fully decoupled ✓ |
| **E — audit fixes** | A40 (light) | emit the v14 **hash manifest**, regenerate ONE canonical v14/state (anchor), rebuild the **stale AL log_T**, wire the shared **freshness-assert utility** into every `--per-fold-transition-dir` consumer | the BLOCKER fix |

> ⚠ **Correction (stress-test):** baseline *code* is windowing-independent, but **B1 CTLE and B2b skip-gram
> pre-train on per-fold train-only sequences** → their pretrain INPUTS (and any cached pretrain artifacts) are
> windowing-dependent and **re-run at the freeze**. Only **B2a POI2Vec** (per-POI embedding) is fully reusable.
> And the C-lane smoke-tests run on the non-overlap base, so they do **not** validate stride-1 behavior
> (causal mask + memory under ~7.5–8.4× more sequences) — budget that the "one re-run" may surface
> OOM/convergence issues at stride-1.

> ⚠ **Candidate advisory lever to run alongside G0.1-advisory (§8 #5):** **loss-scale normalization** — divide
> each task CE by `log(num_classes)` before the static `cw=0.75` (cat `ln 7 ≈ 1.95` vs reg `ln ~9000 ≈ 9.1` ⇒ a
> ~4.7× built-in magnitude gap; `cw=0.75` may be partly undoing it). The one CHEAP, recipe-touching lever aimed
> at the region-sacrifice weakness that is **neither tested nor excluded**, and mechanistically distinct from
> every closed R-gate (a *magnitude* problem → survives the P4 gradient-conflict null). A few lines, no new
> infra. ≥0.3 pp → a v17 candidate like the closed R-gates; null → exclude on the record.

**Throwaway if started early (HOLD until freeze):** seeded per-fold log_T, sequence/fold rebuilds, and **any**
model board run — all windowing- AND recipe-dependent.

## 3 · G0.1 split (removes the p-hacking surface)

- **G0.1-advisory:** current base, AL+FL seed0 — a fast, **non-binding** signal (can run now in Lane B).
- **G0.1-binding:** on whatever base the freeze pins (overlap if adopted), full **{0,1,7,100}**, with the
  **0.3 pp gate pre-registered**. Only the binding run can re-pin recipe v16 → v17. State this in advance so
  there is no temptation to pick the favorable base/seed-subset.
- ⚠ aligned-pairing may interact with stride-1's ~7.5× denser per-sample supervision → the current-base
  result may **not** transfer; the binding run on the frozen base is what counts.

## 4 · The freeze (P2) — SIX axes (one commit)

1. **Recipe** = v16 / champion G, unless **G0.1-binding** promotes → v17. (Confirm `--category-weight 0.75` is
   LIVE under static_weight+onecycle — NOT the historical alternating-step no-op.)
2. **Substrate** = v14, materialized as **ONE hash-manifested canonical artifact per state** (fixed
   machine+seed; CA/TX/GE builds anchored to it; C1/A2/A4 absolutes re-derived against it). ⚠ **GE (Georgia) is
   a first-class M0 BLOCKER, not a presumed sync** — its v14 substrate is ABSENT on the local box (only raw
   `Georgia.parquet` exists); verify-or-build it (substrate + TIGER tracts + region cardinality + `poi_to_region`
   + folds + seeded log_T) on the SAME machine+seed as the hash anchor before the freeze.
3. **Windowing** = the ADOPT/KEEP decision + stride — **validated at FL/small-state first** (§1b). **Window
   length = 9: KEEP (literature-confirmed, §9).** **`MIN_SEQUENCE_LENGTH` = 10 (user-side; NO POI filter)** at
   the frozen base — the field-aligned 10-core *user* side + 9+1-window alignment (every kept user fills ≥1 full
   window); the POI-side 10-core is next-POI-specific (omitted — our targets are category/region). Code:
   `src/data/inputs/core.py:17` 5→10, applied at the **P3 base rebuild** (M0b). ⚠ **Hold MIN at 5 during the
   overlap-validation (Lane 2) to isolate the overlap effect** vs the AL prior — don't confound two base changes.
   Massive-STEPS keeps its native filter (≥2/trail, ≥3 trails/user), disclosed.
4. **Second-dataset windowing + region granularity** = mirror overlap on Massive-STEPS (§1a, user-chosen) + the
   region label: **Istanbul mahalle (520) PRIMARY + H3-9 (2,585) as a sensitivity row; NYC TIGER tracts (1,912)**;
   gap-to-ceiling / lift-over-floor framing mandatory.
5. **Label-space** (now explicit, frozen by VALUE): the 7-root FSQ/Gowalla **category map**; region =
   census/TIGER **tract** on Gowalla + NYC, **mahalle**-primary on Istanbul; **per-state region cardinalities**
   (AL ~1.1k … CA ~8.5k; NYC 1,912; IST 520) — every B1–B5 reg run-spec threads these.
6. **RUN_MATRIX** = the signed cell + baseline-inclusion inventory (pin the E2E set HARD — §6).

Plus the protocol constant: 6 states × {0,1,7,100} × 5 folds (n=20), user-disjoint frozen folds, matched
scorer, geom_simple selector. **NOT pinned:** the numbers (P3 output) and the prose (story locked, numbers
provisional until M4). **Cannot commit** until all three gates close (G0.1-binding, overlap-validated, B1–B5
inclusion), the audit prereqs are met (hash manifest + AL log_T fresh + freshness preflight centralized), **GE
is verified-or-built**, and the §4b decisions are pinned.

### 4b · Freeze-record decisions to PIN (completeness audit, 2026-06-18)

The 4-doc-tree sweep found **zero uncaptured base-changer** — these are "pin-so-it-is-not-silently-inconsistent"
items (all cheap; recommended default in **bold**):
- **Execution config: `--compile --tf32` ADOPTED for the P3 board (user-confirmed 2026-06-19).** torch.compile
  + TF32, GPU-resident dataset (auto-fit). EMPIRICALLY result-neutral: champion-G FL non-overlap seed0 5f
  vs the byte-identical no-knobs baseline → cat +0.046 pp, reg +0.065 pp (both ≪ ±1 pp seed/fold noise);
  ~15% faster/fold. These are **execution knobs, NOT recipe identity** (v16 model/heads/loss unchanged;
  frozen embeddings untouched — compile is training-only). PIN: the whole P3 board runs with `--compile
  --tf32` ONCE (never mix compiled/non-compiled cells); reviewers reproduce *with* them. The no-knobs
  byte-identical anchors (e.g. FL 73.01/73.54) are the reference; board cells sit ~0.05 pp above. torch
  stays **2.11** (§10 NO-GO: `torch_cluster` cu128 wheel + topk re-baseline); `num_workers` **skipped**
  (non-bottleneck with GPU-resident data + non-deterministic). Eval: `pre_freeze_gates/SPEED_LEVERS.md`.
- **T4 Δm metric:** the Δm joint-score table uses reg-**MRR** as primary while the board headline is reg-**Acc@10**
  → pin **Acc@10-primary** (headline-consistent), or compute both + footnote metric-robustness.
- **Cross-state reporting convention:** main-board reg Acc@10 is NOT comparable across states (tracts 1.1k–8.5k) →
  pin the **gap-to-ceiling / lift-over-floor** convention for the main board (consistent with the second dataset),
  or accept absolute-Acc@10 cross-state with a stated caveat.
- **Selection-on-reporting-fold (B7):** no held-out test; selection + reporting share the val fold; §0.1 is
  diagnostic-best → pin **disclosure-only (pair every diagnostic-best number with the deployable geom_simple
  number + state the shared-fold bias)**; one nested-selection/per-state-holdout B7 pass is OPTIONAL.
- **Task-scope pivot (SIGNED):** record the foundational decision — **drop next-POI ranking; report next_category
  (7-class macro-F1) + next_region (Acc@10)** — in the freeze record / PAPER §Methods, with the **affirmative
  task-pairing defense** (the "next-POI MTL with the main task amputated" attack; deployment/sparsity/privacy
  rationale, positioning §2.2; `task_pivot_memo.md` drafts the sentence).
- **Leak re-audit enumerates THREE stride-1 fold paths** (the stride-9 CLEAN verdict does NOT cover them): (a) MTL
  `StratifiedGroupKFold(userid)`, (b) STL-NEXT user-disjointness, (c) **the category-STL plain `StratifiedKFold`
  carve-out** — the one NON-user-grouped fold path (`FOLD_LEAKAGE_AUDIT` line 108), where stride-1 per-(POI,window)
  rows could straddle the cat fold boundary — plus the second-dataset E2 chronological split. Anchor to Luca et al.
  (ML 2023).

## 5 · Post-freeze board (P3) + external validity (Phase V)

- **M0b:** rebuild seeded per-fold log_T + sequences + frozen folds on the **adopted windowing** (FL: consolidate
  the multi-seed log_T from the A40 first).
- **M1 → M2 → M3** (A40, the multi-day bulk): STL cat/reg ceilings → champion G at all 6 states (save per-task-best
  snapshots for the C1 supportive panel) → suite cells (T3/§0.1 first) + **all chosen baselines mirroring the
  adopted windowing**.
- **M4:** single matched-metric board re-score → the paper's source of truth (provisional → final here). **CA/TX
  are the expected-but-unmeasured PARETO cells — the C2 headline (Pareto at all 6 states) cannot be written
  final until these land; if they surprise, the headline is honestly re-scoped.**
- **Phase V (L4, CUDA):** champion G + STL ceilings + Markov-1 on NYC/Istanbul (within-user CV = Gowalla-parity),
  on the frozen substrate **and the §1a windowing decision**.

## 6 · Defer to camera-ready / a follow-up (to cap the open-ended cost)

The full board + all baselines + Phase V is the largest-possible scope. Cap it: **pin the E2E baseline set HARD
at P1b** and treat any post-freeze addition as a camera-ready item, **not** a freeze re-open. Recommended
deferrals (none weaken the core 4 beats):
- **E2/A5 chronological temporal bridge** — beat 4 (external validity) is carried by the within-user CV
  Gowalla-parity comparison; the chrono bridge is a *strengthening* robustness check and is exactly the hardest
  leak-re-audit-under-overlap. **Defer unless cheap.**
- **B4-faithful (CSLSL/CatDM E2E) and B5 DeepMove** — B3 HMT-GRN-style + B5 Flashback-only + the existing
  faithful set already establish the frontier-negative. Recommend the **cheap B4 substrate-column cascade**;
  defer faithful-B4 + DeepMove.
- **F2 scale-progression scatter** (visualizes the dissolved gap → likely DROP); **T2 / §0.8** story-dependent
  (single-v14 board drops T2; §0.8 log_T-KD is NULL on G → do NOT run).
- **EXCLUDE composite / dual-substrate routing from the board** — they forfeit the single-model thesis; if ever
  shown, supportive-panel only, never a freeze-reopening run.

## 7 · Compute realism (honest)

- **Pre-freeze:** CA/TX v14 builds = the metered H100 burst (**measure CA first — TX is ~3× FL and a
  check-in-level HGI build at that scale plausibly exceeds 6 h; route TX → A40 if so**). G0.1-advisory + overlap
  validation + leak re-audit ≈ 1–2 A40-days. Baseline code + hash manifest + reading lane ≈ dev time.
- **Post-freeze (the dominant spend, A40 unmetered):** the board = [champion G + 2 STL ceilings + composite +
  up-to-7 net-new baselines + the existing faithful E2E set] × 6 states × 4 seeds × 5 folds. ⚠ **Under stride-1,
  CA/TX carry ~7.5–8.4× more sequences than non-overlap** — the FL-anchored "hours-to-a-day" estimate is
  optimistic; the full external-baseline suite at n=20 across 6 states under overlap is plausibly an
  **order of magnitude more GPU-hours** and a genuine multi-day-to-week A40 run. **The overlap adoption is the
  cost multiplier.** This is the strongest argument for pinning the baseline set tight (§6).
- **Phase V:** modest (2 cities × 4 seeds × {champion + ceilings + Markov}, small-to-mid corpora).

## 8 · Decisions — the 4 base-forks RESOLVED (user, 2026-06-18) + new audit items

**RESOLVED ✓ (all four):**
1. **Validate overlap FIRST** ✓ — run the FL/small-state multi-seed reproduction + leak re-audit before committing
   the base change.
2. **Second-dataset windowing** ✓ — **mirror overlap on Massive-STEPS** (the complete-picture option).
3. **Baseline set** ✓ — INCLUDE B1 CTLE + B2a/B2b + B3 HMT-GRN-style + B5 Flashback-only + the existing faithful
   set + the **cheap** B4 SC cascade; **defer** faithful-B4 + DeepMove + E2/A5 + F2.
4. **G0.1 binding on the frozen base** ✓ — {0,1,7,100}, 0.3 pp gate pre-registered.

**RESOLVED ✓ (user, 2026-06-18):**
5. **Loss-scale normalization** → **RAN 2026-06-18 (A40, AL+FL seed0) → EXCLUDED (harmful at scale).** Not a
   v17. AL cat +0.61 (small-state rebalance artifact) but **FL reg −37.81 pp** (degenerate head, reached
   epoch 2-6) — normalization divides reg CE by `ln(n_reg)≈8.46` vs cat `ln7≈1.95`, starving the large reg head
   on top of `cw=0.75`; worsens with region cardinality (worst at CA/TX). Recipe stays v16. Verdict:
   [`pre_freeze_gates/LANE1_LOSSSCALE_VERDICT.md`](pre_freeze_gates/LANE1_LOSSSCALE_VERDICT.md).
7. **B7 selection-on-reporting-fold** → **disclosure-only** (pair every diagnostic-best number with the
   deployable geom_simple number + state the shared-fold bias; no separate B7 run).

**6. Windowing / filter — literature-informed (search 2026-06-18; §9):**
- **Window = 9 → KEEP (confirmed).** Typical for the daily-session family + coarse targets; ablations saturate
  early; Massive-STEPS has no window standard. Locked into axis 3.
- **`MIN_SEQUENCE_LENGTH` → RESOLVED ✓ (user, option A): raise the user-side floor 5→10** (`core.py:17`),
  **no POI filter**. Rationale: matches the Gowalla **10-core** field mode on the user side (comparable to
  GETNext/STAN/SGRec), aligns with the 9+1 window (every kept user fills ≥1 full window → removes all-padding
  users; window/sample count drops far less than the user count), and the POI-side 10-core is next-POI-specific
  (rare-POI *targets*) so it is correctly omitted for our category/region targets. Applied at the **P3 rebuild**;
  **held at 5 during the Lane-2 overlap-validation** to isolate the overlap effect. **Second dataset keeps
  Massive-STEPS' native filter (≥2/trail, ≥3 trails/user), disclosed** — not force-10-cored (gap-to-ceiling
  framing normalizes the cold-start population).

None of these blocks the §2 NOW-lanes; they gate the freeze. **GE (Georgia) is a hard M0 BLOCKER** (axis 2) —
not a decision, a build.

## 9 · Windowing / filter literature (search 2026-06-18)

**Window length.** Field is bimodal: daily-session 2–10 (DeepMove, LSTPM, GETNext avg ~7–8) · fixed-RNN ≈20
(Flashback, SGRec, PLSPL) · long attention caps =100 (STAN, GeoSAN, padding-dominated). **Window=9 is
typical/short-but-normal**; ablations show next-POI accuracy *saturates early* and very long windows add noise;
coarse targets need less context → 9 is sufficient/generous for next-category/next-region. **Massive-STEPS**
(Wongso, Xue, Salim; arXiv:2505.11239, ACM 2026): no window standard — variable-length trails grouped by an
**8 h inter-check-in gap**, mostly 2–10 check-ins; baselines inherit LibCity defaults.
**Activity filter.** Mode = **10-core** (user ≥10 AND POI ≥10: GETNext, SGRec, LSTPM, GeoSAN, PLSPL); stricter
outliers Flashback ≥100, HMT-GRN 20–50; Massive-STEPS lenient (≥2/trail, ≥3 trails/user). **Our 5 is lenient +
one-sided.** Sources: Flashback (IJCAI'20, code `setting.py` `sequence_length=20`/`min_checkins=101`),
DeepMove (WWW'18, `session 2–10`/`trace_min=10`), GETNext (SIGIR'22, daily ≈7–8/`<10` filter), STAN (WWW'21,
max=100), SGRec (IJCAI'21, `n=20`/10-core), CTLE (AAAI'21), HMT-GRN (SIGIR'22, 20–50/geohash regions),
Massive-STEPS (arXiv:2505.11239).

## 10 · Torch upgrade decision — NO-GO (2026-06-19, eval-backed)

User strategy (sound): adopting overlap re-runs all MODEL training anyway, so a torch upgrade's re-baseline
cost is largely amortized → bundle it into the one freeze. **But evaluated → NO-GO for the freeze.** Make-or-break:
`torch_cluster` (load-bearing: the check2hgi embedding builder imports `torch_geometric.nn.Node2Vec` →
`torch_cluster.random_walk` at module-load) has **ZERO torch-2.12 wheels** (cu126/cu130/cu132 all confirmed
absent at https://data.pyg.org/whl/; sdist frozen at 1.6.3). `torch_scatter`/`torch_sparse` DO have pt212 wheels;
`torch_cluster` does not. Migrating ⇒ from-source compile against cu126/130 nvcc on a shared box hosting live
runs — high-risk. Benefit ~nil (2.12's TopK RadixSelect speedup is sub-noise for our few-thousand-region Acc@10
topk; no quality win). Driver 580 supports cu126/cu130 (not the blocker). **Decision: stay torch 2.11.0+cu128
for the entire freeze.** The torch-build guard (lane1_run.sh, commit b5332b2e) enforces it. Revisit 2.12
post-paper IF torch_cluster publishes pt212 wheels OR the Node2Vec import is made lazy. ⇒ CA/TX v14 builds on
2.11 are FINAL (not throwaway) — Lane 3 unblocked.

## 11 · PR #28 (A40 pre-freeze lanes) — audit outcome + post-merge follow-ups (2026-06-19)

**MERGED.** Very-critical audit (10-dim + lean 7-dim workflows + 3 hand-verified dims): **no blockers, no gate
rework** — every gate verdict holds. G0.1 advisory NULL (correct *single-joint-loader* impl; AL aligned HURTS
cat −4.77 → random pairing is a beneficial augmentation; v16 stands). **Loss-scale EXCLUDED (FL reg −37.8 pp,
harmful at scale).** Overlap ADOPT (training-length confound REFUTED via convergence curves; leak clean; MIN
held at 5; AL cat +8.12/reg +3.57, FL cat +3.64/reg +0.62). v14 6-state recipe-identical — the builder
`build_design_k_delaunay.py` is **UNCHANGED**, so the OOM/streaming work never touched the substrate. 7 baselines
faithful (POI2Vec φ unit-tested). Champion G bit-identical; guidelines followed (C28, validate-first, MIN=5 held,
no commit-to-main).

### Resolved-here (doc-overclaim corrections, applied 2026-06-19)
- **byte-identity:** S2 chunked-val was never empirically run (default-OFF) → byte-identical *by construction*
  only; "4-dp exact" softened (baseline at 2 dp). (`FL_NEWCODE_SCAN.md`.)
- **leak:** (c) category-STL is flat one-row-per-POI → **NOT a stride-1 leak surface** (resolves §4b's
  carve-out worry); (d) chrono "stride-1 reproduction" was actually stride-9 (off-critical-path, clean by
  construction). (`STRIDE1_LEAK_REAUDIT.md`.)
- **v14:** "all 6 on one machine+seed" overstated — **AL/AZ/FL/GE are the existing canonical artifacts (Jun 2–3),
  only CA/TX fresh (Jun-19 A40)**; all recipe-identical (builder unchanged). `V14_HASH_MANIFEST.json` proves
  OUTPUT integrity but records no build-env metadata → strict one-environment identity is not provable from the
  manifest (low risk: deterministic recipe + seed 42).

### Known bug (inert for champion G; fix before relying on it)
- **`train.py --aligned-pairing` crashes** (`dataclasses.replace` on the undeclared `aligned_pairing` field).
  INERT for champion G (default off; the G0.1 advisory ran via the working `lane1_run.sh` → `FoldCreator(
  aligned_pairing=)` path). **The binding G0.1 must use `lane1_run.sh`**, or declare+wire the field first.

### P3 verification checklist (deferred — execute at/after the freeze; NOT rework)
- **Binding G0.1** on the frozen base, full {0,1,7,100}, 0.3 pp gate pre-registered — via `lane1_run.sh`.
  Expected null (advisory is clearly null/negative).
- **S2-ON byte-identity A/B** before the board enables `MTL_CHUNK_VAL_METRIC` (needed for large-state overlap
  val OOM).
- **Empirical stride-1 leak check** on the adopted overlap sequences ((a)/(b), the real board surface).
- **v14 build-env note** — record per-state build provenance (date/machine/torch/seed) when the board re-hashes
  against the manifest.
- **Streaming-path regression test** + the frozen-substrate clobber guard on the geotree/b2b builders.

### ⚠ TWO USER DECISIONS (do not block the merge; gate the P3 board)
1. **Overlap board hardware/data-path.** FL/CA/TX overlap-MTL **OOMs the A40** at 8.5× sequences. The P3 board
   needs either the CPU-resident-dataset path (`MTL_DATASET_CPU=1`, slower) or a bigger GPU (H100) —
   ~order-of-magnitude more GPU-hours. *This is the cost the "full picture" decision implies; confirm the
   hardware/data-path before the board runs.*
2. **compile+TF32 pin.** Pinned for the P3 board as result-neutral (+0.05 pp, within ±0.8 fold std; ~15% faster;
   fp32-eval hatch intact → the matched scorer stays fp32). Marked "user-confirmed" on the A40 side —
   **confirm the pin explicitly** (it re-baselines the board's absolutes by +0.05 pp; built once, reviewers
   reproduce *with* the flags). [→ §12: ADOPTED board-wide via PR #29, misdiagnosed warmup fixed.]

## 12 · PR #29 (board-enabling) — audit outcome + board prep (2026-06-21)

**MERGED.** Rigorous 6-dim audit: **no blockers.** M1 tail-gate LEGITIMATE (principled OOB de-skewing of
last-POI target over-representation; keeps the full-context prediction; train/eval consistent; no leak;
non-overlap byte-identical) — framing fix applied (dropped windows are valid examples, NOT "trivial padding").
OOM fixes (lazy per-fold + drop `FoldData.x` + streaming build + S2 auto-enable) byte-identical +
non-overlap-§0.1-safe by construction. Windowing math sound (0 leaks / 3359 windows). Guards real;
default-preserving; 179 tests pass.

**Resolves last turn's hardware decision (§11 #1):** CA/TX overlap-MTL now **fits the A40** (auto-fit,
CPU-resident dataset, ~11 h/TX; NEVER `MTL_DATASET_GPU=1`); the A100 adds parallel throughput. **compile+TF32
ADOPTED board-wide** (quality-neutral, ~13–21% faster; the "32-min warmup" was an eager-fallback cache bug,
fixed via `MTL_COMPILE_DYNAMIC=1` + shared `TORCHINDUCTOR_CACHE_DIR` + cache_size_limit 8→64; p1 STL-reg
ceiling now compiled too → fair). Apply UNIFORMLY (mixing confounds paired Δ at ±0.05 pp).

### 🔴 The headline-relevant finding — gated overlap widens the reg gap
Under gated overlap, FL reg MTL 75.52 vs STL 76.64 = **−1.12 (s42) / −1.21 (s0)** — SYSTEMATIC (vs −0.35
non-overlap; mechanism: overlap lifts the STL reg ceiling MORE than MTL reg). Cat strengthens (+3.12, beats).
The reg "matches" claim moves from "−0.3 visibly ties" to "non-inferior within a TOST margin." **Audit fix
(F3-2):** the δ=2 pp TOST in `STATISTICAL_AUDIT §0.3` is the **SUBSTRATE-axis** margin — the **MTL-vs-STL**
reg-parity claim needs its OWN pinned margin (see [`closing_data/STATISTICAL_PROTOCOL.md §3`](closing_data/STATISTICAL_PROTOCOL.md);
recommend δ_reg=2 pp justified for THIS axis, user-confirm at P2). **F3-3:** the MTL headline JSONs are not
committed → the P3 board MUST commit the MTL artifacts (C28).

### Board run-spec additions (resolve-here, audit F5)
- **⚠ Cross-GPU rule (A40 + A100 parallelization):** partition the board **BY STATE** — each state's MTL + its
  STL ceilings + its baselines run **entirely on ONE GPU**; **never split a comparison group across GPUs**
  (torch.compile emits arch-specific kernels + TF32 differs across architectures → A40/A100 cells of the same
  config can differ by ~fp-noise, confounding paired Δ). **Before parallelizing, run a cross-GPU equivalence
  A/B** (one FL cell on A40 vs A100, compiled+tf32) — confirm |Δ| ≤ fp-noise (~±0.05 pp); if it exceeds,
  by-state partition is mandatory and cross-GPU absolutes carry a caveat.
- **Pinned methodology:** [`closing_data/STATISTICAL_PROTOCOL.md`](closing_data/STATISTICAL_PROTOCOL.md) (paired
  Wilcoxon for superiority; TOST per-axis for equivalence; pairing discipline; Holm-Bonferroni) +
  [`RUN_MATRIX §2.5`](closing_data/RUN_MATRIX.md) baseline comparison design (substrate-column = embedding
  isolation; end-to-end native = system; both on OUR protocol, only emb/arch varies).

### ⚠ THE de-risk gate before the full board (advisor Condition 1 — endorsed)
Run **ONE TX (large-state) gated-overlap reg cell** (1 seed, 5-fold; matched B-A2: G-MTL FULL top10 vs STL
`next_stan_flow` ceiling, BOTH on overlap). **Rule:** |Δreg| ≤ ~1.5 pp (inside δ_reg=2 pp) → adopt overlap
board-wide; > 2 pp → DO NOT adopt, keep the non-overlap base (paper-grade at −0.09…−0.31). The mechanism warns
CA/TX (4703–8501 regions) could be worse than FL's −1.2; this single cell de-risks the multi-day board.

### USER DECISIONS — RESOLVED (user, 2026-06-21)
1. **Gated overlap → ADOPTED unconditionally** (user: "make the methodology clean, otherwise it's a mess" — ONE
   base for everything beats a conditional/mixed base; this IS the plan's "freeze one base, no moving target"
   discipline). The TX cell is **no longer a go/no-go gate** — it is **early insurance**: run TX (+CA)
   gated-overlap reg EARLY (before the full multi-day board) so the **reg-claim framing is known up front**
   (non-inferior within δ_reg, or — if CA/TX exceed 2 pp — an honest re-scope / the composite-reg fallback). We
   adopt overlap regardless; the early cell only fixes how we WRITE the reg claim, not whether to freeze.
2. **δ_reg = 2 pp → CONFIRMED** for the MTL-vs-STL reg-parity TOST (its own axis; `STATISTICAL_PROTOCOL.md §3`).
3. **A100 equivalence A/B + by-state partition → CONFIRMED.**

## 13 · Board launch sequence + device allocation (2026-06-21)

**THE GOVERNING RULE (extends the cross-GPU rule to the Macs):** every PAIRED comparison runs end-to-end on ONE
device-class. ⇒ partition the board **BY STATE** across {A40, A100, M2 Pro, M4 Pro}; a state's full cell set
(MTL + STL ceilings + its baselines) runs on ONE device. Every per-state Δ (the headline: MTL-vs-STL,
baseline-vs-ours) is then **device-internal and clean**; only the ABSOLUTE cross-state table carries a
device-class footnote (small states MPS-fp32; large states CUDA-compiled-tf32). **Never split one comparison
across device-classes** (MPS fp32 vs CUDA tf32+compile differ well beyond the ±0.05 pp effect size).

### Using the Macs for the light cells (user, 2026-06-21)
- **(preferred) Macs BUILD the light baseline embeddings** (CTLE / POI2Vec / skip-gram / one-hot pretrains,
  train-only per fold) in parallel — embeddings are device-tolerant *inputs*; the matched-head comparison then
  runs on the uniform CUDA board → cleanest (no device footnote). This is the highest-value Mac use.
- **OR Macs OWN whole small states** (AL on M2 Pro 32 GB, AZ on M4 Pro 24 GB): the state's MTL + STL + baselines
  all on that Mac → Δ's MPS-internal; footnote the device class on absolutes. ⚠ first confirm AL/AZ overlap-MTL
  fits MPS memory + the wall-time is acceptable (MPS fp32, no compile). **Do NOT** split a state's baselines onto
  a Mac while its STL/MTL is on CUDA (confounds the paired Δ).

### Launch sequence (to the freeze and the board)
1. **A100-equivalence A/B** (one FL cell A40 vs A100, compiled+tf32) — |Δ| ≤ ±0.05 pp; lets a state move between
   the two CUDA cards without drift. [gates by-state parallelization]
2. **TX (+CA) gated-overlap reg cell, EARLY** (1 seed × 5 folds, matched B-A2, both overlap) — insurance for the
   reg-claim framing (not a stop-gate); records the large-state Δreg before the full spend.
3. **P2 FREEZE** (one commit): recipe v16 · substrate v14 6-state (hashed) · windowing = **gated stride-1
   overlap, MIN_SEQ=10** · label-space · signed RUN_MATRIX (carrying §2.5 baseline design + STATISTICAL_PROTOCOL).
4. **P3 board** — all states × {0,1,7,100} × 5 folds, partitioned by state across A40/A100/Macs, CUDA cells
   compiled, **committing the MTL + STL + baseline artifacts** (C28; closes F3-3). Substrate-column + end-to-end
   native baselines per RUN_MATRIX §2.5.
5. **Stats** per STATISTICAL_PROTOCOL (paired Wilcoxon superiority + TOST δ_reg=2 pp non-inferiority + Holm).
6. **L4** second-dataset Phase V (mirror the overlap windowing).

### Reg-claim fallback (insurance, not plan)
If CA/TX blow the 2 pp margin under overlap: (a) re-scope honestly ("non-inferior at AL/AZ/FL/GE; within X pp at
CA/TX"), or (b) the 2-model composite reg panel (SUPPORTIVE only — forfeits the single-model property, never the
headline).

## 14 · Board results + the reg-base decision (2026-06-22) — DECISION C

**Board WIP (seed 0, gated overlap, frozen v14, compiled+tf32; PRs #30/#31/#32):** FL Δcat **+3.37** / Δreg
**−1.29** (H100; A40 MTL cat 79.01 / reg 75.50). **Cross-GPU (A40 vs H100, SAME frozen v14): reg robust
(Δ0.08 pp) but cat diverges 0.49 pp** (Ampere vs Hopper + compile) → **by-state partition MANDATORY, vindicated
by data** (never compare cat *absolutes* across GPUs). **TX Δreg −2.41 (EXCEEDS δ_reg=2 pp).** **CA pending**
(H100 Lightning studio interrupted, exit 137 — needs restart; FS survived). Cat beats everywhere.

**DECISION (user, 2026-06-22) = C: gated overlap STAYS the headline base; RE-SCOPE the reg claim.** The reg gap
is **cardinality-dependent** (FL 4703 reg −1.3 ✅ → TX 6553 −2.4 ✗ → CA 8501 pending/likely-worse). The reg
claim is therefore **NOT flat parity** — it is:
> *MTL is non-inferior on region (TOST δ=2 pp) at the ≤~5k-region states (AL/AZ/FL/GE) and trails by ~2.4–3 pp
> at the largest (TX/CA) — a cost that grows with region cardinality — while beating the category ceiling +3–4
> pp everywhere, in a single model.*
The cat win is the robust headline; the reg side is a **cardinality-binned non-inferiority + an honest
large-state cost** (mechanism: dense-data overlap lifts the STL reg ceiling more than MTL at high region
counts). ⇒ update `STATISTICAL_PROTOCOL.md §3` (the reg TOST is **per-state / cardinality-binned**, not pooled)
+ the netcore C2 reg prose (provisional → finalized at M4). (Note: the non-overlap base would have given a flat
clean Pareto −0.09…−0.31; the user chose overlap for methodological cleanliness + the dense-data argument,
accepting the cardinality-binned reg framing.)

**Next:** (1) **restart the H100 → run CA** (confirms the trend; most-at-risk reg cell); (2) **p3_board.sh fix**
(the 3 blockers → build via `build_overlap_probe_engine.py` [separate engine, no frozen clobber] + stride/
engine-aware next_region + log_T); (3) **M4 disk policy** = keep embeddings + provenance, regen next/next_region
on CUDA; (4) PRs #30/#31/#32 stay WIP until CA + a cleanup pass. Then **P2 freeze** (windowing = gated-overlap
MIN_SEQ=10) → **P3 board**.
