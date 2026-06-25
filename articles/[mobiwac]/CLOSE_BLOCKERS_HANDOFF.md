# MobiWac 2026: Close-the-Blockers Handoff (for the executing agent)

> **Why this exists.** The freeze-readiness audit (2026-06-25) found the paper **CLOSEABLE-WITH-CAVEATS on the
> REGULAR track**: every headline is backed by final on-disk data; what remains is **three submission blockers
> that are cheap recomputes / one short run, no new architecture**. Close these three and the paper can be
> submitted (regular track). **Read first:** [`RESULTS_BOARD.md`](../../docs/studies/closing_data/RESULTS_BOARD.md)
> (the board + the §3 file map + §4 baseline status), [`BASELINE_HANDOFF.md`](BASELINE_HANDOFF.md) (the locked
> baseline plan + the per-baseline drivers), and [`PAPER_PLAN.md §10`](PAPER_PLAN.md) (the verdict).
>
> **House rules (non-negotiable):** the board base is **gated stride-1 overlap, engine `check2hgi_dk_ovl`,
> MIN_SEQ=10**; **seed 0 × 5 folds (n=5)**; **fp32** for large-state CUDA cells (Ampere bf16 grad-NaN); leak-clean
> **per-fold train-only** (NEVER `--folds 1` for an SC build, it leaked 81.8% of val users); user-disjoint
> `StratifiedGroupKFold`; **set the AMP gate** (`DISABLE_AMP=1` / `MTL_DISABLE_AMP=1`) on any reg/joint cell and
> verify healthy late best-epochs (the fp16 NaN silently collapses a fold). Scope = **5 Gowalla states
> (AL/AZ/FL/CA/TX) + Istanbul**; GE (Georgia) is out of paper scope, build it only if a dependency forces it.
> Commit incrementally (per cell + a one-line finding); never cite a void fp16/bf16 JSON.

---

## BLOCKER 1 ⛔ (highest priority): the FL CTLE artifact gap

**The problem.** At the *headline* state (FL), the CTLE comparand the paper relies on is cited from **files that do
not exist on disk**: FL CTLE-SC (`florida_ctle.json`) ran **fold-0 only**, and FL CTLE-E2E
(`results/ctle_e2e_b1/florida/ctle_e2e_seed0.json`, the cited 29.65) **was never created**. As written, the
headline representation contrast is **non-reproducible** (reject-grade if a reviewer requests artifacts), and it
poisons trust in the whole Part-1 table because it sits next to a real, large, defensible +37 gap.

**Do NOT re-diagnose.** The frozen-below-floor (CTLE-SC < bigram floor) is already **forensically diagnosed REAL**
at AL (`docs/results/closing_data/baseline_compare/alabama_ctle_DIAGNOSIS.md`): the substrate is genuinely swapped
(cosine(CTLE, Check2HGI) ≈ 0.01; the *identical* head reaches 55.6 on Check2HGI vs 17.8 on CTLE, same rows/folds).
It is not a pipeline/leak bug. So just **run FL at full 5 folds**.

**Tasks (commit both JSONs):**
1. **FL CTLE-SC, 5 folds.** Build the frozen CTLE embedding at FL (`scripts/baselines/build_ctle_substrate.py`,
   pinned MIN_SEQ=10 to match `dk_ovl`), feed it to our `next_gru` category head, **leak-clean per-fold (`--folds 5`,
   per-fold train-only pretrain)**, matched rows/folds vs the FL Check2HGI-SC comparand (Check2HGI-SC FL ≈ 73.47).
   Exact command + the CUDA path: `RESULTS_BOARD.md §4` (FL CTLE-SC was queued there).
2. **FL CTLE-E2E, 5 folds.** `scripts/baselines/ctle_e2e.py` at FL (CTLE's native fine-tuned form; AL E2E = 21.24
   on disk at `results/ctle_e2e_b1/alabama/ctle_e2e_seed0.json` is the template). Commit
   `results/ctle_e2e_b1/florida/ctle_e2e_seed0.json`.
3. **Strike the phantom numbers.** In `RESULTS_BOARD.md` and
   `docs/.../H100_FL_BASELINES_FINDINGS.md`, replace the fold-0 SC (27.98/73.00) and the never-run E2E (29.65)
   with the real 5f values, or remove them until the JSONs land. Update `docs/baselines/next_category/ctle.md`.

**Acceptance:** FL CTLE-SC 5f and FL CTLE-E2E 5f both on disk and committed; Check2HGI-SC > CTLE-E2E > or ≈
CTLE-SC at FL; the FL representation block is reproducible (no phantom paths cited anywhere). **Presentation rule
(from the audit):** present CTLE as a *ladder* (one-hot / skip-gram / POI2Vec / CTLE-SC, all frozen → our head)
with CTLE-E2E beside it, and say "even in its best (E2E) form CTLE is well below ours; under matched frozen
capacity ours is the exception"; **never "we crushed CTLE."**

---

## BLOCKER 2: Tbl 2 overlap re-score (one windowing for the whole paper)

**The problem.** Part 2 is on the **overlap** board; the Part-1 substrate comparison (Tbl 2: Check2HGI vs HGI
category macro-F1 + the per-visit share) is still on the **non-overlap** base (the +15…+29 numbers). Mixed
windowing is exactly what the plan forbids; a reviewer asks "why two windowings?".

**What already exists vs what is missing.** The Check2HGI arm under overlap is **already on the board** (it is the
STL category ceiling: AL 55.87 / AZ 57.13 / FL 75.15 / CA 70.26 / TX 69.95; `RESULTS_BOARD §1`). Only the **HGI
arm under overlap** (and the POI-pooled arm, for the per-visit share) is missing.

**Tasks:**
1. **HGI-category-STL under `dk_ovl`** at AL/AZ/FL/CA/TX + Istanbul. Build the HGI-substrate inputs under the
   overlap windowing (MIN_SEQ=10), then `train.py --task next --engine hgi --cat-head next_gru`, seed 0 × 5f, on
   the same folds. **Prereq:** HGI embeddings exist at AL/AZ/FL; **build HGI at CA/TX** first (HGI is windowing-
   independent; see the substrate-build drivers referenced in `RESULTS_BOARD §3` / `HANDOFF_A40.md`).
2. **POI-pooled Check2HGI-category-STL under `dk_ovl`** (for the per-visit-context share = canonical vs
   POI-mean-pooled), same states, same protocol. (Optional if time-pressed: the per-visit *share* is windowing-
   robust enough to footnote as the non-overlap CH19 value, but the cleanest is to re-score.)

**Acceptance:** Tbl 2 = Check2HGI-cat-STL (board) vs HGI-cat-STL (new, overlap) at the 5 states + Istanbul, on one
windowing; the substrate margin reported (expect ~+15…+29 magnitude); the per-visit share re-scored or footnoted.
Update Tbl 2 in `PAPER_PLAN.md` and mark it no longer "non-overlap". (Fig 4 embedding-geometry is windowing-robust
and needs **no** re-score.)

---

## BLOCKER 3 ✅ DONE (2026-06-25, macOS CPU): Tbl 1 dataset-statistics windows column (pure recompute, no training)

> **Closed.** Overlap window counts + max/avg seq-len + sparsity computed for all 6 states and written to
> `PAPER_PLAN.md §5` (Tbl 1) and `docs/studies/second_dataset/STATS_T1.md`. Recompute **validated** against the
> one on-disk dk_ovl parquet: AL = 96,326 (= on-disk exactly); FL = 1,274,418 (= the value cited here). Windows:
> AL 96,326 · AZ 200,895 · FL 1,274,418 · CA 2,925,466 · TX 3,830,414 · Istanbul 270,217 (mahalle substrate).
> Script: `scripts/closing_data/tbl1_overlap_stats.py` (uses `generate_sequences`, stride-1/MIN_SEQ=10/emit_tail=False).


**The problem.** Tbl 1's "Windows" column holds **non-overlap** counts (FL 159,175); the paper uses overlap (FL ≈
1,274,418). Plus max/avg sequence length and sparsity are unfilled.

**Tasks (no GPU):**
1. **Overlap window counts** per state + Istanbul: count rows in the frozen `dk_ovl` category input parquet
   (`output/check2hgi_dk_ovl/<state>/input/next.parquet`), gated MIN_SEQ=10.
2. **Max / average sequence length** from `data/checkins/<state>.parquet` (groupby `userid` size: max, and mean
   (verify mean on the parquet, do not just divide, because null-coordinate drops shift it for Istanbul)).
3. **Sparsity** = 1 − check-ins / (users × POIs), per state.

**Acceptance:** Tbl 1 in `PAPER_PLAN.md` (and `docs/studies/second_dataset/STATS_T1.md`) has overlap window
counts + max/avg sequence length + sparsity for AL/AZ/FL/CA/TX + Istanbul.

---

## SHOULD-FIX (bundle these while you are in the closing pass; not blockers)

- ✅ **DONE (2026-06-25, macOS): TOST / power statement** for the small-state region "matches" (AL −0.18,
  AZ −0.06, Istanbul −0.52). Paired per-fold TOST at δ=2 pp: **all three non-inferior** — AL p=7e-5 / 90% CI
  (−0.46,+0.09); AZ p=1.5e-4 / (−0.41,+0.29); Istanbul (s0 paired arm, Δ=−0.50) p=2e-5 / (−0.65,−0.35); every CI
  sits well inside ±2 pp. Per-fold σ is small (0.16–0.37 pp) → power ≈1.0 to declare a true match, ≥95% to reject a
  true 2-pp gap. Results pinned in `STATISTICAL_PROTOCOL.md §3.4`, surfaced in `PAPER_PLAN.md §6.2`. Script:
  `scripts/closing_data/region_match_tost.py`. (Also fixed a stale Istanbul −0.58 → −0.52 in PAPER_PLAN.)
- ✅ **W6 encoder-isolation probe — DONE 2026-06-25 (PR #48, A40), VERDICT = trunk, NOT transfer.** With the
  region stream frozen-at-init (`--freeze-reg-stream`, reg-loss off), the category head keeps the ENTIRE joint
  lift: probe cat **AL 63.50 / AZ 63.67 / FL 79.79 ≈ full-MTL cat (±0.3 pp) and ≫ STL ceiling (+4.6…+7.6 pp)**.
  → the joint category win is the shared TRUNK (architecture), not region→category transfer. **§6.2 mechanism is
  BACKED** — cite this category-side probe (`W6_ENCODER_ISOLATION.md`, RESULTS_BOARD §1c), NOT the wrong-direction
  F49. Sanity gates passed (reg group 0 trainable params, 0 nan, fp32). n=5 provisional.

---

## After the runs: paper-doc updates

- `PAPER_PLAN.md`: fill Tbl 1 (overlap windows), Tbl 2 (overlap substrate gap), §6.1/§6.2 numbers; if W6 landed,
  promote the encoder mechanism from hypothesis to finding (§2 spine, §6.2, §9); if TOST computed, state it in
  §5.3/§6.2.
- `BASELINE_HANDOFF.md` / `ctle.md` / `RESULTS_BOARD.md §4`: real FL CTLE 5f numbers; strike the phantom paths.
- Keep the **CSLSL = tie at equal cost** framing (it is a defense, never "we beat the cascade").

## Priority order (UPDATED 2026-06-25 — W6 + Blocker 3 + TOST DONE; FL CTLE partial)
1. **Blocker 2 (HGI-cat-STL under overlap):** the A40's continuing GPU work — needs the canonical CA/TX HGI build
   first (CPU, kick off early). The last open blocker requiring new runs. See `HANDOFF_A40.md §2`.
2. **Blocker 1 (FL CTLE):** H100 got **2/5 folds** (PR #47, partial; cat ~28 ≪ comparand 73.47, reg ~73 ≈ 72.71);
   W3 is already satisfied at AL/AZ/Istanbul, so FL is corroborating — finish the 5f if a card frees, else cite 2/5.
3. ✅ **Blocker 3 (Tbl 1), TOST/power, W6** — all DONE (below).

## Close-readiness checklist (the gate for "submit regular")
- [~] FL CTLE-SC **2/5** committed (PR #47, partial); FL CTLE-E2E done; W3 closed at AL/AZ/Istanbul (FL corroborates).
- [ ] HGI-category-STL under overlap (Blocker 2) committed at 5 states + Istanbul; Tbl 2 on one windowing. **← A40 next**
- [x] Tbl 1 overlap windows + max/avg seq-len + sparsity filled. **(DONE 2026-06-25)**
- [x] (should) TOST/power statement for the small-state region matches. **(DONE 2026-06-25 — tested non-inferiority, power≈1.0)**
- [x] (should) W6 probe run → **DONE (PR #48): trunk, not transfer; §6.2 mechanism BACKED.**
- [ ] No void fp16/bf16 JSON cited; AMP gate verified on every reg/joint cell.
