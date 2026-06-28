> ⛔ **ARCHIVED 2026-06-28 — the work this document drove is DONE.** Kept for provenance, not live
> state. Current status and pointers: [`../CLAUDE.md`](../CLAUDE.md). (Relative links below may be off by one
> directory level after the move to `archive/`.)

# MobiWac 2026: Baseline Execution Handoff (for the executing agent)

> **Status: DECISIONS LOCKED (Vitor + advisor, 2026-06-24).** This tells an executing agent exactly what to run,
> on which states, under which protocol, where the numbers go, and the traps to avoid. **Read first:**
> [`BASELINE_AUDIT.md`](BASELINE_AUDIT.md) (the why), [`../../docs/studies/closing_data/RESULTS_BOARD.md`](../../docs/studies/closing_data/RESULTS_BOARD.md)
> (our result + board protocol/device rules), [`../../docs/baselines/README.md`](../../docs/baselines/README.md)
> (table schema), and the per-baseline docs under `docs/baselines/`.
>
> **Istanbul baselines** — DONE (PR #51/#53): Markov-9-cat (24.55), POI-RGNN (30.12) and faithful STAN (61.86) are
> filled in Table 3; only ReHDM-faithful remains (deferred to CUDA). Results + the ReHDM run command:
> [`../../docs/studies/closing_data/ISTANBUL_BASELINES_RESULTS.md`](../../docs/studies/closing_data/ISTANBUL_BASELINES_RESULTS.md).
>
> **House rules:** SLIM tables (we already have a lot of data; extra columns confuse); American English, no
> em-dash; tasks named **next-category** / **next-region**; numbers from committed JSONs only; n=5 (seed 0 × 5
> folds) reduced board, {1,7,100} top-up post-deadline; fp32 for large-state CUDA cells; leak-free per-fold
> train-only priors; user-disjoint `StratifiedGroupKFold`.

---

## 1 · The three baseline ROLES (protect this separation in the writing)

The story is solid but easy to muddle. Every baseline serves exactly one of three distinct questions. **In the
text, keep these three in three clearly-labeled places; do not let a reader conflate them** (this is the BRACIS
clarity lesson):

1. **Task SOTA — "is our prediction good vs the external state of the art?"**
   next-category: **POI-RGNN** + **Markov-9-cat** floor. next-region: **ReHDM** + **STAN** (Markov-1 region
   DROPPED 2026-06-24, we beat ReHDM + STAN-on-substrate, so the floor is redundant; one-clause text mention at most).
2. **Representation — "is our Check2HGI embedding the source of the category gain (not just any contextual
   embedding, not just feature injection)?"** → **CTLE** (the closest prior contextual check-in embedding) +
   the **feature-concat control**, at **FL only**.
3. **Multi-task — "is joint training better than the published alternatives?"** → **CSLSL cascade** (preferred)
   with **HMT-GRN** kept as the already-run safety net.

---

## 2 · Context (the result these baselines frame)

`RESULTS_BOARD.md` (2026-06-24): the single joint model **beats the dedicated category ceiling at every state
(+4.7 … +8.1 pp)** and **beats the region ceiling at the large states** (FL +0.57, CA +2.18 [5f], TX +2.17
[2/5f]), **matching within 2 pp at the small** (AL −0.18, AZ −0.06, Istanbul −0.58). MTL region clears Markov-1 by
+12 … +23 pp everywhere. The baselines must show our STL and MTL beat the external SOTA and that the
representation gain is real.

---

## 3 · The protocol (all SOTA baselines)

**"Faithful execution, our data"** = the baseline's own published architecture, trained from its own raw inputs
(no leak of our pretrained substrate into a SOTA row), on **our Gowalla states (+ Istanbul where feasible)** under
**our user-disjoint `StratifiedGroupKFold` folds** (paired, fair), leak-free, matched metric (cat macro-F1; reg
Acc@10), seed 0 × 5 folds (n=5). This is the `faithful` variant in `docs/baselines/`. The one exception is STAN
(see §5, role-1) and the FL representation controls (role-2), which are substrate-column runs by design.

---

## 4 · Tasks by role

### ROLE 1 — Task SOTA

**next-category**
- **POI-RGNN (faithful, all states) — DONE, just tabulate.** Cite the **canonical** numbers FL 34.49 / CA 31.78 /
  TX 33.03 (+ AL/AZ/GA from the audited JSONs), NOT the superseded `next_category/comparison.md` snapshot
  (`POI_RGNN_AUDIT.md`). **The "protocol mismatch" is NOT a concern** — by design we compare the authors' faithful
  implementation on our data against our solution; that is the intended apples comparison. **Acceptance:** our MTL
  cat beats it +40 … +48 pp everywhere (expected-huge).
- **Markov-9-cat floor — DONE, tabulate.** AL 20.50 / AZ 23.92 / FL 29.74 / CA 27.59 / TX 28.67. The honest cat
  floor.

**next-region**
- **ReHDM (faithful) — done at AL/AZ/FL** (66.06 / 54.65 / 65.68; `rehdm.md`). CA/TX faithful is deferred
  (~75–120 h/state). **Action:** confirm AL/AZ/FL; **at CA/TX, footnote "faithful infeasible at scale" — do NOT
  burn the compute** (LOCKED D2). **Acceptance:** our MTL reg beats ReHDM AL +3.75 / AZ +4.69 / FL +11.6 (AL is
  the thinnest region margin in the paper — keep it honest and n=5-provisional).
- **STAN → use the `stl_hgi` variant (LOCKED D1), NOT faithful.** AL 62.88 / AZ 54.86 / FL 73.58 / **CA 60.45 /
  TX 62.70** (`stan.md`). Faithful STAN (AL 34.46, *below* Markov-1) is a strawman + infeasible at CA/TX; do NOT
  headline it (footnote its existence at AL/AZ/FL with the from-scratch/cold-user caveat). **Frame `stl_hgi`
  honestly:** "the STAN architecture given a shared pretrained substrate" — this is the *tightest, fairest,
  scalable* competitor; beating it ("even handed our embedding, STAN loses": FL +3.7, CA +5.2, TX +4.4) is the
  strongest claim, and it is the only external region SOTA point at CA/TX. **Acceptance:** our MTL reg beats STAN
  `stl_hgi` at every state.
- **Markov-1 region floor — DROPPED (2026-06-24).** Not a table row. The joint model clears it by +12 … +23 pp
  everywhere; keep at most a one-clause text mention (we already beat ReHDM and STAN-on-substrate, the stronger
  comparison).

### ROLE 2 — Representation (FL ONLY, the novelty validation; keep out of the main tables, a small block)

- **CTLE at FL — the key comparison; ⛔ SUBMISSION BLOCKER (artifact gap, 2026-06-25 audit).** The FL CTLE
  numbers in the docs are cited from **files that do not exist on disk**: `florida_ctle.json` (SC) ran **fold-0
  only**, and `results/ctle_e2e_b1/florida/` (E2E 29.65) **was never created**. At the headline state, the
  representation contrast is currently **non-reproducible as written**. **Must run + commit FL CTLE-SC 5f AND FL
  CTLE-E2E 5f before submission.** (The frozen-below-floor is already DIAGNOSED REAL at AL — a genuine
  substrate-swap, not a bug — so the +37 gap is defensible once FL has real 5f numbers.) CTLE is the closest prior
  contextual check-in embedding, so it is the one a knowledgeable reviewer wants. **Tasks:** (1) the below-floor
  frozen number is **diagnosed REAL** (substrate genuinely swapped, identical head reaches 55.6 on Check2HGI vs
  17.8 on CTLE), not a pipeline bug, so do not re-diagnose; just run the FL 5f; (2) **run CTLE-E2E** at FL (`scripts/baselines/ctle_e2e.py`) for CTLE's *true*
  fine-tuned strength (~21 at AL note); (3) **present CTLE fairly:** report **CTLE-E2E** as the headline CTLE
  number, with **CTLE-SC** as the matched-frozen-capacity companion, both at FL, vs **Check2HGI-SC** (matched
  head `next_gru`, matched folds, matched min_seq=10 rows, leak-clean per-fold — NOT `--folds 1`). The honest
  reading is "even in its best (E2E) form CTLE is well below ours, and under matched frozen capacity ours is the
  exception" — never "we crushed CTLE." **Acceptance:** Check2HGI > CTLE-E2E and > CTLE-SC at FL, with the
  framing above. **Writing rule:** the novelty claim stays scoped to "a specific combination (check-in +
  hierarchical graph + infomax)"; cite + conceptually distinguish CTLE (masked-LM, next-POI) in related work.
- **feature-concat control at FL — DECISIVE, cheap.** HGI per-place vector ⊕ raw per-visit features (category
  one-hot + hour/day sin/cos) → same `next_gru` head, same folds. **No new embedding training** (concat existing
  vectors); write the thin input-builder wrapper if none exists. Answers "is the gain just feature injection, not
  the hierarchy?" **Acceptance:** Check2HGI > (HGI ⊕ features) at FL.

### ROLE 3 — Multi-task comparator

- **CSLSL cascade (category→region) — DONE (AL/AZ/FL, canonical dk_ovl 5f).** Scaffold: `scripts/baselines/b4_cascade.py`.
  **Result: a DEAD TIE** with our parallel champion-G on the joint objective (Δjoint AL +0.02 / AZ +0.00 / FL ≈0,
  all ≪ fold-std). **Frame it honestly: our parallel model TIES the dominant published cascade at EQUAL cost** (it
  reuses identical heads, the same frozen representation, one forward pass). This is a **defense** (it rules out
  that a cheaper cascade would have matched our STL-ceiling lift), **NOT a "we beat the cascade" claim** —
  contribution 2 is anchored to the §1 STL-ceiling lift, not to this cell. CA/TX deferred (post-deadline, optional).
- **HMT-GRN — KEPT as the safety net (LOCKED D4), already run.** AL 57.05 / FL 63.74 / CA 49.61 (Mac/MPS, PR #38
  audited; the old 62.37 was an anomaly), well below our MTL. Keep it as the external-MTL row. **If CSLSL lands,
  HMT-GRN gracefully demotes to a footnote; if CSLSL slips on time, HMT-GRN is the external-MTL baseline and the
  MTL-story hole stays closed either way.**

---

## 5 · DROPPED — do NOT re-add

- **CTLE as a SOTA row / full SC cross-check program across states** — CTLE stays ONLY as the FL representation
  control (role 2). No CTLE-SC at other states, no ReHDM-SC, no full ladder.
- **MHA+PE** (redundant with POI-RGNN; slim).
- **STAN-faithful as a headline** (footnote-only; strawman + no CA/TX).
- **SC-region** (quarantined `INVALID_PENDING_RERUN`; region substrate-isolation is weak and not the story).
- **LLM zero-shot, new from-scratch region SOTA training, the 2022-25 graph-wave as run baselines.**

---

## 6 · Final table shapes (slim) + the writing-care note

- **Table A — next-category (macro-F1):** ours STL + ours MTL · POI-RGNN · Markov-9-cat. Per state.
- **Table B — next-region (Acc@10):** ours STL + ours MTL · ReHDM (AL/AZ/FL; CA/TX footnoted) · STAN-`stl_hgi`
  (all states). Per state. (Markov-1 region dropped.)
- **Validation block (FL only, small):** Check2HGI-SC · CTLE-E2E · CTLE-SC · (HGI ⊕ features). ~4 rows.
- **MTL comparator:** CSLSL cascade (if it lands) and/or HMT-GRN, in/near Table B or a small joint-objective table.
- Mark every cell **n=5 provisional**; never print a 5f TX mean (TX MTL is 2/5 folds); never cite the VOID
  fp16/bf16 JSONs (`RESULTS_BOARD §3`).
- ⚠ **WRITING CARE (Vitor):** the three roles (§1) must read as three separate questions. Suggested phrasing:
  "We compare against the external state of the art (Table A/B); we isolate the contribution of our representation
  against the closest contextual embedding, CTLE, and a feature-concat control (FL); and we compare joint training
  against the cascade alternative." Do not let CTLE (a representation control) blur into the SOTA tables, and do
  not let the `stl_hgi` STAN variant blur into "STAN faithful".

---

## 7 · Priority order

1. **Tabulate what exists** (POI-RGNN canonical, Markov-9-cat, ReHDM AL/AZ/FL, STAN `stl_hgi` all states, HMT-GRN)
   into Tables A/B — zero new compute, shows the SOTA-vs-ours story immediately.
2. **ROLE-2 FL block:** diagnose+fix CTLE (E2E + SC), run the feature-concat control. Small, decisive for novelty.
3. **ROLE-3 CSLSL cascade** (`b4_cascade.py`) — the one new run with the highest story value; HMT-GRN already
   covers the fallback.
4. Post-deadline: {1,7,100} → n=20 top-up; CA/TX faithful region SOTA only if compute frees up.

---

## 8 · Locked-decision record
- **D1** STAN = `stl_hgi` (not faithful). ✅
- **D2** CA/TX region SOTA = footnote faithful-infeasible; `stl_hgi` is the CA/TX point. ✅
- **D3** Representation control = **CTLE** (option b), with the frozen problem fixed (E2E + fair framing); NOT
  POI2Vec/skip-gram. ✅
- **D4** HMT-GRN kept as the external-MTL safety net; CSLSL attempted as the preferred comparator. ✅
- POI-RGNN "protocol mismatch" = not a concern (intentional: faithful author impl on our data). ✅
