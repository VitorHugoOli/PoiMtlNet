# MOBILITY_PLAN.md — venue-bridge integration plan (2026-07-06)

> **What this is.** A briefing for the future agent that will integrate mobility(-management) more deeply
> into the paper: the author will return with direction on *how far* the integration goes; this doc holds
> everything that agent needs — the settled constraints, the current bridge inventory (with quotes), the
> ready-to-apply venue-local citation, the full evaluation of the geographic near-miss metric, and the
> option ladder with costs. Companion: [`RELATED_WORK_TRIAGE.md`](RELATED_WORK_TRIAGE.md) (§4.1a the
> handover cite, §5 the rebuttal kit, §7 the two-communities/novelty analysis).
>
> **Scope note (2026-07-07):** this doc covers the *networking/telecom-management* framing axis only (L0
> applied). A separate, complementary gap — grounding the paper in actual mobility/urban-computing
> **science** (network-science evidence about how tourist check-in data behaves) — is planned independently
> in [`MOBILITY_SCIENCE_BRIDGE_PLAN.md`](MOBILITY_SCIENCE_BRIDGE_PLAN.md), built around the Moura, Aquino,
> Loureiro MSWiM 2025 paper and a light replication experiment on our own six datasets. Read both; they do
> not conflict.

## 0 · Settled constraints — do NOT violate without an explicit author decision

1. **Decisions ledger** ([`CLAUDE.md §3`](CLAUDE.md), "Venue bridge"): mobility management is
   **motivation only**. Banned: any measured network result, any prefetch/coverage curve, examples sized
   above tract granularity ("a census tract is not a radio cell"). §3 of the paper already concedes
   "even the ten most likely tracts are far too coarse to drive cell association or handover" — anything
   added must compose with that disclaimer, not contradict it.
2. **Writing law** ([`GLOSSARY.md`](GLOSSARY.md)): no em-dash, plain words for the networking audience,
   verbs bound to tests, no repo codenames.
3. **Page budget**: the **8-page EDAS no-fee budget** (re-ruled 2026-07-19; supersedes the earlier 10-page
   fee-variant ruling; the 2026-07-20 trim campaign returned the build to 8 pages), **FULL**. Every added
   line must name what it displaces.
4. **Claim discipline**: numbers only from the board ([`RESULTS_BOARD.md §1/§3`](../../docs/studies/closing_data/RESULTS_BOARD.md));
   no placeholders in shipped prose (folder convention).

## 1 · Where the bridge lives today (prose inventory, verified quotes)

The mobility-management thread currently appears in **seven places**; §6 (results) is deliberately clean:

| # | Location | Current text (load-bearing part) |
|---|---|---|
| 1 | Abstract (`main.tex:57-58`) | "If we can anticipate the next visit, mobile and urban services can act ahead of demand." |
| 2 | Intro ¶1 (`01_introduction.tex:15-20`) | "a service that can anticipate the next move can prepare ahead of time instead of reacting after the fact… staging the right content in the area a user is heading to, or planning capacity there before demand arrives" |
| 3 | Related §2.2 (`02_related.tex:48-50`) | "We choose them because they are what a mobility-aware service can act on, not to make the task easier." |
| 4 | Problem §3 (`03_problem.tex:26-38`) | the full motivation paragraph: "two kinds of preparation a mobility-aware service makes… A census tract is a neighborhood, not a radio cell… even the ten most likely tracts are far too coarse to drive cell association or handover… We do not build or evaluate such a service, or any network system, here; the application is the reason the predictions matter, not a result we claim." |
| 5 | Method §4.2 (`04_method.tex:58-60`) | "a service that wants both answers runs a single model once, instead of two separate dedicated single-task models." |
| 6 | Setup §5.3 (`05_setup.tex:76-79`) | "A mobility-aware service acts on which region will be busy, not on a single rank position: it anticipates regional demand, stages content for the right region, or plans capacity at the census-tract level." (the TOST-margin rationale) |
| 7 | Discussion §7 (`07_discussion.tex:19-29`) | the usage sketch: "one model, one forward pass, can anticipate both what a user will do next and where… The next category then hints at what to prepare in them." + the guard "It remains motivation only, since we measure no service here." |

**Structural read:** the bridge is consistent and well-fenced, but it is entirely *argued* — zero
citations to the networking side, zero measured quantity interpreting prediction error for a service.
The two cheapest reinforcements are exactly §2 (a venue-local citation) and §3 (the near-miss metric).

## 2 · The venue-local citation — ready to apply (triage action #1)

**Fact:** the bibliography contains **zero MobiWac papers** while the intro's whole motivation is
anticipatory service adaptation. **Verified target** (web-verified 2026-07-06, DOI + author list checked):

> C. L. Vielhaus, J. V. S. Busch, P. Geuer, A. Palaios, J. Rischke, D. F. Külzer, V. Latzko,
> F. H. P. Fitzek, *"Handover Predictions as an Enabler for Anticipatory Service Adaptations in
> Next-Generation Cellular Networks"*, **MobiWac '22**, pp. 19–27, DOI 10.1145/3551660.3560913.

**Exact insertion point:** `01_introduction.tex`, a **new final sentence of ¶1**, immediately after
"…and the same signal also helps recommendation and urban analysis." (line 20). Do NOT hang a bare
`\cite` on the "prepare ahead of time" sentence — that would misattribute *our* framing sentence to
Vielhaus et al. A dedicated closing sentence presents their work as the network-side instance of the same
anticipation principle at finer granularity, which *sets up* (rather than fights) §3's "too coarse to
drive handover" disclaimer.

**Ready-to-paste drafts (GLOSSARY-compliant):**
- Primary: *"The same logic, acting before the user moves instead of after, is established on the network
  side, where handover predictions let cellular services adapt in advance \cite{vielhaus2022handover}; we
  study its coarser, city-scale form."*
- Shorter variant if ¶1 must stay tight: *"In cellular networks the same logic already drives handover
  prediction, which lets a service adapt before the user moves \cite{vielhaus2022handover}."*

**Bib entry:** drafted in [`RELATED_WORK_TRIAGE.md §4.1(a)`](RELATED_WORK_TRIAGE.md). Per folder
convention, add a primary-source verification quote as a bib comment at add time.
**Cost:** ~2 lines. **Status: APPLIED ✅ 2026-07-06** (author go): the primary-variant sentence closes
intro ¶1, `vielhaus2022handover` + `luca2021mobilitysurvey` added to `references.bib` (new venue-bridge
section), plus the Luca formulation sentence in §2.2. Compile verified: 10 pages, 0 undefined, 0 bibtex
warnings, 33 rendered refs [superseded 2026-07-19: 8-page EDAS budget, no fee]. L0 (and the §2.2 half of
L1) of the ladder is therefore DONE.

## 3 · Geographic near-miss metric — full evaluation (verdict: NOT in the submission; register + piggyback P1)

> **STATUS UPDATE 2026-07-08: COMPUTED — decoupled from P1.** The metric ran standalone on the A40 (PR #59,
> 2026-07-07: AL/AZ/FL/Istanbul, v17 `dk_ovl` seed-0 5f, `MTL_DUMP_VAL_PREDS=1`; median in-distribution miss
> 3.16–8.13 km) and the random-pair floor was added 2026-07-08 (`analysis/near_miss_floor.py`: 20–241 km).
> Record: [`analysis/near_miss_RESULTS.md`](analysis/near_miss_RESULTS.md). The §3.4 placeholder-gated drafts
> below are **superseded** by the certified C4 text in
> [`MOBILITY_SCIENCE_BRIDGE_PLAN.md §12.3/§13`](MOBILITY_SCIENCE_BRIDGE_PLAN.md); execution tracking is
> `CLOSER_HANDOFF.md §P7`. The section below is kept as the original registration record.

**Definition.** For each test visit whose top predicted region is wrong: the distance (km) between the
centroid of the predicted tract (mahalle for Istanbul) and the centroid of the true one; report per-state
P50/P90 and the distribution, per fold. Visits whose true region is absent from training (the OOD share
the Acc@10 discount removes) are reported separately — they have no in-vocabulary correct answer.

### 3.1 Why it fits (and where the line is)

It **strengthens** the motivation-only ruling rather than skirting it: it is computed entirely on the
model's own test predictions (a prediction-quality metric in km), involves no network/service/coverage
quantity, and its unit IS the census tract — exactly the granularity the ruling right-sizes to.
Epistemically it matches the already-settled §5.3 margin rationale (argue from what a service acts on,
measure only the prediction). **Two hazards to respect:** (a) a distance CDF with a service radius or
threshold overlaid IS the banned coverage curve in disguise — keep bare percentiles or a bare CDF;
(b) keep the measurement verb on the prediction ("we measured the distance between the predicted and the
true region") and service value as explicitly untested interpretation, mirroring §7's guard.

### 3.2 Feasibility — the hard facts (repo forensics, verified 2026-07-06)

**Verdict: `needs-rerun` unless piggybacked.** Nothing computable exists in committed artifacts:
- The citable cell JSONs (`docs/results/closing_data/…_score.json`) hold **fold-level aggregates only**
  (13 keys; `cat_per_fold`, `reg_per_fold`, best-epochs). The TX/CA folders add per-**epoch** val CSVs —
  still aggregates.
- **Per-sample predictions are definitively absent everywhere.** `mtl_eval.py`'s `evaluate_model()`
  *computes* per-sample top-k indices and rank-of-target internally (chunked S2 path) and then reduces
  to scalars and **discards them**; nothing in `src/tracking/` or the runners writes per-sample rows.
  (BRIDGING_METRICS.md's "saved logits" phrase refers to rundir metric CSVs on the run machines, not
  per-sample logits.)
- **Checkpoints were never written**: best states live in the in-memory `BestModelTracker` and die with
  the process; the board driver (`p3_board.sh`) does not set `--save-task-best-snapshots`. There is
  nothing to re-evaluate from.
- **Geometry IS locally available** (no census downloads needed): `output/check2hgi/<state>/temp/boroughs_area.csv`
  = GEOID + full WKT polygon (EPSG:4326); the graph artifact `temp/checkin_graph.pt` holds
  `region_to_idx` (class index → GEOID), `region_adjacency`, `region_area` (verified at AL: 1,109
  regions). ⚠ GEOIDs are stored as int64 with the leading zero dropped (AL `1073012918` =
  `01073012918`) — zero-pad to 11 chars before any TIGER join.

**Minimal pipeline** (~0.5–1 day human, near-zero GPU if piggybacked):
1. **Code (local, ~50–100 lines):** env-gated dump (e.g. `MTL_DUMP_VAL_PREDS=1`) in
   `mtl_eval.py`/`mtl_cv.py`: at each reg diagnostic-best improvement, overwrite
   `<rundir>/metrics/fold{N}_reg_val_preds.parquet` with true region_idx + top-10 predicted idx
   (int32+zstd; ~10–20 MB/fold at TX/CA scale, KB–MB at AL/AZ). ⚠ Align the dump trigger to
   `top10_acc_indist` (the matched scorer's selection metric) — the internal tracker may monitor a
   different key.
2. **Run:** let the flag ride the **P1 n=20 top-up** on the H100 (<1 % overhead; P1 retrains every cell
   from scratch anyway). Optionally also pass `--save-task-best-snapshots` during P1 so fold weights
   survive for ANY future re-scoring (today none exist). Dump parquets ride the existing rundir-sidecar
   autocommit.
3. **Offline (local CPU, minutes):** join predicted/true idx → GEOID → WKT centroid, haversine, per-state
   P50/P90 + CDF; OOD visits reported separately.

> ⚠ **TIME-SENSITIVE COUPLING:** the dump flag must be merged into the branch the H100 lane runs
> **before P1 launches**. Missed, the metric costs its own dedicated ~1–2 H100-days re-run (A40 lane
> verified infeasible for FL/CA/TX).

### 3.3 Placement verdict (the "is it worth it" answer)

- **Submission build: NO.** The numbers do not exist; the folder's discipline forbids placeholders; the
  10 pages are full [superseded 2026-07-19: 8-page EDAS budget, no fee]; and a fig4 CDF panel was **assessed and rejected** (the current single-column axes
  cannot honestly host a second panel, and a thresholded CDF edges toward the banned coverage curve).
- **Registered ✅ (2026-07-06, zero page cost):** now in [`BRIDGING_METRICS.md`](BRIDGING_METRICS.md) as
  **deferred item 4** (definition, OOD handling, guardrails, and the P1-piggyback source plan), keeping
  one provenance trail with the three existing deferred re-scores.
- **Rebuttal:** it is the prepared answer to "Acc@10 over 8,501 classes says nothing about how bad the
  misses are" (rebuttal kit, `RELATED_WORK_TRIAGE.md §5`).
- **Camera-ready (if computed AND if misses are in fact near):** exactly **one sentence**, first choice
  end of the §6.2 region-result paragraph (after "the opposite of a cost that grows with the size of the
  spatial problem"), second choice inside §7's usage sketch. Not both plus a figure. Buy-back options:
  compress the §6.2 ordering-caveat sentence ("We order by region count because…") or trim the §7
  sketch-closer clause that duplicates limitation 3.

### 3.4 Ready-to-paste drafts (ALL gated on computed numbers — never ship placeholders)

- §6.2: *"A rank metric does not say how far a miss lands, so we also measured the distance between the
  predicted and the true region on the visits where the top guess is wrong: half of those misses fall
  within [X] km of the true tract and ninety percent within [Y] km (per-fold medians, seed 0)."*
- §7: *"On our test folds, most wrong region predictions land close to the true tract (half within [X]
  km), so even a miss usually names a nearby part of the city; whether that nearness is enough for a
  given service is a question we do not test."*
- Standalone mini-figure caption (camera-ready only, never a fig4 panel): *"Distance from the predicted
  to the true region when the model's top guess is wrong. Half of the misses fall within [X] km. This is
  a property of the predictions on our test folds, not a measured service result."*

## 4 · Integration option ladder (pick when the author returns)

| Level | What | Cost | Status / risk |
|---|---|---|---|
| **L0** | Venue-local handover cite (§2) | ~2 lines + 1 bib | **Recommended now.** Zero risk; defuses "why here?" |
| **L1** | Networking-side prose thickening: 1–2 sentences in §2.2 or §3 citing the coarse-location lineage that makes region targets native at networking venues (next-cell prediction; e.g. "Next-cell and mobility prediction in new generation cellular systems", Computer Networks 2024; handover-prediction line) | 2–4 lines + 1–2 bibs | Optional. Strengthens the two-communities positioning (`RELATED_WORK_TRIAGE.md §7`); risk: dilutes §2's focus if it grows past two sentences. Verify each candidate primary-source before citing. |
| **L2** | Near-miss proxy metric (§3) | 0 pages now (register + P1 flag); 1 sentence at camera-ready | Verdict above: not in submission; time-sensitive P1 coupling. |
| **L3** | A measured service result (prefetch/staging hit-rate over predicted tracts, capacity-planning simulation) | new experiments + a results subsection | **BANNED by the settled ruling** (motivation only). Reopening is an author-only decision; it would change the paper's genre and is post-MobiWac / thesis material. If ever reopened: the near-miss dump (§3.2) is its data prerequisite too. |

## 5 · Adjacent defense worth prepping (from the novelty review, R1)

**The POI-then-derive baseline gap.** No baseline in the paper trains on the exact-POI vocabulary and
*derives* category+region from the predicted venue (faithful STAN was retargeted to rank regions
directly; the cascade tests coupling topology inside our own model). An IR-trained reviewer can ask for
it. Cheap rebuttal-experiment option (post-deadline, NOT a submission blocker): run faithful STAN in its
native next-POI form at AL/AZ/FL/Istanbul (CA/TX infeasible per the board), map the top-k predicted POIs
to their tract/category, score against the same folds. Whatever the outcome, it is reportable: either the
derive route is worse (the argument becomes measured) or it is competitive (an honest scope note). Keep
with the rebuttal kit; decide only if a review asks.

## 6 · Open questions for the author (answer on return)

1. **Scope:** which ladder level (L0 / L0+L1 / L0+L2)? L3 stays closed unless you explicitly reopen it.
2. **P1 coupling:** do we merge the `MTL_DUMP_VAL_PREDS` flag (and optionally
   `--save-task-best-snapshots`) before the P1 launch? This is the only time-sensitive decision here.
3. **Storage:** commit near-miss dump parquets for all P1 seeds or seed-0 only (~hundreds of MB vs tens)?
4. **Handover-cite wording:** primary or shorter variant (§2)?
5. **Istanbul geometry:** confirm the mahalle polygons exist in the Istanbul artifacts as they do for the
   Gowalla states (verified at AL; Istanbul assumed, not yet verified).

---
*Prepared 2026-07-06 from three verified analysis lanes (survey/novelty evidence sweep, repo feasibility
forensics, placement analysis) + hand spot-checks (intro insertion point, BRIDGING_METRICS content, STAN
retargeting). Nothing in the paper or BRIDGING_METRICS.md has been edited; §2 and the §3 registration are
ready to apply on author go.*
