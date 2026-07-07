# MOBILITY_SCIENCE_BRIDGE_PLAN.md — giving the paper a real mobility contribution, not a citation (2026-07-07, revised)

> **What this is.** Originally scoped as "add one citation for mobility-science grounding" (§7 below is that
> first pass, kept for the record). The author pushed back, correctly: a citation is applying the paper's
> topic to ours, not giving our own results a mobility tone. This revision is built around that critique. It
> is grounded in two independent, already-existing reviews of this exact paper that name the same weakness,
> and it proposes fixes that make the paper's *own* numbers argue for mobility relevance, escalating (per
> author decision, 2026-07-07) into an actual code + experiment track for the one thing on record that a
> simulated reviewer said would flip a reject.

## 0 · The evidence this isn't a hypothetical worry

Two independent reviews of this paper, at different times, by different mechanisms, converge on the same
weakness:

**1. The archived simulated MobiWac panel** (`archive/REVIEW_PANEL.md`, 2026-06-23, five reviewers + an Area
Chair synthesis over `PAPER_PLAN.md`). Reviewer 1 (venue fit) recommends **Reject**:

> "Stripped of the lanyard, a TPC member reads a generic LBSN recommendation paper that could go to RecSys,
> SIGSPATIAL, CIKM, or a mobility-data venue with no edits... The networking motivation is unfalsifiable
> decoration, not a thread the paper can be evaluated on... even the *motivation* is mismatched to the
> metric, which a hard-networking TPC will catch immediately."

The Area Chair ranks this **W4, MAJOR** ("no amount of re-measurement fixes it") and states plainly: *"this
is the dominant reason a hard-networking TPC votes reject."* Required-change #9 in that review is explicit
about what would actually move the needle: *"Add a scope-fit subsection and, if at all feasible, one
lightweight management-side proxy evaluation... Even a trace-driven simulation with a management metric
converts the §7 'usage illustration' from forbidden decoration into the contribution the venue asks for."*
Reviewer 1's own bullet list ranks a **real proxy evaluation** first, a **re-derived network-meaningful
spatial unit** second, and an **explicit scope-fit paragraph** third, explicitly calling the third option
"the weakest... it does not manufacture an in-scope contribution, it just makes the gamble legible."

**2. The much more recent human co-author review** (`REVIEW_GERMANO.md`, advisor pass 2026-06-29 — not
hypothetical, on the actual near-final draft). Its own "whole-paper claim-discipline risks" section says,
unprompted:

> "Venue-fit register (cumulative). The service disclaimer recurs (§3 ×2, §5.3, §7 ×2, the radio-cell
> scoping); **the more it apologizes, the more it invites 'then why MobiWac?'**. Consolidate to one confident
> scoping in §3 (folds into OP4)."

That consolidation (`OP4` in `review/review_v2.md`) is tracked as **TODO, not yet run** (`review/README.md`:
"Phase 2 (OP2/OP3/OP4 sweeps)... NOT RUN as designed"). So as of today, the paper still repeats an apologetic
"we don't measure a service" disclaimer roughly five times across §3, §5.3, and §7, with no fix yet applied.

**Conclusion:** the author's instinct is right, and it is independently corroborated by both reviews. My
first pass at this plan (§7 below) proposed adding one more citation. That is precisely the class of fix
Reviewer 1 already anticipated and rejected in advance: a lanyard, not a load-bearing argument. It needed to
change.

## 1 · Why a citation alone doesn't fix this

The weakness isn't "we haven't cited enough mobility papers." It's that **nothing in the paper argues why its
own metrics (category macro-F1, region Acc@10) are the right proxy for a mobility-management decision**, and
the paper's repeated response to that gap is to apologize for it rather than answer it. Adding Moura et al.
as a citation would be true, verified, legitimate — and would not touch this problem at all, because the
problem is structural (argument, not citation count).

The fix has to make the paper's *already-computed numbers* say something a mobility reader can act on.
Nothing below requires a new model or a changed result.

## 2 · Fix 1 (ready now, zero new experiments): quantify the shortlist, don't just gesture at it

`07_discussion.tex` already gestures at exactly the right idea and never grounds it:

> "A service could treat the model's top-ranked next regions as an anticipatory set: a small list of areas
> worth preparing in advance."

"A small list" is the unfalsifiable decoration Reviewer 1 is talking about. The paper already has the exact
numbers to make this concrete, in Table 3 and the already-computed random-top-10 floors
(`BRIDGING_METRICS.md`): region count per state, and the joint model's Acc@10.

| State | regions (N) | joint Acc@10 | random top-10 floor (10/N) | enrichment over chance |
|---|---:|---:|---:|---:|
| California | 8,501 | 65.66% | 0.12% | **≈547×** |
| Texas | 6,553 | 67.02% | 0.15% | **≈447×** |
| Florida | 4,703 | 77.28% | 0.21% | **≈368×** |
| Arizona | 1,547 | 59.34% | 0.65% | **≈91×** |
| Alabama | 1,109 | 69.81% | 0.90% | **≈78×** |
| Istanbul | 520 | 74.28% | 1.92% | **≈39×** |

**Ready-to-paste replacement** for the quoted `07_discussion.tex` sentence:

> *"A service could treat the model's top-ranked next regions as an anticipatory set: at California, for
> example, a shortlist of ten regions out of 8,501 candidates, about a tenth of a percent of the space,
> contains the true next region 65.66 percent of the time, more than five hundred times better than picking
> ten at random; at Alabama, ten out of 1,109 candidates capture it 69.81 percent of the time, a similar
> order of enrichment. Concentrating preparation on that shortlist, rather than the full set of regions, is
> the kind of set a service could act on. The next category then hints at what to prepare in it."*

This directly answers Reviewer 1's sharpest, most specific line ("Acc@10... is far too loose to 'set aside a
connection' or 'get a handover ready'") — not by disputing it (it's correct for handover), but by giving the
*actually intended* use (a shortlist for staging/planning, not a single connection decision) a number instead
of an adjective. Cost: about 3-4 lines, in the exact spot the paper already spends words on this idea.

## 3 · Fix 2: reposition Moura et al., from "design rationale" to "application-legitimacy precedent"

My first pass (§7) cited Moura et al. to justify the *graph representation* (§2.1, a method-design argument).
That is honest but doesn't touch the venue-fit weakness, which is about whether the paper's *tasks and
metrics* matter to anyone, not whether a graph is the right ML architecture.

The stronger use of the same paper: Moura et al.'s own applications section explicitly maps **POI/census-tract-granularity
network analysis** to a list of concrete mobility-management uses:

> "Urban planners benefit from insights into city hotspots and flow bottlenecks... planners can better
> allocate different types of resources (e.g., public washrooms and rest areas), transit links, or event
> zones... For smart city system developers, the degree distribution and centrality metrics inform adaptive
> transportation logistics (e.g., dynamic routing of buses and shared mobility fleets)... Real-time systems
> can use this structure to feed context-aware alerts, digital signs, or augmented reality tools."

Every one of those examples operates at **exactly our granularity** (tract/POI, not radio cell) and is
recognizably "mobility management" to a MobiWac reader, without any pretense of a networking/handover claim.
Citing this alongside the shortlist quantification (§2) turns "we picked census tracts and Acc@10 for no
stated reason" into "we picked the same granularity and the same class of downstream use (resource
allocation, transportation logistics, crowd/content staging) that a venue-adjacent, recently published
mobility-science paper argues is the natural home for this kind of prediction." This is a precedent argument
for *scope*, not a design-rationale citation — a materially different and more load-bearing job for the same
reference.

The co-visitation-network light experiment (`analysis/covisitation_network_findings.md`, unchanged from the
first pass, see §7 below for the numbers) still backs this: our own check-in graphs show the same
small-world, hub-dominated structure Moura et al. document, across 5 of our 6 datasets. Keep it as
supporting evidence for "this class of data and this granularity is a legitimate object of mobility-science
study," available for the rebuttal kit; it does not need to appear in the main text given the page budget.

## 4 · Fix 3: one confident scope-fit paragraph, paid for by the apologies it replaces

The reviewers already found the funding source for this. §0 above quotes the June-29 advisor pass naming
five-ish scattered instances of the same disclaimer (§3 ×2, §5.3, §7 ×2) as actively counterproductive
("the more it apologizes, the more it invites 'then why MobiWac?'"), and that consolidation is already a
tracked, owed task (`OP4`, TODO, not yet run). This plan does not invent a new cut; it proposes what the
freed room should be spent on when OP4 runs:

**Replace** the scattered "we do not build or evaluate this, we measure no service" apologies with **one**
confident paragraph, placed once (§3 is the natural home, since that's where the task and its motivation are
first stated), doing three things at once: (a) names the concrete management uses this class of prediction
serves (citing Moura et al. per §3 above, not hedging that it might be relevant), (b) states the shortlist
enrichment argument (§2) as the reason the chosen metric is a legitimate proxy, (c) states once, plainly,
what is out of scope (radio-cell/handover-granularity claims) without repeating the concession five times.
One honest boundary stated with confidence reads very differently from five apologies for the same boundary.

**Illustrative shape** (final wording is an OP4/prose-pass job, not this plan's to finalize):

> *"These predictions serve the class of mobility-management uses that operate at neighborhood granularity:
> resource allocation, transportation logistics, and content or capacity staging, the same uses a recent
> network-science study of tourist check-in mobility identifies for tract/POI-level analysis
> \cite{moura2025mobilityaware}. [shortlist-enrichment sentence, §2]. We do not claim radio-cell-granularity
> uses such as handover or cell association, which need finer spatial resolution than a census tract can
> give; everything else in this paper is scoped to the neighborhood-level uses above."*

This is a genuinely different rhetorical move from "add a citation": it makes an affirmative, bounded,
numbers-backed claim about what the paper's predictions are good for, once, instead of restating what they
are not good for, five times.

## 5 · Fix 4 (escalated per author decision, 2026-07-07): the geographic near-miss metric, for real

Reviewer 1's own ranking put a **real, even lightweight, management-side proxy metric** above the scope-fit
paragraph as the fix most likely to flip a verdict. The paper already has one fully scoped and never built:
the geographic near-miss metric (`MOBILITY_PLAN.md §3`, `BRIDGING_METRICS.md` deferred item 4) — for each
wrong region prediction, how far (km) is the predicted region's centroid from the true one. If most misses
land nearby, that is a real, quantified, non-decorative argument that even the model's *errors* are
locally-actionable for a mobility-management reader, not just its successes (§2's shortlist argument).

**Status as of this revision: code implemented, independently verified, local smoke test attempted and
blocked by an unrelated environmental guard (details below). Ready to run on the A40.**

1. `MTL_DUMP_VAL_PREDS=1` (default off) in `src/training/runners/{mtl_eval,mtl_cv}.py` — dumps per-sample
   true region + top-10 predicted region + an in-distribution/OOD flag to
   `<rundir>/metrics/fold{N}_reg_val_preds.parquet`, firing on exactly the same event the BestModelTracker
   already uses for the region task's diagnostic-best (`top10_acc_indist` for the check2HGI next_region
   preset — no separate hook invented). Verified independently (not just the implementing agent's
   self-report): read the full diff line by line (151 lines across 2 files, nothing else touched), confirmed
   every new code path is behind the `MTL_DUMP_VAL_PREDS` env-var gate, and re-ran `tests/test_training/`
   myself (89 passed, 8 skipped — matches, unchanged from before the change).
2. `articles/[mobiwac]/analysis/near_miss_distance.py` — offline (CPU-only, read-only, no training
   dependency) join of predicted/true region index to GEOID to centroid (via
   `output/check2hgi/<state>/temp/boroughs_area.csv` / `temp/checkin_graph.pt`, handling the documented GEOID
   zero-pad gotcha and Istanbul's non-numeric OSM relation-id GEOIDs), haversine distance, per-fold P50/P90,
   OOD visits reported separately, no service-radius/CDF overlay (matches the venue-bridge guardrail).

**Local smoke test attempt (2026-07-07):** tried a minimal, bounded run (`--state alabama --engine check2hgi
--epochs 2 --only-folds 0`, `MTL_DUMP_VAL_PREDS=1`, no board flags since `--compile`/`--tf32` are CUDA-only
and the board driver itself says "GPU job, do NOT run on CPU-only/review machines" and pins
`torch==2.11.0+cu128`). It failed before reaching any of the new code, at data loading, on this repo's own
`_guard_cpu_resident_ram` safety check: the machine had **12.5 GB free out of 32 GB** at the time (confirmed
via `vm_stat`, other applications using the rest), below the guard's required headroom. **I did not override
this guard.** It exists for exactly the class of risk this repo's own history already flags (a prior local
training run caused this machine to restart under memory pressure) — a machine at ~60% RAM committed to
other things is a bad time to add a training job, board-recipe or not. This is a real, useful signal, not a
bug: it reinforces that the real run belongs on the A40, not local MPS, even for a small state.

**Ready-to-run on the A40** (a smoke-validation variant first, to confirm the dump writes a real, sane
parquet before spending time on a full board-recipe run):

```bash
# Smoke validation (bare defaults, ~seconds-to-minutes on a real GPU; NOT board-recipe-accurate numbers,
# just proves the dump mechanism works end to end on a real training loop):
MTL_DUMP_VAL_PREDS=1 python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state alabama --engine check2hgi --epochs 2 --only-folds 0

# Then check the dump landed:
ls results/check2hgi/alabama/*/metrics/fold0_reg_val_preds.parquet

# And run the distance computation against it:
python "articles/[mobiwac]/analysis/near_miss_distance.py" --state alabama --rundir <the rundir printed above>
```

For paper-citable numbers, the real run needs the board recipe (`scripts/closing_data/p3_board.sh`, or the
canonical NORTH_STAR invocation in the repo-root `/CLAUDE.md`, with `MTL_DUMP_VAL_PREDS=1` added), on the
`check2hgi_dk_ovl` engine, matching whichever state(s)/seed(s) the author wants near-miss numbers for. **I
have not launched this** — it's a real resource commitment (GPU time, and it retrains from scratch) that
should be confirmed (state, seed count, 1 fold vs full 5) once the A40 session is available, not assumed.

## 6 · Updated recommendation, in order

1. **Now:** apply Fix 1 (§2, the shortlist quantification) and reposition Moura et al. per Fix 2 (§3) —
   both are ready-to-paste, zero-new-experiment, and directly answer Reviewer 1's most specific complaint.
2. **When OP4 runs** (already an owed task, not new work this plan invented): apply Fix 3 (§4), consolidating
   the five scattered apologies into one confident, numbers-backed scope-fit statement, funded by the room
   the consolidation itself frees up.
3. **In progress:** Fix 4 (§5), the near-miss metric. Code first (underway), then a scoped, confirmed
   decision on which state(s) and which machine before any real run.
4. Keep the Moura bib entry + verification comment from §7 below; it is still correct, just repurposed.

## 7 · First-pass content (superseded in framing, not in facts — kept for the record)

The research below is still accurate and still useful; only its *use* changes (§3 above). Kept verbatim from
the original pass:

### 7.1 · The source paper

D. L. L. Moura, A. L. L. Aquino, A. A. F. Loureiro, "On the Design of Mobility-Aware Systems: A Tourist's
Perspective," **2025 IEEE MSWiM**, pp. 667-674, DOI 10.1109/MSWiM67937.2025.11308734 (Barcelona, Oct 2025).
Builds an undirected co-visitation graph from 18 months of Foursquare check-ins in Rome (nodes = POIs, edges
= shared visitors), finding small-world/hub-dominated structure and a length-of-stay segmentation (long-stay
visitors form dense, cohesive networks; short-stay visitors form sparse, fragmented ones), with concrete
downstream applications in urban planning, transportation logistics, and crowd/context-aware services.

### 7.2 · Independent verification (unchanged)

Confirmed via a 3-agent research pass: (a) genuinely new ground — not cited, discussed, or evaluated anywhere
in this repo before, and missed entirely by the prior 13-agent `RELATED_WORK_TRIAGE.md` triage; (b) legitimate
and indexed (IEEE Xplore, dblp, Google Scholar all agree on authors/venue/DOI); too recent for a citation
count; (c) MSWiM and MobiWac are confirmed co-located ACM/IEEE sister venues, so citing one in the other is
standard practice; (d) a credible backup/companion citation exists and is still worth keeping documented: T.
H. Silva, A. C. Viana, F. Benevenuto, L. Villas, J. Salles, A. Loureiro, D. Quercia, "Urban Computing
Leveraging Location-Based Social Network Data: A Survey," ACM Computing Surveys 52(1), Art. 17, 2019, DOI
10.1145/3301284 (same senior author, broader/foundational, not currently cited).

### 7.3 · The light experiment (unchanged, still valid, still useful as rebuttal-kit material)

Full write-up: `analysis/covisitation_network_findings.md` (+ `covisitation_network.py`,
`covisitation_network_results.json`). Replicated Moura et al.'s co-visitation methodology on our own 6
datasets. **Verdict, unchanged:** the small-world/giant-component/clustering-vs-density signature replicates
cleanly and even more strongly than Rome across 5 of 6 datasets we could build (Texas's graph construction
crashed the machine, reported not forced); the activity-intensity segmentation (our analogue to length-of-
stay) reproduces Rome's dense-vs-fragmented topology split at Alabama/Arizona/Florida. The "heavy-tailed /
scale-free" framing specifically is a clean match only at Istanbul; do not claim it broadly. This stays
out of the main text (too nuanced for the page budget) and lives as rebuttal-kit / precedent-for-scope
material, per §3 above.

### 7.4 · Bib entry (unchanged, ready to paste)

```bibtex
% verified: 2025 IEEE International Conference on Modeling, Analysis and Simulation
% of Wireless and Mobile Systems (MSWiM), pp. 667-674, DOI
% 10.1109/MSWiM67937.2025.11308734 (PDF-verified 2026-07-07 against
% articles/[mobiwac]/mobility/On_the_Design_of_Mobility-Aware_Systems_A_Tourists_Perspective.pdf:
% title page confirms authors Douglas L. L. Moura (UFMG), Andre L. L. Aquino (UFAL),
% Antonio A. F. Loureiro (UFMG); co-located with MobiWac (MSWiM's sister ACM/IEEE
% symposium), same mobile-systems community. Used as precedent that tract/POI-granularity
% network analysis of check-in mobility maps to concrete management uses (resource
% allocation, transportation logistics, context-aware alerts), not as a design-rationale
% citation for our graph representation (revised framing, 2026-07-07 — see
% MOBILITY_SCIENCE_BRIDGE_PLAN.md §3). Independently checked against our own check-in
% graphs (analysis/covisitation_network_findings.md): small-world/clustering signature
% replicates across 5/6 of our datasets; heavy-tailed/scale-free framing specifically
% replicates cleanly only at Istanbul, so that narrower claim is not echoed in our prose.
@inproceedings{moura2025mobilityaware,
  author    = {Douglas L. L. Moura and Andre L. L. Aquino and Antonio A. F. Loureiro},
  title     = {On the Design of Mobility-Aware Systems: A Tourist's Perspective},
  booktitle = {2025 IEEE International Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSWiM)},
  pages     = {667--674},
  year      = {2025},
  doi       = {10.1109/MSWiM67937.2025.11308734},
}
```

(Verify "Andre" vs "André" against IEEE Xplore's own metadata before finalizing, per the file's existing care
with accented names.)

## 8 · Open questions for the author

1. **Fix 1 + Fix 2 (§2-3): apply now?** Both are ready-to-paste and zero-risk to the frozen results; only
   question is exact final wording, which is a prose-pass detail.
2. **Fix 3 (§4): timing.** Bundle into the already-owed OP4 pass, or pull it forward as its own smaller edit
   before OP4 runs in full?
3. **Fix 4 (§5): once the code review lands, which state(s) and which machine?** Alabama/Arizona locally on
   MPS as a first real validation, versus going straight to the A40 for board-consistent numbers. My
   recommendation once the diff is in hand: validate the pipeline at the smallest state first (wherever it
   runs), but treat any number meant for the paper as needing the same machine discipline as every other
   board cell (i.e., the A40, not a local MPS run) so it doesn't introduce a new precision/reproducibility
   question this late in the process.
4. **If Fix 4 lands with a real, favorable number:** where does it go — a sentence in §6.2 (paired with the
   shortlist enrichment), or in the §7 usage sketch alongside Fix 1's rewrite? Recommend picking one, not
   both, per the paper's own no-repeat-the-same-point discipline (the concern OP4 already exists to fix).

## 9 · Provenance

Original pass (citation-only framing) prepared 2026-07-07 from a 3-agent research pass (co-visitation
experiment, bib/PAPER_PLAN check, web credibility check). This revision prepared the same day after the
author's pushback, grounded in a full read of `archive/REVIEW_PANEL.md` (all reviewers, not just Reviewer 1)
and the relevant sections of `REVIEW_GERMANO.md` + `review/review_v2.md` + `review/README.md`, plus the
paper's own Table 3 / §6 results numbers (`06_results.tex`, `tbl3_results.tex`) for the shortlist-enrichment
arithmetic. Fix 4's code implementation was launched as a background agent the same session; its diff is
pending review before any real run is confirmed.
