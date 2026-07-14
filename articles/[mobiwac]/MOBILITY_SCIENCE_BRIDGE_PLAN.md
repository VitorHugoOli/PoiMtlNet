# MOBILITY_SCIENCE_BRIDGE_PLAN.md — giving the paper a real mobility contribution, not a citation (2026-07-07, revised)

> **What this is.** Originally scoped as "add one citation for mobility-science grounding" (§7 below is that
> first pass, kept for the record). The author pushed back, correctly: a citation is applying the paper's
> topic to ours, not giving our own results a mobility tone. This revision is built around that critique. It
> is grounded in two independent, already-existing reviews of this exact paper that name the same weakness,
> and it proposes fixes that make the paper's *own* numbers argue for mobility relevance, escalating (per
> author decision, 2026-07-07) into an actual code + experiment track for the one thing on record that a
> simulated reviewer said would flip a reject.
>
> **Status (2026-07-07, end of day): all four fixes have concrete material in hand.** Fix 4 (the near-miss
> metric) ran for real on the A40 across four states (PR #59, independently audited and merged) — median
> in-distribution misses land 3-8 km from the truth. Nothing has been written into the paper's `.tex` yet;
> §6 is now the single, unified next step.

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

**Status as of this revision: DONE. Real numbers, on the A40, all four feasible states, independently
audited (PR #59, merged 2026-07-07).**

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

**The real run (2026-07-07, A40, `PR #59`).** Champion **v17** recipe (bs8192 + per-head cat-lr,
`MTL_ONECYCLE_PER_HEAD_LR=1`), the gated stride-1 overlap engine `check2hgi_dk_ovl`, fp32, seed 0 × 5 folds —
verified byte-for-byte against the two real canonical run scripts in `docs/studies/closing_data/`
(`run_catx_v17_n20.sh`, `run_catx_v17_seed0_5f.sh`; flag-set diff is empty), not an approximation. Results,
independently audited (diff read, recipe cross-checked, numbers cross-checked against the frozen board
cells, internal-consistency checks — OOD count vs OOD-miss count, top-1 rate vs known Acc@1 ceiling — all
passed) and merged to main:

| State | regions | reg Acc@10 (this run vs. board) | in-dist miss **P50** | P90 | mean |
|---|---:|---|---:|---:|---:|
| Alabama | 1,109 | 69.8 (board 69.81) | **8.13 km** | 38.47 | 20.38 |
| Arizona | 1,547 | 59.7 (board 59.34) | **8.05 km** | 30.71 | 17.35 |
| Florida | 4,703 | 77.4 (board 77.28) | **7.04 km** | 37.58 | 20.56 |
| Istanbul | 520 | 75.4 (board 74.28, 4-seed avg) | **3.16 km** | 14.91 | 5.75 |

(California and Texas not attempted — the largest states, and not needed to make the point; four states
already gives a consistent, non-cherry-picked pattern.) Full provenance, per-fold breakdowns, source rundirs,
and the exact reproduce command: `analysis/near_miss_RESULTS.md`.

**This changes the recommendation.** `MOBILITY_PLAN.md §3.3`'s own pre-registered placement rule said:
*"Camera-ready (if computed AND if misses are in fact near): exactly one sentence... never a fig4 panel."*
Both conditions are now met — computed, and near (a median of 3-8 km, well inside "a nearby part of the
city"). This is no longer a registered-but-hypothetical idea; it is a ready sentence with real numbers.

## 6 · Updated recommendation, in order

All four fixes now have concrete, ready material — this is no longer four separate asks at different
readiness levels, it is one prose-writing job with all its inputs in hand:

1. **Merge Fixes 1+ 2 + 3 + 4 into a single rewrite** of the mobility-motivation material currently scattered
   across §3, §5.3, and §7 (the five apology instances your reviewers already flagged as counterproductive).
   One confident paragraph that: (a) quantifies the shortlist-enrichment argument (Fix 1, e.g. California's
   ~550×-over-chance shortlist), (b) cites Moura et al. as precedent that this granularity and class of use
   (resource allocation, transportation logistics, content/capacity staging) is recognized mobility-management
   territory, not our invention (Fix 2), (c) adds the near-miss number now that it is real (Fix 4: "a median
   miss of 3-8 km, a neighboring region, not a random one"), and (d) states once, plainly, what is out of
   scope (radio-cell/handover claims) — replacing five apologies with one affirmative, numbers-backed claim.
2. **Page budget:** this is a genuinely bigger ask than the original "~2 lines" L0 citation. Rough accounting:
   removing 4-5 scattered one-sentence apologies frees perhaps 4-8 lines; the merged paragraph (enrichment +
   Moura + near-miss + one scope line) is perhaps 8-12 lines. Close to a wash, not a blank check, but it needs
   an actual page-budget check after drafting, not an assumption either way.
3. **Placement:** §3 (Problem and Tasks) is the natural single home — it is where the task and its
   motivation are first stated, and where most of the apology instances already live. §7's usage sketch then
   gets to *reference* the established argument instead of re-litigating it (fixing the OP4 redundancy
   finding at the same time).
4. Keep the Moura bib entry + verification comment (§7 below); still correct, now doing double duty (Fix 2's
   precedent argument, and unchanged as context for the light experiment).
5. The co-visitation-network light experiment (§7.3) and the OOD/small-state caveats stay out of the main
   text — supporting/rebuttal material, not headline claims, per the same discipline that shaped everything
   above.

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

All four fixes now have real material; the open questions are about the single merged rewrite (§6), not
about whether to build anything further:

1. **Sign off on the merge-into-one-paragraph plan (§6)?** Or keep the pieces as separate insertions across
   §3/§5.3/§7 rather than consolidating? The consolidation is what turns five apologies into one confident
   claim (the thing your reviewers actually flagged), so I'd push back gently on keeping them separate, but
   it's your call.
2. **Exact final wording** is a prose-pass job (naturally bundled with the already-owed `OP4` sweep, since
   both touch the same scattered sentences) — do you want to draft it together now, or land the numbers-only
   version first and do the full prose polish as its own pass?
3. **California/Texas near-miss:** worth running later (there's GPU capacity now that the pipeline is
   proven), or is four states (including the one Istanbul-vs-Gowalla contrast) enough to make the point
   without over-investing in a motivation-only metric?
4. **The Silva et al. CSUR 2019 survey** (§7.2, verified, not currently cited): still just a documented
   backup, or worth a slot now that the mobility material is getting a real rewrite rather than one citation?

## 9 · Provenance

Original pass (citation-only framing) prepared 2026-07-07 from a 3-agent research pass (co-visitation
experiment, bib/PAPER_PLAN check, web credibility check). Revised the same day after the author's pushback,
grounded in a full read of `archive/REVIEW_PANEL.md` (all reviewers, not just Reviewer 1) and the relevant
sections of `REVIEW_GERMANO.md` + `review/review_v2.md` + `review/README.md`, plus the paper's own Table 3 /
§6 results numbers (`06_results.tex`, `tbl3_results.tex`) for the shortlist-enrichment arithmetic. Fix 4's
code (the `MTL_DUMP_VAL_PREDS` dump + `analysis/near_miss_distance.py`) was implemented as a background
agent the same session, independently verified (diff read line by line, `tests/test_training/` re-run
myself), and merged to main. The real A40 run landed as PR #59 (four states, 2026-07-07): recipe
independently cross-checked against `docs/studies/closing_data/archive/run_logs/run_catx_v17_n20.sh + archive/run_logs/run_catx_v17_seed0_5f.sh` (flag-set
diff empty), numbers cross-checked against the frozen board cells and internal consistency checks (OOD
count vs. OOD-miss count, top-1 rate vs. the known Acc@1 ceiling), audited and merged to main the same day.

## 10 · Draft prose (NOT yet applied to any `.tex` file — pending adversarial review, 2026-07-08)

This is a concrete before/after for every location the merged rewrite (§6) touches, mapped against the
**exact, complete inventory** of the five apology instances the June-29 advisor pass counted (`§3 ×2, §5.3,
§7 ×2`) — re-derived here from a full re-read of `01_introduction.tex`, `03_problem.tex`, `05_setup.tex`,
`07_discussion.tex`, `08_conclusion.tex`, confirming there are exactly five, no more, no fewer:
1. `03_problem.tex`: "A census tract is a neighborhood, not a radio cell... far too coarse to drive cell
   association or handover."
2. `03_problem.tex`: "We do not build or evaluate such a service, or any network system, here; the
   application is the reason the predictions matter, not a result we claim."
3. `05_setup.tex` §5.3: "A mobility-aware service acts on which region will be busy... stages content for
   the right region, or plans capacity at the census-tract level" (restates #1's list almost verbatim).
4. `07_discussion.tex`: "It remains motivation only, since we measure no service here... Turning that
   sketch into a measured result... is future work, not a claim we make."
5. `07_discussion.tex` (limitations list): "although a mobility-aware service is our motivation throughout,
   we do not build or evaluate one."

`01_introduction.tex` and `08_conclusion.tex` carry **no** apology instances (confirmed by full re-read) —
the intro's mobility framing (with the L0 Vielhaus cite already applied) and the conclusion (pure
results-summary) do not need touching. `02_related.tex` §2.2 already carries the Luca et al. citation for
the region-as-target formulation; Moura et al. is a *scope/motivation* precedent, not a competing-method
citation, so §3 (Problem and Tasks), not §2 (Related Work), is its home.

### 10.1 · §3 (`03_problem.tex`) — replaces apology instances #1 and #2

**Current** (last five sentences of the section):
> "These examples are scoped to what we actually measure. A census tract is a neighborhood, not a radio
> cell (the small coverage area a single base station serves); even the ten most likely tracts are far too
> coarse to drive cell association or handover. We therefore keep the motivation at the regional level:
> demand and load anticipation, content staging, and capacity planning. With that framing in place, we
> study how well one model can anticipate both the category and the region of the next visit. We do not
> build or evaluate such a service, or any network system, here; the application is the reason the
> predictions matter, not a result we claim."

**Proposed:**
> "These are recognized mobility-management uses at neighborhood granularity, not our own invention: a
> recent network-science study of tourist check-in mobility identifies the same class of uses, resource
> allocation, transportation logistics, and context-aware services, for analysis at exactly this
> granularity~\cite{moura2025mobilityaware}. A census tract is a neighborhood, though, not a radio cell (the
> small coverage area a single base station serves), so we scope our claims accordingly: demand and load
> anticipation, content staging, and capacity planning, not cell association or handover, which need finer
> resolution than a census tract gives. With that framing in place, we study how well one model can
> anticipate both the category and the region of the next visit; Section~\ref{sec:discussion} returns to
> what these predictions would let a service do, and by how much."

**Judgment call to flag for review:** this drops the explicit "we do not build or evaluate a service" line
from §3 entirely, deferring that honesty statement to §7 where real numbers appear. Is a reader misled in
the gap between §3 and §7 (through §4 Method and §5 Setup, neither of which claims a system result), or is
one clear statement at the point numbers appear (§7) sufficient and actually clearer than restating it
before any evidence exists?

### 10.2 · §5.3 (`05_setup.tex`) — replaces apology instance #3

**Current:**
> "A mobility-aware service acts on which region will be busy, not on a single rank position: it
> anticipates regional demand, stages content for the right region, or plans capacity at the census-tract
> level. A two-point shift in Acc@10 is therefore below the granularity at which such a service would
> behave differently."

**Proposed:**
> "As in Section~\ref{sec:problem}, a mobility-aware service acts on which region will be busy, not on a
> single rank position. A two-point shift in Acc@10 is therefore below the granularity at which such a
> service would behave differently."

This keeps the paragraph's actual job (justifying the TOST margin) and removes only the restated
content/capacity-staging list, which now lives once, in §3.

### 10.3 · §7 (`07_discussion.tex`) — replaces apology instances #4 and #5 (partially)

**Current usage sketch:**
> "A short usage sketch makes the motivation concrete. It remains motivation only, since we measure no
> service here. A service could treat the model's top-ranked next regions as an anticipatory set: a small
> list of areas worth preparing in advance. The next category then hints at what to prepare in them.
> Turning that sketch into a measured result, with its own service baselines and costs, is future work, not
> a claim we make."

**Proposed:**
> "A short usage sketch makes the motivation concrete, now with numbers rather than adjectives. A service
> could treat the model's top-ranked next regions as an anticipatory set: at California, for example, a
> shortlist of ten regions out of 8,501 candidates, about a tenth of a percent of the space, contains the
> true next region 65.66 percent of the time, more than five hundred times better than picking ten at
> random; at Alabama, ten out of 1,109 candidates capture it 69.81 percent of the time, a similar order of
> enrichment. Even when the top guess is wrong, it typically lands close by: across the four states where we
> measured it, the predicted region's centroid sits a median of 3 to 8 kilometers from the true one, a
> neighboring region rather than an arbitrary one. The next category then hints at what to prepare there.
> This remains motivation, not a measured service result: we report properties of the model's own
> predictions, not a system's cost or a service's outcome."

**Current limitations-list instance (#5):** "although a mobility-aware service is our motivation throughout,
we do not build or evaluate one; tying these predictions to a concrete service is the natural next study."

**Proposed:** unchanged. **Judgment call to flag for review:** is this a sixth restatement to also trim, or
is it legitimately different (a formal limitations-list entry, conventional and expected, versus the
argumentative-prose apologies elsewhere) and fine to keep as the one place this scope boundary is stated as
a limitation rather than an aside? I lean toward keeping it (limitations sections are supposed to state
limits plainly), but this is exactly the kind of line-drawing worth a second opinion on.

### 10.4 · Page-budget arithmetic (honest, not yet resolved)

Rough word counts: §3 nets **+20 words** (~+1 line) with the Moura citation added; §5.3 nets **-13 words**;
§7's usage sketch nets **+75 words** (~+3-4 lines) for the shortlist + near-miss numbers; §7's limitations
instance is unchanged. **Net: roughly +4 lines**, not free, not a blank check either. The paper compiled at
exactly 10 pages with no observed slack (§0). This is an open cost, not a hidden saving — flagged for the
review pass and for your decision, not resolved here.

### 10.5 · What is deliberately NOT touched, and why

- `01_introduction.tex`: no apology instances found; the L0 Vielhaus cite already does its job there.
- `08_conclusion.tex`: pure results-summary, already at its repetition budget per the OP4 finding
  (the region-scales-with-count claim already appears 3× across abstract/§7/§8); adding mobility framing
  here would be a sixth restatement of something, not a fix.
- `02_related.tex`: Moura et al. is a scope/motivation precedent, not a competing-method citation; forcing
  it into §2 would misplace it next to Luca et al.'s formulation-precedent role, which is answering a
  different question (why census tracts as the target unit) than Moura's (why this class of application).
- Abstract, Table 3 / Table 3 caption: untouched — the shortlist/near-miss numbers are motivation-layer
  interpretation, not the paper's audited headline results, and inserting them there would blur the
  claim-discipline line the paper has held everywhere else (`PAPER_PLAN.md §3`: numbers in results contexts
  trace to the board; these numbers are a derived, motivation-only reading of already-reported numbers).

### 10.6 · Review status

**Reviewed 2026-07-08** by three independent agents: a simulated MobiWac-specialist reviewer (the same
persona and rigor as `archive/REVIEW_PANEL.md`'s Reviewer 1, who gave this paper's original submission a
Reject on venue-fit), a general editorial advisor (factual accuracy, GLOSSARY/claim-discipline compliance,
redundancy), and a ripple-effect mapper (the rest of the paper). Full verdicts in §11. **Verdict: apply with
named fixes, not as drafted verbatim** — several real problems found, one of them (the Moura granularity
claim) genuinely serious. §10's text above is v1 and is superseded by §11's fixes; nothing has been applied
to any `.tex` file.

## 11 · Adversarial review verdict (2026-07-08)

Three independent reviews of §10's draft, run in parallel, each with full paper context (all 8 sections,
the abstract, tables/figures, GLOSSARY, PAPER_PLAN §3, the near-miss source data, and — for the MobiWac
reviewer — the actual Moura et al. PDF, read directly rather than trusting this plan's paraphrase).

### 11.1 · Headline verdicts

- **MobiWac-specialist reviewer:** moves from **Reject toward Weak Reject**, not Accept. The shortlist and
  near-miss numbers genuinely fix the "unfalsifiable decoration" half of the original complaint — real,
  falsifiable, source-traceable numbers replace adjectives. But the paper still has **zero evaluated
  mobility-management artifact**; the reviewer's first-ranked fix (a real, even lightweight, management-side
  proxy metric with an actual cost/hit-rate trade-off) still doesn't exist, only better-supported motivation
  does. And the Moura precedent — this draft's one attempt at an application-legitimacy argument beyond raw
  numbers — **does not survive a check against the primary source** (see 11.2, this is the most important
  finding below).
- **Editorial advisor:** **"apply with named fixes, not as drafted verbatim."** Confirms the five-apology
  count independently (exactly five, correctly located), confirms no core content is lost or diluted,
  confirms GLOSSARY/em-dash/spelling compliance is clean, confirms the §3→§7 forward-reference is honestly
  fulfilled — but fails the draft on one claim-discipline point (the Moura "exactly this granularity" line)
  and flags a missing in-distribution qualifier on the near-miss sentence.
- **Ripple-effect mapper:** mostly clean (abstract, conclusion, tables/figures correctly left untouched, no
  new cross-reference breaks) — but catches a real mechanical gap (the `moura2025mobilityaware` bib entry
  was drafted back in §7.4 but never actually added to `references.bib`; landing the `.tex` citation alone
  would break the paper's "0 undefined refs" state) and a real GLOSSARY violation (the near-miss metric is
  introduced with no definition and no reference-point floor, which `GLOSSARY.md §4` explicitly requires for
  every number in the paper).

### 11.2 · The serious finding: the Moura precedent does not hold up

Read in full by the MobiWac reviewer directly against the source PDF, not this plan's summary. Two
independent problems, either one sufficient to sink the claim:

- **Granularity mismatch, not match.** Moura et al.'s co-visitation graph nodes are **individual named
  POIs** (the Colosseum, at degree 131,269, is their own highest-degree example) — not census
  tracts/neighborhoods. A tract in our own task aggregates hundreds to thousands of such POIs into one
  prediction class. POI-level and tract-level are opposite ends of the same granularity axis, not the same
  point on it. The drafted §3 sentence ("recognized mobility-management uses at neighborhood granularity...
  for analysis at exactly this granularity") asserts the opposite of what a ten-minute check of the primary
  source shows.
- **Task-type mismatch, not match.** Moura et al. do no prediction at all — it is a retrospective, purely
  descriptive network-science analysis of a static graph (degree distribution, centrality, robustness to
  node removal). Precedent that structural analysis of check-in data is mobility-management-relevant does
  not transfer to "therefore a supervised next-region classifier is a legitimate proxy for the same uses,"
  which is the actual claim on trial.
- **Meta-level problem:** Moura's own applications paragraph (the one quoted) is itself written in the same
  unmeasured, conditional mood ("planners CAN better allocate...") this paper's rewrite is trying to escape.
  It is a discussion-section wish list, not a validated finding — citing it as "recognized... not our own
  invention" overstates what one paper's own unvalidated discussion section can license.
- This is flagged as a **direct recurrence** of a weakness the original archived review already named:
  *"the case-for-relevance rests on a single cherry-picked precedent and treats it as license rather than as
  a bar to clear"* (`archive/REVIEW_PANEL.md:123`). The rewrite doesn't fix that pattern, it re-executes it
  with a different paper — one that turns out to be at the wrong granularity and the wrong task type besides.

**Two options, not resolved here — an explicit decision point:**
- **(a) Narrow, don't drop:** keep the citation but remove the false equivalence — drop "exactly this
  granularity" and "recognized... not our own invention," reframe as "an active line of mobility/network-
  science research examines check-in-derived urban structure for infrastructure and service uses" without
  claiming granularity or task-type identity with our own work. (The editorial advisor's lighter-touch fix;
  sufficient to clear the claim-discipline FAIL, but the MobiWac reviewer would likely still read it as a
  weak, single-paper legitimation move — see the "house habit" note below.)
- **(b) Drop it as a scope-legitimizing citation entirely:** let Fixes 1 and 4 (the real, defensible,
  source-traceable numbers) carry the §3/§7 argument on their own, without leaning on an external precedent
  that doesn't actually match. The MobiWac reviewer's explicit recommendation.
- Independent of (a)/(b): the reviewer also notes the paper now runs the same "one citation licenses a
  design/scope choice" move **twice** (Luca et al. in §2 for the region-as-target formulation, Moura et al.
  in §3 for the application scope) — individually defensible, but a critical reviewer who catches the second
  one weak is primed to go back and discount the first one too.

### 11.3 · Required fixes (mechanical, low-risk, apply regardless of the Moura decision above)

1. **Add the in-distribution qualifier** to the §7 near-miss sentence. As drafted it says "when the top
   guess is wrong... a median of 3 to 8 kilometers" with no scope; OOD misses are 1.5-3x larger
   (`analysis/near_miss_RESULTS.md`) and the source material's own guardrail says in-dist/OOD must never be
   pooled or presented without the distinction. Three words fix it: "...for visits whose true region was in
   the training vocabulary, a median of 3 to 8 km..."
2. **Add the near-miss metric's definition and a reference point.** Per `GLOSSARY.md §4` ("undefined
   metrics: never give a number without its reference point"), every other number in the paper has a floor
   (majority-class, Markov-1, random-top-10); this one currently has none. Add a short parenthetical: what
   it is (great-circle distance between predicted and true region centroids, in-distribution visits only),
   and ideally one comparator if cheap (e.g., a typical inter-region centroid distance for the state).
3. **Fix "the four states" → "the four datasets" / "the four settings."** Istanbul is a city, not a state,
   in this paper's own careful vocabulary everywhere else (Advisor + ripple-mapper both caught this
   independently).
4. **Add the `moura2025mobilityaware` BibTeX entry** (drafted in §7.4 above) to `references.bib` in the same
   edit as the `.tex` citation — it is not yet in the file; landing the citation alone breaks the paper's
   "0 undefined refs" state.
5. **Re-verify the page-budget cost by actually inserting and recompiling**, not by word-count estimate.
   The advisor independently measured the real compiled PDF's body-text density at 8-11 words/line (not the
   ~20 words/line this plan's §10.4 assumed) and recomputed the net cost at **roughly 10 lines, not 4** —
   given the paper is at exactly 10 pages with no observed slack, this needs a real compile check before any
   claim about "the room this frees up," not an assumption in either direction.

### 11.4 · Two structural fixes to the §3/§7 disclaimer split

1. **The §3→§7 gap is a real exposure window, not a safe simplification.** Removing "we do not build or
   evaluate a service" from §3 entirely (replacing it with a forward reference to §7) means a reader who
   reads §3 through §6 encounters **zero** explicit no-service-measured statements until the *second*
   paragraph of §7 — and even there, it now comes *after* the shiny new numbers, the wrong rhetorical order
   for a paper trying to look less apologetic, not more oversold. Fix: keep one compact clause in §3, e.g.
   extend the forward-reference sentence to "...Section~\ref{sec:discussion} returns to what these
   predictions would let a service do, and by how much; we build and evaluate no such service here."
2. **§7 now states the disclaimer twice, back-to-back**, once closing the rewritten usage sketch and once
   opening the (unchanged) limitations-list bullet a few sentences later — a new, local redundancy even
   though the global count went down. Fix: reword the limitations-list entry so it doesn't repeat verbatim,
   e.g. "a concrete service evaluation, with its own baselines and costs, is deferred to future work."

### 11.5 · Optional / lower-priority polish (author's call, not blocking)

- **The spatial-compactness gap (MobiWac reviewer's sharpest new point, worth taking seriously even though
  optional):** "500x over chance" is a *relative* enrichment claim; it says nothing about whether the top-10
  shortlist is spatially compact (a real, actionable cluster) or scattered across a whole state. The reviewer
  notes this is likely cheap to close: the top-10 predicted indices are already in the `MTL_DUMP_VAL_PREDS`
  parquet dumps, so a bounding-box or centroid-spread statistic for the shortlist itself (not just the top-1
  miss distance) could probably be computed offline, no retraining needed. This is the one item that could
  meaningfully strengthen the case rather than just defend it — flagged for a decision, not done here.
- Trim the untouched §3 opening sentence (the "staging content... planning capacity" example, lines 28-30)
  so the same triad isn't stated twice in one paragraph after the rewrite (ripple-mapper finding A) — also
  helps the page-budget arithmetic.
- "network-science study" risks a momentary misread as "networking science" for this specific audience;
  consider "mobility-science study" instead (ripple-mapper finding C).
- The paragraph's flagship example swaps from California (shortlist) to a 4-state pool excluding CA/TX
  (near-miss) without acknowledging the swap (ripple-mapper finding D) — cheap to close with one clause.
- The intro's opening handover analogy (correctly left untouched) still primes a reader to expect a
  handover-adjacent contribution three sections before the rest of the paper disowns exactly that claim —
  not fatal, not something this rewrite is obligated to fix, but worth knowing it's still there in tension
  with everything downstream of it.

### 11.6 · The bigger, unresolved strategic question

The MobiWac reviewer's ceiling verdict — Weak Reject, not Accept, because there is still no evaluated
management artifact — points at a fork this plan has not resolved and should not resolve unilaterally:

- **Stay at "much better supported motivation," accept Weak Reject as the realistic ceiling** for a paper
  whose settled ruling is motivation-only (no measured network/service result). Apply §11.3/§11.4's fixes,
  make the Moura call (11.2), and treat this as done — a real, honest, meaningfully improved position, just
  not one that converts venue-fit from MAJOR to a non-issue.
- **Reopen the settled "no measured network result" ruling** to build the one thing the reviewer says would
  actually close the gap: a real, even lightweight, management-side proxy (a simulated hit-rate-vs-
  preparation-cost trade-off, or a paging/staging-cost curve over the predicted region distribution). This
  is explicitly what `MOBILITY_PLAN.md`'s ladder marks **L3, banned, "reopening is an author-only decision"**
  — a genuinely bigger, riskier commitment (new experiment design, new code, likely more GPU time) than
  anything built so far, not a small extension of Fix 4.
- Not resolved here on purpose — this is exactly the kind of call that belongs to the author, not something
  a plan document should quietly decide by drafting toward one answer.

## 12 · v3 — the KEEP-Moura reframing, adversarially certified (2026-07-08)

> **Author decision (2026-07-08): the Moura citation STAYS.** It is the most on-subject POI/LBSN paper
> available, peer-accepted at the venue itself, and from a group (Loureiro, UFMG) the community knows. The
> task was therefore not keep-or-drop but "find the framing that survives the primary-source check that
> killed v1." This section records that framing, certified by a second adversarial round (a reframe critic
> reading the Moura PDF directly, a web-verification agent, and a page-budget auditor measuring the real
> compiled PDF).

### 12.1 · Two discoveries that strengthen the keep decision

1. **Moura et al. is venue-local to MobiWac itself, not just the sister conference.** Verified against the
   MobiWac 2025 program page: the paper was presented in the MobiWac 2025 symposium program (Tuesday, Oct
   28), published in the joint IEEE MSWiM'25 volume. The venue-precedent argument is stronger than §7.2
   assumed.
2. **Moura's own reference [1] is the Silva et al. CSUR 2019 survey** — the exact broad companion §11's
   review suggested pairing it with. The survey+instance two-legged pattern is therefore the source's own
   citation lineage, not a construction of ours. (Both share senior author Loureiro; acceptable per the
   critic — CSUR is the standard survey of the subfield — but do NOT add a third Loureiro-group cite.)

### 12.2 · The four anchors and their verdicts (critic, PDF-verified)

Every quote below was re-verified verbatim against the PDF before judging. Verdicts:

| Anchor | What it claims | Verdict | Notes |
|---|---|---|---|
| **A** (intro ¶1, ~1.5 lines) | check-in mobility analysis is venue-local design input for mobility-aware services | ADOPT-WITH-EDIT, optional | first sacrifice if trimming; makes 3 total Moura cites |
| **B** (§3 tail, the main one) | visits concentrate on hubs → demand concentrates; Moura ties this to planning/adaptive strategies; we are the predictive complement | ADOPT-WITH-EDIT (edits mandatory) | the "reads past structure / we predict two properties of the next visit" sentence is what makes v2/v3 survivable where v1 died |
| **C** (§7 final ¶, 1 line) | the descriptive line itself names ML as its next step; we move in that direction | ADOPT-WITH-EDIT | "calls" plural died (it is ONE hedged sentence in their conclusion); referent fixed |
| **D** (dataset characterization in-text) | our own check-in graphs show the same structure | **KILL for main text; rebuttal kit only** | Texas hole (5/6), not Table-1-class numbers, re-opens the granularity axis v1 died on |

Clauses of the v2 draft that **died under the PDF check** and are fixed in the v3 text below: "crowds and
demand concentrate too" in our voice (Moura's method explicitly discards time — co-visitation degree cannot
show crowding, which is simultaneity; must be attributed: "argues that... as well"); "shows why anticipation
matters" (Moura never uses predictive framing; the anticipation inference is ours and must stay in our
voice); "directly" (overstates their "contribute to the broader discussion" hedge); "we predict the next
one" (GLOSSARY next-place blur risk; must be "two properties of the next one"); "calls in the mobility
literature" plural (one hedged sentence).

### 12.3 · Final prose (v3 — critic-corrected, floor-anchored; the apply-ready text)

**C1 — intro ¶1 (Anchor A, optional, author call).** Replace the current Vielhaus sentence tail:
> "The same logic, acting before the user moves instead of after, is already established at the network
> level, where predicted handovers let cellular services adapt in advance~\cite{vielhaus2022handover}; at
> the city scale, check-in mobility analysis is likewise treated as a design input for mobility-aware
> services~\cite{moura2025mobilityaware}. We ask the predictive, city-scale version of the question."

**C1b — intro ¶1, zero-word INV6 closure (new, from the web round).** Hang the verified proactive-caching
citation on the existing concrete example (the cite supports the example, not our framing — exactly what
REVIEW_GERMANO #57 / INV6 asked for): "...staging the right content in the area a user is heading
to~\cite{bastug2014edge}, or planning capacity there before demand arrives..."

**C2 — §3 tail (Anchor B, the main edit).** Replace the current final five sentences:
> "These uses have support in the mobility literature. Urban computing has long built city services on
> location-based social network traces~\cite{silva2019urbancomputing}. A recent analysis of tourist
> check-in mobility finds that visits concentrate on a few hub places, argues that crowds and demand
> therefore concentrate as well, and ties this structure to infrastructure planning and adaptive
> strategies~\cite{moura2025mobilityaware}. That analysis reads the structure of past visits; we predict
> two properties of the next one, which indicates where demand is heading before it lands. A census tract
> is a neighborhood, not a radio cell (the small coverage area a single base station serves), so we scope
> our claims to neighborhood-level preparation: demand and load anticipation, content staging, and capacity
> planning, not cell association or handover. We build and evaluate no such service here;
> Section~\ref{sec:discussion} quantifies what these predictions would give one."
("quantifies" is valid ONLY because C4 lands in the same commit; if ever split, write "returns to".)

**C2b — §3 same-paragraph trim (mandatory pair of C2; kills the triad duplication AND the float-spill
risk).** Replace the current 82-word "The motivation is practical... before the load arrives." passage with:
> "The motivation is practical: the category says what kind of place comes next, which indicates what a
> user will want; the region says where, which indicates where to prepare."

**C3 — §5.3.** Replace the service sentence with:
> "As in Section~\ref{sec:problem}, a mobility-aware service acts on which region will be busy, not on a
> single rank position."
(The TOST-margin sentences that follow stay untouched.)

**C4 — §7 usage sketch (floor-anchored; replaces the current sketch paragraph).**
> "A short usage sketch makes the motivation concrete, now with numbers rather than adjectives. A service
> could treat the model's top-ranked next regions as an anticipatory set: at California, a shortlist of ten
> regions out of 8{,}501 candidates, about a tenth of a percent of the space, contains the true next region
> 65.66 percent of the time, more than five hundred times better than picking ten at random; at Alabama,
> ten out of 1{,}109 capture it 69.81 percent of the time. Misses are typically local as well: on the four
> datasets where we measured it (Alabama, Arizona, Florida, and Istanbul), when the top guess is wrong and
> the true region was seen in training, the predicted region sits a median of 3 to 8 kilometers from the
> true one, measured between region centroids, where two randomly drawn candidate regions of the same map
> sit a median of 20 to 241 kilometers apart; visits to regions never seen in training are rarer and miss
> by more. The next category then hints at what to prepare there. This remains motivation, not a measured
> service result: we report properties of the model's own predictions, not a system's cost or a service's
> outcome."

**C5 — §7 limitations clause reword** (kills the back-to-back §7 redundancy):
> "a concrete service evaluation, with its own baselines and costs, remains the natural next study."

**C6 — §7 final paragraph append (Anchor C):**
> "...are natural directions to explore. Structural analysis of check-in networks has itself named machine
> learning as a next step~\cite{moura2025mobilityaware}; learning over check-in structure, as this study
> does, moves in that direction."

### 12.4 · The floor number is real and committed

`analysis/near_miss_floor.py` (+ the Floors section appended to `near_miss_RESULTS.md`): random-pair
inter-centroid distance over the model's exact region vocabulary (all GEOIDs matched at every state; 200k
pairs, seed 0). P50: Alabama 170.67 km, Arizona 120.32, Florida 241.22, Istanbul 20.45 — so the model's
misses (8.13 / 8.05 / 7.04 / 3.16 km) land **~21x / ~15x / ~34x / ~6x closer than chance** on the same
candidate set. This satisfies the GLOSSARY reference-point rule that §11.3 item 2 flagged, with numbers, not
a promise. C4's "20 to 241 kilometers" clause is backed by this artifact.

### 12.5 · Page budget: CONFIRMED TO FIT (measured, not estimated)

The auditor measured the real compiled PDF (9.9 words/line) and — the finding that settles §11.3 item 5 —
located the document's one real slack: **~41 footnotesize lines (~5.2 column-inches) of white space at the
end of the references on page 10**, which all downstream growth drains into. With the C2b trim taken: net
+114 words ≈ +11-12 body lines + 2-3 bib entries ≈ ~24-29 of those 41 lines consumed → **fits inside 10
pages with ~1.5-2 column-inches to spare**. One hard condition: **C2b is mandatory** — without it, upstream
growth (+4 lines before §6) risks slipping the Table III float anchor, the one configuration that spills to
page 11. Six further optional trims (~-7 lines) are catalogued in the auditor's report if headroom is ever
needed. Final verification remains an actual recompile at apply time.

### 12.6 · New verified citations (web round; all primary-source verified with quotes)

- **`silva2019urbancomputing`** — Silva, Viana, Benevenuto, Villas, Salles, Loureiro, Quercia, "Urban
  Computing Leveraging Location-Based Social Network Data: A Survey," ACM Computing Surveys 52(1), Art. 17,
  39 pp., 2019, DOI 10.1145/3301284. Scope verified beyond the abstract (author-hosted PDF read): §4
  taxonomy includes Urban Mobility; §3.2 "Analytics and Development of Services and Applications"; ~115
  citations (Semantic Scholar). Anchor B's first leg.
- **`bastug2014edge`** — Bastug, Bennis, Debbah, "Living on the Edge: The Role of Proactive Caching in 5G
  Wireless Networks," IEEE Communications Magazine 52(8), pp. 82-89, 2014, DOI 10.1109/MCOM.2014.6871674.
  ~1,000+ citations. Quotable tie: "peak traffic demands can be substantially reduced by proactively serving
  predictable user demands via caching." **Closes INV6** (the author-pending review item #57) at zero word
  cost (C1b). Optional modern complement if ever wanted: Yu et al., IEEE T-ITS 22(8) 2021, DOI
  10.1109/TITS.2020.3017474 (mobility-aware proactive edge caching, ~300 citations).
- **Moura BibTeX correction:** publisher metadata (IEEE PDF byline + Crossref) renders the second author
  **"Andre L. L. Aquino"** — no accent, no "de" (dblp normalizes differently; follow the publisher). §7.4's
  entry is otherwise correct. Citation count 0 (published Oct 2025 — expected).
- **Venue scan bonus (optional, not required):** Kouam, Viana, Beiró, Ferres, Pappalardo, "Beyond
  Aggregates: A Fine-Grained Analysis of Individual Mobility and Traffic Dependencies," MSWiM 2025, pp.
  201-210, DOI 10.1109/MSWiM67937.2025.11309071 — individual mobility + network-traffic joint modeling from
  cellular data. Could honestly pluralize Anchor A ("mobility analysis is likewise treated as a design
  input... \cite{moura2025mobilityaware,kouam2025beyond}") and diversify beyond the Loureiro group at zero
  word cost; needs its own primary-source framing check before citing. Otherwise the 2022-2025 venue scan
  is null (nothing else on POI/check-in prediction for services) — the venue-local cites we have are what
  exist.

### 12.7 · Shortlist-compactness: specced, A40-dependent, not yet run

Confirmed: the committed JSONs hold per-miss distances only; the top-10 predicted indices exist only in the
A40 rundirs' `fold{N}_reg_val_preds.parquet`. The auditor wrote the 5-line spec (inputs: the parquets +
`load_region_centroids`; statistic: per-visit centroid spread + bounding-box diagonal of the 10 predicted
regions, in-dist/OOD separate, bare percentiles; output: `analysis/shortlist_compactness_<state>.{json,md}`).
This is the remaining cheap strengthening item — it converts "500x enrichment" from a relative-lift claim
into a spatial-actionability claim ("the shortlist is also compact"), the sharpest residual counter a
networking reviewer has left against C4. Runs in CPU-minutes on the A40 against the existing dumps; no
retraining.

### 12.8 · Updated shipping recommendation

1. **Ship B (C2+C2b), C3, C4, C5, C6** — the certified core. **A (C1) recommended too** given the author's
   pro-Moura direction and the confirmed budget; accept that Moura then appears 3 times (the critic's
   sacrifice order if that ever needs undoing: D first — already killed — then A, then C; never B).
2. **C1b (Bastug on the staging example)** — zero-cost, closes INV6.
3. **Same commit must carry:** `moura2025mobilityaware` + `silva2019urbancomputing` (+ `bastug2014edge`)
   into `references.bib` (grep-verified absent today; the citation without the entry breaks the 0-undefined
   state), the C2b trim, and a real recompile + page-count check.
4. **Next A40 session:** shortlist-compactness (12.7) — and optionally CA/TX near-miss if ever wanted;
   the Istanbul near-miss seed-rigor caveat from §11.3 stands (seed-0 vs the paper's 4-seed Istanbul bar
   elsewhere; either note it or top it up when the H100/P1 lane runs).
5. **The L3 fork (§11.6) remains the author's** — everything above stays inside the settled motivation-only
   ruling.

## 13 · Green-light gate (2026-07-08): GATE GREEN, with named apply-time fixes

An independent advisor re-verified §12 end to end (A-L checklist: prose blocks, bib mechanics, the floor
artifact, the budget logic, settled-decisions consistency — spot-checking Moura quotes against the PDF
again) and a citation scout answered the two open author questions. **GATE: GREEN** — nothing red; the
following named fixes MUST be carried into the apply commit (they are one-word/one-anchor edits, listed here
so the applier cannot miss them):

1. **C2 opener:** the C2b trim removes the concrete-uses list that C2's "These uses have support..."
   pointed at — open C2 with **"These preparations have support in the mobility literature."**
2. **C4:** drop **", now"** ("makes the motivation concrete, with numbers rather than adjectives" — the
   "now" is revision-note leakage with no referent for a first-time reader). Prefer **"about 3 to 8
   kilometers"** (Alabama's median is 8.13; "about" is airtight).
3. **C5 anchor (critical):** replace ONLY the final clause "tying these predictions to a concrete service
   is the natural next study" with "a concrete service evaluation, with its own baselines and costs,
   remains the natural next study" — **KEEP "we do not build or evaluate one"** in the limitations bullet
   (a whole-bullet replacement would delete the limitations list's only no-service statement).
4. **Bib mechanics:** (a) §7.4's `% verified:` comment still carries the killed v1 granularity framing —
   rewrite it to the v3 framing (venue-local MobiWac/MSWiM'25 instance; attributed descriptive anchor)
   before pasting; (b) draft `silva2019urbancomputing` + `bastug2014edge` BibTeX blocks with `% verified:`
   comments (fields verified in §12.6); (c) confirm the Moura booktitle against the PDF header — "2025
   International Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSWiM)",
   with NO "IEEE" inside the container title (Crossref/PDF agree; the §7.4 draft says "2025 IEEE...").
5. **At apply time:** update CLOSER_HANDOFF §0's "31 cited refs" count (→34, or 35 with C7); the Istanbul
   seed-rigor caveat (§12.8.4) travels with C4.

### 13.1 · C7 (NEW, scout-endorsed, recommended): the Song 2010 feasibility clause

The one genuinely uncovered mobility-science leg: nothing cited establishes that the next visit is
predictable *at all* (`luca2021mobilitysurvey` covers formulation/methods and itself cites Song rather than
establishing predictability). The canonical primary source, metadata verified:

```bibtex
% verified: Science 327(5968), pp. 1018-1021, 2010, DOI 10.1126/science.1177170
% (web-verified 2026-07-08). Abstract, verbatim: "By measuring the entropy of each
% individual's trajectory, we find a 93% potential predictability in user mobility
% across the whole user base." Cited for the feasibility premise only (mobility is
% predictable in principle); the 93% figure is for cell-tower trajectories and is
% deliberately NOT transplanted into our prose (check-in streams are sparser).
@article{song2010limits,
  author  = {Chaoming Song and Zehui Qu and Nicholas Blumm and Albert-L{\'a}szl{\'o} Barab{\'a}si},
  title   = {Limits of Predictability in Human Mobility},
  journal = {Science},
  volume  = {327},
  number  = {5968},
  pages   = {1018--1021},
  year    = {2010},
  doi     = {10.1126/science.1177170},
}
```

**Placement + wording** (intro ¶1, ~+9 words): current "A natural and useful question is what a person will
do next, because a service that can anticipate the next move can prepare ahead of time..." becomes:
> "A natural and useful question is what a person will do next; individual mobility is highly predictable
> in principle~\cite{song2010limits}, and a service that can anticipate the next move can prepare ahead of
> time instead of reacting after the fact."

**Rejected by the scout (recorded so nobody re-litigates):** González et al. 2008 (Nature) — the precursor;
Song is the direct quantification and one of the pair suffices under this budget. Alipour et al. MSWiM 2019
(next-location prediction for edge caching) — venue-family but WLAN campus traces, and its role is already
covered by Bastug + Vielhaus. LBSN-predictability one-offs (MDPI Entropy 2016 etc.) — non-canonical.

### 13.2 · Kouam co-citation: optional, reworded only

Kouam et al. MSWiM 2025 (abstract verified via HAL: joint modeling of individual mobility and traffic,
cellular XDRs) can honestly join Moura ONLY in a one-level-up sentence ("individual-level mobility analysis
increasingly drives the design of mobility-aware services, from tourist check-in networks
\cite{moura2025mobilityaware} to mobility-traffic dependencies in cellular networks \cite{kouam2025beyond}");
a check-in-specific sentence breaks with it. Adds venue-family currency, no new argumentative leg — skip
first under budget pressure. Note: its IEEE pages were unverifiable at scan time (Xplore/HAL fetch blocks);
verify before any citation.

### 13.3 · The two author questions, answered

- **"Is there a better paper than Moura?" NO — confirmed irreplaceable.** The final bounded scan (venue
  family back through 2019 + PerCom adjacents) found nothing else combining venue-local + POI/check-in/LBSN
  subject + mobility-aware-systems framing. The nearest miss (Alipour MSWiM'19) fails the subject match
  (WLAN traces). Moura is the anchor; Silva 2019 is its broad leg; Song 2010 fills the feasibility leg;
  Bastug 2014 backs the staging example; Vielhaus 2022 holds the network side. **The set is complete.**
- **"Is L3 worth doing?" Recommendation: NO for this cycle.** (a) A weak L3 is worse than no L3: a credible
  service evaluation needs cost assumptions (staging cost, hit value, cache size) — each one fresh attack
  surface for the exact reviewers being defended against, and the archived AC warned it changes the paper's
  genre; (b) most cheap L3 shapes (hit-rate vs. prepared-set-size curves) are precisely the banned
  coverage-curve-in-disguise the settled ruling names; (c) the deadline may be past and P1 owns any
  recovered compute; (d) the achievable motivation-side gain inside the ruling is now captured (enrichment
  + near-miss + floor + compactness pending) — the residual Weak-Reject→Accept delta needs a REAL system
  study, which the near-miss dump infrastructure has already de-risked as a **follow-up paper / thesis
  chapter**, where it can be done properly with its own baselines. The ruling stays closed unless the
  author explicitly reopens it.

## 14 · APPLIED + post-apply review (2026-07-08) — the venue-fit verdict trajectory completes

**The edit set is in the paper.** PR #60 (shortlist compactness, A40-6) was audited and merged the same day;
its result (in-dist shortlist spread P50 2.86-7.53 km, ~7-32x tighter than the random-pair map scale) was
folded into C4 alongside the near-miss + floor numbers. All blocks applied (C1+C1b+C7 intro, C2b+C2 §3, C3
§5.3, C4-adapted+C5+C6 §7) + 4 bib entries. **Compile: 10 pages, 0 undefined, 0 bibtex warnings, 0
overfull, 37 rendered refs** — the refs-end slack absorbed the growth exactly as §12.5 measured.

**Post-apply review (3 independent lenses on the applied text):**

- **Venue-fit re-review** (the original Reject persona): **"Weak Reject, now genuinely at the borderline —
  no longer a relevance reject I would argue for in committee."** Trajectory: plan = Reject → draft =
  "toward Weak Reject" → applied = borderline. Its remaining sharpest attack (per-user prediction vs.
  aggregate-demand uses) was answered same-day with the §3 "aggregated over users those predictions point
  to..." clause; its RecSys-hook complaint (the intro's "recommendation and urban analysis" aside) was cut.
  What would move it further is only L3 (closed this cycle, §13.3).
- **Claim/number audit:** every number digit-verified (PASS across the board); all §13 named fixes confirmed
  applied; **one FAIL found and fixed** — the post-certification C4 compactness clause ("cluster within a
  median spread") read as containment when the statistic is mean-distance-to-center (actual bbox extent
  10.9-32.1 km); reworded to the exact statistic ("sit, on average, about 3 to 8 kilometers from the
  shortlist's own geographic center (the median across visits, computed from region centroids)").
- **Flow cold read:** two must-fixes (both applied): the abstract's "that model" antecedent bug (the region
  claim read as beating the *category* model — pre-existing, now "the dedicated region model") and the C4
  double "about 3 to 8 kilometers" mis-parse (now split into two labeled sentences, "The miss distance is
  similar:" marking the echo deliberate, "while" replacing the garden-path "where"). High-value polish also
  applied: Song sentence split (restores the ¶1 hook beat), the triple before/after echo cut, Moura tagged
  "read from past traces" (descriptive), "Such preparation has support..." antecedent fix, the silva
  sentence fused into a general-to-specific ladder, "would give such a service", the ";. not cell
  association" disambiguation. Verified-fine items left alone (cross-section escalation, disclaimer
  placement, the single-seed parenthetical).

**Recompiled after fixes: 10 pages, 0 undefined, 0 warnings, 37 refs — unchanged.** Remaining open:
the L3 fork (author-only, §13.3), the Istanbul dk_ovl/n=20 top-up path for the two geometry metrics
(rides A40-2/H3 or P1), and P1 itself (the science verdict-changer, unaffected by all of this).
