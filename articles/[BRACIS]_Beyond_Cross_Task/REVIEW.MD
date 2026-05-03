# Review of First Full Draft

## Overall judgment

The paper already has the **right scientific spine** for BRACIS: substrate task-asymmetry first, classic MTL tradeoff
second, and the Alabama per-visit decomposition as the mechanism anchor. That is the correct story from the study.

The current manuscript is **not yet submission-grade prose**. The main problems are not the core results; they are:

- weak paper framing in the introduction and related work
- inconsistent novelty positioning
- too much internal/protocol language leaking into reader-facing text
- several over-strong interpretations
- table/figure presentation that is still closer to lab notes than to a polished conference paper

If this were submitted in the current state, I think reviewers would say:

- the empirical story is interesting
- the writing is not yet sharp enough
- the attribution between prior work, substrate, and architecture is still under-explained
- the presentation is not yet polished to BRACIS main-track standard

## Priority findings

- **High: the prior-work framing is under-cited and therefore not defensible as written.** The introduction says “Two
  recent works frame the bottleneck,” but only DGI and HGI are actually cited, not the two prior papers being
  summarized. This happens in `src/sections/intro.tex:6-10`. The same problem repeats in
  `src/sections/related.tex:19-21`, where the earlier MTL paper and the follow-up embedding paper are described without
  proper citations. This is one of the clearest reviewer attack surfaces. In double-blind, those papers should still be
  cited in third person.

- **High: the manuscript still spends too much headline budget on internal protocol language instead of the scientific
  problem.** “Leak-free per-fold transition priors” appears in the abstract (`src/samplepaper.tex:57-66`),
  introduction (`src/sections/intro.tex:10`), related work (`src/sections/related.tex:21`), setup (
  `src/sections/setup.tex:11`), and discussion (`src/sections/discussion.tex:10`). This is not a paper contribution. It
  is a protocol detail that should be stated once, cleanly, in setup/method. Right now it dilutes the story and reads
  like internal correction history, not conference prose.

- **High: the novelty positioning is over-defensive and partially self-defeating.** The introduction says: “we propose
  no new balancer, head, or attention variant, and do not claim Check2HGI as our model” in `src/sections/intro.tex:10`.
  This goes too far. It protects against over-claiming, but it also makes the reader ask why the paper is interesting at
  all. The study canon is subtler: the paper’s contribution is the controlled empirical reattribution and the
  task-asymmetric substrate result. The text should say that clearly, without sounding as if the paper contributes
  almost nothing.

- **High: the fairness question ‘is this transfer or just a more complex network?’ is not answered clearly enough in
  prose.** The science does answer it through matched-head STL ceilings, but the paper does not explain that logic
  sharply enough for a reviewer scanning quickly. The key sentence in setup is only one line long at
  `src/sections/setup.tex:16`. This needs to become explicit and memorable: same heads, same substrate, only the sharing
  architecture changes, therefore the delta is architectural rather than head-capacity inflation.

- **High: the FL-only ablation panel is overgeneralized into a five-state structural claim too strongly.** The mechanism
  section says the cost is “therefore structural” and that the structural reading “transfers” because the sign is
  consistent across five states in `src/sections/mechanism.tex:25-27`. That is too strong. FL-only negative ablations
  are evidence consistent with a structural cost, not definitive proof of a five-state architectural law. The discussion
  repeats the same jump in `src/sections/discussion.tex:10`. This needs weaker wording.

- **High: the external-baselines section risks overreaching relative to what is actually comparable.** The category side
  says the model exceeds POI-RGNN by `+27` to `+33` pp and then says the gap is substrate-driven in
  `src/sections/results.tex:36-42`. That is too confident. The folds are not directly comparable to the published
  POI-RGNN setup, and the “substrate-driven” interpretation is not identified by an external-baseline experiment. This
  section needs tighter caveats and less triumphalist tone.

## Major issues

- **Major: the abstract is still too protocol-heavy and too dense compared with accepted BRACIS papers.** In the BRACIS
  2025 samples I checked, abstracts follow a much cleaner four-step rhythm: problem, gap, method, result/insight. Ours
  at `src/samplepaper.tex:54-68` front-loads internal evaluation protocol (“user-disjoint folds,” “leak-free per-fold
  transition priors,” matched-head nuance) before the reader has even absorbed the paper question. Those details matter,
  but not at this density in the abstract. The abstract should sound more like a conference paper and less like a
  statistical closure note.

- **Major: the introduction still lacks a strong applied motivation paragraph.** It explains the tasks, but it does not
  yet sell why POI category and region prediction matter in urban computing, recommendation, mobility support, or LBSNs
  with the same strength as the prior CBIC/CoUrb papers. `src/sections/intro.tex:4-10` is technically correct but too
  cold. BRACIS papers usually need one clearer motivation beat before the method pivot.

- **Major: acronym discipline is below conference standard.** Some acronyms are spelled out correctly, but the paper
  still assumes too much familiarity too early. The most visible issues are MTL/STL/TOST/Acc@10/MRR/POI usage across
  `src/sections/intro.tex`, `src/sections/setup.tex`, and `src/sections/results.tex`.
  Alabama/Arizona/Florida/California/Texas are also reduced to AL/AZ/FL/CA/TX very quickly. This is fixable, but it
  needs a systematic pass.

- **Major: the paper still mixes conceptual names and code names too aggressively.** `next_gru`, `next_stan_flow`, and
  `task_weight = 0` appear often in reader-facing text, especially in `src/sections/method.tex:27-34`,
  `src/sections/setup.tex:16`, and `src/sections/mechanism.tex:29-32`. Some code names are unavoidable, but right now
  the balance is off. Main text should privilege conceptual labels, with code identifiers secondary.

- **Major: the architecture figure is doing too much.** The current caption at `src/sections/method.tex:27-31` suggests
  a dense explanatory figure with many implementation details. Combined with the current manuscript’s heavy use of code
  names, this likely makes the visual harder, not clearer. Your concern here is valid: the figure should become simpler,
  more conceptual, and visually cleaner.

- **Major: the table design is still too compressed and not yet publication-quality.** `src/tables/mtl_vs_stl.tex:1-21`
  and `src/tables/external.tex:1-36` rely on `\resizebox{\textwidth}{!}{...}` and pack too much into dense cells. The
  “delta + p” in a single cell is especially hard to scan. Your concern that the columns feel too close is correct. The
  tables need a readability redesign, not only small spacing tweaks.

- **Major: the paper does not yet use a clear best-value convention.** In the external table, only the MTL row is
  bolded (`src/tables/external.tex:28`), which visually privileges the proposed model rather than the best value per
  column. That is not ideal for a rigorous conference paper. Best values should be bold according to a declared
  convention, and the proposed model can be marked separately if needed.

- **Major: the bibliography still contains at least one clearly non-final entry.** `li2025rehdm` in
  `src/references.bib:229-235` is not in acceptable final form: authors are just surnames, and the entry is incomplete.
  There may be other entries that need a final pass too, but this one is enough to say the bib is not ready.

## Medium issues

- **Medium: the related-work opening is slightly misframed for the actual task pair.** `src/sections/related.tex:4` says
  “prior multi-task work on the POI category and next-POI pair,” but the paper is about category and region. This may
  seem small, but it makes the section feel less precise than the study actually is.

- **Medium: some prose is still too lab-internal or developmental.** “Single-modality variants collapsed one head in
  development, so two streams are the default” in `src/sections/method.tex:16` reads like research log language, not
  paper language.

- **Medium: the discussion sentence ‘empirically vacuous on the harder task’ is too sharp for this draft.**
  `src/sections/discussion.tex:6` uses language that the article-side planning docs had already tried to control
  carefully. The claim can be made, but the current wording invites pushback and makes the paper sound combative rather
  than precise.

- **Medium: the sklearn split caveat should not be in the main paper.** Your concern is correct. The last limitation in
  `src/sections/discussion.tex:10` is too developer-facing. It belongs in repository documentation or appendix
  reproducibility notes, not in the main narrative of a BRACIS paper.

- **Medium: the conclusion mispositions the CoUrb connection slightly.** `src/sections/discussion.tex:12` and
  `src/sections/conclusion.tex:3-4` frame a future fused substrate as if this paper had not already inherited part of
  that logic. Your concern is valid: the continuity with CoUrb should be described more carefully, since Check2HGI is
  not conceptually orthogonal to temporal/spatial/category structure in the way the current wording suggests.

- **Medium: the external-baseline caption has a duplicated word and a slightly messy narrative.**
  `src/tables/external.tex:3` says “ReHDM and faithful STAN faithful runs,” which is plainly an editing mistake.

- **Medium: the category-side contribution may now be under-sold.** The manuscript tries so hard not to overclaim
  Check2HGI that `src/sections/method.tex:11` and `src/sections/intro.tex:10` may cause the reader to miss that the
  substrate result is actually the main positive finding of the paper.

## Small issues and polish

- `src/sections/intro.tex:10` uses state acronyms before giving the full names in prose. The setup table fixes this
  later, but first-mention expansion would still read better.
- `src/sections/results.tex:24` says “monotonic shrink,” then immediately says TX breaks it. The intended meaning is
  clear, but the phrasing can be cleaner.
- `src/sections/results.tex:31` says FL is “the strongest paired result in the paper.” That is arguable, but it reads a
  bit promotional.
- `src/sections/conclusion.tex:3-4` is very dense. It compresses too many results into one paragraph and would benefit
  from a cleaner hierarchy of result importance.
- `src/tables/mtl_vs_stl.tex:3` says the Wilcoxon tests are one-sided at `n = 20`, but AL is reported with `p = 0.036`,
  which reads as a two-sided negative-direction result in the surrounding prose. The table and statistical narration
  should stay perfectly aligned.

## Assessment of your current concerns

- **“Many abbreviations are not explained on first use.”** I agree. This is a real problem and needs a systematic
  acronym pass.
- **“MTL is used without explicit acronym introduction.”** I agree. Same for STL and some metric/test abbreviations.
- **“We should not use leak-free language.”** I mostly agree. Keep the protocol fact once in setup/method, but remove
  the branding from abstract, introduction, related work, and discussion.
- **“The introduction needs more explanation of the problem and POI importance.”** I agree strongly.
- **“We cite ‘Two recent works’ but only reference one.”** I agree strongly. This is a serious citation problem.
- **“We use state siglas without full names.”** I agree. Fix at first mention.
- **“We reference STL without explaining it.”** I agree.
- **“If we say we propose no new head, is that true?”** Partially agree. The current wording is too absolute. The reg
  head is an adapted implementation component, even if it is not a claimed paper contribution.
- **“Is Check2HGI not a contribution?”** Partially agree. The draft currently over-deflates it. The paper should not
  pivot into “new model architecture” mode, but it should more clearly claim the check-in-level substrate result as one
  of the paper’s real contributions.
- **“The architecture image is not good.”** I agree.
- **“The tables are too close/confusing.”** I agree strongly.
- **“Best values should be bold.”** I agree, with a declared and neutral convention.
- **“Are we really answering sharing transfer vs just a more complex network?”** Partially. The science does, but the
  prose does not explain that clearly enough.
- **“The references need checking one by one.”** I agree. At least the ReHDM entry is visibly not ready.
- **“The sklearn limitation should not be discussed.”** I agree.
- **“The conclusion misreads the Paiva fusion direction.”** I agree.

## Benchmark against accepted BRACIS papers

I also compared the draft against accepted papers from the BRACIS 2024 and BRACIS 2025 Springer proceedings you
downloaded. I sampled:

- **BRACIS 2025:** *A Categorical Kalman Filter for Human Activity Recognition*
- **BRACIS 2025:** *Enhancing Graph Neural Networks for Multi-target Activity Prediction Through Multi-task Learning and
  Knowledge Distillation*
- **BRACIS 2025:** *Enhancing Framingham Cardiovascular Risk Score Transparency Through Logic-Based XAI*
- **BRACIS 2024:** *A Contrastive Objective for Training Continuous Generative Flow Networks*

These are not identical to your topic, but they are close enough in style and conference expectations to be useful
benchmarks.

### What accepted BRACIS papers do consistently

- They open with a **clear application problem first**, before the methodological nuance. The reader understands why the
  task matters before being asked to care about the architecture.
- They state the **gap** in plain language. The gap is usually one sentence, not a long defensive paragraph.
- They describe the contribution in **positive terms**. Even when they are careful, they do not spend precious intro
  space saying what they are *not* claiming.
- Their abstracts are **lighter**. They mention protocol only when necessary, and they do not try to settle every
  fairness objection in the abstract itself.
- Their related-work sections are **shorter and more decision-oriented**. They cite enough to be credible, but they do
  not overload the reader with too many technical branches unless those branches are central to the paper’s comparison
  logic.
- Their conclusions are **clean summaries**, not compressed audit logs.

### Where this draft is still below that benchmark

- **The paper still reads too much like a study closure document.** Accepted BRACIS papers read like a finished research
  argument. This draft still sometimes reads like it is defending the research process itself.
- **The intro is more intricate than necessary.** The accepted papers tend to introduce the task, the gap, and the main
  idea with less friction. Your current introduction is scientifically richer, but rhetorically heavier.
- **The contribution list is stronger scientifically than rhetorically.** It contains the right claims, but the
  sentences are too long and too numerically dense for first contact. Accepted papers usually keep the list easier to
  parse, then move the precise statistical closure to results/setup.
- **The paper is too eager to pre-defend itself.** The line “we propose no new balancer, head, or attention variant” is
  a good internal guardrail, but not good conference rhetoric in its current position and phrasing.
- **Some section prose is more codebase-aware than reader-aware.** Accepted papers may include model details, but they
  usually translate implementation choices into conceptual language before naming the exact component.

### What this means in practice

The good news is that your paper is **not** weaker scientifically than those BRACIS papers. In some ways, the study
discipline here is stronger. The problem is that the manuscript does not yet present the work with the same economy and
confidence.

If you want the paper to feel like a strong accepted BRACIS paper rather than a technically correct draft, the next
rewrite should aim for:

- a simpler abstract
- a more applied first page
- cleaner positive contribution framing
- less protocol language outside setup
- shorter, more declarative paragraphs in introduction, related work, and conclusion

## Rewrite priorities

1. **Rewrite the introduction.** Add a stronger problem/motivation opening, cite the two prior papers properly, define
   acronyms on first use, and remove the defensive “we do not claim Check2HGI as our model” language.
2. **Clean the novelty positioning.** State clearly that the contribution is a controlled empirical reattribution and a
   task-asymmetric substrate result, not a new balancer paper, but do not undercut the paper’s value.
3. **Strip internal protocol language from reader-facing sections.** Keep the fold-built transition-prior fact in
   setup/method only.
4. **Tighten the claims that currently overgeneralize from FL-only ablations and external comparisons.**
5. **Redesign the tables and simplify the architecture figure.** This is necessary for conference readability, not
   cosmetic preference.
6. **Do a full bibliography pass.** Fix incomplete entries, verify all self-citations, and make the reference list
   uniformly professional.
7. **Rewrite discussion and conclusion for better continuity with the broader study arc.** Especially the
   CoUrb-to-Check2HGI relation and the future-work positioning.

## Bottom line

The paper is **scientifically promising but rhetorically underpowered** in its current form. The main risk is not that
the results are weak; it is that the prose still oscillates between:

- over-defensive
- overly internal
- occasionally over-strong

The next pass should not chase more results. It should make the paper:

- easier to read
- cleaner in attribution
- stricter in claims
- much more polished in presentation

If that rewrite is done well, this can become a strong BRACIS submission.
--------

# Other authours consernes

- The text is start to getting traction but I am felling that we are stuck in the five state in some sections, while we
  should advocated to
  the whole article, we should focus on the inovations and how this inovations was made, the data to support it should
  be somthing more avegrave instead of locallize a exemple the Per-visit context decomposition image is a exclent image,
  but didn't we coud did a average of all the state instead of focusing in just one ? wouldntd this be more strongers ?
- As I have discuss before I belive we need to tail more the story and don't be afraid of huge change in how the sotry
  is going or what we have written, the importance is to have a good story
- Remadeir of keep my style and tom, form the previus articles and not use AI Watermarking/Fingerprints

--------

# Independent review (Claude — second pass after Codex)

I read the rendered PDF, every section file, every table file, the architecture figure, the per-visit
figure, the bib, the prior CBIC and CoUrb articles for voice calibration, AGENT.md, PAPER_STRUCTURE.md,
and the study north star. I did not have the BRACIS 2024/2025 proceedings PDFs locally, so I rely on
Codex's benchmarking for genre comparisons and add the points below from the manuscript itself.

I agree with the bulk of Codex's findings. Rather than repeat them, this review concentrates on what I
think they missed or under-weighted, then answers your six concerns directly, then proposes a concrete
rewrite order.

## The credibility-blocking issue Codex missed

At Florida, the Markov-1-region floor reaches Acc@10 = 65.05 ± 0.93 (Table 4 / `src/tables/external.tex:15`),
and the proposed MTL row reaches **63.27 ± 0.10** on the same metric (Table 3 / `src/tables/mtl_vs_stl.tex:13`,
Table 4 / `src/tables/external.tex:28`). The proposed model sits below a one-step Markov chain on its
primary next-region metric at the headline state. The discussion never says this. AGENT.md `§6` flagged
"FL `next_region` Acc@10 sits in a Markov-saturated regime" as a required limitation, but
`src/sections/discussion.tex:10` lists five limitations and skips this one.

A reviewer who reads the tables side by side will notice in under a minute. If you do not surface it
before they do, it is a paper-killer in review. The fix is honest: add one paragraph to §5.2 (after
Table 3) and one limitation entry to §7. Frame it the way the AGENT.md already does — Markov saturation
at FL on Acc@10 is real, the MRR axis tells a different story, MTL is below the Markov floor on Acc@10
but the substrate ablation already showed that POI-stable embeddings are the natural fit for the region
head and the MTL backbone pays an additional structural cost on top of that. State the numbers, do not
spin them.

This is the single most important change. It costs you nothing scientifically (the data already says it)
and it converts a reviewer attack surface into a credibility signal.

## The framing question you keep half-asking

The current paper carries two stories in parallel: (1) substrate task-asymmetry as the load-bearing
phenomenon, and (2) the classic MTL tradeoff as the secondary observation. The title commits to the
first. The contribution bullets in `src/sections/intro.tex:14-15` give roughly equal weight to both. The
results section dedicates 5.1 to substrate, 5.2 to MTL-vs-STL, 5.3 to Δm, 5.4 to externals. The
discussion §7 also splits attention between them. By the time the reader gets to §6.1's per-visit
counterfactual, the substrate story has been re-introduced four times and never quite landed.

The empirics decide the framing. Substrate is paper-grade significant at all five states. MTL is positive
at four of five (AL is small-significantly negative). The substrate story is stronger and earned a
five-state pass; the MTL story is at four-of-five with one negative. Commit to substrate as the headline
phenomenon and demote MTL-vs-STL to its consequence. Concretely:

- §1 contribution C1 = "controlled five-state ablation showing the substrate is task-asymmetric." That is
  the contribution. The numbers are the evidence. The current bullet (`src/sections/intro.tex:14`)
  reads as if the asymmetry is the contribution; the contribution is the controlled ablation that
  identified it.
- §1 contribution C2 should be reframed from "classic MTL tradeoff" (a finding) to "we measure the MTL
  tradeoff under the load-bearing substrate, separating substrate effects from architectural effects."
  That is what is novel here, and it is what the matched-head ceiling actually buys you.
- §5 should put the substrate ablation and the per-visit counterfactual closer together, even at the cost
  of breaking the current ordering. The mechanism (§6.1) is what makes §5.1 interpretable; reading them
  35 pages apart fragments the story.

This addresses your concern about "tailing more the story." The story tightens once you commit to "the
substrate is the lever, here is the mechanism, the architecture pays a known cost on top of it" rather
than running substrate and MTL as parallel lines.

## The cross-attention encoder co-adaptation point is under-used, not under-promoted

I disagree with the impulse to promote §6.3 to a full contribution; AGENT.md `§2` is explicit that the
methodological note must not move into the abstract, and that constraint is reasonable for a 15-page
double-blind submission with C1+C2 already carrying the empirical weight. The fix is not promotion but
visibility. The encoder-frozen isolation observation deserves a clear sentence in §3.3 (training
protocol) before the reader hits §6.3. Right now `src/sections/method.tex:50` mentions the caveat in one
clause near the end of the training-protocol paragraph. A reader scanning method-side does not see
that the paper has thought about why the architectural decomposition is non-trivial.

A clean two-sentence statement in §3.3 — "Under cross-attention, zeroing one task weight does not
isolate the silenced encoder; we use encoder-frozen isolation throughout, and Section 6.3 develops the
note." — costs you nothing on page budget and makes the §6.3 finding feel earned rather than tucked away.

The architecture figure (`src/figs/architecture.tex`) currently shows the K/V cross-feeds as small
gray dashed arrows. Those arrows are the load-bearing object the §6.3 finding rests on. If you keep the
figure, recolor the cross-feed in a way that signals "this is the thing." Black, thicker, with a brief
inline label like "K/V cross-feed (encoder co-adaptation, §6.3)" would do it.

## The Per-visit decomposition is single-state — and you already have a path out

Your concern about the figure is right but the framing matters. The bar chart at AL is excellent; the
problem is not the chart but that the mechanism reading rests on one state. The fix is not "average
across states" because the data does not exist at the other four. The fix is to actually run the
counterfactual at one more state.

AZ is the cheapest second state to add (3.3K users, ~10 min on H100 per the AGENT.md timings). The
counterfactual is a single STL run with POI-pooled vectors. If AZ shows a similar
canonical-vs-pooled-vs-HGI ordering and a comparable per-visit share (say within ±10pp of 72%), the
mechanism reading goes from "single-state evidence" to "two-state replicated." That is a real upgrade
for reviewers, not a cosmetic one. If AZ shows a different split, that is also valuable — it tells you
the mechanism is state-dependent, which the paper currently has no way to assess.

This is a one-day pre-submission run, not a future-work note. I would do it before the deadline rather
than ship it as a limitation.

## Direct responses to your six concerns

1. **"We are stuck in the five state in some sections."** Partially right. The substrate ablation is a
   genuine five-state result and should stay disaggregated; that is the strongest claim in the paper.
   What feels stuck is that §5.2 reports five MTL-vs-STL numbers state-by-state without ever
   synthesising the pattern. One synthesis sentence — "the MTL trade is sign-consistent across all
   five states; the magnitude varies non-monotonically with cardinality" — closes the gap without
   collapsing the per-state numbers. The §6.2 ablation panel (FL only) is genuinely state-local and
   should be honest about that.

2. **Per-visit figure averaged across states.** Cannot be done with current data; recommend running AZ
   instead (see above). If compute does not allow, change the §6.1 framing from "explains the
   asymmetry" to "consistent with the asymmetry, single-state instance." The current framing
   over-promises.

3. **"Focus on the innovations and how this innovations was made."** Agreed. The current contribution
   bullets list findings, not innovations. Innovations made: (a) a controlled substrate ablation with
   matched-head STL ceilings that decouples embedding effects from head-capacity effects, (b) a
   per-fold leak-free transition prior protocol that lets the MTL-vs-STL contrast remain valid at
   n=20, (c) a per-visit counterfactual via POI-pooled embeddings as a substrate decomposition tool,
   (d) the encoder-frozen isolation protocol for cross-attention MTL. These are protocol-level
   innovations, not new layers, and that is OK if the paper says so plainly.

4. **"Tail more the story, don't be afraid of huge change."** I would consider one structural change:
   merge the current §5.1 (substrate ablation) with §6.1 (per-visit counterfactual) into a single
   §5.1 "the substrate is task-asymmetric and we know why" subsection. The paper currently splits
   the finding from the mechanism by ten pages. They should land together.

5. **CBIC/CoUrb voice.** The current draft drifts toward AGENT-doc voice — "leak-free per-fold
   transition priors," "matched-head ceilings," "sign-consistent" — language that reads as internal
   hygiene. CBIC's intro lands the central hypothesis in plain prose ("a standard hard
   parameter-sharing MTL architecture will face significant limitations"), and CoUrb does the same in
   Portuguese. The BRACIS draft should imitate this tone: state the question, state what you found,
   defend the finding. The matched-head and leak-free protocol facts go once in §3 or §4, not in the
   abstract, intro, related work, and discussion.

6. **AI watermarking.** The current draft has occasional AI-pattern phrasings: "the conventional MTL
   framing where joint training transfers signal to lift both heads is empirically vacuous on the
   harder task" (`src/sections/discussion.tex:6`), "the textbook tradeoff" used five times,
   numerically dense single sentences in the abstract. The fixes are mostly already on Codex's list;
   I would add one rule: any sentence with three numbers and two p-values should split into two
   sentences.

--------

# Independent review (Codex — final pass on the current PDF)

I re-read the current section files, tables, bibliography, and the rendered `samplepaper.pdf` dated
2026-05-03. Some earlier review items are now resolved in the manuscript itself. Most notably, the
Florida Markov-floor issue is no longer hidden: it is now stated explicitly in `src/sections/results.tex:26`
and carried into `src/sections/discussion.tex:10`. The points below are therefore limited to what I
still think is blocking or fragile in the **current** version.

## New high-priority issues still present

- **High: the paper's core manipulated variable is still under-specified in the main text.** The title,
  abstract, and headline claim all depend on Check2HGI, but `src/sections/related.tex:9` and
  `src/sections/method.tex:9` only give a conceptual gloss and then defer "the recipe" to
  supplementary material. For a paper built around a new embedding granularity, reviewers need enough
  in the main manuscript to know exactly what changed from HGI: what graph/view is constructed at the
  check-in level, what the positive/negative contrast is, how temporal context enters, and how the
  per-region vectors used by the region head are derived. As written, the central independent variable
  remains partly a black box.

- **High: the paper uses the same phrase, "matched-head STL ceiling," for materially different
  estimators across sections, and the prose does not warn the reader.** The substrate table reports the
  single-seed five-fold substrate axis, while Tables 3 and 4 report multi-seed pooled ceilings for the
  architectural contrast. This is visible immediately at Florida: category is `63.43` in
  `src/tables/substrate.tex:14` but `67.16` in `src/tables/mtl_vs_stl.tex:13`; region is `69.22` in
  `src/tables/substrate.tex:29` but `70.62` in `src/tables/mtl_vs_stl.tex:13`. The underlying science
  may still be correct, but the presentation currently invites a reviewer to think the tables
  contradict each other. Either unify the reported ceiling convention or explain very explicitly that
  Section 5.1 and Section 5.2 are using different estimands for different purposes.

- **High: the caption of `src/tables/substrate.tex` is internally inconsistent and should be treated as
  a correctness issue, not a style issue.** It says "Check2HGI at AL/AZ/FL is multi-seed mean ± seed
  sigma over seeds `{0, 1, 7, 100}`", but the Florida value shown in the table is `63.43`, which is
  the single-seed matched-head STL figure used on the substrate axis, not the multi-seed ceiling used
  elsewhere (`67.16`). At minimum the caption is wrong. A reviewer who notices this will start
  distrusting the entire evidence table.

## Major issues

- **Major: the manuscript still contains "camera-ready" promise language inside the submission
  artifact.** `src/tables/external.tex:33` says "ReHDM at CA/TX deferred (camera-ready, compute
  budget)." That is bad submission rhetoric. Reviewers do not score future intentions; they score the
  current paper. If CA/TX ReHDM is out of scope, state that once as a limitation and stop there. Do
  not tell the reviewer that the comparison will be supplied later.

- **Major: the paper is already exactly 15 pages, so the next fix must be structural, not additive.**
  The current PDF is at the hard cap. Any attempt to address the missing Check2HGI specification or the
  ceiling-reporting confusion by only adding prose will likely make the paper worse. The next pass
  should cut low-yield protocol wording first and spend that space on self-contained method detail and
  cleaner table logic.

## Medium issues

- **Medium: C3 still reads more like a methodological caution than a conference-level contribution.**
  I agree with the earlier note that the encoder-frozen isolation point is real and useful, but in the
  current paper it is argued briefly and without a substantial experimental panel of its own. If page
  pressure forces prioritization, this is the first contribution I would demote from the intro bullet
  list into supporting method/discussion prose.

- **Medium: the bibliography still contains a visibly unfinished entry in the rendered PDF.**
  `li2025rehdm` currently renders as "Li, Gu, Yao, Zhou, Zhu, Zhao, Du" with `p.~0343`, which looks
  like placeholders rather than a finished IJCAI reference. This is not just a `.bib` hygiene issue;
  it is visible in the paper the reviewer reads.

## What I would actually do, in order

1. **Add the Markov-1-at-FL paragraph to §5.2 and §7.** This is the highest-priority fix and the
   shortest. Two paragraphs, total. Without this the paper has a credibility hole at the headline
   state.
2. **Run AZ per-visit counterfactual** if compute permits. If not, soften §6.1 framing to
   "single-state instance, consistent with."
3. **Cite Silva 2025 and Paiva 2026** in third person at `src/sections/intro.tex:6` and
   `src/sections/related.tex:19`. This is the second-most reviewer-visible problem and Codex flagged
   it correctly.
4. **Strip "leak-free" branding from abstract / intro / related / discussion.** Keep the protocol fact
   in setup once. Codex covered this; I am repeating because it is a high-value low-effort fix.
5. **Rewrite the contribution bullets** to lead with what you built and ablated (innovations), not
   what you found (findings). Concrete sentences in your prior CBIC voice are easy templates.
6. **Make the encoder-frozen isolation visible in §3.3** without moving it to the abstract.
7. **Redesign Tables 3 and 4** to drop `\resizebox` and split "Δ (pp), p" into two columns. Declare a
   bolding convention in the table notes ("Best per column in **bold**; the proposed model is
   marked with †") and apply it consistently to T4 — currently the MTL row is bolded for being ours
   even when STAN-on-HGI is the best per column.
8. **Redraw the architecture figure** with shared/task-specific visual hierarchy and the K/V
   cross-feed highlighted as the load-bearing object.
9. **Acronym pass.** Expand STL, MTL, MRR, Acc@10, TOST, POI on first use. State acronyms at first
   mention should give the full name (Florida, California, Texas, Alabama, Arizona), then the
   abbreviation. Codex covered this; mentioning so it appears on the action list.
10. **Bibliography pass.** ReHDM (`src/references.bib:229-235`) is a clear must-fix; check every
    other entry for completeness while you are there.

Steps 1, 3, 5, 6 are the highest-leverage rhetorical fixes. Step 2 is the highest-leverage scientific
fix. Steps 7–10 are presentational and Codex covered most of the territory.

## On the broader structural question

You asked me to be free to judge. The honest take is that this is a stronger paper than the prose
currently lets it be. The substrate task-asymmetry is a genuinely interesting finding for the POI-MTL
literature; the per-visit counterfactual is a clean mechanism probe; the encoder co-adaptation note
under cross-attention is the kind of methodological observation that reviewers remember. The work is
not weaker than typical accepted BRACIS papers. The prose is.

If you fix the Markov-1 honesty gap and trust the substrate-first framing without hedging, this can
land as a confident submission with a clear story. If you keep oscillating between substrate and MTL
as co-equal contributions and leave the Markov-1 gap unaddressed, you risk a reviewer summary that
reads "interesting empirics, defensive prose, two open methodological questions" — which is the
borderline-reject zone the acceptance rate makes harder than it has to be.

I would not change the title. "Substrate Carries, Architecture Pays" already commits. The body should
match the title's confidence.
------
Authour considerations:

 We are dicussing in the intro about our methodology, like: `a length-9 window` and `7-way decision      
  shaped by per-visit` shouldnt we let it for the methodology section ? and be more focus on the problematic   
  in this begin of the introductionl, If you agree and belive that is areal problem take this same think and   
  expanded it for the rest of the article, analysing if we are not puting information out of place