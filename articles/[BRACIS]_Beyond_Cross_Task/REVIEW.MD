# Review status

This file tracks **what is still pending** on the BRACIS submission. The full
review history (six rounds of Codex + Claude reviews and the author's running
notes) is preserved in git log; the items below are the ones that were not
fully closed in the last pass and are worth picking up before the camera-ready
deadline. The PDF currently compiles to **15 pages** (LNCS hard cap); any new
edit must hold or reduce the count.

--------

# Pending — pre-submission

Round 6h (Codex critical external-style review, 2026-05-04) surfaced four
items that I (Round 6i, same date) re-audited against source files. **Two of
the four (P1, P4) are valid but lower-severity than Codex assigns; one (P2)
is misdiagnosed — the article is internally correct, the stale working
memos are the actual problem; one (P3) is a strategic framing concern that
matches my own prior critical verdict.** Severity and fix path corrected
below; full Codex Round 6h review preserved at the bottom of this file as
the original critique, and Round 6i (my critical audit of that critique)
is appended after it.

## P1. User-disjoint vs transductive substrate — **CLOSED Round 6j Batch 2+3 (2026-05-04)**: parenthetical added to abstract + intro

**Verified.** Abstract (`samplepaper.tex:59`) and intro (`intro.tex:10`)
use "user-disjoint folds" without the transductive qualifier. **However**,
the qualifier IS disclosed in three other sections: `setup.tex:11` (in the
same paragraph as the user-disjoint statement, three sentences later),
`method.tex:19`, and `discussion.tex:6` (limitations). So the disclosure is
not hidden — Codex's "the paper wants the rhetorical benefit while burying
the qualifier" framing is too strong.

**Fix (cheap):** add one parenthetical to the abstract — *"five-state
Gowalla ablation under user-disjoint downstream folds (transductive at the
substrate level; see \S\ref{subsec:setup-data})"* — and a similar
phrase to the intro setup sentence. ${\sim}10$ extra characters total; no
page-budget hit. Closes the issue without running new experiments.

## P2. POI-RGNN inconsistency — **CLOSED Round 6j Batch 1 (2026-05-04)**: working memos marked SUPERSEDED; article unchanged

**Codex's three-source contradiction is real**, but the location of the fix
is wrong. After re-reading the canonical sources:

- `docs/studies/check2hgi/baselines/next_category/poi_rgnn.md` (canonical
  baseline doc) explicitly states: *"We use `StratifiedGroupKFold(5,
  seed=42)` on userid stratified by target_category... cold-user; train
  and val users are disjoint."* — **the reproduction IS user-disjoint.**
- `docs/studies/check2hgi/results/RESULTS_TABLE.md §0.6` lists the
  canonical numbers as **34.49 / 31.78 / 33.03** for FL / CA / TX.
- The article (`external.tex`, `results.tex`) inherits both: user-disjoint
  protocol claim and the canonical numbers. **Article-side is consistent
  with the canonical sources.**
- `STATISTICAL_AUDIT.md:184` text *"did not use user-disjoint folds"* is
  **stale** — file dated 2026-05-02, before the audit cycle closed. It
  reflects an earlier protocol claim that was superseded.
- `comparison.md` JSON values (33.35/30.71/32.08) are pre-bugfix per the
  explicit `COORDINATOR_HANDOFF.md:355` note: *"JSON values diverge
  (33.35/30.71/32.08) — JSONs may be pre-bugfix. Trusted §0.6."*

**Fix (the right one):** mark `STATISTICAL_AUDIT.md` and `comparison.md`
as superseded by `RESULTS_TABLE.md §0.6` (or rewrite/archive them).
Article needs **zero changes**. The Codex recommendation to "until the
protocol and numbers are reconciled, the doubling claim is not stable" is
overstated — the article's claim IS stable; the stale internal memos are
not. **HIGH severity for repo-hygiene if the anonymous code release
exposes these working memos; LOW severity if it exposes only `src/`.**

## P3. Architecture as contribution vs failed add-on — VALID, STRATEGIC

This matches my own prior critical verdict (see Round 6 — critical paper
review at the end of this file). The proposed cross-attention MTL backbone
is empirically dominated by single-task baselines and by a 1-step Markov
chain at most states. The current paper has tension between "we propose
this architecture" and "this architecture loses on every region task".

This is **not a pre-submission blocker** — the paper does honestly disclose
the negative; the title "Architecture Pays" already telegraphs it; the
conclusion's three open lines (asymmetric routing, encoder enrichment,
region-side dynamic prior) all explicitly target the architecture's
weakness. But the underlying tension is real and remains unresolved at the
reframe level. Two strategic options (deferable to a resubmission cycle if
this version is rejected):

1. Drop the MTL framing entirely; reframe as a substrate-axis paper with
   matched-head STL ceilings as the headline.
2. Lead with §6.3 encoder-isolation as the methodological contribution,
   demote the MTL backbone to "the architecture our methodological note
   was developed against".

**For this submission cycle: accept the borderline status.** The paper is
honest about the negative; reviewers who value rigor + honesty will accept,
those who require positive algorithmic advance will reject. P3 is not
fixable in two days.

## P4. Per-visit mechanism over-emphasis — **CLOSED Round 6j Batch 2+3 (2026-05-04)**: AZ paired-Wilcoxon ran (p=0.0312 each at n=5, 5/5 positive); §6.1 upgraded from descriptive to inferential replication

**Verified.** The 72/64 claim appears in abstract (line 68), intro C1
(`intro.tex:15`), conclusion (`conclusion.tex:4`), and Fig. 2 caption. AZ
is descriptive-only per `mechanism.tex:11` (no paired test). FL/CA/TX have
no mechanism replication.

**Cleanest fix is already documented as D3 (deferred camera-ready):** run
the AZ paired Wilcoxon on the existing fold-level deltas. One-day run,
zero new training compute, converts AZ from descriptive to inferential.
Resolves the over-emphasis concern without rewording.

**If D3 is not run pre-submission**, second-best fix: add the phrase
*"(inferential at AL, descriptive replication at AZ)"* to the abstract's
72/64 sentence. ${\sim}45$ extra characters; honest scoping; preserves the
contribution.

--------

# Deferred — camera-ready window

These items are not blocking submission; they are the natural polish path
for the camera-ready cycle (or follow-up work) once the reviewer feedback
arrives.

## D1. Architecture figure: cross-attention label collision — **CLOSED Round 5g (2026-05-04)**

Resolved with option 1 (drop `(Eq.~\ref{eq:crossattn})` from the box title);
the cross-ref to the equation is preserved in the figure caption. See
"Round 5g — closure" at the bottom of this file for the evaluation that
considered an alternative TikZ rewrite (`/Users/vitor/Downloads/architecture.tex`)
and the reasoning for keeping Version A.

## D2. 5-state per-visit decomposition figure

The §6.1 figure is currently a two-panel AL + AZ render (commit shipped
2026-05-04). The 5-state launcher and renderer are in place
(`scripts/run_pervisit_counterfactual_FL_CA_TX.sh`,
`src/figs/render_per_visit_5state.py` — auto-falls-back to AL + AZ if
FL/CA/TX JSONs are missing). Running the launcher (3-6 H100-hours for the
nine cells: 3 states × {canonical, pooled, HGI}) lifts the §6.1 mechanism
from "AL paired-Wilcoxon + AZ descriptive replication" to a 5-state
replicated panel, which converts the substrate-axis claim into 5-state
inferential evidence. Strictly stronger than the AL+AZ two-panel; deferred
on compute, not on intent.

## D3. AZ paired-Wilcoxon on the per-visit counterfactual folds

`mechanism.tex` reports the AZ replication as "consistent with the AL
ordering, with both components positive in matched-head means" and
explicitly says "we do not run a paired test across the AZ folds and report
the second state as descriptive replication, not inferential confirmation."
If a reviewer asks for inferential support, running the paired Wilcoxon on
the AZ fold-level deltas would convert "descriptive" to "two-state
replicated" without changing the numbers. One-day pre-camera-ready run.

--------

# Audit-trail summary (what was closed across the six rounds)

Each item below was checked against the source files; full review text is in
the prior commits on this branch.

- **Round 1 (initial Codex review).** Resolved: prior-work citations
  (silva2025mtlnet, paiva2026courb), "leak-free" branding stripped from
  abstract/intro/related/discussion, over-defensive novelty wording removed,
  fairness sentence in setup made explicit, Markov-1-at-FL honesty paragraph
  added, external-baseline caveats added, code names (`next_gru`,
  `next_stan_flow`) replaced by conceptual labels in body prose, table
  spacing relaxed (`\arraystretch=1.1` across T1--T4), best-per-column
  bolding convention declared in T4 caption.

- **Round 2 (Claude — second pass).** Resolved: Markov-1-at-FL surfaced in
  §5.2 + §7; substrate-first framing reflected in title/contributions;
  F-arch K/V cross-feeds recoloured bold black; AZ counterfactual run
  integrated; §3.3 made the encoder-frozen note visible.

- **Round 3 (Codex — final pass on PDF).** Resolved: Check2HGI
  under-specification (§3.1 expanded); ceiling-estimator labelling clarified
  in §4.3; substrate caption corrected; "camera-ready (compute budget)"
  promise language removed from external.tex.

- **Round 4 (Codex — methodological-faithfulness round).** Resolved:
  scale-conditional optimiser disclosure in §3.3 (FL/CA/TX vs AL/AZ recipe
  split); transductive substrate disclosure in §4 + §7; ReHDM coverage
  statement aligned with what the table shows; AZ replication wording
  softened to descriptive; "supplementary material" deferrals replaced by
  anonymous-code-release pointers; intro method-detail trimmed; abstract
  updated for AL+AZ replication.

- **Round 5 (Claude — meticulous numerical + storytelling).** Resolved:
  H1 Markov-1 cross-state pattern reframed (no longer FL-only); H2 FL
  saturated/Pareto-positive reconciled in §5.3; H3 substrate Table 1(a)
  protocol switched to seed-42 throughout so cell-cell = Δ; H4 AZ HGI
  drift between Table 1(a) and §6.1 reconciled at $28.99 \pm 0.51$;
  H5 "faithful STAN" prose rewritten; M1 $\lambda_{\text{cat}}$ sensitivity
  pointer; M2 head-invariant probes pointer; M4 Acc@10 cost ranges verified
  clean; per-visit shares verified clean.

- **Round 5b (Author considerations).** Resolved: A1 per-visit figure
  upgraded to two-panel AL+AZ (5-state deferred to D2); A2 doubling-vs-
  external claim added to §5.4 + §7 with substrate attribution preserved;
  A2-followup applied as **Option B** — Check2HGI promoted to C0
  contribution, abstract opens with "We introduce Check2HGI", §3.1 expanded
  with the 3-boundary MI loss equation and embedding-level corruption note;
  conclusion opens "We introduced Check2HGI"; §7 scope sentence pre-empts
  the genericity question; A3 future-work refined.

- **Round 5c (BRACIS proceedings style audit).** Resolved: B1 architecture
  figure simplified from 12+ boxes to 6; B2 contribution bullets unbolded;
  B3 voice trim ("load-bearing" 5×→2×, "textbook tradeoff" 4×→3×);
  B4 math expansion under Option B; B5/B6/B7 (table density / §7
  separate Limitations / reference depth) verified appropriate;
  B8 punch-list collapsed.

- **Round 5d (Self-audit of post-Round-5/5b/5c edits).** Resolved:
  numerical-consistency cross-check across abstract / intro / discussion /
  conclusion / mechanism (range $+14$--$29$ pp; AL/AZ shares); flagged H6
  Markov-1 cross-state numbers absent from any visible table; H7 GETNext
  named without `\cite{}` after `yang2022getnext` was bib-deleted; M3 stale
  "Total entries: 30" comment; M6 HMT-GRN removal documented.

- **Round 5e (Bibliography web-audit + figure render audit).** Resolved
  HB1 `capanema2022poirgnn` → `capanema2023poirgnn` (Ad Hoc Networks
  138:103016, 2023, DBLP-verified); HB2 `li2025rehdm` UNVERIFIED tag
  dropped, full author names + pages 3081--3089 + DOI added (IJCAI 2025
  proceedings page); MB1 `zeng2019mhape` DOI chapter `_22` → `_21`;
  10 other bib entries verified clean (caruana, chen2020hmrm, cho2011gowalla,
  huang2023hgi, liu2023famo, luo2021stan, maninis2019attentive,
  senushkin2023aligned, vandenhende2022mtl, velickovic2019dgi);
  paiva2026courb / silva2025mtlnet trust-by-provenance (CBIC/CoUrb local
  source files). Architecture-figure-resize over-shrink flagged (N2);
  per-visit figure unchanged from AL-only flagged (F-2). M3 stale comment
  fixed to "Total active entries: 16" with full provenance for each
  removed/restored entry.

- **Round 5f (Self-audit of post-Round-5e edits).** Resolved: N1 BibTeX
  `@Comment{...}` wrapper applied to commented-out `lim2022hmtgrn` block
  (build log now zero errors); N2 architecture figure resize closed at
  `0.60\linewidth` (tested 0.65/0.62 — both overflowed to 16 pp; 0.60
  holds at 15 pp with all box labels legible); F-2 per-visit figure
  regenerated as two-panel AL+AZ at $0.72\linewidth$; H6 Markov-1 inline
  numbers absorbed at `results.tex:26`; H7 `yang2022getnext` restored to
  bib with provenance comment; one residual cosmetic (D1 above) noted;
  5-state per-visit launcher in place (D2).

- **Author considerations across rounds.** Voice / tone alignment with
  CBIC / CoUrb prior articles, avoidance of AI-watermarking phrasing,
  reduced over-localised per-state framing in favour of synthesis
  sentences, BRACIS-style calibration without losing the rhetorical
  framing that makes the paper memorable. Addressed across all rounds.

--------

# Final state at submission (Round 6j closure, 2026-05-04)

- Page count: 15 (hard cap, no overflow).
- BibTeX log: zero errors, zero warnings.
- 16 active references; all 16 cites resolve in body; zero unresolved
  cross-references.
- Numerical consistency: substrate Table 1(a) cell − cell = Δ at all 5
  states; per-visit shares (72\%/64\%) match across mechanism / figure /
  contribution C1; range $+14$--$29$ pp consistent across abstract / intro
  / discussion / conclusion. Round 6j numerical adds verified vs §0.1
  ($67.16$ FL STL category ceiling) and §0.6 ($34.49$ POI-RGNN faithful
  FL).
- Statistical rigour: paired Wilcoxon at $n=5$ ceiling (substrate, AL +
  AZ per-visit, Round 6j Batch 3) and multi-seed pooled at $n=20$ (MTL
  vs STL); TOST non-inferiority at $\delta\in\{2,3\}$ pp; $\Delta_m$ on
  two axes (MRR primary, Acc@10 secondary); FL Pareto-positive cell at
  $n=25$.
- Honest framing: AL category negative reported as is; cross-state
  Markov-1 disclosure absorbed into the joint-cost story; literature-
  faithful STAN omission stated explicitly in Table 4 caption + §7;
  user-disjoint vs transductive substrate disclosed in abstract + intro
  parentheticals (Round 6j Batch 2).
- Architecture: Check2HGI claimed as C0 contribution with the 3-boundary
  MI loss specification; cross-attention multi-task backbone diagrammed
  with the bidirectional K/V cross-feed highlighted; §6.3 encoder-frozen
  isolation note now backed by the AZ targeted probe (encoder-frozen
  Acc@10 $60.72$ vs loss-side $62.70$, $+1.98$ pp K/V co-adaptation
  gap; Round 6j Batch 4).
- C2 contribution framed as a *characterisation* of the multi-task
  tradeoff with a structural cost of the cross-attention shared-backbone
  interaction with the region head — not a loss-recipe artefact. Aligns
  with the conclusion's three open lines (asymmetric routing, encoder
  enrichment, region-side dynamic prior; Round 6j Batch 5).
- Reproducibility: anonymous code release, per-fold transition priors,
  fold splits, regression tests all referenced; AZ Wilcoxon run artefact
  saved at `docs/studies/check2hgi/research/AZ_PERVISIT_WILCOXON.json`
  (Round 6j Batch 3).

## Round 6j commit pointers

| Batch | Commit | Closes |
|---|---|---|
| 1 — SUPERSEDED banners | `badc362` | P2 |
| 2+3 — disclosure + softening + AZ Wilcoxon | `21677ff` | P1, M3, A2, P4 |
| 4+5 — §6.3 evidence + C2 softening | `074c831` | A1, A3 |

## Strategic item left open

- **P3 (architecture-as-failed-add-on)**: not implementable without an
  author-level reframe decision (drop the MTL framing entirely, or lead
  with §6.3 as the methodological contribution). Not a pre-submission
  blocker; the paper honestly discloses the negative across §5.2, §5.3
  Δm-Acc@10, §7 limitations, and the conclusion's three open lines.

**Recommendation: submit.**

--------

# Round 5g — closure (2026-05-04)

Single item: **N4 (D1) architecture-figure label collision — CLOSED.**

## Alternative TikZ evaluated and rejected

The author surfaced a candidate replacement at
`/Users/vitor/Downloads/architecture.tex` ("Version B"). I dispatched an
expert-agent review comparing it against the in-repo "Version A"
(`src/figs/architecture.tex`). Verdict: **keep A, apply one-line N4
patch**. Key rationale (full review in agent transcript):

- **Effective rendered font size after `\resizebox{0.60\linewidth}{!}{...}`
  is the binding constraint, not the source-level font choice.** Version
  B's natural TikZ width is ${\sim}118$ mm (driven by the wider 118 mm
  cross-block); LNCS `\linewidth` is ${\sim}117$ mm so target after resize
  is ${\sim}70$ mm. Scale factor: B = $70/118 \approx 0.595$, A =
  $70/78 \approx 0.90$. Effective font: B's `\small` (9 pt) renders at
  ${\sim}5.4$ pt; A's `\scriptsize` (7 pt) renders at ${\sim}6.3$ pt. **B
  is below LNCS readability floor (typ. 6 pt) on a figure already squeezed
  to fit the page cap.**
- **Vertical footprint**: B adds ${\sim}2$ mm rendered height — won't
  push to 16 pp on its own, but the font shrink is already disqualifying.
- **N4 fix in B**: structurally cleaner (kvleft/kvright side-labels +
  centred $\longleftrightarrow$ symbol below the title), genuinely
  resolves the collision rather than relocating it.
- **LNCS aesthetic**: B's `\small\bfseries` on every block title +
  T-stem fork between backbone and heads reads as "tutorial slide";
  A's understated style is more idiomatic Springer LNCS.

## N4 patch applied (option 1)

Of the three documented one-line fixes, the cheapest at our 0.60 cap:

```diff
% --- cross-attention (single block spanning both streams) ---
+% N4 fix (Round 5g): "(Eq.)" cross-ref dropped from box title to free
+% the box centre for the K/V cross-feed label (caption already cites Eq.~2).
 \node[crossbox, below=8mm of $(xc)!0.5!(xr)$] (cross)
-    {Cross-attention (Eq.~\ref{eq:crossattn})};
+    {Cross-attention};
```

Option 2 (bump `crossbox` `minimum height` 10 mm → 14 mm) was tried
first per the original advisor recommendation; it overflowed the PDF to
16 pages. Reverted and replaced with option 1, which has zero
dimensional impact.

## Verification

- **PDF page count: 15** (unchanged from Round 5f).
- **bibtex.log: clean** (no errors, no warnings).
- **Visual**: page 6 architecture figure now shows the cross-attention
  block with title "Cross-attention" alone; the bidirectional
  $K^{c}/V^{c} \leftrightarrow K^{r}/V^{r}$ label sits cleanly above
  the title with the white-fill not overlapping any descender of the
  shorter title text. Caption unchanged at `method.tex:36` and still
  references Eq.~\ref{eq:crossattn}.
- **Equation cross-ref preserved**: caption (`method.tex:36`) and
  body prose (`method.tex:23` "Eq.~\eqref{eq:crossattn} is the
  architectural object the result tables interrogate") both still
  cite Eq.~2; the only loss was the parenthetical inside the figure
  box, which was redundant.

## What is *not* changed

- `src/figs/architecture.tex` font sizes (`\scriptsize`/`\tiny`)
  retained; not bumped to `\small` despite the agent's note that 6.3 pt
  is "small" — bumping would force figure shrink to keep page cap.
- Resize ratio retained at `0.60\linewidth` (Round 5f locked).
- Heads, backbone, K/V flow geometry untouched.
- `/Users/vitor/Downloads/architecture.tex` left in place as a
  reference / camera-ready candidate; not adopted.

## Working tree state

D1 marker in the "Deferred" section above flipped from open to **CLOSED
Round 5g**. No commit made; staged in working tree for author commit.
The submission is now in shape against every HIGH / MEDIUM / LOW item
across Rounds 1–5g.

--------

# Round 6h — critical paper review (2026-05-04)

This round was done from the standpoint of a demanding BRACIS/LNCS reviewer,
not an implementation auditor. I re-read the current paper source
(`src/sections/*.tex`), the rendered `samplepaper.pdf`, the study-side claim
files, and sampled BRACIS 2024/2025 proceedings pages for style calibration.
Verdict: **the draft is promising, but not yet submission-safe**. The main
remaining risk is not missing experiments; it is that several parts of the
paper still read like an internal closure memo compressed into LNCS form.

## Findings (ordered by severity)

### H1. The evaluation story overstates "user-disjoint" generalisation when the substrate is transductive

**Why this matters.** The abstract and intro lead a reviewer to expect a
held-out-user evaluation. But `method.tex:19` and `setup.tex:11` reveal that
both Check2HGI and HGI are pre-trained on each state's **full** check-in graph,
so held-out users influence the substrate before downstream folds are created.
That may be acceptable as a transductive protocol if argued explicitly, but it
does **not** support the stronger reading that the paper is evaluating a clean
new-user setting end to end.

**Where it bites the draft now.**

- `samplepaper.tex` abstract: "user-disjoint folds" is foregrounded without the
  transductive qualifier.
- `src/sections/intro.tex:10` uses "under user-disjoint folds" as part of the
  paper's main setup sentence.
- `src/sections/discussion.tex:6` demotes the issue to the fourth limitation,
  which is too late relative to how central the evaluation claim is.

**What I would change.** Either run one small held-out-user substrate retrain
audit and report it honestly, or move the transductive qualifier into the
abstract/intro and frame the paper as a controlled **relative substrate study**
under a transductive embedding setup. Right now the paper wants the rhetorical
benefit of "user-disjoint" while burying the strongest qualifier.

### H2. The POI-RGNN comparison is not yet trustworthy enough to anchor the external story

**Why this matters.** Reviewers do not need every baseline to be perfect, but
they do need one stable story. The current paper-side material still contains
multiple, conflicting POI-RGNN stories:

- `src/tables/external.tex:3,17` says POI-RGNN is our **user-disjoint
  reproduction** and gives FL/CA/TX = **34.49 / 31.78 / 33.03**.
- `src/sections/results.tex:38` repeats the same framing and uses it for the
  "roughly doubles" claim.
- `docs/studies/check2hgi/baselines/next_category/comparison.md:13,56-58`
  reports **33.35 / 30.71 / 32.08** from the baseline JSON path.
- `articles/[BRACIS]_Beyond_Cross_Task/STATISTICAL_AUDIT.md:184` still says the
  reproduction "*did not* use user-disjoint folds".
- `COORDINATOR_HANDOFF.md:355` explicitly notes that the JSON values diverge
  from the canonical article values.

This is not a cosmetic mismatch. It means the article's external-comparison
axis depends on numbers and protocol claims that are not yet internally
reconciled.

**What I would change.** Before submission, choose one auditable POI-RGNN
source of truth and make every article-side file inherit from it. If the
baseline is user-disjoint, say so everywhere and explain why the JSON summary
differs. If it is not, stop calling it user-disjoint. Until that is fixed, I
would remove or soften any sentence built on "Check2HGI roughly doubles
POI-RGNN".

### H3. The paper still has not decided whether it is proposing an MTL model or diagnosing why MTL is not the answer

**Why this matters.** The substrate result is strong and clean. The architecture
result is scientifically interesting, but mostly because it is a careful
negative result: MTL helps category a bit at four states, hurts it at AL, and
is bad on region everywhere. The manuscript partly knows this, but the story is
not yet aligned around it.

**Symptoms in the current draft.**

- The title and C2 framing are already moving toward "architecture pays", but
  the paper still spends a lot of energy presenting the MTL backbone as if it
  were the natural deployment answer.
- `src/sections/results.tex:24-28` and `discussion.tex:4-6` acknowledge the
  region cost, yet the paper never cleanly states the practical takeaway:
  **if you care about performance, use Check2HGI + STL; if you care about a
  single joint model, you accept a region penalty.**
- `src/tables/external.tex` leaves the best region row to STAN, and
  `results.tex:26` admits MTL is below Markov-1 at four of five states.

**What I would change.** Reframe the paper more explicitly as:

- contribution A: Check2HGI exposes a task-asymmetric substrate effect;
- contribution B: under that substrate, cross-attention MTL does **not**
  recover bidirectional gains and reveals a structural tradeoff.

That is a strong paper if owned directly. It is a weaker paper if written as a
"new model paper" whose own best recommendation is not to use the new model.

### H4. The per-visit mechanism is headline material with non-headline evidence

**Why this matters.** The mechanism story is elegant, but the paper currently
gives it more rhetorical weight than the evidence supports.

**Current evidence.**

- AL: inferential mechanism split, good.
- AZ: descriptive replication only; `src/sections/mechanism.tex:11` explicitly
  says no paired test is run.
- FL/CA/TX: no mechanism replication.

**Current emphasis.**

- abstract ends with the 72/64 claim;
- `intro.tex:15` makes it a contribution bullet;
- `conclusion.tex:4` repeats it as one of the closing facts;
- the figure is a major visual anchor.

I would either demote it slightly or scope it much more aggressively
("two-state mechanism probe, inferential at AL and descriptive at AZ"). Right
now the reader sees a polished mechanistic result but gets narrow support when
they inspect the section closely.

### M1. The prose still sounds too much like an internal project memo

Sampled BRACIS papers in the 2024/2025 proceedings are much plainer in tone.
This draft is better than earlier versions, but expressions such as
"load-bearing", "headline scale", "smaller-scale anchors", "textbook tradeoff",
"honest reattribution", and repeated "carries/pays" phrasing still make the
paper sound self-aware in a way LNCS reviewers may read as rhetorical pressure.
The scientific content is strong enough that it does not need this vocabulary.

Concrete pressure points:

- `intro.tex:8,15-16`
- `related.tex:9,19,21`
- `results.tex:15,24-28,33,38`
- `discussion.tex:4-6`

Recommendation: simplify the voice another notch. Favor conventional claims
("suggests", "indicates", "is consistent with") over interpretive slogans.

### M2. Check2HGI is still under-specified relative to how central C0 is

If Check2HGI remains a named contribution (`intro.tex:14`), the paper should
be reproducible at the method level without forcing the reviewer into the code
release. `method.tex:9-18` gives the hierarchy and loss weights, but the paper
still outsources too much of the actual recipe: graph construction details,
negative sampling/corruption choices, and the exact pretraining protocol are
mostly gestured at. For a derivative engine this may be acceptable **only if**
the contribution is framed as an empirical substrate study rather than as a new
embedding method. Right now the paper tries to have both.

### M3. A few interpretive sentences are stronger than the evidence on the page — **CLOSED Round 6j Batch 2+3 (2026-05-04, commit 21677ff)**: results.tex:26 Markov-1 sentence rewritten; results.tex:33 Δm "precisely because" → "is consistent with"

Examples:

- `results.tex:26` moves from "MTL trails Markov-1 on Acc@10" to "what a
  one-step Markov chain captures is what the cross-attention region head
  underfits." Plausible, but not directly shown.
- `results.tex:33` says MRR "sees through" the Acc@10 saturation regime. Again
  plausible, but the paper has not established that mechanism, only the metric
  divergence.

These lines should be softened. They currently read as causal explanations for
patterns that are still observational.

## Bottom-line recommendation

Do **not** treat the current article as "ready except cosmetic edits". The
strong part of the paper is strong enough to submit, but only if the review-team
first resolves the trust issues above:

1. Make the transductive substrate assumption impossible to miss.
2. Reconcile the POI-RGNN protocol and numbers everywhere.
3. Decide, explicitly, that the architecture result is diagnostic/negative, and
   write the story around that choice.
4. Reduce the rhetorical weight on the AL/AZ mechanism split unless more
   evidence is added.

If those four are fixed, the paper becomes much more reviewer-proof without
needing new large-scale experiments.

--------

# Round 6i — critical audit of the Codex Round 6h review (2026-05-04)

The Codex review above is generally well-reasoned. I cross-checked each
finding against source files; the per-item verdicts are summarised at the
top under "Pending — pre-submission" (P1--P4). What follows audits the M-tier
items that did not get propagated to the pending list, plus three additional
considerations the Codex review did not surface.

## Audit of Codex M-tier items

### M1 (prose internal-memo voice) — DIMINISHING RETURNS

Voice-trim has been done across Rounds 1, 5c, 5d. Current state:
"load-bearing" $2\times$ (intro + related), "textbook tradeoff" $3\times$
(abstract + intro + related). Each remaining instance is in a load-bearing
position. The Codex push for further trim ("headline scale", "smaller-
scale anchors", "carries/pays") is reasonable but the listed phrases
mostly have no idiomatic LNCS replacement that preserves meaning:

- "headline scale" $\leftrightarrow$ "the FL/CA/TX states" (loses the
  hierarchical framing)
- "smaller-scale anchors" $\leftrightarrow$ "the AL/AZ states" (loses the
  comparative framing the §5.1 substrate analysis depends on)
- "carries / pays" $\leftrightarrow$ "is the determining factor / incurs
  a cost" (loses the title's rhetorical pair)

**Verdict: KEEP CODEX RECOMMENDATION as cosmetic camera-ready polish.**
Not a pre-submission blocker. The paper's voice is already two BRACIS-
calibration rounds in; further trim risks losing the memorable framing
that makes the paper stick in a reviewer's mind among 393 submissions.

### M2 (Check2HGI under-specified vs C0 promotion) — DEFENSIBLE TRADE-OFF

Codex's tension is real: under Option B (Check2HGI as C0 contribution),
the method spec at `method.tex:9-18` gives the 4-level hierarchy and the
3-boundary MI loss equation but defers graph construction, negative
sampling, exact pretraining protocol to the anonymous code release.

**My read**: this is a defensible LNCS pattern — the loss equation is the
load-bearing architectural claim; graph construction details are
implementation-grade and standard in the GCN literature. A reviewer who
wants to reproduce can use the code release; a reviewer who wants to
understand the method has the equation and the corruption-trick sentence.

**However**, if a reviewer pushes on "where is the algorithmic
contribution beyond the loss equation?", the answer is thin. This loops
back to P3 (architecture as contribution) — the C0 promotion under
Option B is itself a strategic risk. **No pre-submission action needed**;
the trade-off is already chosen.

### M3 (interpretive overreach) — VALID, ONE-LINE FIX

Codex flags two specific lines:

- `results.tex:26` *"what a one-step Markov chain captures is what the
  cross-attention region head underfits"* — interpretive causal claim
  beyond what the data shows.
- `results.tex:33` *"the same FL state is exceptional on $\Delta_m$-MRR
  precisely because the Acc@10-saturated regime described above is what
  MRR sees through"* — same pattern: causal explanation of a metric-
  divergence pattern that has only been observed.

**Both are real overreaches.** One-line softening fixes:

1. `results.tex:26` change *"what a one-step Markov chain captures is what
   the cross-attention region head underfits"* $\to$ *"the cross-attention
   region head underperforms on the short-horizon transition structure
   that a one-step Markov chain already captures"* — same observation,
   no causal claim about what the head "captures".
2. `results.tex:33` change *"precisely because"* $\to$ *"is consistent
   with"*. From causal to correlative.

**Effort: ${\sim}1$ minute.** Recommend applying.

## Additional considerations Codex did not surface

### A1 (NEW). §6.3 encoder-isolation note is asserted without numerical evidence — **CLOSED Round 6j Batch 4 (2026-05-04)**: AZ probe quoted in §6.3 (encoder-frozen 60.72 vs loss-side 62.70 Acc@10, +1.98 pp K/V co-adaptation gap) sourced from F49_LAMBDA0_DECOMPOSITION_RESULTS.md

The §6.3 cross-attention $\lambda=0$ vs encoder-frozen finding is a
methodological claim the paper relies on (it justifies the encoder-frozen
ablation protocol throughout). But the body shows **zero numerical
comparison** between the two protocols — the paper says "regression tests
in the code release" and stops there.

If a reviewer reads §6.3 and asks "show me the difference between
$\lambda=0$ and encoder-frozen at one state, one head, one fold", there is
nothing in the body to point at. This is a real reviewer attack surface,
especially because the framing leans on §6.3 as the methodological
contribution that anchors the cross-attention story.

**Fix:** add one sentence with concrete numbers in §6.3, e.g., *"At AL,
the loss-side $\lambda_{\text{cat}}=0$ ablation drops region Acc@10 by
$X.X$ pp; the encoder-frozen isolation drops it by $Y.Y$ pp, confirming
that the surviving stream's K/V projections continue to carry category
gradient through the targeted encoder."* If the numbers do not exist
yet, run a one-day comparison at one state.

**Severity: MEDIUM.** The claim is correct and important; the absence of
in-body evidence weakens the only methodological contribution that
distinguishes this paper from a substrate-axis ablation.

### A2 (NEW). The "doubling vs POI-RGNN" claim quotes external numbers, not multi-task numbers — **CLOSED Round 6j Batch 2+3 (2026-05-04, commit 21677ff)**: results.tex:38 + discussion.tex:4 reframed as "matched-head STL category ceiling on Check2HGI ($67.16$ at FL)"

`results.tex:38` and `discussion.tex:4` say *"Check2HGI's matched-head
category ceiling at FL ($67.16$) roughly doubles the strongest
user-disjoint POI-RGNN reproduction ($34.49$)"*. The 67.16 is the
**STL** category ceiling on Check2HGI — i.e., the doubling claim is for
the substrate + matched-head GRU, not for the proposed cross-attention
MTL model.

This is not strictly dishonest — the paper says "matched-head category
ceiling" — but a casual reviewer reading the abstract / discussion will
hear "our model doubles POI-RGNN" and not parse the matched-head
qualifier. The MTL row itself ($68.56$ at FL) is only marginally above
the STL ceiling, and trails by $-7$ to $-17$ pp on region.

**Fix:** rewrite *"Check2HGI's matched-head category ceiling at FL
($67.16$)"* $\to$ *"the matched-head STL category ceiling on Check2HGI
($67.16$ at FL, single-task)"* — making explicit that the doubling is a
single-task substrate effect, not a multi-task model claim.

This also helps mitigate P3 (architecture-as-contribution) by being clear
that the dramatic external lift comes from substrate, not architecture.

**Severity: MEDIUM.** Reviewer-trust issue if a reader misreads the
qualifier; honest disclosure if read carefully.

### A3 (NEW). The conclusion's "open lines" implicitly admit the architecture failed — **CLOSED Round 6j Batch 5 (2026-05-04)**: intro.tex:16 C2 bullet softened to "direct characterisation of the multi-task tradeoff..." with explicit "structural cost of the cross-attention shared-backbone interaction with the region head" framing

`conclusion.tex` lists three follow-up directions: *"asymmetric routing
across the shared backbone to relax cross-attention interference, encoder
enrichment fusing per-visit contrast with the explicit spatial-temporal
sub-encoders of [CoUrb], and a region-side dynamic prior conditioned on
the user's recent trajectory."*

**All three target the architecture's weakness** — interference,
encoder limitations, region-head failure. This is honest and
reviewer-positive (the paper acknowledges the architecture needs work).
**But it also implicitly confirms P3** (the architecture is a diagnostic
result, not a deployment recommendation). A senior reviewer will read the
three open lines and conclude "the authors agree the architecture
doesn't work; future work is needed". That reading is correct — the
question is whether it sinks the paper or strengthens the honesty signal.

**No action needed**, but worth noting that the conclusion's framing
already takes the strategic position P3 recommends. **The contradiction
is between the conclusion's honest reframe and the C2 contribution
bullet's confident "direct measurement of the multi-task tradeoff"
framing in the intro.** Aligning these two would close the strategic gap
without changing experiments.

## Net verdict on the Codex Round 6h review

Codex produced a substantive, well-targeted critique. **Two-thirds of the
findings are valid and have one-line fixes.** The misdiagnosis on P2 is
the most consequential issue with the Codex critique itself — pointing
the fix at the article rather than at the stale working memos would have
caused unnecessary article-side edits.

**Combined pre-submission punch list (Round 6h Codex + Round 6i mine):**

| # | Item | Effort | Severity |
|---|------|------|------|
| **P2-fix** | Mark `STATISTICAL_AUDIT.md` and `comparison.md` as superseded by `RESULTS_TABLE.md §0.6` (or archive them outside the anonymous-code-release scope) | $\sim 5$ min | HIGH if anon code release exposes them |
| **P1-fix** | Add transductive parenthetical to abstract + intro setup sentence | $\sim 5$ min | MEDIUM |
| **P4-fix (preferred)** | Run AZ paired Wilcoxon (D3) — converts AZ from descriptive to inferential | 1 day | MEDIUM |
| **P4-fix (fallback)** | Add "(inferential at AL, descriptive at AZ)" to the abstract 72/64 sentence | $\sim 1$ min | MEDIUM |
| **M3-fix** | Soften two interpretive causal sentences at `results.tex:26` and `:33` | $\sim 2$ min | LOW |
| **A1-fix** | Add one numerical comparison sentence to §6.3 encoder-isolation note | 1 day if numbers must be generated; $\sim 5$ min if they exist | MEDIUM |
| **A2-fix** | Rewrite doubling claim to make "matched-head STL" explicit | $\sim 2$ min | MEDIUM |
| **A3-note** | Align C2 intro framing with conclusion's honest reframe | $\sim 5$ min | LOW |

**Pre-submission core: ~15 minutes** if P4 takes the fallback wording
fix and A1 numbers already exist in the code release. **Pre-submission
expanded: ~1 day** if P4 runs AZ paired Wilcoxon and A1 generates the
missing comparison numbers.

P3 (architecture-as-failed-add-on) is the only finding without a sub-day
fix path — and it is genuinely a strategic/structural concern that requires
either a major reframe or accepting the borderline acceptance probability.
**My recommendation matches the prior verdict: submit with the current
framing this cycle; if rejected, reframe as substrate-axis paper for
resubmission elsewhere.**
