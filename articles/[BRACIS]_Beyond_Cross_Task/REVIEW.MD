# Review status

This file tracks **what is still pending** on the BRACIS submission. The full
review history (six rounds of Codex + Claude reviews and the author's running
notes) is preserved in git log; the items below are the ones that were not
fully closed in the last pass and are worth picking up before the camera-ready
deadline. The PDF currently compiles to **15 pages** (LNCS hard cap); any new
edit must hold or reduce the count.

--------

# Pending — pre-submission

*(none reviewer-blocking)*

The six-round audit cycle (Rounds 1--5f) closed every HIGH and MEDIUM finding.
The PDF builds to 15 pages with zero BibTeX errors, all 16 citations resolve,
and Table 1(a) cell -- cell math reconciles for every state. The submission
package is in shape.

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

# Final state at submission

- Page count: 15 (hard cap, no overflow).
- BibTeX log: zero errors, zero warnings.
- 16 active references; all 16 cites resolve in body; zero unresolved
  cross-references.
- Numerical consistency: substrate Table 1(a) cell − cell = Δ at all 5
  states; per-visit shares (72\%/64\%) match across mechanism / figure /
  contribution C1; range $+14$--$29$ pp consistent across abstract / intro
  / discussion / conclusion.
- Statistical rigour: paired Wilcoxon at $n=5$ ceiling and multi-seed
  pooled at $n=20$; TOST non-inferiority at $\delta\in\{2,3\}$ pp;
  $\Delta_m$ on two axes (MRR primary, Acc@10 secondary); FL Pareto-
  positive cell at $n=25$.
- Honest framing: AL category negative reported as is; cross-state
  Markov-1 disclosure absorbed into the joint-cost story; literature-
  faithful STAN omission stated explicitly in Table 4 caption + §7.
- Architecture: Check2HGI claimed as C0 contribution with proper
  3-boundary MI loss specification; cross-attention multi-task backbone
  diagrammed; encoder-frozen isolation note in §6.3 carries the
  methodological caution.
- Reproducibility: anonymous code release, per-fold transition priors,
  fold splits, regression tests all referenced.

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
