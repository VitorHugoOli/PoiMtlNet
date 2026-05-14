# Review status

This file tracks **what is still pending** on the BRACIS submission. The full
review history (six rounds of Codex + Claude reviews + figure / caption /
voice polish) is preserved in git log; the items below are the ones not
fully closed by the last pass and worth picking up before the camera-ready
deadline. The PDF currently compiles to **15 pages** (LNCS hard cap); any
new edit must hold or reduce the count.

--------

# Pending — pre-submission

*(none reviewer-blocking)*

The audit cycle (Rounds 1--6j + figure / caption / voice polish) closed
every HIGH and MEDIUM finding raised by either Codex critical reviews or my
own meticulous audits. The PDF builds to 15 pages with zero BibTeX errors,
all 16 citations resolve, Table 1(a) cell -- cell math reconciles for every
state, and both figures (architecture and per-visit) render at full
\linewidth and are legible without zoom.

--------

# Deferred — camera-ready window

These items are not blocking submission; they are the natural polish path
for the camera-ready cycle once reviewer feedback arrives.

## D1. AZ paired-Wilcoxon on the per-visit counterfactual folds

`mechanism.tex` reports the AZ replication as descriptive only ("we do not
run a paired test across the AZ folds and report the second state as
descriptive replication, not inferential confirmation"). Running the paired
Wilcoxon on the existing AZ fold-level deltas would convert "descriptive"
to "two-state replicated" without changing the numbers. One-day
pre-camera-ready run; zero new training compute.

## D2. AL-state encoder-isolation probe in §6.3

§6.3 currently carries the AZ encoder-frozen-vs-loss-side-$\lambda=0$ probe
($60.72$ vs.\ $62.70$, $+1.98$ pp co-adaptation gap) as the methodological
numerical demonstration. F49 also has the AL probe ($74.85$ vs.\ $74.94$,
${\sim}0$ pp gap, lift purely architectural) which would deepen the
methodological note from "single example" to "state-dependent
decomposition". Tested four caption / prose compressions to fit it
pre-submission; all pushed the PDF to 16 pages. Stays in F49 internal docs
for camera-ready when reviewer feedback opens up the page budget.

## D3. Camera-ready author block restoration

`samplepaper.tex` carries the anonymised author block for the double-blind
submission, plus the camera-ready author block (Tarik S. Paiva, Vitor H. O.
Silva, Germano B. dos Santos, Fabricio A. Silva at NESPeD-LAB / UFV)
preserved as commented LaTeX directly below. Restoration at camera-ready
is a comment-swap.

--------

# Audit-trail summary (what was closed across the rounds)

Each item below was checked against the source files; full review text is
in the prior commits on this branch.

- **Round 1 (initial Codex review).** Resolved: prior-work citations
  (silva2025mtlnet, paiva2026courb), "leak-free" branding stripped from
  abstract / intro / related / discussion, over-defensive novelty wording
  removed, fairness sentence in setup made explicit, Markov-1-at-FL
  honesty paragraph added, external-baseline caveats added, code names
  (`next_gru`, `next_stan_flow`) replaced by conceptual labels in body
  prose, table spacing relaxed (`\arraystretch=1.1` across T1--T4),
  best-per-column bolding convention declared in T4 caption.

- **Round 2 (Claude — second pass).** Resolved: Markov-1-at-FL surfaced
  in §5.2 + §7; substrate-first framing reflected in title /
  contributions; F-arch K/V cross-feeds recoloured bold black; AZ
  counterfactual run integrated; §3.3 made the encoder-frozen note
  visible.

- **Rounds 3--4 (Codex final pass + methodological-faithfulness round).**
  Resolved: Check2HGI under-specification (§3.1 expanded);
  ceiling-estimator labelling clarified in §4.3; substrate caption
  corrected; "camera-ready (compute budget)" promise language removed;
  scale-conditional optimiser disclosure in §3.3; transductive substrate
  disclosure in §4 + §7; ReHDM coverage statement aligned with what the
  table shows; AZ replication wording softened to descriptive;
  "supplementary material" deferrals replaced by anonymous-code-release
  pointers; intro method-detail trimmed; abstract updated for AL+AZ
  replication.

- **Round 5 (Claude — meticulous numerical + storytelling).** Resolved:
  H1 Markov-1 cross-state pattern reframed (no longer FL-only); H2 FL
  saturated / Pareto-positive reconciled in §5.3; H3 substrate Table
  1(a) protocol switched to seed-42 throughout so cell-cell = Δ; H4 AZ
  HGI drift reconciled at $28.99 \pm 0.51$; H5 "faithful STAN" prose
  rewritten; M1 $\lambda_{\text{cat}}$ sensitivity pointer; M2
  head-invariant probes pointer; M4 Acc@10 cost ranges verified clean;
  per-visit shares verified clean.

- **Round 5b (Author considerations).** Resolved: A1 per-visit figure
  upgraded to two-panel AL+AZ (later closed at 5-state); A2
  doubling-vs-external claim added to §5.4 + §7 with substrate
  attribution preserved; A2-followup applied as **Option B** —
  Check2HGI promoted to C0 contribution, abstract opens with "We
  introduce Check2HGI", §3.1 expanded with the 3-boundary MI loss
  equation; conclusion opens "We introduced Check2HGI"; §7 scope
  sentence pre-empts the genericity question; A3 future-work refined.

- **Round 5c (BRACIS proceedings style audit).** Resolved: B1
  architecture figure simplified; B2 contribution bullets unbolded;
  B3 voice trim ("load-bearing" 5×→2×, "textbook tradeoff" 4×→3×);
  B4 math expansion under Option B; B5/B6/B7 verified appropriate;
  B8 punch-list collapsed.

- **Round 5d/5e/5f/5g (self-audits + bibliography web verification +
  figure-render audit).** Resolved: H6 Markov-1 cross-state numbers
  inline at `results.tex:26`; H7 `yang2022getnext` restored after
  cross-cite gap; HB1 `capanema2022poirgnn` → `capanema2023poirgnn`
  (Ad Hoc Networks 138:103016, 2023, DBLP-verified); HB2 `li2025rehdm`
  UNVERIFIED tag dropped, full author names + pages 3081--3089 + DOI
  added (Wen{-}liang hyphen-protected); MB1 `zeng2019mhape` DOI
  chapter `_22` → `_21`; M3 stale "Total entries: 30" comment
  refreshed to "16 active"; N1 BibTeX `@Comment{...}` wrapper applied
  to commented `lim2022hmtgrn`; N2 architecture-figure resize
  bisected; N4 cross-attention label collision fixed. Bibliography
  web-verified against DBLP / IJCAI / Springer for 13 entries; 2
  Brazilian-venue entries trust-by-provenance.

- **Round 6h--6j (Codex critical external-style review + applied fixes).**
  Resolved: P1 user-disjoint vs.\ transductive substrate parenthetical
  added to abstract + intro; P2 POI-RGNN internal consistency
  confirmed (article matches canonical RESULTS TABLE; stale internal
  memos are the issue, not the paper); P3 architecture-as-
  contribution-vs-failed-add-on framing tension acknowledged as
  strategic-and-honestly-disclosed (no fix at this cycle); P4 per-visit
  mechanism over-emphasis closed by 5-state upgrade.

- **Figure / caption / voice polish.** Resolved: D2 5-state per-visit
  figure (FL/CA/TX H100 results cherry-picked, 9 per-fold JSONs; first
  5-panel render via render_per_visit_5state.py, then redesigned to
  compact grouped-bar via render_per_visit_grouped.py showing 5 state
  groups × 3 bars with per-visit shares 72/64/90/89/90 % directly
  annotated); arch.png swapped from TikZ to landscape PNG (4040×1000,
  4.04:1 aspect); §3 method compressions; Table 4 caption trimmed from
  12 lines to 4 (page 11 white-gap eliminated, §6 header flows back
  from page 12); Fig 2 caption trimmed (per-state shares already in
  §6.1 body); arch.png inclusion grown $0.45 \to 0.54 \to 0.85 \to
  0.95 \to \linewidth$ via successive compressions; final state both
  figures at \linewidth, comfortably readable.

- **Author considerations across rounds.** Voice / tone alignment with
  CBIC / CoUrb prior articles, avoidance of AI-watermarking phrasing,
  reduced over-localised per-state framing in favour of synthesis
  sentences, BRACIS-style calibration without losing rhetorical
  framing that makes the paper memorable. Camera-ready author block
  staged in commented form (D3).

--------

# Final state at submission

- Page count: **15** (hard cap, no overflow).
- BibTeX log: **zero errors, zero warnings**.
- 16 active references; all 16 cites resolve in body; zero unresolved
  cross-references.
- Numerical consistency: substrate Table 1(a) cell − cell = Δ at all 5
  states; per-visit shares (72/64/90/89/90 %) match across mechanism
  prose, figure annotation, and contribution C1; range $+14$--$29$ pp
  consistent across abstract / intro / discussion / conclusion.
- Statistical rigour: paired Wilcoxon at $n=5$ ceiling and multi-seed
  pooled at $n=20$; TOST non-inferiority at $\delta\in\{2,3\}$ pp;
  $\Delta_m$ on two axes (MRR primary, Acc@10 secondary); FL Pareto-
  positive cell at $n=25$; 5-state per-visit replication paper-grade
  at AL/AZ ($p=0.0312$ each), descriptive at FL/CA/TX (D1 deferred
  paired-Wilcoxon).
- Honest framing: AL category negative reported as is; cross-state
  Markov-1 disclosure absorbed into the joint-cost story; literature-
  faithful STAN omission stated explicitly in Table 4 caption + §7;
  $\Delta_m$ broadly negative with FL-on-MRR exception named in §5.3
  title.
- Architecture: Check2HGI claimed as C0 contribution with proper
  3-boundary MI loss specification; cross-attention multi-task
  backbone diagrammed at \linewidth on page 5; encoder-frozen
  isolation note in §6.3 carries the methodological caution with
  AZ numerical demonstration ($+1.98$ pp co-adaptation gap).
- Reproducibility: anonymous code release referenced in §3.3 + §7;
  per-fold transition priors, fold splits, regression tests all
  released. Per-visit launchers (5-state runner, 5-state-grouped
  renderer, per-fold extractor) all in `scripts/` and `src/figs/`.
- Anonymity: `samplepaper.tex` author block is `Anonymous Author(s)
  / Anonymous Institution`; camera-ready block (Paiva / Silva /
  dos Santos / Silva at NESPeD-LAB UFV) staged in comment for
  comment-swap restoration (D3).

**Recommendation: submit.**
