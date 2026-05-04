# Review status

This file tracks **what is still pending** on the BRACIS submission. The full
review history (four rounds of Codex + Claude reviews and the author's running
notes) is preserved in git log; the items below are the ones that were not
fully closed in the last pass and are worth picking up before submission or at
camera-ready.

The PDF is at the 15-page hard cap; any new edit must hold or reduce the count.

--------

# Pending — pre-submission

## ReHDM bibliography entry remains incomplete

`src/references.bib` has the `li2025rehdm` entry tagged `% UNVERIFIED:` because
the author given names and the page field were never recovered from a primary
source. It currently renders as "Li, Gu, Yao, Zhou, Zhu, Zhao, Du" with `p.~0343`
(which is almost certainly an IJCAI paper-ID rather than a page range). A
reviewer who looks at the bibliography will notice. **Fix:** look up the IJCAI
2025 proceedings page for the paper title (*"Beyond Individual and Point: Next
POI Recommendation via Region-aware Dynamic Hypergraph with Dual-level
Modeling"*) and replace the placeholder fields. Low effort; high reviewer
visibility.

--------

# Deferred — picked up if compute / page budget allow, otherwise camera-ready

## Per-visit figure: AL+AZ two-panel upgrade

**Status.** `src/figs/render_per_visit.py` and `src/figs/per-visit.png` show the
single-state Alabama panel. `mechanism.tex §6.1` already documents both states
in prose (canonical $43.17$, POI-pooled $34.09$, HGI $28.99$ at AZ) but the
figure visualises only AL.

**Proposed change.** Two side-by-side panels (AL and AZ), three bars each
(HGI / POI-pooled Check2HGI / canonical Check2HGI), with per-visit and
training-signal brackets and the share split. This shows the two-state
replication visually rather than relying on §6.1 prose alone.

**Why averaging across n=2 is the wrong move.** The author's instinct in
earlier review notes was "average across the states" — that hides the per-state
shares (72\% vs 64\%) which *are* the mechanism evidence. The replication
signal is "both positive, similar magnitude," not "the mean is X." The
two-panel format preserves the per-state shares.

**Drop-in code (prototyped and rendered cleanly).** Two-panel version with
shared y-axis, panels for `("Alabama (AL)", 25.26, 29.57, 40.76)` and
`("Arizona (AZ)", 28.99, 34.09, 43.17)`, at `figsize=(8.5, 3.4)`. The figure
caption in `src/sections/mechanism.tex:18` would need to be widened to describe
both panels, and `\includegraphics[width=0.55\linewidth]` should grow to about
`0.75\linewidth` for legibility.

**Page-budget caveat.** The two-panel figure is wider than the current single
panel; a future agent must re-check the page count after the swap, since the
paper sits at the 15-page hard cap. If the swap pushes the PDF to 16 pages,
revert and ship the AL-only figure with §6.1 prose carrying the AZ replication.

## AZ paired-Wilcoxon on the per-visit counterfactual folds

`mechanism.tex:13` currently reports the AZ replication as "consistent with the
AL ordering, with both components positive in matched-head means" and
explicitly says "we do not run a paired test across the AZ folds and report the
second state as descriptive replication, not inferential confirmation." If a
reviewer asks for inferential support, running the paired Wilcoxon on the AZ
fold-level deltas would convert "descriptive" back to "two-state replicated"
without changing the numbers. One-day pre-camera-ready run; not blocking.

--------

# Audit-trail summary (what was closed in earlier rounds)

For continuity if a new agent picks this up. Each item below was checked
against the source files on 2026-05-03; full review text is in the prior
commits on this branch.

- **Round 1 (initial Codex review).** Resolved: prior-work citations
  (silva2025mtlnet, paiva2026courb), "leak-free" branding stripped from
  abstract/intro/related/discussion, over-defensive novelty wording removed,
  fairness sentence in setup made explicit, Markov-1-at-FL honesty paragraph
  added, external-baseline caveats added, code names (`next_gru`,
  `next_stan_flow`) replaced by conceptual labels in body prose, table
  spacing relaxed (`\arraystretch=1.1` across T1–T4), best-per-column bolding
  convention declared in T4 caption.

- **Round 2 (Claude — second pass).** Resolved: Markov-1-at-FL surfaced in
  §5.2 + §7; substrate-first framing reflected in title/contributions; F-arch
  K/V cross-feeds recoloured bold black (load-bearing for §6.3); AZ
  counterfactual run and integrated; §3.3 made the encoder-frozen note
  visible.

- **Round 3 (Codex — final pass on PDF).** Resolved: Check2HGI under-
  specification (§3.1 expanded); ceiling-estimator labelling clarified in
  §4.3; substrate caption corrected; "camera-ready (compute budget)" promise
  language removed from external.tex.

- **Round 4 (Codex — revised draft / methodological-faithfulness round).**
  Resolved: scale-conditional optimiser disclosure in §3.3 (FL/CA/TX vs AL/AZ
  recipe split); transductive substrate disclosure in §4 + §7; ReHDM coverage
  statement aligned with what the table shows; AZ replication wording
  softened to descriptive; "supplementary material" deferrals replaced by
  anonymous-code-release pointers; intro method-detail trimmed; abstract
  updated for AL+AZ replication.

- **Author considerations.** Voice/tone alignment with CBIC/CoUrb prior
  articles, avoidance of AI watermarking phrasing, reduced over-localised
  per-state framing in favour of synthesis sentences. Addressed across all
  rounds.
