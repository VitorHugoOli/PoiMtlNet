# ADAPTATION LEDGER — Chapter 3 (CBIC 2025 re-typeset)

> Source of record: `articles/CBIC___MTL/` (main.tex + sections/{intro,basis,method,results,
> conclusion}.tex + tables/ + imgs/), published at CBIC 2025, DOI 10.21528/CBIC2025-1191324.
> Target: `articles/dissertacao/src/chapters/3_cbic.tex`.
> Every departure from the published text is listed here (feeds Appendix B; reconciliation
> gate: fixes applied == fixes listed). Citation errata #1–#9 in `articles/CBIC___MTL/ERRATA.md`
> are Phase-4 bibliography work and are NOT applied here; all \cite keys kept as-is.

## A. Structural re-typeset changes (format only, no content change)

| # | Change | Where |
|---|---|---|
| A1 | Abstract and IEEEkeywords dropped (coletânea chapters carry no abstract; NORTH_STAR §3). | whole chapter |
| A2 | Paper `\section` levels became `\section` under the `\chapter`; labels renumbered to the chapter-prefixed scheme: `sec:related → sec:cbic:related`, `sec:methodology → sec:cbic:methodology`, `sec:mtl → sec:cbic:mtl`, `sec:method:mtl → sec:cbic:mtlarch`, `sec:nash_mtl → sec:cbic:nash`, `sec:experiments → sec:cbic:experiments`, `sec:conclusion → sec:cbic:conclusion`, `fig:arch → fig:cbic:arch`, `eq:loss → eq:cbic:loss`, `eq:nbs_objective → eq:cbic:nbs`, `table:cat → tab:cbic:category`, `table:next → tab:cbic:next`, `table:convergence_comparison → tab:cbic:convergence`. | whole chapter |
| A3 | "this paper" → "this chapter" (4 instances: §Intro "In this work, we investigate" → "In this chapter"; "contributions of this paper" → "of this chapter"; "The rest of this paper is organized" → "of this chapter"; §Method "This work computes" → "This chapter computes"; §Conclusion "In this paper, we introduced" → "In this chapter"). | intro, method, conclusion |
| A4 | Italic time-capsule preface added before §3.1 (`chapterpreface` environment): venue in full, DOI, first author, of-the-time framing, pointer to Chapters 4–5, one clause on the later Nash-MTL implementation finding (sanctioned by NORTH_STAR §4 Ch.3 claim-discipline note). Not part of the published text. | chapter opening |
| A5 | `IEEEeqnarray` (2-col line-broken DGI loss) reflowed to a single-line `equation` for 1-col; content identical. `\argmax` (paper-preamble macro) → `\operatorname*{arg\,max}` inline. `\resizebox` wrapper around the parameter-partition display removed (not needed at 1-col); the Θ_specific set line-broken for width. | method |
| A6 | Tables converted from vertical-ruled `tabular` + `\resizebox` to booktabs (no vertical rules), captions moved ABOVE tables (WRITING_LAW §5), captions rewritten as self-contained descriptions naming metric, spread convention, and bolding convention (original captions: "Table comparing the results of ..."). Cell values, bolding, and underlining copied verbatim from the published tables (see B7 for the one bolding-convention note). `{\ul ...}` (ulem) → `\underline{...}`. | results tables |
| A7 | Lead takeaway sentence added before the convergence table ("The convergence measurements show a clear wall-time disadvantage...") per WRITING_LAW §5 (tables introduced by a lead sentence); the two results tables were already introduced by the paper's own discussion paragraphs. | results |
| A8 | Figure: `figure*` (2-col span) → `figure`; `imgs/mtlnet_poi.drawio.png` copied to `figures/cbic_mtlnet_arch.png`; placed at `\textwidth` (the paper's figure* also spanned the full page width, so this is not a stretch; bitmap is 1200 px wide, ~200 dpi at text width — acceptable for a diagram, but flag: a TikZ/drawio re-export at higher resolution would be better; sources `mtlnet_poi_horizontal.drawio` / `_vertical.drawio` exist in `articles/CBIC___MTL/imgs/`). | method figure |
| A9 | Em-dashes and free-standing en-dashes removed per WRITING_LAW §1: challenge-list separators ("Negative Transfer — Unrelated..." etc.) → colons; parenthetical en-dashes in intro ("POIs – specific physical locations ... –") → commas; "Nash–MTL" heading → "Nash-MTL"; "$p_1$–$p_9$" → "$p_1$--$p_9$" (range en-dash kept in TeX form). | intro, basis, method |
| A10 | Commented-out draft blocks (old intro draft, section to-do notes in PT, commented citations) not carried over. Acknowledgment section not carried (frame-level matter). | whole chapter |

## B. Errata applied silently in the text (ERRATA.md non-citation set; each also lands in Appendix B)

| # | Erratum | Applied fix |
|---|---|---|
| B1 | Unfilled dataset placeholders `N_users`, `N_poi`, `N_checkins` in results.tex. | NOT invented. Rendered as visible placeholders "[$N_{\text{users}}$; VERIFY: recompute per ERRATA.md]" (same for poi/checkins) + a % comment block naming the sanctioned path (repo-committed recompute script over the CBIC-era Florida pipeline, author-approved; CoUrb FL row is cross-check only). **AUTHOR ACTION REQUIRED before handoff.** |
| B2 | Prose "almost four times more wall time" contradicts table (80.88 s vs 34.97 s = 2.3×). | Reconciled to the table: "required 80.88 s of wall time, about 2.3 times the cumulative 34.97 s of the individual single-task models." Values 80.88/16.26/18.71 from the published table; 34.97 and 2.3× quoted from ERRATA.md (not agent-computed). |
| B3 | Prose MFLOPs "roughly double" contradicts the table (MTL 0.234 vs Category 2.315 + Next 0.012). | Reconciled to the table: prose now states the table does not show a higher MFLOPs cost for MTL and quotes the three cell values verbatim. Follow-on edits for consistency: "incurred a significantly higher overhead in both time and computational resources" → "in wall time"; "where convergence speed and computational efficiency are critical factors" → "where convergence speed is a critical factor"; "a more convenient and resource-efficient strategy" → "a more convenient strategy"; §Conclusion "higher computational demands in terms of convergence time and MFLOPs" → "in terms of convergence wall time". |
| B4 | Broken cross-ref: `\label{sec:method:single_task_heads}` sat on the Dataset subsection, and method.tex pointed to it as if it described the task heads. | Dataset subsection relabeled `sec:cbic:dataset`; the dangling pointer sentence in §Method ("as detailed in Section~\ref{sec:method:single_task_heads}") dropped: "passed to dedicated, unshared task-specific heads. These heads generate...". No section describing single-task heads exists in the published paper, so no target could be substituted. |
| B5 | Typo "spatio-tegm\nmporal" in basis.tex. | → "spatio-temporal". |
| B6 | Nash-MTL claim discipline (NORTH_STAR §4). | The paper's own §Method sentence ("Nash-MTL consistently yielded a better overall performance") is preserved verbatim as published text; NOT amplified anywhere; the preface carries the one-clause of-the-time caution. |
| B7 | (Noted, not an ERRATA.md item) The published category table bolds the better of MTL/Single per row and never bolds HMRM, even where HMRM is numerically highest (Recall/Nightlife: HMRM 42.13 > Single 36.77 bold). Emphasis preserved exactly as published; the new caption states the convention explicitly so the bolding is not misread as "best per row". | tab:cbic:category |

## C. Writing-law substitutions in preserved prose (minimal, lint-gate-driven; WRITING_LAW §4 banned words)

| # | Original (published) | Chapter text |
|---|---|---|
| C1 | "By leveraging shared information, MTL can potentially mitigate..." (intro) | "By using shared information, ..." |
| C2 | "solutions that leverage shared representations across related POI tasks" (basis) | "solutions that use shared representations..." |
| C3 | "by leveraging shared information across correlated tasks" (basis §MTL-in-POI) | "by sharing information across correlated tasks" |
| C4 | "Moreover, we utilize a Deep Graph Infomax..." / "Moreover, $\mathcal{D}$ is a discriminator..." (method) | "In addition, ..." (both) |
| C5 | "These findings underscore that..." (conclusion) | "These findings indicate that..." |
| C6 | "Furthermore, investigating advanced multi-task optimizers..." (conclusion) | "In addition, investigating..." |
| C7 | "Furthermore, the computational cost ... was roughly double" (results) | superseded by erratum B3 rewrite |
| C8 | "where different sequential patterns might be leveraged" (results) | "might be exploited" |

## D. [VERIFY] / [NEEDS SIGN-OFF] items

1. **[VERIFY — blocks handoff]** B1 dataset statistics: `N_users`, `N_poi`, `N_checkins` must be
   recomputed by the sanctioned repo script and author-approved before the placeholders are
   replaced. Visible in the rendered PDF by design.
2. **[NEEDS SIGN-OFF]** B4: the dangling task-heads cross-reference was resolved by deletion
   (no valid target exists in the published paper). Alternative: point to §3.3.2 (the MTL
   architecture's task-heads paragraph) if the author prefers a pointer over deletion.
3. **[NEEDS SIGN-OFF]** Preface wording, in particular the Nash-MTL clause ("weakened by a
   later finding about the optimizer implementation") — the solver-bug detail (repo memory
   2026-04-10) is deliberately not spelled out; confirm the level of specificity wanted.
4. **[FLAG]** A8: `cbic_mtlnet_arch.png` is a 1200×336 px bitmap (~200 dpi at text width).
   Legible, but a higher-resolution re-export from the .drawio sources would be better for the
   final build.
5. **[NOTE]** Citation errata #1–#9 (ERRATA.md) untouched here; keys `velivckovic2017graph`,
   `velivckovic2018deep`, `nash`, `yu2019mmoe`, `chen2020modeling`, `misra2016cross`,
   `zhang2021survey` etc. appear in this chapter exactly as in the paper and await the Phase-4
   global bib merge.

## E. Numbers ledger (every numeral in the chapter → source)

All performance numbers (per-category F1/precision/recall means ± std, both tables) and all
convergence cells (16.26 / 18.71 / 80.88 s; 3.8 / 3.2 / 3.2 epochs; 2.315 / 0.012 / 0.234
MFLOPs) are copied verbatim from `articles/CBIC___MTL/tables/{category_result,next_result,
converge_result}.tex`. In-prose numbers (62.51±0.94, 46.69±0.81, 57.43±1.46, 28.44±0.42,
53.11±0.58, 47.75±0.89, 43.47±0.50, 43.53±1.66, 64.61±1.11, 22.07±0.52, 26.06±1.01,
22.45±0.81, targets 47 and 32.2) are verbatim from `sections/results.tex`. 34.97 s and 2.3×
are quoted from `articles/CBIC___MTL/ERRATA.md` (sanctioned reconciliation), not computed by
the agent. Method constants (64-d, L_h=9, 9×64=576, seven categories, 5-fold, <5 visits)
verbatim from `sections/method.tex` and `sections/results.tex`. Hardware/software footnote
verbatim from `sections/results.tex`. Dataset counts: placeholders only (B1).
