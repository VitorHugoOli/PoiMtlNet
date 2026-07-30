# LEFT_OUT.md — the register of findings deliberately not put in the text

**Purpose.** Some things this project established are true, verified, and *deliberately absent* from
the dissertation. Without a register, an absence is indistinguishable from an oversight: a later
reader (a reviewer persona, an agent, the author in six months) finds the measurement in a report,
does not find it in the text, and either re-does the work or "fixes" the omission. This file is the
difference between "we did not notice" and "we decided."

**Standing rule.** An entry belongs here when *all three* hold: (a) the finding is established, not
merely suspected; (b) it is absent from the dissertation body, or present in a narrower form than
the finding supports; and (c) the absence is a **decision** with a named decider and a date. A
finding that is absent because nobody got to it belongs in `PENDENCIAS.md`, not here.

**This file is not a hiding place.** Every entry names where the full finding lives, so nothing is
lost by not printing it. If an entry ever needs to say "and we hope nobody asks," it does not belong
in the register; it belongs in the text.

---

## LO-1 · The tuning budget for Chapters 3 and 4

**Finding.** The number of hyperparameter configurations tried for the CBIC and CoUrb studies is
**not recoverable**. No search harness has ever existed in either codebase, run outputs were
gitignored, and only the hand-copied best runs survive.

**What the text says instead.** Neither chapter asserts a configuration count, and the protocol
sentences were written specifically so that they do not imply one. **Since 2026-07-30 the appendix
also states the absence outright, for both studies** (author decision, `PENDENCIAS.md` 2.7:
*"Documentar no letf_out.md e adcionar esse ponto no appendix B"*): the reproduction appendix's
Article 1 section says *"The number of model configurations examined during the study is not
recoverable from the released material, and no sentence of the chapter asserts one"*, and Article 2
now says the same of Chapter 4. Rendered in the supplementary volume, pp. 8 and 9 (read from
`build/main_extra.pdf`, not from the source).

> **Half of this entry's decision was applied and half was not, for a day, which is worth recording
> because it is this project's most expensive defect class.** The Article 1 sentence had been there;
> Article 2 had nothing (`configuration`, `tuning`, `recoverable`, `harness`, `hyperparameter` all
> returned 0 in the comment-stripped Article 2 section), while the finding in
> `10_protocol_recovery.md` §1.4 is explicitly about **both** codebases. A finding credited to a
> whole when only one part moved is exactly `AGENT_GUARDRAILS` §4b V14's second consequence.

**Why it is out.** Because asserting a count would be a claim about conduct with no artifact behind
it. The author's recollection ("we did not change much") is consistent with the code, but a
recollection is not a record, and `AGENT_GUARDRAILS N1` forbids writing a number that cannot be
traced to a source.

**Where the full finding lives.** `src_utils/_round6/10_protocol_recovery.md` §1.4.

**Decided by.** The protocol-recovery pass, 2026-07-28, under the existing number protocol. Not a
discretionary call: writing the number was not available.

---

## LO-2 · The published CBIC joint-model next-category column is not reproducible from this repository

**Finding.** Three of the four published CBIC result columns reproduce **exactly** against committed
run artifacts, 21 of 21 cells each. The fourth, the joint model's next-category column, matches
**no** artifact in this repository: 0 of 21 against the nearest committed run, and a best of 1 of 7
F1 means when tested against every `summary_next_metrics_formatted.csv` ever committed, across both
codebase commits and all six states. The run that produced that column was never committed.

**What the text says instead.** Nothing. No sentence in the dissertation claims the CBIC results are
reproducible from the released code.

**Why it is out.** Because the honest form of this is a negative about one column of a published
table, and it does not change any conclusion: the split code, seed values and checkpoint rule are
identical across the entire window in which all the published runs fall, so the protocol records
cover that run whether or not its outputs survive. Printing it would raise a question about the
published article that the finding itself answers in the article's favor.

**The constraint it imposes.** No future sentence may claim the CBIC numbers are reproducible from
this repository without excluding that column. That constraint is the reason for this entry.

**Where the full finding lives.** `src_utils/_round6/10_protocol_recovery.md` §1.5.

**Decided by.** Recorded 2026-07-28. **Decidido pelo autor (2026-07-30):** *"Documentar no left_out.md"*. Era exatamente isso, e este registro e o desfecho — nao ha divulgacao pendente no Apendice B.

---

## LO-3 · The `\campus` field renders nowhere

**Finding.** `\campus{Campus Florestal}` is set correctly, and is consumed only by
`\imprimircampus` inside `\imprimircapa`, which **neither build calls**. The campus therefore
appears in no output.

**What the text says instead.** Nothing, in either build.

**Why it is out.** Because there is no cover page. The field is correct as data and will start
rendering the moment one is added. Removing it would be worse than leaving it.

**Where the full finding lives.** `src_utils/_round6/15_frontmatter_names.md`; the mechanism is
already commented at `src/0_main.tex:139-141`.

**Decided by.** No decision needed; recorded so it is not "found" again.

---

## LO-4 · The contribution-to-claim table and the consolidated results table

**Finding.** Two of the three "excellence moves" proposed for the SBC-CTD lens were measured against
the five exemplar dissertations: **zero of five** carry a contribution-to-claim table, and **one of
five** carries a consolidated cross-chapter results table.

**What the text has instead.** Neither table. The third move, a reproducibility appendix, had **four
of five** support and is the one being written.

**Why they are out.** The author's words: *"Eu acho que o A e o com menor ganho"*, and the
measurement agreed with his instinct. In a collection-format dissertation the
contribution-to-chapter mapping is explicit by construction, and Chapter 5's results table already
is the consolidated result.

**Where the full finding lives.** `src_utils/PENDENCIAS.md` §2b.3, with the five-exemplar table.

**Decided by.** The author, 2026-07-27: *"Concordo, vamos com a opc: C)."*

---

## LO-5 · The rotated placement of the Chapter 5 data-flow figure

**Finding.** `fig1_dataflow` renders its labels at 7.93 pt against an 11.96 pt body, 66 percent.
`WRITING_LAW §5` asks for in-figure text near body size. A rotated full-page placement was **built
and measured** at 10.53 pt, 88 percent, building clean at `tex_errors=0` with zero overfull boxes.

**What the text has instead.** The upright placement at 66 percent, with the item left flagged.

**Why it is out.** The author chose the upright placement when asked directly, on the grounds that
rotating the page turns the diagram's deliberate left-to-right data flow into a bottom-to-top one
and costs the reader a page turn. The presentation cost was judged higher than the type-size gain.

**Where the full finding lives.** `src_utils/_round6/12_figures.md` §3, with both measurements.

**Decided by.** The author, 2026-07-28, answering a direct question with all three options measured.

---

## LO-6 · The label size of BOTH published architecture rasters (Ch.3 and Ch.4)

**Finding.** The two raster architecture figures carried over from the published papers print their
labels at roughly **45 percent of body size**, and they are the two smallest in the document:

| Figure | File | Raster | Estimated label type | Percent of 11.96 pt body |
|---|---|---|---:|---:|
| Ch.3, MTLnet architecture | `figures/cbic_mtlnet_arch.png` | 1200 x 336 | 5.42 pt | **45.3** |
| Ch.4, ST-MTLNet architecture | `figures/courb/arquitetura_modelo.png` | 1102 x 348 | 5.31 pt | **44.4** |
| Ch.5, `fig2_model` (rescaled this round) | `figures/mobiwac/fig2_model.pdf` | vector | 11.15 pt | 93.2 |
| Ch.5, `fig1_dataflow` (rescaled this round) | `figures/mobiwac/fig1_dataflow.pdf` | vector | 7.93 pt | 66.3 |

**So the two figures the audit tracked under COD-017 are the two LARGEST of the four.** The two it did
not track are half body size. Measured independently twice: the visual persona reported 44.2 percent
for the Ch.3 figure from raster cap-ink at threshold 100, and a separate measurement here on modal
glyph height gave 45.3 percent. Both instruments agree the figure is at about 45 percent; neither
number is a font size the file declares, because a raster declares none.

The Ch.3 figure is **byte-identical** to `articles/CBIC___MTL/imgs/mtlnet_poi.drawio.png` (sha256
matches, 1200 x 336 both), so it is the published artifact exactly as published.

**What the text has instead.** Both figures at their published label size. The Ch.4 one carries this
round's six Portuguese-to-English label translations and nothing else; the Ch.3 one is untouched.

**Why it is out.** The sanctioned change to a published figure was the *language* of six labels in one
of them. Raising type size is a presentation change to a published artifact, and for Chapter 4 to a
co-authored one. Neither was authorized. A `.drawio` source exists for both
(`articles/CBIC___MTL/imgs/mtlnet_poi_{horizontal,vertical}.drawio` and the CoUrb source), so the fix
is available and cheap: raise `fontSize` from 13 to about 20 and re-export at the same pixel width.
What is missing is the decision, not the capability.

**Where the full finding lives.** `src_utils/_round6/12_figures.md` for the Ch.4 figure and the fix
recipe; `src_utils/_round6/18_visual_ufv_latex.md` finding V-1 for the Ch.3 figure and the
measurement method.

**Decided by.** Deferred, not declined. **Open for the author, and now covering both figures** --
this entry originally recorded only Chapter 4, which left the smaller of the two unregistered.
Extended 2026-07-28 after the visual pass measured it.

---

## LO-7 · The sub-area selection rule behind the state-distribution figure

**Finding.** The published figure is not reproducible from the corpora, because the rule that
selected each panel's sub-area is recorded nowhere in either tree. Each panel's window holds 531 to
703 Food and Shopping POIs, the caption says about 100 per region, and marker counting suggests
roughly 163 to 307 plotted.

**What the text has instead.** The published figure, unchanged. It needed no change: its labels are
already fully English, which was the reason it was examined.

**Why it is out.** The pass failed closed rather than shipping a regenerated lookalike. A figure that
resembles the published one but was produced by a different, invented rule would be worse than one
that cannot be regenerated.

**Where the full finding lives.** `src_utils/_round6/12_figures.md` §2.

**Decided by.** The figures pass, 2026-07-28, under the fail-closed rule.

---

## LO-8 · The preamble that produced the two Chapter 5 figure PDFs

**Finding.** The TikZ sources for both figures **do exist** in the MobiWac folder, contrary to the
premise this round started from. But they compile 6 to 8 pt wider than the committed PDFs under
every preamble variant tried, so the exact build that produced the committed bytes is not
recoverable.

**What the text has instead.** The committed PDFs, with only their placement changed.

**Why it is out.** Regenerating at a larger type size would have changed the figures' geometry, and
at 11 pt one annotation was measured crossing 2.18 pt into an adjacent box. Placement was the
change available without redesigning a submitted paper's figure.

**Where the full finding lives.** `src_utils/_round6/12_figures.md` §3.

**Decided by.** The figures pass, 2026-07-28.

---

## LO-9 · The conditions on the Nash-MTL guarantee, beyond the two the paper names [!]

**Finding.** `arXiv:2202.01017v2` was fetched and read this session (19 pages). Its guarantee is real
and it is **conditional**, on more than the dissertation states:

1. **Assumption 5.1, p.6** (informally p.3): "if $\theta$ is not Pareto stationary then the gradients
   are linearly independent". Claim 3.1's characterization of the solution is stated only "if $\theta$
   is not on the Pareto front".
2. **The solution is approximated, not solved.** p.4: the $\alpha$ is obtained by a concave-convex
   procedure limited to 20 iterations. The exact Nash bargaining solution is not computed.
3. **The authors' own hedge, p.6:** "Since our update rule is a descent direction for all tasks, we
   can reasonably assume that our algorithm avoids local maxima points." *Reasonably assume*, not
   prove.

**What the text says instead.** The dissertation names the **two conditions the paper itself
foregrounds** and stops. It does not claim the guarantee is unconditional, and it does not import the
CCP approximation cap or the authors' hedge.

**Why it is out.** Because the two named conditions are what a reader needs in order not to
over-read the method, and the remaining detail is a discussion of someone else's method that would
not change any decision in this dissertation. Naming all three in the frame would give an
external method more space than this document's own contribution.

**A separate flag inside this entry.** `src/references.bib:741` carries `pages = {16428--16446}` for
the ICML 2022 record. OpenAlex returns **only** the arXiv preprint (`W4225981399`, type `preprint`,
no page range), so that page range is **not confirmed by any source of record reachable this
session**. The venue itself is confirmed by the paper's own arXiv comment field. This is a
`[VERIFY]`, not an error: the pages may well be right.

**Where the full finding lives.** `src_utils/_round6/10_protocol_recovery.md` §3.1.

**Decided by.** The protocol-recovery pass, 2026-07-28, applied at `1fa930e0`.
**Open for the author** only on the page range.

---

## LO-10 · The provenance-relocation option for the comment volume

**Finding.** Comment volume was measured by block: 1,217 of 1,269 comment lines (95%) carry a
traceable fact, and the fact-free remainder is structural banners plus the author's own sign-off
queue. Two reductions were identified that lose nothing: **41 purely decorative rule lines**, and
**moving the provenance blocks** out of the `.tex` files into per-chapter files under `src_utils/`
with a one-line pointer at each site. The second would remove far more volume than the first.

**What the text says instead.** Neither was applied to the chapters. The decorative lines are being
removed in round 7's comment pass; the relocation was **not** proposed for application.

**Why it is out.** Because relocation trades away the property that makes the provenance work. A
comment beside its value is read by whoever edits that value; a file one directory away is not. The
number protocol (`AGENT_GUARDRAILS` N3) depends on that adjacency, and it is what caught the
wrong-quantity defect recorded inside one of those very blocks. The `tables/` reorganization did the
same move successfully, but there the hoisted text was **identical** across 16 files, so nothing was
separated from anything.

**Where the full finding lives.** `src_utils/_round6/14_comments_measured.md`.

**Decided by.** Recorded 2026-07-28; recommended against. **Open for the author** if he wants the
volume gone and accepts the trade.

---

## LO-11 · The author's per-role credit for the CoUrb paper

**Finding.** The author holds three roles in the CoUrb paper that the dissertation does not state
anywhere. In his own words, answering COD-018 (`PENDENCIAS.md` §5.8): *"Meu papel no courb foi na
implementação, auxilo ao meu aluno de graduação na sua pesquisa pelos modelos de embedding, e escrita
da parte do MTL e parte da conclusão."* That is implementation, supporting his undergraduate
student's research on the embedding models, and writing the multi-task learning section plus part of
the conclusion.

**What the text says instead.** The Chapter 4 preface states the roles that are matters of public
record: Tarik S. Paiva is first author, the author of this dissertation is second author, presented
the paper at the workshop, and is the first author of the baseline model MTLnet
(`chapters/4_courb.tex`:19). Appendix A describes the platform and the ETL and attributes no
per-function role for CoUrb. So the credit is present in a narrower form than the finding supports.

**Why it is out.** The author decided so, twice. His `PENDENCIAS.md` §5.8 answer is *"Não precisa
mexer nisso, pode remover essa preocupação."* On 2026-07-30 a round-8 track was briefed to add it
anyway; the track stopped and asked rather than writing it, and he chose to honor the recorded
decision. Two reasons this is his call and not an agent's: authorship credit on a co-authored paper
is a claim only he can make (`AGENT_GUARDRAILS` C2), and naming an undergraduate student in the
dissertation is a decision about a third party.

**Consequence for the round-8 gate, so nobody reads it as a silent narrowing.**
`src_utils/check_audit_claims.py` carried a probe asserting this credit was PRESENT, which was
written from the audit's expectation rather than from his decision. The probe is retired into that
file's `RETIRED` table, which prints on every run with his quote as the reason, so the gate reports
the withdrawal instead of dropping it. Measured after the change: 8 of 8 probes hold, 1 withdrawn,
and a sabotage test (flipping one probe's expectation) still makes the gate exit 1.

**Where the full finding lives.** `PENDENCIAS.md` §5.8, which quotes him verbatim, and the
`RETIRED` entry in `src_utils/check_audit_claims.py`.

**Decided by.** The author, in `PENDENCIAS.md` §5.8; reconfirmed 2026-07-30. **Reversible at any
time:** the input is one sentence in Appendix A, matched to the register the Chapter 4 preface uses,
and it would carry `[NEEDS SIGN-OFF: COD-018]`.

---

## How to add an entry

Copy the shape above: the finding, what the text says instead, why it is out, where the full finding
lives, and who decided with the date. If you cannot fill "decided by," the item is a pendency, not an
omission, and it goes in `PENDENCIAS.md`.
