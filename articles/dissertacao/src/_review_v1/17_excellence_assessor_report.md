# Reviewer 17 — Excellence Assessor — Complete-v1 Scorecard

> Scope: full defense build (`src/main_defense.pdf`, 87 pp) + all chapter/appendix source.
> Method: score against the positive OUTSTANDING descriptors (evidence file
> `docs/research/dissertation_excellence_2026-07-20.md`), never by counting flaws.
> Persona: `reviewers/17_excellence_assessor.md`. Read-only.
> Status: **COMPLETE** — scorecard, gap report, award lens, protect list, and verdict below.

## Reading log (what I have read firsthand this session)
- Persona 17 + reviewers/README common protocol — done
- Rubric evidence file (Lovitts / Mullins&Kiley / Holbrook&Bourke / Sharmini / ACM / SBC CTD / CAPES) — done
- NORTH_STAR (arc, spine, chapter map, errata) + CLAUDE.md ledger — done
- `0_main.tex` (preamble, Resumo/Abstract, siglas, structure) — done
- Ch.1 Introduction (`chapters/1_introduction.tex`) — done
- Ch.2 Fundamentals (`chapters/2_fundamentals.tex`) — done
- Ch.6 Conclusion (`chapters/6_conclusion.tex`) — done
- Ch.3/4/5 prefaces + bridges; appendices A/B/C — PENDING
- Rendered PDF spot pages — PENDING

All chapters + appendices + front matter read firsthand from source; defense PDF (87 pp)
spot-rendered (front matter, TOC, Ch.2 lineage table p.20, Ch.5 results table p.67 + deltas
figure p.68, Ch.6 limitations p.74). Prior reports 05 (citations) and 12 (banca) are empty
stubs this session — nothing to defer to.

---

## 1 · THE SCORECARD

Scored against the OUTSTANDING positive descriptors (evidence file), not by counting flaws.
Calibrated for a master's coletânea (Bourke & Holbrook 2013: same dimensions as PhD, contribution
threshold proportionately lower).

| # | Dimension | Score | One-line evidence |
|---|-----------|-------|-------------------|
| 1 | Problem framing & significance | **OUTSTANDING** | Quotable load-bearing question, bold inline in §1.2 and mirrored in the Abstract; §1.1 argues why-it-matters beyond the lab (urban planning, disease, pollution via `luca2021mobilitysurvey`) and anchors on the 93% predictability ceiling. Not framed as an extension of the advisor's work. |
| 2 | Contribution clarity & unity (coletânea-critical) | **GOOD** | All three unity mechanisms present and well-built (intro states a claim no paper makes alone; time-capsule prefaces bridge chapters; §6.2 answers the RQ). BUT the four-group contribution taxonomy in §1.6 (Theoretical/Software/Empirical/Practical) is never reconciled in §6.1 (which narrates by chapter), and no contributions→claims mapping table exists — the unity lives in prose, not in the one artifact that makes it examiner-legible at a glance. |
| 3 | Command & critical use of literature (chapter-2 test) | **OUTSTANDING** | §2 takes positions, not a catalog: "This lineage is background for the present work, not its target" (§2.1); "a fixed-weight baseline is a serious competitor, and a balancer earns its place only by outperforming it" (§2.3). The model-lineage table is a synthesis artifact; §2.5 maps three gaps 1:1 onto Ch.3/4/5. 99 refs, foundational + 2024/2025 current. |
| 4 | Methodological rigor & justification | **OUTSTANDING** | "we chose A over B because…" everywhere with advantages AND disadvantages: parallel-vs-cascade coupling *tested* not asserted (§5.6.2, honestly noting CSLSL reports the opposite on its own benchmarks); class-weighting disadvantage disclosed; guards designed in (per-fold prior, label-free objective, train-users-only rebuild). |
| 5 | Statistical & empirical rigor (ML) | **OUTSTANDING** | n=20 (4 seeds × 5 folds), ± sd in every headline cell, paired-t + 90% CIs + Holm + pre-registered TOST; budget-matched capacity baseline (4.2M vs 0.6M, §6.2); self-downgrading generality ("a finding for this pair of tasks, not a general rule"; "we read the trend across the points rather than as a precise law"). Exceeds most published work. |
| 6 | Originality & insight | **OUTSTANDING** | The named reframing "the representation, not the architecture, is the bottleneck" is delivered and threaded across all three frame chapters; Check2HGI (four-level infomax down to the check-in) is a genuine new tool. Field-level consequence stated modestly (correctly, per claim discipline) — delivery can sharpen without widening scope. |
| 7 | Critical self-assessment & honest negatives | **OUTSTANDING** | The dissertation's superpower, delivered as reframing not summary: "The negative result was not an obstacle… worked through, it was the contribution's first half" (§6.5); the task-pair-confound limitation (§6.3.6) could only be written by someone who ran the experiments; the BRACIS correction-trail (Appendix A) confronts a contradiction between own works (fp16 artifact) head-on. |
| 8 | Reproducibility & artifacts | **GOOD** | The work is genuinely reproducible (pinned configs, released code, public figshare data, pre-registered analysis plan). But the products are scattered across three GitHub footnotes + one figshare link + one data cite; no reproducibility statement/appendix, no consolidated per-chapter repo/artifact inventory, no dataset version/hash in the text — the rubric's OUTSTANDING evidence ("a reproducibility statement or appendix; repository links per chapter") is missing. |
| 9 | Writing, structure, voice | **GOOD** | Confident, economical, point-of-view prose; visible spine (§2 opener, §2.5 gap→chapter map, prefaces); clean render. Held from OUTSTANDING by the notation-dialect seam — the same joint-vs-dedicated contrast reads as MTL/Single (Ch.3), MTLNet/baseline (Ch.4), Joint/Dedicated (Ch.5); the frame uses canonical terms but the reproduced chapters keep per-venue dialects with no bridge. (Abstract ~250 words vs the ≤200 ideal — nit. Title placeholder in the built PDF is an owned open decision — out-of-scope handoff below.) |
| 10 | External validation & impact trail | **GOOD** | Co-authorship ownership note handled well (the load-bearing element per Sharmini's 69%): CoUrb contribution stated in §1.5 AND the Ch.4 preface. Venues appropriate for a CS master's (2 published + 1 ACM-symposium under review). But no CTD-style consolidated products list, and the relevance argument (§1.1) is not carried into an explicit impact close in §6. |

**Cross-cutting test A — the chapter-2 test (authority by end of the literature review):** **PASS.**
§2 establishes command — it takes evaluative positions, builds the lineage taxonomy, and its gaps
map 1:1 onto the research questions. Examiner authority is won here, which is the decisive first
impression (Mullins & Kiley).

**Cross-cutting test B — the intro-conclusion loop test:** **PASS with one seam.** The RQ is answered
verbatim and directly (§6.2 opens "Does multi-task learning help point-of-interest prediction? …
conditional, and the condition is the finding"); objectives 1-3 → §6.1 chapter paragraphs, objective
4 → §6.2 controls. The one seam: the §1.6 four-group contribution taxonomy is not reconciled in the
conclusion, so an examiner tracing contributions specifically finds the intro's frame (four *kinds*)
and the conclusion's frame (three *chapters*) do not line up.

**Profile: 6 OUTSTANDING · 4 GOOD · 0 BELOW.** Every GOOD is in the delivery/packaging register
(unity artifact, reproducibility inventory, notation unification, products list); the science is
uniformly OUTSTANDING. This is the textbook "very good → outstanding" gap: outstanding substance
with an outstanding core (dimension 7), seams in how unity and products are *packaged* for the
examiner.

---

## 2 · THE GAP REPORT (moves to lift each non-OUTSTANDING dimension)

Ranked by leverage-per-hour. All moves are delivery/synthesis or packaging — none touches the
science or widens any claim (hard limit). The frame dominates, as expected.

### ★ TOP-3 MOVES (highest leverage)

**★1 — Add a contributions→claims mapping table in §1.6, and close the taxonomy loop in §6.1.**
*(Dimension 2; serves "a contributions table mapping papers → thesis-level claims" and Lovitts
"connects components in a seamless way".)*
Where: §1.6 (a small table: rows = Ch.3/4/5 + consolidation; columns = the thesis-level claim each
instantiates, the artifact, the status). Then in §6.1, name the four contribution *kinds* against
what was delivered so the intro frame and the conclusion frame line up.
Why highest: the chapter-2 authority is already won; what remains on the table is the *whole-document
unity judgment*, and right now that judgment must be assembled by the reader from prose. One table
makes "these three papers are one argument" legible at a glance — exactly the coletânea-critical
signal. Cost: ~1-2 h, ~0.5 pp.

**★2 — Add one consolidated cross-chapter results view.**
*(Dimension 2 + dimension 5 delivery; this is THE signature missed-connection — "a consolidated
cross-chapter results perspective" the structure begs for but the text does not run.)*
Where: §6.1 or §6.2, a compact table showing, per chapter, the headline joint-vs-dedicated (or
joint-vs-single/baseline) contrast in one place — MTL≈Single (Ch.3, place-level), the +20.2…+22.0 pp
category jump (Ch.4, decomposed input), the +5.3…+9.4 category / 4-of-6 region result (Ch.5,
check-in level). Numbers quoted from each chapter's own source of truth, nothing recomputed.
Why: the arc's punchline — representation dominates — is currently provable only by flipping between
three result sections. A single view lets the examiner *see* the monotone climb. Cost: ~1-2 h, ~0.5 pp.

**★3 — Add a short "Artifacts and reproducibility" appendix (or §6 subsection).**
*(Dimension 8 + dimension 10 + CAPES III + the CTD products list — triple-counts.)*
Where: new Appendix D (or fold into §1.6 Software). Consolidate: the three repository URLs + the
MobiWac branch, the figshare dataset + its version, Massive-STEPS, the seed set (n=20 = 4×5), and the
location of the pre-registered analysis plan. One paragraph + a small table.
Why: the work IS reproducible; the committee just cannot find the evidence in one place. This is the
single move that most raises the two award-lens axes at once (see §3). Cost: ~1-2 h, ~1 pp.

### Remaining moves

**4 — Notation bridge across the three dialects.** *(Dimension 9 unity.)* One clause per paper-chapter
preface (or a single line in §2.3) mapping each paper's vocabulary onto the canonical pair — "what
this article calls MTL vs Single (Ch.3) / MTLNet vs baseline (Ch.4) is the joint model vs the
dedicated single-task model of this dissertation." Fidelity-safe: touches only frame/preface prose,
never the reproduced tables. Cost: ~0.5 h.

**5 — Sharpen the field-level reframing (delivery only).** *(Dimension 6 delivery.)* In §6.2/§6.5,
state the portable lesson once — that representation granularity should match the unit being predicted
— explicitly bounded to these POI tasks and this representation line (no scope widening). Turns a
reported finding into a lesson the field can carry. Cost: ~0.5 h. Optional.

**6 — Trim the Abstract toward ≤200 words.** *(Dimension 9, nit.)* It hits all four beats
(problem-approach-result-implication) but runs ~250 words. Cost: ~0.5 h.

---

## 3 · THE AWARD LENS — could this compete at SBC CTD?

**Answer: yes, with edits — and the edits are exactly TOP-moves ★1-★3.**

(a) **Can the problem → contributions → impact story be told in a 10-page summary?** *Yes, already.*
The frame (Ch.1 + Ch.2 + Ch.6 ≈ 13 pp) carries a crisp arc; a CTD extended abstract is largely
extractable from Ch.1 §1.1-1.6 + Ch.6 §6.1-6.2. The one-sentence contribution exists and is quotable.

(b) **Is the products list where a committee finds it?** *No — this is the concrete gap.* CTD/SBRC
require a byproducts document (papers, software, datasets); here they are three footnotes and a
figshare link. Move ★3 closes this directly and is the highest-value single edit for CTD readiness.

(c) **Do originality and relevance (the double-weighted CAPES axes I+II) get argued explicitly, or
only implied?** *Originality: explicitly* (the reframing, delivered and threaded). *Relevance:
argued in §1.1 but not carried into an explicit close* — add one relevance sentence to §6 (which
downstream services the two predictions enable, at neighbourhood scale, per the scoped §5.7 framing)
so criterion II is stated where a committee scores it, not just implied.

The venue tier (ACM MobiWac international + two published Brazilian venues) plus demonstrable
candidate ownership (the CoUrb note) already function as pre-scored evidence on CAPES I/II/VI. Whether
to actually submit to CTD is the author's decision and out of scope; the test is a quality instrument,
and the verdict is that three low-cost frame edits move this from "competitive" to "committee-ready."

---

## 4 · THE PROTECT LIST (already outstanding-grade — do NOT dilute in any edit)

1. **The time-capsule prefaces** (each paper chapter, `0_main.tex` `chapterpreface` env). The single
   mechanism that makes three differently-shaped papers read as one document. Do not trim or merge.
2. **The honest-arc narration** — §1.2 ("the journey is the contribution") and §6.5 ("worked through,
   it was the contribution's first half"). This is the dissertation's rarest asset (dimension 7). Do
   not soften into neutral summary.
3. **The claim-verb discipline** — outperforms/matches bound to tests, AZ never upgraded, "read the
   trend… rather than as a precise law." Do not "smooth" into stronger verbs for readability; the
   restraint is what an expert committee rewards.
4. **The six concrete limitations, especially §6.3.6 (the task-pair confound).** The mark of someone
   who ran the experiments. Do not generalize into boilerplate caveats.
5. **The BRACIS correction-trail (Appendix A).** Confronting a contradiction between own works is
   excellence evidence, not a weakness to hide. Keep the "corrected by MobiWac" framing.
6. **The capacity-matched baseline + freeze control + leak audit** (Ch.5 §5.5.2, §6.2). Award-grade
   rigor. Do not compress away when trimming for length.
7. **The Ch.2 evaluative positions** — the fixed-weight-baseline stance (§2.3) and the region-as-end-
   target gap. This is where authority is won; do not flatten into a neutral catalog.
8. **The CoUrb ownership note** (§1.5 + Ch.4 preface). Load-bearing co-authorship disclosure; keep verbatim.

---

## 5 · OUT-OF-SCOPE HANDOFFS (not my gate; one line each)

- **Title is a placeholder in the built PDF** (`[TITLE — OPEN DECISION NORTH_STAR §5.8]`, p.1 + Resumo/
  Abstract headers). Owned open decision; must clear before the banca build or it becomes a sloppiness-
  cascade trigger. → persona 13 / author (working title in `1_introduction.tex` header is strong and
  names both factors).
- **CBIC dataset placeholders render visibly in the body** (`[$N_{users}$; VERIFY…]`, Ch.3 §3.4.1;
  errata Table B.1 row 5 = "pending"). → persona 06 number gate / author recompute.
- **Prior reports 05 (citations) and 12 (banca) are empty stubs** — those gates have not yet completed.

---

## 6 · VERDICT (Lovitts' terms) + the single highest-leverage investment

This dissertation is on an **outstanding trajectory**, currently reading as **very good with an
outstanding core.** The science is uniformly outstanding — the empirical rigor (pre-registered TOST,
budget-matched capacity baseline, freeze control, leak audit) exceeds most published work, and the
critical-self-assessment dimension is the rare, genuine article: a published null result, its
mechanism-level diagnosis, and a correction trail narrated as "what we got wrong and how we found
out." The chapter-2 authority test is passed. What holds it at "very good" at the whole-document level
is not any weak component (there is no BELOW) but a set of *packaging* seams — the unity of the three
papers is delivered in prose rather than in the one or two synthesis artifacts (a contributions→claims
map, a consolidated results view) that would let an examiner *see* it at a glance, and the reproducible
products are scattered rather than inventoried. These are exactly the "misses opportunities to explore
connections" and "uneven components" patterns that demote outstanding to very good — and every one is
low-cost, frame-only, and claim-safe.

**The single highest-leverage remaining investment: the frame unity artifacts (moves ★1 + ★2) — a
contributions→claims mapping table and a consolidated cross-chapter results view.** The authority is
already won in Chapter 2; what is left on the table is the whole-document synthesis judgment, and two
small tables in the frame convert an arc the reader must currently assemble into one the examiner is
handed. Roughly half a day of work, no new claims, and it is the difference between "three strong
papers with a good wrapper" and "one document that changes how you read the three."

