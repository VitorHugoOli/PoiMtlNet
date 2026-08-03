# Round 11, item 32 — GER-02, second part: improve the explanation of HGI and how it works

**Report path:** `src_utils/_round11/32_hgi_explanation.md`
**Baseline commit:** 28df0cc0
**Date:** 2026-08-03
**File edited:** `src/chapters/2_fundamentals.tex`, and nothing else.

---

## 1 · The item and the author's ruling, verbatim

Germano's remark, as the author transcribed it (a paraphrase of a verbal comment, never his
written words), followed by the author's own decision:

> "De fato isso ja ta bem explicado no texto, entao nao vamos mexer. Algo que podemos melhorar e
> explicacao do hgi e como ele funciona, vide que essa e uma das aboardagens mais importantes para
> as contribuicoes da dissertacao."

and today:

> "Quanto ao FAB-15 e o GER-02 segunda parte, tambem podemos seguir com isso"

and, on the page budget:

> "pode melhorar o texto da fundamentacao sem preocupacao de paginas, vamos remover algumas coisas
> de appendix em sequencia."

The FIRST half of GER-02 is settled as "do not touch" and was not touched. Only the HGI
explanation is in scope.

---

## 2 · What the LIVE text said before the edit

Located in the live file immediately before writing (not from the brief's line numbers, which were
correct here: the subsection head was at `:206`). The block that changed, quoted exactly:

```
used in the first study. Hierarchical Graph Infomax (HGI) extends that objective
across POI, region, and city levels~\cite{huang2023hgi}. It begins with category
embeddings, updates each POI from its spatial neighborhood, and aggregates POIs into
regions with multi-head attention. A region graph then propagates information between
adjacent regions, and region representations are aggregated into a city summary. The
training objective brings POIs into agreement with their regions and regions into
agreement with the city while contrasting corrupted pairs. This hierarchy lets a
region representation carry both local POI information and city-level context.

HGI was developed and evaluated for urban region representation. This dissertation
repurposes its POI-level output for sequential prediction. In this role, HGI supplies
the place-level baseline in the later studies and the direct basis of Check2HGI.
```

Everything above that block (the opening purpose sentence at `:208-209`, the MINE / Deep InfoMax /
DGI lead-in at `:211-217`) is unchanged.

**Nothing in the old text was wrong.** No clause of it is contradicted by the new text; every
statement it made survives in substance. It was terse, which is what the author asked to fix.

---

## 3 · What changed and where

`src/chapters/2_fundamentals.tex`, subsection "Hierarchical graph infomax". The prose now runs
`:206` to `:276` (was `:206` to `:228`); the source-comment ledger runs `:277` to `:355`. The
subsection that follows, "From a fixed place vector to visit context", now begins at `:366`.

Six paragraphs replace the two. In order:

1. **`:220-234` — why the mechanism matters here, and what "infomax" means concretely.** Names
   HGI's own stated goal (unsupervised urban region representations from POIs), the extension of
   DGI's two scales to three, and then the thing the old text left implicit: *what* is maximized
   (mutual information between representations at two adjacent levels), and that the quantity is
   never evaluated in closed form. HGI is contrastive and scores pairs. Names the bilinear
   discriminator and the logistic function (both registered GLOSSARY terms, already used by the
   Check2HGI equations later in this same section), and states that no downstream label enters the
   comparison.
2. **`:236-248` — the five stages, so a reader knows what the four levels *are*.** Pretrained
   category encoder for the initial features; the Delaunay POI graph and the direction of its edge
   weights; one graph convolution layer and what it produces; multi-head attention with one head
   per perspective; the region graph over border-sharing regions; the area-weighted city
   embedding.
3. **`:245-251` — the paper's own two examples, which answer "why a hierarchy at all".** The two
   hotels (POI level) and the two company-plus-restaurant regions (region level). These are the
   source's examples, attributed as such in the text.
4. **`:253-260` — what the discriminator compares and what a negative sample is.** One term per
   adjacent level pair; the POI-region term's true and false pairs; the region-city term's false
   pair built by shuffling the POI feature matrix row by row, described concretely (every POI keeps
   its place in the graph and receives another POI's category embedding).
5. **`:261-270` — how a POI embedding ends up carrying regional context.** This is the property the
   dissertation later leans on and the one the old text did not explain. Two routes: the joint
   training of the three modules against a region-membership objective, and region membership
   entering earlier through the POI-graph edge weights. Closes with the authors' own property claim
   for their own product (region representations both locally and globally relevant).
6. **`:272-276` — the honest caveat, kept and made specific.** The old two sentences are retained
   in substance and gain their evidence: the reported experiments (urban functional distributions,
   population density, housing price; Xiamen Island and Shenzhen), the fact that the evaluated
   object there is the region embedding and the POI embedding is an internal stage, and the
   explicit statement that this dissertation's use is one the original evaluation does not cover.
   The final sentence ("In this role, HGI supplies the place-level baseline in the later studies
   and the direct basis of Check2HGI") is unchanged word for word.

One wording fix after the first render: the examples paragraph originally ended "the region graph
is what makes the difference visible", a cleft of exactly the shape WRITING_LAW §1 bans ("What
carries that diagnosis is..."). Rewritten to "the region graph makes the difference visible", and a
neighboring "which is the work of the POI level" tail became the plain sentence "The POI level does
that work." Both were caught by reading the rendered page, not by a gate.

---

## 4 · Sources opened this session

**One external source, on disk, opened in full.** The budget allowed six; five went unused, which is
disclosed here rather than padded.

| Source | Identifier | Where opened | What was located |
|---|---|---|---|
| Huang, Zhang, Mai, Guo, Cui, "Learning urban region representations with POIs and hierarchical graph infomax", *ISPRS Journal of Photogrammetry and Remote Sensing* 196 (2023) 134–145 | `doi:10.1016/j.isprsjprs.2022.11.021`; bib key `huang2023hgi`, already cited in the chapter | `science/articles/Learning urban region representations with POIs and hierarchical graph infomax.pdf`, on disk in this repo; extracted with pypdfium2, 12 PDF sheets, journal pages 134–145 | every mechanism claim below |

**Not reached, and named rather than skipped:** the HGI reference implementation at
`github.com/RightBank/HGI` (the paper's own link, p.134 and p.136); Huang et al. 2022, the
semantics-preserved POI category encoder HGI uses as its stage 1; Lee et al. 2019 (set transformer),
the source of the AGG_poi-region form; Veličković et al. 2019 (DGI), which the chapter cites and
which a prior round verified; and Wang & Isola 2020 (alignment and uniformity), which HGI invokes at
p.138. None of them is needed for a claim written here: every sentence attributes a property of HGI
to HGI's own paper.

### 4.1 · Claim ledger — every mechanism claim to its page

Journal pagination cited; PDF sheet number in parentheses. This ledger is duplicated verbatim in the
source-comment block at `2_fundamentals.tex:277-355`, matching the standard of the Pareto block at
`:518-573`.

| Claim in the new prose | Page | Located text |
|---|---|---|
| stated goal, fully unsupervised | p.134 (sheet 1), abstract | "learning urban region representations (vector embeddings) with points-of-interest (POIs) in a fully unsupervised manner" |
| extends DGI's two scales to three | p.136 (sheet 3), end of §2 | "we extend DGI to model the interactions between three scales, i.e., POI, region, and city" |
| what is maximized | p.134 abstract; p.139 (sheet 6) | "trained through maximizing the mutual information among the POI – region – city hierarchy"; "a region-centered hierarchical mutual information maximization" |
| contrastive, not a closed-form MI estimate | p.138 (sheet 5) §3.7 | "The general idea of HGI follows the paradigm of contrastive learning". The MI is never evaluated in closed form: Eq. 12–13 (p.139) are log-scores of the discriminator and nothing else |
| bilinear discriminator + logistic function | p.139, below Eq. 13 | `D_pr(p_i, r_k) = sigma_L(p_i W_pr r_k)`, "sigma_L is a sigmoid function" (prose says "logistic function" per GLOSSARY §3) |
| no downstream label enters | p.138–139 §3.7 | "The training of HGI only relies on POIs and region boundaries, and not on any ground truth data in the downstream tasks." |
| five stages | p.136 (sheet 3) §3.1, items (1)–(5); same list in the abstract | — |
| pretrained category encoder, and why | p.136 §3.1(1) | "A POI category encoder phi_c is pretrained to generate the initial POI features, as categorical information generally plays a key role in defining the meaning of a POI." Paraphrased in prose to avoid the banned "plays a crucial role" template shape |
| Delaunay graph; edge weights fall with distance and are reduced across regions | p.137 (sheet 4) §3.3, Eq. 2 | "all the POIs in a study area are connected using DT to form a graph"; `a_p = log((1+L^1.5)/(1+l^1.5)) x w_r`, "l represents the spatial distance between the two POIs ... w_r is a factor to differentiate intra- (w_r = 1) and cross-region (w_r = 0.4) edges" |
| one graph convolution layer; what it produces | p.137 §3.3; p.138 (sheet 5) top | "we apply a one-layer GCN encoder"; "the embedding of each POI is updated to be a transformed combination of the information from itself and its spatial context, which captures its uniqueness" |
| one head per perspective | p.138 §3.4, closing sentence | the region raw embedding "reflects the different importance levels of the POIs from several perspectives (each head represents a perspective)" |
| region graph over border-sharing regions | p.138 §3.5 | "we conceptually view each region as a node, and build edges between the regions that share parts of borders"; "We employ a one-layer GCN" |
| area-weighted city embedding | p.138 §3.6, Eq. 10 | "area-weighted summarization"; `aw_i` "is the area proportion of region r_i in the study area (city)" |
| the two-hotels example | p.137 §3.3 | "a hotel within an airport complex and another hotel in a university campus ... have diverse semantics" |
| the two-regions example | p.135 (sheet 2), end of §2.1 | two regions each with a company and a restaurant, one "surrounded by commercial regions", the other "surrounded by several industrial areas with factories"; they "can hardly be differentiated unless their contextual information is incorporated" |
| positives and negatives, both terms | p.139 §3.7 | for a region, positives are "the POI embeddings P_i within the region", negatives "the POI embeddings P_j in another region r_j"; for the city, positives are the region embeddings R, negatives come from "row-wise shuffling of the POI graph's feature matrix X_p ... (replace the category embedding of each POI with a category embedding from another randomly picked POI to form a corrupted graph)" |
| one weight balances the two terms | p.139, Eq. 11 | `L = alpha L_pr + (1 - alpha) L_rc`; "The strengths of L_pr and L_rc are controlled by alpha" |
| the three modules train together | p.136 §3.1(6) | the hierarchical infomax objective trains "the components of phi_p, AGG_poi-region, and phi_r" |
| locally and globally relevant | p.139, last sentence of §3.7; also the abstract | "the information from local-scale POIs and the global-scale city both flows to the learned region embeddings, making them both globally and locally relevant" |
| what HGI was evaluated FOR (the caveat) | p.134 abstract; p.139 (sheet 6) §4.1 | three downstream tasks, "estimating urban functional distributions, population density, and housing price"; study areas "Xiamen Island and Shenzhen, China" |

### 4.2 · Deliberate omissions

Written down so a later reader does not mistake them for oversights. All are supported by the
source; none is needed by a fundamentals chapter, and including them would turn the subsection into
a paper summary, which the item forbids:

- hard negative sampling and its cosine-similarity window [0.6, 0.8] (p.139);
- the PReLU activations of both graph convolution encoders (p.137, p.138);
- the Laplacian-Eigenmaps term and the full objective of the category encoder (p.136, Eq. 1);
- the paper's fallback rule for regions that share no border (p.138);
- every reported result and baseline comparison.

### 4.3 · Numbers

**No number is quoted in the new prose, by design.** Two candidates were considered and both
rejected:

- **`w_r = 0.4`** (the published cross-region edge weight). Rejected: the repo retunes this weight
  to 0.7 for this project's data, which the orphaned NUM-4 ledger block earlier in §2.2
  (`2_fundamentals.tex:180-193`) records with its sources. The live prose carries no value for it.
  Writing the published 0.4 here, three subsections above a ledger that documents the retuning,
  would create exactly the collision N5 forbids. Prose states the DIRECTION only: weights decrease
  with distance and are reduced further across regions. Both directions are read straight off Eq. 2
  with no computation.
- **`alpha`** (the balancing weight of Eq. 11). Rejected: the paper states only that it "should be
  tuned", so there is no single value of record to quote.

Consequently N3 has nothing to trace and the numeral-extraction gate acquires no new rows.

---

## 5 · Compliance

**GLOSSARY (fail-closed).** No new term is introduced, and **no new registry row is proposed**.
Every technical word in the new prose is already registered or already live in this chapter:
*bilinear discriminator* and *logistic function* (GLOSSARY §3, registered 2026-07-28 for the
Check2HGI equations that appear later in this same section); *POI / place*, *region*, *check-in*,
*DGI*, *HGI*, *Check2HGI* (§2, §3). *Delaunay graph*, *graph convolution*, *multi-head attention*
and *mutual information* are the source's own mechanism names and are already in dissertation prose
at `chapters/3_cbic/method.tex:35` (Delaunay, DGI, mutual information, corrupted graph) and
`chapters/4_courb/related.tex:18` (hierarchical levels, graph convolutions, multi-attention).

**Consistency with what the chapter already commits to about HGI**, the constraint the item flagged
as likely to bite. Checked against both sites:

- The lineage table row, `src/tables/frame/lineage.tex:26`: "Extends graph infomax over the POI,
  region, and city hierarchy to yield region-aware place embeddings." The new paragraph at
  `:261-270` is the *explanation* of that row's "region-aware", not a competing claim. Nothing in
  the table changed.
- §2.2's framing of place-level embeddings, and Definition "Place embedding" in the following
  subsection: HGI's output is one vector per POI, static across visits. The new text says nothing
  that would soften that; it explains where the vector comes from, not that it varies.
- The honest caveat is not weakened. It is strengthened: the old text asserted HGI's target was
  region representation, and the new text names the three evaluated tasks, the two study areas, the
  fact that the region embedding is the evaluated object, and the fact that the POI-level use here
  is outside the original evaluation.

**WRITING_LAW.** No em-dash (gate green). No contraction. American English. No repo codename. No
banned word from §4 in the new prose (checked by grep over the added span and by the register gate).
The "carry/carries" metaphor budget of the idiom rule
is ≤3 per chapter, and Chapter 2 measures **4** in prose lines after this edit, against **5** at
baseline 28df0cc0 (measured both ways with the same command, comment lines excluded). The chapter is
still one over budget and was two over before; my edit added none and removed one, which sat in the
sentence "This hierarchy lets a region representation *carry* both local POI information and
city-level context" and did not return. The remaining four are pre-existing and outside this item's
file scope to touch. Burstiness preserved: the added prose mixes short
sentences ("The name states what is maximized.", "The POI level does that work.") with long ones,
and no two consecutive paragraphs open the same way.

**No claim about this dissertation's own results appears in the new text**, so C1/C2 have nothing to
gate. Every sentence is a statement about HGI, attributed to HGI's paper, plus the two sentences
about how this project repurposes it, both of which are the old text's own surviving claims.

---

## 6 · Build gates (rule 6) — each run separately from `src/`, exit code read directly

| Command | Exit code |
|---|---|
| `make defense` | **0** (108 pp, tex_errors=0) |
| `make ppgc` | **0** (109 pp, tex_errors=0) |
| `make check` | **0** (25 gates) |
| `make selftest` | **0** |

`make academico` was also run (rc 0, 105 pp) because the page-count gate measures all three volumes
and would otherwise compare a stale academico log.

**Page counts, before and after.**

| Volume | Before (baseline 28df0cc0) | After |
|---|---|---|
| defense | 106 | **108** |
| academico | 103 | **105** |
| ppgc | 107 | **109** |
| extra | 22 | 22 (untouched) |

The defense volume grew by two pages. That is the expected cost of the item: the subsection went
from roughly 12 lines of prose to roughly 57.

Note on the +2 for academico and ppgc: another agent is editing §2.2 in this same file concurrently
(splitting the former Definition 2.7 into two, at what is now `:366` onward), and the introduction
carried modifications from an earlier item at baseline. Those edits are in the tree, so the measured
counts above are the counts of the *tree*, not of my edit in isolation. My edit alone accounts for
the growth in §2.2's HGI subsection.

**`sync_page_counts.py --write` WAS RUN, once, from `src/`, and this is the authorized case.** The
first `make check` returned rc 2 on exactly one gate, "recorded page counts vs the measured build",
with four stale claims (CLAUDE.md defense and ppgc, PLAN.md defense, src_utils/codex_reviewer.md
defense). The author suspended the page budget today and authorized the volume to grow, so the
document genuinely changed size by his decision and the recorded counts were the stale artifact, not
the defect. Seven claims were updated across CLAUDE.md, PLAN.md and src_utils/codex_reviewer.md.
**No other gate was red at any point**, and nothing was weakened, reworded, or repointed to make a
probe pass. Recording the distinction the brief drew, because it is the point: in the previous round
the identical command would have masked a real defect, an agent correctly refused it, and that
refusal was right. The two cases differ by authorization, not by convenience.

---

## 7 · Concurrency

The item warned that another agent was editing §2.2 at `:235-249` in this same file. I stayed above
that boundary: my edit begins inside the "Hierarchical graph infomax" subsection and ends before
`\subsection{From a fixed place vector to visit context}`. I re-read the live file immediately
before writing and again after, and the other agent's split of the place-embedding definition is
present and intact in the tree (its own round-11 comment block is at what is now `:371-380`). No
overwrite occurred in either direction; the builds above are of the combined tree.

---

## UNFINISHED

Nothing in the item's scope. The subsection was rewritten, all four gates are green, and the page
counts are synced under the author's authorization.

Three things this item deliberately did **not** do, so they are visible rather than silent:

1. **The lineage table row for HGI was not touched.** It remains "Extends graph infomax over the
   POI, region, and city hierarchy to yield region-aware place embeddings." It is consistent with
   the expanded prose and the item named only the chapter file as editable.
2. **The reference implementation at `github.com/RightBank/HGI` was not opened.** Nothing written
   depends on it, but if a later round wants to state what this project's HGI *run* did (as opposed
   to what the paper describes), that code and `research/embeddings/hgi/` are where it would have to
   be verified.
3. **The published `w_r = 0.4` versus this project's retuned 0.7 is still carried only by the
   orphaned NUM-4 comment block at `:180-193`, not by prose.** That block is marked ORPHANED from
   the author's clean-tree pass. Whether the retuning deserves a sentence in the chapter is an
   author decision, not one this item was authorized to make. Flagging it because the new HGI
   paragraph now describes the edge weights qualitatively, which makes the absence of the retuning
   slightly more visible than it was.
