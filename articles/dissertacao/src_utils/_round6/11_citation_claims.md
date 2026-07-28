# 11_citation_claims.md — the per-chapter citation claim-support audit, consolidated

**Round 6, 2026-07-28.** Written against the live source tree; PDFs rebuilt after the two
bibliography edits of ITEM 2 (`tex_errors=0`, 0 overfull boxes, 105/100 pp; see section 7).

This report covers the three items of the task: the per-chapter claim-support pass the author
asked for by name under COD-008 (ITEM 1), the two preprint-to-version-of-record bibliography
upgrades (ITEM 2), and the Standley citation audited against its own commit history (ITEM 3).

---

## 0 · How this was run, and one deviation to declare

**Deviation.** The task specifies dispatching one sub-agent per chapter with `host.delegate`.
`host.delegate` is **not available in this frame** — it is root-only and this frame is a leaf, so
the call raises rather than dispatching. I ran every unit myself instead, unit by unit, over the
whole citation surface rather than a sample. The per-chapter reports the task names were still
written, one per unit, and each carries its own counts, verdict table and source ledger. What is
lost by not fanning out is the independence of a fresh-eyes reader per chapter (AGENT_GUARDRAILS
L6); what is kept is that no unit was sampled and every verdict traces to a source I opened.
**The author should treat this report as one auditor's pass, not seven.**

**Method, in the order it ran.**

1. **Measured the surface.** Parsed all eleven chapter files plus `0_main.tex` and the six
   `\input` table files, stripping `%` comments so only rendering sites count. Matched
   `\cite`, `\citep`, `\citet`, `\textcite`, `\parencite` and `\onlinecite`.
2. **Resolved every bibliography entry at its source of record.** 100 entries: Crossref REST
   for every DOI, the arXiv API for every arXiv identifier, OpenAlex (with the configured API
   key, never anonymously and never with a `mailto` parameter) for entries carrying neither, and
   Semantic Scholar for the abstracts Crossref does not deposit. Six references were read as
   full PDFs: five already in the repository at `science/articles/`, plus Caruana 1997 fetched
   open-access from Springer, plus Standley 2020 fetched from arXiv for ITEM 3.
3. **Screened every citing sentence** against the retrieved record and abstract. This screen is
   a triage instrument only; per AGENT_GUARDRAILS R5, AI output is not a source.
4. **Verified by hand every site the screen did not clear**, plus every site whose verdict
   depended on a detail an abstract cannot carry. 38 sites were adjudicated against the sources
   directly; 15 more were closed by reading a paper body or a repository document. Six of the
   screen's flags were **overturned as false positives** and are recorded as such, because a
   later pass would otherwise re-raise them.
5. **Traced provenance** of every failure: whether the citing sentence is the author's own frame
   prose or verbatim reproduced article prose, by exact-string matching against
   `articles/CBIC___MTL/sections/`, `articles/CoUrb_2026/src_en/sections/` (and the PT source of
   record) and `articles/[mobiwac]/src/sections/`. This decides the errata regime for each row.

## 1 · Counts (measured, not sampled)

| Unit | `\cite` commands | Source lines carrying them | Key instances (audited) | Distinct keys | SUPPORTED | PARTIAL | NOT-SUPPORTED | UNVERIFIABLE |
|---|---|---|---|---|---|---|---|---|
| `1_introduction` | 8 | 8 | 9 | 9 | 8 | 1 | 0 | 0 |
| `2_fundamentals` | 69 | 69 | 70 | 67 | 70 | 0 | 0 | 0 |
| `3_cbic` | 57 | 37 | 64 | 31 | 49 | 8 | 6 | 1 |
| `4_courb` | 50 | 32 | 53 | 28 | 47 | 4 | 2 | 0 |
| `5_mobiwac` | 56 | 43 | 60 | 33 | 57 | 3 | 0 | 0 |
| `6_conclusion` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `appendices` | 7 | 7 | 9 | 9 | 9 | 0 | 0 | 0 |
| **total** | **247** | **196** | **265** | **100** | **240** | **16** | **8** | **1** |

**On the task's figure of 334 citation instances.** I measure **247 rendering `\cite` commands**
and **265 key instances** across the eleven units. The gap is accounted for and is not a
discrepancy in the document: raw `grep` over the same files finds 255 `\cite` commands and 273
comma-expanded keys, of which the difference is inside `%` comments (audit-trail notes that do
not render), plus 3 keys in `tables/frame/lineage.tex` (counted here under Chapter 2, which
`\input`s it) and one `\citebrackets` command in `0_main.tex` that is a style declaration, not a
citation. 334 does not correspond to any count I can reproduce over `src/`; the author should
treat 247/265 as the measured figure and this paragraph as the audit trail for it.

**Bibliography integrity, measured while resolving.** 100 entries, 100 distinct keys, **zero**
duplicate keys, **zero** cited-but-missing keys, **zero** uncited entries. 99 of the 100 resolve
at an external source of record; the one that does not is `santos2024urban`, a UFV master's
dissertation with no DOI, which I verified against the document itself (see section 4).

## 2 · The consolidated failure table, ranked by how load-bearing the claim is

Ranking is by what the claim carries in the document, not by verdict severity. A NOT-SUPPORTED
citation on a background bullet costs less than a PARTIAL on a sentence that motivates a design
choice. "Load" is my judgment and the author may re-rank it; the evidence for each row is in the
per-chapter report and the reasoning is in section 3.

| # | Load | Verdict | Site | Key | Regime | Recommended disposition |
|---|---|---|---|---|---|---|
| 1 | high | NOT-SUPPORTED | `3_cbic.tex:213` | `ruder2017sluice` | PUBLISHED, Appendix B row required | CHANGE THE CITATION. Swap to `baxter2000model`, which is already in the bibliography and is cited for exactly this claim at `4_courb.tex:116`. Appendix B row (published prose). |
| 2 | high | NOT-SUPPORTED | `3_cbic.tex:214` | `standley2020tasks` | PUBLISHED, Appendix B row required | NARROW THE PROSE. Draft sentence and Appendix B row in section 5 of this report. NEEDS-AUTHOR: it is a claim, not a typo. |
| 3 | high | NOT-SUPPORTED | `4_courb.tex:161` | `sun2020go` | PUBLISHED, Appendix B row required | NARROW THE PROSE. Drop the semantics half, or keep it and cite a source for it. `cho2011gowalla` supports temporal and spatial periodicity of visits; nothing in the bibliography supports temporal cycles revealing place FUNCTION. Appendix B row. |
| 4 | high | NOT-SUPPORTED | `4_courb.tex:219` | `belkin2003laplacian` | PUBLISHED, Appendix B row required | CHANGE THE CITATION. `Xu2023` (TME) regularizes category relatedness with a predefined category hierarchy, which is what this term does; it is already in the bibliography and cited in the same chapter. Or drop the citation and present the term as the implementation's own. Appendix B row. |
| 5 | high | PARTIAL | `5_mobiwac.tex:44` | `caruana1997multitask` | UNDER REVIEW, two-file change + article errata | ADD A SECOND CITATION. `standley2020tasks` states "this often leads to inferior overall performance as task objectives can compete", which is the mechanism the sentence describes; it is already in the bibliography and cited for this at `2_fundamentals.tex:310`. Two-file change (dissertation + `articles/[mobiwac]/src/`), then that article's errata record. |
| 6 | medium | NOT-SUPPORTED | `3_cbic.tex:97` | `zhang2021survey` | PUBLISHED, Appendix B row required | NARROW THE ATTRIBUTION. The five dimensions listed are `yu2024survey`'s five areas, renamed. Keep the list on `yu2024survey`; cite `zhang2021survey` for the survey framing only, or drop it here (it is cited three more times in the chapter). Appendix B row. |
| 7 | medium | NOT-SUPPORTED | `3_cbic.tex:114` | `nash` | PUBLISHED, Appendix B row required | NARROW THE PROSE or MOVE THE KEYS. Both keys are mis-slotted: they belong to the gradient-conflict bullet immediately above, where both are already cited. Either narrow the data-heterogeneity bullet to what these works support, or cite a survey that does treat heterogeneity. Appendix B row. |
| 8 | medium | NOT-SUPPORTED | `3_cbic.tex:114` | `standley2020tasks` | PUBLISHED, Appendix B row required | As the row above: same bullet, same disposition. |
| 9 | medium | PARTIAL | `3_cbic.tex:244` | `nash` | PUBLISHED, Appendix B row required | LEAVE AND RECORD as a `[VERIFY]`. The clause needs a page or section from arXiv:2202.01017; the same subsection has already had two other cost claims corrected against this paper, so the paper has been read here before. |
| 10 | medium | PARTIAL | `3_cbic.tex:306` | `chen2020modeling` | PUBLISHED, Appendix B row required | NARROW THE PROSE. "is designed for POI category classification" -> "is applied here to POI category classification". One clause, no result affected. Appendix B row. |
| 11 | medium | PARTIAL | `4_courb.tex:134` | `russwurm2024geographiclocationencodingspherical` | PUBLISHED, Appendix B row required | ADD ONE CLAUSE. State that the encoder used is the sinusoidal-representation-network component of that work; the paper itself reports the two components as separable and each competitive alone. Same fix serves `:153`. Appendix B row. |
| 12 | medium | PARTIAL | `4_courb.tex:153` | `russwurm2024geographiclocationencodingspherical` | PUBLISHED, Appendix B row required | As `:134`, one shared clause covers both. |
| 13 | medium | PARTIAL | `5_mobiwac.tex:153` | `Lim2022` | UNDER REVIEW, two-file change + article errata | LEAVE. The two sentences following the citation already draw the distinction the finding is about. If tightened: say the coarse target is trained but subordinate. |
| 14 | low | NOT-SUPPORTED | `3_cbic.tex:123` | `Zhang2020` | PUBLISHED, Appendix B row required | NARROW THE PROSE. Restate as its authors do: an interactive multi-task framework whose temporal-aware activity encoder handles uncertain check-ins. Appendix B row (wording table). |
| 15 | low | PARTIAL | `1_introduction.tex:50` | `wu2024torchspatial` | frame, no erratum; [NEEDS SIGN-OFF] | NARROW ONE WORD. "first validated" -> "validated". True of both co-cited works under the weaker verb. Frame chapter, so no erratum; `[NEEDS SIGN-OFF]`-class. NOTE: the published CoUrb introduction carries the same "originally validated" attribution to the same key, so the two should take the same disposition. |
| 16 | low | PARTIAL | `3_cbic.tex:102` | `caruana1997multitask` | PUBLISHED, Appendix B row required | LEAVE AND RECORD. The regularization half is supported; the popularity half is a bibliometric claim a 1997 paper cannot carry. Low exposure. |
| 17 | low | PARTIAL | `3_cbic.tex:115` | `zhang2021survey` | PUBLISHED, Appendix B row required | LEAVE AND RECORD. The surveys treat cost growth with task count; "super-linearly" is a shape claim neither abstract states. Challenge-list bullet, no result depends on it. |
| 18 | low | PARTIAL | `3_cbic.tex:115` | `yu2024survey` | PUBLISHED, Appendix B row required | As the row above. |
| 19 | low | PARTIAL | `3_cbic.tex:123` | `Liao2018` | PUBLISHED, Appendix B row required | NARROW THE PROSE. "temporal attention mechanisms" -> "context-aware recurrent units", which is what MCARNN's authors describe. Appendix B wording row. |
| 20 | low | PARTIAL | `3_cbic.tex:125` | `Xia2020` | PUBLISHED, Appendix B row required | NARROW THE PROSE. Drop "LSTMs and", or verify against the paper body. The sentence also needs a grammatical repair ("improve multi-task POI recommendation both location and temporal context"), which is a wording row regardless. |
| 21 | low | PARTIAL | `3_cbic.tex:127` | `Xu2023` | PUBLISHED, Appendix B row required | NARROW THE PROSE. "graph-based encoders" -> "a tree-guided multi-task embedding". Appendix B wording row. |
| 22 | low | PARTIAL | `4_courb.tex:60` | `rahmani2019category` | PUBLISHED, Appendix B row required | LEAVE AND RECORD as a narrow `[VERIFY]` on the sequential clause. Crossref returns a truncated abstract for this paper. |
| 23 | low | PARTIAL | `4_courb.tex:82` | `Xia2020` | PUBLISHED, Appendix B row required | Same work and same defect as `3_cbic.tex:125`. Fix both or neither, so the two chapters do not describe one system two ways. |
| 24 | low | PARTIAL | `5_mobiwac.tex:181` | `caruana1997multitask` | UNDER REVIEW, two-file change + article errata | LEAVE, or move the fixed-weighting clause onto `kurin2022scalarization` / `xin2022domtl`, both cited two lines later and both of which do establish a fixed or uniform weighting as the baseline to beat. |
| 25 | low | UNVERIFIABLE | `3_cbic.tex:145` | `huang2022estimating` | PUBLISHED, Appendix B row required | RECORD as a `[VERIFY]` with a named check: locate the edge-weight formula in the cited paper, or restate it as this work's own construction. |

**Shape of the result.** 25 rows out of 265 key instances: 8 NOT-SUPPORTED, 16 PARTIAL, 1
UNVERIFIABLE. **Twenty-one of the 25 are in reproduced article prose** (15 in Chapter 3, 6 in
Chapter 4), so they are errata-policy decisions rather than free edits. The frame chapters the
author wrote himself carry **one** PARTIAL between them (`1_introduction.tex:50`) and Chapter 2,
the most heavily cited unit at 70 key instances, carries **none**. That asymmetry is the finding
worth naming: the citation risk in this document is concentrated in the inherited related-work
sections of the two published papers, not in the new writing.

**Five rows share two underlying defects.** `Xia2020` is described with LSTMs at both
`3_cbic.tex:125` and `4_courb.tex:82`; `russwurm2024...spherical` is described as SIREN-only at
both `4_courb.tex:134` and `:153`; and the "originally validated" attribution to
`wu2024torchspatial` appears both at `1_introduction.tex:50` and in the published CoUrb
introduction. Fixing each pair together costs one decision instead of two and prevents the
document describing one system two ways.

## 3 · Why each NOT-SUPPORTED is a failure (the four highest-load rows)

The full reasoning for all 25 rows is in the per-chapter reports. The four that carry the most
are set out here, since these are the ones the author will have to rule on.

### `3_cbic.tex:213` — `ruder2017sluice` cited for hard-sharing regularization

The bullet reads: "By constraining the hypothesis space, hard sharing acts as a regularizer, often
leading to more generalizable models, especially when tasks are related." The cited work's title
of record at arXiv is **"Latent Multi-task Architecture Learning"** (arXiv:1705.08142; the bib
entry carries the earlier "Sluice Networks" title, which is an attribute defect in its own right).
Its stated contribution is learning **what and how much to share**, and it reports that this
"consistently outperforms previous approaches to learning latent architectures". It is a
soft-sharing method presented as an improvement on fixed sharing: evidence against the bullet it
is attached to. **`baxter2000model` is the right citation and is already in the bibliography**,
cited for precisely this claim at `4_courb.tex:116`; its abstract states that "the learner can
search for a hypothesis space that contains good solutions to many of the problems", which is the
hypothesis-space-constraint argument the bullet makes.

### `4_courb.tex:161` — `sun2020go` cited for temporal cycles revealing place function

The sentence claims that cyclical regularities such as meal times and weekly movements "carry
discriminative information about the functional nature of the visited POIs". The cited paper is
LSTPM (AAAI 2020), which models long- and short-term **user preference** for next-POI
recommendation with a nonlocal network and a geo-dilated RNN. It makes no claim about temporal
signal predicting place semantics. The temporal-regularity half is common ground in the field and
`cho2011gowalla` supports it directly ("periodic behavior explains 50% to 70%"); the semantics
half has no support anywhere in this bibliography, which I checked by re-reading the abstracts of
`kazemi2019time2vec` and `Xu2023`, the two other candidates in the chapter.

### `4_courb.tex:219` — `belkin2003laplacian` cited for a hierarchical embedding regularizer

The cited object is an L2 penalty pulling a subcategory embedding toward its parent category
embedding over a known label tree. Laplacian eigenmaps is a nonlinear **dimensionality-reduction**
method for data on a low-dimensional manifold. The link is thematic at best (graph-Laplacian
smoothness) and the sentence attributes the term itself to the paper. `Xu2023` (TME), already in
this chapter's citation list, "utilizes the predefined category hierarchy to regularize the
relatedness among categories" — the same construction the term implements.

### `5_mobiwac.tex:44` — `caruana1997multitask` cited for the compromise mechanism

This one is PARTIAL rather than NOT-SUPPORTED but ranks high because it opens Chapter 5 and sets
up the whole study. The sentence says shared parameters "can converge to a compromise optimal for
neither task, helping one while hurting the other". Caruana 1997 is the origin of the shared
representation and argues the **positive** direction: MTL "improves generalization by using the
domain information contained in the training signals of related tasks as an inductive bias".
`standley2020tasks` states the negative direction in as many words — "this often leads to inferior
overall performance as task objectives can compete" — and is already in the bibliography, cited
for exactly this at `2_fundamentals.tex:310`. Adding it beside Caruana closes the gap without
removing the historically correct citation. Chapter 5 is under review, so the change is a
two-file change and goes in that article's errata record, not Appendix B.

## 4 · Six false positives the screen produced, overturned here

Recorded because each would otherwise be re-raised by the next checker, and because two of them
are systematic classes rather than one-offs.

| Site | Key | Screen said | Actual verdict | Why the screen was wrong |
|---|---|---|---|---|
| `5_mobiwac.tex:96` | `silva2025mtlnet` | NOT-SUPPORTED ("architecture described in reverse") | **SUPPORTED** | The CBIC **abstract** compresses the architecture ("shares lower-level embeddings and sequence encoders while maintaining task-specific heads"); the CBIC **method section** of record states the chapter's version verbatim: inputs "are first processed by separate, task-specific encoders", then FiLM conditioning, then shared residual layers, then task-specific heads, with Nash-MTL aggregating gradients. Abstract-only checking inverts this. |
| `apx_b_errata.tex:220` | `silva2025mtlnet` | NOT-SUPPORTED | **SUPPORTED** | An errata row necessarily cites the source that contradicts the text being corrected. The row says the *submitted MobiWac manuscript* mis-described the CBIC work; the cited abstract is the evidence for the erratum, not the claim under audit. Systematic class for errata registers. |
| `2_fundamentals.tex:57` | `Xu2023` | PARTIAL ("TME uses check-in context, not static features") | **SUPPORTED** | The sentence attributes the **task** (labeling a POI with its category, which TME's abstract states as its problem), not a feature set. "from static features rather than from a sequence" is the dissertation's contrast between task types. |
| `2_fundamentals.tex:420` | `wongso2025massivesteps` | PARTIAL ("15 cities, not Istanbul") | **SUPPORTED** | The chapter says the benchmark *supplies* Istanbul check-ins, not that it is Istanbul-only, and the over-reliance claim is verbatim in the abstract ("the over-reliance on older datasets from 2012-2013"). |
| `3_cbic.tex:108` and `2_fundamentals.tex:320` | `liu2019dwa` | PARTIAL / UNVERIFIABLE | **SUPPORTED** | The paper ("End-To-End Multi-Task Learning With Attention", CVPR 2019) introduces both MTAN and Dynamic Weight Averaging. Its abstract leads with the architecture, so an abstract-only check underreads the DWA attribution, which is correct. |
| `appendices apx_d_ceiling.tex:55` | `kohavi1995crossval` | PARTIAL ("recommends ten folds, not five") | **SUPPORTED** | The citation supports cross-validation as the estimation protocol; the fold count and the user grouping are the dissertation's own choices, stated in the same sentence. Same construction as the scikit-learn site the author already ruled on. |

## 5 · ITEM 3 — the Standley citation at `3_cbic.tex:214`

The author asked for four things here: what the paper does and does not support, a commit-history
check for an earlier reference at that site, whether a reference that DOES support the claim exists,
and a drafted narrowed sentence with its Appendix B row. **`3_cbic.tex` was not edited.**

### 5.1 · What the paper supports, read at the source

Resolved as arXiv:1905.07553 (arXiv API: submitted 2019-05-18, last revised 2020-09-03, six
authors, no `journal_ref` and no DOI on the preprint record) and as the ICML 2020 version at
OpenAlex (venue "International Conference on Machine Learning", pp. 9120-9132, PMLR v119). I then
downloaded and read the full paper (13 pages, arXiv v3). The bibliography entry's ICML venue is
correct and the dropped page range is the fail-closed behaviour Appendix B already records.

The citing bullet claims two things. **Both fail, in different ways.**

*First half, "hard parameter sharing frequently matches or exceeds the performance of more complex
architectures on many benchmarks".* The paper argues the opposite direction as its motivating
premise: multi-task learning "often leads to inferior overall performance as task objectives can
compete", and in the body, "multi-task performance can suffer so much that smaller independent
networks are often superior". Its own contribution is a framework for **assigning** tasks across
several networks so that cooperating tasks share one and competing tasks do not. On the specific
comparison the bullet makes, the paper reports its groupings beating a single traditional
multi-task network at every budget above 1.5 SNT: "solutions that utilize multiple networks
outperform this traditional strategy for every budget > 1.5". The one result that leans the
bullet's way is narrower and budget-conditional: "when the single-task networks are shrunk so that
they fit within the same total budget as the multi-task network, multi-task networks with 3, 4, or
5 tasks outperform the single-task networks on average. Nevertheless, two-task networks still do
not compare favorably." **This chapter's model has exactly two tasks**, so the sentence the paper
does support is the one that excludes this chapter's own case.

*Second half, "while offering faster training and inference".* The string "faster training" does
not occur in the paper. It does support the inference half: multi-task learning "can save
computation at inference time as only a single network needs to be evaluated", and the paper's
framework "offers a time-accuracy trade-off". On training the paper says the opposite of a benefit
in the passage nearest the claim: UberNet's authors, using hard sharing, "focus on reducing the
computational cost of training for hard parameter sharing, but experience a rapid degradation in
performance as more tasks are added". Training cost is also the axis on which this chapter's own
result went the other way, which Appendix B already records.

So the site is not a mis-citation of the wrong paper. It is a **correctly identified paper cited
for the reverse of its finding**, which is the class AGENT_GUARDRAILS §1 calls claim-not-supported
and which existence checking cannot catch. Note that the same key is cited four times in Chapter
3, and three of those four are sound: `:118` (task clustering), `:191` (mitigating negative
transfer, alongside `perez2018film`) and `2_fundamentals.tex:310` (joint training can hurt as
easily as it helps) are all SUPPORTED. Only `:214` and `:114` fail.

### 5.2 · The commit history: was a different reference ever used at that site?

**No. The answer is negative and I can date it.** Run through the shell over both paths the author
named:

```
git log --oneline -S"standley2020tasks"    -- articles/CBIC___MTL/sections/ articles/dissertacao/src/chapters/3_cbic.tex
git log --oneline -S"matches or exceeds"   -- (same two paths)
git log --oneline -S"Empirical Performance"-- (same two paths)
```

All three return the **same two commits**, and only those two:

| Commit | Date | What it is |
|---|---|---|
| `223f5df7` | 2025-10-21 12:22 | first import of the CBIC article tree (`CBIC___MTL/` at the repo root) |
| `643c686e` | 2025-10-21 12:23 | the same content one minute later, after the `articles/` reorganization |
| `1a29b545` | 2026-07-23 | the dissertation re-typeset that copied the sentence into `3_cbic.tex` |

`articles/CBIC___MTL/sections/method.tex` has been touched by **exactly one commit in its entire
history** (`643c686e`). The bullet is present in the very first committed version of that file, at
line 85, with `\cite{standley2020tasks}` already attached and the wording byte-identical to today:

> \item \textbf{Empirical Performance:} In practice, hard parameter sharing frequently matches or
> exceeds the performance of more complex architectures on many benchmarks, while offering faster
> training and inference \cite{standley2020tasks}.

I then checked the dissertation side the other way, printing the `Empirical Performance` line from
**every one of the 14 commits** that has ever touched `3_cbic.tex`: the line is byte-identical in
all 13 versions in which it exists, and absent only from `a735b8f3` (the skeleton commit, before
the chapter had content). I also ran `-S` for seven plausible alternative keys
(`vandenhende2022mtl`, `crawshaw2020multi`, `ruder2017overview`, `zhang2021survey`,
`kokkinos2016ubernet`, `caruana1997multitask`, `baxter2000model`) against the CBIC method file:
only `caruana1997multitask` and `baxter2000model` appear at all, each in the single import commit,
and neither has ever stood at this bullet.

**Conclusion.** No earlier or different reference ever stood at that site. The mis-support is
original to the article as first written, not introduced by the dissertation's re-typeset or by any
later repair. That is worth recording precisely because the neighbouring Mikolov site *was* a
repair-introduced defect (`church2017word2vec` -> `mikolov2013word2vec` fixed an existence error and
left a support error); this one has a different history and no repair to blame.

One further fact from the history, offered because it bears on how the site should be fixed: the
same key is cited in the CBIC architecture sentence (`method.tex`, now `3_cbic.tex:191`) for
"modulating task interactions and mitigating negative transfer", alongside `perez2018film`. That
use is **sound** and is what the paper is actually about. The author's own instinct in citing
Standley was right; only this one bullet's direction is wrong.

### 5.3 · Does a reference in the bibliography support the claim as stated?

**Not as stated, and I would not add one that appeared to.** I checked the three candidates whose
stated findings are closest, all already in `references.bib` and all read at their sources this
session:

| Candidate | Identifier | What it actually supports | Verdict as a replacement |
|---|---|---|---|
| `kurin2022scalarization` | arXiv:2201.04122 (OpenAlex) | "unitary scalarization, coupled with standard regularization and stabilization techniques ... matches or improves upon the performance of complex multi-task optimizers" | Supports "matches or improves" about **optimizers**, not about architectures. Closest in wording, wrong object. |
| `xin2022domtl` | arXiv:2209.11379 (arXiv API) | "MTO methods do not yield any performance improvements beyond what is achievable via traditional optimization approaches" | Same: optimizers, not architectures. |
| `vandenhende2022mtl` | DOI 10.1109/TPAMI.2021.3054719 (Crossref) | an architecture-plus-optimization survey with "an extensive experimental evaluation across a variety of dense prediction benchmarks to examine the pros and cons of the different methods" | The only candidate whose object is architectures and which compares them on benchmarks. It does not state the bullet's direction in its abstract, so citing it for "frequently matches or exceeds" would repeat the present defect one step removed. |

The honest reading is that the bullet's first half, as written, is **not a claim the MTL literature
supports in general** — which is precisely what this dissertation's own arc goes on to find. So the
repair is to narrow the sentence to what is supported, not to hunt for a citation that licenses the
original. Two things *are* supported and are worth keeping:

- **the inference-time saving**, by Standley directly ("can save computation at inference time as
  only a single network needs to be evaluated");
- **the fixed-weight-baseline result**, by `kurin2022scalarization` and `xin2022domtl`, which is the
  defensible version of "the simple thing is hard to beat" and which Chapter 2 already states at
  `2_fundamentals.tex:331-340`.

### 5.4 · The drafted narrowed sentence (for the author to approve; NOT applied)

The bullet lives in an itemized list of three under "Rationale for Hard Parameter Sharing". The
draft keeps the list structure, keeps the bullet's role in the argument (why this chapter chose hard
sharing), and reduces the claim to what the cited work supports. Claim strength goes **down**, never
up, which is the direction the writing law requires for a substitution.

**Published wording** (`articles/CBIC___MTL/sections/method.tex`, reproduced at `3_cbic.tex:214`):

> \item \textbf{Empirical Performance:} In practice, hard parameter sharing frequently matches or
> exceeds the performance of more complex architectures on many benchmarks, while offering faster
> training and inference \cite{standley2020tasks}.

**Draft replacement** (option A, the conservative one — narrows to the inference claim the paper
makes and drops the comparative claim it does not):

```latex
    \item \textbf{Inference Cost:} A single shared network is evaluated once at inference time
    rather than once per task, which is where the joint model's computational advantage lies
    \cite{standley2020tasks}.\footnote{The published sentence read ``In practice, hard parameter
    sharing frequently matches or exceeds the performance of more complex architectures on many
    benchmarks, while offering faster training and inference''. Neither half is supported by the
    cited work, which argues that joint training ``often leads to inferior overall performance as
    task objectives can compete'' and reports its own task groupings outperforming a single
    multi-task network at every inference budget it tests. The claim is corrected here rather than
    reproduced. See Table~\ref{tab:apx:cbic-errata}.}
```

**Draft replacement** (option B, if the author prefers to keep a performance claim in the bullet —
it then has to be the one the literature supports, and the citation changes):

```latex
    \item \textbf{Empirical Performance:} A simple shared architecture trained with fixed loss
    weights is a strong baseline: controlled comparisons report that specialized multi-task
    optimizers do not consistently improve on it \cite{kurin2022scalarization,xin2022domtl}. The
    inference cost is also lower, since a single network is evaluated once rather than once per
    task \cite{standley2020tasks}.
```

I recommend **option A**. It is the smaller change, it keeps the bullet's function, and it does not
import a claim about optimizers into a paragraph about architectures. Option B is defensible but it
adds two citations to published prose and shifts the bullet's subject.

Two consequential notes on the surrounding text if option A is taken. The bullet's `\textbf{}` label
changes from "Empirical Performance" to "Inference Cost", which is a heading change inside published
prose and needs the same sign-off as the sentence. And the paragraph's lead-in at `:210` says the
choice "is motivated by its efficiency and regularization benefits" — after this change the list's
three bullets are efficiency, regularization and inference cost, so the lead-in still reads
correctly, but the first and third bullets both now concern cost. The author may prefer to merge
them; that is a structural edit and I have not drafted it.

### 5.5 · The Appendix B row (drafted, NOT applied)

This goes in `src/tables/cbic/errata.tex`, the content-errata table for the CBIC chapter, whose
existing rows I read and matched for voice (defect stated flatly in the left cell, correction with
its evidence and its direction in the right cell; the Nash rows are the closest models). It is a
content correction, not a wording substitution, so it belongs in `errata.tex` and not in
`errata_wording.tex`.

```latex
\addlinespace
The rationale for hard parameter sharing states that the approach ``frequently matches or exceeds
the performance of more complex architectures on many benchmarks, while offering faster training
and inference''. The cited work argues the opposite direction: joint training ``often leads to
inferior overall performance as task objectives can compete'', and its own contribution is a
framework for splitting tasks across several networks, which it reports outperforming a single
multi-task network at every inference budget it tests. The phrase ``faster training'' does not
occur in it. &
Narrowed to the claim the cited work supports, that a single shared network is evaluated once at
inference time rather than once per task, with the correction recorded in a footnote. The one
result in that work which favors joint training is conditioned on a matched parameter budget and
on three or more tasks, and it excludes the two-task case this chapter studies. The correction
removes a claim in the chapter's own favor, so it runs against the chapter's interest. \\
```

The reconciliation header at the top of `apx_b_errata.tex` counts itemized rows ("6 + 13 + 3 + 14 =
36"); adding this row makes the CBIC content table 7 and the total 37. **That header comment must be
updated in the same commit as the row**, or the appendix will misstate its own count — the defect
class the file's own history records at the "11 rows" line. I have not made either change: applying
chapter prose edits is a later pass, and I did not want a half-applied erratum in the tree.

## 6 · ITEM 2 — the two bibliography records, re-verified and applied

I re-verified both at Crossref in this session rather than trusting the audit's values. Both came
back exactly as the audit reported, with one typographic difference worth recording.

### 6.1 · `kokkinos2016ubernet`

**Before** (`references.bib:395-400`): `@article` with
`journal = {arXiv preprint arXiv:1609.02132}`, year 2016, no DOI.

**Crossref, DOI 10.1109/cvpr.2017.579**, fetched this session:

| Field | Value returned |
|---|---|
| title | UberNet: Training a Universal Convolutional Neural Network for Low-, Mid-, and High-Level Vision Using Diverse Datasets and Limited Memory |
| container-title | 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) |
| type / publisher | proceedings-article / IEEE |
| issued | 2017-07 |
| page | 5454-5463 |
| author | Iasonas Kokkinos (single) |

Cross-checked at the arXiv API for 1609.02132: same single author, same title, submitted
2016-09-07, and **no `journal_ref` and no DOI on the preprint record**, which is why the conference
version does not resolve from the preprint alone. Every value the audit reported is confirmed:
CVPR 2017, DOI 10.1109/cvpr.2017.579, pp. 5454-5463.

**One difference from Crossref I did not adopt.** Crossref renders the title without the quotation
marks around *Universal* that the author's own preprint title carries (\`Universal'). I kept the
author's typography, since R2 asks that a work be described as its authors describe it, and
recorded the difference in the entry's provenance comment.

**After** (`references.bib:394-412`): re-typed to `@inproceedings` with the CVPR booktitle, pages
`5454--5463`, `doi = {10.1109/CVPR.2017.579}`, and `note = {arXiv:1609.02132}` keeping the preprint
identifier. **The key is unchanged, so no citing site moves.** The single citing site is
`1_introduction.tex:72`, which cites it for "single networks that handle many vision tasks at once";
the abstract supports that and the site is SUPPORTED in the Chapter 1 report.

### 6.2 · `mai2023sphere2vecgeneralpurposelocationrepresentation`

**Before** (`references.bib:640-648`): `@misc` with `eprint = {2306.17624}`, `archivePrefix`,
`primaryClass`, `url`, year 2023, no venue at all.

**Crossref, DOI 10.1016/j.isprsjprs.2023.06.016**, fetched this session:

| Field | Value returned |
|---|---|
| title | Sphere2Vec: A general-purpose location representation learning over a spherical surface for large-scale geospatial predictions |
| container-title | ISPRS Journal of Photogrammetry and Remote Sensing |
| type / publisher | journal-article / Elsevier BV |
| issued | 2023-08 |
| volume / page | 202 / 439-462 |
| ISSN | 0924-2716 |
| authors | Gengchen Mai; Yao Xuan; Wenyun Zuo; Yutong He; Jiaming Song; Stefano Ermon; Krzysztof Janowicz; Ni Lao (8) |

Cross-checked at the arXiv API for 2306.17624: the record's `journal_ref` field reads **"ISPRS
Journal of Photogrammetry and Remote Sensing, 2023"**, and the author list and title match. Every
value the audit reported is confirmed: ISPRS J. Photogramm. Remote Sens. 202:439-462, DOI
10.1016/j.isprsjprs.2023.06.016.

**After** (`references.bib:639-657`): re-typed to `@article` with the journal abbreviated exactly as
the sibling entry `huang2023hgi` abbreviates the same journal (`ISPRS J. Photogramm. Remote Sens.`),
volume 202, pages `439--462`, the DOI, and `note = {arXiv:2306.17624}`. **The key is unchanged, so
none of the five citing sites moves.**

### 6.3 · The two errata rows, and the overfull box they caused

Both rows were added to `src/tables/frame/bib_errata.tex` immediately before the CoUrb-Gowalla row,
matching the existing rows' voice (defect in the donor list on the left, correction with its
verification source on the right, the phrase "arXiv identifier kept as a note" reused from the GAT
and Rußwurm rows).

**First attempt introduced a defect, which the rebuild caught.** Printing
`\texttt{mai2023sphere2vecgeneralpurposelocationrepresentation}` (52 characters) in the
`p{0.42\textwidth}` left column produced `Overfull \hbox (113.58371pt too wide)` at
`bib_errata.tex:112-113` — a `\texttt` key does not line-break. It was the only overfull box in
either build. The fix follows the precedent already in the table: the Rußwurm row **names the work**
instead of printing the key, and so does this row now ("The Sphere2Vec location encoder of Mai et
al. was typed as an arXiv preprint, with no venue recorded at all"). The constraint is recorded in a
comment at the top of the file so the next editor does not re-introduce a long key in a cell.

**The braces are intact.** The `{\small ...}` group whose lost opening brace broke every build from
`6d780b58` to `a880632b` is untouched: I verified the balance mechanically on the comment-stripped
file (44 open, 44 close, delta 0), confirmed line 15 is still `{\small` and the file still ends
`\end{longtable}` then `}`, and rebuilt.

**Both rows render.** Read out of the PDF text layer, not assumed: the two rows are on **p. 95** of
the defense build and **p. 90** of the final build, carrying "kokkinos2016ubernet: typed as an arXiv
preprint", "DOI 10.1109/CVPR.2017.579, pages 5454-5463", "The Sphere2Vec location encoder of Mai et
al." and "DOI 10.1016/j.isprsjprs.2023.06.016". The two upgraded entries also render in the printed
bibliography on **p. 81**: `[8] KOKKINOS, I. Ubernet: ... In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR). 2017. p. 5454-5463. ArXiv:1609.02132.` and `[5]
MAI, G. et al. Sphere2vec: ... ISPRS J. Photogramm. Remote Sens., v. 202, p. 439-462, 2023.
ArXiv:2306.17624.`

## 7 · Build result

Run after the two `src/` edits, from `src/`, with `src_utils/texenv.sh` sourced:

```
make defense   ->  build/main.pdf        105 pages
make final     ->  build/main_final.pdf  100 pages
bash ../src_utils/build.sh . both
  DEFENSE: pages=['105'] tex_errors=0 overfull_hbox=0 overfull_vbox=0 undef_cite=0 undef_ref=0
           bibtex_problems=0 oversized_floats=0
  FINAL:   pages=['100'] tex_errors=0 overfull_hbox=0 overfull_vbox=0 undef_cite=0 undef_ref=0
           bibtex_problems=0 oversized_floats=0
```

**The gate was validated in both directions before I trusted it**, per AGENT_GUARDRAILS §7 ("a check
that has never fired is not a check"). I copied `src/` to a scratch tree and re-injected the exact
historical defect, removing the opening brace of the `{\small` group in `bib_errata.tex`. On that
tree `build.sh` reports `tex_errors=1` with `! Extra }, or forgotten \endgroup`, prints its own
warning that the recovered PDF "is not the document", and **exits 1**; `make defense` produces no PDF
at all and exits 2. On the real tree `build.sh` reports `tex_errors=0` and **exits 0**. So the zero
above is an observation the checker is capable of contradicting.

`tex_errors=0` on both, which is the part of the claim that matters after `ba90aa6d`. Zero overfull
boxes after the long-key fix described in 6.3. Both builds are converged: I ran each twice and the
byte size and page count stopped moving, and the `Label(s) may have changed` warning is gone from
the final pass.

**On the page count, stated carefully because ANCHORS.md records 104/99 and this build reports
105/100.** The delta is **not** mine. Other agents are working in this same tree and have
uncommitted edits in it, including `src/chapters/5_mobiwac.tex` (19 lines) and a replaced CoUrb
figure. I measured the attribution rather than assuming it: I copied `src/` to a scratch tree,
reverted **only my two files** to their `HEAD` versions, and rebuilt. That tree also produces **105
pages**. So the document had already grown to 105/100 from another agent's work before my edits, and
my two errata rows cost **zero pages** — they fit in the existing longtable overflow on p. 95. The
committed `dissertacao.pdf` at `HEAD` is 104 pages, which is what ANCHORS.md measured.

`make check` **fails**, on one gate, and it is not mine:

```
== recorded page counts vs the measured build ==
src/build/main.log has no page count -- the build did not finish
  -> run: python3 src_utils/sync_page_counts.py --write
```

Two things to note about that gate. First, its message is **wrong about the cause**: the build did
finish and `build/main.log` does contain `Output written on build/main.pdf (105 pages, ...)`. The
check reads `main.log` before the final `pdflatex` pass has rewritten it, or reads it while `make`
is mid-run; either way the failure is a stale-read in the checker, not a build failure. Second, the
recorded page counts in the governance files are now stale by one page for a reason that is not my
change. **I did not run `sync_page_counts.py --write`**: it edits `CLAUDE.md`, `PLAN.md` and other
files outside my remit, and the page count is still moving while other agents work. That is a
hand-off, recorded here rather than acted on. Every other `make check` gate passes: banned words,
codenames, unresolved refs and cites, bibtex, sweep-guard self-tests, word-count reconciliation,
torn sentences (0), trapped prose (0/10 fixtures failing).

## 8 · Edits applied

| File | Lines | Change |
|---|---|---|
| `src/references.bib` | 394-412 | `kokkinos2016ubernet` re-typed `@article` -> `@inproceedings`, CVPR 2017 booktitle, pp. 5454--5463, `doi = {10.1109/CVPR.2017.579}`, `note = {arXiv:1609.02132}`, with a 10-line provenance comment naming what was verified where. Key unchanged. |
| `src/references.bib` | 650-670 | `mai2023sphere2vecgeneralpurposelocationrepresentation` re-typed `@misc` -> `@article`, ISPRS J. Photogramm. Remote Sens. v.202 pp. 439--462, `doi = {10.1016/j.isprsjprs.2023.06.016}`, `note = {arXiv:2306.17624}`, with a 9-line provenance comment. Key unchanged. |
| `src/tables/frame/bib_errata.tex` | 15-22 | Comment recording the two new rows and the long-key overflow constraint, so it is not re-introduced. |
| `src/tables/frame/bib_errata.tex` | 114-124 | The two errata rows, in the existing rows' voice, placed before the CoUrb-Gowalla row. |

Nothing else in `src/` was touched. `3_cbic.tex` was **not** edited (ITEM 3 is a hand-off);
`apx_b_errata.tex` was **not** edited (its reconciliation header needs the Standley row and the
count update in the same commit as the prose change, which is the later pass).

## 9 · Source ledger

Every one of the 100 bibliography entries, the identifier it resolved by, and where I opened it this
session. Channels: **Crossref REST** (`api.crossref.org/works/{doi}`), **arXiv API**
(`export.arxiv.org/api/query?id_list=`), **OpenAlex** (`api.openalex.org`, with the configured API
key on every request; never anonymous, never with a `mailto` parameter), **Semantic Scholar**
(`api.semanticscholar.org/graph/v1`) for abstracts Crossref does not deposit, and **PDF** where a
full text was read.

| Key | Identifier | Opened at | Record as returned |
|---|---|---|---|
| `bastug2014edge` | DOI 10.1109/MCOM.2014.6871674 | Crossref REST; OpenAlex API | Living on the edge: The role of proactive caching in 5G wireless networks \| IEEE Communications Magazine \| 2014 \| type journal-article |
| `baxter2000model` | DOI 10.1613/jair.731 | Crossref REST; OpenAlex API | A Model of Inductive Bias Learning \| Journal of Artificial Intelligence Research \| 2000 \| type journal-article |
| `belghazi2018mine` | arXiv:1801.04062 | arXiv API; OpenAlex API | MINE: Mutual Information Neural Estimation \| ICML 2018 \| 2018 \| type posted-content |
| `belkin2003laplacian` | DOI 10.1162/089976603321780317 | Crossref REST; OpenAlex API | Laplacian Eigenmaps for Dimensionality Reduction and Data Representation \| Neural Computation \| 2003 \| type journal-article |
| `capanema2023poirgnn` | DOI 10.1016/j.adhoc.2022.103016 | Crossref REST; OpenAlex API | Combining recurrent and Graph Neural Networks to predict the next place’s category \| Ad Hoc Networks \| 2023 \| type journal-article |
| `caruana1997multitask` | DOI 10.1023/A:1007379606734 | Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf | Multitask Learning \| Machine Learning \| 1997 \| type journal-article |
| `chen2018gradnorm` | no identifier in the bib entry | arXiv API; OpenAlex API | GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks \| Proceedings of the 35th International Conference on Machine |
| `chen2020modeling` | DOI 10.1109/TKDE.2020.3001025 | Crossref REST; OpenAlex API | Modeling Spatial Trajectories With Attribute Representation Learning \| IEEE Transactions on Knowledge and Data Engineering \| 2022 \| type journal-ar |
| `cho2011gowalla` | DOI 10.1145/2020408.2020579 | Crossref REST; OpenAlex API | Friendship and mobility \| Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining \| 2011 \| type proceedi |
| `du2019beyond` | DOI 10.1109/ICDM.2019.00026 | Crossref REST; OpenAlex API | Beyond Geo-First Law: Learning Spatial Representations via Integrated Autocorrelations and Complementarity \| 2019 IEEE International Conference on Da |
| `feng2017poi2vec` | DOI 10.1609/aaai.v31i1.10500 | Crossref REST; OpenAlex API | POI2Vec: Geographical Latent Representation for Predicting Future Visitors \| Proceedings of the AAAI Conference on Artificial Intelligence \| 2017 \| |
| `feng2018deepmove` | DOI 10.1145/3178876.3186058 | Crossref REST; OpenAlex API | DeepMove \| Proceedings of the 2018 World Wide Web Conference on World Wide Web - WWW '18 \| 2018 \| type proceedings-article |
| `gambs2012mmc` | DOI 10.1145/2181196.2181199 | Crossref REST; OpenAlex API | Next place prediction using mobility Markov chains \| Proceedings of the First Workshop on Measurement, Privacy, and Mobility \| 2012 \| type proceedi |
| `grover2016node2vec` | DOI 10.1145/2939672.2939754 | Crossref REST; OpenAlex API | node2vec \| Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining \| 2016 \| type proceedings-article |
| `Halder2021` | DOI 10.1007/978-3-030-75765-6_41 | Crossref REST; OpenAlex API; Semantic Scholar API | Transformer-Based Multi-task Learning for Queuing Time Aware Next POI Recommendation \| Lecture Notes in Computer Science \| 2021 \| type book-chapter |
| `Halder2022` | DOI 10.1007/s10618-022-00865-w | Crossref REST; OpenAlex API | POI recommendation with queuing time and user interest awareness \| Data Mining and Knowledge Discovery \| 2022 \| type journal-article |
| `hamilton2017graphsage` | arXiv:1706.02216 | arXiv API; OpenAlex API | Inductive Representation Learning on Large Graphs \| arXiv preprint \| 2017 \| type posted-content |
| `hazimeh2021dselectk` | arXiv:2106.03760 | arXiv API; OpenAlex API | DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning \| arXiv preprint \| 2021 \| type posted-conten |
| `hjelm2019dim` | arXiv:1808.06670 | arXiv API; OpenAlex API | Learning deep representations by mutual information estimation and maximization \| arXiv preprint \| 2018 \| type posted-content |
| `holm1979` | no identifier in the bib entry | OpenAlex API | A Simple Sequentially Rejective Multiple Test Procedure \| Scandinavian Journal of Statistics \| 1979 \| type article |
| `huang2022estimating` | DOI 10.1080/13658816.2022.2040510 | Crossref REST; OpenAlex API | Estimating urban functional distributions with semantics preserved POI embedding \| International Journal of Geographical Information Science \| 2022  |
| `huang2023hgi` | DOI 10.1016/j.isprsjprs.2022.11.021 | Crossref REST; OpenAlex API; Semantic Scholar API; PDF in repo: Learning urban region representations with POIs and hierarchical graph infomax.pdf | Learning urban region representations with POIs and hierarchical graph infomax \| ISPRS Journal of Photogrammetry and Remote Sensing \| 2023 \| type j |
| `huang2024cslsl` | DOI 10.1140/epjds/s13688-024-00460-7 | Crossref REST; OpenAlex API | Human mobility prediction with causal and spatial-constrained multi-task network \| EPJ Data Science \| 2024 \| type journal-article |
| `jure2014snap` | no identifier in the bib entry | OpenAlex API | {SNAP Datasets}: {Stanford} Large Network Dataset Collection \| (no venue in record) \| 2014 \| type article |
| `kazemi2019time2vec` | arXiv:1907.05321 | arXiv API; OpenAlex API | Time2Vec: Learning a Vector Representation of Time \| arXiv preprint \| 2019 \| type posted-content |
| `kendall2018uncertainty` | DOI 10.1109/CVPR.2018.00781 | Crossref REST; OpenAlex API | Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics \| 2018 IEEE/CVF Conference on Computer Vision and Pattern Reco |
| `kipf2017gcn` | arXiv:1609.02907 | arXiv API; OpenAlex API | Semi-Supervised Classification with Graph Convolutional Networks \| arXiv preprint \| 2016 \| type posted-content |
| `kohavi1995crossval` | no identifier in the bib entry | OpenAlex API | A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection \| (no venue in record) \| 1995 \| type article |
| `kokkinos2016ubernet` | arXiv:1609.02132 | arXiv API; OpenAlex API | UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory \| arXiv  |
| `kong2018hstlstm` | DOI 10.24963/ijcai.2018/324 | Crossref REST; OpenAlex API | HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network for Location Prediction \| Proceedings of the Twenty-Seventh International Jo |
| `kurin2022scalarization` | no identifier in the bib entry | arXiv API; OpenAlex API | In Defense of the Unitary Scalarization for Deep Multi-Task Learning \| arXiv preprint \| 2022 \| type posted-content |
| `lakens2017tost` | DOI 10.1177/1948550617697177 | Crossref REST; OpenAlex API | Equivalence Tests \| Social Psychological and Personality Science \| 2017 \| type journal-article |
| `li2025rehdm` | DOI 10.24963/ijcai.2025/343 | Crossref REST; OpenAlex API | Beyond Individual and Point: Next POI Recommendation via Region-aware Dynamic Hypergraph with Dual-level Modeling \| Proceedings of the Thirty-Fourth  |
| `lian2020geosan` | DOI 10.1145/3394486.3403252 | Crossref REST; OpenAlex API | Geography-Aware Sequential Location Recommendation \| Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mi |
| `Liao2018` | DOI 10.24963/ijcai.2018/477 | Crossref REST; OpenAlex API | Predicting Activity and Location with Multi-task Context Aware Recurrent Neural Network \| Proceedings of the Twenty-Seventh International Joint Confe |
| `Lim2022` | DOI 10.1145/3477495.3531989 | Crossref REST; OpenAlex API | Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation \| Proceedings of the 45th International ACM SIGIR Conference on Research  |
| `lin2021ctle` | DOI 10.1609/aaai.v35i5.16548 | Crossref REST; OpenAlex API | Pre-training Context and Time Aware Location Embeddings from Spatial-Temporal Trajectories for User Next Location Prediction \| Proceedings of the AAA |
| `lin2022rlw` | arXiv:2111.10603 | arXiv API; OpenAlex API | Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning \| arXiv preprint \| 2021 \| type posted-content |
| `lipton2015learning` | arXiv:1511.03677 | arXiv API; OpenAlex API | Learning to Diagnose with LSTM Recurrent Neural Networks \| arXiv preprint \| 2015 \| type posted-content |
| `liu2016strnn` | DOI 10.1609/aaai.v30i1.9971 | Crossref REST; OpenAlex API | Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts \| Proceedings of the AAAI Conference on Artificial Intelligence \| |
| `liu2019dwa` | DOI 10.1109/CVPR.2019.00197 | Crossref REST; OpenAlex API | End-To-End Multi-Task Learning With Attention \| 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) \| 2019 \| type proceeding |
| `liu2021cagrad` | arXiv:2110.14048 | arXiv API; OpenAlex API | Conflict-Averse Gradient Descent for Multi-task Learning \| arXiv preprint \| 2021 \| type posted-content |
| `liu2023famo` | no identifier in the bib entry | arXiv API; OpenAlex API | FAMO: Fast Adaptive Multitask Optimization \| arXiv preprint \| 2023 \| type posted-content |
| `luca2021mobilitysurvey` | DOI 10.1145/3485125 | Crossref REST; OpenAlex API | A Survey on Deep Learning for Human Mobility \| ACM Computing Surveys \| 2021 \| type journal-article |
| `luo2021stan` | DOI 10.1145/3442381.3449998 | Crossref REST; OpenAlex API | STAN: Spatio-Temporal Attention Network for Next Location Recommendation \| Proceedings of the Web Conference 2021 \| 2021 \| type proceedings-article |
| `ma2018mmoe` | DOI 10.1145/3219819.3220007 | Crossref REST; OpenAlex API | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts \| Proceedings of the 24th ACM SIGKDD International Conference o |
| `mai2020multiscalerepresentationlearningspatial` | arXiv:2003.00824 | arXiv API; OpenAlex API | Multi-Scale Representation Learning for Spatial Feature Distributions using Grid Cells \| ICLR 2020, Apr. 26 - 30, 2020, Addis Ababa, ETHIOPIA \| 2020 |
| `mai2023sphere2vecgeneralpurposelocationrepresentation` | arXiv:2306.17624 | arXiv API; OpenAlex API | Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions \| ISPRS Journal of Pho |
| `maninis2019attentive` | DOI 10.1109/CVPR.2019.00195 | Crossref REST; OpenAlex API | Attentive Single-Tasking of Multiple Tasks \| 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) \| 2019 \| type proceedings-a |
| `mikolov2013negsampling` | arXiv:1310.4546 | arXiv API; OpenAlex API | Distributed Representations of Words and Phrases and their Compositionality \| arXiv preprint \| 2013 \| type posted-content |
| `mikolov2013word2vec` | arXiv:1301.3781 | arXiv API; OpenAlex API | Efficient Estimation of Word Representations in Vector Space \| arXiv preprint \| 2013 \| type posted-content |
| `misra2016cross` | DOI 10.1109/CVPR.2016.433 | Crossref REST; OpenAlex API | Cross-Stitch Networks for Multi-task Learning \| 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) \| 2016 \| type proceedings-ar |
| `moura2025mobilityaware` | DOI 10.1109/MSWiM67937.2025.11308734 | Crossref REST; OpenAlex API | On the Design of Mobility-Aware Systems: A Tourist’s Perspective \| 2025 International Conference on Modeling, Analysis and Simulation of Wireless and |
| `nash` | no identifier in the bib entry | arXiv API; OpenAlex API | Multi-Task Learning as a Bargaining Game \| arXiv preprint \| 2022 \| type posted-content |
| `paiva2026stmtlnet` | DOI 10.5753/courb.2026.22960 | Crossref REST; OpenAlex API | ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa \| Anais do X Workshop de Computação Urbana (CoUrb 2026 |
| `pedregosa2011sklearn` | no identifier in the bib entry | arXiv API; OpenAlex API; PDF in repo: Pedregosa2011_ScikitLearn.pdf | Scikit-learn: Machine Learning in Python \| Journal of Machine Learning Research (2011) \| 2012 \| type posted-content |
| `perez2018film` | DOI 10.1609/aaai.v32i1.11671 | Crossref REST; OpenAlex API | FiLM: Visual Reasoning with a General Conditioning Layer \| Proceedings of the AAAI Conference on Artificial Intelligence \| 2018 \| type journal-arti |
| `perozzi2014deepwalk` | DOI 10.1145/2623330.2623732 | Crossref REST; OpenAlex API | DeepWalk \| Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining \| 2014 \| type proceedings-article |
| `rahmani2019category` | DOI 10.1145/3341981.3344240 | Crossref REST; OpenAlex API | Category-Aware Location Embedding for Point-of-Interest Recommendation \| Proceedings of the 2019 ACM SIGIR International Conference on Theory of Info |
| `ruder2017mtloverview` | arXiv:1706.05098 | arXiv API; OpenAlex API | An Overview of Multi-Task Learning in Deep Neural Networks \| arXiv preprint \| 2017 \| type posted-content |
| `ruder2017sluice` | arXiv:1705.08142 | arXiv API | Latent Multi-task Architecture Learning \| arXiv preprint \| 2017 \| type posted-content |
| `russwurm2024geographiclocationencodingspherical` | arXiv:2310.06743 | arXiv API; OpenAlex API | Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks \| Published as a conference paper at ICLR 2024 \| 2023 \ |
| `santos2024urban` | no identifier in the bib entry | NOT RESOLVED at any source of record | None \| (no venue in record) \| None \| type None |
| `sener2018mgda` | arXiv:1810.04650 | arXiv API; OpenAlex API | Multi-Task Learning as Multi-Objective Optimization \| arXiv preprint \| 2018 \| type posted-content |
| `senushkin2023aligned` | no identifier in the bib entry | Crossref REST; OpenAlex API | Independent Component Alignment for Multi-Task Learning \| 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) \| 2023 \| type  |
| `silva2019urbancomputing` | DOI 10.1145/3301284 | Crossref REST; OpenAlex API | Urban Computing Leveraging Location-Based Social Network Data \| ACM Computing Surveys \| 2019 \| type journal-article |
| `silva2025mtlnet` | DOI 10.21528/CBIC2025-1191324 | Crossref REST; OpenAlex API | An Investigation into Multi-Task Learning for Point-of-Interest Category Classification and Next-POI Prediction \| Anais do XVII Congresso Brasileiro  |
| `sitzmann2020implicit` | arXiv:2006.09661 | arXiv API; OpenAlex API | Implicit Neural Representations with Periodic Activation Functions \| arXiv preprint \| 2020 \| type posted-content |
| `sokolova2009measures` | DOI 10.1016/j.ipm.2009.03.002 | Crossref REST; OpenAlex API; PDF in repo: sokolova2009.pdf | A systematic analysis of performance measures for classification tasks \| Information Processing &amp; Management \| 2009 \| type journal-article |
| `song2010limits` | DOI 10.1126/science.1177170 | Crossref REST; OpenAlex API; PDF in repo: 201002-19_Science-Predictability.pdf | Limits of Predictability in Human Mobility \| Science \| 2010 \| type journal-article |
| `standley2020tasks` | arXiv:1905.07553 | arXiv API; OpenAlex API | Which Tasks Should Be Learned Together in Multi-task Learning? \| arXiv preprint \| 2019 \| type posted-content |
| `sun2020go` | DOI 10.1609/aaai.v34i01.5353 | Crossref REST; OpenAlex API | Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation \| Proceedings of the AAAI Conference on Artific |
| `sun2024mcmg` | DOI 10.1145/3592789 | Crossref REST; OpenAlex API | A Multi-channel Next POI Recommendation Framework with Multi-granularity Check-in Signals \| ACM Transactions on Information Systems \| 2023 \| type j |
| `sun2024transtarec` | DOI 10.1109/ICCEA62105.2024.10603711 | Crossref REST; OpenAlex API | TransTARec: Time-Adaptive Translating Embedding Model for Next POI Recommendation \| 2024 5th International Conference on Computer Engineering and App |
| `sun2025kgtb` | DOI 10.48550/arXiv.2509.12350 | arXiv API; OpenAlex API | Knowledge Graph Tokenization for Behavior-Aware Generative Next POI Recommendation \| arXiv preprint \| 2025 \| type posted-content |
| `tang2020ple` | DOI 10.1145/3383313.3412236 | Crossref REST; OpenAlex API | Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations \| Fourteenth ACM Conference on Recomme |
| `vandenhende2022mtl` | DOI 10.1109/TPAMI.2021.3054719 | Crossref REST; OpenAlex API | Multi-Task Learning for Dense Prediction Tasks: A Survey \| IEEE Transactions on Pattern Analysis and Machine Intelligence \| 2021 \| type journal-art |
| `vaswani2017attention` | arXiv:1706.03762 | arXiv API | Attention Is All You Need \| arXiv preprint \| 2017 \| type posted-content |
| `velickovic2019deep` | no identifier in the bib entry | OpenAlex API | Deep Graph Infomax \| Apollo (University of Cambridge) \| 2018 \| type conference-paper |
| `velivckovic2017graph` | arXiv:1710.10903 | arXiv API | Graph Attention Networks \| arXiv preprint \| 2017 \| type posted-content |
| `vielhaus2022handover` | DOI 10.1145/3551660.3560913 | Crossref REST; OpenAlex API | Handover Predictions as an Enabler for Anticipatory Service Adaptations in Next-Generation Cellular Networks \| Proceedings of the 20th ACM Internatio |
| `wang2025hamtl` | DOI 10.1007/s11227-025-07643-7 | Crossref REST; OpenAlex API | Hierarchy aware-based multi-task learning for user location prediction \| The Journal of Supercomputing \| 2025 \| type journal-article |
| `wei2022finetuned` | URL https://openreview.net/forum?id=gEZrGCozdqR | arXiv API; OpenAlex API | Finetuned Language Models Are Zero-Shot Learners \| arXiv preprint \| 2021 \| type posted-content |
| `wilcoxon1945` | DOI 10.2307/3001968 | Crossref REST; OpenAlex API; PDF in repo: wilcoxon1945.pdf | Individual Comparisons by Ranking Methods \| Biometrics Bulletin \| 1945 \| type journal-article |
| `wongso2025massivesteps` | no identifier in the bib entry | arXiv API; OpenAlex API | Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins -- Dataset and Benchmarks \| arXiv preprint \| 2025 \| type posted-conten |
| `wu2024torchspatial` | arXiv:2406.15658 | arXiv API; OpenAlex API | TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning \| arXiv preprint \| 2024 \| type posted-content |
| `Xia2020` | DOI 10.3390/app10196664 | Crossref REST; OpenAlex API | MTPR: A Multi-Task Learning Based POI Recommendation Considering Temporal Check-Ins and Geographical Locations \| Applied Sciences \| 2020 \| type jou |
| `xin2022domtl` | no identifier in the bib entry | arXiv API; OpenAlex API | Do Current Multi-Task Optimization Methods in Deep Learning Even Help? \| arXiv preprint \| 2022 \| type posted-content |
| `Xu2023` | DOI 10.1145/3582553 | Crossref REST; OpenAlex API | TME: Tree-guided Multi-task Embedding Learning towards Semantic Venue Annotation \| ACM Transactions on Information Systems \| 2023 \| type journal-ar |
| `yang2015tsmc` | DOI 10.1109/TSMC.2014.2327053 | Crossref REST; OpenAlex API | Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs \| IEEE Transactions on Systems, Man, and Cybernetics:  |
| `yang2020flashback` | DOI 10.24963/ijcai.2020/302 | Crossref REST; OpenAlex API | Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States! \| Proceedings of the Twenty-Ninth International Joint Co |
| `yang2022getnext` | DOI 10.1145/3477495.3531983 | Crossref REST; OpenAlex API | GETNext \| Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval \| 2022 \| type proceedings |
| `ye2013nextmove` | DOI 10.1137/1.9781611972832.19 | Crossref REST; OpenAlex API | What's Your Next Move: User Activity Prediction in Location-based Social Networks \| Proceedings of the 2013 SIAM International Conference on Data Min |
| `yu2020catdm` | DOI 10.1145/3366423.3380202 | Crossref REST; OpenAlex API | A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data \| Proceedings of The Web Conference 2020 \| 2020 \| type procee |
| `yu2020pcgrad` | no identifier in the bib entry | arXiv API; OpenAlex API | Gradient Surgery for Multi-Task Learning \| arXiv preprint \| 2020 \| type posted-content |
| `yu2024survey` | arXiv:2404.18961 | arXiv API; OpenAlex API | Unleashing the Power of Multi-Task Learning: A Comprehensive Survey Spanning Traditional, Deep, and Pretrained Foundation Model Eras \| arXiv preprint |
| `zeng2019next` | DOI 10.1007/978-3-030-30146-0_21 | Crossref REST; OpenAlex API; Semantic Scholar API | A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-attention \| Lecture Notes of the Institute for Computer Sciences, So |
| `Zhang2020` | DOI 10.24963/ijcai.2020/491 | Crossref REST; OpenAlex API | An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins \| Proceedings of the Twenty-Ninth International Joi |
| `zhang2021survey` | DOI 10.1109/TKDE.2021.3070203 | Crossref REST; OpenAlex API | A Survey on Multi-Task Learning \| IEEE Transactions on Knowledge and Data Engineering \| 2022 \| type journal-article |
| `zhu2022drrgnn` | DOI 10.1145/3529091 | Crossref REST; OpenAlex API | Predicting a Person’s Next Activity Region with a Dynamic Region-Relation-Aware Graph Neural Network \| ACM Transactions on Knowledge Discovery from D |

**Six references read as full PDFs**, not only as records:

| Key | PDF | Why the full text was needed |
|---|---|---|
| `standley2020tasks` | arXiv 1905.07553v3, downloaded this session | ITEM 3: the abstract alone cannot establish what the paper does and does not support |
| `caruana1997multitask` | fetched open-access, DOI 10.1023/A:1007379606734 | cited 13 times, the most-cited key in the document |
| `huang2023hgi` | `science/articles/Learning urban region representations ... .pdf` | 8 citing sites, all needing the POI-region-city hierarchy and the corruption mechanism |
| `song2010limits` | `science/articles/201002-19_Science-Predictability.pdf` | the 93 percent figure in Chapter 1 is a quoted number |
| `pedregosa2011sklearn` | `science/articles/Pedregosa2011_ScikitLearn.pdf` | the twice-ruled site at `2_fundamentals.tex:465` |
| `sokolova2009measures` | `science/articles/sokolova2009.pdf` | the macro-F1 weighting claim in Appendix D |

Also read from the repository, not from a publisher: `moura2025mobilityaware`
(`articles/[mobiwac]/mobility/On_the_Design_of_Mobility-Aware_Systems_A_Tourists_Perspective.pdf`,
for the machine-learning-as-future-work claim at `5_mobiwac.tex:810`) and `santos2024urban`
(`articles/dissertacao/exemples/germano/Dissertação_Mestrado___Germano.pdf`, the only entry with no
external source of record).

**Numbers quoted in this report, and where each comes from.** No number here was computed by me;
counts were produced by a script over the source and are reported as it printed them.

| Number | Source | Convention |
|---|---|---|
| 247 sites / 265 key instances | parse of the eleven chapter files plus `tables/frame/lineage.tex`, `%` comments stripped | a multi-key `\cite` counts once per key; comment-only sites excluded |
| 255 raw `\cite` / 273 raw keys | `grep` over the same files **including** comments | given only to account for the difference |
| 100 bib entries, 0 duplicate keys, 0 dangling cites, 0 uncited | parse of `src/references.bib` against the union of used keys | |
| 105 / 100 pages | `Output written on` in `build/main.log` and `build/main_final.log` | after my edits, both builds converged |
| 104 pages at HEAD | `git show HEAD:articles/dissertacao/src/dissertacao.pdf`, page count read from the PDF | the committed artifact ANCHORS.md measured |
| `Overfull \hbox (113.58371pt too wide)` | `build/main.log`, first attempt at the Sphere2Vec errata row | pt too wide, as pdflatex reports it |
| 44 open / 44 close braces | brace count over the comment-stripped `bib_errata.tex` | delta 0, the `{\small}` group intact |
| 244,987 characters | extracted text of the Germano dissertation PDF | the corpus searched for ethics-committee terms |

## 10 · `[VERIFY]` flags

Four, all narrow and each with a named check that would close it:

1. **`[VERIFY: 3_cbic.tex:244, nash]`** — "task weights can be updated less frequently,
   significantly reducing runtime while maintaining performance". Not in the abstract of
   arXiv:2202.01017. Check: locate the passage in the paper body. This same subsection has already
   had two other cost claims corrected against this paper, so it has been read here before.
2. **`[VERIFY: 3_cbic.tex:145, huang2022estimating]`** — the edge-weight formula
   `w_ij = log((1+D^1.5)/(1+d_ij^1.5))` is attributed to the cited work. Check: locate the formula
   in the paper, or restate it as this work's own construction.
3. **`[VERIFY: 4_courb.tex:60, rahmani2019category]`** — Crossref returns a truncated four-sentence
   abstract for this ICTIR paper. The categorical half of the citing sentence is supported; the
   sequential half is not visible in the record. Check: the paper body.
4. **`[VERIFY: 5_mobiwac.tex:750, huang2024cslsl]`** — the sentence cites CSLSL's own ablation (its
   chain against a shared-trunk parallel variant). An abstract does not carry ablations. Check: the
   paper's results section.

A fifth, narrower still and recorded rather than flagged: `5_mobiwac.tex:409` asserts that CTLE's
training never sees the category vocabulary. That is consistent with the abstract, which names only
locations and temporal information as inputs, but an abstract cannot prove an absence.

## 11 · What I could not confirm

Stated plainly, since a smoothed-over gap is the defect this repository keeps catching.

1. **The fan-out the task specified did not happen.** `host.delegate` is unavailable in this frame
   (root-only; this frame is a leaf). I ran all seven units myself. The per-chapter reports exist and
   nothing was sampled, but the **independence** of seven fresh readers does not. AGENT_GUARDRAILS L6
   asks for fresh eyes; this is one pair. Treat it accordingly.
2. **Four entries return no abstract at any source of record I can reach**, so for those the check
   ran on the record and title only: `capanema2023poirgnn` (Elsevier), `wang2025hamtl` (Springer),
   `zeng2019next` (Springer chapter), and `Halder2021` (Springer, cited once at `3_cbic.tex:125`).
   Their publisher landing pages are outside the network allowlist; Crossref deposits no abstract and
   Semantic Scholar returns the records with empty abstract fields. Every citing site for these keys
   is an identity-of-baseline or pattern-continuation pointer that the resolved record supports; no
   mechanism claim rests on any of them. But I did not read those four papers.
3. **The comparative results claims at `5_mobiwac.tex:664-666` were not audited.** What a citation
   must support there is each baseline's identity, and each record does. Whether the joint model is
   in fact above every external baseline on both tasks across all six datasets is a **number** claim
   under AGENT_GUARDRAILS §2, whose single source of truth for Chapter 5 is `RESULTS_BOARD.md`. That
   is a different audit and I did not perform it. The same applies to every reported figure in
   Chapters 3 and 4 that sits beside a citation.
4. **The `[VERIFY]` flags in section 10 are open**, four of them, each needing a page from a paper
   body I did not obtain.
5. **`santos2024urban` has no external identifier and never will**, being a UFV master's
   dissertation. I verified it against the document in the repository, which is the best available
   evidence and is not a source of record in the Crossref sense. The entry should carry a bib comment
   saying so, or a future existence-checker will read the absence as a defect. I did not add that
   comment: it is a bib edit outside ITEM 2's remit.
6. **The `make check` page-count gate fails and I did not fix it.** Its own message misdiagnoses the
   cause (`main.log` does carry the page count). Running `sync_page_counts.py --write` would edit
   `CLAUDE.md`, `PLAN.md` and other files outside my remit while the count is still moving under
   other agents' work. Hand-off, not a fix.
7. **I did not verify the load-bearing ranking in section 2 against the author's own sense of the
   argument.** It is my reading of what each claim carries. The dispositions follow the ranking, so if
   the ranking is wrong the priorities are wrong; the evidence per row does not depend on it.

## 12 · Recommended order of work for the author

1. **Rule on the Standley site** (section 5.4, option A recommended). It is the one row the author
   asked for by name, it is a claim rather than a typo, and the drafted sentence and Appendix B row
   are ready to apply.
2. **Take the four other high-load rows together**, since three of the four are one-key swaps to
   references already in the bibliography: `ruder2017sluice` -> `baxter2000model` at `3_cbic.tex:213`,
   `belkin2003laplacian` -> `Xu2023` at `4_courb.tex:219`, and adding `standley2020tasks` beside
   Caruana at `5_mobiwac.tex:44`. Only `4_courb.tex:161` needs a prose decision rather than a key.
3. **Fix the three paired defects as pairs** (`Xia2020` at two sites, the Rußwurm encoder at two
   sites, the TorchSpatial "first validated" attribution at a frame site and in the published CoUrb
   introduction), so the document does not describe one system two ways.
4. **Batch the remaining low-load Chapter 3 wording rows** into one Appendix B wording-table pass:
   `:102`, `:115`, `:123` (both keys), `:125`, `:127`. Each is one clause.
5. **Close or accept the four `[VERIFY]` flags.**
6. **Re-run `sync_page_counts.py --write`** once the other agents' edits have landed and the page
   count has stopped moving.

---

**Companion reports** (one per unit, each with its own counts, full verdict table and source ledger):
`11_claims_1_introduction.md`, `11_claims_2_fundamentals.md`, `11_claims_3_cbic.md`,
`11_claims_4_courb.md`, `11_claims_5_mobiwac.md`, `11_claims_6_conclusion.md`,
`11_claims_appendices.md`. The machine-readable verdict set is `_final_verdicts.json` in this folder,
with the intermediate resolution and screening data alongside it.

