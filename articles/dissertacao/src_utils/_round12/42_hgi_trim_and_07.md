# Item 42 - shorten the HGI mechanism detail, and add the 0.7 sentence

Round 12. Baseline commit **8f17f294**. File edited: `src/chapters/2_fundamentals.tex` (only).
`src_utils/check_audit_claims.py` was **not** edited: R11-hgi still matches its pinned wording
verbatim after the rewrite (evidence below), so no repin was owed.

## 1 - The author's instruction, verbatim

> "Vamos tentar diminuir um pouco dos detalhes do HGI, quem tiver interessado pode ir ao texto
> original, a explicacao aqui deve se conter no que fizemos de diferente apra adapa-lo para
> checking, e ja explicamos isso, apesar de nao entrar em muitos detalhes. Sobre citar que na
> dissertacao usamos 0,7, podemos citar por alto, algo como: 'Em experimentos avaliamos que para
> nossos datasets 0,7 gerava os melhores resultados', como vamos remover o appendix E nao precisa
> referenciar ele."

And, from his numbered list:

> "HGI -- diminuir os detalhes de mecanismo. Quem tiver interesse vai ao artigo original. A
> explicacao aqui deve se conter no que fizemos de diferente para adapta-lo ao check-in, e isso ja
> esta explicado, mesmo sem muitos detalhes."

And the non-negotiable, his words:

> "a ressalva honesta ... NAO e detalhe de mecanismo e deve sobreviver ao corte."

## 2 - The LIVE text before the edit (quoted from the file, not from the brief)

`chapters/2_fundamentals.tex:220-276`, six paragraphs, read immediately before editing:

```
220: HGI is the place-level representation that the final study replaces, so its mechanism
221: fixes the reference point of the main comparison of this dissertation. Its own stated
222: goal is to learn urban region representations from POIs in a fully unsupervised
223: manner, and it reaches that goal by extending the two scales of DGI to three: POI,
224: region, and city. The name states what is maximized. Training raises the mutual
225: information between representations at two adjacent levels of that hierarchy, and it
226: does so without evaluating that quantity in closed form. HGI follows the contrastive
227: paradigm and scores pairs instead. A bilinear discriminator combines two embeddings
228: through a learned weight matrix and passes the result through a logistic function,
229: and the loss rewards a high score for a true pair and a low score for a false pair.
230: No label of any downstream task enters that comparison, so the representation is
231: obtained without supervision.
232:
233: The hierarchy is assembled in five stages. A pretrained category encoder supplies the
234: initial POI features, because the category of a POI already accounts for much of its
235: meaning. All POIs of a study area are then connected into a Delaunay graph whose edge
236: weights decrease with the distance between two POIs and are reduced further when the
237: two POIs lie in different regions. One graph convolution layer updates each POI
238: embedding into a transformed combination of the POI itself and its spatial context.
239: The POI embeddings inside one region are aggregated by multi-head attention, where
240: each head represents one perspective on how important a POI is in defining its
241: region. A second graph, whose nodes are regions and whose edges join regions that
242: share a border, propagates information between neighbors and yields the region
243: embeddings. An area-weighted sum over those embeddings produces one city embedding.
244:
245: Two examples from the original paper show what the levels are for. A hotel inside an
246: airport complex and a hotel on a university campus share a category and differ in
247: meaning, and only the spatial neighborhood separates them. The POI level does that
248: work. Two regions that each contain one company and one restaurant are almost
249: indistinguishable from their own contents. If one of them is surrounded by commercial
250: regions and the other by industrial areas, the region graph makes the difference
251: visible.
252:
253: The objective has one term for each adjacent pair of levels, and a single weight
254: balances them. The first term treats a region embedding and the POI embeddings
255: located inside that region as a true pair, and takes POI embeddings from another
256: region as the false pair. The second term treats the city embedding and the region
257: embeddings as a true pair. Its false pair uses region embeddings recomputed after the
258: POI feature matrix has been shuffled row by row, so every POI keeps its place in the
259: graph but receives the category embedding of a different POI.
260:
261: One consequence of this design matters for the later chapters. The training signal
262: updates the POI encoder, the aggregation function, and the region encoder together,
263: so a POI embedding is optimized to score high against the embedding of its own region
264: and low against the embeddings of other regions. Region membership also enters
265: earlier, through the edge weights of the POI graph. The POI-level output is therefore
266: not a description of the place in isolation. It already reflects the region the place
267: belongs to, and the authors state the matching property for their own product, whose
268: region representations are both locally and globally relevant.
269:
270: HGI was developed and evaluated for urban region representation. Its reported
271: experiments estimate urban functional distributions, population density, and housing
272: price in Xiamen Island and Shenzhen, and the object under evaluation there is the
273: region embedding. The POI embedding is an internal stage in producing it. This
274: dissertation repurposes that POI-level output for sequential prediction, a use the
275: original evaluation does not cover. In this role, HGI supplies the place-level
276: baseline in the later studies and the direct basis of Check2HGI.
```

Also read live before editing, and this is what decided constraint (c): **the file contained no
`\ref{apx:hgi-tuning}` to remove.** Measured from `src/`, comments stripped:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao/src
grep -vn '^[[:space:]]*%' chapters/2_fundamentals.tex | grep -c "apx:hgi-tuning"   # -> 0
```

So the "removing that reference is YOUR job" half of the item was already satisfied at baseline.
Nothing was added, and the count is still 0 after my edit (re-measured, same command).

## 3 - What I changed, and where

### 3a - The trim: six paragraphs to two (2_fundamentals.tex:220-246 after the edit)

Cut, as mechanism detail a reader can get from the original paper:
the five-stage enumeration as an enumeration; the two worked examples (airport hotel / campus
hotel, and the two one-company-one-restaurant regions); the whole loss-term paragraph (positive
and negative pairs, the row-wise feature shuffling for the city term, the single balancing
weight); the sentence naming the contrastive paradigm as such; the per-head reading of multi-head
attention; the region graph as its own sentence; the authors' locally-and-globally-relevant
property.

Kept, because the later chapters rest on them: the stated goal and the unsupervised training; the
extension of DGI's two scales to three; that mutual information is raised without being evaluated
in closed form; the bilinear discriminator and its logistic function; and, whole and unaltered,
the paragraph that carries the honest caveat.

Kept **compressed into one sentence** rather than cut, because the second surviving paragraph is
the one that explains *how region context reaches a POI embedding*, which is exactly what the
later chapters use: the category encoder, the one graph convolution layer over the Delaunay graph,
the multi-head aggregation, the area-weighted city sum.

Added one sentence, at 220-222, that does the referral he asked for:

> The mechanism itself is set out in the original paper, and the account below is limited to what
> the later chapters rely on.

The honest caveat paragraph (`:248-254` after the edit) is **byte-identical** to the live text at
baseline. It was not touched.

### 3b - Orphaned ledger lines, cut or repointed

The `[round11]` source ledger at `:275+` page-anchors every mechanism claim to `huang2023hgi`.
Six rows were rewritten or removed so that no comment claims to source a sentence that no longer
exists:

| Ledger row | Action |
|---|---|
| `five stages` | REMOVED (the phrase is gone from prose) |
| `hotel example` | REMOVED with the example |
| `two-regions example` | REMOVED with the example |
| `one balancing weight` (Eq. 11 alpha) | REMOVED with the paragraph |
| `locally and globally relevant` | REMOVED with the clause |
| `contrastive, not a closed-form MI` | RENAMED to `no closed-form MI` and annotated: the contrastive-paradigm sentence was cut, and what the row now sources is the surviving "without evaluating that quantity in closed form" |
| `one GCN layer + what it produces` | NARROWED: the paper's "transformed combination ... captures its uniqueness" quote left prose; the row now sources only "adds spatial context" |
| `multi-head heads` | NARROWED to `multi-head attention`: the per-head sentence left prose; the aggregation survives |
| `positives/negatives` | NARROWED to the region-level pair only, which is what "score high against ... its own region and low against ... other regions" rests on; the city-level shuffling row is annotated as cut with its prose |
| `Delaunay + weights` | REPOINTED: it used to say "NO NUMBER ... quoting 0.4 here would collide". The published 0.4 is now stated once, in the adaptation paragraph, so the row records that the same Eq. 2 sentence sources both values and why they do not collide |

Every removal is annotated `[round12, item 42]` in place, so a later reader sees a cut and not a
gap.

### 3c - The 0.7 paragraph (2_fundamentals.tex:256-263 after the edit)

Placed immediately after the honest-caveat paragraph, which is where the reader has just been told
what HGI was evaluated for:

> The move to other study areas left one setting of that baseline open. Huang et al.\ reduce an
> edge between POIs of different regions to 0.4 of its weight in the experiments named above, and
> this work did not transfer that value untested. In experiments on one of the state datasets of
> this dissertation, over the same five folds used elsewhere, a value of 0.7 gave the best
> category F1 among the values tried, and the later studies adopt it. What that comparison settles
> is one hyperparameter of the place-level baseline. It is not evidence about how HGI and
> Check2HGI compare.

**(a) Number protocol.** The paragraph quotes exactly one number of its own, `0.7`, and that
number is not a result: it is the shipped default,
`research/embeddings/hgi/preprocess.py:23`, `DEFAULT_CROSS_REGION_WEIGHT = 0.7`. The *result*
behind it is stated qualitatively, "the best category F1 among the values tried", which is
readable off the source of record without computing anything.
Source of record, `research/embeddings/hgi/README.md:544` (table header, "5 folds x 50 epochs")
and `:548-551`:

```
| w_r | Cat F1              |
| 0.4 | 0.7388 +/- 0.0205   |
| 0.5 | 0.7678 +/- 0.0211   |
| 0.6 | 0.7944 +/- 0.0186   |
| 0.7 | 0.8186 +/- 0.0123   |
```

The largest of the four is the 0.7 row, so "best among the values tried" is the table's own
ordering. It contradicts none of those cells, and none of them are restated in Chapter 2: the
values with their spreads and their averaging convention live in `chapters/apx_g_hgi_tuning.tex`,
which I did not edit. **NUM-4 pins `0.8186` in that file and reports `holds` after my edit**
(`/tmp/c2.log:111`). The chapter writes "category F1" and not "macro-F1" because the standing
`[VERIFY: averaging convention of the swept "Cat F1"]` in the NUM-4 ledger earlier in section 2.2
is still open; my sentence inherits that hedge rather than resolving it.
"one of the state datasets" is Alabama (`README.md:544` heads the table "Alabama w_r sweep"). The
state is not named in prose because the sweep is a single state and the sentence claims no more
than that; naming it would invite the reader to expect a per-state sweep the source does not have.

**(b) Scope limit: I carried it INSIDE the sentence's vicinity, and here is why.** The warning
used to be the last two sentences of `chapters/apx_g_hgi_tuning.tex`. That appendix is no longer
in the defense volume: another agent removed it from `src/content.tex` this round on the author's
instruction ("remover A.1, C.3 e E, mandando esse conteudo para o material extra"), and it now
renders in the supplementary volume, which grew from 22 to 26 pages. Confirmed in the rendered
defense PDF: the string "Adaptation of the HGI Baseline" is **absent**. A warning in a volume the
Chapter 2 reader no longer has is not a warning, so the limit is now the paragraph's own last two
sentences. I did not edit the appendix file (not mine, and another agent is in it), so the warning
now exists in both volumes, which is correct rather than duplicative: each volume's reader gets it
once.

**(c) No reference to appendix E.** There was none to remove (measured above), and none was added.
The paragraph names no appendix and no label.

**The published 0.4 was located firsthand this session** in the source PDF on disk,
`science/articles/Learning urban region representations with POIs and hierarchical graph infomax.pdf`,
**p.137 (PDF sheet 4), section 3.3**, Equation 2 and the sentence under it:

> "wr is a factor to differentiate intra- (wr = 1) and cross-region (wr = 0.4) edges"

and, in the next sentence, "The choices of the parameters ... and wr = 0.4 (cross-region) are in
view of the previous practices". Prose states it as a reduction *to 0.4 of* the edge weight, which
is what `wr = 1` versus `wr = 0.4` in that sentence says. "the experiments named above" is the
preceding paragraph's Xiamen Island and Shenzhen, so the study areas are named once in the
subsection and not twice.

## 4 - Sources opened this session, with the claim located in each

| Source | Identifier | Where opened | Claim located |
|---|---|---|---|
| Huang et al., "Learning urban region representations with POIs and hierarchical graph infomax", ISPRS J. Photogramm. Remote Sens. 196 (2023) 134-145 | doi `10.1016/j.isprsjprs.2022.11.021` | the PDF on disk, `science/articles/Learning urban region representations with POIs and hierarchical graph infomax.pdf`, text layer extracted with pypdfium2 | **p.137 (sheet 4) sec. 3.3, Eq. 2**: "wr is a factor to differentiate intra- (wr = 1) and cross-region (wr = 0.4) edges" -> the published 0.4 in my new sentence. **p.134 (sheet 1) abstract**: the three downstream tasks and "the study areas of Xiamen Island and Shenzhen, China" -> the honest caveat, which I kept unchanged |
| The repo sweep table | `research/embeddings/hgi/README.md:544, :548-551` | read this session | the four swept weights, their category F1 with spreads, "5 folds x 50 epochs", and the table title "Alabama w_r sweep" -> "over the same five folds used elsewhere" and "best ... among the values tried" |
| The shipped default | `research/embeddings/hgi/preprocess.py:23` (and the comment at `:36-40`) | read this session | `DEFAULT_CROSS_REGION_WEIGHT = 0.7` -> "the later studies adopt it" |
| The appendix whose warning I relocated | `src/chapters/apx_g_hgi_tuning.tex` (read only) | read this session | its closing two sentences, "This sweep supports adaptation of one baseline hyperparameter; it is not evidence about the relative performance of HGI and Check2HGI" -> the substance of my last two sentences |
| The probe file | `src_utils/check_audit_claims.py:95-96` (NUM-4), `:470-478` (R11-hgi) | read this session | the pinned patterns, verified against the live file |

No new bibliography entry was added; `huang2023hgi` was already cited in the subsection.
No new glossary term was introduced. Every technical word in the new paragraph is already
registered or already live: HGI, Check2HGI, POI, region, place-level baseline, fold, category F1.
`hyperparameter` was already in the live text of the appendix I quoted from.

## 5 - R11-hgi: it holds, and I validated the instrument by sabotage

The rewrite did not touch the pinned sentence, so no repin was owed. But "it holds" from a probe I
did not prove could fail is an assertion, so:

```bash
# occurrences of the target string in the WHOLE file, prose and comments: 1
# (so a count=1 replace was not the trap here, but I replaced ALL occurrences anyway)
# mutate: "repurposes that POI-level output" -> "reuses that POI-level output", every occurrence
python3 src_utils/check_audit_claims.py   # from articles/dissertacao/
```

Result read BEFORE restoring:

```
rc = 1
NOT APPLIED R11-hgi   the HGI explanation keeps the honest caveat ...
FAIL: R11-hgi are recorded as APPLIED and are not in the document.
```

Restored, and byte-identity of the restored file confirmed in the same cell
(`restored identical: True`). On the live file the probe reports `holds` (`/tmp/c2.log:182`).

## 6 - The six exit codes (rule 6), each run separately from `src/`, exit code read directly

| Command | rc |
|---|--:|
| `make defense` | **0** |
| `make academico` | **0** |
| `make ppgc` | **0** |
| `make extra` | **0** |
| `make check` | **2** |
| `make selftest` | **0** |

`make check` is rc 2, and **the failing gate is not mine.** One gate fails, "the author-facing
verification commands actually return what they claim", on block 6 of
`src_utils/_round6/VERIFY_LIST.md`:

```
FAIL  VERIFY_LIST.md: python3 -c "
      output does not contain 'repair_in_prose: True'
```

That block asserts a sentence in `chapters/apx_a_contributions.tex`. Attribution, measured:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
git show 8f17f294:./src/chapters/apx_a_contributions.tex \
  | grep -c "stratified its folds by sample rather than by user"      # -> 1  (present at baseline)
git diff 8f17f294 -- src/chapters/apx_a_contributions.tex | grep -n "stratified its folds"
# -> 76:-Chapter~\ref{ch:courb} stratified its folds by sample rather than by user and ran in a
```

A concurrent agent deleted that line (that file lost 76 lines this round, per `git diff --stat`).
It is not a file I may edit, and the probe's other half, which reads *my* file, passes
(`retired_clause_in_prose: False`). The gate was already red before my edit for the same reason.
I did not weaken, reword, or repoint it. **Flagged for whoever owns `apx_a_contributions.tex`
this round: block 6 of `_round6/VERIFY_LIST.md` needs repointing to wherever that sentence went,
or the sentence needs restoring.**

The page-count gate went red after my edit, as the suspended budget predicts, and I ran the
sanctioned fix:

```bash
cd src && python3 ../src_utils/sync_page_counts.py --write   # 7 claim(s) updated
```

After the sync the page-count gate is green (no `STALE` lines in the re-run). Every other gate
reports OK, and the two probes this item bears on both report `holds`: **NUM-4** and **R11-hgi**.

## 7 - Page counts

| Volume | Before (baseline record) | After (measured from the build) |
|---|--:|--:|
| defense | 108 pp | **104 pp** |
| academico | 105 pp | **101 pp** |
| ppgc | 109 pp | **105 pp** |
| extra | 22 pp | **26 pp** |

Measured with pypdfium2 on `src/build/main.pdf`, `main_academico.pdf`, `main_ppgc.pdf`,
`main_extra.pdf`, and independently by the page-count gate from the build logs, which agrees.
The four-page drop and the four-page rise in the supplementary volume are **not both mine**: the
concurrent appendix relocation moves pages between the two volumes. My own trim removes about
thirty lines of prose and adds eight, so roughly half a page of the defense-volume drop is mine.
I did not attempt to separate the two, and I am not claiming a per-item page delta.

## 8 - Verification in the rendered PDF, both directions

`src/build/main.pdf`, text layer via pypdfium2, whitespace-normalized (so the assertions survive
LaTeX line wrapping):

PRESENT:
- "a value of 0.7 gave the best category F1 among the values tried"
- "It is not evidence about how HGI and Check2HGI compare."
- "What that comparison settles is one hyperparameter of the place-level baseline."
- "The mechanism itself is set out in the original paper"
- "multi-head attention aggregates the POI embeddings of a region"
- "repurposes that POI-level output for sequential prediction, a use the original evaluation does not cover" (the caveat, intact)

ABSENT:
- "The hierarchy is assembled in five stages"
- "A hotel inside an airport complex"
- "Two regions that each contain one company and one restaurant"
- "The objective has one term for each adjacent pair of levels"
- "HGI follows the contrastive paradigm and scores pairs instead"
- "shuffled row by row"
- "both locally and globally relevant"
- "Two examples from the original paper"

One PRESENT assertion initially read as MISSING: "reduce an edge between POIs of different regions
to 0.4 of its weight". The extracted text shows why, and it is a false alarm rather than a defect:
a running header splits the sentence across the page break. The rendered paragraph, quoted from
the PDF text layer:

> "The move to other study areas left one setting of that baseline open. Huang et al. reduce
> `Chapter 2. Fundamentals 22` an edge between POIs of different regions to 0.4 of its weight in
> the experiments named above, and this work did not transfer that value untested. In experiments
> on one of the state datasets of this dissertation, over the same five folds used elsewhere, a
> value of 0.7 gave the best category F1 among the values tried, and the later studies adopt it.
> What that comparison settles is one hyperparameter of the place-level baseline. It is not
> evidence about how HGI and Check2HGI compare."

Also verified in the defense PDF: `0.8186` does **not** appear (it lives in the supplementary
volume with its table), and "Adaptation of the HGI Baseline" does **not** appear, which is the
independent confirmation that the appendix left this volume and that carrying the scope limit
inline was necessary rather than merely tidy.

## 9 - Rule-4 compliance on the new prose

No em-dash (checked by the em-dash gate, OK). No contraction (contractions gate, OK). No repo
codename (codenames gate, OK). American English (register gate, OK). "outperforms" and "matches"
do not appear in anything I wrote; the strongest verb in the new paragraph is "gave the best
category F1 among the values tried", which is an ordering read off a table and not a test verdict,
and it is immediately scoped by the two sentences that follow it.

## UNFINISHED

Nothing in this item is unfinished. Two things belong to other owners and are recorded here rather
than fixed by me:

1. **`make check` rc 2**, from block 6 of `src_utils/_round6/VERIFY_LIST.md`, whose asserted
   sentence was deleted from `chapters/apx_a_contributions.tex` by a concurrent agent this round.
   Not my file, and red before my edit for that same reason. Needs repointing or restoring by
   whoever owns that file.
2. **The standing `[VERIFY: averaging convention of the swept "Cat F1"]`** in the NUM-4 ledger is
   still open. My sentence inherits its hedge and writes "category F1" rather than "macro-F1"; the
   author still owes the convention, or the decision to keep the clause qualitative.

One thing for the author to decide, not a defect: the scope limit on the sweep now appears in both
volumes, inline in Chapter 2 and again in the supplementary appendix, because I may not edit that
appendix. If he prefers it in one place only, the appendix's closing two sentences are the ones to
drop.
