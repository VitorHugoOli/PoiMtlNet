# 01 — "are other terms shown before being explained?" — the mechanical sweep

Round 13, 2026-08-03. Baseline commit `5d31632a`. The author asked two things after FAB-34 landed:
put the fuller `mahalle` account in the fundamentals dataset section, and *"have you eval if there
was other similar points that we are showing before explaning?"*

## 1. Why this sweep was re-run rather than reused

The sweep I ran with FAB-32/33/34 used a **hand-typed list of terms**, so it could only find defects
I had already thought of. That is the instrument weakness `check_register.py`'s own header warns
about, and this repository has paid for it more than once. These three tests derive their candidates
**from the document**.

## 2. The three tests

**Test 1 — the `\emph{}` convention.** The document's rule is that `\emph{term}` marks a term's
introduction. So for every `\emph{}` term, did the same words appear in **plain** text earlier?
Candidates come from the file, not from me.
**Result: zero in Chapter 1, zero in Chapter 2.** The convention is clean.

**Test 2 — acronyms extracted by SHAPE.** Every CamelCase or all-caps token, checked against the
`siglas` block (12 entries: CBIC, Check2HGI, CoUrb, DGI, FiLM, HGI, LBSN, MobiWac, MTL, POI, SBRC,
TOST) and an initials test on a window around the first use. The discriminator that made this usable:
**a token with a `\cite` within ~130 characters is a cited system's proper name** and needs no
expansion. That correctly cleared ST-RNN, HST-LSTM, DeepMove, GeoSAN, GETNext, CTLE, CatDM, RGNN,
DeepWalk, GraphSAGE, Deep InfoMax, DSelect-k, CAGrad, GradNorm, FAMO, MCARNN, CSLSL, TME, HAMTL,
IeMTLF, SIREN and Massive-STEPS.

**Test 3 — foreign-language and administrative terms**, which is the class FAB-34 belonged to.

## 3. What survived verification: two findings

**S-1 — `PCGrad` was used 323 lines before it was introduced.** First use at
`2_fundamentals.tex:988` in §2.3.2 ("PCGrad changes a direction only when this cosine is negative"),
no citation, no gloss. The sentence that introduces it ("Yu et al. introduce PCGrad, which projects
away a conflicting component", with `yu2020pcgrad`) is at `:1311`, in §2.3.5. Same shape as
FAB-32/33: a named thing arriving before the sentence that names it.
**Applied:** a five-word appositive plus the citation at first use. §2.3.5 keeps its full treatment
and now reads as the fuller account rather than the first mention. Probe `R13-s1pcgrad`.

**S-2 — the introduction named two external baselines with no citation and no gloss.**
`1_introduction.tex:455-457` had "HMT-GRN keeps its shared multitask skeleton" and "STAN has its
output layer adapted" as bare names. Chapter 2 introduces both properly (`:329` STAN, `:345`
HMT-GRN) and both bib keys existed, but in Chapter 1 they appeared **only in the comment ledger**,
never in prose.
**Applied:** each gets the gloss Chapter 2 already uses, plus its citation, so the two chapters
describe them identically. Probes `R13-s2base`, `R13-s2base2`.

## 4. What the sweep raised and verification KILLED

| candidate | why it is correct as it stands |
|---|---|
| `MTLnet`, Ch.1 `:151` | introduced in the same sentence ("introduces MTLnet, the first joint model developed in this research") |
| `OOD`, Ch.2 `:1597` | expanded in place ("out-of-distribution discounted, or OOD-discounted") |
| `MacroF1`, Ch.2 `:1563` | a LaTeX operator name inside the equation that defines it, not prose |
| `XVII`, `DOI`, `ACM`, `CBIC2025`, Ch.1 | venue metadata and an identifier, not document vocabulary |
| `PENDENCIAS`, `FAB`, Ch.1 `:292` | inside a comment, not prose |
| "the three studies", Ch.2 `:13` | Chapter 1 introduces them first; reading order is fine |

## 5. An instrument defect I found and fixed mid-sweep

My first acronym pass reported line numbers **wrong by about twenty lines**, because it searched a
LaTeX-stripped string while mapping positions from the unstripped one. Fixed by masking commands
with **equal-length spaces** so every character position is preserved, asserted with
`len(masked) == len(joined)`. The candidate list did not change; the coordinates did, and I would
have published wrong ones. Recorded because a coordinate I cannot reproduce is worse than no
coordinate.

## 6. The author's first request: the fuller account in §2.4.1

Both units are now defined there, and **both** rather than only the Turkish one. Measured across the
whole live tree first: **neither unit had ever been defined anywhere.** "census tract" appears in six
files and is never explained; the closest thing is Chapter 5's "A census tract is a neighborhood, not
a radio cell", which says what it is *not*. Glossing only `mahalle` would have implied the American
unit needs no explanation, which is the asymmetry a reader outside the United States notices first.

The paragraph now says which dataset supplies which unit, then what kind of object each is, then what
makes them comparable. Probes `R13-mahfull`, `R13-mahfull2`.

**A regression the gate caught, and it was mine.** My first version of this paragraph defined both
units but dropped the clause binding each unit to its datasets, which broke `R13-aut18b` — an
existing probe whose whole purpose is that §2.4.1 names the unit each dataset actually supplies. The
binding is restored. This is the second time this round that an existing probe caught me removing a
property while adding a different one.

## 7. [PROPOSED REGISTRY ROW — needs the author's approval before it can be used]

The Turkish word for the elected head of a `mahalle` is **not** in `GLOSSARY.md`, and §1's rule is
fail-closed: an agent proposes, the author approves, and the row lands **before** the term reaches
prose. So §2.4.1 says "an elected local administrator" and the term is proposed here instead of
minted in the chapter. Probe `R13-mahterm` fails if it appears in Chapter 2 prose while no row exists.

| Term | Proposed definition | Notes |
|---|---|---|
| muhtar | The elected head of a `mahalle`, the smallest unit of Turkish local administration. | Would let §2.4.1 name the office instead of describing it. **Only needed if you want the term itself in the text**; the current wording works without it. PT: *muhtar* |

## 8. Source ledger

| Claim in the text | Source | Where it was checked |
|---|---|---|
| A census tract is a statistical area drawn by the Census Bureau to hold a roughly stable population, revised each decennial census, with no government of its own | U.S. Census Bureau geographic-area definitions | opened this session |
| A mahalle is the smallest unit of Turkish local administration, one level below the district, with an elected local administrator | Turkish local-government administrative-division references | opened this session |
| HMT-GRN uses a predicted region to constrain the search for a place | `Lim2022` — Lim, Hooi, Ng, Goh, Weng, Tan, *Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation*, ACM SIGIR 2022, DOI 10.1145/3477495.3531989 | bib entry read in `references.bib`; gloss copied from Ch.2 `:345`, which already cites it |
| STAN connects non-adjacent visits through spatio-temporal correlations | `luo2021stan` — Luo, Liu, Liu, *STAN: Spatio-Temporal Attention Network for Next Location Recommendation*, WWW 2021, DOI 10.1145/3442381.3449998 | bib entry read in `references.bib`; gloss copied from Ch.2 `:329`, which already cites it |
| PCGrad projects away a conflicting gradient component | `yu2020pcgrad`, already cited at `:982` and `:1311` in this chapter | existing citation reused, not a new reference |

**No new bibliography entry was added.** The two citations added to Chapter 1 are keys that already
existed and were already cited in Chapter 2, so no reference was introduced on my word.

## 9. [VERIFY] flags

None outstanding for the applied text. One thing the author may want to rule on: whether §2.4.1
should also name the office (item 7), which is a wording preference rather than a defect.
