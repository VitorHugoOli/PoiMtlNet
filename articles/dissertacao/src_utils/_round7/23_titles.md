# 23_titles.md — candidate titles, grounded in defended and CTD-selected work

**The advisor's guidance (2026-07-28):** a dissertation title should be plainer and more direct than
an article title, because that is what the community keeps; and CTD is where to look.

## The evidence base

Two corpora, both on disk, both measured rather than recalled:

- **CTD 2026**, all 16 selected theses and dissertations of the *XXXIX Concurso de Teses e
  Dissertações*, fetched to `exemples/ctd2026/` with `PROVENANCE.md`. These are the national
  competition's own selections, so their conventions carry a verdict rather than a preference.
- **The five UFV exemplars** already in `exemples/`, one of which (Germano) is *also* CTD-selected
  and shares this dissertation's advisor and lab.

Measured over the sixteen:

| property | value |
|---|---|
| words | mean **9.7**, median **10**, min **7**, max **15** |
| use a colon | **9 of 16** — so a colon is the majority form, not a flourish |
| open with a rhetorical *From / Toward / Beyond / Rethinking* | **0 of 16** |
| open with a gerund (*Automating*, *Designing*, *Managing*) | 3 of 16 |
| lead with a named artifact (*Caramel:*, *Oraculum:*, *FoT-PDS:*) | 4 of 16 |
| of the colon titles, words **before** the colon | mean **2.7** (min 1, max 7) |

Reproduce any of it:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
python3 - <<'PY'
import json, statistics
recs = json.load(open('/tmp/ctd2026.json'))          # or re-derive from exemples/ctd2026/PROVENANCE.md
W = [len(r['title'].split()) for r in recs]
print(len(recs), 'titles; mean', round(sum(W)/len(W),1), 'median', statistics.median(W),
      'min', min(W), 'max', max(W))
print('colon:', sum(1 for r in recs if ':' in r['title']))
PY
```

## What this says about the current title

    From Representations to a Single Joint Model: Multi-Task Learning for
    Point-of-Interest Category and Region Prediction

Three findings, in order of how much they matter:

1. **The rhetorical opener is the outlier.** *"From X to Y"* appears in **0 of 16** selected titles.
   This is the single clearest signal in the data, and it is exactly what the advisor was pointing at.
2. **The structure is inverted relative to the convention.** The CTD colon pattern puts the **object**
   first and the **approach** second, with a short left side (mean 2.7 words). Germano's — same
   advisor, same lab, CTD-selected — is the clean instance: *Urban Region Representation Learning*
   (the topic, 4 words) : *A Positional and Structural Graph Approach* (the method, 6 words). The
   current title puts a **narrative** first (7 words naming no object) and the topic second.
3. **The colon itself is fine.** 9 of 16 use one. Nothing here argues for removing it.

The length is defensible either way: the current title is at the top of the observed range, not
outside it.

## Candidates

Each keeps the two tasks findable, uses only names in `GLOSSARY.md` (verified: *next category*,
*next region*, *check-in-level representation*, *place embedding*, *joint model*, *multi-task
learning* are all registered), and drops the rhetorical opener. Word counts are measured, not typed.

| | words | colon | candidate |
|---|--:|:-:|---|
| **A** | 13 | yes | Multi-Task Learning for Next-Category and Next-Region Prediction: The Role of the Check-in-Level Representation |
| **B** | 6 | no | Check-in-Level Representations for Multi-Task Point-of-Interest Prediction |
| **C** | 11 | yes | Multi-Task Learning for Point-of-Interest Category and Region Prediction: A Check-in-Level Study |
| **D** | 7 | no | Representation and Sharing in Multi-Task Point-of-Interest Prediction |
| **E** | 9 | no | A Check-in-Level Joint Model for Next-Category and Next-Region Prediction |

### What each gains and gives up

**A — Multi-Task Learning for Next-Category and Next-Region Prediction: The Role of the Check-in-Level Representation**

- 13 words, inside the CTD range; colon, object first.
- Gains: Object first (the task pair), method second — Germano's structure exactly.
- Gives up: at 13 words it is the longest here, though still inside the observed 7-15 range.

**B — Check-in-Level Representations for Multi-Task Point-of-Interest Prediction**

- 6 words, **below** the CTD range (observed 7-15); no colon.
- Gains: Names the contribution as an object, no colon, and it is the plainest form on the list. At 6 words it is shorter than any of the sixteen, which reads as confident rather than terse.
- Gives up: Does not name the two tasks; a reader must open the abstract to learn which.

**C — Multi-Task Learning for Point-of-Interest Category and Region Prediction: A Check-in-Level Study**

- 11 words, inside the CTD range; colon, object first.
- Gains: closest to the current title with the opener removed and the object promoted; colon, object first, both tasks findable.
- Gives up: 'Study' is weaker than the result warrants (one joint model beat both dedicated models).

**D — Representation and Sharing in Multi-Task Point-of-Interest Prediction**

- 7 words, inside the CTD range; no colon.
- Gains: names the two variables the dissertation actually manipulated (representation, sharing).
- Gives up: Abstract; a committee member scanning a list may not see the tasks.

**E — A Check-in-Level Joint Model for Next-Category and Next-Region Prediction**

- 9 words, inside the CTD range; no colon.
- Gains: names the artifact AND both tasks, uses registry terms verbatim, and needs no colon.
- Gives up: Foregrounds the third paper; the arc across all three is implicit.

## The recommendation, and it is a recommendation only

**E** — *A Check-in-Level Joint Model for Next-Category and Next-Region Prediction* — sits closest to
the measured centre: 9 words against a mean of 9.7, no rhetorical opener, both tasks
findable by someone scanning a list, and every term drawn from the registry.

**C** is the conservative choice if you would rather change as little as possible: it is the current
title with the opener removed and the object promoted.

The reservation about **E**, stated plainly because it is the reason not to pick it: it names the
third paper's artifact, so the honest arc across all three papers becomes implicit rather than
announced. If you want the arc in the title, **A** carries it but runs long.

**[NEEDS SIGN-OFF]** The title is the author's decision. Nothing here is applied; `0_main.tex` still
carries the current title.
