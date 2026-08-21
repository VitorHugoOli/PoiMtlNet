# Research question and the arc — the settled content

> The one question and the three-beat arc as the frame tells them.
> Rendered in: Ch.1 §1.3 (`drafts/1_introduction.tex`), closed by Ch.6 §6.2.
> Process trail: `archive/process/01_arc_and_logline/`, `11_full_arc_rereview/`.

## The research question (verbatim spine)

**Does multi-task learning help point-of-interest prediction — next category and next region —
and what does the answer depend on?**

## The arc, beat by beat (with the licensed wording)

1. **CBIC (Ch.3, setup)** — MTLnet: place-level graph embedding + hard sharing beneath the two
   task heads; tasks = category classification + next category. "Did not consistently
   outperform" the dedicated models ("beat" is banned) and "cost more to train". Reported as a
   finding with THREE candidate explanations (task dissimilarity / representation too poor /
   restrictiveness of hard sharing). Time-indexed: "held for the configuration of its time".
   NEVER: "CBIC called for better representations" (its future work proposed the architecture
   door). Approved form: the null was hypothesized and the results "lend weight to" it.
2. **CoUrb (Ch.4, diagnosis)** — tested the representation explanation FIRST "as the cheapest
   controlled test among the three" (no door metaphor in prose). Architecture fixed, input
   replaced (64-d monolithic DGI → decomposed spatial+temporal+categorical). Category macro-F1
   +20.2 to +22.0 pp (AUDITED values — the paper's published 16/21 was recounted to 15/21+1 tie;
   use audited numbers + errata note). Diagnosis time-indexed: "at that stage of the research".
   Boundary (approved Item 6): CoUrb's only baseline is MTLnet — it does not revisit
   MTL-vs-single-task; Chapter 5 reopens that question.
3. **MobiWac (Ch.5, resolution)** — the mechanism-as-hypothesis sentence: any place-level
   embedding gives a place the same vector on every visit; "cannot tell a weekday lunch from a
   Saturday night out". Check-in level = one vector per visit. TWO changes, both named, always:
   representation AND sharing topology (cross-attention between two task-specific streams,
   acting on another of CBIC's candidate explanations). The pair settles on next category +
   next region. Payoff with bound verbs: category outperforms at all six datasets (five U.S.
   states + Istanbul — count precedes the claim); region outperforms at four of six, TOST
   non-inferior (±2 pp) at Alabama and Arizona. Status: submitted, under review.

## The two-factor law (F1)

Every payoff summary names BOTH factors — the check-in-level representation AND the redesigned
sharing topology. Never representation alone.

## Working title (D2, advisor decides)

"Multi-Task Learning for Point-of-Interest Classification and Prediction Tasks: The Role of the Check-in-Level Representation" — with three alternates in the `drafts/1_introduction.tex`
header comment block.