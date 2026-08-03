# 45_prose_diagnostic.md — the AI-tell pattern, measured before any rewriting

Round 12, 2026-08-03. Baseline `8f17f294`. The author's complaint, verbatim: "os textos estao com cara
I.A. Eles comecam com uma oracao pequena seguida por ponto final e sem conexao segue para uma sequencia
de outras oracoes. Sem contar que varios conceitos logicos estao sem conexao dificultando o fluxo de
leitura do texto."

This is a `WRITING_LAW` §4 complaint, and it is measurable, so it was measured BEFORE any prose was
touched. Otherwise "improved" is an impression and the improvement cannot be checked.

## Method

Comments stripped. Display environments (equation, align, table, tabular, tabularx, figure, itemize,
enumerate, definition) removed, so the numbers describe running prose and not markup. Citation and
reference macros collapsed to a single token so they count as one word rather than zero. Paragraphs
shorter than 120 characters ignored. "Connectives" counts a fixed list of subordinating and coordinating
words (however, therefore, because, since, although, while, whereas, which, thus, hence, but, yet, as a
result, in contrast, instead, rather, then, when, if, unless, after, before, once) normalized per
sentence, which is a proxy for clause linkage and not a measure of quality.

## Baseline

| file | paragraphs | median opening sentence | paragraphs opening <=12 words | median sentence | sd | connectives per sentence |
|---|--:|--:|--:|--:|--:|--:|
| `2_fundamentals.tex` | 59 | 16 w | **41%** | 16 w | 7.6 | **0.35** |
| `1_introduction.tex` | 16 | 13 w | **50%** | 18 w | 7.4 | **0.23** |
| `6_conclusion.tex` | 16 | 16.5 w | **38%** | 16 w | 8.3 | **0.21** |

His description is accurate. Between 38 and 50 percent of paragraphs open on a clause of twelve words or
fewer, and connective density sits between 0.21 and 0.35 per sentence, meaning roughly three sentences in
four carry no explicit link to the one before it.

**24 paragraphs show BOTH symptoms at once** (opening sentence of 11 words or fewer AND 0.34 or fewer
connectives per sentence). Those are the targets. The list is in `/tmp/prose_worst.json` at measurement
time and reproducible with the method above.

## What the numbers are NOT

They do not say the prose is wrong. Several of the flagged paragraphs are correct and deliberately
clipped: a short opener followed by three qualifying sentences is the honest shape for a limitation, and
`6_conclusion.tex`'s "That gain needs two qualifications." is doing exactly the job it should. A rewrite
that raises the connective count by welding those clauses together would damage them.

So the metric is a FINDING AID, not a target to optimize. The instruction to any rewriting pass is to
read the 24 paragraphs and fix the ones where the reader genuinely loses the logical thread, then
re-measure to show the direction moved, without treating a specific number as the goal.

## One instance is mine, from today

`1_introduction.tex`: "The prediction targets are the next category and the next region." followed by two
unconnected sentences. I wrote that sentence this morning under FAB-15. It is a fair example of the exact
pattern he is objecting to, produced while I was busy being careful about a different rule.
