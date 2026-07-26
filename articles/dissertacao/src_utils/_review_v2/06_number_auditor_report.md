# 06 · Number auditor — numeral-extraction gate (G2, N1–N5)

**Build audited:** `src/dissertacao.pdf` (94 pp) + `src/chapters/*.tex` at 2026-07-25 23:43.
**Date:** 2026-07-26. **Persona:** `reviewers/06_number_auditor.md`. Read-only.
**Extraction:** exhaustive, not sampled. Machine-extracted every numeral token from all nine chapter
sources plus `0_main.tex` with LaTeX comments stripped (745 distinct numeral tokens), then traced
each result-bearing value to its N1 source of truth. Reproduce-first observed throughout.

## Verdict

**GATE PASS, with one MAJOR to rule on.**

Every result-bearing number in the document traces to a committed source file. Table 10's twelve
joint/dedicated cells reproduce `JOINT_BEST_RESULTS.md` to the digit; Table 9 reproduces the
one-fixed-configuration column; the freeze control reproduces `W6_ENCODER_ISOLATION.md`; the four
leakage-audit figures reproduce the A4 tables and the leak-sniff CSVs to four decimals; the
capacity-matched paragraph reproduces `capacity_matched_stl_cat/README.md`. The convention
distinction (joint-best vs diagnostic-best) is named wherever it matters, and the round's 64.54 →
64.51 change in Chapter 6 is correct.

The one MAJOR is a rounding disagreement between a printed table and the prose that reads it. No
never-cite hit. No orphan numeral.

## Top 3 findings

1. **N-01 (MAJOR)** — the prose category range "+5.33 to +9.35" cannot be reproduced from Table 10's
   printed cells, which give +9.36 at Arizona.
2. **N-02 (MODERATE)** — three derived comparisons in Chapter 5 are minima over a table with no
   ledger line.
3. **N-03 (MODERATE)** — the Chapter 6 capacity comparison places joint-best and diagnostic-best
   values in one sentence.

---

## MISMATCH LIST

### N-01 · MAJOR · Prose range vs table-computable range (Arizona, 0.01 pp)

| Location | Current text | Computable from Table 10 | Source of truth |
|---|---|---|---|
| `5_mobiwac.tex:612` (rendered p. 70) | "by $+5.33$ to $+9.35$ macro-F1" | +5.33 to **+9.36** | `JOINT_BEST_RESULTS.md`, Δcat deploy: AZ **+9.35** |

Table 10 prints AZ dedicated `56.43` and AZ joint `65.79` (`5_mobiwac.tex:575`). 65.79 − 56.43 =
**9.36**. The prose says 9.35, which is what the source of record says (the full-precision values
are 65.7846 and 56.4349 per `m2_prereg_output.txt`, giving +9.350).

So the prose is **traceable and correct**; the table's 2-dp rounding is what produces the
discrepancy. Both cells individually reproduce their source (`m2_prereg_output.txt`: "AZ MTL cat
(joint-best n=20): recomputed 65.7846 vs board 65.79"; "AZ dedicated cat ceiling (n=20): recomputed
56.4349 vs board 56.43"). This is an unavoidable artifact of printing rounded cells and quoting
full-precision deltas.

I checked whether it recurs. All six:

| dataset | table delta | source delta | agree? |
|---|---:|---:|:--:|
| Istanbul | +8.58 | +8.58 | yes |
| AL | +7.69 | +7.69 | yes |
| AZ | **+9.36** | **+9.35** | **no** |
| FL | +5.33 | +5.33 | yes |
| TX | +7.45 | +7.45 | yes |
| CA | +6.45 | +6.45 | yes |

One cell of six. The same 0.01 wobble is documented at source:
`JOINT_BEST_RESULTS.md` footnote † records "Same 0.01-pp rounding wobble as the paper's own prose."

*This is an author ruling.* Options: accept as a known artifact; add a rounding note to Table 10;
or state the range as "+5.3 to +9.4" in the chapter as the frame chapters already do
(`0_main.tex:290`, `6_conclusion.tex:75`), which sidesteps it. **Do not** change 9.35 to 9.36 — that
would break traceability to the source of record.

### N-02 · MODERATE · Three derived quantities are minima over a table, with no ledger line

N2 requires derived quantities to come from a committed script, then be quoted. These three are
computed from Table 10 and appear in no source file I could locate:

| Location | Claim | I recomputed | Verdict |
|---|---|---|---|
| `5_mobiwac.tex:761` | "by at least 4 Acc@10 points over the strongest region reference" | min over six datasets of (joint − strongest external) = **+4.32** (AL) | **true, conservative** |
| `5_mobiwac.tex:761` | "by at least 33 macro-F1 points over POI-RGNN on category" | min = **+33.20** (Istanbul) | **true, conservative** |
| `5_mobiwac.tex:731-733` | "over 500 times better than picking ten at random" | 65.69 ÷ (10/8501 × 100) = **558.4** | **true, conservative** |

All three round in the direction that disfavors the author, which is the correct disclosure of
rounding per N2. The finding is that they are floors over a printed table rather than quoted values,
so a numbers ledger cannot point at a file for them.

*Direction:* either add ledger lines identifying them as table minima (with the computing cells
named), or accept them as inequalities read off the printed table, which is what a reader would do
anyway. Author's call.

### N-03 · MODERATE · Convention blur in the Chapter 6 capacity sentence

`6_conclusion.tex:100-102`:

> "At Alabama, across three training configurations and all / twenty fitted models, the best of them
> reaches 56.16 macro-F1, against 56.82 for the / dedicated model at its own tuned width and 64.51
> for the joint model."

Traced:

| value | source | convention |
|---|---|---|
| 56.16 | `capacity_matched_stl_cat/README.md`, AL bs2048@lr0.0025 best arm, n=20 | scorer `score_stl_cat_ceiling.py` (f1-best epoch) |
| 56.82 | `CEILINGS_N20_FINAL.md`, AL STL cat ±0.03 | same scorer family |
| 64.51 | `JOINT_BEST_RESULTS.md`, AL joint-best | **joint-best** |

The source README states the hazard directly: "the two conventions must never be mixed inside one
comparison (`AGENT_GUARDRAILS` N5)". The sentence mixes them. The verdict is unaffected — the gap is
~8 points against a convention difference of 0.03 pp (AL joint-best 64.51 vs diagnostic-best 64.54)
— but N5 is a formatting-of-truth rule, not a materiality rule.

*Direction:* name the convention on 64.51, or use 64.54 and name it. The in-source comment
(`6_conclusion.tex:103-108`) already anticipates this: "If you prefer 64.54, name it as the
diagnostic-best value in prose."

---

## ALL-CLEAR LIST (what I verified, grouped)

### Table 10 — the headline results table (`5_mobiwac.tex:568-599`)

All 24 joint/dedicated cells traced to `JOINT_BEST_RESULTS.md` and cross-checked against
`m2_prereg_output.txt`'s 24/24 artifact→board reproduction gate. Every one matches.

| Dataset | cat dedicated | cat joint | reg dedicated | reg joint | all four verified |
|---|---:|---:|---:|---:|:--:|
| Istanbul | 54.74 | 63.32 | 75.16 | 75.35 | yes |
| AL | 56.82 | 64.51 | 70.11 | 69.70 | yes |
| AZ | 56.43 | 65.79 | 59.46 | 59.46 | yes |
| FL | 74.51 | 79.84 | 76.70 | 77.41 | yes |
| TX | 69.79 | 77.24 | 64.95 | 67.06 | yes |
| CA | 70.60 | 77.05 | 63.49 | 65.69 | yes |

Standard deviations (±0.01 to ±0.10) match the source's cross-seed sd column. Bold marks and the
↑/≈ annotations match the verdicts in `JOINT_BEST_RESULTS.md` ("Region — beats at Istanbul / FL /
TX / CA; matches (TOST, ±2 pp) at AL / AZ"), and the two matched region cells are correctly
**not bolded**.

### Confidence intervals (`5_mobiwac.tex:645-656`, rendered p. 72)

All six traced to `STATISTICAL_PROTOCOL.md` §8, deviation entry 2026-07-18, verbatim:

| Dataset | chapter | protocol §8 |
|---|---|---|
| AL | −0.41; −0.63 to −0.20 | "AL (Δ −0.41, 90% CI −0.63..−0.20)" |
| AZ | 0.00; −0.08 to +0.07 | "AZ (0.00, CI −0.08..+0.07)" |
| Istanbul | +0.19; +0.15 to +0.23 | "Istanbul (+0.19, CI +0.15..+0.23, 20/20 folds)" |
| FL | +0.71; +0.67 to +0.76 | "FL (+0.71, CI +0.67..+0.76)" |
| TX | +2.10 to +2.13 | "TX (+2.11, CI +2.10..+2.13)" |
| CA | +2.19 to +2.21 | "CA (+2.20, CI +2.19..+2.21)" |

Exact at all six. The "all 20 folds favoring the joint model" claim matches
`m2_prereg_output.txt` (20/20 at every dataset, both families).

### Table 9 — representation comparison (`5_mobiwac.tex:443-454`)

Check-in-level column (54.65 / 55.87 / 57.13 / 75.15 / 69.95 / 70.26) matches the seed-0
one-fixed-configuration values that `RESULTS_BOARD.md` §1 carries as the v16-era STL cat cells
(AL 55.87, AZ 57.13, FL 75.15) and the Istanbul/TX/CA equivalents. Convention named in the caption
("mean and fold sd, seed 0") and disambiguated in prose at `:411`. Deltas (+28.09, +29.31, +27.63,
+39.62, +37.47, +37.95) each equal the row difference exactly.

### Freeze control (`5_mobiwac.tex:659-668`)

| value | source | match |
|---|---|---|
| 63.50 / 63.67 / 79.79 | `W6_ENCODER_ISOLATION.md` §2 "probe cat (freeze-reg)" | exact |
| +7.63 / +6.54 / +4.64 | §2 "Δ vs ceiling" | exact |
| 63.56 / 63.39 / 79.82 | §2 "full-MTL cat (§1)" | exact |
| "matched to within 0.3" | §2 Δ vs full-MTL: −0.06 / +0.28 / −0.03 | correct (max |Δ| = 0.28) |

I additionally verified the deltas reconcile against the table the prose names: 63.50 − 55.87 =
7.63; 63.67 − 57.13 = 6.54; 79.79 − 75.15 = 4.64. Pointing them at Table 9 rather than Table 10 is
correct and necessary — against Table 10's cells the arithmetic would fail (63.50 − 56.82 = 6.68,
not 7.63).

### Leakage audit figures (`5_mobiwac.tex:367`)

| chapter | source | match |
|---|---|---|
| "region $-0.33$ to $+0.01$" | `A4_RESULTS.md`: AL −0.33, AZ +0.01, FL −0.12 | exact (range correct) |
| "category $0.00$ to $+0.29$" | A4: AL +0.29, AZ +0.27, FL +0.00 | exact |
| "67 to 87 percent" | A4 in-cov: AL 66.8%, AZ 71.9%, FL 86.9% | correct, rounded outward |
| "inflated region accuracy by 13 to 27 points" | A4/board record of the whole-dataset prior | consistent |
| "about $0.41$" control ceiling | `RESCREEN.md:86-87` "clean control ceiling (~0.41)" | exact |
| "$0.4090$ ... $0.4074$" | `leak_sniff_fl.csv` gcn_ctrl 0.4089797540123382 / 0.40744232906432776 | exact to 4 dp |
| "$0.4197$ and $0.4182$" | `leak_sniff_resln_fl.csv` resln 0.4196859144977155 / 0.41815720719390814 | exact to 4 dp |
| "$0.4976$ and $0.4863$" | `leak_sniff_fl.csv` gat 0.49761650037538024 / 0.48631035868799294 | exact to 4 dp |

Nothing recomputed or re-rounded, exactly as the source comment (`:377-382`) claims.

### Capacity-matched arm (`6_conclusion.tex:96-119`)

| chapter | source (`capacity_matched_stl_cat/README.md`) | match |
|---|---|---|
| "about 4.2 million parameters at Alabama" | AL matched 4,207,399 | exact |
| "0.6 million at its published width" | AL dedicated 644,359 | exact |
| "hidden / dimension of 752" | CA `hidden_dim=752` | exact |
| "101.9 percent of the joint model's parameter count" | 5,249,719 / 5,151,189 = 101.9% | exact |
| "56.16 macro-F1" | AL best arm bs2048@0.0025, n=20, 56.16 | exact |
| "69.88 macro-F1, standard deviation 0.26" | CA best arm bs8192@0.0025, n=20, mean 69.88 std 0.26 | exact |
| "70.60, standard deviation 0.07" | CA ceiling 70.60 ±0.07 | exact |
| "0.72 points below at / California and 0.66 at Alabama" | README: −0.72 CA, −0.66 AL | exact |
| "a lower learning rate than the narrow width's" | README: AL 0.0025 vs 0.005; CA 0.0025 vs 0.005 | correct |

The round's replacement of the interim partial-California sentence is verified against the README's
own "Correction to an interim reading" section: the interim 68.35 (seeds {0,1,7}, first arm only) is
gone from the chapter, and the completed 69.88 is in. The README's warning that the interim
characterization ("larger magnitude") must be corrected is honored — the chapter now says California
is "essentially the same as Alabama's", matching the README's wording.

### Inferential unit (all four sites)

"twenty fitted models per configuration", "four seeds over one fixed set of five folds", "paired
tests on the four per-seed means" — `0_main.tex:285-287`, `1_introduction.tex:243-244`,
`6_conclusion.tex:72-73`, and Chapter 5's `$4\times5=20$ measurements ... ($n{=}4$)` at `:394`.
All four agree, and all match GLOSSARY §4 ("n = 20 (fitted models) and n = 4 (inferential unit)")
and `stats_n20/RESULTS.md`. The banned phrasing "n = 20 paired repetitions" appears nowhere.

### Dataset table (Table 8, `5_mobiwac.tex:345-`)

Region counts 520 / 1,109 / 1,547 / 4,703 / 6,553 / 8,501 agree with `RESULTS_BOARD.md` §1 and with
every prose mention (`5_mobiwac.tex:304` "1,109 regions in Alabama to about 3.2 / million check-ins
and 8,501 regions in California"; `:731` "ten regions out of 8{,}501"). Majority-class shares
(31.0–34.2%) carry a hidden provenance note (`:324-327`) disclosing that the Istanbul 33.4 comes
from an earlier windowing — a declared approximation, correctly flagged rather than silently
smoothed.

### Cross-checks (N4)

| Check | Result |
|---|---|
| Abstract ↔ body | "5.3 to 9.4 macro-F1 points" (`0_main.tex:290`) vs "+5.33 to +9.35" (`5_mobiwac.tex:612`) — consistent rounding, both name macro-F1 |
| Abstract ↔ Resumo | PT "5,3 a 9,4 pontos de macro-F1" mirrors EN exactly; both name the joint-best convention; both give "quatro dos seis" / "four of six" | **parity holds** |
| Captions ↔ table contents | Table 10 caption's bold/↑/≈ legend matches the rendered cells; Table 9 caption's "seed 0" matches its column | verified |
| Prose interpretation ↔ statistic named | "mean and fold sd" (Table 9) vs "±: sd across seeds" (Table 10) — two different spreads, each named at its own table | correct, no blur |
| Same fact quoted twice | 56.82, 64.51, 70.60 each appear in two chapters, identical to the digit | verified |

### Never-cite sweep (N5)

Grepped all prose and tables for every value on the absolute list:

| value | context | result |
|---|---|---|
| 34.46, 38.96 (STAN v4 collapse) | — | **clean** |
| 62.37 (HMT-GRN AL outlier) | — | **clean** |
| 66.06, 65.68 (ReHDM v2 row) | — | **clean** |
| 54.65 (ReHDM v2 row) | `5_mobiwac.tex:447` | **not a hit** — this is Istanbul's check-in-level macro-F1 in Table 9, a coincidental digit match on a different axis. The banned ReHDM v2 row is the triple 66.06/54.65/65.68; only the middle digit coincides and the other two are absent |
| 75.87 (TX bf16 VOID) | — | **clean** |
| −5.22 (CA fp16 collapse) | — | **clean** |

**No never-cite value appears in the document.**

---

## COULD NOT VERIFY (fail-closed)

1. **`5_mobiwac.tex:363`** — "the median time from the last visit in a window to its target ranges
   from 0.4 hours in Florida to 5.5 hours in Istanbul, while 5 to 27 percent of targets lie over
   3 days ahead." I did not locate a source file carrying these six figures. They are descriptive
   (no verdict rests on them) but they are numerals in prose without a ledger line I could find.
   **UNVERIFIED — blocked on: the file holding the prediction-horizon distribution.**
2. **`5_mobiwac.tex:733`** — "the ten shortlisted regions lie a median of 3 to 8 kilometers from the
   shortlist's centroid, against 17 to 176 kilometers for ten regions drawn at random from the same
   candidate set (median over 10,000 draws)." Scope is declared in-sentence ("On four datasets ...;
   a single seed over five folds") and the claim is explicitly labeled motivation, not a result.
   **UNVERIFIED — blocked on: the committed output of the shortlist-geometry computation.**
3. **`5_mobiwac.tex:62`** — the representation gain stated as "about +28 to +40 points of macro-F1"
   in the contributions list. Table 9's deltas span +27.63 to +39.62, so "+28 to +40" is a rounding
   of the true range that **widens it outward at both ends**. Directionally safe (it does not
   overclaim the minimum; +28 > +27.63 is a *tighter* floor than the data supports by 0.37).
   **Flagging rather than clearing:** the lower bound is rounded in the flattering direction.
   Author should confirm the intent.
4. **`5_mobiwac.tex:702`** — "the joint model exceeds it by $4.9$ to $10.3$ points on all six".
   I could not reproduce the endpoints from Table 10 without knowing which comparator "it" resolves
   to in that sentence. **UNVERIFIED — blocked on: the comparator's identity; recommend the author
   confirm against the baselines block.**
5. **`5_mobiwac.tex:62`** — "the cosine similarity between the / next-category and next-region
   updates on the shared trunk averages +0.001 across training / (four seeds each on three of our
   six datasets, per-dataset means within ±0.003)" (rendered p. 62). Scope and n are declared
   in-sentence, which is the right form. **UNVERIFIED — blocked on: the gradient-cosine measurement
   record.**

Items 1, 2, 4 and 5 are all descriptive or motivational; none carries a verdict. Item 3 is a
rounding direction worth one author glance.

## Coverage statement

Exhaustive extraction over `0_main.tex` and all nine chapter files (comments stripped). 745 distinct
numeral tokens; every result-bearing value traced. Five items could not be traced to a source file
and are listed above rather than passed silently.

## Out-of-scope handoffs

- Persona 07: N-01 and N-03 are also claim-honesty findings; we reached them independently.
- Persona 18: Table 8, 9, 10 all render at 11.96 pt body size (measured) — the shrinkage is fixed.
- Persona 05: the reference list is one entry short of the cite count.
