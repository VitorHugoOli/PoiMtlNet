# 07 · Claim & honesty auditor — G2 claim-registry gate (C1–C4 + WRITING_LAW §3)

**Build audited:** `src/dissertacao.pdf` (94 pp) + `src/chapters/*.tex`, both at 2026-07-25 23:43.
**Date:** 2026-07-26. **Persona:** `reviewers/07_claim_honesty_auditor.md`. Read-only, fresh eyes.

**Licenses consulted:** `docs/studies/closing_data/RESULTS_BOARD.md` §1; `.../joint_best/JOINT_BEST_RESULTS.md`;
`.../v17_completion/CEILINGS_N20_FINAL.md`; `.../v17_completion/STATISTICAL_PROTOCOL.md` §7–§8;
`.../v17_completion/stats_n20/RESULTS.md` + `m2_prereg_output.txt`; `docs/studies/pre_freeze_gates/A4_RESULTS.md`;
`docs/results/embedding_eval/rescreen_cat/RESCREEN.md` + its CSVs;
`docs/studies/closing_data/archive/findings/W6_ENCODER_ISOLATION.md`;
`docs/results/closing_data/capacity_matched_stl_cat/README.md`.

## Verdict

**GATE PASS.**

Every verdict verb in the document is bound to a test that the record actually ran. The two claims
this round rewrote — the Conclusão Geral's previously unqualified "outperforms both dedicated
models", and Chapter 5's representation-integrity paragraph — are now *narrower* than their
evidence rather than wider, which is the correct direction. Arizona is never upgraded, at any of the
six sites where the region verdict appears. The never-cite list is at zero. BRACIS containment
holds.

I found one MAJOR claim-scope issue that the round introduced (C-01, a claim narrowed past its own
evidence in a way that misstates the source), three MODERATE, and four MINOR. None is a gate failure:
none is an unlicensed claim, a verb-test mismatch, a C3/C4 violation, or a lost hygiene device.

## Top 3 findings

1. **C-01 (MAJOR)** — Chapter 5's Istanbul external-validity sentence attributes the +8.58 category
   gain to a comparison the chapter's own Table 9 does not support.
2. **C-02 (MODERATE)** — "The equivalence is well powered" is an unsourced power claim in a chapter
   that otherwise binds every statistical word to a named test.
3. **C-03 (MODERATE)** — the Chapter 6 capacity-matched paragraph mixes the joint-best and
   diagnostic-best conventions across one comparison, the exact blur N5 forbids.

---

## Ranked findings

### C-01 · MAJOR · Istanbul external-validity delta points at the wrong comparand

`5_mobiwac.tex:717-719` reads (rendered p. 73):

> "joint model outperforms the dedicated category ceiling by $+8.58$ macro-F1"

with `54.74` and `63.32` on the same line (`:718`) and `75.16`/`75.35` at `:719`. Those are Table
10's cells, so the arithmetic is right: 63.32 − 54.74 = 8.58. **This one is fine.**

The problem is one line up. The *same* +8.58 is claimed at `5_mobiwac.tex:613`:

> "and $+8.58$ at Istanbul"

inside the sentence that opens "the joint model outperforms the dedicated ceiling on / every
dataset, by $+5.33$ to $+9.35$ macro-F1 (smallest at Florida, largest at / Arizona, ...)"
(`:611-613`). I recomputed all six deltas from Table 10's printed cells:

| dataset | Table 10 dedicated | Table 10 joint | delta |
|---|---:|---:|---:|
| Istanbul | 54.74 | 63.32 | **+8.58** |
| AL | 56.82 | 64.51 | **+7.69** |
| AZ | 56.43 | 65.79 | **+9.36** |
| FL | 74.51 | 79.84 | **+5.33** |
| TX | 69.79 | 77.24 | **+7.45** |
| CA | 70.60 | 77.05 | **+6.45** |

The prose says the range is "+5.33 to +9.35". The maximum computable from the printed table is
**+9.36** (Arizona). The source of record gives +9.35 (`JOINT_BEST_RESULTS.md`, "Δcat deploy" column
for AZ), so the *prose* number is the traceable one and the table's rounding produces 9.36. This is
a 0.01 pp rounding disagreement between a table and the prose that reads it, and it is the kind of
thing an examiner with a calculator finds in ninety seconds.

*Licensed phrasing:* the source of record states, verbatim, "**Category — beats at all six**
(+5.33 … +9.35)" (`JOINT_BEST_RESULTS.md`). The prose matches the license exactly. The fix, if any,
is a note that table cells are rounded to 2 dp and deltas are computed at full precision — **or**
nothing at all, and this becomes a known-and-accepted rounding artifact. It is an author ruling,
not mine.

*Severity note:* MAJOR because the document's own honesty discipline is "quote, never compute", and
here a reader computing from the page gets a different digit than the page states.

### C-02 · MODERATE · "well powered" is asserted, not tested

`5_mobiwac.tex:394`:

> "The equivalence is well powered: the paired difference's standard deviation is 0.01 to 0.18 points
> across the datasets, and the intervals pass a margin as small as one point at Alabama and Arizona"

The supporting facts (the sd range, the one-point margin) are real and are the right *kind* of
evidence. But "well powered" is a statistical term of art, and WRITING_LAW §3 requires
"'significant' only with the test named" — the same logic binds "powered". `STATISTICAL_PROTOCOL.md`
§8 and the persona's own attack surface (item 8) specifically warn against "post-hoc power dressed
as design power". What is offered here is an observed-precision argument, which is the honest and
defensible version, but it is labeled with the word that means the other thing.

*Direction:* the sentence already contains its own better formulation. Leading with the precision
facts and dropping the label would say the same thing without the term of art. Author's call.

### C-03 · MODERATE · The capacity-matched comparison mixes two selection conventions

`6_conclusion.tex:100-102`:

> "At Alabama, across three training configurations and all / twenty fitted models, the best of them
> reaches 56.16 macro-F1, against 56.82 for the / dedicated model at its own tuned width and 64.51
> for the joint model."

The round changed 64.54 → 64.51 here, and the in-source comment (`:103-108`) explains why: 64.51 is
the joint-best value that Chapter 5's Table 10 reports, and N5 forbids blurring joint-best with
diagnostic-best. Correct as far as it goes.

But the source of record for the capacity arm states the opposite pairing:

> "Convention note: the joint values above are **diagnostic-best**, the convention in which the
> ceiling deltas are reported in `CEILINGS_N20_FINAL.md`. Chapter 6 quotes the **joint-best** AL
> value (64.51) to match Chapter 5's Table 3; the verdict is identical under either basis, but the
> two conventions must never be mixed inside one comparison"
> — `docs/results/closing_data/capacity_matched_stl_cat/README.md`

So: 56.16 (capacity arm) and 56.82 (ceiling) are scored under the diagnostic-best-family scorer
`score_stl_cat_ceiling.py`; 64.51 is joint-best. The three numbers in that one sentence do not all
share a convention. The source document flags this exact hazard and the chapter does the thing it
warns about — while its own comment cites N5 as the reason for the change. The verdict does not
move (the README says so, and the gap is ~8 points against a ~0.03 pp convention difference), but
the sentence as written is the blur.

*Direction:* name the convention on the joint value, or use the diagnostic-best 64.54 with the
convention named. Either is defensible; leaving all three unlabeled in one comparison is not.

### C-04 · MODERATE · Two derived comparisons in the Chapter 5 conclusion are computed, not quoted

`5_mobiwac.tex:761`:

> "by at least 4 Acc@10 points over the strongest region reference (HMT-GRN, STAN, or ReHDM; the
> first two on our folds, ReHDM under its own protocol) and by at least 33 macro-F1 points over
> POI-RGNN on category"

I recomputed both from Table 10:

- region, joint minus the strongest external per dataset: Istanbul +6.02, AL +4.32, AZ +6.46,
  FL +4.42, TX +5.39, CA +7.17 → **minimum +4.32**, so "at least 4" is true and conservative.
- category, joint minus POI-RGNN: Istanbul +33.20, AL +40.71, AZ +38.15, FL +45.35, TX +44.21,
  CA +45.27 → **minimum +33.20**, so "at least 33" is true and conservative.

Both claims are **sound and correctly conservative**. The finding is process, not fact: N2 says
agents quote rather than compute, and these two floors do not appear in any source-of-truth file I
could find — they are minima over a table. They are reported as inequalities, which is the safest
possible form, and the direction of rounding disfavors the author in both cases. Recording them so
the author can decide whether to add a ledger line.

A third, same class, `5_mobiwac.tex:731-733`: "at California, ten regions out of 8{,}501 contain the
true next region 65.69 / percent of the time, over 500 times better than picking ten at / random."
Recomputed: 65.69 / (10/8501 × 100) = **558.4**, so "over 500" holds with margin.

### C-05 · MINOR · Verb-test binding: audited at every site, all bound

The full inventory (prose only, comments excluded). Every "outperforms" traced to a test that ran:

| Site | Verb + scope | Test backing it | Verdict |
|---|---|---|---|
| `0_main.tex:288-294` | outperforms cat at all six; region four of six; matches TOST two-point at other two | m2 per-fold Wilcoxon Holm m=6 all reject; TOST AL/AZ | **bound** |
| `1_introduction.tex:129-133` | same scope | same | **bound** |
| `2_fundamentals.tex:618-621` | "by paired / superiority tests, outperforms ... everywhere it is tested and on the next region at four of six ... matches ... by non-inferiority testing" | verb explicitly names its test in-sentence | **bound, exemplary** |
| `5_mobiwac.tex:611-622` | cat every dataset +5.33..+9.35; region FL/TX/CA/Istanbul; TOST AL/AZ | Holm-corrected paired t + registered per-fold Wilcoxon, both reported at `:639-645` | **bound** |
| `5_mobiwac.tex:761` | same | same | **bound** |
| `6_conclusion.tex:20-22, 74-77` | same scope, both sites | same | **bound** |

Banned verbs swept across all prose: `beats` 0, `wins` 0 (one occurrence inside an Appendix B
*quotation* of published CoUrb wording at `apx_b_errata.tex:250,262`, which is a quotation of a
source being corrected, correctly framed), `ties` 0, `Pareto` 0 outside the Nash-MTL method
description in `3_cbic.tex:108` where it is the cited technique's own term ("MGDA finds
Pareto-optimal descent directions").

### C-06 · MINOR · Arizona: never upgraded, at six sites

The zero-delta dataset is the standing trap. Every site:

- `5_mobiwac.tex:592-593` — table rows carry `$^{\approx}$`, **not bolded** (the bold marks
  supported improvement; AL and AZ region cells are unbolded). Verified in the rendered PDF p. 71.
- `5_mobiwac.tex:645-648` — "Alabama and Arizona are tested for equivalence (TOST, / $\pm2$~pp) and
  pass ... Arizona ($0.00$; $-0.08$ to $+0.07$). At Arizona, the interval is centered on zero, so we
  report a match, / not a gain."
- `5_mobiwac.tex:621-622`, `:761`, `0_main.tex:292-294`, `1_introduction.tex:131-133`,
  `2_fundamentals.tex:619-621`, `6_conclusion.tex:21-22, 76-77` — all "four of six" + TOST.

The Alabama deficit is stated rather than hidden: "At Alabama, the whole interval lies below zero, a
small but / statistically significant deficit, still well within the two-point margin"
(`:649-650`). That sentence costs the author nothing and buys a great deal of credibility.

### C-07 · MINOR · Time-indexing: intact, and improved this round

CBIC's null and CoUrb's protocol both read as conclusions of their time:

> "Its / conclusions are the conclusions of the time, for the configuration studied here"
> — `3_cbic.tex:20-21`

> "That conclusion held for / the configuration of its time, a place-level input under hard
> sharing"
> — `6_conclusion.tex:39-40`

The round's Nash-pointer correction (`3_cbic.tex:25-27`) is itself a time-indexing fix and is
correct: Chapter 4 *does* use Nash-MTL (`4_courb.tex:115`), so narrowing "the following chapters do
not rely on it" to "Chapter~\ref{ch:mobiwac} does not rely on it" removes a false statement.
Chapter 5's own treatment is consistent — Nash-MTL appears only in the related-work list of
balancers that "rarely improve on a" fixed-weight baseline (`5_mobiwac.tex:183`).

### C-08 · MINOR · Honesty devices: inventory verified present

The mandated keeps, all confirmed in the current build. **Do not edit these away.**

| Device | Location | State |
|---|---|---|
| "We do not predict the exact next place" | `5_mobiwac.tex:226` (rendered p. 62) | present, early, once |
| Hygiene sentence, splitting | `5_mobiwac.tex:365` "so all of a user's windows fall in the same fold and overlap cannot leak" | present |
| Hygiene sentence, region-transition prior | `5_mobiwac.tex:367` "is built per fold from training data only, after an earlier whole-dataset version inflated region accuracy by 13 to 27 points" | present, and it *quantifies the counterfactual* — unusually strong |
| Hygiene sentence, baseline representations | `5_mobiwac.tex:367` "HGI is pre-trained once on the whole dataset, like ours, and CTLE per fold on training users only" | present |
| Selection-bias limitation | `5_mobiwac.tex:736` "every absolute score reported here is optimistic" | present (new this round) |
| Non-cancellation clause | `5_mobiwac.tex:736` "It does not follow that the bias cancels exactly." | present — and the in-source comment at `:740-742` records that this **weakens** a drafted sentence that had claimed exact cancellation. This is the anti-dilution rule running in the correct direction. |
| Service claim bounded | `5_mobiwac.tex:733` "This remains motivation, not a measured service result." | present |
| Trend not a law | `5_mobiwac.tex:624-625` "we read the trend across the points rather than / as a precise law" | present |
| Attribution declared a finding | `5_mobiwac.tex:668` "We report this attribution as a finding, not a hypothesis." | present |
| CoUrb contribution note | `4_courb.tex:13` preface, Tarik first author / Vitor second and presenter | present |
| Negative result written with care | `6_conclusion.tex:32-40`; `3_cbic.tex:20-23` | present — the CBIC null gets a full paragraph in the conclusion, framed as the arc's foundation, not rushed past |

### C-09 · MINOR · C1/C3/C4 sweeps

- **C1 whitelist conformance:** every Chapter 5 claim I traced appears in the CAN-say set of the
  board/joint-best documents. The must-NOT-say phrasings ("beats region everywhere", "ties",
  "Pareto") are at zero.
- **C3 never-cite:** grepped all prose+tables for the eight banned values. `34.46`, `38.96`,
  `62.37`, `66.06`, `65.68`, `75.87`, `-5.22` — **all clean**. One apparent hit on `54.65` at
  `5_mobiwac.tex:447` is **not** the banned ReHDM v2 value: it is Istanbul's check-in-level macro-F1
  in Table 9, a coincidental digit match. Verified against `RESULTS_BOARD.md` (ReHDM v2 row is
  66.06/54.65/65.68 as a triple; the Table 9 cell is a different quantity on a different axis).
  No action.
- **C4 BRACIS containment:** BRACIS appears in prose only at `apx_a_contributions.tex:111-116` (as a
  submitted-and-rejected other contribution) and in the abbreviations list (`0_main.tex:346`).
  Zero occurrences in Chapters 1–6. The superseded region-cost claim does not appear anywhere. **Held.**

---

## The four grounds of the representation-integrity paragraph

Audited separately because the round rewrote it and it is the chapter's most attackable passage.
Full text at `5_mobiwac.tex:367`. Each ground checked against its source:

| Ground | Claim | Source | Verdict |
|---|---|---|---|
| First (label-free objective) | "its training objective is label-free ... That bounds the training signal and not the inputs, since each visit's own category enters as a node feature" | `RESCREEN.md:56` confirms category is an input node feature | **honest** — the ground *states its own limit* in the same sentence and forwards to ground four |
| Second (transductive channel) | "moves both tasks by at most a third of a point (region $-0.33$ to $+0.01$; category $0.00$ to $+0.29$, at Alabama, Arizona, and Florida)" and "covers the visits whose places appear in training (67 to 87 percent)" | `A4_RESULTS.md`: reg AL −0.33 / AZ +0.01 / FL −0.12; cat AL +0.29 / AZ +0.27 / FL +0.00; in-coverage AL 66.8% / AZ 71.9% / FL 86.9% | **exact.** Every figure traces. The coverage caveat and the unseen-places residual are both stated |
| Third (region-transition prior) | "built per fold from training data only, after an earlier whole-dataset version inflated region accuracy by 13 to 27 points. Our joint and dedicated models do not use this prior" | consistent with the A4 record and the baselines section | **honest** — discloses a past leak and its magnitude |
| Fourth (forward-edge channel) | control ceiling ~0.41; FL "$0.4090$ ... $0.4074$"; residual variant "$0.4197$ and $0.4182$"; attention encoder "$0.4976$ and $0.4863$, above the ceiling, and was disqualified" | `leak_sniff_fl.csv`: gcn_ctrl 0.4089797540123382 / 0.40744232906432776; gat 0.49761650037538024 / 0.48631035868799294. `leak_sniff_resln_fl.csv`: resln 0.4196859144977155 / 0.41815720719390814 | **exact to four decimals.** Nothing recomputed |

The three stated limits ("the probe is linear, it was run at Florida alone at one random
initialization over five user-grouped folds, and it was run on those ancestor builds of the
representation rather than on the one that produced the results reported here") match the audit's
own residuals in `RESCREEN.md:94-95` and the source comment at `5_mobiwac.tex:382-385`.

**The strongest single sentence in the paragraph** is the self-undercut:

> "The same record shows why the linear form is a screen and not a proof, since one encoder that
> passed it leaked under a downstream sequence model."

That is the R-GCN case from `RESCREEN.md` (passed per-step at 0.414, leaked at 0.754 under the GRU),
reported against the author's own interest. The paragraph as rewritten claims **less** than the
evidence would allow and names what it cannot reach. This is the round's best work.

---

## NEW-CLAIM list for author sign-off

Claims that assert what the arc "shows" and go beyond a single chapter (rule C2). None is
unlicensed; all need the author's word because the frame owns them:

1. `6_conclusion.tex:89-90` — "The representation, together with the / sharing topology built on it,
   is what the answer depends on." The dissertation-level synthesis. Supported by the arc but
   asserted by no chapter. **The abstract hedges the same claim more carefully** ("depends on the
   input representation and the sharing topology built on it", `0_main.tex:295-296`, after the
   round's "determines whether"→"depends on" change per the comment at `:299-300`). Consistent.
2. `6_conclusion.tex:95-96` — "it comes from a / stronger shared trunk." Licensed by W6
   (`W6_ENCODER_ISOLATION.md` verdict: "the joint CATEGORY win is the shared TRUNK ... NOT
   region→category transfer"), scoped to three datasets in the same sentence. Sign-off is for the
   frame-level generalization only.
3. `2_fundamentals.tex:371-372` — "a fixed-weight baseline is a serious competitor, and a / balancer
   earns its place only by outperforming it." A methodological position, not a result. Supported by
   the three cited studies. Frame claim; author owns it.
4. `6_conclusion.tex:96-119` — the entire capacity-matched paragraph is post-submission frame-level
   analysis (the source README marks it "POST-SUBMISSION ... never enter Chapter 5"). Correctly
   placed in Chapter 6 and correctly declared there ("run after the / Chapter~\ref{ch:mobiwac}
   manuscript was submitted and reported here as a frame-level analysis", `:96-97`).

---

## Open questions only the author can answer

1. C-01: accept the 9.35/9.36 rounding disagreement as a known artifact, or add a rounding note to
   Table 10?
2. C-03: which convention should the Chapter 6 capacity sentence carry — joint-best 64.51 with the
   convention named, or diagnostic-best 64.54 with the convention named?
3. C-04: do the two "at least" floors need ledger lines, given they are minima over a printed table
   rather than values in a source file?

## Out-of-scope handoffs

- Persona 06: the 9.36-vs-9.35 rounding; the "at least 4" / "at least 33" floors.
- Persona 09: the "well powered" sentence (C-02) is also a statistics finding.
- Persona 05: `russwurm...` renders as `(??)`; the Sluice Networks title.
