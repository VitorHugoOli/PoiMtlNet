# Banca Simulator (persona 12) — Report on Complete-v1 Defense Build

> Reviewer: simulated UFV/PPGCC defense-committee member (professor doutor, ML / urban computing).
> Scope: FULL defense build (all chapters, front matter, appendices). Complete-v1 dry run.
> Build under review: `articles/dissertacao/src/main_defense.pdf` (87 pp) + `main_final.pdf` (83 pp).
> Read-only. Findings only; edits are a separate, author-approved step.
> STATUS: IN PROGRESS — written incrementally so a restart cannot lose work.

---

## 0 · Reading log (what I have read this session)

- [x] `reviewers/12_banca_simulator.md` (my persona)
- [x] `reviewers/README.md` (Common protocol)
- [x] `articles/dissertacao/CLAUDE.md`
- [x] `articles/dissertacao/NORTH_STAR.md` §1–§4, §6
- [x] `articles/dissertacao/UFV_COMPLIANCE.md` §3
- [x] The dissertation build: `main_defense.pdf` (87 pp) — all six chapters + 3 appendices + front matter, read via source `.tex` and PDF text layer
- [x] Sources of truth for spot-checks: `docs/studies/closing_data/RESULTS_BOARD.md`, `storyline/audit/capacity_baseline_experiment.md`, the CBIC/CoUrb reproduced tables, `docs/studies/pre_freeze_gates/A4_RESULTS.md` (leak audit numbers cited in Ch.5)

---

## 1 · Annotation list (pre-read; quote + page + what bothers me)

### A-1 [MAJOR] Same quantity, two values across chapters: AL joint next-category macro-F1 is 64.51 (Ch.5 table) vs 64.54 (Ch.6 §6.2)
- Ch.5 `tab:mobiwac:results` (p.66): Alabama, Next-category, Joint (ours) = **64.51** ±0.09.
- Ch.6 §6.2 (p.73): "...against 56.82 for the dedicated model at its own tuned width and **64.54** for the joint model."
- The dedicated value (56.82) matches the table exactly, which makes the 64.51/64.54 mismatch conspicuous.
- Source trace: `docs/studies/closing_data/RESULTS_BOARD.md` L27 gives AL joint = **64.54**; `storyline/audit/capacity_baseline_experiment.md` L92/L113 gives joint v17 = **64.54**. So Ch.6 is faithful to the board; Ch.5 reproduces the submitted manuscript's 64.51 (per its errata policy, "no correction applied"). Both are internally sourced; the defect is that the text never tells the reader the same cell reads differently in the two chapters, or why (a later re-run).
- This is the build's single most concrete banca gotcha. Not a BLOCKER (0.03 pp, no verdict changes), but a flip-trigger by this persona's own red-flag list ("numbers that differ between chapters or between text and tables"). Trivial honest fix.

### A-2 [BLOCKER for the defense build] Dissertation title is still a bracket placeholder on the cover, Resumo, and Abstract
- `0_main.tex`: `\titulo{...[TITLE --- open decision NORTH\_STAR §5.8]...}`, and the Resumo/Abstract headers both render `\textbf{[TITLE --- open decision NORTH\_STAR §5.8]}`.
- A banca receives a defense PDF whose folha de rosto, Resumo, and Abstract show a literal placeholder instead of a title. Working title exists (author 2026-07-23, chapters/1_introduction.tex header) but was not wired in. Must be resolved before the text ships (Art. 22, >=20 days before defense).

### A-3 [MAJOR] Front-matter placeholders in the defense build: approval sheet, banca members, defense date
- `0_main.tex`: approval-sheet placeholder block; `\membrobancaA{[Banca member 1 --- pending]}`, `\membrobancaB{[... pending]}`, `\databanca{[defense date --- pending]}`.
- Expected at this stage (banca not yet formed), but they are visible in the built PDF and are prerequisites for the >=20-day submission. Flag, not a science defect.

### A-4 [MAJOR] Editorial scaffolding leaks into the numbered body: the CBIC dataset sentence renders "VERIFY: recompute per ERRATA.md" on page 35
- Physical/numbered page 35, §3.4.1: rendered text reads "This subset comprises a total of [N_users; VERIFY: recompute per ERRATA.md] users, [N_poi; VERIFY: recompute per ERRATA.md] unique Points-of-Interest (POIs), and [N_checkins; VERIFY: r..." (verified from the PDF text layer).
- This is the errata "Pending" row (Appendix B, `tab:apx:cbic-errata`) surfacing as visible incompleteness plus internal review instructions ("VERIFY: recompute per ERRATA.md") in the middle of a reproduced article. A banca reading page 35 sees the scaffolding, not just a gap. By this persona's red-flag list this is a flip-trigger. The values must be recomputed and inserted (author-approved script per NORTH_STAR §4 Ch.3), or the sentence recast to avoid rendering internal tokens, before the text ships.
- Honesty note: the pending recompute itself is defensible (values not invented). The defect is that the raw internal marker prints in the defense PDF.

### A-5 [MAJOR] The frame's verb-binding law names a test the resolution chapter does not use, and claims a scope the reproduced chapters do not honor
Two-part defect in the dissertation's own honesty device (the "verbs bound to tests" law).
- (a) **Test-name mismatch, Ch.2 vs Ch.5.** Ch.2 §2.4 (p.22): "The paired Wilcoxon signed-rank test ... is the test that licenses the verb ``outperforms''." But Ch.5 §5.5.3 (p.64): "superiority is tested with a paired $t$ on the per-seed means," and §5.6 (p.68): "each gain is significant after a Holm correction ... (paired $t$, corrected $p<0.001$)." Ch.5 never uses Wilcoxon; Ch.2 never mentions the paired t. The frame chapter describes a superiority test the resolution chapter does not use.
- (b) **Scope overclaim.** Ch.2 §2.4: "The verbs and the tests are bound together **throughout the document**: ``outperforms'' follows only from a paired superiority test." This is contradicted by the reproduced article chapters. Ch.3 (CBIC) §3.4.2 (p.36): "both our MTL and Single models **significantly outperform** HMRM across all POI categories," with NO paired test anywhere in the chapter, and the same chapter's own §3.1/§3.4.2 concede the MTL-vs-single differences "fall within standard deviations, suggesting ... statistical performance was largely comparable." Ch.4 (CoUrb) uses "outperform" 10 times (e.g. "outperform the original MTLNet in all 21 category-state combinations") with no significance test in the chapter at all.
- Why it matters: the fundamentals chapter's most quotable rigor sentence promises a document-wide standard the document does not meet. A stats-minded examiner reads Ch.2, then finds "significantly outperform" untested in Ch.3. The honest scoping is available and true: the binding law governs the frame's own claims and Ch.5's consolidated analysis; the reproduced articles are of-their-time texts whose internal verb usage predates the law (the time-capsule device already says as much for conclusions).
- Suggested direction (author decides): (a) in Ch.2, either name the paired t used in Ch.5 or generalize to "a paired significance test" so the frame and Ch.5 agree; (b) scope "throughout the document" to the frame + Ch.5, or add one clause acknowledging the reproduced chapters use the verb as published. Do NOT retro-fit tests onto the reproduced articles.
- Severity: MAJOR, not BLOCKER. No result changes; the Ch.5 claims (the ones the dissertation stakes itself on) ARE properly tested. The defect is a frame overclaim about rigor + a test-name inconsistency, both cheaply fixed and both squarely on this persona's flip-list ("numbers/claims that differ between chapters").

### A-6 [MINOR] Terminology bridge missing at the chapter boundary: "Next-POI Prediction" (Ch.3/Ch.4 titles and sections) vs the frame's "next category" + "next place is never predicted"
- Ch.2 §2.1 (p.17) fixes canonical names and states "It does not predict the exact next place." Ch.3 is then titled "...Category Classification and **Next-POI Prediction**" (p.25) and Ch.4 §4.4.3 is "**Next-POI Prediction**" (p.53). A reader meeting those titles right after the scope statement can momentarily read them as next-place prediction.
- Mitigation already present: both reproduced chapters define the term internally (Ch.3 L35/L146: "Predicting the category of the next POI"), and Ch.1 §1.1 explains the task-pair evolution. So it is recoverable, not wrong.
- Gap: neither the Ch.3 nor the Ch.4 preface tells the reader "what this article calls next-POI prediction is the next-category task of Chapter 2." One clause in each preface (or a footnote at the first chapter-title occurrence) closes it. This is the concrete example a banca would raise under the coletanea terminology-consistency probe (Q22).
- Severity MINOR: the internal definitions carry it; the fix is one sentence per preface.

### A-7 [MINOR] The capacity-matched baseline (Ch.6 §6.2) is a post-submission frame-level analysis with an incomplete run
- Ch.6 §6.2 (p.73): "a capacity-matched dedicated baseline, run after the Chapter 5 manuscript was submitted ... A partial California run, fifteen of twenty repetitions at the time of writing, shows the same direction."
- This control is load-bearing: it closes the "the joint model just has more parameters" objection (a natural banca follow-up). Introducing it only in the Conclusion, and with one dataset still at 15/20 reps, is defensible but leaves an opening: an examiner can ask why the strongest counter-explanation to the headline claim is relegated to the frame and left partially run.
- Not a defect of honesty (the partial status is disclosed). The risk is presentational: complete the California run before the defense, or state plainly that the Alabama result (full n=20) is the primary evidence and California is confirmatory. The candidate must own the 15/20 in the room.

### A-8 [MINOR] No privacy or ethics treatment of individual mobility data anywhere in the document
- Searched all chapters: no "privacy", "ethic", "sensitive", "consent", or "anonymiz*" in rendered prose. The work models individual users' movement (check-in traces, user-disjoint splits keyed on `userid`), which is personal mobility data.
- A banca almost always asks this (bank question Q18). The text pre-answers most attacks but not this one. A short paragraph (public datasets, aggregate/region-level targets, no re-identification attempted, user-disjoint evaluation) in Ch.6 limitations or a one-line note would pre-empt it at low cost.
- Severity MINOR: not a methodological flaw; a missing defensive paragraph on a predictable question.

### A-9 [NIT] The 93% predictability figure appears in Ch.1 §1.1 as motivation without the scope caveat that Ch.2 §2.4 supplies
- Ch.1 §1.1 (p.12): "estimated the potential predictability of an individual's next location at about 93 percent."
- Ch.2 §2.4 (p.23) handles it correctly: explicitly NOT a ceiling on seven-class category macro-F1 or region ranking. Ch.1 uses it loosely as "regularity," not as a bound, so it is not an overclaim.
- A reader who knows Song et al. 2010 will note the resolution-dependence. A forward-reference from Ch.1 to the Ch.2 scoping (or a half-clause) removes the momentary friction. The correct handling already exists downstream; this is polish only.

### Out-of-scope handoffs (one line each, not my scope)
- Font-cache regenerations in `missfont.log` (ts1-qtmr, t1xtt, txsys) resolved at build time; the PDF built. Whether `\texttt` in the appendices renders in the Times family or a fallback -> persona 18 (visual) / 13 (format).
- 32 Overfull \hbox warnings (one at 29.76pt, lines 214--232; rest minor) and 11 Underfull -> persona 18 (visual-presentation) / 02 (line editor).
- Whether B4/B2 Qualis clears the Art. 21 internal-resolution bar (UFV_COMPLIANCE open item) -> persona 13 (UFV compliance).

---

## 2 · Dimension scores (10 × 1–5)

| # | Dimension | Score | Evidence line |
|---|-----------|:---:|---------------|
| 1 | Problem clarity & delimitation | **5** | Bold inline research question (Ch.1 §1.2); three tasks kept formally distinct (Ch.2 §2.1); "the exact next place is not predicted anywhere in this work" stated in Ch.1 §1.4 AND Ch.2 AND Ch.6. |
| 2 | Command of the state of the art | **4** | Ch.2 is critical, not a catalog: places the work against next-place lineage, CTLE as the contextual hinge, the scalarization-skeptic MTL literature (Kurin, Xin, RLW). Current (Massive-STEPS 2025, ReHDM 2025). Loses one point: the frame's MTL-in-mobility coverage is thinner than the per-task depth. |
| 3 | Methodological coherence | **5** | Every choice justified against alternatives: cheapest-controlled-test logic for the CoUrb representation move (Ch.1 §1.2); cross-attention over expert-gating (Ch.2 §2.3, Ch.5); three-role baseline design (Ch.5 §5.5.4); TOST margin justified by service granularity. No "the advisor suggested it." |
| 4 | Rigor & honesty of results | **4** | Pre-registered superiority/non-inferiority assignment, Holm, user-disjoint CV, an honest leak audit that discloses a caught leak (region prior, +13–27 pp) and a residual coverage gap. Docked one point for A-1 (64.51 vs 64.54 across chapters) and A-5 (frame names Wilcoxon; Ch.5 uses paired t; reproduced chapters say "significantly outperform" untested). |
| 5 | Contribution | **5** | A nameable dissertation-level delta beyond any single paper: the conditional answer (representation + sharing topology decide whether MTL helps these tasks), consolidated across a published null, its diagnosis, and its resolution. Master's standard cleared comfortably. |
| 6 | Recognition of limitations | **5** | Six concrete limitations volunteered (Ch.6 §6.3), each with a consequence and a 1:1 future-work item — including the self-incriminating task-pair confound ("no single controlled ablation separates the representation-and-topology change from the task-pair change"). This is rare and impressive. |
| 7 | Candidate ownership | **4** | Co-authorship handled explicitly (CoUrb preface + Ch.1 §1.5: 2nd author, presenter, author of the MTLnet baseline). Every number traceable to a source of truth. Docked one point: the capacity baseline is 15/20 at California (A-7) and the candidate must own that live; A-1 shows one cross-chapter value not yet reconciled. |
| 8 | Text quality | **4** | Organization and register are strong and consistent; the arc reads as one document. Docked for the visible scaffolding on p.35 ("VERIFY: recompute per ERRATA.md", A-4) and the title placeholder (A-2) — exactly the sloppiness signals that trigger examiner hypercriticism, though both are pre-defense-build artifacts. |
| 9 | Coletânea unity (fio condutor) | **5** | The correction-trail arc IS the thread; the Introdução Geral states it, the time-capsule prefaces enforce it, and the Conclusão Geral claims something no single paper claims (the consolidated conditional answer + two controls). Inter-paper differences are confronted, not hidden (Ch.6 §6.2 reads the null and the positive together to bound the claim). |
| 10 | Defense-readiness of the text | **4** | Pre-answers the leak kill-shot, the fair-baseline question, the parameter-count objection, the external-validity question (Istanbul), and the fixed-weight-baseline question. Gaps: no privacy paragraph (A-8), and the frame verb-binding overclaim (A-5) hands a stats examiner one clean opening. |

**Aggregate:** 45/50. Consistent with the modal outcome (aprovado com correcoes menores): a strong, honest, coherent dissertation with a handful of concrete, cheap-to-fix text/consistency defects and two predictable-but-unanswered probes.

---

## 3 · Arguição transcript (12 questions)

> Posed as a UFV/PPGCC examiner would (PT-BR), each tied to a specific annotation. For each:
> what it **tests**, what a **strong answer** contains, and what the **current text supports**
> (if the text already answers it, that is a pass and the candidate should quote it). This
> doubles as the author's defense-preparation sheet.

### Contribution & positioning

**Q1 (bank #1).** "Em uma frase: qual e a contribuicao original desta dissertacao, da dissertacao, nao de cada artigo?"
- *Tests:* whether a dissertation-level delta exists above the three papers.
- *Strong answer:* the conditional answer, the representation together with the sharing topology built on it decides whether MTL helps these POI tasks, established across a published null, its diagnosis, and its resolution; no single paper states this.
- *Text supports:* YES. Ch.1 §1.6 (Theoretical contribution) and Ch.6 §6.2 both state it verbatim. **Pass** — quote Ch.6 §6.2's "The representation, together with the sharing topology built on it, is what the answer depends on."

**Q2 (bank #2).** "O que seu trabalho mostra que um MTL de mobilidade ja existente ainda nao mostrava?"
- *Tests:* positioning against prior MTL-in-mobility.
- *Strong answer:* prior mobility MTL uses region/category as auxiliary signals toward next place (MCARNN, CSLSL, HMT-GRN, CatDM); none predicts next region as a co-equal end target alongside next category, and none isolates the representation as the deciding variable.
- *Text supports:* YES. Ch.2 §2.3 ("no multi-task model among them predicts the next region as a co-equal end target") and Ch.5 §5.5.4. **Pass.**

### Method justification

**Q3 (bank #4).** "Por que MTL? Mediu a comparacao com dois modelos separados com o mesmo orcamento de tuning dos dois lados?"
- *Tests:* the fair-baseline standard and the parameter-count confound.
- *Strong answer:* the dedicated single-task model is the operative ceiling, tuned at its own optimum (Ch.5); and a capacity-matched dedicated category model widened to the joint budget (4.2M vs 0.6M at Alabama) does NOT recover the gain, so it is not parameters, it is the second task's training signal.
- *Text supports:* PARTIALLY. Ch.6 §6.2 carries the capacity-matched baseline, but it is a post-submission frame analysis with California at 15/20 (A-7). **The candidate must own the partial run in the room** and lead with the full-n Alabama result (56.16 capacity-matched vs 64.54 joint).

**Q4 (bank #6).** "Por que um balanceador de gradientes e nao uma soma ponderada estatica? A soma estatica nao venceu de fato no seu caso final?"
- *Tests:* the dangerous question — the fixed-weight baseline in fact sufficed in Ch.5.
- *Strong answer:* concede it plainly. Ch.2 §2.3 already takes the cautious position (balancers often do not beat a tuned fixed weight); in the Ch.5 configuration the two tasks' gradients are near-orthogonal, so a balancer had little to correct and a tuned fixed weighting sufficed. CBIC's Nash-MTL preference is time-indexed and weakened by a later optimizer-implementation finding (Ch.3 preface).
- *Text supports:* YES, and honestly. Ch.6 §6.2 ("why gradient-balancing optimizers had little to correct ... and why a tuned fixed weighting sufficed"). **Pass** — this is a strength, not a trap, because the text concedes it first.

**Q5 (bank #7).** "Justifique a representacao. Que alternativas considerou e por que as descartou?"
- *Tests:* command of the representation design space (the spine of the thesis).
- *Strong answer:* one-hot to distributed to graph-infomax place embeddings (DGI, HGI) to the check-in level; CTLE is the closest contextual prior and is run as a control; a feature-concatenation control and a place-level-embedding control isolate what the per-visit representation adds.
- *Text supports:* YES. Ch.2 §2.2 (the lineage) + Ch.5 §5.5.4 (CTLE and concat controls). **Pass.**

### Experimental rigor

**Q6 (bank #8, the classic kill-shot).** "Como garantiu que nao ha vazamento? Em particular, uma representacao pre-treinada no dataset inteiro nao passa informacao do teste para o treino?"
- *Tests:* leakage from the whole-dataset transductive representation.
- *Strong answer:* three grounds (Ch.5 §5.5.2): label-free training objective; a per-fold rebuild moves both tasks by at most a third of a point at AL/AZ/FL; and the one component that could pass information between visits (the region-transition prior) is built per fold, after an earlier whole-dataset version was caught inflating region accuracy by 13–27 points. Residual gap disclosed: visits to places unseen in training (the 13–33% out-of-coverage tail).
- *Text supports:* YES, exemplary. **Pass** — this is the strongest single passage for the defense; the candidate should walk the committee through it. Own the residual coverage gap as a stated limitation, not a hole.

**Q7 (bank #10).** "Os ganhos sao significativos? Quantas seeds, quantos folds, qual teste, e a variancia entre folds e maior que o ganho?"
- *Tests:* the statistical protocol and whether the frame and the chapter agree.
- *Strong answer:* n=20 (4 seeds x 5 folds), paired t on per-seed means (n=4), Holm across six datasets, category p<0.001 everywhere; region superiority at 4/6 and TOST non-inferiority (±2 pp) at AL/AZ with 90% CIs inside two points.
- *Text supports:* YES for Ch.5 — BUT **the candidate will be caught by A-5**: Ch.2 §2.4 tells the reader "the paired Wilcoxon signed-rank test ... licenses the verb outperforms," while Ch.5 uses a paired t. Expect: "O senhor disse Wilcoxon no Capitulo 2 e t pareado no Capitulo 5. Qual foi usado, e por que a discrepancia?" Fix the frame before the defense; if asked live, concede it is a frame editing slip and the paired t on per-seed means is the test actually run.

**Q8 (bank #11).** "Os baselines foram tunados com o mesmo esforco que o seu metodo?"
- *Tests:* baseline fairness.
- *Strong answer:* per-task SOTA re-implemented on the same folds/initialization (POI-RGNN, HMT-GRN, STAN); dedicated models tuned at their own optimum (best-vs-best); ReHDM reported under its own published protocol (disclosed as such).
- *Text supports:* YES. Ch.5 §5.5.4. **Pass** — note the honest disclosure that HMT-GRN/STAN are adapted to the region target and stripped of their next-place machinery (a fairness concession, not a hidden edge).

### Coletanea-specific (>=4 required)

**Q9 (bank #19).** "Qual e o fio condutor? Convenca-me de que isto e uma dissertacao e nao artigos grampeados."
- *Tests:* coletanea unity.
- *Strong answer:* the correction trail — null, diagnosis, resolution — with each chapter revising the previous one's conclusion, stated in the Introducao Geral and enforced by the time-capsule prefaces.
- *Text supports:* YES, strongly. Ch.1 §1.2 + the three prefaces + Ch.6 §6.2. **Pass** — this is the dissertation's best feature.

**Q10 (bank #20).** "O que o artigo 3 corrige do artigo 1? Os numeros divergem, qual versao devo acreditar, e a Conclusao Geral discute essa evolucao?"
- *Tests:* whether inter-paper contradictions are confronted.
- *Strong answer:* Ch.5 does not contradict Ch.3; read together they bound the claim (MTL is neither a free gain nor a dead end). CBIC's null holds for its configuration (place-level input, hard sharing); Ch.5 changes the representation and topology. The Conclusao Geral §6.2 states exactly this.
- *Text supports:* YES. **Pass.** Caveat for the candidate: be ready for the sharper form — "the Alabama joint category number is 64.51 in Chapter 5 and 64.54 in Chapter 6 (A-1); which is right?" Answer: 64.54 is the audited board value from a later re-run; 64.51 is the submitted-manuscript value reproduced verbatim per the errata policy. Reconcile the two in the text (one footnote) before the defense so the question never arises.

**Q11 (bank #21).** "No artigo em coautoria (CoUrb), o que exatamente foi contribuicao sua?"
- *Tests:* individual contribution in the 2nd-author chapter (a real concern for this format).
- *Strong answer:* Vitor is the author of the MTLnet baseline the study builds on (his 1st-author CBIC work), the second author of the paper, and presented it at the workshop; the CoUrb study holds his architecture fixed and varies only the input.
- *Text supports:* YES. Ch.4 preface + Ch.1 §1.5 state all three roles. **Pass** — the disclosure is explicit and correctly placed.

**Q12 (bank #24).** "Que tentativas falhas ficaram de fora? A iteracao do BRACIS, por exemplo, e a afirmacao de custo de regiao que ela fazia."
- *Tests:* whether negative/failed attempts are documented, not buried.
- *Strong answer:* the BRACIS submission (rejected, unpublished) is in Appendix A as an earlier iteration whose central region-cost claim was corrected by MobiWac (traced to an fp16 numerical-precision artifact + older protocol); no result of it is cited as evidence.
- *Text supports:* YES. Appendix A states it plainly, including that the reviewers' leakage objection is answered by the Ch.5 audit. **Pass** — the containment is exactly right; the candidate should present it as evidence of self-correction, not hide it.

---

## 4 · Verdict + corrections list

### Verdict: **APROVADO COM CORREÇÕES MENORES**

This is a defensible master's dissertation and, on the text, a good one. It has a sharp research
question, a genuine dissertation-level contribution that no single paper carries, an honest
correction-trail arc that makes three differently-shaped papers read as one investigation, and a
rigor-and-honesty posture (pre-registered tests, a leak audit that discloses a caught leak,
volunteered limitations including a self-incriminating confound) that is above the master's bar. I
want to pass this candidate and the text supports it.

The verdict is "com correções menores" and not "sem ressalvas" for two reasons, both cheap to fix
and neither touching a result: (1) two visible incompleteness artifacts in the defense build (the
title placeholder on the folha de rosto/Resumo/Abstract, and the "VERIFY: recompute" scaffolding
rendered on body page 35), which are exactly the sloppiness signals that flip an examiner into
hypercritical mode; and (2) one real cross-chapter consistency defect (the same Alabama number
reads 64.51 in Ch.5 and 64.54 in Ch.6) plus one frame overclaim about the statistical protocol
(Ch.2 names Wilcoxon; Ch.5 uses a paired t; the reproduced chapters say "significantly outperform"
with no test). None is invalidating; all are the kind of thing the banca files as obrigatória and
expects fixed in the corrected version.

No BLOCKER endangers the contribution or the method. A-2 (title placeholder) is marked BLOCKER only
in the sense that the text cannot be shipped to the banca with a bracket where the title goes; it
is a wiring step, not a scientific defect.

### Correções OBRIGATÓRIAS (blockers/majors the banca would file)

1. **[A-2] Wire the dissertation title** into `\titulo`, the Resumo header, and the Abstract header.
   The defense PDF currently shows `[TITLE — OPEN DECISION NORTH_STAR §5.8]` on the folha de rosto,
   the Resumo, and the Abstract (verified in the PDF text layer). A working title already exists
   (chapters/1_introduction.tex header). Must be final before the ≥20-day submission (Art. 22).
2. **[A-4] Remove the editorial scaffolding from body page 35.** §3.4.1 renders
   "[N_users; VERIFY: recompute per ERRATA.md] users, [N_poi; VERIFY: ...] ... POIs, [N_checkins;
   VERIFY: r..." Either insert the recomputed values (author-approved script per NORTH_STAR §4
   Ch.3) or recast the sentence so no internal marker prints. Appendix B already flags this row as
   "Pending"; it must not print raw in the reproduced article.
3. **[A-1] Reconcile the Alabama joint next-category value across chapters.** Ch.5 table = 64.51;
   Ch.6 §6.2 = 64.54. Both are internally sourced (Ch.5 = submitted manuscript; Ch.6 = audited
   board). Add one footnote at the Ch.6 use (or the Ch.5 cell) noting the two values and why they
   differ (a later re-run), so a committee does not read it as a careless disagreement.
4. **[A-5] Fix the verb-binding law in Ch.2 §2.4.** (a) Either name the paired t used in Ch.5 or
   generalize to "a paired significance test" so the frame and Ch.5 agree; (b) scope "throughout
   the document" to the frame + Ch.5, or add a clause acknowledging the reproduced articles use
   verdict verbs as published. Do NOT retro-fit significance tests onto the reproduced CBIC/CoUrb
   texts (that would violate the reproduction fidelity rule).
5. **[A-3] Front-matter completion** (approval sheet, banca members, defense date) once the banca is
   formed — a submission prerequisite, expected open now.

### Sugestões (would strengthen; not blockers)

6. **[A-8] Add a short privacy/ethics note** (public datasets, region-level aggregate targets, no
   re-identification, user-disjoint evaluation) in Ch.6 limitations. Pre-empts a near-certain banca
   question (Q18) at one paragraph's cost.
7. **[A-7] Complete the California capacity-matched run** (15/20 → 20/20) before the defense, or
   state in §6.2 that Alabama (full n=20) is the primary evidence and California is confirmatory.
   The candidate should be ready to own the partial run aloud.
8. **[A-6] Add a one-clause terminology bridge** in the Ch.3 and Ch.4 prefaces: "what this article
   calls next-POI prediction is the next-category task defined in Chapter 2." Closes the momentary
   next-place/next-category ambiguity at the chapter boundary.
9. **[A-9] Forward-reference the 93% predictability scope** from Ch.1 §1.1 to the Ch.2 §2.4 caveat
   (or half-clause it in Ch.1), so the figure never reads as a ceiling on category/region.

### Defense-preparation priorities (from the arguição, §3)

- **Own the numbers you did not fully run:** the California 15/20 capacity baseline (Q3, A-7). Lead
  with full-n Alabama.
- **Pre-empt the Wilcoxon-vs-t question (Q7):** fix the frame; if asked, concede a frame edit slip
  and state the paired t on per-seed means is the test actually run.
- **Walk the committee through the leak audit (Q6):** it is the single strongest passage; present
  the caught region-prior leak (+13–27 pp) as evidence of discipline, and own the coverage-tail gap.
- **Present BRACIS as self-correction (Q12), not as a skeleton in the closet.**

---

## 5 · What impressed me (do not edit away)

1. **The honest arc as the contribution.** A published negative result, its diagnosis, and its
   resolution, narrated as a correction trail where each chapter states which earlier conclusion it
   supersedes (Ch.1 §1.2, the three time-capsule prefaces, Ch.6 §6.2). This is the fio condutor most
   coletâneas lack, and it is executed, not merely asserted. Do not let a trimming pass flatten it
   into three paper recaps.
2. **The leak audit (Ch.5 §5.5.2).** Three grounds, a disclosed caught leak (the whole-dataset
   region prior that inflated accuracy by 13–27 points, now built per fold), and an honestly stated
   residual coverage gap. This pre-answers the classic ML-banca kill-shot. It is the best-defended
   passage in the document.
3. **Cost honesty (Ch.5, "operational rather than arithmetic").** The text states plainly that the
   joint model is larger (4.2M vs 1.1M combined at Alabama) and costs more compute, and defines the
   benefit as operational (one artifact, one forward pass). No compute-saving overclaim. The Ch.1
   §1.1 guard against promising lower cost held all the way through.
4. **Volunteered limitations with consequences (Ch.6 §6.3).** Six concrete limits, each tied 1:1 to
   a future-work item — including the task-pair confound, which the candidate could have hidden and
   instead names as "no single controlled ablation separates the representation-and-topology change
   from the task-pair change." Volunteering the weakness that most threatens the headline is exactly
   what distinguishes a candidate who knows their work's limits.
5. **The capacity-matched baseline** closing the parameter-count explanation (Ch.6 §6.2). Even
   partial, running it at all is the mark of someone who anticipated the obvious counter-explanation
   and tested it rather than waving it away.
6. **Co-authorship handled cleanly** (Ch.4 preface + Ch.1 §1.5): 2nd author, presenter, author of
   the baseline the study builds on — stated where the reader meets the chapter, not buried.
7. **BRACIS containment (Appendix A).** A rejected iteration presented as documented self-correction,
   with its corrected claim traced to a numerical-precision artifact and no result cited as evidence.

---

_End of report. Read-only review; no dissertation files modified. Final message to the author is
the verdict above._
