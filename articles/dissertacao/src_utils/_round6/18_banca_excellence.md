# 17_banca_excellence.md — personas 12 (banca simulator) and 17 (excellence assessor), round 6

**Written 2026-07-28.** Read-only pass. Two personas in one report because they share one reading of
one artifact: the **defense build** `src/build/main.pdf`, **108 pp**, snapshot SHA1
`c8b2b1e5b6a8…`, taken at 17:35 and pinned to `/tmp` before any further build could move it.
Companions measured: `main_final.pdf` **105 pp** (`81af41ba…`), `main_ppgc.pdf` **109 pp**
(`29319a58…`). All three page counts confirmed by opening the PDFs, not by reading a log.

**Mid-session structural change, handled.** Commit `4e84cf7a` split the three paper chapters into
per-section files while this pass was running. I re-extracted the full text of the rebuilt
`main.pdf` and compared it page by page against my pinned snapshot: **0 of 108 pages differ**. Every
finding below therefore stands, and every source coordinate in it was re-resolved against the new
per-section files (`chapters/3_cbic/`, `chapters/4_courb/`, `chapters/5_mobiwac/`) and carries
today's line number.

**Instruments, so a later reader can distrust the right one.** Rendered prose was read from the
PDF text layer (PyMuPDF), not from `.tex`. Table bolding was measured from **span font names** in
the page dictionary, not inferred from source macros. Repository claims were checked against
`git cat-file` on the **published remote refs** (`git ls-remote` confirms `origin/main` =
`8c1b534e`, `origin/mobiwac` = `3c57197c`), not against the working tree, because the working tree
is 90 commits ahead of what a reader can fetch. That distinction produced the largest finding here.

---

## PART I — PERSONA 12: THE DEFENSE DRY RUN

### I.0 Verdict

> **Aprovado com correções menores.** The modal real outcome, and here it is the honest one. The
> document survives every attack I could aim at it from inside its own pages: the null result is
> framed as evidence rather than apologized for, the co-authored chapter's contribution note is on
> the page, the region verbs are bound to their tests everywhere I checked, and the leak audit is
> written out at a length no 8-page paper could afford. **Three items would be filed as
> obrigatórias**, and only one of them is about the science (F-01, the reproducibility appendix
> names paths that are not in the repository a reader can fetch). The rest is a corrections list.

### I.1 Dimension scores

| # | Dimension | Score | Evidence line (measured) |
|---|---|---|---|
| 1 | Problem clarity and delimitation | **5** | The question is stated once, in display, p.12; echoed verbatim as the opening of §6.2, p.77. `next place` is excluded on p.13, p.16, p.61 and in both abstracts. |
| 2 | Command of the state of the art | **4** | Ch.2 §2.1–2.3 positions rather than catalogs: it separates "region/category as means" (HMT-GRN, CatDM, p.17) from "as end" (DRRGNN, POI-RGNN), and closes the MTL-optimizer section on the *negative* consolidation (random weighting competitive; unitary scalarization matching), p.22. Not a 5: the gap statement rests on "to our knowledge" (p.58, p.60) with no systematic search described. |
| 3 | Methodological coherence | **4** | Alternatives are named and rejected with reasons: cascade vs parallel is *tested*, not asserted (p.73); class weighting was tried and lowered both metrics (p.63); nineteen balancers screened (p.60). Not a 5: the choice of a fixed 0.75/0.25 weighting is "tuned once on validation during development" (p.63) with no budget stated. |
| 4 | Rigor and honesty of results | **5** | Every result I sampled carries its convention. p.70 Table 10 bolding measured against the caption's rule: in the region block only Istanbul/FL/TX/CA are bold, AL and AZ are regular with `≈`. Epoch-selection optimism is volunteered (p.74), the freeze control's scope is stated three times, and the AZ 0.00 is reported as a match, never upgraded. |
| 5 | Contribution | **4** | Named, falsifiable, and not inflated: the representation-plus-topology finding (p.15, p.77). Not a 5 for a coletânea reason — see Q1 in the transcript; the *dissertation-level* delta is stated in the frame but never given its own consolidated evidence display. |
| 6 | Recognition of limitations | **5** | Six numbered limitations, p.78–79, each tied 1:1 to a future-work item; the task-pair confound (limitation 6) is a concession a committee would otherwise have to extract. Appendix E adds "this work adds no de-identification of its own", p.107. |
| 7 | Candidate ownership | **4** | The CoUrb contribution note is in the organization section (p.14) *and* the chapter preface (p.42). Not a 5: Appendix A describes the platform in the collective ("this research produced"), and no per-artifact role split exists (see F-06). |
| 8 | Text quality | **4** | Zero undefined refs, zero `??`, zero literal macro leakage across 108 pages (measured). One torn sentence renders on p.77 (F-02). |
| 9 | **Coletânea unity** | **5** | The strongest dimension. Ch.2 §2.5 is a real hinge, not a summary: its three clauses map onto Ch.3/4/5 in order (p.24–25). Ch.5 §5.2.1 opens by naming what Chapters 3 and 4 left open (p.59). Ch.6 §6.2 confronts the inter-paper divergence head-on: "do not contradict each other; read together, they bound the claim" (p.78). |
| 10 | Defense-readiness of the text | **4** | The text pre-answers the leak attack, the parameter-count attack, the transfer attack, and the cascade attack. It does not pre-answer the reproducibility attack (F-01) or the "is a null result enough" attack in the terms a committee will use (Q2). |

### I.2 Corrections list the banca would file

**Obrigatórias**

1. **The reproducibility appendix names paths a reader cannot reach.** F-01. Appendix A §A.2 says
   "Every path is relative to the platform repository, which Chapter 5 footnotes at first mention"
   (`apx_a_contributions.tex:98`); that footnote is `.../PoiMtlNet/tree/mobiwac`. **Nine of the
   thirteen paths named in Appendices A and D do not exist on `origin/mobiwac`.**
2. **One sentence renders without its subject.** F-02. p.77: "California run, completed since,
   repeats the pattern." The article was lost in an edit; `6_conclusion.tex:132`.
3. **`make check` exits non-zero.** F-03. The brief says all gates pass; measured, it fails — at
   `01915ba7` as well as at `4e84cf7a`, so this is not the split's doing.

**Sugestões** — F-04 (Gowalla date range differs between Ch.4 and Ch.6 with no reconciling note),
F-05 (Ch.2 promises two metrics the document never uses), F-06 (no per-artifact role split in
Appendix A), F-07 (two terms in the new Ch.2 equations are unregistered in the GLOSSARY),
F-08 (Ch.2's characterization of Ch.3's split axis is weaker than what Ch.3 now states).

### I.3 The arguição transcript

Fifteen questions, in the order a UFV/PPGCC committee asks them: contribution first, then method,
then the coletânea block (which this format invites), then rigor, then limitations. For each: what
it tests, the strongest **honest** answer *from material actually in the document*, and — where the
document cannot answer — that, said plainly.

---

**Q1.** *"Em uma frase: qual é a contribuição original desta dissertação — da dissertação, não de
cada artigo?"*

- **Tests:** whether the candidate owns a thesis-level delta rather than three paper-level ones.
- **Strongest honest answer, from the text:** "That whether multi-task learning helps this pair of
  POI tasks is decided by the input representation and the sharing topology built on it, and not by
  the sharing architecture alone — and that no single paper of the three shows this, because it
  takes the null result of the first, the input-only intervention of the second, and the check-in
  level of the third to establish it." This is on the page twice: Contributions, Theoretical (p.15)
  and §6.2 (p.77). The second half of the sentence — *no single paper shows it* — is the part that
  answers the question as asked, and it is the sentence to rehearse verbatim.
- **What the text supports:** fully. This is a pass.

**Q2.** *"O primeiro resultado é nulo. Uma dissertação de mestrado cujo primeiro artigo não
encontra ganho tem contribuição suficiente?"*

- **Tests:** whether the candidate can defend a negative result as evidence rather than as a
  setback. The committee is not hostile here; it is checking that the candidate is not embarrassed.
- **Strongest honest answer, from the text:** three moves, in this order. (i) The null result is
  not the contribution; it is the *instrument* that located the variable: Ch.3 closed with three
  candidate explanations (task dissimilarity, an input representation too poor for both tasks,
  architectural restrictiveness — p.41), and the dissertation tested two of them and reports which
  one moved the result. (ii) The null result is what makes the final positive result interpretable:
  §6.2 "read together, they bound the claim. Multi-task learning is not a free gain that any
  architecture collects, and it is not a dead end that no representation can rescue" (p.78). A
  positive result alone would not bound anything. (iii) It was published, peer-reviewed, at CBIC
  2025 — the community accepted it as a finding.
- **What the text supports:** fully, and §6.5 says it in one sentence the candidate should quote:
  "The negative result was not an obstacle on the way to the contribution; worked through, it was
  the contribution's first half" (p.79).

**Q3.** *"O terceiro artigo, que carrega o resultado positivo, não está aceito. Se ele for
rejeitado, o que sobra desta dissertação?"*

- **Tests:** whether the candidate has conflated peer-review status with evidential status, and
  whether the coletânea still satisfies Art. 21 without it.
- **Strongest honest answer, from the text:** the requirement is already satisfied without it —
  CBIC 2025 is published with a DOI (p.14) and CoUrb 2026 is a second published article with a DOI
  (p.14), and UFV Normas §2.3(iii) admits submitted articles as chapters regardless. On the
  evidence: the Chapter 5 result does not depend on the review outcome; it depends on the protocol,
  which is in the chapter — user-disjoint five-fold CV, four initializations over one fixed fold
  partition, paired tests on four per-seed means with Holm correction, TOST for the match claims
  (p.66–67). A rejection would change where it is published, not what was measured. The status
  wording is correct at all four places it appears in the document ("submitted … under review",
  pp.12, 14, 57, and Appendix B); nothing anywhere reads "accepted".
- **What the text supports:** the status discipline, yes, measured. The Art. 21 arithmetic is in the
  project's compliance record, **not in the dissertation body** — the candidate should carry it to
  the defense verbally, and have the CBIC comprovante on file.

**Q4.** *"No artigo em coautoria (CoUrb), o que exatamente foi contribuição sua? Por que ele conta
como capítulo desta dissertação?"*

- **Tests:** the format's single most-probed weakness. (Sharmini et al.: co-authorship opacity is
  the top examiner concern in publication-based theses.)
- **Strongest honest answer, from the text:** three specific contributions, all of them on the
  page: "this author is the second author, contributed the MTLnet baseline on which the study
  builds, and presented the paper at the event" (p.14, repeated p.42). The load-bearing one is the
  first: MTLnet is the candidate's own first-authored artifact from Chapter 3, and Chapter 4 is an
  intervention *on that artifact* that holds it fixed and changes only its input — so the
  candidate's contribution is the object under test. On eligibility: nothing in Normas §2.3/§2.6
  requires first authorship; articles must be pertinent to the research and published, accepted, or
  submitted.
- **What the text supports:** the contribution note, fully. The *norms* argument is again in the
  project record and not in the body; carry it verbally.

**Q5.** *"Qual é o fio condutor? Convença-me de que isto é uma dissertação e não três artigos
grampeados."*

- **Tests:** coletânea unity — the "colcha de retalhos" attack.
- **Strongest honest answer, from the text:** the thread is one artifact lineage plus one variable.
  Point at three concrete devices rather than asserting unity: (i) Table 1, p.20, traces DGI → HGI →
  MTLnet → ST-MTLNet → Check2HGI → the joint model, one row each, with the chapter that introduces
  it; (ii) the paragraph on p.20 that establishes the final model is a *specialization of the
  MTLnet class* that overrides exactly one component, the shared middle — which is why the Ch.3
  negative and the Ch.5 positive can be read against each other at all; (iii) each chapter preface
  states what later chapters revise in it (pp.26, 42, 57).
- **What the text supports:** fully. This is the document's best answer to its hardest
  format-specific question, and it is not an accident of assembly.

**Q6.** *"O que o artigo 3 corrige do artigo 1? Os dois discordam. Em qual devo acreditar?"*

- **Tests:** whether inter-paper contradiction is confronted or hidden.
- **Strongest honest answer, from the text:** neither is superseded; they are measurements of
  different configurations, and the document says which. Ch.3's conclusion is time-indexed in its
  own preface: "with a place-level embedding and hard parameter sharing, multi-task learning did not
  consistently improve on the dedicated single-task models" (p.26) — a scoped statement, not a
  general one. §6.2 states the resolution: "With a place-level embedding and naive hard sharing, no
  … With a check-in-level representation and a sharing topology built for it, yes" (p.77). Believe
  both, at their own scopes.
- **What the text supports:** fully. Appendix B additionally records the one place where Ch.5's
  *submitted* text mis-described Ch.3, and that this was corrected in both the chapter and the
  version of record (p.97) — a committee reading that will treat it as a point in the candidate's
  favor.

**Q7.** *"Por que multi-task learning, se o modelo conjunto é maior e mais caro? Onde está o
ganho?"*

- **Tests:** whether the candidate is selling an efficiency claim the evidence does not support.
- **Strongest honest answer, from the text:** the cost is disclosed, and the claim is explicitly
  *not* efficiency: "the joint model has about 4.2 million parameters at Alabama against 1.1 million
  for the two dedicated models combined … What the single model provides is operational rather than
  arithmetic: one artifact to train, version, and deploy, and one forward pass" (p.63). Ch.3 also
  reports its own joint model cost 2.3× the cumulative wall time (p.40). The gain is *quality* at
  operational parity: the joint model outperforms both dedicated models on category everywhere.
- **What the text supports:** fully, and the cost sentence appears before any reader could feel
  misled. Do not improve this passage; see the protect list.

**Q8.** *"Como garantiu que não há vazamento — em especial de uma representação treinada sobre todo
o dataset?"*

- **Tests:** the classic kill-shot for a transductive representation.
- **Strongest honest answer, from the text:** the document bounds four channels and states what each
  measurement covers, rather than claiming there is no leak (p.65–66): the training objective is
  label-free; the transductive channel was measured by rebuilding per fold (region −0.33 to +0.01,
  category 0.00 to +0.29 at AL/AZ/FL); the region-transition prior is built per fold from training
  data only, *after* a whole-dataset version inflated region accuracy by 13 to 27 points — a
  disclosed near-miss; and the forward-edge channel was probed against a clean reference encoder.
  The honest core of the answer is the sentence the text itself uses: the measurement "bounds this
  channel rather than closing it", with three limits named (linear probe, Florida only, ancestor
  builds).
- **What the text supports:** fully, and better than the paper could. The candidate should also know
  the *one* residual: visits to places unseen in training are the part the measurement cannot reach
  (p.65) — volunteer it, do not wait to be asked.

**Q9.** *"Os ganhos são significativos? Quantas seeds, qual unidade inferencial, e a variância entre
folds não é maior que o ganho?"*

- **Tests:** whether the candidate knows the statistical footing precisely, including its weakness.
- **Strongest honest answer, from the text:** four initializations × five folds = 20 fitted models
  per cell, but the reported test pairs the **four per-seed means**, so n = 4, and the document says
  so in the abstract, in the contributions list, and in §5.5.3 (pp.3, 15, 66). The registered
  Wilcoxon floors at p = 0.0625 with four pairs, which is why a paired t carries the verdict and the
  registered test is reported alongside it and agrees; both are released with the code. On variance:
  the joint-vs-dedicated separation is large relative to it — category gains +5.33 to +9.35 against
  per-seed sds of 0.01 to 0.10 (Table 10, p.70), and all 20 folds favor the joint model at every
  dataset. On the honest weakness: "All four seeds reuse the same fold partition, so the reported
  intervals do not cover uncertainty over resampled user splits" (p.15, p.74).
- **What the text supports:** fully. This is the answer to give slowly.

**Q10.** *"Os baselines foram tunados com o mesmo esforço? Um modelo conjunto que ganha de um
dedicado mal ajustado não prova nada."*

- **Tests:** the fairness of the central comparison.
- **Strongest honest answer, from the text:** the asymmetry runs *against* the candidate, and the
  text says so: the dedicated category model is tuned per dataset over batch size and learning rate,
  while the joint model uses one configuration held fixed across all six datasets (p.70, p.74).
  "The residual therefore favors the comparator, which makes the reported difference conservative."
  The document immediately adds the honest qualifier: "It does not follow that the bias cancels
  exactly."
- **What the text supports:** fully.

**Q11.** *"O senhor atribui o ganho ao compartilhamento. Mas mediu que os gradientes das tarefas são
ortogonais. Então de onde vem o ganho — e não é só capacidade a mais?"*

- **Tests:** the mechanism, and whether the candidate over-attributes.
- **Strongest honest answer, from the text:** the document rules out two explanations and refuses to
  name a third. Not cross-task transfer: with the region pathway frozen, the category gain survives
  at AL/AZ/FL (p.71). Not parameter count: a dedicated category model widened to the joint budget
  recovers none of the gain, at Alabama (56.16 vs 56.82 narrow vs 64.51 joint) and at California
  (69.88 vs 70.60) — and at both the sweep found the wide model's own better learning rate, so the
  comparison is not rigged (p.77–78). What remains is "the joint architecture itself, and the
  control does not say which part of it" (p.72). The candidate's line: *the document names what the
  gain is not, and declines to name what it is, because the controls run do not license that.*
- **What the text supports:** fully, including the refusal. A committee rewards this.

**Q12.** *"Dados do Gowalla de 2009 a 2011 representam mobilidade hoje?"*

- **Tests:** external validity, and whether limitations are volunteered.
- **Strongest honest answer, from the text:** no, and it is limitation 1 (p.78), tied to a
  future-work item (newer and denser traces, p.79). The mitigation that exists *now* is Istanbul,
  from Massive-STEPS, a collection whose own argument is that the field has leaned too long on
  decade-old data (p.22); the U.S. result repeats there on a different continent and a different
  region unit (p.73). **Caveat the candidate must know before this question:** Ch.4 says the Gowalla
  records were collected "between February 2009 and October 2010" (p.56) and Ch.6 says "between
  2009 and 2011" (p.78). Both are defensible — different extractions of different releases — but
  the document does not say so, and a committee member who notices will ask. See F-04.
- **What the text supports:** the limitation, fully; the date reconciliation, **not at all**.

**Q13.** *"O que um leitor pode efetivamente re-executar? O senhor diz que o código está
disponível."*

- **Tests:** reproducibility as a claim, not as a gesture.
- **Strongest honest answer, from the text:** be precise about the three tiers the document itself
  establishes. Appendix A §A.2 names, per protocol element, the code that implements it and the file
  its output lives in — the fold builder with its `StratifiedGroupKFold` and partition seed 42, the
  four seeds, the per-fold prior, the joint-best scorer, the two test scripts (pp.88–89). It also
  scopes itself honestly: "the list covers the final study alone", and Chapters 3 and 4 ran under
  the weaker protocol, the second in a separate repository (p.90). **But:** I checked those paths
  against the published branch, and nine of thirteen are not there (F-01). Until that is fixed, the
  honest answer to this question is *narrower* than the appendix implies, and the candidate should
  not claim more than the branch contains.
- **What the text supports:** the intent and the structure, yes. The specific promise, **no** — this
  is the one question where the document currently overstates. Fix before the defense.

**Q14.** *"Dados de mobilidade individual são sensíveis. Que considerações de privacidade se
aplicam, e houve comitê de ética?"*

- **Tests:** whether the candidate has thought about it or is improvising.
- **Strongest honest answer, from the text:** Appendix E, and it is unusually candid. Pseudonymity
  is not anonymity (p.107). "This work adds no de-identification of its own. No coordinate is
  perturbed, rounded, generalized, or masked" — with the reason: the Chapter 5 target is itself
  spatial, so coarsening would change the measured quantity. Exposure is limited by restriction
  instead: the social-graph and user-profile files in the deposit are never read, no check-in data
  is redistributed. On ethics review: "the author's position is that review by a research ethics
  committee was not required … It records no approval and no exemption, because none was sought and
  none is claimed" (p.107), with a same-program 2024 precedent consulted and named as a precedent
  rather than a rule.
- **What the text supports:** fully. Say the last sentence exactly as written; do not paraphrase it
  into a claim of exemption.

**Q15.** *"Se recomeçasse hoje, o que faria primeiro — e essa pergunta saiu de qual resultado?"*

- **Tests:** whether future work is derived or decorative.
- **Strongest honest answer, from the text:** the fixed-pair ablation — training the Chapter 5 joint
  model on the Chapter 3 task pair under the check-in-level representation — because it is the one
  experiment that closes limitation 6, the task-pair confound (p.79). The candidate should name the
  confound before naming the experiment: the task pair changed *together with* the representation
  and topology, so no single controlled ablation separates them in the final result; Ch.4 is the
  fixed-pair control for the diagnosis but not for the win.
- **What the text supports:** fully, and the 1:1 limitation-to-future-work mapping in §6.4 means
  every variant of this question has an answer on the page.

### I.4 What impressed me (do not edit these away)

1. **The cost sentence at p.63**, placed before any reader can feel oversold, and its refusal to
   claim arithmetic savings.
2. **The four-channel leak treatment at p.65–66**, especially "bounds this channel rather than
   closing it" and the disclosed 13-to-27-point near-miss from an earlier whole-dataset prior. Very
   few master's texts disclose the version of a mistake that was caught.
3. **The refusal at p.72** to name the shared trunk as the source of the gain, with both controls'
   scopes stated. The temptation to over-attribute here was large.
4. **§6.2's confrontation of the inter-paper divergence** (p.78) — the sentence that makes this a
   dissertation rather than a stack.
5. **Appendix E §E.3's last paragraph** (p.107): records a position and its basis, claims no
   approval. This is what honest looks like when the honest answer is "nothing was sought".
6. **Appendix B's two self-damaging errata** (p.93): both corrections raise the reported cost or
   remove a stated advantage of the architecture the chapter adopts, and the appendix says so.

---

## PART II — PERSONA 17: THE EXCELLENCE SCORECARD (SBC-CTD LENS)

`LEFT_OUT.md` read first, as instructed. **LO-4 declines** the contribution-to-claim table and the
consolidated cross-chapter results table on measured exemplar evidence (0/5 and 1/5 of five
exemplars carry them), and records that the third move, the reproducibility appendix, was taken and
is now Appendix A. **Neither declined move is re-recommended below.** Where a gap I find would
naturally be answered by one of them, I say so and propose a different instrument.

### II.1 Scorecard

| # | Dimension | Verdict | Evidence (measured) |
|---|---|---|---|
| 1 | Problem framing & significance | **OUTSTANDING** | One quotable question, displayed once (p.12), answered as the opening move of §6.2 (p.77). Why-it-matters argued beyond the lab in three registers on p.11: what a service does with each prediction, mobility's societal uses, and the field's own tool imports. |
| 2 | Contribution clarity & unity | **OUTSTANDING** | The intro claims what no single paper claims (p.15). Explicit "Chapter N showed X, which forced Y" bridges exist in both directions: forward at p.12 (each study named as acting on the previous one's candidate explanation) and backward at p.59 (Ch.5 opening by naming what Ch.3 and Ch.4 left open). |
| 3 | Command & critical use of literature | **GOOD** | The chapter-2 test passes on *use*: it builds a means-vs-ends taxonomy and its gaps map 1:1 onto the three questions (p.24–25). Short of outstanding for one measurable reason: two metrics are defined and then never used (F-05), which is the signature of coverage exceeding use. |
| 4 | Methodological rigor & justification | **OUTSTANDING** | Guards designed in, not patched on: the per-fold prior, the user-grouped splitter, the pre-registered analysis plan with its departure disclosed (p.66). Alternatives carry disadvantages as well as advantages (cascade, p.73; class weighting, p.63). |
| 5 | Statistical & empirical rigor | **OUTSTANDING** | n = 4 inferential unit named rather than inflated from n = 20; the fixed-partition limitation stated in three places; verbs bound to tests throughout; AZ's 0.00 reported as a match. Table 10's bolding measured to match its caption exactly. |
| 6 | Originality & insight | **GOOD** | The reframing exists and is quotable — "the representation, more than the architecture, decides whether multi-task learning helps" (p.17). It is *stated*; it is not yet *taught*. See Move 2. |
| 7 | Critical self-assessment & honest negatives | **OUTSTANDING** | This document's superpower, delivered: the arc is narrated as a correction trail (p.13), the null result is defended as evidence (p.79), and two appendix errata run against the chapters' own interest (p.93). |
| 8 | Reproducibility & artifacts | **BELOW** | Appendix A §A.2 exists and is well built — but nine of the thirteen paths it and Appendix D name are absent from the published branch its own footnote points at (F-01). A CTD committee checks exactly this. This is the single dimension where the document currently claims more than a reader can verify. |
| 9 | Writing, structure, voice | **GOOD** | Zero undefined refs/cites, zero macro leakage, zero em-dashes in prose across 108 pages (measured). One torn sentence renders (F-02), and `make check` exits non-zero (F-03) — the sloppiness threshold that flips examiners is not crossed, but these two are exactly the class that flips it. |
| 10 | External validation & impact trail | **GOOD** | Two DOIs plus one under-review submission, each with venue and status on the page (p.14); the CoUrb ownership note is present and load-bearing. Short of outstanding: no products list gathers code, protocol, and platform where a committee finds them (Move 1). |

**The chapter-2 test:** **PASS.** By the end of Ch.2 I knew what the tasks are, why place-level
vectors are the wrong instrument, that MTL balancers usually do not beat a tuned fixed weighting,
and what would license the word "outperforms". Authority is won here. The one thing that leaks it is
F-05: promising MRR and a relative multi-task metric and then never using them.

**The intro-conclusion loop test:** **PASS, tightly.** §6.2 opens with the introduction's question
verbatim and answers it in the introduction's own terms (conditional; representation plus topology).
The four objectives of §1.3 map onto §6.1's three chapter paragraphs plus §6.2's consolidation.
One seam: objective 4 promises to "consolidate the evidence of the three studies" under Ch.5's
leakage-guarded protocol, and Appendix A states plainly that the protocol "is not retroactive"
(p.88). Both are true; the objective's verb is the loose one. See Move 3.

### II.2 The gap report — moves ranked by leverage per hour

**Move 1 — publish the branch the document points at, then add a six-line products paragraph.**
*Serves: dimension 8 (BELOW → OUTSTANDING), dimension 10, and the CTD lens directly.*
Fixing F-01 is not optional (Part I files it as obrigatória), but the *excellence* move is what
comes after: once the paths resolve, one short paragraph at the end of Appendix A §A.2 naming the
five products — the two published articles with DOIs, the submitted manuscript, the platform, the
released protocol implementation — turns a scattered trail into a products list. **Cost: the
publish is repository work, not writing; the paragraph is ~30 minutes and ~6 lines.** This is not
the declined consolidated-results table (LO-4); it is an artifact inventory, which the same
exemplar measurement found in four of five.

**Move 2 — let the reframing teach once, in Ch.2 §2.5.**
*Serves: dimension 6 (GOOD → OUTSTANDING).*
The finding is stated as a property of this study ("the representation … decides whether multi-task
learning helps", p.17). Outstanding is one sentence further: what a *reader of the MTL literature*
should do differently. The document has already earned the sentence and it needs no new claim —
§2.3 establishes that balancers rarely beat a tuned fixed weighting, and §2.5 already argues the
representation is the lever; the missing move is joining them: that an MTL result reported without
its input representation held fixed is not interpretable, which is why this dissertation's own null
and positive results are compatible. **Cost: 2–3 sentences, ~1 hour, no new evidence, no new claim
scope.** Highest intellectual return per line in the document.

**Move 3 — tighten objective 4's verb to match Appendix A's honesty.**
*Serves: the intro-conclusion loop, dimension 2.*
`1_introduction.tex:158` reads "Anchor the final answer … and consolidate the evidence of the three
studies under it". Appendix A says the protocol is not retroactive; Ch.2 p.23 says the tests license
verbs in Ch.5 alone. The fix is a verb, not a paragraph: *anchor the final answer … and read the
evidence of the three studies against it*. **Cost: one line, ~10 minutes.** It removes the only
seam a careful examiner can open in the loop test.

**Move 4 — one sentence reconciling the two Gowalla date ranges.**
*Serves: dimension 9, and closes Q12's caveat.*
Appendix B §B.4 already does this work for the *counts* (two extractions of the same corpus, p.98);
the date ranges are the same story and are not covered. One clause added to that section — that the
earlier chapters read the SNAP-era release and Chapter 5 the category-annotated deposit, whose
measured span runs to 2011 — closes it where the reader already looks. **Cost: one sentence in an
existing section, ~20 minutes.**

**Move 5 — spend the two unused metric promises, or withdraw them.**
*Serves: dimension 3 (GOOD → OUTSTANDING).*
`2_fundamentals.tex:571` promises MRR "where the joint comparison needs a rank-sensitive figure";
`:576` promises the relative multi-task performance change as the cross-task aggregate. Neither
appears anywhere else in the document (measured: zero hits outside Ch.2). Withdrawing them is ~15
minutes and costs nothing. *Post-defense category:* actually computing them would strengthen the
region story, but that is new measurement and the calendar does not have it.

**Not recommended, and why:** a contribution-to-claim table and a consolidated cross-chapter results
table (LO-4, declined by the author on measured exemplar evidence — and I agree with the reasoning:
in a collection the contribution-to-chapter map is explicit by construction, and Table 10 already
*is* the consolidated result).

### II.3 The award lens — could this compete at SBC CTD?

**Yes, with edits — and the edits are Moves 1 and 2, in that order.**

- **(a) Can the problem → contributions → impact story be told in ten pages?** Yes, and the
  material is already shaped for it: the p.12 question, the p.13 three-study arc, the p.20 lineage
  table, Table 10 (p.70), Figure 7 (p.71), the six limitations (p.78). A CTD summary would be
  assembly, not authorship.
- **(b) Is the products list where a committee finds it?** **No.** The products exist — two DOIs and
  the status of the third are in §1.5 (p.14), the code footnotes are in Ch.3 and Ch.5 (pp.27, 58),
  the platform and the protocol implementation are in Appendix A (pp.88–89) — but they are in four
  places and the appendix's paths do not currently resolve on the published branch. Move 1 fixes
  both halves.
- **(c) Are originality and relevance — the double-weighted CAPES axes — argued explicitly, or only
  implied?** Relevance: **argued explicitly** (p.11, and Ch.5 §5.7's anticipatory-set quantification
  at p.74, correctly labeled motivation rather than a measured service result). Originality:
  **argued, but defensively.** The novelty claim rests on "to our knowledge" (pp.58, 60) and on the
  Ch.2 statement that no multi-task model treats next region as a co-equal end target (p.22). That
  is honest, and it is *weaker delivery than the evidence licenses*: the document has a taxonomy
  (means vs ends) into which every cited system falls on the means side. Move 2 is where that
  becomes a positive argument instead of an absence claim.

**On the CTD-competitiveness question itself:** the axis where this document is unusually strong
against a CTD field is dimension 7. A published null result, a diagnosis that names the variable,
and a resolution that outperforms — narrated as a correction trail with the superseded conclusions
still legible — is rarer in that pool than any single positive result would be.

### II.4 Anti-pattern screen (12 documented top-rating killers)

| Anti-pattern | Present? | Measured basis |
|---|---|---|
| Stapled compilation | **No** | Lineage table p.20; bidirectional bridges pp.12/59; §6.2 confronts divergence p.78. |
| Co-authorship opacity | **No** | Three specific contributions named, pp.14 and 42. |
| Catalog literature review | **No** | Means-vs-ends taxonomy, gaps mapped 1:1 to the three questions, pp.17–25. |
| "Published equals untouchable" posture | **No** | Appendix B corrects published prose, including two corrections against the chapters' own interest, p.93. |
| Uneven components | **Partly** | Ch.3 (16 pp) and Ch.4 (15 pp) carry no significance testing; Ch.5 (19 pp) carries the full protocol. Inherent to the arc and *disclosed* (p.23, p.88), so not a defect — but a committee will feel the asymmetry. |
| Broken intro-conclusion contract | **No** | Loop test passes; one loose verb (Move 3). |
| Contribution inflation | **No** | Cost disclosed p.63; gain not attributed to a named component p.72; AZ never upgraded. |
| Unexamined limitations | **No** | Six, numbered, each tied to future work, pp.78–79. |
| Reproducibility gesture without substance | **Partly** | The substance is written (Appendix A §A.2) but nine of thirteen named paths do not resolve on the published branch (F-01). |
| Sloppiness | **Marginal** | Zero undefined refs/cites/macro leaks in 108 pages; one torn sentence renders (F-02); `make check` non-zero (F-03). |
| Notation drift across chapters | **No** | MTLnet/ST-MTLNet/Check2HGI spelling consistent; the three tasks kept distinct at every site checked. |
| Ethics as boilerplate | **No** | Appendix E states what was *not* done (no de-identification) and claims no approval it did not obtain. |

### II.5 The protect list (do not dilute in the correction pass)

1. p.63 the cost-disclosure sentence and its "operational rather than arithmetic".
2. p.65–66 the four-channel integrity treatment, all three stated limits, and the 13-to-27-point
   disclosed near-miss.
3. p.72 the refusal to name the shared trunk, with both controls' scopes.
4. p.74 "The residual therefore favors the comparator … It does not follow that the bias cancels
   exactly."
5. p.78 §6.2's "they bound the claim" paragraph.
6. p.79 §6.5's closing sentence on the negative result.
7. p.107 Appendix E §E.3's final paragraph.
8. p.93 Appendix B's two self-damaging errata rows.
9. Table 1, p.20, and the MTLnet-descent paragraph beside it — the strongest unity device in the
   document.

### II.6 One honest paragraph

In Lovitts' terms this is a **very good dissertation with two outstanding-grade features and one
verifiable overclaim.** The outstanding features are real and rare: the coletânea unity is achieved
rather than asserted, and the critical self-assessment is at a level most examiners never see at
master's level — a published null result defended as an instrument, controls whose scopes are stated
against the author's own interest, and appendix errata that raise the cost of the method the
document adopts. What holds it at "very good" is not a missing experiment; it is that one dimension
(reproducibility) currently promises what the published repository does not deliver, and that the
central reframing is stated rather than taught. **The single highest-leverage investment remaining
is Move 1** — publish the branch the document already points at, then gather the products in six
lines. It converts the one BELOW into an OUTSTANDING, it is the exact thing a CTD committee checks
first, and unlike every other move on the list, it costs repository work rather than new prose.

---

## PART III — FINDINGS (each with its anchor, today's line, the measurement, and what closes it)

### F-01 · **BLOCKER** · Appendix A and D name paths that are absent from the published repository

**Anchor phrase:** "Every path is relative to the platform repository, which Chapter 5 footnotes at
first mention" — `src/chapters/apx_a_contributions.tex:98` (2026-07-28); renders **p.88**.
The footnote it defers to is `5_mobiwac/01_introduction.tex:27`, giving
`https://github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac`.

**What I measured.** `git ls-remote --heads origin` confirms the live published tips:
`origin/mobiwac` = `3c57197c`, `origin/main` = `8c1b534e`. I then resolved each of the **13 paths
printed in Appendix A §A.2 and Appendix D** with `git cat-file -e` against both refs. The local
worktree has all 13; the published branch does not.

| Path as printed | on `origin/mobiwac` | on `origin/main` |
|---|---|---|
| `src/data/folds.py` | present | present |
| `scripts/closing_data/score_joint_best.py` | present | present |
| `scripts/closing_data/superiority_wilcoxon.py` | present | present |
| `scripts/closing_data/region_match_tost.py` | present | present |
| `scripts/build_phase3_per_fold_transitions.sh` | **absent** | present |
| `scripts/embedding_eval/autocorrelation_ceiling.py` | **absent** | **absent** |
| `docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md` | **absent** (lives at `analysis_protocol/STATISTICAL_PROTOCOL.md`) | present |
| `docs/studies/closing_data/joint_best/JOINT_BEST_RESULTS.md` | **absent** (lives at `analysis_protocol/JOINT_BEST_RESULTS.md`) | present |
| `.../stats_n20/m1_stats_n20.py` | **absent** (lives at `scripts/closing_data/m1_stats_n20.py`) | present |
| `.../stats_n20/m2_prereg_perfold.py` | **absent** (lives at `scripts/closing_data/m2_prereg_perfold.py`) | **absent** |
| `.../stats_n20/m1_full_output.txt` | **absent** (basename absent from branch) | present |
| `.../stats_n20/m2_prereg_output.txt` | **absent** (lives at `analysis_protocol/m2_prereg_output.txt`) | **absent** |
| `docs/results/embedding_eval/rescreen_cat/autocorrelation_ceiling.json` | **absent** (basename absent) | **absent** |

**9 of 13 absent on `origin/mobiwac`; 4 of 13 absent on `origin/main`.** Two files
(`autocorrelation_ceiling.py`, the `.json` it writes) are on **no** origin branch: I checked every
`origin/*` ref for the basename and found zero hits. Local `main` is **90 commits ahead** of
`origin/main`, which is how the paths can be simultaneously true on disk and unreachable to a
reader.

**Conclusion.** The single most load-bearing reproducibility promise in the document — a per-element
map from protocol to code to output file — does not resolve for a reader who follows the chapter's
own footnote. Four of the paths would resolve if the reader guessed `main` instead; four resolve
nowhere published. Appendix D's computation script and its output file are in the second group, so
the label-history benchmark that Chapter 5's screening argument rests on is currently unreproducible
from public material. This is a claim about conduct that a CTD committee, or any careful examiner,
verifies by clicking. It is *not* a claim about a result: no number moves.

**What would close it.** Either (a) push the missing files to the branch the footnote names and
re-verify all 13 paths against `origin/mobiwac`, or (b) if the published layout is deliberately
different, correct the printed paths to the published ones and state which ref they are relative to.
Whichever is chosen, the closing evidence is a re-run of the 13-path resolution against the live
remote, not against the worktree — the worktree is what made this invisible.

### F-02 · **MAJOR** · A sentence renders without its subject on p.77

**Anchor phrase:** "California run, completed since, repeats the pattern" —
`src/chapters/6_conclusion.tex:132` (2026-07-28); renders **p.77**, mid-paragraph in §6.2.

**What I measured.** The rendered text layer reads: "… and 64.51 for the joint model. **California
run, completed since, repeats the pattern.**" The preceding line ends in a full stop, so the new
sentence begins with a bare "California". Traced in history: at `70e794f1` the sentence read "A
partial | California run, fifteen of twenty repetitions …" with the article "A partial" at the end
of the previous source line; the REV-013 rewrite at `59de8280` replaced the continuation line only,
and the opening words went with the previous line's tail. The three `make check` prose gates do not
catch it: the torn-sentence checker requires a **lowercase** opener (`check_torn_sentences.py`,
`LEGIT_OPENER` + `first[0].islower()`), and "California" is capitalized, so this class — a torn
sentence whose orphan opener happens to be a proper noun — falls through by construction.

**Conclusion.** A grammatical defect on a page of §6.2, the consolidated answer: the highest-traffic
prose in the document after the abstract. Persona 12's role model reads the conclusion closely and
flips into hypercriticism on exactly this class of finding. Zero numbers are affected; the sentence's
figures (69.88 / 0.26 / 70.60 / 0.07) all check against the surrounding prose.

**What would close it.** Restore an article ("A California run, completed since, repeats the
pattern", or "The California run …") and re-render p.77 to confirm. Separately worth doing, though
it is a gate matter rather than a text matter: the torn-sentence checker's lowercase requirement is
now known to have a blind spot with proper-noun openers — validate any widening in both directions
before trusting it, per the guardrails' never-fired-gate rule.

### F-03 · **MAJOR** · `make check` exits non-zero; the brief states all gates pass

**Anchor phrase:** "This article differs from the other two in a way that changes what this section
has to record." — `src/chapters/apx_b_errata.tex:307` (2026-07-28); renders **p.97**.

**What I measured.** Ran the documented recipe. `make check` returns **exit 1** and prints
`make: *** [check] Error 1`. Walking `src_utils/check.sh`: the `'this paper' / 'this article'` gate
(line 19) sets `FAIL=1` on any non-comment hit, and this sentence is the one hit. I then confirmed
it is **not** a consequence of the mid-session chapter split: I extracted the tree at `01915ba7`,
the commit the brief measures, and the same sentence is present at the same line with the same
narrower `CH=chapters/*.tex` glob — the gate failed there too. Two secondary observations from the
same run: the page-count gate reported "the build did not finish" on my first invocation because
`main.log` was mid-write while the tree was being rebuilt (it reconciled to "defense 108 pp, final
105 pp" once the build settled — the gate behaved correctly, I had caught it in a race), and the
`banned verdict verbs` block prints four hits that are informational by design (`|| echo OK`, no
`FAIL`) — all four are legitimate technical uses of "Pareto".

**Conclusion.** The brief's "`make check` all gates pass" is not what the gate reports at the state
under review. The underlying prose is almost certainly correct — Appendix B is *about* the three
articles, so "This article differs from the other two" is the accurate subject — which makes this a
**gate-specification** defect rather than a prose defect: the sweep exists to catch leftover "this
paper" from re-typeset chapters and has no exemption for the appendix that discusses the articles as
objects. The risk is the documented one from the guardrails' bias table: a gate that fails for a
known-benign reason trains its readers to skip its output, and the next real hit rides along.

**What would close it.** Either reword to "Article 3 differs from the other two …" (zero content
change), or exempt `apx_b_errata.tex` from that one sweep the way it is already exempted from the
banned-words sweep at line 26, with the exemption's reason in a comment. Then `make check` must exit
0, and the exit code — not the printed lines — is the evidence.

### F-04 · **MINOR** · Two different Gowalla collection ranges, no reconciling note

**Anchor phrases:** "collected between February 2009 and October 2010" —
`src/chapters/4_courb/conclusion.tex:20`, renders **p.56**; and "collected between 2009 and 2011" —
`src/chapters/6_conclusion.tex:200`, renders **p.78** (limitation 1).

**What I measured.** Both statements render; I searched all 108 pages for every date range and these
are the only two. Ch.4's is published prose (the CoUrb paper's own limitation sentence). Ch.6's has
a source-side provenance comment at `5_mobiwac/05_setup.tex:20` recording a measurement on the
parquet, 2009-01-21 to 2011-08-16, and noting that the SNAP/`cho2011` dump (Feb 2009–Oct 2010) is
not the data source for Chapter 5. So both numbers are defensible and they describe **different
releases**. Appendix B §B.4 (p.98) already reconciles the *count* difference between the same two
chapter groups for the same reason, and does not mention the dates.

**Conclusion.** A reader who reads Ch.4's limitation and then Ch.6's limitation sees the corpus
vintage change with no explanation, two pages after §6.2 asked them to trust the document's
precision. The information that resolves it exists in a source comment and in the adjacent
appendix section's logic, but not on any rendered page.

**What would close it.** One clause in Appendix B §B.4, where the reader already goes for exactly
this class of question, naming the two releases and their spans; or a parenthetical at
`6_conclusion.tex:200`. Closing evidence: both rendered pages state ranges a reader can connect.

### F-05 · **MINOR** · Chapter 2 defines two metrics the document never uses

**Anchor phrases:** "which is why mean reciprocal rank accompanies it where the joint comparison
needs a rank-sensitive figure" — `src/chapters/2_fundamentals.tex:571`, renders **p.23**; and "the
aggregate is the relative multi-task performance change" — `:576`, renders **p.23**.

**What I measured.** Grepped every chapter and appendix source (`chapters/*.tex` **and**
`chapters/*/*.tex`, the post-split set) for "reciprocal", "MRR", "relative multi-task", "Delta_m":
**zero hits outside `2_fundamentals.tex`**. Confirmed against the render: no rendered page other
than p.23 contains either term. Chapter 5's region results are reported in Acc@10 only.

**Conclusion.** Two promises the document does not keep. The first is the more exposed: it states
MRR accompanies Acc@10 "where the joint comparison needs a rank-sensitive figure", and the joint
comparison in Table 10 has no rank-sensitive figure. This is the one place where Ch.2 reads as
coverage rather than use — the exact failure mode the chapter-2 authority test is looking for, and a
committee member who notices will ask which table to look in.

**What would close it.** Withdraw both clauses (Ch.2 does not need them; the Acc@10 definition and
the dedicated-model reference point carry the section), or compute them for Table 10. The first is
minutes and inside the calendar; the second is new measurement and is a post-defense item.

### F-06 · **MINOR** · Appendix A describes the platform collectively, with no per-artifact roles

**Anchor phrase:** "this research produced a reusable software platform" —
`src/chapters/apx_a_contributions.tex` §A.1, renders **p.88**.

**What I measured.** Read §A.1 in the render. It attributes in the collective throughout ("this
research produced", "The platform provides") and names no individual's role for any component. I
grepped `apx_a_contributions.tex` for "second author", "first author", "presented", "Tarik": zero
hits in prose (one hit in a source comment at `:63`). Cross-checked the platform counts against the
worktree, which supports them: `find src -name '*.py'` gives **192** files and **28,644** lines,
matching "a 192-module, 28,644-line source tree"; `src/losses/` holds 21 balancer directories
(claim: twenty-one); `research/embeddings/` holds 8 engine directories (claim: eight);
`src/models/mtl/` holds 13 backbone directories (claim: thirteen).

**Conclusion.** The numbers hold. What is missing is the ownership split, and the document
establishes elsewhere that ownership is not uniform: Ch.4's study "ran in a separate repository"
(p.88) whose public address is a different author's (`TarikSalles/Spatial_Embeddings`, Ch.4
footnote 1, p.44). A committee that has just probed co-authorship in Ch.4 (Q4) and is then handed a
collectively-attributed platform appendix will ask which parts are the candidate's — and the
document cannot currently answer.

**What would close it.** One sentence in §A.1 naming what is the candidate's own work and what is
shared or inherited. This is *not* the declined contribution-to-claim table (LO-4): that mapped
contributions to claims across chapters; this is a single attribution sentence inside one appendix
section.

### F-07 · **MINOR** · Two terms in the new Chapter 2 equations are unregistered in the GLOSSARY

**Anchor phrase:** "built from a bilinear discriminator that scores how compatible two embeddings
are" — `src/chapters/2_fundamentals.tex:250`, renders **p.19**.

**What I measured.** `GLOSSARY.md` §1's maintenance rule is fail-closed: "a term not in this
registry may not be used in dissertation prose", and the entry lands *before* the term does.
Grepped `GLOSSARY.md`: `bilinear discriminator` → **0** occurrences; `logistic function` → **0**.
The equations landed at `456eaa72`; the only later glossary commit is `01915ba7`, which registered
nine Portuguese Resumo terms and neither of these. The drafting agent flagged the dependency itself
in a source comment at `:308` and proposed the entries for author approval, so this is a queue that
has not been drained, not an unnoticed slip. Precedent noted in the same comment and confirmed:
"discriminator" alone is already in published Ch.3 prose (`3_cbic/method.tex`, p.32), so only the
modifier and the name of σ are new.

**Conclusion.** A fail-closed rule is currently violated by two terms on p.19. The prose itself is
correct and the equations were transcribed from `docs/context/check2hgi_overview.tex` and
independently confirmed against the running code (`alpha_c2p=0.4, alpha_p2r=0.3, alpha_r2c=0.3` in
`Check2HGIModule.py`, assembled as a weighted sum). The defect is registry hygiene, and it matters
because this registry is the mechanism the project uses instead of trusting cross-chapter memory.

**What would close it.** Two rows in `GLOSSARY.md` §4 or §2 (author's call), or a rewording that
uses only registered terms. Closing evidence: a grep of `GLOSSARY.md` returning both terms.
*Related, and for the author rather than for me:* the same passage carries a `[VERIFY]` on whether
Ch.5's two auxiliary terms (weights 0.3 and 0.1, p.62) should be named alongside Equation 2.1. I
confirmed the code defaults for both auxiliaries are 0.0 and that they are enabled per run, so
Equation 2.1 as printed is the three-boundary objective and the prose does not claim more. The
`[VERIFY]` is about completeness, not correctness.

### F-08 · **MINOR** · Chapter 2's account of Chapter 3's split axis is weaker than Chapter 3's own

**Anchor phrases:** "Chapter 3 reports five-fold cross-validation without identifying the split
axis" — `src/chapters/2_fundamentals.tex:601`, renders **p.23**; against "The folds are formed by a
stratified splitter over the samples rather than over the users, so the check-ins of one user may
appear in both training and validation" — `src/chapters/3_cbic/results.tex:30`, renders **p.36**.

**What I measured.** Both sentences render. Ch.2 (p.23) tells the reader that Ch.3 does not identify
its split axis and that only Ch.4 discloses sample stratification; Ch.3 (p.36) identifies its split
axis explicitly, in the same words Ch.4 uses. The Ch.3 sentence is one of the protocol facts added
this round and recorded as an addition in Appendix B (p.94: "that the folds are stratified by sample
rather than by user"). So Ch.2's characterization describes the *published article*, which is
accurate about that artifact, while the *chapter the reader is holding* now says more.

**Conclusion.** A reader who reads p.23 and then p.36 finds the chapter better than the frame said
it was. Harmless to every result, and it is the good direction for an error to run — but it is a
cross-chapter concordance slip in the paragraph whose whole purpose is to tell the reader how the
protocol strengthened across the arc, and it slightly undersells the round's own work.

**What would close it.** Update `2_fundamentals.tex:601` to say Ch.3 stratifies by sample as Ch.4
does (both now disclosed as recovered from the released code, Appendix B), keeping the "only
Chapter 5 splits by user" contrast, which remains exactly true. Closing evidence: p.23 and p.36
agree in the render.

---

## PART IV — WHAT I VERIFIED AS SOUND (so no one "fixes" it)

Recorded because a finding list read alone misrepresents the document, and because each of these
cost a measurement.

1. **Table 10's bolding matches its caption's rule.** Measured from span font names on p.70, not
   from source: region block bold = Istanbul/FL/TX/CA (the `↑` rows); AL and AZ are
   `TeXGyreTermesX-Regular` with `≈`. The caption's promise and the typography agree.
2. **Resumo/Abstract claim parity holds, 19/19.** Ran `_check_pair_parity.py`. Its **defaults are
   stale** — `PT_PAGE=3 EN_PAGE=4`, while the pair now renders on **pp.2 and 3** — and at the
   defaults it reports 19 spurious failures. At the correct pages: **0 failures**, PT 310 words /
   11 sentences, EN 271 / 11. I independently extracted both blocks and confirmed twelve
   PT/EN claim pairs by hand. *A later reader running this checker at its defaults will conclude the
   pair is broken; it is not.*
3. **No undefined references, no macro leakage, no `??`.** Zero pages of 108 contain `??`, `[?]`, or
   literal `\ref`/`\cite`/`\label`/`\includegraphics` text. The doubled-backslash class that printed
   a raw label on a defense page earlier this round does not recur: the checker now covers **49**
   files (it covered 31 before the split fix) and passes its own two-way self-test.
4. **Em-dashes: zero in prose.** Two occurrences exist in the document, both on bibliography pages
   (p.83 in a dataset title, p.86 in a thesis-type field) — reference-list typography, correctly
   outside the prose law.
5. **The CoUrb audited numbers reproduce.** Recomputed both per-state means from the rendered Tables
   6 and 7 (best-of-two-encoders per row, mean over the seven categories): static task FL +20.24,
   CA +20.91, TX +21.98 — matching the "20.2 to 22.0 percentage points" the chapter and Ch.6 report,
   and matching `slides/judge_feedback.md`. Sequential task, same rule: FL +1.86, CA +2.28,
   TX +5.43, with 15/21 strict wins and Florida Outdoors as the technical tie (21.61 vs 21.59) —
   matching the audited count the chapter uses. **The 20.2–22.0 range belongs to the static task,
   and both frame sites attach it correctly to "category macro-F1".** I checked this specifically
   because attaching the static-task range to the sequential task would have been the natural error,
   and it is not made.
6. **Appendix A's platform counts hold** against the worktree (192 modules / 28,644 lines / 21
   balancers / 8 embedding engines / 13 MTL backbones). Only their *location* is at issue (F-01).
7. **Appendix C's AI disclosure no longer claims the reviewer panel.** The false
   "eighteen-reviewer panel" sentence is absent from all 108 rendered pages; the only surviving
   trace is the source comment recording its removal. What remains is scoped correctly, including
   "each pass run by an agent that did not write the text under review" and "Their findings were
   corrections for the author to accept or reject, not approvals".
8. **Appendix B §B.5 is genuinely one line from suppression.** `\input{chapters/apx_b_static_scope}`
   at `apx_b_errata.tex:407` is the only reference to that section; the Ch.4 preface pointer names
   the appendix rather than `\ref`-ing the label (`4_courb.tex:18`), so commenting the input leaves
   no dangling reference. The author's suppression requirement is satisfied as specified.
9. **MobiWac status wording is correct everywhere.** Four rendered sites (pp.12, 14, 57, and
   Appendix B), all "submitted … under review"; zero occurrences of "accepted" for it.
10. **The three tasks stay distinct.** "The exact next place is not predicted" or equivalent appears
    at pp.13, 16, and 61, and both abstracts; no site conflates the three.
11. **Model-name casing is consistent, and the two apparent violations are the sanctioned ones.**
    Counted across the 108-page text layer: `MTLnet` 51, `ST-MTLNet` 35, `Check2HGI` 11, `MTL-Net` 0,
    `ST-MTLnet` 0. The three bare `MTLNet` hits are all legitimate: two are the published expansion
    "ST-MTLNet (Spatial-Temporal MTLNet)" that the GLOSSARY preserves, and the third is Appendix B
    stating that the published article typesets the baseline as `MTLNet`. The two `MtlNet` hits are
    both inside the repository URL `PoiMtlNet`, which Appendix B explicitly exempts as "a literal
    address rather than the model name" (p.95). The errata note is accurate about its own scope.

---

## PART V — UNVERIFIED, blocked on X

1. **Whether the missing Appendix A files are absent from the published branch by intent or by
   omission.** Measured: what is on each ref. Not knowable from the repository: which layout the
   author means a reader to fetch. Blocked on the author. This determines whether F-01's fix is
   "push files" or "correct paths".
2. **Whether Ch.5's two auxiliary loss terms belong in Ch.2's Equation 2.1.** I established the code
   defaults are 0.0 and that they are per-run flags, so the printed equation is not wrong. Whether
   the *shipped* representation for the Ch.5 results enabled them requires that run's configuration,
   which I did not locate. Blocked on the run config. The existing `[VERIFY]` at
   `2_fundamentals.tex:308` states this correctly.
3. **Whether the banca will accept the CoUrb chapter at all.** The norms argument (no first-authorship
   requirement in Normas §2.3/§2.6) is in the project's compliance record and is not printed in the
   dissertation. I can verify the contribution note is present; I cannot verify the Comissão's
   ruling. Blocked on the advisor conversation already logged as open.
4. **Art. 21 comprovante and the anti-plagiarism certificate.** Neither is a document-body matter and
   neither appears in the render (measured: zero pages contain "anti-plagiarism" or "Qualis"). Both
   are defense blockers held outside the text. Blocked on the secretariat; flagged because Q3 and
   Q13 will touch them.
5. **Figure legibility after this round's rescale.** Out of my remit (persona 18 owns it) and I did
   not re-measure glyph geometry. Recorded only because `ANCHORS.md` §2 documents that the obvious
   instrument (`FPDFText_GetFontSize`) is blind to `\includegraphics` scaling — a later reader
   should not "confirm" it with that API.

---

## PART VI — OUT-OF-SCOPE HANDOFFS (one line each)

- **Persona 19 / gate engineering:** `check_torn_sentences.py` cannot see a torn sentence whose
  orphan opener is a proper noun (F-02's class); any widening needs two-way validation.
- **Persona 19:** `_check_pair_parity.py` ships stale default page numbers (3/4 vs the actual 2/3)
  and reports 19 false failures at its defaults.
- **Persona 06 / number auditor:** `_check_pair_parity.py`'s docstring records "PT p.3 310 w / 11 s,
  EN p.4 271 w / 11 s" — the counts are right, the page numbers in the same sentence are not.
- **Persona 13 / UFV compliance:** 43 `[NEEDS SIGN-OFF]` markers remain in source across 18 files
  (`ANCHORS.md` §2 recorded 32 on 2026-07-28); Appendix C asserts the author approved the final
  text, so the queue and that assertion should be reconciled before the deposit.
- **Persona 03 / style:** the `banned verdict verbs` sweep prints four "Pareto" hits that are all
  legitimate technical usage; consider narrowing the pattern so the block's output stays meaningful.

---

*Read-only pass. No file in `src/` was modified. Measurement artifacts (extracted per-chapter text,
the 108-page text layer, and the measurement ledger) are under
`src_utils/_round6/_rev6work/` for anyone re-checking the numbers above.*
