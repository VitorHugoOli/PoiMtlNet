# Banca Simulator (specialist) — Report on CORRECTED v1 Defense Build (round 2)

> Reviewer: simulated UFV/PPGCC defense-committee member (professor doutor, ML / urban computing).
> Scope: FULL corrected-v1 defense build (all chapters, front matter, appendices), corrections round 2 applied.
> Build under review: `articles/dissertacao/src/dissertacao.pdf` (87 pp) + `build/main_final.pdf` (83 pp).
> Read-only. Findings only; edits are a separate, author-approved step.
> Baseline for comparison: round-1 banca pass = aprovado com correções menores, 45/50.
> STATUS: IN PROGRESS — written incrementally so a restart cannot lose work.

---

## 0 · Reading log (what I have read this session)

- [x] `reviewers/12_banca_simulator.md` (my persona) + `reviewers/README.md` (Common protocol)
- [x] `articles/dissertacao/CLAUDE.md`
- [x] `articles/dissertacao/NORTH_STAR.md` §1–§4 (+§5–§6 for arc/errata)
- [x] `articles/dissertacao/UFV_COMPLIANCE.md` §3
- [x] Round-1 banca report (`_review_v1/12_banca_simulator_report.md`) — for delta only; findings re-derived, not echoed
- [x] Corrected build chapters (source .tex + PDF render): 1 intro, 2 fundamentals, 3 CBIC, 4 CoUrb, 5 MobiWac, 6 conclusion, Appendices A/B/C
- [x] Sources of truth for spot-checks: `src_utils/cbic_recompute_result.md`; the reproduced CBIC/CoUrb tables in the chapters; Ch.5 §5.4.2 leak-audit prose; the 87-pp `dissertacao.pdf` text layer (pypdfium2) for what actually renders
- [x] Round-1 banca report read for delta only; every finding below re-derived from the corrected text, not echoed

**Verification method note.** All "renders / does not render" claims are from the PDF text layer of `dissertacao.pdf` (87 pp, extracted with pypdfium2 this session). Page numbers are physical PDF pages. Numbers are traced to the sources named in each finding, never to memory.

---

---

## 1 · Round-2 delta first (did the four named fixes change my verdict?)

The brief asks specifically whether the round-2 fixes move the verdict versus the round-1 banca
pass (*aprovado com correções menores*, 45/50). I verified all four against the rendered PDF and
the source of truth, re-deriving rather than trusting the changelog.

| Round-1 finding | Round-2 status | Evidence (this session) |
|---|---|---|
| **A-2 title placeholder** on folha de rosto / Resumo / Abstract (marked BLOCKER for the build) | **FIXED** | Folha de rosto (PDF p.1) renders `FROM REPRESENTATIONS TO A SINGLE JOINT MODEL: MULTI-TASK LEARNING FOR POINT-OF-INTEREST CATEGORY AND REGION PREDICTION`; the Resumo (p.3) and Abstract (p.4) headers echo it verbatim; zero `[TITLE --- open decision]` placeholders anywhere in the 87-pp text layer. |
| **A-4 scaffolding on body page 35** (`[N_users; VERIFY: recompute per ERRATA.md] ...`) — the flip-trigger | **FIXED** | Full-document token scan (`VERIFY`, `N_users`, `N_poi`, `N_checkins`, `recompute per`, `NEEDS SIGN-OFF`, `GATE FIX`, `diag-best`, `RESULTS_BOARD`, `champion`, `substrate`) returns **zero hits in the rendered body**. Page 35 now reads the clean recomputed sentence (see A-1 below). |
| **A-1 cross-chapter value** (AL joint next-category 64.51 in Ch.5 vs 64.54 in Ch.6) | **FIXED** | Ch.6 §6.2 (p.73) now reads **64.51**, matching Ch.5 Table 3 (p.66). The `B-2` gate note in source documents the joint-best-vs-diagnostic-best basis (AGENT_GUARDRAILS N5). The 56.16 capacity arm and 56.82 dedicated ceiling are unchanged; no conclusion moves. |
| **B.1 CBIC misattribution in Ch.5** (old text: CBIC "studied next-region and observed negative transfer") | **FIXED** and correctly ledgered | Ch.5 intro (p.55 area) and §5.2.3 recap now read "no consistent multi-task advantage for the paired category tasks ... this chapter introduces the next-region task." Appendix B §B.3 (p.83) records the correction, states it was applied to the version of record too, and explains it removes an internal contradiction with the chapter's own novelty claim. |

**CBIC dataset counts** (the round-2 recompute, `src_utils/cbic_recompute_result.md`): wired into
Ch.3 §3.4.1 (PDF p.35) in the safest both-bases form — "21,052 users and 76,544 unique POIs across
1,407,034 check-ins; after discarding users with fewer than five visits ... 13,935 users and 76,266
POIs across 1,392,262 check-ins remain." The values match the recompute file exactly. They remain
`[VERIFY]` in a LaTeX comment (author confirms the basis), which does not render — correct handling.

**Net effect on the verdict:** the round-1 verdict was *aprovado com correções menores* driven by
two visible-incompleteness artifacts (title, page-35 scaffolding) plus two real consistency defects
(64.51/64.54, the frame test-name). Round 2 cleared the two flip-triggers and one of the two
consistency defects. The remaining consistency defect (test-name) persists, and two smaller defects
were newly exposed by the very act of filling the CBIC numbers. The verdict **stays
*aprovado com correções menores*** — the correct and expected modal outcome — and the score rises
by one point to **46/50**: the build a banca now receives no longer hands them a first-impression
flip-trigger, which is the single most valuable thing round 2 bought.

---

## 2 · Annotation list (pre-read; quote + page + what bothers me)

> Built before the arguição, as a real member arrives with. Severity per the Common protocol.
> The two flip-triggers from round 1 are gone; what remains is a short list of genuine but cheap
> defects, none touching a result or the contribution.

### N-1 [MAJOR] The frame still names a superiority test the resolution chapter does not use, and claims a document-wide scope the reproduced chapters do not honor

This is round-1 A-5, **unfixed in round 2**, and it is now the most concrete gotcha a stats-minded
examiner will hit, precisely because the page-35 distraction that used to sit next to it is gone.

- **Test-name mismatch.** Ch.2 §2.4 (PDF p.23): "The paired Wilcoxon signed-rank test compares two
  models across the paired results without assuming normality, and it is the test that licenses the
  verb ``outperforms''." But Ch.5 §5.4.3 (p.64): "superiority is tested with a paired $t$ on the
  per-seed means," and §5.5.2 (p.69): "each gain is significant after a Holm correction across the
  six datasets (paired $t$, corrected $p<0.001$)." Ch.5 never runs a Wilcoxon; Ch.2 never mentions
  the paired $t$ that actually licenses every superiority verb in the dissertation's headline result.
- **Scope overclaim.** Ch.2 §2.4: "The verbs and the tests are bound together **throughout the
  document**: ``outperforms'' follows only from a paired superiority test." Contradicted by the
  reproduced chapters: Ch.3 §3.4.2 says "both our MTL and Single models significantly outperform
  HMRM across all POI categories" with no paired test in the chapter, and Ch.4 uses "outperform"
  repeatedly ("outperform the original MTLNet in all 21 category-state combinations") with no
  significance test at all. The reproduced-article usage is legitimate (reproduction fidelity), but
  the frame's "throughout the document" promises a standard the reproduced texts do not meet.
- **Why it matters:** the fundamentals chapter's most quotable rigor sentence names the wrong test
  and over-scopes the honesty law. It is the one clean opening the leak/stats examiner gets, and it
  is self-inflicted — the honest scoping is available and true.
- **Direction (author decides):** in Ch.2, either name the paired $t$ used in Ch.5 or generalize to
  "a paired significance test"; and scope "throughout the document" to the frame plus Ch.5, or add
  one clause acknowledging the reproduced articles keep their published verb usage. Do **not**
  retrofit tests onto the reproduced articles.
- Severity MAJOR, not BLOCKER: no result changes; the Ch.5 claims the dissertation stakes itself on
  are properly tested. This is a frame-consistency defect, squarely on my flip-list ("numbers/claims
  that differ between chapters").

### N-2 [MAJOR] Round-2 side effect: Appendix B still lists the CBIC dataset counts as "Pending", contradicting the now-filled Chapter 3

The round-2 fix filled the CBIC counts into Ch.3 §3.4.1 (p.35) but did not update the errata table
that tracks them.

- Ch.3 body (p.35): the counts are present — 21,052 / 76,544 / 1,407,034 raw, 13,935 / 76,266 /
  1,392,262 filtered.
- Appendix B, Table B.1 (p.83): the last row still reads "Unfilled dataset placeholders
  ($N_{\text{users}}$, $N_{\text{poi}}$, $N_{\text{checkins}}$) in the results section. *Pending.*
  Not invented: **the chapter renders visible placeholders**; the values await recomputation ...".
- This is a within-document self-contradiction of exactly the kind this persona flips on: the errata
  appendix tells a careful reader that page 35 shows placeholders; page 35 shows the numbers. An
  examiner who cross-reads the errata (they do — it is the map of what was changed) sees the two
  disagree.
- **Direction:** update Table B.1's last row to "Corrected" — state the inserted values (both bases)
  and the sanctioned recompute basis (per-state ETL output, `<5`-visit filter), mirroring the
  language already in the Ch.3 comment. Keep the `[VERIFY]` author-confirmation gate in the body
  comment until the basis is confirmed; that gate does not render and is not the contradiction.
- Severity MAJOR: it is a real cross-reference defect introduced by the round-2 edit, cheap to fix,
  and it undercuts the credibility of the errata appendix (the document's own honesty ledger).

### N-3 [MINOR] Data-vintage claim understated in Chapter 6, and the project's own recompute says so

- Ch.6 §6.3, limitation 1 (p.74): "The five state datasets come from Gowalla check-ins collected in
  **2009 and 2010**."
- This round's sanctioned recompute (`src_utils/cbic_recompute_result.md`, "Bonus finding: data
  vintage") measured the Florida check-ins spanning **2009–2011** (2009: 40,304; 2010: 769,792;
  2011: 596,938), and Ch.5's own hidden data-provenance comment records the full Gowalla dump range
  as 2009-01-21 to 2011-08-16. The recompute file explicitly flags this as review item MJ-5 and
  recommends "2009 to 2011".
- Note the asymmetry that makes this a frame-only defect: Ch.4's conclusion (p.55) says the data was
  "collected between February 2009 and October 2010" — that is the cho2011/SNAP reference range,
  reproduced verbatim from the published CoUrb text, and must stay as published. The frame chapter
  (Ch.6), which speaks in the dissertation's own voice about the data actually used, is where the
  measured 2009–2011 range belongs.
- **Direction:** Ch.6 limitation 1 → "collected between 2009 and 2011" (or "2009 to 2011"). One-word
  class of fix; the limitation's force is unchanged.
- Severity MINOR: a factual understatement of a limitation, contradicted by the author's own
  measurement this round. A banca member who knows the Gowalla dump will note it.

### N-4 [MINOR] No privacy or ethics treatment of individual mobility data (round-1 A-8, still open)

- Full-document scan: "sensitive" appears once (Ch.2 p.22, unrelated — sensitivity of a metric); no
  "privacy", "ethic", "re-identification", "anonymiz*", or "consent" in the rendered prose.
- The work models individual users' movement (check-in traces; user-disjoint splits keyed on user
  identity), which is personal mobility data. Bank question Q18 is near-certain at a Brazilian CS
  defense, and the LGPD context makes it sharper than it was for the paper venues.
- The text pre-answers most attacks; this is the one predictable question it does not touch.
- **Direction:** one short paragraph in Ch.6 limitations — public datasets, region-level aggregate
  targets, user-disjoint evaluation, no re-identification attempted. One paragraph pre-empts it.
- Severity MINOR: a missing defensive paragraph on a predictable question, not a methodological flaw.

### N-5 [MINOR] The capacity-matched baseline is a post-submission frame analysis, California still at 15/20 (round-1 A-7, still open)

- Ch.6 §6.2 (p.73): "a capacity-matched dedicated baseline, run after the Chapter 5 manuscript was
  submitted ... A partial California run, fifteen of twenty repetitions at the time of writing, shows
  the same direction."
- This control is load-bearing: it closes the "the joint model just has more parameters" objection
  (a natural follow-up to the disclosed 4.2M-vs-1.1M parameter cost). Introducing the strongest
  counter-explanation to the headline only in the Conclusion, with one dataset at 15/20, is
  defensible but leaves an opening.
- **Direction:** complete the California run to 20/20 before the defense, or state plainly that
  Alabama (full n=20: capacity arm 56.16 vs joint 64.51 vs dedicated 56.82) is the primary evidence
  and California is confirmatory. The candidate must own the 15/20 aloud.
- Severity MINOR: the partial status is honestly disclosed; the risk is presentational.

### N-6 [MINOR] "Next-POI Prediction" chapter/section titles sit right after the frame's "next place is never predicted" (round-1 A-6, still open)

- Ch.2 §2.1 (p.17) fixes the canonical names and states the exact next place "is named only to hold
  it apart from the two the dissertation studies." Ch.3 is then titled with "Next-POI Prediction"
  (running head + preface) and Ch.4 §4.4 is "Next-POI Prediction" (p.51 area). A reader meeting those
  titles right after the scope statement can momentarily read them as next-place prediction.
- Mitigation present: Ch.3 §3.1 and Ch.4 §4.1 both define their "Next-POI Prediction" as "predict the
  category of the next POI", and Ch.1 §1.1 narrates the task-pair evolution. So it is recoverable.
- **Direction:** one clause in the Ch.3 and Ch.4 prefaces — "what this article calls next-POI
  prediction is the next-category task of Chapter 2." Closes the boundary ambiguity; it is the
  concrete example a banca raises under the coletânea terminology probe (Q22).
- Severity MINOR: internal definitions carry it; one sentence per preface closes it.

### N-7 [NIT] Front-matter items still open (expected at this stage)

- Approval sheet (p.2), banca members, and defense date are placeholders (`\membrobancaA{[... pending]}`
  etc.). Expected — the banca is not formed — but they are prerequisites for the ≥20-day submission
  (Art. 22) and render as bracketed text in the defense PDF. Flag, not a science defect.
- The title is "set for now" pending the advisor's final call (three alternates commented in
  `0_main.tex`). Wired consistently at every echo point, so this is a decision to close, not a defect.

### Out-of-scope handoffs (one line each, not my scope)
- Whether `\texttt` fragments in the appendices render in Times or a fallback; overfull/underfull
  boxes → persona 18 (visual) / 02 (line editor).
- Whether the B4/B2 Qualis strata clear the Art. 21 internal-resolution bar → persona 13 (UFV compliance).
- Whether the Resumo/Abstract `[NEEDS SIGN-OFF]` pair is claim-parity-clean → persona 07 (claims) / 08 as a pair.

---

## 3 · Dimension scores (10 × 1–5), corrected v1

| # | Dimension | Score | Evidence line (corrected build) | vs R1 |
|---|-----------|:---:|---------------|:---:|
| 1 | Problem clarity & delimitation | **5** | Bold inline research question (Ch.1 §1.2, p.13); three tasks kept formally distinct (Ch.2 §2.1, p.17); "the exact next place is not predicted anywhere in this work" in Ch.1 §1.4, Ch.2, and Ch.6. | = |
| 2 | Command of the state of the art | **4** | Ch.2 is critical, not a catalog: next-place lineage (ST-RNN→GETNext), CTLE as the contextual hinge, the scalarization-skeptic MTL line (Kurin, Xin, RLW), current anchors (Massive-STEPS 2025, ReHDM 2025). Docked one: the MTL-in-mobility coverage is still thinner than the per-task depth. | = |
| 3 | Methodological coherence | **5** | Every choice justified against alternatives: cheapest-controlled-test logic for the CoUrb representation move (Ch.1 §1.2); cross-attention over expert-gating (Ch.2 §2.3, Ch.5 §5.3.2); three-role baseline design (Ch.5 §5.4.4); TOST margin justified by service granularity (Ch.5 §5.4.3). No "the advisor suggested it." | = |
| 4 | Rigor & honesty of results | **4** | Pre-registered superiority/non-inferiority assignment, Holm, user-disjoint CV, a leak audit that discloses a caught leak (region prior, +13–27 pp) and a residual coverage tail. Docked one: N-1 (frame names Wilcoxon; Ch.5 uses paired $t$; reproduced chapters say "significantly outperform" untested). The A-1 cross-chapter number is now fixed, but the test-name defect that shared the point still stands. | = |
| 5 | Contribution | **5** | A nameable dissertation-level delta beyond any single paper: the conditional answer (representation + sharing topology decide whether MTL helps these tasks), across a published null, its diagnosis, and its resolution. Master's bar cleared comfortably. | = |
| 6 | Recognition of limitations | **5** | Six concrete limitations (Ch.6 §6.3), each with a consequence and a 1:1 future-work item — including the volunteered task-pair confound ("no single controlled ablation separates the representation-and-topology change from the task-pair change"). Rare and impressive. (One limitation, data vintage, is factually understated — N-3 — but the limitation is present and owned.) | = |
| 7 | Candidate ownership | **4** | Co-authorship handled explicitly (Ch.4 preface + Ch.1 §1.5: 2nd author, presenter, MTLnet author). Numbers traceable. Docked one: the capacity baseline is 15/20 at California (N-5), which the candidate must own live. | = |
| 8 | Text quality | **5** | Organization and register strong and consistent; the arc reads as one document. **Up from 4:** the two round-1 sloppiness signals that triggered the flip (title placeholder, page-35 `VERIFY` scaffolding) are both gone — verified zero scaffolding tokens render. The residual text defects (N-2 stale errata row, N-3 vintage) are ordinary cross-reference slips, not visible-incompleteness artifacts in the body. | **+1** |
| 9 | Coletânea unity (fio condutor) | **5** | The correction-trail arc IS the thread: Introdução Geral states it (Ch.1 §1.2), the time-capsule prefaces enforce it, the Conclusão Geral claims something no single paper claims (Ch.6 §6.2). Inter-paper differences confronted, not hidden. B.1 correction (Ch.5 + Appendix B) makes the Ch.5-corrects-Ch.3 relationship accurate. | = |
| 10 | Defense-readiness of the text | **4** | Pre-answers the leak kill-shot, fair-baseline, parameter-count, external-validity (Istanbul), and fixed-weight questions. Gaps: no privacy paragraph (N-4), and the frame verb-binding overclaim (N-1) hands a stats examiner one clean opening. | = |

**Aggregate: 46/50** (round 1: 45/50). The one-point rise is entirely dimension 8 (text quality):
round 2 removed both first-impression flip-triggers from the build the banca receives. The other
nine dimensions are unchanged — the science, the contribution, and the honesty posture were already
above the master's bar in round 1 and are untouched.

---

## 4 · Arguição transcript (12 questions)

> Posed as a UFV/PPGCC examiner would (PT-BR), each tied to a specific annotation. For each:
> what it **tests**, what a **strong answer** contains, and what the **current corrected text
> supports** (if the text already answers it, that is a pass and the candidate should quote it).
> Doubles as the author's defense-preparation sheet. Coletânea-specific block = Q9–Q12 (four).

### Contribution & positioning

**Q1 (bank #1).** "Em uma frase: qual é a contribuição original desta dissertação, da dissertação, não de cada artigo?"
- *Tests:* whether a dissertation-level delta exists above the three papers.
- *Strong answer:* the conditional answer — the representation, together with the sharing topology built on it, decides whether MTL helps these POI tasks — established across a published null, its diagnosis, and its resolution; no single paper states this.
- *Text supports:* YES. Ch.1 §1.6 (Theoretical contribution) and Ch.6 §6.2 both state it. **Pass** — quote Ch.6 §6.2: "The representation, together with the sharing topology built on it, is what the answer depends on."

**Q2 (bank #2).** "O que seu trabalho mostra que um MTL de mobilidade já existente ainda não mostrava?"
- *Tests:* positioning against prior MTL-in-mobility.
- *Strong answer:* prior mobility MTL uses region/category as auxiliary signals toward next place (MCARNN, CSLSL, HMT-GRN, CatDM); none predicts next region as a co-equal end target alongside next category, and none isolates the representation as the deciding variable.
- *Text supports:* YES. Ch.2 §2.3 ("no multi-task model among them predicts the next region as a co-equal end target alongside the next category") and Ch.5 §5.2.3/§5.2.4. **Pass.**

### Method justification

**Q3 (bank #4).** "Por que MTL? Mediu a comparação com dois modelos separados com o mesmo orçamento de tuning dos dois lados?"
- *Tests:* the fair-baseline standard and the parameter-count confound (annotation N-5).
- *Strong answer:* the dedicated single-task model is the operative ceiling, tuned at its own optimum (Ch.5); and a capacity-matched dedicated category model widened to the joint budget (4.2M vs 0.6M at Alabama) does NOT recover the gain — at Alabama, full n=20, best capacity config 56.16 vs joint 64.51 vs dedicated 56.82 — so it is the second task's training signal, not parameters.
- *Text supports:* PARTIALLY. Ch.6 §6.2 carries the capacity-matched baseline, but it is a post-submission frame analysis and California is 15/20 (N-5). **Own the partial run in the room** and lead with the full-n Alabama triple. The number reconciliation to 64.51 (round-2 fix) means the Alabama comparison is now internally consistent between Ch.5 and Ch.6 — a cleaner story than round 1.

**Q4 (bank #6).** "Por que um balanceador de gradientes e não uma soma ponderada estática? A soma estática não venceu de fato no seu caso final?"
- *Tests:* the dangerous question — the fixed-weight baseline in fact sufficed in Ch.5.
- *Strong answer:* concede it plainly. Ch.2 §2.3 already takes the cautious position (balancers often do not beat a tuned fixed weight); in the Ch.5 configuration the two tasks' gradients are near-orthogonal (cosine ≈ +0.001), so a balancer had nothing to correct and a tuned fixed weighting sufficed. CBIC's Nash-MTL preference is time-indexed and weakened by a later optimizer-implementation finding (Ch.3 preface).
- *Text supports:* YES, and honestly. Ch.6 §6.2 ("why gradient-balancing optimizers had little to correct ... and why a tuned fixed weighting sufficed") and Ch.5 §5.2.4. **Pass** — a strength, because the text concedes it first. Note the cosine number's scope travels correctly (four seeds, three of six datasets, earlier data preparation, directional conflict only).

**Q5 (bank #7).** "Justifique a representação. Que alternativas considerou e por que as descartou?"
- *Tests:* command of the representation design space (the spine of the thesis).
- *Strong answer:* one-hot → distributed (skip-gram) → graph-infomax place embeddings (DGI, HGI) → the check-in level; CTLE is the closest contextual prior and is run as a control; a feature-concatenation control and a place-level-embedding control isolate what the per-visit representation adds (Ch.5 §5.5.1: silhouette 0.57 vs 0.00, kNN purity 0.98 vs 0.78).
- *Text supports:* YES. Ch.2 §2.2 (the lineage + Table 2.1) + Ch.5 §5.4.4 (CTLE and concat controls). **Pass** — the lineage table (DGI→HGI→MTLnet→ST-MTLNet→Check2HGI→joint model) is the one-glance answer.

### Experimental rigor

**Q6 (bank #8, the classic kill-shot).** "Como garantiu que não há vazamento? Em particular, uma representação pré-treinada no dataset inteiro não passa informação do teste para o treino?"
- *Tests:* leakage from the whole-dataset transductive representation.
- *Strong answer:* three grounds (Ch.5 §5.4.2): label-free training objective; a per-fold rebuild moves both tasks by at most a third of a point at AL/AZ/FL; and the one component that could pass information between visits — the region-transition prior — is built per fold, after an earlier whole-dataset version was caught inflating region accuracy by 13–27 points, and our own joint/dedicated models do not use it (only the HMT-GRN baseline does). Residual gap disclosed: visits to places unseen in training (the 13–33% out-of-coverage tail).
- *Text supports:* YES, exemplary — the single strongest passage. **Pass.** Walk the committee through it; present the caught leak as evidence of discipline and own the coverage tail as a stated limitation, not a hole.

**Q7 (bank #10).** "Os ganhos são significativos? Quantas seeds, quantos folds, qual teste, e a variância entre folds é maior que o ganho?"
- *Tests:* the statistical protocol — and whether the frame and the chapter agree (annotation N-1, the live opening).
- *Strong answer:* n=20 (4 seeds × 5 folds), paired $t$ on per-seed means (n=4), Holm across six datasets, category p<0.001 everywhere; region superiority at 4/6 and TOST non-inferiority (±2 pp) at AL/AZ with 90% CIs inside two points (AL −0.63..−0.20; AZ −0.08..+0.07).
- *Text supports:* YES for Ch.5 — **BUT the candidate will be caught by N-1**: Ch.2 §2.4 tells the reader "the paired Wilcoxon signed-rank test ... licenses the verb outperforms," while Ch.5 uses a paired $t$. Expect verbatim: "O senhor disse Wilcoxon no Capítulo 2 e $t$ pareado no Capítulo 5. Qual foi usado, e por que a discrepância?" **Fix the frame before the defense**; if asked live, concede it is a frame editing slip and the paired $t$ on per-seed means is the test actually run, reported with 90% CIs. This is the one question where the corrected text still trips the candidate.

**Q8 (bank #11).** "Os baselines foram tunados com o mesmo esforço que o seu método?"
- *Tests:* baseline fairness.
- *Strong answer:* per-task SOTA re-implemented on the same folds/initialization (POI-RGNN, HMT-GRN, STAN); dedicated models tuned at their own optimum (best-vs-best); ReHDM reported under its own published protocol and disclosed as such; HMT-GRN/STAN adapted to the region target and stripped of their next-place machinery (a fairness concession, stated).
- *Text supports:* YES. Ch.5 §5.4.4. **Pass** — the honest disclosure of the adaptations is a strength, not a hidden edge.

### Coletânea-specific (four; the format's real pressure points)

**Q9 (bank #19).** "Qual é o fio condutor? Convença-me de que isto é uma dissertação e não artigos grampeados."
- *Tests:* coletânea unity — the dimension a publication-based thesis lives or dies on.
- *Strong answer:* the correction trail — null, diagnosis, resolution — with each chapter revising the previous one's conclusion, stated in the Introdução Geral and enforced by the time-capsule prefaces.
- *Text supports:* YES, strongly. Ch.1 §1.2 + the three prefaces + Ch.6 §6.2. **Pass** — the dissertation's best feature. The Fundamentals chapter (thin, de-duplicating, with the lineage table) is the structural proof that the three papers were made to read as one document, not stapled.

**Q10 (bank #20).** "O que o artigo 3 corrige do artigo 1? Os números divergem, qual versão devo acreditar, e a Conclusão Geral discute essa evolução?"
- *Tests:* whether inter-paper contradictions are confronted — and directly probes the round-2 B.1 and A-1 fixes.
- *Strong answer:* Ch.5 does not contradict Ch.3; read together they bound the claim (MTL is neither a free gain nor a dead end). CBIC's null holds for its configuration (place-level input, hard sharing, the static-classification + next-category pair); Ch.5 changes the representation, the topology, AND introduces the region task. The Conclusão Geral §6.2 states exactly this.
- *Text supports:* YES, and **round 2 strengthened it two ways.** (a) The B.1 correction means Ch.5 now describes Ch.3 accurately — Ch.3 paired *static category classification* with next-category and *hypothesized* negative transfer on a parity null; it did not study region and did not observe negative transfer. Before round 2 this relationship was misstated, and a banca that had read both chapters would have caught the contradiction. (b) The Alabama joint value is now 64.51 in both Ch.5 and Ch.6, so the sharper form of this question — "64.51 in one chapter, 64.54 in the other; which is right?" — **no longer has a target.** **Pass**, and the candidate can now walk the correction trail without a number disagreement or a misattribution to explain away.

**Q11 (bank #21).** "No artigo em coautoria (CoUrb), o que exatamente foi contribuição sua?"
- *Tests:* individual contribution in the 2nd-author chapter — a real concern for the coletânea format and one I probe explicitly.
- *Strong answer:* Vitor is the author of the MTLnet baseline the CoUrb study builds on (his 1st-author CBIC work), the second author of the paper, and presented it at the workshop; the CoUrb study holds his architecture fixed and varies only the input representation, so the diagnosis it delivers rests on his baseline.
- *Text supports:* YES. Ch.4 preface + Ch.1 §1.5 state all three roles ("first author of the baseline model MTLNet, introduced in Chapter 3 ... second author ... presented the paper"). Appendix B §B.2 also records the CoUrb-side corrections. **Pass** — the disclosure is explicit and placed where the reader meets the chapter. The candidate should add, aloud, that the intellectual dependency runs the right way: CoUrb's finding is a statement about *his* MTLnet, which makes the second-author chapter integral to the arc rather than a borrowed result.

**Q12 (bank #22 + #24).** "A notação muda entre capítulos e alguns experimentos falhos ficaram de fora. O que 'Next-POI Prediction' significa no Capítulo 3 versus o 'next place' que o senhor diz nunca prever? E a iteração do BRACIS, onde está?"
- *Tests:* terminology consistency across chapters (N-6) and documentation of failed attempts (the BRACIS iteration + the corrected region-cost claim).
- *Strong answer:* "Next-POI Prediction" in Ch.3/Ch.4 is the reproduced articles' own name for the next-*category* task (Ch.3 §3.1 and Ch.4 §4.1 define it as "the category of the next POI"); it is not next-place, which is never predicted. The BRACIS submission (rejected, unpublished) is in Appendix A as an earlier iteration whose central region-cost claim was corrected by MobiWac (traced to an fp16 numerical-precision artifact + older protocol); no result of it is cited as evidence.
- *Text supports:* MOSTLY. Appendix A contains the BRACIS containment cleanly and Ch.3/Ch.4 define their terms internally — but the terminology bridge at the chapter boundary is the one gap (N-6): neither preface says "what this article calls next-POI prediction is the next-category task of Chapter 2." **Add the one-clause bridge** to the Ch.3 and Ch.4 prefaces. On BRACIS the text is fully defensible — present it as documented self-correction, not a skeleton in the closet.

---

## 5 · Verdict + corrections list

### Verdict: **APROVADO COM CORREÇÕES MENORES** (46/50)

This is a defensible master's dissertation and, on the text, a good one. It has a sharp research
question, a genuine dissertation-level contribution that no single paper carries, an honest
correction-trail arc that makes three differently-shaped papers read as one investigation, and a
rigor-and-honesty posture — pre-registered tests, a leak audit that discloses a caught leak,
volunteered limitations including a self-incriminating confound — that is above the master's bar.
I want to pass this candidate and the corrected text supports it more cleanly than round 1 did.

**Change versus the round-1 pass (45/50 → 46/50, verdict unchanged):** round 2 did the single most
valuable thing it could for a banca reading — it removed the two first-impression flip-triggers.
The title now renders on the folha de rosto, Resumo, and Abstract; body page 35 no longer prints
`[N_users; VERIFY: recompute per ERRATA.md]` in the middle of a reproduced article. My first
impression, which this persona forms by the end of the fundamentals chapter, is no longer poisoned
by visible incompleteness, so the hypercritical mode never triggers. Round 2 also fixed one of the
two real consistency defects (the 64.51/64.54 cross-chapter number) and corrected the B.1
misattribution that had Chapter 5 misdescribing what Chapter 3 studied — a defect a committee that
read both chapters would have caught and that went to the coletânea's core question (what does
paper 3 correct in paper 1). That is why dimension 8 (text quality) rises from 4 to 5 and the
aggregate from 45 to 46.

The verdict stays *com correções menores* and not *sem ressalvas* because a short list of genuine
defects remains, none touching a result: (1) the frame still names the paired Wilcoxon test while
the resolution chapter uses a paired $t$, and over-scopes the verb-binding law to "the whole
document" (N-1, unfixed from round 1 — now the cleanest live opening for a stats examiner); (2) a
round-2 side effect — Appendix B still lists the CBIC dataset counts as "Pending / renders visible
placeholders" although Chapter 3 now shows them (N-2); and (3) a factual understatement of the data
vintage that the round's own recompute contradicts (N-3). All three are cheap, and two of them are
consequences of round-2 edits that did not propagate to every location. None is a BLOCKER; all are
the kind of thing the banca files as obrigatória and expects fixed in the corrected version.

No BLOCKER endangers the contribution or the method. The front-matter placeholders (approval sheet,
banca members, date; N-7) are submission prerequisites, expected open now, not scientific defects.

### Correções OBRIGATÓRIAS (the banca would file these)

1. **[N-1] Fix the verb-binding law in Ch.2 §2.4 (p.23).** (a) Either name the paired $t$ used in
   Ch.5 or generalize to "a paired significance test" so the frame and Ch.5 agree; (b) scope
   "throughout the document" to the frame + Ch.5, or add a clause acknowledging the reproduced
   articles keep their published verb usage. Do NOT retrofit significance tests onto the reproduced
   CBIC/CoUrb texts (that would break reproduction fidelity). This is the one defect that will be
   raised aloud if unfixed (Q7).
2. **[N-2] Reconcile Appendix B Table B.1 (p.83) with Chapter 3.** Change the last row from
   "Pending / renders visible placeholders" to "Corrected", stating the inserted values (both bases)
   and the sanctioned recompute basis. The errata appendix is the document's honesty ledger; it must
   not tell the reader page 35 shows placeholders when it shows numbers.
3. **[N-3] Correct the data vintage in Ch.6 §6.3 limitation 1 (p.74):** "2009 and 2010" → "2009 to
   2011", per this round's own recompute (`cbic_recompute_result.md`) and Ch.5's measured dump range.
   Leave Ch.4's "February 2009 and October 2010" as published (reproduction fidelity — it is the
   cho2011/SNAP reference range, not the dissertation's own voice).
4. **[N-7] Front-matter completion** (approval sheet, banca members, defense date) once the banca is
   formed — a submission prerequisite (Art. 22), expected open now.

### Sugestões (would strengthen; not blockers)

5. **[N-4] Add a short privacy/ethics note** in Ch.6 limitations (public datasets, region-level
   aggregate targets, user-disjoint evaluation, no re-identification). Pre-empts a near-certain
   LGPD-era banca question (Q18) at one paragraph's cost.
6. **[N-5] Complete the California capacity-matched run** (15/20 → 20/20) before the defense, or state
   in §6.2 that Alabama (full n=20) is the primary evidence and California is confirmatory. Own the
   partial run aloud.
7. **[N-6] Add a one-clause terminology bridge** to the Ch.3 and Ch.4 prefaces: "what this article
   calls next-POI prediction is the next-category task of Chapter 2." Closes the momentary
   next-place/next-category ambiguity at the chapter boundary (Q12).

### Defense-preparation priorities (from the arguição, §4)

- **Pre-empt the Wilcoxon-vs-$t$ question (Q7, N-1):** fix the frame; if asked live, concede a frame
  editing slip and state the paired $t$ on per-seed means (n=4), Holm-corrected, is the test run.
  This is the highest-value single fix — it is the last remaining question the corrected text loses.
- **Own the number you did not fully run (Q3, N-5):** lead with full-n Alabama (capacity 56.16 vs
  joint 64.51 vs dedicated 56.82); name California as 15/20 confirmatory before being asked.
- **Walk the committee through the leak audit (Q6):** the single strongest passage; present the
  caught region-prior leak (+13–27 pp) as evidence of discipline and own the coverage tail.
- **Use the correction trail as the answer to the coletânea probe (Q9/Q10):** the B.1 fix and the
  64.51 reconciliation mean the arc can now be narrated with no internal number or attribution
  contradiction to explain — lean on that.
- **Present BRACIS as self-correction (Q12), not a skeleton in the closet.**

### What impressed me (do not edit away)

1. **The honest arc as the contribution.** A published negative result, its diagnosis, and its
   resolution, narrated as a correction trail where each chapter states which earlier conclusion it
   supersedes (Ch.1 §1.2, the three time-capsule prefaces, Ch.6 §6.2). Executed, not asserted. The
   round-2 B.1 correction protects this: the Ch.5-corrects-Ch.3 relationship is now stated
   accurately, which is the load-bearing joint of the whole arc.
2. **The leak audit (Ch.5 §5.4.2).** Three grounds, a disclosed caught leak (the whole-dataset region
   prior, +13–27 pp, now built per fold), and an honestly stated residual coverage tail. Pre-answers
   the classic ML-banca kill-shot; the best-defended passage in the document.
3. **Cost honesty (Ch.5 §5.3.2, "operational rather than arithmetic").** The text states plainly the
   joint model is larger (4.2M vs 1.1M combined at Alabama) and costs more compute, and defines the
   benefit as operational. No compute-saving overclaim; the Ch.1 §1.1 guard held all the way through.
4. **Volunteered limitations with consequences (Ch.6 §6.3).** Six concrete limits, each tied 1:1 to a
   future-work item — including the task-pair confound the candidate could have hidden and instead
   names ("no single controlled ablation separates the representation-and-topology change from the
   task-pair change"). Volunteering the weakness that most threatens the headline is exactly what
   distinguishes a candidate who knows their work's limits.
5. **The thin Fundamentals chapter with the model-lineage table (Ch.2 §2.2, Table 2.1).** DGI → HGI →
   MTLnet → ST-MTLNet → Check2HGI → joint model in one glance. It is the structural device that makes
   three papers read as one document — the coletânea's hardest requirement, met explicitly.
6. **Co-authorship handled cleanly (Ch.4 preface + Ch.1 §1.5):** 2nd author, presenter, author of the
   baseline the study builds on — stated where the reader meets the chapter, not buried.
7. **BRACIS containment (Appendix A).** A rejected iteration presented as documented self-correction,
   with its corrected region-cost claim traced to a numerical-precision artifact and no result cited
   as evidence.

---

_End of report. Read-only review; no dissertation files modified except this report. Final message
to the author is the verdict above._

---

### Provenance note — open Reviewer findings checked

Before closing I checked the background Reviewer's open findings for this conversation. Two `warn`
findings are held, **both against prior-session reports** (round-1 suite / an earlier specialist
draft, frames dated 2026-07-23), not against this report: (a) an earlier report presented an
external arXiv identifier ("Elich 2311.04698") as an MTL work the author should add; (b) an earlier
report computed a capacity-gap as "+8.38 over the capacity arm", which mixes the 64.54 and 56.16
bases. I verified this report reproduces **neither**: it recommends no external citation to add, and
it quotes the three Alabama values (56.16 / 64.51 / 56.82) as source numbers without computing any
gap from them. I did not edit the prior reports (read-only), so I leave those two findings for the
author; they are noted here only so the loop is closed.
