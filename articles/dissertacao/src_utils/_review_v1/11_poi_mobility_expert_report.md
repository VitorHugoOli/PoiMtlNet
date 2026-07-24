# Review v1 — Persona 11 · POI / mobility expert

**Reviewer:** next-location / POI-recommendation domain expert (critique-canon prior: Dacrema 2019, Sánchez & Bellogín 2022, POI Pitfalls 2025).
**Scope:** Ch.2 (fundamentals), Ch.3–5, and POI/mobility claims in Ch.1/6.
**Build under review:** `articles/dissertacao/src/` — `main_defense.pdf` (87pp), chapter .tex sources.
**Status:** COMPLETE.

---

# ============ FINAL REPORT (output contract) ============

## Overall verdict: **SOUND-WITH-CORRECTIONS**

The POI/mobility science is sound and, in places, exemplary. Chapter 5 is a model of honest
protocol disclosure that most published next-location papers do not reach: the
overlap-cannot-leak argument is stated explicitly, the A4 transductivity audit travels with its
scope and its unseen-places residual, the per-fold region-transition prior carries the historical
13–27 pp inflation as a cautionary record, every external baseline has a provenance sentence with
its asymmetries disclosed at the point of comparison, the result verbs are bound to their tests,
and Istanbul is framed as external validity for the *gain over the ceiling*, not for absolute
Acc@10. Those are real credibility signals and the author should not touch them.

The verdict is not "sound" because one BLOCKER and a cluster of MAJOR items stand between the
current build and a defense. The BLOCKER is a build/interpretability defect (Ch.3 renders raw
`[VERIFY: recompute…]` placeholders in place of its only dataset statistics). The MAJORs are
text-level disclosure and consistency corrections, not failures of the underlying experiments:
a self-contradiction over the 93% predictability ceiling, an unstated split axis in Ch.3 that
breaks the arc's protocol-strengthening story, a data-vintage limitation that misdescribes the
Ch.5 data by the repo's own measurement, unreconciled cross-chapter dataset statistics, and two
missing-disclosure items the field's own critique canon demands (revisitation intuition;
per-user vs per-sample averaging). All are fixable in the text.

## Top 3 findings

1. **[BLOCKER] Ch.3 dataset statistics render as raw `[VERIFY: recompute per ERRATA.md]`
   placeholders in the compiled defense PDF (p.35).** A results chapter presents per-category F1
   tables with no users/POIs/check-ins count at all, and the scaffolding text is visible.
2. **[MAJOR] Ch.2 §2.1 presents Song et al.'s 93% predictability as "the ceiling … against which
   any predictive model should be read," which §2.4 explicitly contradicts** ("it is not … a
   ceiling on … category macro-F1 or … region ranking"). The §2.1 framing is also domain-wrong:
   the 93% bound is for next-location at coarse spatial resolution, not for 7-class category F1
   or census-tract ranking.
3. **[MAJOR] Ch.3 never states its cross-validation split axis (user- vs sample-disjoint).** The
   arc's honesty story is that the protocol strengthens across chapters; Ch.4 discloses its
   weaker sample-stratified split, but Ch.3 is silent, and no CBIC-era code exists in the repo to
   verify it. A reader cannot tell whether Ch.3's null result rests on a leakage-prone split.

---

## Ranked findings (quote + location + severity + suggested direction)

### BLOCKER

**B1 · Ch.3 dataset statistics are unfilled placeholders in the built PDF.** (Lens 9 —
reproducibility; also a build defect.)
- *Quote (main_defense.pdf p.35, from `chapters/3_cbic.tex` §5.1):* "This subset comprises a
  total of [$N_{\text{users}}$; VERIFY: recompute per ERRATA.md] users, [$N_{\text{poi}}$; VERIFY:
  recompute per ERRATA.md] unique Points-of-Interest (POIs), and [$N_{\text{checkins}}$; VERIFY:
  recompute per ERRATA.md] check-ins."
- *Why it matters:* Ch.3 is a results chapter whose Florida F1 tables cannot be interpreted
  without the corpus size, and the raw VERIFY scaffolding in a defense build is a kill-shot at
  the banca. Known and catalogued (CBIC `ERRATA.md` non-citation #1; NORTH_STAR §4).
- *Direction:* Execute the sanctioned recompute (repo-committed script over the CBIC-era Florida
  Gowalla pipeline, <5-visit users dropped), author-approve, insert. CoUrb's published FL row is
  a cross-check only, not a source. Until then the chapter is not defense-ready.

### MAJOR

**M1 · The 93% predictability figure is used as a universal ceiling in §2.1 and disowned in
§2.4.** (Lens 8 — metrics/ceilings; internal contradiction.)
- *Quote (§2.1, `2_fundamentals.tex`):* "a potential predictability of about 93\% on where an
  individual goes next. That ceiling is the reference point against which any predictive model
  should be read."
- *Contradicting quote (§2.4):* "it is not, however, a ceiling on seven-class category macro-F1
  or on region ranking, which are different label spaces."
- *Why it matters:* Song et al.'s Π_max is a next-location bound at cell/antenna resolution; it
  does not bound 7-class category macro-F1 or census-tract Acc@10 — the dissertation's actual
  targets. §2.1 anchors the reader on an inapplicable ceiling and then §2.4 removes it. Ch.1 §1.1
  uses the same number correctly ("next location"), so §2.1 is the lone offender.
- *Direction:* Harmonize §2.1 down to §2.4's already-correct scoping — present 93% as the
  next-location predictability bound that frames why mobility is learnable, and name the dedicated
  single-task model as the operative ceiling for the two tasks studied. The fix language already
  exists in §2.4.

**M2 · Ch.3's split axis is unstated; the arc's protocol-strengthening story requires it.** (Lens
1 — split legitimacy, the #1 lens.)
- *Quote (§5.1, `3_cbic.tex`):* "all experiments were conducted using a 5-fold cross-validation
  methodology." (No user-vs-sample axis anywhere in the chapter.)
- *Why it matters:* Ch.4's preface discloses its split is "by sample rather than by user, a
  weaker protocol"; Ch.5 states user-disjoint with the leakage argument. Ch.3 has no counterpart
  sentence. The GLOSSARY itself flags "Ch.3's split … verify from the CBIC codebase before
  asserting it in prose," and no CBIC-era code is committed (the `articles/CBIC___MTL/` folder is
  .tex/.bib only). The CoUrb codebase (same pipeline family, firsthand-verified in NORTH_STAR §4:
  plain `StratifiedKFold`, userid dropped) is sample-stratified, so the likely truth is that Ch.3
  is *also* not user-disjoint — precisely the leakage-sensitive fact the reader is left to assume.
- *Direction:* Add one preface/methods sentence stating Ch.3's split axis honestly (if
  sample-stratified like CoUrb, say so — it strengthens the arc; if it cannot be verified from a
  surviving artifact, state that it followed the same pipeline as Ch.4 and mark the residual). Do
  not let silence imply user-disjoint.

**M3 · The data-vintage limitation misdescribes the Ch.5 data by the repo's own measurement.**
(Lens 6 — staleness/representativeness.)
- *Quote (Ch.6 limitation 1):* "come from Gowalla check-ins collected in 2009 and 2010."
- *Repo provenance (author's hidden comment, `5_mobiwac.tex` §setup-data):* "Date range MEASURED
  on the parquet 2026-07-09: 2009-01-21 .. 2011-08-16 -> 'collected 2009 to 2011'. The
  SNAP/cho2011 dump (Feb 2009-Oct 2010) is NOT the data source."
- *Why it matters:* Vintage is a core credibility lever (the field's own critique says
  decade-old single-source results are anecdotes). By the author's own measurement the Ch.5
  five-state data extends into Aug 2011, so the global limitation understates the range of the
  data actually used in the resolution chapter. Ch.4 cites "February 2009 and October 2010" for a
  *different* source (liu2014/SNAP), compounding the inconsistency.
- *Direction:* State the measured range for the data each chapter uses (Ch.5: 2009–2011 figshare
  dump; Ch.4: its own source range) and make Ch.6's limitation cover both correctly. Reconcile
  with the number auditor (06).

**M4 · Same states carry different dataset statistics across Ch.4 and Ch.5, never reconciled.**
(Lens 6/9 — representativeness + reproducibility.)
- *Quote (Ch.4 Table `tab:courb:dataset`):* Florida "990,518 / 65,009 / 20,301"; California
  "2,535,573 / 148,314 / 36,106"; Texas "3,355,419 / 135,570 / 37,522."
- *Quote (Ch.5 Table `tab:mobiwac:datasets`):* FL "1,407,034 / 76,544 / 21,052"; CA "3,171,380 /
  169,145 / 37,090"; TX "4,089,892 / 160,938 / 38,644."
- *Why it matters:* Both are legitimate in their own chapter (different Gowalla source, min-visit
  filter 5 vs 10, vintage, and 3 vs 5 states — confirmed as-published in NORTH_STAR §4), but a
  banca member reading both chapters sees Florida as 990k *and* 1.4M with no explanation, which
  reads as an error. The frame never states that the two studies use different Gowalla
  extractions.
- *Direction:* One frame sentence (Ch.2 §2.4 or the Ch.4/Ch.5 prefaces) noting that Ch.4 and Ch.5
  draw on different Gowalla releases and filtering thresholds, so per-state counts are not
  comparable across the two chapters.

**M5 · No revisitation/repeat intuition anywhere; the trivial "predict last region/category"
anchor is absent.** (Lens 4 — floors & popularity bias; POI Pitfalls Pitfall 13; attack Q5.)
- *Evidence of absence:* grep across all six chapters for revisit/repeat/persistence/return
  yields only the Song/Cho *background* framing in §2.1 ("return to a small set of places");
  no measured repeat rate, and no persistence or per-user most-frequent baseline in any results
  table.
- *Why it matters:* In LBSN next-location, much of any Acc@K number is carried by revisitation.
  The first-order Markov region floor is high (51–72 Acc@10, Ch.5), which itself signals heavy
  revisitation, but the reader is never told what fraction of correct predictions are simply
  repeats, and there is no persistence baseline (predict the current region/category). Without it,
  a skeptic cannot separate "the model learned mobility" from "the data is repetitive."
- *Direction:* Add a persistence / most-frequent-per-user floor to the region results (or its
  prose) and one sentence giving the repeat-vs-explore split for the datasets. If revisitation is
  high, say so and frame the gain over the Markov floor as the non-trivial part.

**M6 · The metric averaging axis (per-visit vs per-user) is never stated, and per-user length skew
is extreme with no activity cap.** (Lens 8 — metric conventions; attack Q9/Q11.)
- *Quote (Ch.5 §setup-metrics):* "the fraction of test visits whose true region appears among the
  model's ten highest-scoring guesses" — i.e. per-visit (per-window) averaging.
- *Quote (Ch.5 Table 1):* max per-user length "42,300" (TX), "16,679" (FL), "14,855" (CA) against
  average length "105.8 / 66.8 / 85.5."
- *Why it matters:* With stride-1 overlapping windows and per-visit averaging, a single
  42,300-visit account contributes ~42k windows and dominates the reported mean; there is no
  per-user-averaged figure and no maximum-activity cap disclosed (only min ≥10 visits). Per-visit
  scoring is field-standard, so this is a disclosure gap rather than a methodological error, but a
  banca statistician will ask whether the headline Acc@10 reflects a typical user or a few
  hyperactive (possibly non-human) accounts. The tiny cross-seed sds show the estimate is *stable*,
  not that it is *user-representative*.
- *Direction:* State the averaging axis explicitly (per-check-in, standard in the field); note
  whether any upper activity bound was applied and, if a 42k-visit account is a real user; ideally
  add one per-user-macro-averaged robustness number so the reader knows power users do not carry
  the result.

**M7 · The transductivity audit covers only AL/AZ/FL; the three largest corpora (CA/TX/Istanbul)
are unaudited.** (Lens 3 — transductive-artifact leakage; attack Q3.)
- *Quote (Ch.5 §setup-windows):* "at Alabama, Arizona, and Florida … This measurement covers the
  visits whose places appear in training (67 to 87 percent)."
- *Why it matters:* The A4 numbers are faithful (verified against `A4_RESULTS.md`), and the
  in-coverage caveat and unseen-places residual are stated honestly. But the whole-corpus
  representation is trained once on all six datasets, and the audit's null is measured only on the
  three smallest US states. CA (8,501 regions, 3.2M check-ins), TX (6,553 / 4.1M) and Istanbul
  carry the most opportunity for transductive information flow and are exactly where the largest
  region gains are claimed (+2.10 TX, +2.19 CA). The text does not flag that the audit's
  reassurance does not extend to the cells doing the heaviest lifting.
- *Direction:* Add one sentence scoping the audit to AL/AZ/FL and naming CA/TX/Istanbul as
  unaudited (the repo already lists this as author TODO in a hidden comment); or run the audit
  there before the banca. Also note (from `A4_RESULTS.md`, not in the text) that the audit is
  non-deterministic — a re-run gave category +0.88 pp vs the committed +0.29 — so "at most a third
  of a point" is one draw, not a stable bound. → coordinate with stats/leakage skeptic (09).

### MINOR

**m1 · Cross-convention number slip: Ch.6 quotes 64.54 for the AL joint category cell that Ch.5
Table 3 reports as 64.51.** (Concordance; → number auditor 06.)
- *Quote (Ch.6 §6.2):* "56.82 for the dedicated model … and 64.54 for the joint model." *Ch.5
  Table `tab:mobiwac:results`:* AL Joint "64.51±0.09."
- *Why it matters:* `JOINT_BEST_RESULTS.md` shows 64.54 = diag-best, 64.51 = joint-best(deploy).
  Ch.5 reports the joint-best lane; Ch.6 imports the diag-best value for the same cell. Within one
  sd, but the same quantity should carry one number across chapters.
- *Direction:* Use the joint-best value (64.51) in Ch.6 to match Ch.5's reported convention.

**m2 · "Next-POI Prediction" persists 17× in Ch.3 and 16× in Ch.4 with no in-chapter bridge to
the canonical "next category."** (Lens 7 / attack Q12 — three-task blur.)
- *Quote (Ch.4 §related):* "HMT-GRN combines Next-POI Prediction and prediction of its geographic
  region." Both chapters *define* the term in-body ("predict the category of the next POI"), and
  Ch.1 §1.3 bridges it narratively, but neither preface says "this chapter's 'Next-POI Prediction'
  is the canonical next-category task."
- *Why it matters:* The label reads as next-*place* to a POI examiner scanning tables and
  headings, and banca members read reproduced chapters standalone. The design (reproduced chapters
  keep the paper's terms, the frame bridges) makes this defensible, so it is minor, not major.
- *Direction:* One sentence in each of the Ch.3/Ch.4 prefaces mapping "Next-POI Prediction" to the
  dissertation's "next category" would close the residual blur risk cheaply.

**m3 · Ch.2 §2.4 forward-promises MRR, which Ch.5 never reports.** (Metric convention.)
- *Quote (§2.4):* "mean reciprocal rank accompanies it where the joint comparison needs a
  rank-sensitive figure." Ch.5 reports Acc@10 only; MRR appears in no results table.
- *Direction:* Either drop the MRR promise from §2.4 or report MRR in Ch.5; an unfulfilled metric
  promise invites the question at the defense.

**m4 · Query-time information symmetry is never stated.** (Lens 7 — attack Q7, closing clause.)
- *Observation:* No chapter states whether the target visit's timestamp is available to the model
  at prediction time (it plausibly is, since category/region are predicted for a known next
  time-slot, but the reader is not told, and it affects whether the task is "predict the next
  visit" or "predict the visit at time t+1").
- *Direction:* One sentence in Ch.5's problem statement clarifying what is known about the target
  at query time.

**m5 · Istanbul vintage is unstated while Gowalla vintage is a named limitation.** (Lens 6.)
- *Observation:* Ch.6 limitation 5 caps non-US evidence at "a single city, Istanbul," but the
  Istanbul (Massive-STEPS) collection date is never given, while Gowalla's is a headline
  limitation. For symmetry and honesty the reader should know the Istanbul vintage too.
- *Direction:* State the Massive-STEPS/Istanbul collection period where the vintage limitation is
  discussed.

---

## What holds / what reads well (do NOT touch)

- **Ch.5 leakage hygiene is exemplary.** The overlap-cannot-leak sentence ("a test user's visits
  never appear in training"), the A4 audit with its 67–87% in-coverage caveat and unseen-places
  residual, the per-fold region-transition prior with the 13–27 pp historical-inflation record,
  and "our joint and dedicated models do not use this prior" are exactly the disclosures the
  field's critique canon asks for. Verified faithful to `A4_RESULTS.md`.
- **Baseline provenance is disclosed at the point of comparison** — POI-RGNN re-implemented,
  HMT-GRN region-native and explicitly "not a reproduction of the complete published system,"
  STAN partial folds (TX 4/5, CA 2/5) and ReHDM single-seed/own-protocol all footnoted. This is
  the fairness discipline most next-POI papers lack.
- **Verbs are bound to tests throughout**: "outperforms" only with the paired superiority test,
  "matches" only with TOST within a two-point margin, AZ never upgraded from a match to a win, and
  the region scaling read honestly as a trend ("we read the trend across the points rather than as
  a precise law").
- **Istanbul external validity is framed correctly**: "The comparable quantity is the gain over
  the ceiling, not the absolute Acc@10." This is the right way to use a non-US dataset and should
  be preserved verbatim.
- **The three tasks are kept formally distinct** and "we do not predict the exact next place" is
  stated early (Ch.1 §1.1, Ch.2 §2.1, Ch.5 problem). The label-cardinality table (520–8,501
  regions vs 7 categories) is present and no cross-cardinality Acc@K comparison is implied.
- **Ch.4's split-protocol disclosure** ("by sample rather than by user, a weaker protocol") is the
  honesty device the arc needs; keep it exactly as written (it is what M2 asks Ch.3 to match).
- **The floors are present and protocol-matched**: majority-class ~7% macro-F1 (internally
  consistent with Food ~33%), and the first-order Markov region floor (51–72 Acc@10) computed
  "under our windows and folds."

## Unstated defenses (facts the repo holds but the text does not carry)

- **A4 non-determinism.** `A4_RESULTS.md` records the audit is non-deterministic (a re-run gave
  cat +0.88 pp vs the committed +0.29) and that the category axis is a POI-level proxy on the
  in-coverage subset. Ch.5 states the proxy caveat but not the run-variance; the "at most a third
  of a point" phrasing implies a stable bound the repo does not claim.
- **CA/TX transductivity direction.** The repo TODO ("extend to CA/TX/Istanbul … non-blocking; the
  gate is on null at AL+AZ+FL") shows the unaudited-large-corpora gap is known; the text could name
  it as scoped future work rather than leave it silent (M7).
- **Cross-chapter dataset provenance.** NORTH_STAR §4 documents that Ch.4 (SNAP/liu2014, min-5) and
  Ch.5 (figshare CC0 dump, min-10) use different Gowalla releases — the fact that reconciles M4 —
  but no chapter tells the reader this.
- **AZ ceiling sensitivity.** A hidden Ch.5 comment records the AZ dedicated category ceiling
  (56.43) carries a pending 2-seed top-up that could raise it to ~57.0 (shrinking Δcat to ~+8.7);
  disclosed on-request per author policy, not in the text. Rule-clean per the pre-registered
  estimator, so not a finding — noted so the author knows the caveat exists if a reviewer probes.

## Out-of-scope handoffs (one line each; not my lens)

- **Number auditor (06):** the 64.54/64.51 convention slip (m1); the Gowalla date range
  reconciliation (M3); confirm the audited CoUrb win-counts/pp-gains (15/21 + 1 tie; +20.2…+22.0)
  match `slides/judge_feedback.md` — Ch.4 uses the audited numbers, which is correct.
- **Citation auditor (05):** CBIC `ERRATA.md` #1 marks `capanema2023poirgnn` (POI-RGNN) as
  "[VERIFY at adaptation] — the exact record could not be resolved via OpenAlex this session." That
  reference is load-bearing twice — the sole support for "next category as an end target" in Ch.2
  §2.1 and a Ch.5 category baseline — so its resolution matters for domain claims; confirm the
  record is opened before the bib freezes.
- **Stats/leakage skeptic (09):** A4 run-variance and the POI-proxy scope (M7 tail); the n=4
  per-dataset pairing for TOST/superiority power; the per-visit averaging power question (M6).
- **Style/concordance (03/04):** the em-dash-free / canonical-name checks are not my lens; I did
  not audit them.
- **Translation fidelity (08):** Ch.4 is a PT→EN reproduction; I reviewed its science, not its
  translation faithfulness.

## Open questions only the author can answer

1. Is the CBIC-era Florida split user-disjoint or sample-stratified, and does any surviving
   artifact establish it? (Decides the M2 wording — disclosure vs residual flag.)
2. Is the 42,300-visit Texas account a real user, and was any maximum-activity cap applied at any
   stage? (Decides whether M6 needs a data-hygiene sentence in addition to the averaging-axis
   disclosure.)
3. Will CA/TX/Istanbul get a transductivity audit before the banca, or should M7 be written as a
   scoped limitation + future-work item?
4. Should the vintage limitation state the measured 2009–2011 range for the Ch.5 figshare data
   (M3), given the published articles say 2009–2010?


**Read (session start):** persona file 11; reviewers/README common protocol; CLAUDE.md; NORTH_STAR §1-§6; GLOSSARY.md. (Persona "read first" order followed.)

---

## Working notes (raw findings as I go — reorganized into the output contract at the end)

<!-- appended below as review proceeds -->

### RAW NOTES BY LENS (pre-verification)

**Chapters read in full:** 2_fundamentals (551 l), 5_mobiwac (627 l), 4_courb (351 l), 3_cbic (365 l),
1_introduction (249 l), 6_conclusion (159 l). Build: main_defense.pdf 87pp.

**Lens 1 (split legitimacy):**
- Ch.5 GOOD: "We split by user with stratified five-fold cross-validation, so all of a user's windows fall in the same fold and overlap cannot leak: a test user's visits never appear in training." Overlap-cannot-leak argument explicit.
- Ch.4 GOOD (disclosed): preface + results both say sample-stratified, weaker than Ch.5, userid may span train/val.
- Ch.3 GAP: §5.1 "all experiments ... using a 5-fold cross-validation methodology" — split AXIS NOT STATED (user vs sample). GLOSSARY flags "verify from CBIC codebase before asserting". Arc honesty story is protocol-strengthening → CBIC's axis must be named. MAJOR.

**Lens 2 (window/seq leakage):** Ch.5 EXEMPLARY — min-len 10, stride-1, overlap, padding, end-of-history dedup all disclosed + overlap-cannot-leak. Ch.3/4 non-overlapping, min-len 5 (<5 dropped). Per-chapter disclosed. Note min-len differs 5 vs 10 across chapters (fine, each states own).

**Lens 3 (transductive):** Ch.5 §setup-windows "Integrity" para EXEMPLARY: label-free + A4 audit (region −0.33..+0.01, category 0.00..+0.29 at AL/AZ/FL) + in-coverage caveat (67–87% places seen) + unseen-places residual + per-fold transition prior w/ 13–27pp historical inflation record + prior only in HMT-GRN. BUT audit only AL/AZ/FL; CA/TX/Istanbul (largest corpora = most leak opportunity) UNAUDITED. Honest scope stated, residual on largest sets not flagged. MAJOR-soft.

**Lens 4 (floors + popularity):**
- Majority-class floor STATED (~7% macro-F1). Markov-K (cat) + first-order Markov region floor STATED, protocol-matched ("our windows and folds"), region floor 51–72 Acc@10. GOOD.
- MISSING: persistence (predict last region/category) + per-user most-frequent baseline. No repeat-vs-explore statistic anywhere. High Markov region floor (51–72) itself signals revisitation carries much of the number, but reader gets no repeat-rate intuition. Persona attack Q5 + Pitfall 13. MAJOR.

**Lens 5 (baseline fairness):** Ch.5 EXEMPLARY provenance — POI-RGNN (re-impl from published arch/hparams), HMT-GRN (region-native, our folds, keep MT skeleton + per-fold prior, drop beam/graph, "not a reproduction of the complete published system"), STAN (re-impl, own embeddings/seq, partial folds TX 4/5 CA 2/5 disclosed in footnote), ReHDM (own protocol, single-seed TX/CA disclosed). All asymmetries disclosed at point of comparison. GOOD.
- MINOR: conclusion "at least 4 Acc@10 over strongest region reference" — tightest point is Joint 69.70 vs ReHDM 65.38 at AL = +4.32, but ReHDM is CROSS-PROTOCOL. Disclosed as "own protocol" but aggregate claim leans on it.

**Lens 6 (staleness):**
- Gowalla vintage: Ch.6 lim1 "2009 and 2010"; Ch.4 "February 2009 and October 2010". BUT Ch.5 hidden comment: figshare dump ETL consumes spans "2009-01-21 .. 2011-08-16" measured on parquet; "SNAP/cho2011 (Feb 2009-Oct 2010) is NOT the data source". So Ch.5 data runs into Aug 2011 → stated 2009–2010 limitation MISDESCRIBES Ch.5 data. MAJOR (verify + reconcile). Cross-ref number auditor.
- Istanbul external validity FRAMED CORRECTLY: "The comparable quantity is the gain over the ceiling, not the absolute Acc@10". EXCELLENT credibility signal.
- Istanbul vintage unstated (staleness lim = Gowalla only). MINOR.

**Lens 6b / Lens 9 (cross-chapter dataset consistency):** SAME STATE, DIFFERENT STATS across chapters, unreconciled:
  FL: Ch.4 990,518 ck / 65,009 POI / 20,301 users  vs  Ch.5 1,407,034 / 76,544 / 21,052.
  CA: Ch.4 2,535,573 / 148,314 / 36,106  vs  Ch.5 3,171,380 / 169,145 / 37,090.
  TX: Ch.4 3,355,419 / 135,570 / 37,522  vs  Ch.5 4,089,892 / 160,938 / 38,644.
  Cause: different Gowalla source (SNAP/liu2014 vs figshare), min-visits 5 vs 10, vintage, 3 vs 5 states. NOT reconciled in text; a reader/examiner sees FL=990k (Ch.4) vs 1.4M (Ch.5). MAJOR.

**Lens 7 (formulation comparability):**
- Three tasks kept distinct throughout; "do not predict exact next place" stated early (Ch.1 §1.1, Ch.2 §2.1, Ch.5 problem). GOOD.
- Label cardinality TABLED (Ch.5 Table 1 Regions 520–8,501; category=7). GOOD.
- Region construction (census tract/mahalle) justified ("neighborhood, not radio cell"; official units vs grid). GOOD.
- No cross-cardinality Acc@K comparison: external-validity section forbids it explicitly ("gain over ceiling, not absolute Acc@10"). GOOD.
- Co-equal region novelty claim scoped ("to our knowledge", fine-grained region vs auxiliary grid; DRRGNN + sun2025kgtb named/distinguished). GOOD.
- Query-time info symmetry: text SILENT on whether target timestamp is a model input. MINOR.

**Lens 8 (metrics):**
- macro-F1 for imbalanced category + majority floor beside: GOOD (Ch.5 + fund §2.4 "plain accuracy inflated by Food ~third").
- Acc@10, K=10 motivated operationally (10-region shortlist / anticipatory set). OOD counted as miss, DEFINED. GOOD.
- HMT-GRN scored on FRIENDLIER denominator (visits w/ region in training, >99%) vs joint counts OOD as miss — DISCLOSED w/ subset size (>99%). Acceptable.
- PER-SAMPLE (per-window) averaging ("fraction of test visits") + EXTREME per-user skew (Table 1 Max len up to 42,300 at TX; 16,679 FL; 14,855 CA; avg 105.8 TX). Overlapping stride-1 windows weight users by visit count → power users dominate the reported mean. Undisclosed; no per-user-averaged number. Also 42,300-visit "user" likely non-human/venue acct — was any activity cap applied? MAJOR.
- MRR: fund §2.4 introduces MRR ("accompanies Acc@10 where the joint comparison needs a rank-sensitive figure") but Ch.5 reports no MRR. MINOR (drop forward-ref or report it).

**Lens 9 (reproducibility):**
- Code+data URLs (Ch.5 GitHub+figshare; Ch.3/4 GitHub), seeds (4; n=20=4×5), configs "in released code". GOOD.
- BLOCKER: Ch.3 §5.1 dataset stats UNFILLED — literal "[$N_{users}$; VERIFY: recompute per ERRATA.md]" ×3 in rendered text. Results chapter with no N + visible VERIFY brackets. Uninterpretable + build defect.

**Frame (Ch.1/6):**
- Ch.1 §1.1 "93 percent" correctly scoped to "next location". Ch.2 §2.1 OVERGENERALIZES same 93% as "the reference point against which any predictive model should be read" / "ceiling" — contradicts §2.4's correct scoping ("not a ceiling on ... category macro-F1 or region ranking"). §2.1 anchors reader on a ceiling that does not apply to the dissertation's actual tasks. MAJOR (harmonize §2.1↔§2.4).
- Ch.1 arc verbs bound to tests. GOOD. Task-pair evolution named plainly. GOOD.
- Ch.6 §6.2 capacity-matched baseline (frame-level, disclosed as post-submission). Quotes joint AL category "64.54"; Ch.5 Table 3 = 64.51±0.09. Within 1sd but should be same quantity. MINOR (concordance; number auditor).
- Ch.6 CoUrb "20.2 to 22.0 pp" = audited numbers (not stale 76%/+20-24). GOOD. MobiWac "5.3 to 9.4" ≈ +5.33..+9.35. GOOD.

### NUMBERS TO VERIFY AGAINST SOURCES OF TRUTH (reproduce-first, README §10)
1. Ch.5 Table 3 joint/dedicated cells vs RESULTS_BOARD.md (spot: AL cat 64.51, the 64.54 in Ch.6).
2. A4 audit numbers (−0.33..+0.01 reg; 0.00..+0.29 cat) vs A4_RESULTS.md.
3. Gowalla date range 2009–2011 claim (Ch.5 hidden comment) — the vintage finding.
4. CoUrb dataset table (990,518 etc.) is as-published (cross-chapter inconsistency, not my typo).
5. CBIC placeholder truly unfilled in built PDF.

### VERIFICATION RESULTS (reproduce-first, done)
- **CBIC placeholder — CONFIRMED BLOCKER.** main_defense.pdf p.35 renders verbatim: "This subset comprises a
  total of [N_users; VERIFY: recompute per ERRATA.md] users, [N_poi; VERIFY: recompute per ERRATA.md] unique
  Points-of-Interest (POIs), and [N_checkins; VERIFY: recompute per ERRATA.md] check-ins." Raw VERIFY scaffolding
  visible in the compiled defense build. Ch.3 presents Florida F1 tables with NO dataset size stated.
- **A4 audit — VERIFIED FAITHFUL.** A4_RESULTS.md: AL reg −0.33/cat +0.29@66.8%; AZ reg +0.01/cat +0.27@71.9%;
  FL reg −0.12/cat +0.00@86.9%. Ch.5 text ("region −0.33 to +0.01; category 0.00 to +0.29"; "67 to 87 percent")
  matches exactly. Scope AL/AZ/FL-only confirmed; CA/TX/Istanbul unaudited (Ch.5 states this honestly).
  NOTE (domain, not a text error): A4_RESULTS carries a RUN-VARIANCE caveat NOT in the dissertation — the
  per-fold train-only substrate is non-deterministic, "~±0.5–0.6 pp on cat", and an AL re-run gave cat +0.88 pp
  vs the committed +0.29. So "at most a third of a point" is the single committed draw, not a stable bound; a
  re-run exceeded it. Also cat is a POI-level proxy on the in-coverage subset, which Ch.5 does state. → stats
  skeptic (09) territory; I flag it as an unstated caveat, low severity for my lens (direction/scope honest).
- **Ch.6 "64.54" vs Ch.5 Table 3 "64.51" (AL joint category) — CONFIRMED CROSS-CONVENTION SLIP.**
  JOINT_BEST_RESULTS.md: AL joint category diag-best=64.54, joint-best(deploy)=64.51. Ch.5 Table 3 reports the
  JOINT-BEST lane (64.51) per its own provenance comment; Ch.6 §6.2 capacity-baseline para quotes 64.54 (the
  diag-best value) as "the joint model" (and 56.16 vs "64.54 for the joint model"). Two conventions mixed across
  chapters for the SAME cell. Within 1 sd but should be one number. → number auditor (06); I flag as concordance.
- **Gowalla vintage — CONFIRMED TEXT INCONSISTENCY (author's own note).** Ch.5 hidden provenance comment
  (author-written): the figshare dump the Ch.5 ETL consumes was MEASURED on the parquet as 2009-01-21 .. 2011-08-16
  → intended phrasing "collected 2009 to 2011"; and explicitly "The SNAP/cho2011 dump (Feb 2009-Oct 2010) is NOT
  the data source." YET Ch.6 limitation 1 renders "collected in 2009 and 2010" and Ch.4 "February 2009 and October
  2010". By the author's own measurement the Ch.5 five-state data runs into Aug 2011, so the stated vintage
  limitation under-states the range for the data actually used in Ch.5. (Ch.4 uses a different source, may be
  2009–2010 correctly — but Ch.6's limitation is global and covers the Ch.5 datasets.) The vintage limitation is a
  core credibility lever (Lens 6); it currently misdescribes the Ch.5 data per the repo's own provenance. MAJOR.
- **Majority floor ~7% — internally consistent** (Food ~33% → macro-F1 of always-Food ≈ 7.1%). Not a finding.
- **Cross-chapter dataset stats — confirmed both as-published.** NORTH_STAR §4 confirms CoUrb FL row
  990,518/65,009/20,301 is the genuine published number; Ch.5 FL 1,407,034 is from the board. Both legitimate in
  their own chapter; the FRAME never reconciles that Ch.4 and Ch.5 use different Gowalla extractions/filters/vintage.


