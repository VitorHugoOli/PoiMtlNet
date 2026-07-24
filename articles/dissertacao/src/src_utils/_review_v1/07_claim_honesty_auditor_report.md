# 07 · Claim & Honesty Auditor — Report on dissertation v1

> Persona 07 (claim-registry gate G2, rules C1–C4 + WRITING_LAW §3 honesty law).
> Scope: all six chapters (`src/chapters/1_introduction.tex` … `6_conclusion.tex`),
> appendices (`apx_a`/`apx_b`/`apx_c`), and front matter (`src/0_main.tex`).
> Read-only. Findings quote the licensed form; they never rewrite.
> Status: IN PROGRESS (written incrementally; a restart must not lose work).

---

## Working reference — the licenses (quoted from source, do not re-derive)

### MobiWac / Ch.5 CAN-say (PAPER_PLAN §3 + CLAUDE §2/§3)
- **Category beats ceiling everywhere:** Δ = AL +7.69 / AZ +9.35 / FL +5.33 / CA +6.45 / TX +7.45 / Ist +8.58 (vs n=20 best-vs-best ceilings; per-cell Holm m=6 all reject, worst adj p=1.0e-06). Joint cat cells: Ist 63.32 / AL 64.51 / AZ 65.79 / FL 79.84 / TX 77.24 / CA 77.05.
- **Region:** outperforms (paired Wilcoxon superiority, 90% CI>0) at **Ist +0.19, FL +0.71, TX +2.11, CA +2.20**; **matches** (TOST non-inferior, δ=2pp) at **AL −0.41, AZ 0.00**. Joint reg cells: Ist 75.35 / AL 69.70 / AZ 59.46 / FL 77.41 / TX 67.06 / CA 65.69. **NEVER upgrade AZ (0.00, CI straddles zero).**
- **Check2HGI vs HGI category margin:** +29.31 / +27.63 / +39.62 / +37.95 / +37.47 (AL/AZ/FL/CA/TX); two bands ~+28–29 (small) and ~+37–40 (large); Istanbul dk_ovl +28.09. HGI cat ≈ 0.46–0.52× Check2HGI cat. On next-region the two representations are within ~1.6–3.1 pts (HGI slightly ahead): representation benefit is **category-only**.
- **Markov region floor (text only, §6.2):** joint exceeds stride-1 Markov-1 floor (Acc@10 51–72 across six) by 4.9–10.3 pts.
- **n = 20** = 4 seeds × 5 folds; Holm across six; user-disjoint CV; A4 leak audit null (region ≤0.33 pp, category ≤0.29 pp at AL/AZ/FL).
- Scaling: region gain rises with region count **across the five U.S. states**; Istanbul (fewest regions) also positive.
- Cascade (CSLSL pattern) = **tie at equal cost** (Δ ≤ 0.02), framed as defense not win; it is a coupling-topology variant of *our own* model, NEVER a CSLSL re-implementation.
- Verdict verbs: **"outperforms"** (paired Wilcoxon) / **"matches"** (TOST); never "beats"/"wins"/"ties"/"Pareto".

### MobiWac must-NOT-say
"beats region everywhere"/"Pareto-dominates"; "ties" on region; "cost grows with region count"/cardinality-cost/TX −2.4; headline a two-model composite / per-task routing / two-substrate; old region numbers 7–17 or pre-2026-05; "trivial"/"padding" for dropped overlap windows; "we beat STAN-on-our-representation" (stl_hgi, sits above us at AL 70.35 vs 69.81); "we beat the cascade"; "we ran/benchmarked CSLSL".

### CBIC / Ch.3 (published, time-indexed)
"MTL does not help" = conclusion **of the time, for that configuration**. Nash-MTL "consistently better" predates the solver-bug discovery (NashMTL collapse to [1,1]) — do NOT amplify in the frame.

### CoUrb / Ch.4 (published, time-indexed)
Audited numbers: **15/21 strict wins + 1 technical tie** (NOT "16/21 (76%)"); per-state category means **+20.2…+22.0 pp** (NOT "+20–24 pp"). Split is **sample-stratified, not user-disjoint** (verified firsthand — plain StratifiedKFold, userid dropped) — say so plainly. Contribution note for Vitor mandatory. Required floor sentence (approved): "this chapter isolates the representation effect with MTLNet as its only baseline; it does not revisit the MTL-versus-single-task question, which Chapter 5 reopens."

### BRACIS (C4 containment)
Appears ONLY as "an earlier unpublished iteration"; its region-cost claim (MTL pays 7–17 pp on region) appears ONLY as corrected history (fp16 artifact + older protocol), never live.

### Story spine (C2 — frame arc claims must match NORTH_STAR §6)
Signed-off additions to the Intro arc paragraph (AVAL rounds 1–2, 2026-07-22): task-pair evolution named plainly; three-legged task-choice defense (leg 2 comparative form stays [VERIFY]); "unnatural" not "incoherent"; N2 caution form only (never "CBIC's future work called for better representations"); mechanism sentence as hypothesis. Ch.6: task-pair confound concession; N3 negative-transfer-reversal beats with cosine +0.001 full scope (four seeds, three of six datasets, directional conflict only, this pair not a general rule); "sharing stopped hurting" never "tasks teach each other"; never credit parameter count.

---

## FINDINGS (incremental)

### Front matter (0_main.tex) — Resumo/Abstract pair
- Abstract (EN) and Resumo (PT) are a claim-parity pair. Verbs bound: "outperforms … at all six … by 5.3 to 9.4 macro-F1 points" (category); region "outperforms at four of six … and statistically matches, with non-inferiority within a two-point margin (TOST), at the other two." PT mirror: "supera … por 5,3 a 9,4 pontos"; region "supera em quatro dos seis … e equipara-se estatisticamente … (TOST), nos outros dois." HOLDS. AZ not upgraded (folded into "the other two", not named as a win). Headline 5.3–9.4 matches CAN-say (+5.33…+9.35). Null result stated as "a null result reported as a finding" / "um resultado nulo relatado como um achado" — honest, not rushed. Both carry [NEEDS SIGN-OFF] comments (author-owned).

### Ch.1 Introduction — arc narrative (C2 spine conformance)
HOLDS against NORTH_STAR §6 signed-off spine. Verified present:
- Task-pair evolution named plainly (§1.1 "A fourth task also appears…"; §1.2 "The task pair therefore evolved…") — not narrated as one fixed experiment. ✓ (spine addition a)
- Three-legged task-choice defense (§1.1): "chosen for what a mobility-aware service can act on, and both are established end targets in the literature on the way to the harder next-place problem." Leg 2 in fallback form (established end targets + feeds next-place), NOT the [VERIFY] comparative "most-cited" form. ✓ (spine addition b)
- "less natural fit" (§1.2), NOT "incoherent"/"unnatural-as-incoherent". ✓ (spine addition c) — note spine said "unnatural"; chapter uses "less natural fit", same polarity, acceptable.
- N2 caution form (§1.2): "tested the representation explanation first, as the cheapest controlled test among the three" — NOT "CBIC's future work called for better representations", no foresight framing. ✓ (spine addition d)
- Mechanism sentence as hypothesis (§1.2): "That observation is the hypothesis the final study tests." ✓ (spine addition e)
- Region verbs (§1.2, §1.6 Practical): "four of six, with statistical non-inferiority within a two-point margin (TOST) at the other two" / "outperforming or remaining non-inferior". ✓ AZ not upgraded.
- Compute-cost honesty (F3 guard): "cost more to train" stated for CBIC (§1.2); operational wish framed as "single model to maintain … single forward pass", never lower compute. ✓

### Ch.2 Fundamentals — HOLDS on claims
- MTLnet null time-indexed: "does not outperform the dedicated single-task models, a result that holds for that configuration" (§2.3). ✓ Nash-MTL NOT amplified (named only in the balancer list, no benefit claim). ✓
- Verb law: §2.3 ledger notes all "beat" removed → "outperform". Region-as-end-target scoped to the multi-task co-equal setting (single-task region prediction acknowledged via zhu2022drrgnn). ✓
- §2.5 result wording bound to tests: "by paired superiority tests, outperforms … on the next category everywhere … and on the next region at four of six datasets, and matches … within a two-point margin, by non-inferiority testing, at the other two." ✓ AZ/AL not upgraded; forward-points to Ch.5.
- song2010limits 93% scope-corrected (§2.1, §2.4): explicitly "not … a ceiling on seven-class category macro-F1 or on region ranking"; dedicated single-task model is the operative ceiling. ✓ (honesty: number carries its reference-point scope)
- Lineage table: Check2HGI + joint model status "submitted, under review" in caption. ✓
- OPEN QUESTION Q1 (see below): "roughly a third" Food figure provenance.

### Ch.6 Conclusion — HOLDS on claims; two items to confirm
- §6.1 CoUrb: "+20.2 to 22.0 percentage points across the three states tested" = audited CoUrb numbers (NORTH_STAR §4), NOT the stale "+20–24 pp". ✓
- §6.1/§6.2 MobiWac verbs bound to tests, four-of-six named (Ist/FL/CA/TX), AL/AZ matched via TOST, never upgraded. ✓
- §6.2 capacity-matched baseline (POST-SUBMISSION frame analysis, licensed by D1 contract + NORTH_STAR §6): numbers trace to storyline/audit/capacity_baseline_experiment.md §5.3 — best arm 56.16 (±1.88 not quoted in prose), dedicated ceiling 56.82, joint 64.54, AL 4.2M vs 0.6M published width, partial CA n=15/20 "same direction". VERIFIED against source. Reading (i) reported honestly per contract §3.2 (unfavorable outcome would have been reportable; favorable one is not overstated). ✓
- §6.2 freeze control: "at the three datasets where the control was run (Alabama, Arizona, Florida)" = NORTH_STAR §6 N3 scope. ✓ Cited as a Ch.5 finding (licensed).
- §6.2 mechanism: cosine +0.001 full scope travels verbatim ("over four seeds on three of the six datasets … a finding for this pair of tasks rather than a general rule"). "sharing stopped hurting" present; "does not come from the region task teaching the category task" — never "tasks teach each other"; parameter count never credited (disclosed as cost). ✓ (NORTH_STAR §6 N3 beats)
- §6.3 limitation 6 (task-pair confound) + §6.4 future-work fixed-pair ablation present = signed-off 2026-07-22 additions. ✓

### Ch.3 CBIC — HOLDS (reproduced published text, correctly time-indexed)
- Preface time-indexes BOTH load-bearing conclusions: the null ("conclusions of the time, for the configuration studied here: with a place-level embedding and hard parameter sharing, multi-task learning did not consistently improve on the dedicated single-task models") AND the Nash-MTL preference ("likewise a conclusion of the time, weakened by a later finding about the optimizer implementation, and the following chapters do not rely on it"). ✓ (NORTH_STAR §4 Ch.3 claim-discipline)
- Null written with care, not rushed: three candidate explanations laid out in §conclusion. ✓ (checklist 9)
- The reproduced "Nash-MTL consistently yielded a better overall performance" (§sec:cbic:nash) is preserved verbatim by design (Appendix B preservation note) with the preface carrying the of-the-time caution — NOT amplified in frame prose. ✓
- CBIC-era task naming ("Next-POI Prediction" = the label is the next POI's category) is the chapter's own published usage; the frame (Ch.1 §1.1, Ch.2 §2.1 mapping) canonicalizes it. ✓
- Dataset placeholders render as visible "[VERIFY: recompute per ERRATA.md]" markers (not fabricated); Appendix B row 5 marks this "Pending, Not invented." Correct fail-closed handling — see MINOR-1.

### Ch.4 CoUrb — HOLDS (audited numbers, time-indexed, ownership disclosed)
- Audited numbers used throughout, NOT the stale published ones: "20.2 to 22.0 percentage points" (not "20–24"); "15 of the 21 ... with one additional technical tie" (not "16/21 (76%)"). Present in §intro, §results, §conclusion, and Appendix B. ✓ (NORTH_STAR §4 Ch.4)
- Ownership honesty (checklist 10): preface states "Tarik S. Paiva is the first author ... the author of this dissertation is the second author, presented the paper at the workshop, and is the first author of the baseline model MTLNet." Also in Ch.1 §1.5. ✓
- Protocol honesty: preface + §experimental-setup both state the split "is stratified by sample, not by user, so the check-ins of one user may appear in both training and validation; Chapter 5 adopts a stricter user-disjoint protocol." Weaker protocol disclosed plainly, strengthens the arc. ✓ (VERIFIED FIRSTHAND per NORTH_STAR §4, UW-3 closed)
- Required floor sentence present verbatim in preface: "This chapter isolates the representation effect with MTLNet as its only baseline; it does not revisit the MTL-versus-single-task question, which Chapter 5 reopens." ✓ (NORTH_STAR §6 Ch.4 approved Item 6)
- Verdict verb: "outperforms the baseline in most scenarios" (Appendix B records the "wins"→"outperforms" fix). ✓

### Ch.5 MobiWac — HOLDS (whitelist-exact; the highest-risk chapter, clean)
- **Category:** "outperforms ... on every dataset ... by +5.33 to +9.35 macro-F1 (smallest at Florida, largest at Arizona, and +8.58 at Istanbul)" = CAN-say exactly. Table III joint cat cells 63.32/64.51/65.79/79.84/77.24/77.05 match the whitelist memory-aid. ✓
- **Region verbs bound to tests:** Table III uses ↑ (superiority) at Ist/FL/TX/CA, ≈ (TOST non-inferior) at AL/AZ; the two matched cells are NOT bolded. Prose: "outperforms ... at Florida, Texas, California, and Istanbul, and stays a non-inferior match (TOST, ±2 pp) at Alabama and Arizona." ✓
- **AZ never upgraded** (the cardinal rule): "At Arizona, the interval is centered on zero, so we report a match, not a gain" (§results-part2). AL handled honestly: "the whole interval lies below zero, a small but statistically significant deficit, still well within the two-point margin." ✓
- **Scaling scoped to the five U.S. states:** "Across the five U.S. states, the region gain rises with region count"; "region count and corpus size co-vary here, so we read the trend across the points rather than as a precise law." ✓ (no over-claimed law)
- **Cascade = defense, not win:** "We read this as a defense of the parallel design, not a claim that we outperform the cascade"; "we test the cascade inside our own model rather than re-implementing those systems." Two qualifications present (parallel-tuned recipe; form fixed in advance). ✓ (decisions ledger cascade framing)
- **Never-cite (C3) absent:** STAN faithful (AL 60.72/AZ 49.86, not the 34.46/38.96 v4-collapse); ReHDM v4 row (69.33/65.38/53.00/64.49/48.81/50.26, not the v2 66.06/54.65/65.68); no HMT-GRN 62.37 outlier (Table III AL = 57.05); no fp16/bf16 VOID cells. ✓
- **stl_hgi / "beat STAN-on-our-representation" absent.** ✓
- **Compute-cost honesty (F3):** "the joint model has about 4.2 million parameters at Alabama against 1.1 million for the two dedicated models combined (5.2 against 2.0 at California) ... What the single model provides is operational rather than arithmetic." Never claims lower cost. ✓
- **Hygiene / evidence-guard (checklist 8) intact:** leak-audit prose restored (§setup-windows "Integrity of the representation", three grounds, region prior "built per fold from training data only" after "an earlier whole-dataset version inflated region accuracy by 13 to 27 points"); per-step hygiene sentences present; STAN partial-fold (†) and ReHDM single-seed (‡) disclosures kept; transductive limitation stated. ✓
- **Freeze control** reported as a finding, scoped: "the full category gain survives at Alabama, Arizona, and Florida"; "We report this attribution as a finding, not a hypothesis." ✓
- **Status wording:** "submitted to MobiWac 2026, under review" (preface + §intro). Never "published/accepted." ✓
- **N3 cosine +0.001 full scope** travels in §related-mtl: "averages +0.001 across training (four seeds each on three of our six datasets, per-dataset means within ±0.003) ... a finding for this pair of tasks, not a general rule." ✓

---

## TOP 3 FINDINGS

1. **[MAJOR] Cross-chapter numeric blur — the AL joint-model next-category cell reads 64.51 in Ch.5 but 64.54 in Ch.6** (joint-best vs diagnostic-best convention leak; N5 "the joint-best vs diagnostic-best distinction must never blur").
2. **[MAJOR] Data-vintage inconsistency — Ch.6 limitation 1 states "2009 and 2010", but Ch.5's own measured data provenance is 2009–2011** for the five-state datasets, and explicitly says the 2009–2010 range "is NOT the data source."
3. **[MINOR] Ch.3 CBIC dataset placeholders unresolved** ($N_{users}$/$N_{poi}$/$N_{checkins}$ render as visible `[VERIFY]` markers) — correctly not fabricated, but a number-completion blocker before the final build.

---

## RANKED FINDINGS (quote + location + rule + suggested direction)

### MAJOR-1 — AL joint-category value blurs joint-best (64.51) and diagnostic-best (64.54)
- **Ch.5** `chapters/5_mobiwac.tex:479` (Table III, joint-best convention): AL Joint category = `\textbf{64.51}\sd{0.09}`. The chapter states Table III is joint-best and that diag-best "would change every joint result by at most 0.06 (category)."
- **Ch.6** `chapters/6_conclusion.tex:77-78`: "its best configuration reaches 56.16 macro-F1, against 56.82 for the dedicated model at its own tuned width and **64.54** for the joint model."
- **Source:** `storyline/audit/capacity_baseline_experiment.md §5.3` quotes "joint v17 = **64.54** (n=20)" — this is the diagnostic-best board-of-record cell, 0.03 above the joint-best 64.51 the dissertation reports in Table III.
- **Rule:** WRITING_LAW §3 / AGENT_GUARDRAILS N5 — "the MobiWac joint-best vs diagnostic-best distinction must never blur." The same quantity (AL joint next-category macro-F1) reads two ways ~25 pages apart.
- **Impact:** the claim's direction (parameter count "yields nothing"; 56.16 ≪ joint) is TRUE under either value, so it does not mislead on the result. It is a convention blur a careful examiner cross-referencing Table III would catch.
- **Suggested direction (author rules):** reconcile Ch.6 to the dissertation's reported convention (64.51, the Table III joint-best value; the gap becomes +8.38 over the capacity arm, +7.69 over the ceiling = the whitelist's AL category Δ), OR add a half-clause naming the 64.54 as the diagnostic-best board value. Do not invent a third number. **Shared with persona 06 (value check).**

### MAJOR-2 — Gowalla vintage stated as 2009–2010 contradicts the project's own measured provenance (2009–2011)
- **Ch.6** `chapters/6_conclusion.tex:107-108` (limitation 1): "The five state datasets come from Gowalla check-ins collected in **2009 and 2010**."
- **Ch.5** `chapters/5_mobiwac.tex` data-provenance note (hidden comment, ~L295): "Date range MEASURED on the parquet 2026-07-09: 2009-01-21 .. 2011-08-16 -> 'collected 2009 to 2011'. **The SNAP/cho2011 dump (Feb 2009-Oct 2010) is NOT the data source.**" The five-state MobiWac data is the figshare CC0 dump (2009–2011), not the cho2011/SNAP Gowalla.
- **Ch.4** `chapters/4_courb.tex:349` says "collected between February 2009 and October 2010" — but that is the CoUrb/liu2014 Gowalla (published reproduced text), a different data source than Ch.5's five-state datasets.
- **Rule:** WRITING_LAW §3 (limitations are concrete AND honest — "2009-2010 Gowalla" is named as the model honesty item, so the concrete vintage must be the correct one) + AGENT_GUARDRAILS N1 (a frame chapter "may only repeat numbers already sourced in a chapter, with the same hedges"; the 2009–2010 vintage is not sourced in any chapter's prose and is contradicted by Ch.5's measurement).
- **Impact:** load-bearing honesty sentence; a banca reading limitation 1 against Ch.5's provenance would see a self-inconsistency. Does not touch a performance claim.
- **Suggested direction (author rules — this is genuinely ambiguous, so it is a QUESTION):** either (a) correct limitation 1 to "2009 to 2011" to match the figshare dump Ch.5 actually consumed, or (b) if the intent is to name the reference-dataset (cho2011) vintage generically across all three studies, say so explicitly and reconcile with Ch.5's measured range. Note Ch.3/Ch.4 use SNAP/liu2014 Gowalla (2009–2010) while Ch.5 uses the figshare dump (2009–2011) — the Conclusion consolidates all three, so the sentence needs to be precise about which data it bounds. **Shared with persona 06 (value check).**

### MINOR-1 — CBIC dataset statistics unresolved (visible placeholders)
- `chapters/3_cbic.tex:~330`: "[$N_{\text{users}}$; VERIFY: recompute per ERRATA.md] users, [$N_{\text{poi}}$; ...] unique Points-of-Interest, and [$N_{\text{checkins}}$; ...] check-ins."
- **Honesty verdict: CORRECT handling** — the values were never filled in the published paper; the chapter renders visible placeholders rather than fabricating them, and Appendix B (`tab:apx:cbic-errata` row 5) declares them "Pending. Not invented." This is exactly the fail-closed behavior AGENT_GUARDRAILS mandates.
- **Rule:** N2/N3 — the sanctioned path (repo-committed recompute over the CBIC-era FL pipeline, author-approved; CoUrb's FL row a cross-check only) must run before the final build; the placeholders cannot ship in the defense PDF.
- **Suggested direction:** number-completion task for persona 06 / the author; not a claim-honesty defect. **Handoff to persona 06.**

### MINOR-2 — Ch.2 "roughly a third" Food-share provenance pointer needs correcting
- `chapters/2_fundamentals.tex:~/§2.4`: "The Food class alone accounts for roughly a third of the check-ins in a representative state."
- The §2.4 ledger flags the provenance pointer should be the check-in distribution table (Alabama Food 34.2%), not the 32.5% POI-count table. The CLAIM ("roughly a third") is qualitative and TRUE under either figure (34.2% or 32.5%), so it holds; only the ledger's internal source pointer needs the fix.
- **Rule:** N3 (traceability of the number behind the claim). Claim intact; provenance line to tidy. **Handoff to persona 06.**

### NIT-1 — Ch.5 Table 1 Istanbul Majority (33.4%) from earlier windowing
- `chapters/5_mobiwac.tex:315` Istanbul Majority = 33.4; the table comment discloses it is "from the earlier windowing of the same visits (raw visit share 27.0; recompute on the dk_ovl inputs ... if exactness is wanted)." The §setup-windows prose range "from about 25 percent of visits in Florida to 34 percent in Alabama" uses the clean Gowalla cells (FL 24.7, AL 34.2), so the claim in prose is unaffected. Value-exactness item only. **Handoff to persona 06.**

---

## NEW-CLAIM LIST (C2 — for author sign-off)

All arc/frame claims trace to the NORTH_STAR §6 approved spine (verified above); no claim exceeds it. The following frame passages are marked `[NEEDS SIGN-OFF]` in the source and are author-owned by construction — listed so the author confirms them, not because they introduce unlicensed claims:
1. **Resumo + Abstract** (`0_main.tex`): claim-parity pair, drafted from Ch.1+Ch.6 only. Audit them as a pair (done — they match; headline 5.3–9.4 = rounded whitelist +5.33…+9.35). Author confirms the PT/EN wording is his.
2. **Ch.5 preface + recap subsection** (`5_mobiwac.tex`): new-to-chapter time-capsule prose; claims from the approved spine + whitelist, no numbers quoted. Confirm.
3. **Appendices A/B/C**: new frame prose (BRACIS containment, errata catalogue, AI disclosure). A satisfies C4 (BRACIS as "earlier unpublished iteration"; region-cost claim only as corrected history). Confirm scope + AI-tool naming (Appendix C leaves model-version naming to the author).
4. **Ch.6 §6.2 capacity-matched baseline paragraph**: POST-SUBMISSION frame analysis licensed by the D1 contract; reading (i) reported honestly. Confirm prominence/placement (D1 contract §3.2 leaves length discretionary; the outcome-binding floor is met). Fix MAJOR-1 (64.54→64.51) here.

---

## HONESTY DEVICES INTACT (the mandated keeps, verified present)

| Device | Location | Status |
|---|---|---|
| Time-capsule prefaces (venue/status/what-is-revised) | Ch.3, Ch.4, Ch.5 prefaces | ✓ all three present |
| CBIC null time-indexed + Nash-MTL of-the-time caution | Ch.3 preface; Ch.1 §1.2; Ch.6 §6.1 | ✓ |
| CoUrb sample-stratified-split disclosure | Ch.4 preface + §exp-setup; Ch.2 §2.4 GLOSSARY note | ✓ plain, not hidden |
| CoUrb ownership/contribution note | Ch.4 preface; Ch.1 §1.5 | ✓ |
| CoUrb "does not revisit MTL-vs-STL" floor sentence | Ch.4 preface | ✓ verbatim |
| AZ never upgraded (0.00 = match, not gain) | Ch.5 Table III (≈, unbolded), §results-part2, Ch.6, Abstract | ✓ everywhere |
| Region verbs bound to tests (↑ superiority / ≈ TOST) | Ch.5 Table III + prose; Ch.6; frame | ✓ |
| Scaling claim scoped to five U.S. states | Ch.5 §intro, §results-part2; "not a precise law" | ✓ |
| Cascade = defense at equal cost, not a win; not a re-implementation | Ch.5 §results-part2, §setup-baselines | ✓ |
| Compute-cost honesty (joint model larger; operational not arithmetic) | Ch.1 §1.2; Ch.5 §method-model; Ch.6 §6.2 | ✓ never promises lower cost |
| Leak-audit / hygiene sentences (per-fold prior, label-free, 13–27 pp inflation disclosed) | Ch.5 §setup-windows | ✓ restored + intact |
| Freeze control + capacity baseline as findings (gain in the shared trunk, not task-teaching, not parameter count) | Ch.5 §results-part2; Ch.6 §6.2 | ✓ "sharing stopped hurting", never "tasks teach each other" |
| N3 cosine +0.001 full scope (four seeds, three of six, this pair not a rule) | Ch.5 §related-mtl; Ch.6 §6.2 | ✓ travels verbatim |
| Negative result written with care (arc's foundation) | Ch.1 §1.2; Ch.6 §6.1, §6.4 | ✓ |
| BRACIS containment (corrected history only) | Appendix A | ✓ C4 satisfied |
| Task-pair confound concession + fixed-pair ablation future work | Ch.6 §6.3 lim 6, §6.4 | ✓ signed-off additions present |
| Placeholders left visible, not fabricated | Ch.3 dataset stats; Appendix B "Pending, Not invented" | ✓ fail-closed |

---

## OUT-OF-SCOPE HANDOFFS (one line each)
- **Persona 06 (numbers):** MAJOR-1 (64.51/64.54), MAJOR-2 (vintage), MINOR-1 (CBIC placeholders), MINOR-2 (Food-share pointer), NIT-1 (Istanbul majority) — all carry a value dimension; verify the values, I verified the claims around them.
- **Persona 08 (translation fidelity):** the Ch.4 English title rendering ("ST-MTLNet: Spatio-Temporal Point-of-Interest Representations for Multi-Task Learning") vs the published PT title is a translation-fidelity check, not a claim check.
- **Persona 04 (concordance):** MAJOR-1 and MAJOR-2 are also cross-chapter concordance items.

---

## WHAT HOLDS / WHAT READS WELL (do not touch)
- The arc is honest and the honesty devices are load-bearing and present. The verb-test binding is airtight across all six chapters — this is the single hardest thing to get right in this document and it is right.
- Ch.5 (the whitelist-governed chapter, highest fabrication risk) is whitelist-exact: no never-cite value, no AZ upgrade, cascade correctly framed, scaling correctly scoped, compute cost disclosed.
- The CBIC placeholders and the CoUrb audited-number corrections show the fail-closed discipline working as designed: nothing was smoothed over or invented.
- Ch.6's capacity-baseline paragraph is a model of honest post-hoc reporting: an unfavorable outcome would have been reportable (D1 contract), and the favorable one is not overstated.

## OPEN QUESTIONS (only the author can answer)
1. **MAJOR-2 vintage:** which data vintage bounds Ch.6 limitation 1 — the figshare dump Ch.5 measured (2009–2011), or a generic reference-dataset window? The five-state data Ch.5 uses is 2009–2011; the sentence currently says 2009–2010.
2. **MAJOR-1 convention:** reconcile Ch.6 to the joint-best 64.51 (dissertation's reported convention) or name 64.54 as the diagnostic-best board value?
3. **Appendix C:** name specific model versions (author-sourceable) or keep the family-level disclosure as drafted?

---

## VERDICT

**GATE PASS.**

No fail trigger is present: zero unlicensed claims (every arc/frame claim traces to the NORTH_STAR §6 approved spine and the MobiWac whitelist), zero verb-test mismatches (superiority↔"outperforms", TOST↔"matches" bound everywhere; AZ never upgraded; scaling scoped to the five U.S. states), zero C3 never-cite values, C4 BRACIS containment satisfied (Appendix A), and every mandated hygiene/fairness/honesty device is present and intact (inventory above).

Two **MAJOR** cross-chapter consistency findings remain (MAJOR-1: the 64.51/64.54 joint-best/diagnostic-best blur; MAJOR-2: the 2009–2010 vs 2009–2011 data vintage). Neither meets a gate-fail trigger — both preserve the direction and licensing of every result — but both are real defects a careful examiner would catch, both straddle persona 06's value-check scope, and both should be resolved before the advisor handoff. The three open questions are the author's to rule.
