# MobiWac Paper — Review Considerations (v3)

Line-item review comments on the current paper draft. Each entry quotes the passage under
discussion, followed by the reviewer's comment.

> **Audit + decision record appended 2026-07-18 (Fable).** Each entry now carries an
> **Audit** block: verdict, evidence, the decision agreed in discussion (or my lean where the
> author delegated), and the exact proposed replacement text. Every proposed rewrite was checked
> against GLOSSARY.md (no em-dash, banned-word lists, naming/verb rules, honesty rule) and, where
> factual, traced to code or result JSONs.
>
> **EXECUTED 2026-07-18 (author go-ahead in chat):** all checklist edits applied in one pass;
> build clean (pdflatex+bibtex, **9 pages**, 0 warnings, 0 undefined refs). Two upgrades landed
> beyond the original checklist, both author-intent driven: (1) **CA/TX upgraded to n=20**
> (A1 top-up; `m1_stats_n20.py` re-run locally, all reproduction gates PASS); (2) **Table III
> switched to the JOINT-BEST convention** (one saved model per fold, both tasks read at its
> validation-selected epoch) per the author's #10 note; every pre-registered test re-verified on
> the joint-best arrays before the switch (all pass; details under #10 below). Figure 1 redrawn
> with the two Check2HGI outputs; Figure 4 re-rendered with the joint-best deltas.

---

### 1

**Quote:**
> "At every dataset, it outperforms a dedicated category model (about +5 to +9 macro-F1), and on
> the next-region task it outperforms the dedicated region model at four of the six datasets,
> while matching it (statistically, within two points) at the other two. Across the five U.S.
> states, the gain on region grows with the number of regions. On the non-U.S. city too, the
> joint model outperforms on category and is slightly ahead on region."

**Comment:** This last part of the abstract is very repetivie the 4 of six state, also we could sumarrize this better,
no ?

> **Audit (E1) — PARTIALLY AGREE; fix by merging, not cutting.** The redundancy is real: the
> Istanbul sentence re-states claims already covered by "at every dataset" (category) and "four
> of the six" (region). But the honesty rule (GLOSSARY §1) requires Istanbul to stay visibly
> positive next to the U.S.-only scaling claim, or a reader infers the fewest-regions city should
> be negative. **Decision (my lean, delegated): apply this replacement** for the final three
> sentences of the abstract:
>
> *"It outperforms a dedicated category model at every dataset (about $+5$ to $+9$ macro-F1) and
> outperforms the dedicated region model at four of the six, matching it (statistically, within
> two points) at the other two. Across the five U.S. states, the region gain grows with the
> number of regions; Istanbul, the city with the fewest regions, is also slightly ahead on
> region."*
>
> Keeps: the exact ledger-mandated TOST string, "outperforms" as the only superiority verb, the
> +5-to-+9 number, the U.S.-scoped scaling claim, "slightly ahead" for Istanbul. Saves ~25 words
> and removes the fourth stacked scope-phrase.

---

### 2

**Quote:**
> "The California and Texas joint results use a single random initialization; the other four
> datasets use four for both models (Section VI-B)."

**Comment:** We need to be very carreful with "single random initialization", since this term can be confusing and some
reviews can not understand what his can mean. What do you think ?

> **Audit (E2) — PARTIALLY AGREE, and now SUPERSEDED by the author-approved CA/TX n=20 upgrade.**
> The term itself is locked: the GLOSSARY §3 seed rule (author ruling 2026-07-10) forbids "seed"
> in the abstract/intro, so "random initialization" is the only legal term there; the real defect
> was the ellipsis ("use four" — four what? and what do the CA/TX dedicated models use?).
> **That is now moot: the A1 top-up (CA/TX seeds {1,7,100}) completed 2026-07-11**
> (`docs/results/closing_data/catx_v17_n20/`), and the pre-registered 6-dataset stats were re-run
> and PASS (M1-FULL rev 4, 2026-07-13, `docs/studies/closing_data/v17_completion/stats_n20/RESULTS.md`):
> category Holm-adjusted p ≤ 8.9e-07 at all six (CA Δ+6.45, TX Δ+7.45); region CA Δ+2.20
> (90% CI +2.19…+2.22, entirely above the two-point margin), TX Δ+2.11 — **no verdict changes**.
> **Decision (author, 2026-07-18): upgrade CA/TX to n=20 across the paper.** The single-seed
> disclosure retires everywhere. Touch points (edit E2, applied together):
>
> 1. `01_introduction.tex:45` — replace the asymmetry sentence with: *"All joint and dedicated
     > results average four random initializations (Section~\ref{sec:results-part2})."*
> 2. `tbl3_results.tex` — CA/TX joint cells → n=20 means ± cross-seed std: TX cat
     > 77.24\sd{0.01}, TX reg 67.06\sd{0.01}, CA cat 77.05\sd{0.01}, CA reg 65.69\sd{0.02};
     > remove the two $^{\circ}$ markers and the $^{\circ}$ footnote ($^{\dagger}$ STAN partial and
     > $^{\ddagger}$ ReHDM footnotes stay); update the provenance comment block.
> 3. `05_setup.tex:43` — statistics paragraph: drop the CA/TX single-seed fold-paired Wilcoxon
     > branch; all six datasets are now seed-level paired (n=4 per dataset), Holm across the six
     > category comparisons; region stays TOST at AL/AZ and superiority at Istanbul/FL/TX/CA.
     > Re-check the TOST power sentence ("0.04 to 0.15 … at the four datasets with four seeds")
     > while there.
> 4. `06_results.tex:86-93` — replace the single-seed disclosure + "More seeds at these two
     > states are the remaining confirmation" with the n=20 confirmation (updated Δs: cat CA +6.45
     > / TX +7.45; reg CA +2.20 / TX +2.11).
> 5. `07_discussion.tex:21` — limitation #1 (CA/TX single seed) is deleted; "Three limits" →
     > "Two limits" (interacts with E13 below).
> 6. `figs/fig4_deltas.py` — CA/TX delta bars move by ≤0.01; update values and re-render the PDF.
> 7. Post-edit doc sync (separate commit, after the paper edits): GLOSSARY honesty-rule footing,
     > `articles/[mobiwac]/CLAUDE.md` §2, PAPER_PLAN §3 whitelist — retire the "CA/TX single-seed /
     > provisional" clauses.
>
> Verification step before applying numbers: re-run
> `docs/studies/closing_data/v17_completion/stats_n20/m1_stats_n20.py` (CPU-only, reads committed
> artifacts, aborts on any board mismatch).

---

### 3

**Quote:**
> "rather than switching to a sequence model as CTLE does"

**Comment:** What is sequence model in this context ? U didn't explain this early. And if the cehck2hgi is not
sequencial what is is ? We should think on this sentence better

> **Audit (E3) — AGREE.** "Sequence model" appears exactly once, unglossed, and the sentence
> never says what Check2HGI is instead. Web-verified: CTLE (Lin et al., AAAI 2021) is a
> bidirectional Transformer over the check-in sequence, so the gloss is honest; naming another
> system's architecture is explicitly allowed (GLOSSARY §4). Check2HGI is a network (graph)
> model; visit order still enters as time-decayed links between consecutive check-ins
> (`04_method.tex:20`, `research/embeddings/check2hgi/README.md`). **Decision (my lean,
> delegated): apply**, replacing the second sentence of the ¶ at `02_related.tex:24-28`:
>
> *"Our representation, Check2HGI, also works at the check-in level but through a different
> construction. CTLE is a sequence model, a Transformer that reads the check-in sequence itself;
> Check2HGI remains a network model, keeping the hierarchical place-to-region-to-city network and
> the same infomax objective, now extended one level deeper, from individual places down to
> individual check-ins (Fig.~\ref{fig:dataflow}). The order of a user's visits still enters the
> network, as links between consecutive check-ins (Section IV-A ref)."*
>
> (Use the actual §4.1 label at edit time.) This also removes one "rather than" (density 10→9).

---

### 4

**Quote:**
> "A census tract is a neighborhood, not a radio cell, so we scope our claims to
> neighborhood-level preparation: demand and load anticipation, caching content ahead of time,
> and capacity planning; not cell association or handover. We build and evaluate no such service
> here;"

**Comment:** This sentence is good, and contains important information, but I not belive that is well scrutured and the
flow of it is broken, we could make somthing more easy to read and undurstand

> **Audit (E4) — AGREE.** One ~40-word sentence doing four jobs. Constraint: "a neighborhood,
> not a radio cell" is a ledger-mandated keep (GLOSSARY §7) and survives verbatim. **Decision
> (my lean, delegated): apply this restructure** at `03_problem.tex:21` (also converts the
> elliptical "; not cell association or handover" into a full sentence, satisfying §9.2):
>
> *"A census tract is a neighborhood, not a radio cell. We therefore scope our claims to
> neighborhood-level preparation: demand and load anticipation, caching content ahead of time,
> and capacity planning. Cell association and handover are radio-level decisions and remain out
> of scope. We build and evaluate no such service here; Section~\ref{sec:discussion} quantifies
> what these predictions would give such a service."*

---

### 5

**Quote:**
> "wo input streams feed the model: a semantic one carrying the window of per-visit vectors, and
> a spatial one in which each visit is represented by the trained vector of its region node from
> the same graph."

**Comment:** This is semms to be wrong the two inputs in our models is the checking 9 window visits for the next
category. For the next region we use the a region embeddion 9 window vist. Check this in the code.

> **Audit (E5) — YOUR FACTS ARE RIGHT; the sentence is not wrong, but it is unclear — and the
> literal per-task version would over-correct.** Code-verified (`src/data/folds.py:1427-1448`,
> `src/data/inputs/region_sequence.py`, `src/models/mtl/mtlnet_crossattn/model.py:404-465`,
> dualtower `model.py:87-129`, board flags `--task-a-input-type checkin --task-b-input-type
> region`): the category task's input IS the check-in 9-window and the region task's input IS the
> region-embedding 9-window over the SAME nine visits. But the bidirectional cross-attention
> trunk makes each prediction depend on both windows, so "input X → task A only" would be false;
> the fix is to state the stream-to-task binding while keeping the meet at the trunk. The region
> head additionally consumes the raw spatial window through the private path
> (`raw_region_seq=next_input`, fusion_mode=aux). **Decision (my lean, delegated): apply both
> edits** at `04_method.tex:29-30` and `:33`:
>
> Main: *"Each task has its own input. The category task reads the window of per-visit vectors
> (the semantic stream); the region task reads the same window of visits, each visit now
> represented by the trained vector of its region node from the same graph (the spatial stream).
> Both pass through private per-task encoders (a small input network per task, with no shared
> weights) into the shared trunk, a cross-attention stack of two blocks, so each prediction still
> uses both inputs."*
>
> Line 33: *"…the region output keeps a private spatial path, a small branch inside the one model
> (not a second model) that reads the raw spatial window directly, bypassing the shared trunk,
> and that the category task does not touch."*

---

### 6

**Quote:**
> "Third, the one component that could pass visit-to-visit information, a region-transition prior
> (a table of how often one region follows another), inflated region accuracy by 13 to 27 points
> when an earlier version built it on the whole dataset; any such table is therefore built per
> fold, from training data only, in every repetition of the experiment. The joint and dedicated
> models do not use this prior;"

**Comment:** This is a whole unecessary phrase, since this was an artifact of the study phase and is not delivery in the
final model. I belive that we can remo all of it. Futhermore, the `Integrity of the representation.`, cna be reduce
drastically to one phrase or less, my point is: I know of its importantece but this is evidences that we can showe upon
request this don't need to be in the main text I belive, aslo this paragh is writte due a report of one reviewer, I
belive that we need to address if way brief as it is today. Also this is a confgusing and tecnichal part, so put in the
test can casu more harm that good.

> **Audit (E6) — DISAGREE on deletion; AUTHOR RULED (2026-07-18): keep, with the
> floor-compliant compression, and prioritize clarity (this ¶ is currently the least clear in
> the paper).** Two factual corrections to the premise: (a) the prior is NOT a study-phase relic
> — it is live in HMT-GRN, the primary region baseline (`05_setup.tex:48` "add a region-transition
> prior built per fold from training data"), so the sentence certifies a live baseline's hygiene
> and states our headline numbers owe nothing to a component with a demonstrated 13–27-point
> leak; (b) GLOSSARY §9.4 was written against exactly this cut (the previous submission was
> rejected partly on a leakage accusation): "compress for clarity … never below the floor: the
> three grounds + the audit numbers + the per-fold prior construction". **Proposed replacement
> for the whole ¶ at `05_setup.tex:31`** (~205 words vs 293; shorter sentences, one signposted
> claim each, all floor items + both fairness signals kept):
>
> *"\emph{Integrity of the representation.} We train the representation once on the whole dataset
> and feed it to every model, dedicated and joint. We verify that it passes no usable information
> about the test visits, on three grounds. First, its training objective is label-free: it
> contrasts real graph neighborhoods against shuffled ones and never sees a next-category or
> next-region target. Second, we measured what the whole-dataset graph could pass to the test
> side. Rebuilding the representation per fold, from that fold's training users only, moves both
> tasks by at most a third of a point (region $-0.33$ to $+0.01$; category $0.00$ to $+0.29$, at
> Alabama, Arizona, and Florida), within fold noise. This measurement covers the visits whose
> places appear in training (67 to 87 percent); visits to places unseen in training are the one
> part it cannot reach. Third, the one component that could pass information between visits, a
> region-transition prior (a table of how often one region follows another), is built per fold
> from training data only, after an earlier whole-dataset version inflated region accuracy by 13
> to 27 points. Our joint and dedicated models do not use this prior; it appears only in the
> HMT-GRN baseline (Section~\ref{sec:setup-baselines}). The baseline representations meet the
> same standard: HGI is pre-trained once on the whole dataset, like ours, and CTLE per fold on
> training users only."*

---

### 7

**Quote:**
> "Where the joint model is expected in advance to outperform, we test superiority."

**Comment:** This phrase is a little bit thorough way withou context, no? I didn't get it.

> **Audit (E7) — AGREE.** The ¶ opens with the rule's second half and never says why two tests
> coexist or that the assignment (not only the margin) was fixed in advance; the pre-registration
> (`v17_completion/STATISTICAL_PROTOCOL.md`) supports the missing context. **Decision (my lean,
> delegated): apply** at `05_setup.tex:43` — new opening:
>
> *"A claimed gain and a claimed match require different tests, because a difference that fails
> to reach significance is not by itself evidence of a match. We fix the assignment in advance,
> before reading any results: where the joint model was expected to outperform, we test
> \emph{superiority}; where it was expected only to match the dedicated model, we test
> \emph{non-inferiority}."*
>
> and shorten the later TOST sentence to avoid stating the rule twice: *"The
> \emph{non-inferiority} claim is that the joint model is no worse than the dedicated model by
> more than a two-point margin; we test it with the two one-sided tests (TOST)
> procedure~\cite{lakens2017tost}."* (Merges with the E2 rewrite of this ¶ — one combined edit.)

---

### 8

**Quote:**
> "To calibrate the two scales:"

**Comment:** Calibrated what ? in reference of which data? The phrase is seems incomplete

> **Audit (E8, carries E9) — AGREE.** It is a floors sentence and should say so, reusing
> "reference point" (already established in §5.3; anti-synonym-cycling). **Decision (my lean,
> delegated): apply** at `06_results.tex:36-37`:
>
> *"To give both metrics a reference point: on category, always predicting the most common of the
> seven categories reaches a macro-F1 of only about $7\%$ (the majority-class floor of
> Section~\ref{sec:setup-metrics}); on region, with 520 to 8{,}501 regions, a random top-ten
> guess is right at most about two percent of the time."*
>
> The random-guess clause then appears in both §5.3 and §6.2; kept deliberately (table-side
> readability).

---

### 9

**Quote:**
> "a majority-class floor of about 7%"

**Comment:** I didn't get the mean of it here

> **Audit (E9) — AGREE the parenthetical is too compressed; the number is right.** Verified in
> the JSONs (`docs/results/P0/simple_baselines/*/next_category_f1.json`, per-dataset 5.66–7.28):
> it is the macro-F1 of always predicting the majority class — that class alone scores ~0.4–0.5
> F1, the other six score zero, and the 7-way macro average lands near 7. NOT 1/7 chance. The E8
> rewrite carries the in-line gloss; no further change.

---

### 10

**Quote:**
> "Throughout, each task is scored at its own validation-best epoch, under the same rule for the
> joint and dedicated models, hence two epochs per fold for the joint model; a single jointly
> selected saved model reproduces every joint result within 0.06 (category) and 0.11 (region)
> points, so no reported claim depends on that choice."

**Comment:** So we are saying that in the dedicated model we have useds the result of the best epoch during the train,
but for the joint model we have used the best joitly result. Is that ? At lest I belive it what should be. But, my point
is the phrase is a little bit hard to read and undrstand.

> **Audit (E10) — AGREE, and note your paraphrase of it is exactly the misreading the sentence
> invites (both models score each task at its OWN validation-best epoch, same rule; the
> jointly-selected saved model is the robustness check, not the reported number).** The
> 0.06/0.11 bounds re-verified against `joint_best/data/j1_results.json` (max deviations
> 0.051 cat / 0.107 reg, both at AZ). **Decision (my lean, delegated): apply** at
> `06_results.tex:42-45`:
>
> *"Throughout, the joint and the dedicated models are scored under one rule: each task is read
> at the epoch where its own validation score peaks, so the joint model's category and region
> results can come from two different epochs of the same training run. If we instead keep the
> single saved joint model per fold, one set of weights serving both tasks, every joint result
> changes by at most $0.06$ (category) and $0.11$ (region) points, so no reported claim depends
> on this choice."*
>
**Author**: Reading the phrase: "If we instead keep the
single saved joint model per fold, one set of weights serving both tasks, every joint result
changes by at most $0.06$ (category) and $0.11$ (region) points, so no reported claim depends
on this choice." is still a little bit confuse for me. So, to be in the same level let's first stablish my line of
thoughs: My aim was to use the best jointly result of the mtl model, so in the epoch that give the best jointly result
we should use the reuslt of it, not the results of the individual best task result in the eopchs. Said that, if we
already done it I belive that we can cut this phrase.

> **ANSWER + EXECUTED (2026-07-18).** Important clarification: the paper had NOT done what you
> describe — Table III reported each task at its OWN best epoch (diagnostic-best), and the
> confusing sentence disclosed that. Your stated aim (report the one saved model per fold at the
> epoch the joint validation score selects, both tasks read there) is the prepared "joint-best"
> lane. Before switching, an agent re-ran every pre-registered test on the joint-best arrays
> (all six datasets n=20): category Holm all reject (worst adjusted p=1.0e-06); AL TOST passes
> (Δ −0.41, CI −0.63 to −0.20, far inside ±2); AZ TOST passes (0.00, CI −0.08 to +0.07); region
> superiority holds at Istanbul (+0.19, CI +0.15 to +0.23, 20/20 folds), FL (+0.71), TX (+2.11),
> CA (+2.20); the scaling claim becomes strictly monotone across the five states. **No verdict
> changes, so the switch was executed**: Table III now carries the joint-best cells (Ist reg
> 75.35, AL reg 69.70, AZ reg 59.46, FL reg 77.41; cat 63.32/64.51/65.79/79.84/77.24/77.05),
> §6.2's CIs and deltas re-footed, fig 4 re-rendered, §5.3's power flourish re-fitted
> ("half a point" → "one point", AL's joint-best CI needs ±1). The sentence you flagged now
> reads as a simple convention statement with diag-best demoted to the robustness check. Watch
> item: Istanbul region is the thinnest bolded cell (+0.19); statistically solid but visibly
> small — flagged to the review panel.

---

### 11

**Quote:**
> "the like-for-like anchor remains the dedicated column."

**Comment:** What you mean by this ?

> **Audit (E11) — AGREE.** Intended meaning: the externals run on their own embeddings, so
> beating them mixes the Part-1 representation advantage with the joint-training effect; the
> controlled comparison is the Dedicated column (same representation, windows, folds). The idiom
> fails GLOSSARY §8's register test. **Decision (my lean, delegated): apply** at
> `06_results.tex:106` — replace the clause with:
>
> *"; the controlled comparison for the joint model remains the one against the dedicated models
> (the Dedicated column of Table~\ref{tab:results}), which use the same representation, windows,
> and folds."*
>
> Also (register consistency, same edit): `06_results.tex:17` "The comparison is like-for-like:"
> → *"The comparison is controlled:"* (the colon list already spells out what is held equal).

---

### 12

**Quote:**
> "A simple first-order Markov region floor reaches 43 to 65 Acc@10 across the datasets (computed
> under a non-overlapping windowing of the same data, indicative rather than protocol-matched);"

**Comment:** i belive this is wrong, we have runned the markov unve the overleap window, check it. also have we search
and acknoladge that overleap window is wrong since sliding windows meand the same and is the comman name in literature

> **Audit (E12) — DISAGREE on the fact; the paper is correct. RECOMPUTE DONE (2026-07-18),
> verdict-preserving.** Three independent proofs the 43–65 floor was computed on NON-overlapping
> (stride-9) windows: (a) window-count arithmetic — the floor JSONs hold 7–8× fewer windows than
> the stride-1 protocol at every dataset (check-ins/windows ≈ 8.8–9.0); (b) the producing script
> `scripts/compute_simple_baselines.py` hardcodes the frozen non-overlap `check2hgi` engine;
> (c) commit provenance predates the overlap switch. No stride-1 Markov JSON existed in the repo.
> The likely memory: HMT-GRN Istanbul, which really was run under both windowings (56.56
> non-overlap / 60.42 stride-1; the board cites the stride-1 one). Terminology: already settled —
> the 2026-07-10 ruling adopts "sliding windows" as the one name.
>
> **Recompute result (stride-1, window 9, MIN_SEQ=10, same fold protocol; window-count gate vs
> Table 1 EXACT at the five Gowalla states, Istanbul −0.53% substrate-vintage residual, well
> inside its ±0.38 fold-std):** the floor rises to Ist 65.06 / AL 62.26 / AZ 51.23 / FL 72.47 /
> TX 60.10 / CA 59.09 — range 51–72 (was 43–65) — and **the joint model still clears it at all
> six datasets, by +4.95 (FL, tightest) to +10.38 (Istanbul)**. Artifacts:
> `docs/results/closing_data/markov_floor_stride1/*.json` (spot-verified) +
> `docs/studies/closing_data/MARKOV_FLOOR_STRIDE1.md` +
> `scripts/closing_data/compute_markov_floor_stride1.py`; `docs/results/P0/` untouched; the
> category Markov-9 floor deliberately untouched (row-matched to POI-RGNN's ETL).
> **Proposed sentence swap at `06_results.tex:106-109` (my lean: apply in this pass, not at
> camera-ready, since it is clean and the caveat retires):**
>
> *"A simple first-order Markov region floor, recomputed under the same sliding-window protocol
> and fold splits as our models, reaches $51$ to $72$ Acc@10 across the datasets; the joint model
> exceeds it by $5$ to $10$ points at all six datasets."*
>
> Bounds note for your ruling: FL's exact margin is $+4.95$, which rounds to 5; if you prefer a
> margin never rounded up, the exact variant is "by $4.9$ to $10.4$ points". Istanbul's windows
> are the local pre-rebuild stride-1 base (270,217 vs the board's 271,666); documented in the
> findings note, no paper caveat needed in my view.

---

### 13

**Quote:**
> "Third, although a mobility-aware service motivates this work, we do not build or evaluate
> one."

**Comment:** I get the honnest on it, but I don't see the need for ackonladge this. THe text speak for it self since we
don't have this data.

> **Audit (E13) — PARTIALLY AGREE; AUTHOR RULED (2026-07-18): shorten, do not delete, and word
> it so the service reads as background motivation, not the paper's subject (the main claims are
> the prediction results).** The limitations list is where a program-committee member checks
> disclosure, and the venue-bridge ledger ruling rests on this honesty posture; §7's other copy
> ("motivation, not a measured service result") is glossary-protected and stays. **Proposed
> replacement** (note E2 deletes limitation #1, so the list becomes "Two limits" and this becomes
> the second):
>
> *"Second, a mobility-aware service is the background motivation of this work rather than its
> subject; we do not build or evaluate one, and the paper's claims are the prediction results
> themselves."*

---

### 14

**Quote:**
> "Our representation is a fixed per-visit vector that any downstream model can consume;
> structural check-in analysis has named machine learning as a next step [5], and this study
> moves in that direction."

**Comment:** The part A make sense the part B is confuse I didn't get what you ment on it.

> **Audit (E14) — AGREE.** [5] = Moura, Aquino, Loureiro (MSWiM 2025): a descriptive
> network-science analysis of tourist check-in mobility whose conclusion names machine learning
> as a next step. The current sentence over-compressed previously certified wording ("structural
> check-in analysis" reads as a field, not the cited paper; the semicolon welds two unrelated
> thoughts). **Decision (my lean, delegated): apply** at `07_discussion.tex:24-27`:
>
> *"Our representation is a fixed per-visit vector that any downstream model can consume. A
> recent structural analysis of check-in networks names machine learning as a next
> step~\cite{moura2025mobilityaware}; this study, which learns to predict from check-in data,
> moves in that direction."*
>
> Guard: keep the indefinite "a next step" (Moura's sentence is hedged; the certified wording
> forbids strengthening it).

**Author:** "A recent structural analysis of check-in networks names machine learning as a next step~
\cite{moura2025mobilityaware};" I still keeping not undesrtanding thie phrase.

> **ANSWER + EXECUTED (2026-07-18).** The confusion was the abstract subject ("structural
> analysis" read as a field, not a paper). Applied a fully concrete version naming the authors
> and what their paper does:
> *"Moura et al. [5] study the structure of tourist check-ins, describing the network of past
> visits without predicting future ones, and name machine learning as a next step; this study
> moves in that direction, learning to predict the next visit from check-in data."*
> Meaning in one line: the cited MSWiM 2025 paper only describes past mobility and itself says
> prediction with machine learning is the natural follow-on; our paper is that follow-on. The
> hedged "a next step" is kept (their sentence is hedged; the certified wording forbids
> strengthening it).

---

### 15

**Comment:** In the conclusion I belive we miss to talk about if our results overcame other baselines.

> **Audit (E15) — AGREE; AUTHOR RULED (2026-07-18): add the full sentence.** Board-verified:
> the joint model is above every external region baseline by ≥4.4 Acc@10 (worst case FL 77.42 vs
> faithful STAN 72.99) and above POI-RGNN on category by ≥33 macro-F1, at all six datasets.
> Constraints honored: descriptive verb ("remains above", never "outperforms" — that verb is
> bound to the paired test, run only against the dedicated models); the cascade is absent (our
> own model, a tie); STAN-on-our-representation is not a baseline. **Insert before the final
> sentence of `08_conclusion.tex`:**
>
> *"The joint model also remains above every external baseline in Table~\ref{tab:results} at all
> six datasets, by at least 4 Acc@10 points over the strongest region reference (HMT-GRN, STAN,
> or ReHDM, each under its own protocol) and by at least 33 macro-F1 points over POI-RGNN on
> category."*

---

## Other considerations

1. Search common words used by AI in generate text so we can avoid them. if they are not naturally in the text.

> **Audit (E16) — DONE (web-refreshed sweep, 2026-07-18): the draft is CLEAN.** Zero hits on
> the GLOSSARY §7 banned lists across all sections/tables/captions; zero em-dashes; -ly density
> 0.44% (target ≤0.8%); intensifiers down to "far" ×2; the one hit on the NEWER 2025–2026 tell
> lists ("we propose two enhancements") is the C2-mandated lead-in (single load-bearing use;
> "density convicts, one hit doesn't"). **Decision (my lean, delegated): update GLOSSARY §7** —
> add the new tell rows (enhance, emphasize, align with, decorative "key", bolstered, vibrant,
> enduring, commendable, exceptional, invaluable, noteworthy) with a C2 carve-out ("the single
> mandated 'we propose two enhancements' is exempt; never a second use"), and refresh the §7
> audit banner to today's counts. The optional de-echo of the third verbatim "not from extra
> training signal" is NOT applied (GLOSSARY §9.2 prefers exact repetition over synonym-cycling).

2. On the fig 1 I have a consern, since in the check2hgi we output some results not only one, so in the figure we may
   should represent that we have mutiple outrputs fromt he check2hgi and this outputs will be used different for each
   task.

> **Audit (E17) — AGREE.** The trained graph exports per-visit vectors AND region-node vectors
> (`check2hgi.py:856-899`), consumed by different streams; Fig. 1's single "one per-visit
> vector" box is inconsistent with Fig. 2 (which already draws both windows) and §4. **Decision
> (my lean, delegated): apply the TikZ change** — split Stage 3 into two output nodes
> ("per-visit vectors (one per check-in)" / "region vectors (one per region)"), two extract
> arrows, both feeding the windows stage; amend the window sublabel; update the caption to name
> the two vector sets and the two model inputs. No extra "joint model" box (model anatomy stays
> in Fig. 2). Nuance kept: routing is per-stream, and the streams couple in the trunk, so the
> fork to the two output boxes stays where it is.

3. Table II could have compare check2hgi x hgi x CTLE.

> **Audit — AUTHOR RULED (2026-07-18): do NOT add the column.** Clean protocol-matched CTLE
> category cells exist only at Istanbul (25.92), AL (17.77), AZ (19.30); FL is a disclosed
> 2/5-fold partial (28.02); CA/TX were never run (a recorded board decision, ~a GPU-day each).
> A column would surface the missing cells as dashes and weaken the table. The 3-way comparison
> stays where it is, as §6.1 prose with coverage disclosed. **No edit.**

4. The STAN could have been a next-category and next-region baseline as well.

> **Audit (E18) — DISAGREE for this revision; the scoping is principled.** STAN is
> next-POI-native; the region adaptation works because region candidates keep a spatial identity
> (tract centroids feed STAN's defining candidate-distance matching). Categories have no
> coordinates: a category STAN either drops that mechanism (no longer STAN) or predicts next-POI
> mapped to category (new code, 6×5 folds, CA/TX re-pay the cost that already forced partial-fold
> closure). Category is covered by the category-native POI-RGNN + Markov-K + CTLE controls.
> **Decision (my lean, delegated): add one scoping clause** to the STAN sentence in
> `05_setup.tex:48` so it reads as a decision, not an omission:
>
> *"…under one fixed configuration; we adapt its output to region candidates, which keep the
> spatial identity its candidate-distance matching needs, and do not adapt it to category, where
> that mechanism has no meaning."*

---

## Approval checklist — ✅ ALL APPLIED 2026-07-18 (author go-ahead in chat; build clean, 9 pages)

> Every "my lean — apply" and "author-approved" row below was applied in the single edit pass.
> Ruling details fixed at apply time: E12 margin printed as the exact "$4.9$ to $10.3$" (joint-best
> footing; FL tightest at +4.94, Istanbul largest at +10.29); E13 applied as proposed (now the
> second of two limits, CA/TX limit deleted by E2); E10 superseded by the joint-best switch (see
> #10 above); E16's GLOSSARY §7 additions and the honesty-rule/doc sync run as a follow-up pass
> recorded in §Work items. The table below is the pre-approval record.

| ID    | Edit                                                                                                                                                       | Status                                          |
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| E1    | Abstract ending merge (Istanbul folded into scaling sentence)                                                                                              | my lean — apply                                 |
| E2    | **CA/TX n=20 upgrade** (intro, Tbl III cells + footnote, §5.3 stats, §6.2, §7 limits, fig4, then doc sync)                                                 | author-approved 2026-07-18                      |
| E3    | CTLE "sequence model" gloss + Check2HGI "network model" contrast (§2)                                                                                      | my lean — apply                                 |
| E4    | Census-tract ¶ restructure (§3)                                                                                                                            | my lean — apply                                 |
| E5    | Input-streams rewrite: stream→task binding + private-path source (§4)                                                                                      | my lean — apply                                 |
| E6    | Integrity ¶: floor-compliant clarity compression (~205 words)                                                                                              | author-approved 2026-07-18                      |
| E7    | Stats lead-in: why two tests + fixed in advance (§5.3)                                                                                                     | my lean — apply                                 |
| E8+E9 | "Reference point" floors sentence + majority-class gloss (§6.2)                                                                                            | my lean — apply                                 |
| E10   | Epoch-selection two-sentence rewrite (§6.2)                                                                                                                | my lean — apply                                 |
| E11   | "like-for-like anchor" → controlled-comparison clause; §6.1 sibling → "controlled"                                                                         | my lean — apply                                 |
| E12   | Markov floor: stride-1 recompute DONE and verdict-preserving; §6.2 sentence swap retires the caveat (bounds: "5 to 10" vs exact "4.9 to 10.4" — your call) | my lean — apply in this pass                    |
| E13   | Limitation shortened + background-motivation wording (§7)                                                                                                  | author-approved 2026-07-18 (wording to confirm) |
| E14   | Moura sentence split + named referent (§7)                                                                                                                 | my lean — apply                                 |
| E15   | Conclusion baseline-standing sentence (full form)                                                                                                          | author-approved 2026-07-18                      |
| E16   | GLOSSARY §7: new AI-tell rows + banner refresh; no prose edit needed                                                                                       | my lean — apply                                 |
| E17   | Fig. 1: two Check2HGI outputs + caption                                                                                                                    | my lean — apply                                 |
| E18   | STAN scoping clause (§5.4); no category STAN                                                                                                               | my lean — apply                                 |
| —     | Table II CTLE column                                                                                                                                       | REJECTED (author, 2026-07-18)                   |

## Panel follow-up rulings (author, 2026-07-18, post-panel)

The five-member panel's remaining asks were ruled as follows (reports in `panel_2026-07-18/`):

1. **Sub-margin region gains (R1/R2/R3 consensus ask): B — one clause in the conclusion only,
   abstract unchanged.** Applied: "(the gains at Texas and California exceed the two-point
   margin; those at Florida and Istanbul are statistically supported but smaller)".
2. **Edge-direction leakage question (R2 MF1): A — answer from the code, no new runs.** A
   read-only agent is producing the code-cited response-letter memo (edge direction,
   receptive field, what A4 does and does not cover, the cheap masking control described but
   not run).
3. **Feature-concat numbers (R2 MF2): A — trace the JSONs and add the number(s) to §6.1.**
4. **TX/CA leak-audit extension (R1 MF3 / R3 MF2): REJECTED** — no addition; the AL/AZ/FL audit
   is considered near-extensive for the larger states; the §7 transductivity limitation already
   discloses the residual.
5. **Prediction horizon (R1 MF4): A — compute locally under the stride-1 protocol** (new script
   reusing the Markov-floor machinery) and add one sentence to §5.2.
6. **Detail rulings:** gradient-correlation sentence — find the measured numbers (in flight);
   Markov floor stays OUT of Table III (prose only); Table II stays seed-0; HMT-GRN footnote
   gains the seen-region coverage percentage (in flight).

Also applied post-panel (V1's mechanical list): §6.1 lead takeaway sentence; Table III float
moved before §6; joint-selector gloss; CTLE sentence split; §6.1 de-echo; Markov-K gloss;
analysis-plan + paired-t clause in §5.3; Fig. 1 "POI"→"place"; abstract "improves ... by 28 to
40 points" + "(LBSNs)" dropped; conclusion de-numbered; the Fig. 3 caption blocker and two stale
comments fixed. Build clean at every step (9 pages, 0 warnings).

**Prose-only panel round (2026-07-18, evening; reports `panel_2026-07-18/prose_P{1,2,3}*.md`).**
Three text-only readers (cold reader, concordance checker, line editor). Verdicts: "genuinely
readable ... nothing suggests structural rewriting"; "as cross-section machinery this paper is
unusually tight"; "mechanics unusually clean ... no subject-verb errors, no article misuse, one
comma splice". ~35 of their fixes APPLIED, the big ones: the conclusion's "each under its own
protocol" factual slip corrected (HMT-GRN/STAN are on our folds); "at {dataset-count}" → "on"
sweep (the one recurring non-native tell; "at {StateName}" kept); abstract Istanbul sentence
de-contradicted ("nevertheless" replaces "also"); the region pathway defined in §4.2 (grounds
the freeze control); Check2HGI named again in §4.1 (was orphaned after §2); Markov region floor
now introduced in §5.4 before §6.2 uses it; "frozen weights (no fine-tuning)" replaces the
"fixed weights" collision with the loss weights; CTLE sentence de-garden-pathed with direction
named ("the check-in-level representation ahead by ..."); "no second model to serve" clarified;
STAN adaptation clause and HMT-GRN "next-place head" both unpacked; three-roles parallelism;
"To give both metrics a floor" (un-collides "reference point"); dangling "having long built"
fixed; mahalle glossed at first use; non-U.S./international unified; sundry mechanics (out of,
percent style, comma splice, apposition brackets, aux-weights parenthesis, versus→labeled
parens, Fig./Figure style). NOT applied (each collides with a settled ruling): the C2
"enhancements" lead-in; the §9.3 intro self-cite subject rule ("Prior work [7]"); the intro
"first" vs §2.2 "underexplored" pairing (both ledger-backed and substantively consistent); the
§6.1 epoch gloss (glossary mandates it for this audience). Final build: 9 pages, 0 warnings,
0 undefined, residue sweeps clean.

## 8-page trim campaign (2026-07-19/20) — ✅ DONE, build back at 8 pages

The 2026-07-18 additions had pushed the build to 9 pages against the **8-page no-fee EDAS budget**
(Step-3 record; the "10-page fee variant" ledger row was stale and is now corrected). Author
rulings: Tier-2 prose cuts except M1; NO figure work initially, later Fig. 1 only (option B);
advisor-gated throughout. Two advisory rounds ran (reports summarized here; the second round
covered M3/B/YEARS/M8, all APPROVED — with the Fig. 1 ~6.1 pt gray-italic legibility risk on
record, and two standing VETOES honored: the MCMG citation stays (novelty defusal) and "an
earlier preparation of the data" stays (honest vagueness; substrate AND windowing differed).
Applied: M2, M4, M5, M6, M9, M10, M3, M8, FN3, CAP1, CAP2, BIB, V1, V3, V4, F1, the years
parenthetical cut (restores the dataset-years ledger row AND removes the latent 2009-2011 vs
SNAP-2009-2010 vintage mismatch), Fig. 1 at 0.66\textwidth, float-separation/arraystretch
preamble, and ~10 zero-information loose-line word tightenings (orphan-word paragraph endings;
duplicate clauses: "(20 measurements each)", the §6.2 Markov protocol modifier now carried by
§5.4, the Moura clause duplicated by §3). NOT taken: M1 and REF (vetoed), M7/M11–M15 (unneeded),
Fig. 2/Fig. 3 shrinks, title shortening. **Final build: 8 pages, 0 warnings, 0 undefined refs,
0 overfull boxes, references end flush on page 8; Fig. 1 visually verified.** The only
prose-unrecoverable number from the cuts is TX's region delta point estimate (+2.11), bracketed
by its printed CI (+2.10 to +2.13), Fig. 3's label, and Table III.

## Work items in flight

- **Stride-1 Markov floor recompute — ✅ DONE (2026-07-18, ~40 s CPU-only).** Window-count gate
  vs Table 1: EXACT at AL/AZ/FL/CA/TX; Istanbul −0.53% (pre-rebuild stride-1 base; documented).
  Floors 51–72 (was 43–65); joint clears at all six by +4.95 to +10.38. See E12 for the
  approvable sentence swap. Artifacts spot-verified: JSONs at
  `docs/results/closing_data/markov_floor_stride1/`, findings note
  `docs/studies/closing_data/MARKOV_FLOOR_STRIDE1.md`, script
  `scripts/closing_data/compute_markov_floor_stride1.py`. Note for integration: a local
  `.git/info/exclude` rule hides `results/`-named paths from git status — committing the JSONs
  later needs `git add -f`.
- **CA/TX n=20** — no new experiments needed: M1-FULL rev 4 (2026-07-13) already ran the
  pre-registered 6-dataset family from committed artifacts (16/16 reproduction gates PASS).
  Before editing, `m1_stats_n20.py` will be re-run locally as a reproducibility check.
