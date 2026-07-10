1. Summary

  This is a mobility-prediction ML paper, not yet a mobility-systems paper. It proposes a check-in-level graph-infomax representation and a joint cross-attention model for next-category and next-region prediction across six LBSN datasets. The category results are strong and consistently reported; the region results are more conditional, especially because the
  large-state gains that drive the scaling story are single-seed provisional cells. The paper is unusually honest about cost, non-inferiority, and limitations, but the MobiWac fit is still fragile: no resource-allocation, caching, load, handover, latency, or deployment metric is actually evaluated. I would expect a hostile committee discussion to center on “nice
  ML, insufficient systems substance.”

  2. Strengths

  - Clear problem decomposition and metric definitions: macro-F1 for seven-category prediction and Acc@10 for region prediction are defined in articles/[mobiwac]/src/sections/05_setup.tex:41.
  - Good claims discipline in several places: Arizona is explicitly called “a match, not a gain,” and Alabama’s region deficit is disclosed in articles/[mobiwac]/src/sections/06_results.tex:83.
  - The paper does not hide model cost: it states the joint model is larger and more expensive than two dedicated models in articles/[mobiwac]/src/sections/04_method.tex:43.
  - Reproducibility is better than average: code, statistical tests, Gowalla figshare URL, and Massive-STEPS provenance are rendered in articles/[mobiwac]/src/sections/08_conclusion.tex:12.

  3. Weaknesses Ranked By Severity
  4. Systems relevance is still mostly motivational.
     Quote: “Third, although a mobility-aware service motivates this work, we do not build or evaluate one.” articles/[mobiwac]/src/sections/07_discussion.tex:24
     Why it matters at MobiWac: the paper measures prediction quality, not a mobility-management quantity. Acc@10 and centroid sketches do not show better capacity planning, caching, load anticipation, handover behavior, latency, or resource efficiency.

  5. The region scaling headline rests on provisional large-state cells.
     Quote: “California and Texas remain provisional: their dedicated ceilings are (n=20), but the joint cells are a single seed over five folds.” articles/[mobiwac]/src/sections/06_results.tex:85
     Why it matters: the largest-region datasets are exactly the points supporting the “gain rises with region count” story. A single seed plus fold-level Wilcoxon is not enough for a scaling claim at a systems venue.

  6. Potential transductive / temporal leakage concern is not fully closed.
     Quote: “We train the representation once on the whole dataset and feed it to both the dedicated single-task models and the joint model; we argue, and verify, that it carries no usable information about the test visits, on three grounds.” articles/[mobiwac]/src/sections/05_setup.tex:31
     Quote: “This is a place-level proxy and the one residual we cannot fully measure.” articles/[mobiwac]/src/sections/05_setup.tex:33
     Why it matters: online prediction cannot use future test-user trajectory structure. The audit covers only three datasets and only measurable in-coverage visits, so a skeptical reviewer can still question whether the representation is deployable as evaluated.

  7. The “label-free” representation claim is too easy to attack.
     Quote: “Each visit’s category, time of day, and day of week enter as input features of its node, not as edges.” articles/[mobiwac]/src/sections/04_method.tex:20
     Quote: “The training uses no task label: it never sees the next category or the next region.” articles/[mobiwac]/src/sections/04_method.tex:22
     Why it matters: for a next-category task, using category vocabulary inside the representation is not “label-free” in the ordinary reviewer sense. The huge category gain may look like feature injection unless there is a no-category-feature ablation.

  8. External baselines are useful but not clean enough for strong SOTA language.
     Quote: “It is a region-native model, not a reproduction of the complete published system.” articles/[mobiwac]/src/sections/05_setup.tex:50
     Quote: “These externals run on their own embeddings, so this comparison folds in the representation advantage of Section VI-A; the like-for-like anchor remains the dedicated column.” articles/[mobiwac]/src/sections/06_results.tex:104
     Why it matters: the “above every external baseline” result will not survive as a pure architecture comparison. The dedicated columns are the credible comparison; the external rows are contextual.

  9. The operational one-model claim is weak for systems readers.
     Quote: “The joint model carries the cross-attention stack and both task heads, so it is larger than either dedicated single-task model of Table III, and a forward pass costs more compute than running the two small dedicated models.” articles/[mobiwac]/src/sections/04_method.tex:43
     Why it matters: “one artifact” is a software-engineering convenience, not a demonstrated networking or mobility-management benefit.

  10. Detailed Comments

  - Abstract: “mobile and urban services can act ahead of demand” is broader than what is measured.
  - Sec. III: “We build and evaluate no such service here” is honest but damaging; it should appear earlier and be paired with a concrete proxy metric.
  - Sec. V-A: “the sixth is Istanbul, from the Massive-STEPS collection, a non-U.S. check” reads like a compression typo; use “non-U.S. dataset” or “non-U.S. city.”
  - Sec. V-C: “power near 1.0” is too strong for the single-seed cells; remove or restrict to multi-seed datasets.
  - Fig. 2 is legible only with effort after column-width scaling; simplify labels or make it double-column.
  - Fig. 3 caption says the region gain rises with region count but does not disclose CA/TX provisional status; the caption must say this.
  - Table III is self-contained, but the footnote carries too much methodological load.

  5. VERDICT

  Weak Reject, confidence 4/5.
  Estimated acceptance probability for a typical 3-person MobiWac committee: 25–40%.

  The paper is close to Borderline because the empirical work is substantial and the prose is disciplined. I would still vote Weak Reject because the central contribution is ML methodology, while the systems motivation is not evaluated and the most venue-relevant scaling result depends on provisional single-seed cells.

  6. EXTRA COMMENTS

  (a) Fixable In Prose Before Submission

  - articles/[mobiwac]/src/sections/07_discussion.tex:13: Replace “One model is enough” with “One model is competitive for these two prediction tasks.”
    Expected effect: +0.1; reduces overclaim.

  - articles/[mobiwac]/src/figs/fig4_deltas.tex:6: Add “CA/TX joint cells are single-seed provisional.”
    Expected effect: +0.2; prevents the headline figure from overstating evidence.

  - articles/[mobiwac]/src/sections/04_method.tex:20: Replace “label-free” wording with “does not use next-visit task labels” and explicitly say current-visit category is an input feature.
    Expected effect: +0.2; closes an avoidable reviewer attack.

  (b) Response-Letter Material

  - Prepare exact joint-checkpoint selection details for articles/[mobiwac]/src/sections/06_results.tex:41.
    Expected effect: +0.1 to +0.2 if challenged.

  - Prepare full leakage-audit tables, especially why whole-corpus representation does not encode future test-user trajectory information.
    Expected effect: +0.3 if reviewers raise transductive leakage.

  - Prepare baseline protocol appendix text explaining HMT-GRN adaptation, STAN partial folds, and ReHDM own-protocol comparability.
    Expected effect: +0.2; limits damage but will not turn externals into clean SOTA evidence.

  (c) Camera-Ready / New-Experiment Material

  - Run CA/TX joint cells at four seeds over five folds.
    Expected effect: +0.4 to +0.6; makes the scaling claim defensible.

  - Add a no-current-category-feature representation ablation.
    Expected effect: +0.3 to +0.5; tests whether Check2HGI is more than category-history injection.

  - Add a trace-driven anticipatory-service evaluation: top-k region staging, demand captured under fixed budget, false-positive region cost, and distance/load saved versus Markov and dedicated models.
    Expected effect: +0.8 to +1.0; this is the one change most likely to raise my verdict a full step.
------
• You are right. I overweighted the older “networking systems paper” prior.

  After checking the official MobiWac 2026 CFP, the venue scope is broader than my first review assumed. The CFP still says the symposium emphasizes “mobility management and wireless access,” but the topic list explicitly includes AI-based mobility management, Big data analytics for mobile and wireless networks, Context-aware protocols, services, architecture and
  applications, and Social mobile networks and applications. Source: MobiWac 2026 CFP: http://mobiwac-symposium.org/2026/cfp.html

  So I would revise the venue-fit criticism this way:

  - Not fair: “This is not a MobiWac paper unless it evaluates networking/resource-allocation quantities.”
  - Fair: “The paper is within scope, but it should make its fit explicit under AI-based mobility management / social mobile networks / context-aware services, rather than letting reviewers infer it.”

  That changes my verdict calibration. I would move from Weak Reject to Borderline, confidence 3/5, acceptance probability around 40–55%. The main remaining risks are CA/TX single-seed scaling, transductive representation concerns, and the category-feature/“label-free” wording, not basic venue mismatch.

  For EDAS topic tags, I would choose:

  1. AI-based mobility management
  2. Social mobile networks and applications
  3. Big data analytics for mobile and wireless networks

  If only two feel safer: use AI-based mobility management and Social mobile networks and applications. “Context-aware protocols, services, architecture and applications” is also defensible, but your paper is more analytics/prediction than protocol/architecture.

  On the Loureiro point: agreed. If the cited Moura/Loureiro MSWiM 2025 article is itself POI/check-in mobility work, that strengthens the “this community accepts POI/LBSN mobility analysis” argument. I would explicitly use it in the intro as venue-local precedent, not only as generic motivation.