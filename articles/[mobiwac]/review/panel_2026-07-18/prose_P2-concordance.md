P2 CONCORDANCE / CROSS-SECTION CONSISTENCY AUDIT — main.pdf (9 pp), sources in sections/*.tex

**[TOP 1] Conclusion contradicts Section V-D and the Table II footnote on baseline protocols.**
- Conclusion (08_conclusion.tex:12): "by at least 4 Acc@10 points over the strongest region reference (HMT-GRN, STAN, or ReHDM, each under its own protocol)"
- Section V-D: HMT-GRN is "evaluated on the same data, folds, and initialization"; STAN is "re-implemented from its published architecture and trained from the raw check-ins"; only "A ReHDM [32] reference is reported under its own published protocol." Table II footnote agrees: "HMT-GRN (region-native, on our folds, ...)"; "STAN (our re-implementation) and ReHDM (its own protocol)".
- Wrong: "each under its own protocol" flatly contradicts the body's carefully built claim that two of the three run on the authors' folds.
- Replace with: "by at least 4 Acc@10 points over the strongest region reference (HMT-GRN, STAN, or ReHDM; the first two on our folds, ReHDM under its own protocol)"

**[TOP 2] The intro claims "the first"; the section it cites as support claims only "underexplored".**
- Intro contribution 2 (01_introduction.tex:31): "to our knowledge, the first to treat fine-grained region as an end target of equal standing (Section II-B)"
- Section II-B (02_related.tex:54-56): "To our knowledge, fine-grained region as an end target of equal standing, rather than an auxiliary coarse grid cell, is underexplored. The nearest exceptions do not study our exact pairing:"
- Wrong: the contribution's priority claim is stronger than the hedged claim in the very section cited to back it, so a reader who follows the pointer finds the promise walked back.
- Replace (align II-B upward, since it already argues no exception matches): "To our knowledge, no prior work treats fine-grained region as an end target of equal standing, rather than an auxiliary coarse grid cell; the nearest exceptions do not study our exact pairing:" — or, if the authors prefer the softer claim, change the intro to "to our knowledge, fine-grained region as an end target of equal standing is otherwise underexplored (Section II-B)". Either way the two sites must state the same strength.

**[TOP 3] "Region pathway" is used twice at load-bearing moments but never defined; the region side already has two other names.**
- Section IV-B defines "the spatial stream" (the input window) and "a private spatial path, a small branch inside the one model"; then V-D (05_setup.tex:51): "feed the predicted category forward into the region pathway"; and VI-B (06_results.tex:98): "We freeze the region pathway at the start of training".
- Wrong: in the cascade rewiring and especially the freeze control, the reader cannot tell whether "the region pathway" means the spatial stream, the private spatial path, or the whole region half of the model — i.e., what exactly is frozen.
- Replace: append one sentence at the end of the IV-B architecture paragraph (04_method.tex:33): "Together, the spatial encoder, the region output, and this private path form the region pathway." (then the V-D and VI-B uses are grounded).

**4. The representation's name, Check2HGI, appears only in Related Work and then vanishes.**
- II-A (02_related.tex:24): "Our representation, Check2HGI, also works at the check-in level but through a different construction." — but the Method subsection is titled only "The check-in-level representation", and Check2HGI never appears again in Method, Results, tables, or Conclusion (Table III's column is "Check-in level").
- Wrong: a name is introduced and orphaned, so a first-time reader may not connect the thing named in II-A with the thing built in IV-A.
- Replace IV-A's first sentence: "We want each visit, not each place, to have its own vector; we call the resulting representation Check2HGI (Section II-A)." (or delete the name in II-A entirely and say "Our representation also works at the check-in level...").

**5. The abstract's "extra training signal" is never delivered in those words in the body.**
- Abstract (main.tex:74): "most of the gain comes from the per-visit context, not from extra training signal"; intro contribution 1: "attributes most of the gain to the per-visit context, not to extra training signal."
- Body delivery, VI-A: "averaging the per-visit vectors into one vector per place removes roughly 64 to 90 percent of the gain (state-dependent), so most of the gain is the context that each visit carries." — the phrase "training signal" reappears only in VI-B meaning something else ("the harder task's training signal").
- Wrong: the reader cannot match the abstract's negative claim ("not from extra training signal") to any sentence in the body that rules it out.
- Replace the VI-A clause with: "so most of the gain is the context that each visit carries, not the extra training signal that the per-visit features supply."

**6. A baseline appears in Results that Section V-D ("the comparisons play three roles") never introduced — and the category Markov baseline changes name in the same paragraph.**
- VI-B (06_results.tex:110): "A simple first-order Markov region floor, recomputed under the same sliding-window protocol and fold splits as our models, reaches 51 to 72 Acc@10 across the datasets" — absent from V-D. Three lines earlier the same paragraph says "and a Markov floor on category", while V-D and Table II call that baseline "Markov-K".
- Wrong: an unannounced comparison surfaces mid-results, and "Markov floor" now names two different things one sentence apart.
- Replace: in V-D role 1, after the ReHDM sentence, add "We also compute a first-order Markov region floor under the same sliding-window protocol and fold splits as our models."; in VI-B change "and a Markov floor on category" to "and Markov-K on category".

**7. "Reference point" means two different things in V-C and VI-B, and the random-guess sentence is duplicated nearly verbatim.**
- V-C (05_setup.tex:41): "its reference point is the dedicated single-task model." and "For scale, with 520 to 8,501 regions, a random top-ten guess includes the true region at most about two percent of the time."
- VI-B (06_results.tex:38): "To give both metrics a reference point: ... on region, with 520 to 8,501 regions, a random top-ten guess is right at most about two percent of the time."
- Wrong: V-C defines the region metric's "reference point" as the dedicated model, then VI-B reuses the same term for a chance floor and repeats V-C's sentence almost word for word.
- Replace the VI-B opener with: "To give both metrics a floor: on category, always predicting the most common of the seven categories reaches a macro-F1 of only about 7% (the majority-class floor of Section V-C); on region, a random top-ten guess is right at most about two percent of the time (Section V-C)."

**8. The same self-citation is "Prior work" in the intro and "Our earlier work" in Related Work.**
- Intro (01_introduction.tex:16-17): "Prior work observed exactly this for next-category and next-region [7]" — II-B (02_related.tex:48): "Our earlier work [7] established this two-task setup and observed negative transfer".
- Wrong: the attribution voice for the same reference flips between sections, and the intro reads as if [7] were someone else's result.
- Replace the intro with: "Our earlier work observed exactly this for next-category and next-region [7]".

**9. In the conclusion, "the standard place-level one" grammatically points at a task that does not exist.**
- Conclusion (08_conclusion.tex:8): "makes the next-category task far more learnable than the standard place-level one."
- Wrong: "one" attaches to "task" (there is no place-level task); the intended contrast is with the place-level representation.
- Replace with: "makes the next-category task far more learnable than the standard place-level representation does."

**10. "Mahalle" is used in Section III but only glossed two sections later.**
- III (03_problem.tex:13): "The second is its region, the place's census tract (the mahalle for Istanbul)." — the gloss arrives in V-A: "a mahalle (a municipal neighborhood) for Istanbul."
- Wrong: a first-time reader meets an unexplained Turkish term at first use; the definition comes after.
- Replace the III parenthetical with: "(for Istanbul, the mahalle, a municipal neighborhood)".

**11. The "three roles" enumeration in V-D breaks parallel form.**
- V-D: "The first role is the per-task state of the art..." / "The second role is two representation controls attributing the category gain..." / "The third role compares the two ways of coupling the tasks."
- Wrong: the pattern shifts from "the role is NP" (with a number-mismatched "is two controls") to "the role compares", so the scaffold the paragraph promises is not kept.
- Replace with: "The second role is a pair of representation controls that attribute the category gain to..." and "The third role is a comparison between the two ways of coupling the tasks."

**12. Istanbul is "one non-U.S. city" in the abstract but "one international city" in the intro, contribution 3, and the Table I caption.**
- Abstract (main.tex:71): "five U.S. states and one non-U.S. city (Istanbul)"; intro (01_introduction.tex:22): "one international city (Istanbul)"; Table I caption: "ONE INTERNATIONAL CITY"; V-A and VI-C use "non-U.S." again.
- Wrong: one recurring concept, two names, drifting section by section.
- Replace: use "non-U.S. city" everywhere (intro line 22, contribution 3, Table I caption); in V-A also smooth the apposition "the sixth is Istanbul, ..., a non-U.S. check on the findings" to "included as a non-U.S. check on the findings".

**13. Section VI-C opens on an ambiguous singular "the finding" — right after a section with several findings — and breaks the paper's "at {dataset}" convention.**
- VI-C (06_results.tex:126): "The finding holds on a non-U.S. city under the same representation, sliding-window protocol, and training setup as the U.S. states. On Istanbul, the joint model outperforms..."
- Wrong: VI-B established the joint-vs-dedicated result, the freeze control, and the cascade result, so "the finding" has no unique referent; also the paper says "at Alabama / at every dataset" everywhere else but "on Istanbul" here (and in VI-A "The same holds on Istanbul").
- Replace with: "The joint-model results hold at a non-U.S. city under the same representation, sliding-window protocol, and training setup as the U.S. states. At Istanbul, the joint model outperforms..." (and VI-A: "The same holds at Istanbul (+28.09)").

**14. Section III forwards the Istanbul label-mapping to Section V-A, which never delivers it.**
- III (03_problem.tex:13): "Istanbul's source collection maps its places onto the same seven labels (Section V-A)." — V-A says only "Every dataset uses the same seven place categories: Community, Entertainment, Food, Nightlife, Outdoors, Shopping, and Travel."
- Wrong: the forward reference promises detail on the mapping, but the target section merely restates that the categories are shared.
- Replace: add to V-A after that sentence: "For Istanbul, we map the Massive-STEPS place categories onto these seven labels." (or delete "(Section V-A)" in III).

**15. Figure-citation style drifts: "(Figure 3)" in the Discussion vs "(Fig. 3)" in the intro.**
- VII (07_discussion.tex:13): "the region gain rises with the number of regions (Figure~\ref{fig:deltas})." — intro (01_introduction.tex:43): "(Fig.~\ref{fig:deltas})".
- Wrong: mid-sentence parenthetical figure references use two styles (sentence-initial "Figure 3 plots" in VI-B is fine under IEEE style).
- Replace in 07_discussion.tex:13: "(Figure~\ref{fig:deltas})" with "(Fig.~\ref{fig:deltas})".

**Overall verdict.** As cross-section machinery this paper is unusually tight: the abstract's numbers, the contribution list, the results, and the figure all agree to the decimal; category consistently precedes region; the pre-registered superiority/non-inferiority framing is stated once in V-C and used identically in VI-B, VI-C, VII, and VIII; and hard terms (macro-F1, TOST, pp, seed, majority-class floor, ceiling) are glossed at or before first use. The residual problems are few but real and cluster at the seams: two claim-strength mismatches between sections that cite each other (the conclusion's "each under its own protocol" and the intro's "the first" vs II-B's "underexplored"), one undefined term doing real work ("region pathway", which matters for understanding the freeze control), one orphaned name (Check2HGI), and a handful of small naming drifts (Markov floor vs Markov-K, non-U.S. vs international, reference point vs floor, at vs on) that each cost a careful reader a moment of doubt. All fifteen are local edits; none requires restructuring, and fixing the top three would remove the only places where a reviewer could catch the paper disagreeing with itself.