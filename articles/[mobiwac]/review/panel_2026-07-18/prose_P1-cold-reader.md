P1 COLD-READ REPORT — /Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/main.pdf

Read once, start to finish. Findings ranked by how badly they broke the first pass. Quotes are verbatim from the PDF/tex.

---

**1. [TOP 3] Abstract — the Istanbul sentence reads as contradicting the sentence it is attached to.** (tex: main.tex ~L78-80)
> "Across the five U.S. states, the region gain grows with the number of regions; Istanbul, the city with the fewest regions, is also slightly ahead on region."

Where/why I broke: last sentence of the abstract. I had just been told the gain grows with region count; then the city with the *fewest* regions is "also slightly ahead" — the word "also" makes Istanbul sound like a confirmation when it is logically a counterexample, and no explanation is available yet. I stopped and re-read twice.
Fix: "Across the five U.S. states, the region gain grows with the number of regions. Istanbul, our non-U.S. dataset, sits outside this trend: despite having the fewest regions, it too shows a small region gain."

**2. [TOP 3] §VI-B — "no second model to serve" is opaque.** (sections/06_results.tex:100)
> "We therefore attribute the category gain to a stronger shared trunk, not to the region task teaching the category one; the trunk comes with no second model to serve (one model, one forward pass)."

Where/why: end of the freezing-control paragraph. "The trunk comes with no second model to serve" — who serves whom? I could not parse it on first pass; only the parenthetical hinted at a deployment argument.
Fix: "We therefore attribute the category gain to a stronger shared trunk, not to the region task teaching the category one; and since the trunk is part of the deployed model itself, the gain costs no second model at serving time (one model, one forward pass)."

**3. [TOP 3] §V-D — the STAN adaptation clause is unparseable on first pass.** (sections/05_setup.tex:47)
> "we adapt its output to region candidates, which keep the spatial identity its candidate-distance matching needs, and do not adapt it to category, where that mechanism has no meaning."

Where/why: mid-baselines. "keep the spatial identity its candidate-distance matching needs" is three stacked noun phrases with no article — I re-read it three times to find the verb structure.
Fix: "we adapt its output layer to rank region candidates, which are spatial objects and therefore still support STAN's candidate-distance matching; we do not adapt it to category, where distance matching has no meaning."

**4. §II-C — a 60-word sentence with a forward reference to a concept not yet defined.** (sections/02_related.tex:89-95)
> "The reason is visible in the gradients, measured during development on the same joint architecture over an earlier, non-overlapping windowing of the data: the cosine similarity … averages +0.001 across training (four seeds each at three of our six datasets, per-dataset means within ±0.003), so a balancer has no conflict to resolve, consistent with the mechanism test in Section VI-B; this is a finding for this pair of tasks, not a general rule."

Where/why: still in Related Work, and suddenly I am inside an experimental detail; worse, "non-overlapping windowing" means nothing yet — windows are only defined in §V-B. I lost the thread mid-sentence.
Fix (split in three): "The reason is visible in the gradients. Measured during development on the same joint architecture (on an earlier preparation of the data), the cosine similarity between the next-category and next-region updates on the shared trunk averages +0.001 across training (four seeds each at three of our six datasets, per-dataset means within ±0.003). A balancer therefore has no conflict to resolve, consistent with the mechanism test in Section VI-B; this is a finding for this pair of tasks, not a general rule."

**5. §V-D and §VI-A — "fixed weights" (CTLE) collides with the paper's fixed loss weights.** (sections/05_setup.tex:49; 06_results.tex:25)
> "run end to end at Florida and with fixed weights at Alabama, Arizona, and Istanbul" / "With fixed weights, fed to the same single-task model, it repeats the ordering…"

Where/why: I had just read about "fixed-weight" loss training (Eq. 1) and a "tuned fixed task weighting"; here "fixed weights" means *frozen embedding parameters*, which took me a beat to work out.
Fix: "run end to end (fine-tuned) at Florida and with frozen weights (no fine-tuning) at Alabama, Arizona, and Istanbul" and "With frozen weights, fed to the same single-task model, CTLE repeats the ordering…"

**6. §VI-A — "gaps of +37.8, +37.0, and +28.7" never says between what and what.** (sections/06_results.tex:25)
> "it repeats the ordering at Alabama, Arizona, and Istanbul, with gaps of +37.8, +37.0, and +28.7 macro-F1."

Where/why: end of the CTLE paragraph; "it" is CTLE, so a plus-signed "gap" attached to CTLE momentarily reads as CTLE winning. I had to infer the direction.
Fix: "it repeats the ordering at Alabama, Arizona, and Istanbul, where our representation leads CTLE by +37.8, +37.0, and +28.7 macro-F1."

**7. §VI-B — a Markov *region* floor materializes that Baselines (§V-D) never introduced.** (sections/06_results.tex:110)
> "A simple first-order Markov region floor, recomputed under the same sliding-window protocol and fold splits as our models, reaches 51 to 72 Acc@10 across the datasets"

Where/why: §V-D carefully enumerated three roles of comparisons, with Markov only for category; a region Markov appearing here made me flip back to check whether I had missed it. "That's odd" moment.
Fix: add to §V-D after the Markov-K sentence: "For region, we additionally compute a first-order Markov floor over region transitions, under the same windows and folds." (then §VI-B can reference it as introduced).

**8. §V-B — the padded-windows sentence needs two reads.** (sections/05_setup.tex:27)
> "Windows running past the end of a history are padded and would all target the final visit; we keep the one full-context window ending there and drop the padded duplicates."

Where/why: the hypothetical "would all target" plus the unstated fact that several such windows exist per user made me reconstruct the geometry myself.
Fix: "Near the end of a user's history, every start position within the last nine visits yields a shorter, padded window whose target is the same final visit; we keep only the full-length window ending there and drop these padded duplicates."

**9. §IV-B — the private-spatial-path sentence stacks relative clauses, and "raw" contradicts the stream's definition.** (sections/04_method.tex:33)
> "the region output keeps a private spatial path, a small branch inside the one model (not a second model) that reads the raw spatial window directly, bypassing the shared trunk, and that the category task does not touch."

Where/why: the trailing "and that the category task does not touch" arrives after two other clauses and forced a re-read; also "raw spatial window" puzzled me because the spatial stream was just defined as *trained* region vectors — "raw" suggests something untrained.
Fix: "the region output additionally keeps a private spatial path: a small branch inside the one model (not a second model) that reads the spatial input window directly, bypassing the shared trunk. The category task does not touch this branch."

**10. §VI-B — the external-baseline list defeats first-pass grouping.** (sections/06_results.tex:102-106)
> "the primary region-native comparison, HMT-GRN [17], a STAN [13] trained from the raw check-ins and a ReHDM reference [32] on region, and POI-RGNN [31] and a Markov floor on category."

Where/why: "a STAN … and a ReHDM reference … on region, and X and Y on category" — the task-grouping only becomes clear at the very end; I re-parsed it once.
Fix: "on region, the primary comparison HMT-GRN [17], a STAN [13] trained from the raw check-ins, and a ReHDM reference [32]; on category, POI-RGNN [31] and a Markov floor."

**11. §V-D — "the next-place head that we do not predict," and "region-native" coined without a gloss.** (sections/05_setup.tex:47)
> "drop its graph components and hierarchical beam search, which serve the next-place head that we do not predict. It is a region-native model, not a reproduction of the complete published system."

Where/why: one does not "predict a head," so I stalled on the mismatch; and "region-native" is a made-up label whose meaning I had to guess (it recurs in Table II and §VI-B).
Fix: "drop its graph components and hierarchical beam search, which exist to serve its next-place target, a target we do not predict. The result is a region-native model — region is one of its original prediction targets — not a reproduction of the complete published system."

**12. §VI-B — "On principle" is the wrong idiom.** (sections/06_results.tex:115)
> "On principle, we prefer coupling the tasks in parallel rather than in a chain"

Where/why: "on principle" means refusing something as a matter of conscience; here the meaning is a design rationale, so the phrase reads oddly formal-moral.
Fix: "By design, we prefer coupling the tasks in parallel rather than in a chain".

**13. §VI-A — the CTLE-Florida sentence opens with three consecutive "at" phrases, and defines "epoch" to a CS audience.** (sections/06_results.tex:25)
> "Fine-tuned together with the task model at Florida at its authors' defaults, at the same 64 dimensions and windowing as ours, CTLE reaches 33.45 macro-F1 at its best epoch and 29.69 at its final one (an epoch is one pass over the training data), …"

Where/why: the subject (CTLE) arrives after three modifiers, and the aside defining "epoch" — while nearby terms like "region-native" go undefined — was a "that's odd" moment.
Fix: "At Florida, we fine-tune CTLE together with the task model, using its authors' defaults and the same 64 dimensions and windowing as ours. It reaches 33.45 macro-F1 at its best epoch and 29.69 at its final one — about two points below the place embedding under the same best-epoch rule, and far below the check-in-level representation's 75.15." (drop the epoch gloss)

**14. Intro, contribution 2 — "the first to treat" has no noun.** (sections/01_introduction.tex:31)
> "to our knowledge, the first to treat fine-grained region as an end target of equal standing"

Where/why: the bullet's subject is "A single model…", so "the first" dangles — first model? first study? Momentary snag.
Fix: "to our knowledge, the first work to treat fine-grained region as an end target of equal standing".

**15. Intro — "We propose two enhancements" with no object.** (sections/01_introduction.tex:19)
> "We propose two enhancements."

Where/why: enhancements *to what*? Nothing has been introduced to enhance — the previous paragraph poses a question, not a system. Small but real pause.
Fix: "We address this with two contributions." (matches the bullet list that follows)

---

**What read effortlessly (first pass):** §II-A (the DGI→HGI→CTLE→Check2HGI progression is a model of plain explanation for a cold reader); §III (the practical motivation and the honest radio-cell scoping); §IV-A (the graph construction — the best-written section of the paper); §V-C's superiority-vs-non-inferiority setup (dense but never lost me); §VI-C (crisp, ends on the strong "The U.S. result repeats on a different continent and region unit"); the Conclusion (clean, and the final sentence lands); and the Table III footnote pre-empting the 26.56 coincidence, which is exemplary reader care. Figure captions 1-3 all help rather than decorate.

**Overall verdict.** This is a genuinely readable paper for a first-time reader: the vocabulary is plain, nearly every technical term is glossed inline at first use, the argument structure (representation first, then joint model, then external check) is easy to hold, and the authors' habit of pre-answering the skeptical reader ("kept simple by design so that any improvement… comes from the shared representation"; "This remains motivation, not a measured service result") builds real trust. The friction is concentrated, not diffuse: a recurring habit of compressing an entire methodological argument into one sentence with stacked relative clauses (worst in §V-D's baseline descriptions and §II-C's gradient aside), a handful of coined or overloaded phrases that a cold reader cannot decode ("fixed weights" for frozen, "region-native," "no second model to serve," "raw spatial window"), one abstract-level sentence that appears to contradict itself (Istanbul vs. the region-count trend), and the pervasive non-native preposition tic "at every dataset / at Alabama" (harmless to comprehension, but "on" is what the reader expects and it is the one tell of non-native drafting that recurs on every page). Fix the fifteen items above — especially the top three — and the paper reads smoothly end to end; nothing here suggests structural rewriting is needed.