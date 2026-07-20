Findings ranked by value. Files are under `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/`.

---

**1. [TOP] Pervasive non-native preposition: "at" used for datasets, seeds, and task counts** — global, all sections.
- `main.tex` (abstract): "It outperforms a dedicated category model at every dataset"
- `sections/02_related.tex`: "rarely improve on a well-tuned fixed weighting at two tasks"
- `sections/06_results.tex`: "All six datasets are measured at four seeds for both models"

Standard English evaluates "on" a dataset, runs "with" N seeds/tasks; the systematic "at" (a Portuguese "em" transfer) is the single most audible non-native marker in the paper, occurring 20+ times. Fix: "on every dataset", "with two tasks", "with four seeds"; "at Alabama/Florida" (state names read as places) may stay, but "at every dataset / at four of the six datasets / at all six datasets" should become "on". Also flag the pile-up in 06_results.tex: "Fine-tuned together with the task model at Florida at its authors' defaults, at the same 64 dimensions and windowing as ours" → "Fine-tuned together with the task model on Florida, with its authors' defaults and the same 64 dimensions and windowing as ours".

**2. [TOP] Misattached comparison makes a ~42-point deficit read as two points** — `sections/06_results.tex` line 25:
> "CTLE reaches $33.45$ macro-F1 at its best epoch and $29.69$ at its final one (an epoch is one pass over the training data), about two points below the place embedding under the same best-epoch rule and below the check-in-level representation's $75.15$."

"about two points below ... and below ... 75.15" invites the reading "two points below 75.15", which is wrong by forty points. Replace: "..., about two points below the place embedding under the same best-epoch rule, and far below the check-in-level representation's $75.15$."

**3. [TOP] Garden-path apposition in the abstract's key sentence** — `main.tex` lines 70–71:
> "Across six datasets, five U.S. states and one non-U.S. city (Istanbul), this improves next-category prediction..."

"Across six datasets, five U.S. states and..." first parses as a three-item list (six datasets, five states, one city); the appositive needs bracketing. Replace: "Across six datasets (five U.S. states and one non-U.S. city, Istanbul), this improves next-category prediction..."

**4. Dangling modifier: the literature "built city services"** — `sections/03_problem.tex` line 15:
> "The mobility literature supports such preparation, having long built city services on location-based social network traces~\cite{silva2019urbancomputing}."

The participle "having long built" attaches to "The mobility literature", which builds nothing. Replace: "The mobility literature supports such preparation: city services have long been built on location-based social network traces~\cite{silva2019urbancomputing}."

**5. Unparseable baseline list (appositive comma collides with list commas)** — `sections/06_results.tex` lines 103–106:
> "the primary region-native comparison, HMT-GRN~\cite{lim2022hmtgrn}, a STAN~\cite{luo2021stan} trained from the raw check-ins and a ReHDM reference~\cite{li2025rehdm} on region, and POI-RGNN~\cite{capanema2023poirgnn} and a Markov floor on category."

The comma after "comparison" makes HMT-GRN look like a second list item, and the region/category grouping only emerges at the end. Replace: "on region, the primary region-native comparison HMT-GRN~\cite{lim2022hmtgrn}, a STAN~\cite{luo2021stan} trained from the raw check-ins, and a ReHDM reference~\cite{li2025rehdm}; on category, POI-RGNN~\cite{capanema2023poirgnn} and a Markov floor."

**6. Idiom error: "ten regions of 8,501"** — `sections/07_discussion.tex` line 16:
> "at California, ten regions of 8{,}501 contain the true next region 65.69 percent of the time"

"N of M" needs "out of" here; as written it reads like a partitive of the number itself. Replace: "at California, ten regions out of 8{,}501 contain the true next region 65.69 percent of the time".

**7. Number-style inconsistency: "%" and "percent" in adjacent sentences for the same kind of quantity** — `sections/06_results.tex` lines 39–41:
> "reaches a macro-F1 of only about $7\%$ ... a random top-ten guess is right at most about two percent of the time."

One reference-point sentence uses digit+%, the next spells "percent"; prose elsewhere spells "percent" ("5 to 27 percent", "65.69 percent"). Replace "$7\%$" with "7 percent" (keep "%" only in "90\% confidence interval", a fixed statistical form).

**8. Inconsistent name for the same control: "feature-concat" vs "feature-concatenation"** — `sections/06_results.tex` line 25: "A feature-concat control, the place embedding joined with the same raw per-visit features our graph reads" vs `sections/05_setup.tex` line 49: "a feature-concatenation control appending raw per-visit features...". The clipped form appears only once; a first-time reader may not link them. Replace in 06_results.tex: "A feature-concatenation control, ...".

**9. Ambiguous number order after "outperforms"** — `sections/06_results.tex` lines 127–129:
> "outperforms the dedicated category ceiling by $+8.58$ macro-F1 ($54.74$ versus $63.32$) and is slightly above the dedicated region ceiling at $+0.19$ Acc@10 ($75.16$ versus $75.35$)"

The sentence subject is the joint model, but its number comes second in both parentheses, so the reader must guess which figure is whose. Replace with directional phrasing: "($54.74$ to $63.32$)" and "($75.16$ to $75.35$)" — or label: "(dedicated $54.74$, joint $63.32$)".

**10. Agreement-strained equative plus loose participle** — `sections/05_setup.tex` line 49:
> "The second role is two representation controls attributing the category gain to our specific design, not to contextualization in general or extra features:"

A singular "role" *is* not two controls, and "attributing" floats (the controls do not themselves attribute). Replace: "The second role is played by two representation controls that attribute the category gain to our specific design, not to contextualization in general or to extra features:" (and, for parallelism, "The first role is played by the per-task state of the art..." at line 47).

**11. Awkward apposition "a fourth, check-in level"** — `sections/01_introduction.tex` line 20:
> "This adds a fourth, check-in level beneath the place, region, and city levels of hierarchical graph infomax"

"a fourth, X level" works with an adjective, not with the noun "check-in"; it momentarily reads as "a fourth-comma-check-in". Replace: "This adds a fourth level, the check-in, beneath the place, region, and city levels of hierarchical graph infomax".

**12. Misplaced parenthesis: the two weights attach to only the second term** — `sections/04_method.tex` lines 20–22:
> "Two small label-free auxiliary terms are added: a masked reconstruction of each place's aggregated category features and an anchor to a place embedding pre-trained, label-free, on the same data (weights 0.3 and 0.1)."

"(weights 0.3 and 0.1)" covers both terms but sits inside the second item, so it first reads as a property of the anchor. Replace: "Two small label-free auxiliary terms are added (weights 0.3 and 0.1): a masked reconstruction of each place's aggregated category features, and an anchor to a place embedding pre-trained, label-free, on the same data."

**13. Comma splice in the main table caption** — `tables/tbl3_results.tex` line 31:
> "ordered by region count. Category is macro-F1, region is Acc@10."

Two independent clauses joined by a comma. Replace: "Category is macro-F1; region is Acc@10."

**14. Direction of the "gaps" is unstated and the pronoun is far from its referent** — `sections/06_results.tex` line 25:
> "With fixed weights, fed to the same single-task model, it repeats the ordering at Alabama, Arizona, and Istanbul, with gaps of $+37.8$, $+37.0$, and $+28.7$ macro-F1."

"it" (CTLE) is several clauses back, and "gaps of +37.8" does not say who leads. Replace: "With fixed weights, fed to the same single-task model, CTLE repeats the ordering at Alabama, Arizona, and Istanbul, with the check-in-level representation ahead by $+37.8$, $+37.0$, and $+28.7$ macro-F1."

**15. Opaque clause "the trunk comes with no second model to serve"** — `sections/06_results.tex` line 100:
> "We therefore attribute the category gain to a stronger shared trunk, not to the region task teaching the category one; the trunk comes with no second model to serve (one model, one forward pass)."

The trunk does not "come with" models to serve; the intended point (the gain costs no extra deployed model) has to be reverse-engineered. Replace: "We therefore attribute the category gain to a stronger shared trunk, not to the region task teaching the category one; and this gain arrives with no second model to serve (one model, one forward pass)."

---

**Overall verdict.** The mechanics of this paper are unusually clean for a first-submission draft: I found no subject–verb errors, no article misuse (the classic Portuguese-speaker tell), no typos, and only one comma splice, which sits in a caption. What remains is a small set of genuine clarity faults — one systematically repeated preposition choice ("at" a dataset/seed-count where English wants "on"/"with"), a handful of sentences where an appositive or comparison attaches to the wrong thing (findings 2–5 are the ones most likely to actively mislead a first-time reader), and minor consistency slips in naming and number style. The register is exactly what the authors aim for: plain, careful, defensible aloud; sentences are sometimes densely packed with parenthetical qualifications, but they parse. Fixing the top five findings would remove essentially all moments where a reader stumbles or draws the wrong number from a sentence; the rest is polish.