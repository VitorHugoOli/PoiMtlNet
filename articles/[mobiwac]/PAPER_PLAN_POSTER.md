# MobiWac 2026: Poster-Track Cut (4 pages): working plan

> **What this is.** A condensed, 4-page poster/short-paper version of [`PAPER_PLAN.md`](PAPER_PLAN.md). The
> reviewer panel ([`REVIEW_PANEL.md`](REVIEW_PANEL.md)) recommended the poster track because, for a regular paper,
> two blockers are live until the board freezes (the region headline is unmeasured at CA; the leak rebuttal needs
> sourcing). A poster turns those from blockers into an honest "preliminary cross-scale observation." It also
> fits the venue better: a network-motivated mobility-prediction study with no system result is a natural poster.
>
> **Same rules as the regular plan:** American English; plain, motivation-first, first-person plural; simple
> words; tasks named **next-category** and **next-region** (never activity/area); no em-dash; claim discipline
> (say "non-inferior within a 2-point margin", never "ties/Pareto"). Numbers in ⟨angle brackets⟩ are placeholders
> to fill from the board. Citation keys are from the verified BRACIS bib (see PAPER_PLAN §8).
>
> ⚠ **Verify the poster format first:** the MobiWac poster page historically asked for **ACM** style and 4 pages,
> while the regular paper asks for IEEE. Confirm the template and page limit on EDAS / with the TPC chair before
> writing the camera copy.

---

## 0 · What changed from the regular cut

- **Two contributions, not three.** The cross-scale "when it costs" account folds into the results, not a
  separate bullet. Headline = (1) the representation, (2) the joint-model finding.
- **The region story is stronger now (and provisional).** Per Vitor (2026-06-23, new data, unfrozen): the **MTL
  region Acc@10 now beats the STL region ceiling at FL and the small states (AL, AZ)**; **TX and CA are being
  measured.** So the poster's Part 2 is: on the states we measure, one model beats **both** dedicated models. We
  present this as a cross-scale observation, not a monotone law, and we say plainly which states are measured and
  which are in progress.
- **The leak rebuttal is one honest paragraph**, not a 290-word defense.
- **Mobility-management stays motivation only.** One or two sentences, right-sized to tract granularity (regional
  demand, content staging, capacity), not handover or cell selection.
- **Dropped for space:** the related-work survey (compressed to 3 to 4 sentences), the optimizer discussion, the
  recipe details, the per-visit-mechanism deep dive (kept as one figure), the discussion-section limitations
  (compressed to two sentences), and the system "usage illustration".

---

## 1 · Thesis and title

**Thesis (poster).** *A check-in-level representation makes next-category prediction far more learnable than a
fixed per-place embedding, and a single model that predicts next-category and next-region together beats both
dedicated models on the states we have measured so far.*

**Working title:** *One Model for Next-Category and Next-Region Prediction: A Check-in-Level Multi-Task Study.*
(Keep "beats" out of the title until TX and CA are in; a poster title that names the tasks and the method is
safe.)

---

## 2 · The 4-page cut, section by section (condensed draft paragraphs)

### Abstract (about 110 words): draft

> Location-based social networks record where people go and what they do, one check-in at a time, and
> anticipating the next visit can help mobile and urban services act ahead of demand. We study two questions that
> are easier than predicting the exact next place: the next category (the kind of place) and the next region (the
> area). We build a check-in-level representation that describes each visit in its own context, instead of giving
> every place one fixed vector, and we train one model to predict both tasks at once. Across five U.S. states the
> representation improves next-category prediction by a wide margin over a standard place embedding, and on the
> states we have measured a single joint model beats both a dedicated category model and a dedicated region
> model. We report what is measured and what is still in progress.

### §1 Introduction (about half a column): draft

> Location-based social networks such as Gowalla [Cho2011] let people share the places they visit, and together
> these check-ins let us study how people move through a city. A useful question is what a person will do next,
> because a system that can anticipate it can prepare ahead of time instead of reacting; for a mobility-aware
> service this can mean staging content at the right area or planning capacity ahead of demand. Predicting the
> exact next place is hard and often more than a service needs, so we study two coarser questions: the next
> *category* (the kind of place) and the next *region* (a census tract). They pair naturally, one for intent and
> one for geography, and the question we study is whether one model should learn both at once and what sharing
> costs.
>
> We make two changes. First, a check-in-level representation: instead of one fixed vector per place, each
> check-in gets its own vector that carries its context (time, nearby places, recent trail), building on
> hierarchical graph representations of places [huang2023hgi] and the infomax idea behind them
> [velickovic2019dgi]. Second, a single model that predicts the next category and the next region in one forward
> pass. **Our contributions are: (1)** a check-in-level representation that improves next-category prediction by a
> large, consistent margin over a standard place embedding (+⟨15 to 29⟩ macro-F1 across five states), with a
> controlled test showing the gain comes from the per-visit context; **(2)** a multi-task model for next-category
> and next-region that, on the states we have measured, beats both a dedicated category model and a dedicated
> region model in one forward pass, evaluated across five U.S. states of different sizes and a non-U.S. city.

### §2 Method (about three quarters of a column): draft + one figure

> **From place embeddings to per-visit embeddings.** A place embedding turns a point of interest into a vector.
> Deep Graph Infomax [velickovic2019dgi] learns such vectors by contrasting real graph neighborhoods against
> shuffled ones, and Hierarchical Graph Infomax [huang2023hgi] adds structure, place to region to city. Both give
> one vector per place, so two visits to the same café are identical. Our representation keeps that hierarchy but
> adds a check-in level beneath the place, so each visit is represented in its own context: the graph has four
> levels (check-in, place, region, city), visit nodes carry their category and time, and we train it with the
> same infomax objective, then read out one vector per check-in.
>
> **The joint model.** One model with a shared part that both tasks use and two outputs, where the region output
> keeps its own path. The deployment property is simple: one model, one forward pass, two predictions. The loss
> is a fixed-weight sum with both outputs on plain unweighted cross-entropy.

⭐ **Figure 1 (the one figure that carries the method):** the data flow, raw check-in trail to a four-level graph
to a per-visit representation to short windows to two outputs, with the shared part and the private region path
drawn. (Recreate from current results; do not reuse the old BRACIS figures, which show old data and the old
fully-shared architecture.)

### §3 Experimental setup (about half a column): draft + Table 1

> **Data.** Five Gowalla U.S. states of different sizes and one international city (Istanbul, from Massive-STEPS),
> on purpose, to see whether the findings hold across small and large and across U.S. and non-U.S. settings
> (Table 1). The next-category label is one of seven classes; the next-region label is a census tract (a mahalle
> for Istanbul), from about one thousand to about eight thousand classes.
>
> **Protocol.** We use five-fold cross-validation split by user, so a user's visits never cross the train and
> test boundary, and we reuse the same splits across all conditions so paired tests are well defined. The
> single-task ceilings use the same output classes as the joint model, so any difference between them is the
> sharing, not the output capacity. The region-transition prior is built per fold from training data only. The
> representation is trained without task labels, so no label leaks into the model inputs; we also rebuilt it from
> each fold's training users only and saw a negligible change on both tasks (within ⟨a third of a point⟩).
>
> **Baselines and metrics.** We compare against the place embedding (HGI), a check-in-level competitor (CTLE), a
> Markov transition baseline, and a strong region model (STAN). We report next-category macro-F1 (so rare
> categories count) and next-region Acc@10, each against a floor (majority class; Markov-1).

**Table 1 (real numbers, from `PAPER_PLAN.md §5`):** per state and Istanbul: check-ins, users, regions, windows.
(Window counts re-fill under the overlapping windows; max/avg sequence length and sparsity computed from the
data.)

### §4 Results (about one column): two parts, two reads

> **The representation carries category.** The check-in-level representation beats the place embedding on
> next-category macro-F1 by +⟨15 to 29⟩ points across the five states, and a controlled test shows most of the
> gain (about ⟨64 to 90⟩ percent) is the per-visit context itself, not extra training signal. On next-region the
> two representations are even, so the benefit is uneven across the two tasks. *Read it as: per-visit context,
> not extra training, is what makes the category task learnable.*

⭐ **Figure 2:** the per-visit result (place embedding vs per-place-pooled vs check-in-level next-category
macro-F1 across the five states, with the per-visit share). Recreate from current numbers.

> **One model, two predictions.** A single joint model, in one forward pass, beats both dedicated models on the
> states we have measured: it beats the dedicated category model by about +⟨3 to 4⟩ points, and on the new data
> it also beats the dedicated region model at AL, AZ, and FL. The category win comes from the shared part being a
> stronger category encoder at no extra cost, not from the region task teaching category. The two largest region
> states (TX, CA) are being measured; we report the measured states and present this as a cross-scale
> observation, not a law. *Read it as: at the sizes we have measured, sharing helps both tasks at once.*

**Table 2:** joint model vs the two dedicated ceilings, per state, both tasks, with measured cells marked and TX
and CA marked in progress.

### §5 Conclusion (about a quarter column): draft

> A check-in-level representation makes next-category prediction far more learnable, and a single model that
> learns next-category and next-region together beats both dedicated models on the states we have measured. We
> are finishing the two largest region states, and a natural next step is to connect these predictions to a
> concrete mobility-aware service. The takeaway is a simple reading: a per-visit representation carries the
> semantic task, and one shared model can serve both tasks at once.

---

## 3 · Figures, tables, and references (poster budget)

- **Fig 1** (method data flow), **Fig 2** (per-visit category result), **Table 1** (datasets), **Table 2** (joint
  vs dedicated). That is the whole visual budget; a poster should not exceed two figures and two small tables.
- **References:** trim to about 12 to 15, from the verified BRACIS bib: DGI, HGI, STAN, GETNext, HMT-GRN, ReHDM,
  POI-RGNN, HMRM, Gowalla, Massive-STEPS, CTLE, our published CBIC and CoUrb papers (`silva2025mtlnet`,
  `paiva2026courb`), and Caruana for the MTL background.

---

## 4 · Claim discipline for the poster (unchanged from the regular plan)

- Say "**beats both dedicated models on the states we have measured**", never "everywhere" while TX and CA are
  open. On region, if a state is non-inferior rather than a beat, say "**non-inferior within a 2-point margin**",
  never "ties".
- Mark every provisional cell. Keep the per-visit share and the +15 to 29 substrate gain as the solid core.
- The category win is **a stronger shared encoder at no extra cost**, not "the region task teaches category".
- Mobility-management is **motivation only**, right-sized to tract granularity (regional demand, staging,
  capacity), not handover or cell selection.

---

## 5 · Open items before camera-ready

1. **TX and CA region cells** (in progress): they decide whether Part 2 is "beats both at the measured states,
   largest in progress" or can be stated more broadly. The poster is honest either way.
2. **Final numbers:** fill every ⟨placeholder⟩ from the frozen board; recreate both figures from current data.
3. **CTLE** scored leak-clean, so the representation comparison is fair.
4. **Source the leak-rebuild numbers** (the "within a third of a point" claim).
5. **Confirm the poster template and page limit** (ACM vs IEEE ambiguity) and the deadline.
6. **Bibliography:** port the verified BRACIS bib, de-anonymize, trim to the poster set.
