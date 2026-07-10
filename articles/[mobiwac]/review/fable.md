Part 1 — Recent MobiWac papers under this year's topics

Topics chosen (3): AI-based mobility management, Mobility models, control and management, Big data analytics for mobile
and wireless networks.

Venue facts first: there was no 2024 edition (MobiWac 21st = 2023 Montreal, 22nd = 2025 Barcelona, 23rd = 2026 Paris);
the 2025 edition moved from the ACM DL to IEEE Xplore with IEEE-style 8+2 formatting, which your build already matches.
The venue is small (15–17 papers, single track).

Most relevant recent papers, by topic:

AI-based mobility management — Vielhaus et al., "Handover Predictions as an Enabler for Anticipatory Service
Adaptations" (MobiWac 2022; the venue's canonical prediction→proactive-action template, already cited in your intro);
Almutairi et al., GRU blockage mitigation in mmWave (MobiWac 2023); Tsukamoto et al., DRL AP clustering for cell-free
massive MIMO (MobiWac 2023); Abdelreheem et al., attention-DQN Wi-Fi 7 scheduling (MobiWac 2025); Lindner et al.,
predictive QoS with proactive channel switching (MSWiM 2025).

Mobility models, control and management — Moura, Aquino, Loureiro, "On the Design of Mobility-Aware Systems: A Tourist's
Perspective" (MobiWac 2025) — the single closest paper to yours at this venue: Foursquare check-ins, POI co-visitation
graph, mobility-aware design; already cited (moura2025mobilityaware) and correctly positioned in your §3/§7. Also Kouam,
Viana, Pappalardo et al., "Beyond Aggregates: Individual Mobility and Traffic Dependencies" (MSWiM 2025) — venue-blessed
evidence that individual mobility modeling drives network resource outcomes; not currently cited, and it should be (see
change list). Fontes et al., TFT vehicle-trajectory forecasting (MSWiM 2025).

Big data analytics — Capanema, Silva, Villas, Loureiro, non-IID multi-model FL (MSWiM 2025 — note this is exactly the
Brazilian next-POI community; likely reviewers know your problem well); Chitauro et al., autoencoder 5G anomaly
detection (MobiWac 2025); Goyal et al., GAE-LSTM tracking demo (MobiWac 2023 — in-venue precedent for graph-embedding +
sequence hybrids).

What the committee actually accepts: roughly a third of 2023+2025 papers use ML, almost always in service of a network
function; the ML is methodologically modest (DQN, GRU, autoencoders — transformers only arriving in 2025). Formal
statistical testing is uncommon at this venue; a hard systems tie-in is not strictly demanded (Moura et al. got in with
descriptive analysis and no service evaluation), but the rewarded framing is "prediction as an enabler of anticipatory
management." No MobiWac/MSWiM 2022–2025 paper does next-category/next-region prediction from LBSN check-ins,
graph-infomax representations, or multi-task cross-attention — the submission would be novel at the venue and lands on
the 2025-visible human-mobility trend line.

---
Part 2 — TPC Review, MobiWac 2026

1. Summary

The paper studies two coarse next-visit prediction tasks on LBSN check-ins — next-category (7 classes) and next-region (
520–8,501 census tracts / mahalle) — across five Gowalla U.S. states and Istanbul. Part 1 introduces Check2HGI, a
check-in-level extension of hierarchical graph infomax that gives each visit a contextual vector instead of one fixed
vector per place, lifting next-category macro-F1 by +27.6 to +39.6 over a place-level HGI baseline under a matched
protocol. Part 2 trains one joint cross-attention model for both tasks and reports that it outperforms per-dataset-tuned
dedicated category models everywhere (+5.3 to +9.4) and outperforms or TOST-matches (±2 pp) dedicated region models,
with the region gain rising with region count. Four datasets are n=20 (4 seeds × 5 folds) with Holm-corrected paired
tests; California and Texas joint cells are single-seed and labeled provisional. The mobility-management motivation is
explicitly not evaluated as a service; a quantified "usage sketch" (shortlist enrichment, centroid distances) stands in
for a systems result.

2. Strengths

1. Statistical discipline far above this venue's norm (§5.3, §6.2). Pre-registered TOST margin with a
   deployment-grounded rationale, Holm correction, paired designs, power analysis, and — rare anywhere — an honestly
   reported significant deficit: "At Alabama the interval sits entirely below zero, a small but statistically
   significant deficit, still well within the two-point margin" (06_results.tex). Verbs are bound to tests throughout.
2. The leakage story is proactively closed (§5.2). The per-fold rebuild audit ("rebuilding the representation per fold
   from its training users only... The effect is within fold noise"), the 13–27-point region-transition-prior inflation
   disclosure, and the baseline-parity sentence ("no baseline input is held to a weaker standard than ours") anticipate
   exactly the attacks a skeptical committee would mount.
3. Honest cost accounting (§4.2). "about 4.2 million parameters at Alabama against 1.1 million for the two combined...
   What the single model buys is operational rather than arithmetic" — most MTL papers hide this; stating it converts a
   weakness into credibility.
4. Reproducibility is concrete (§8). Code URL, both data sources public with exact links, metric conventions defined
   including the OOD-as-error rule for Acc@10 ("a visit whose true region is absent from that fold's training data
   counts as an error," §5.3), and the epoch-selection rule disclosed with a single-checkpoint reproduction bound (
   §6.2).

3. Weaknesses, ranked by severity

W1 (major) — Nothing measured is a mobility-management quantity; the venue bridge is motivation-only. §3: "We build and
evaluate no such service here"; §7: "This remains motivation, not a measured service result." At MobiWac, this is the
first thing a systems reviewer says aloud. The usage sketch (§7) is thoughtful — shortlist enrichment "more than five
hundred times better than picking ten at random," 3–8 km near-miss medians against a 20–241 km random-pair floor — but
these are still prediction properties, not a service metric (prefetch hit-rate, paging cost, handover preparation lead
time). Mitigating: the 2025 program shows the committee accepts LBSN/POI mobility papers without a service evaluation (
Moura et al.), and the paper is candid rather than decorative about the gap. But among three reviewers, expect one to
score this as "wrong venue," and the paper's defense is one paragraph of honest disclaimers rather than one number a
systems reader could act on.

W2 (major) — The scaling headline leans on single-seed provisional cells. The intro promises "the largest state (
California) showing the largest region gain" (01_introduction.tex, contribution 3), and Fig. 3's monotone trend has its
two rightmost, largest-gain points (TX +2.12, CA +2.20) from "a single seed over five folds (provisional)" (Table III
footnote). The paper says so plainly — "A multi-seed run at these two states is the remaining confirmation" (§6.2) — but
the confirmation is exactly the experiment that decides whether the third contribution survives. Five states with region
count and corpus size co-varying (acknowledged: "region count and corpus size co-vary here, so we read the trend across
the points rather than as a precise law") is already thin for a scaling claim; putting the pivotal points on n=5 makes
it hostile-committee bait. In discussion I would say: contributions 1 and 2 are solid; contribution 3 is a trend at four
confirmed points plus two provisional ones.

W3 (moderate) — The external baseline table is a fairness patchwork. Within one table (Table III): HMT-GRN is adapted ("
we keep its shared multi-task skeleton, add a region-transition prior..., and drop its graph components and hierarchical
beam search... It is a region-native model, not a reproduction of the complete published system," §5.4) and is scored "
on the visits whose region appears in training" (Table III footnote) — a different metric denominator from the
ours-columns' OOD-as-error convention (conservative in your favor, but two conventions in one row of comparisons); STAN
at California is two of five folds; ReHDM runs "under its own published protocol"; CTLE is end-to-end at Florida only,
frozen elsewhere. Every irregularity is disclosed, and the paper correctly anchors on the dedicated column ("the
like-for-like anchor remains the dedicated column," §6.2) — but the sentence "The joint model is also above every
external baseline reported, on both tasks, at every dataset" is doing rhetorical work its evidentiary base can't fully
carry, especially the CA STAN cell (2/5 folds). A reviewer may also ask why no adapted GETNext-class (2022+) or more
recent next-POI system appears on region.

W4 (moderate) — The CTLE control cannot attribute what §2.1 says it attributes. §2.1 claims: "We compare against CTLE
directly to show that the combination, not per-visit context on its own, is the source of the gain." But §6.1 concedes
the confound: "it pretrains on place identifiers and timestamps alone, so the category vocabulary never enters its
training, whereas our graph reads each visit's category and time of day as input features." CTLE differs from Check2HGI
in input information, not just in representation architecture — so the CTLE margin (+28.7 to +37.8) mostly measures the
missing category features, not "hierarchical-graph vs. sequence-model contextualization." The attribution actually rests
on the feature-concat control (place embedding + raw per-visit features, same model), which is the right experiment and
is present — but §2.1's framing sentence oversells the CTLE comparison specifically. A careful ML reviewer will catch
this in minutes.

W5 (minor-moderate) — Four paired observations per test. §6.2: "the tests pair the per-seed means (four paired
observations per dataset)... (paired t, corrected p<0.001 everywhere)." A paired t with n=4 (df=3) has unverifiable
normality; the p<0.001 claims are numerically plausible only because the cross-seed sds are tiny (±0.02–0.20), which
itself deserves a sentence (seed variance is nearly nil because the folds are frozen — say so). Relatedly, the ± symbol
switches meaning across cells ("± is the sd across seeds for four-seed cells, across folds for single-seed cells," Table
III footnote) — disclosed, but a reader comparing ±0.02 to ±0.45 across rows will misread precision unless warned in the
caption, not just the footnote.

W6 (minor) — Compression damage from the 10→8-page cut. Two orphaned antecedents survive: (a) 02_related.tex: "The field
increasingly models several granularities at once; in that work, category and region are auxiliary signals..." — "that
work" has no antecedent (it points at "the field"); (b) 03_problem.tex: "We do not predict the exact next place; these
are easier to learn and, for most uses, enough" — "these" reads as places on first pass; it means the two tasks, which
by this point in the sentence haven't been restated. Otherwise the cut held up well: Fig. 2 remains legible at column
width, the cut Fig. 3's four numbers are indeed fully in §6.1 prose, and the renumbered "Figure 3" references are
consistent.

W7 (minor) — Combination novelty. §2.1: "The novelty is this specific combination: per-visit context inside a
hierarchical graph-infomax representation." Honest, and the defusals (CTLE, DRRGNN, KGTB, HAMTL) are in place — but "
combination novelty" plus a +28–40 margin against a place-level baseline that structurally cannot see per-visit category
features will read to some as a strong result over a handicapped baseline. The feature-concat control is the answer;
make it carry more of the argument (see W4).

4. Detailed comments

- Table I: "Max len" 42,300 (TX) is a single user with forty-two thousand check-ins — a bot or a venue-spam account. One
  sentence on outlier handling (or the absence of filtering) would pre-empt a data-hygiene question.
- §3: "from about five hundred classes (Istanbul)" — it's 520; "about five hundred" is fine, but §5.3 writes "five
  hundred twenty" — pick one style.
- §6.2: the Markov region floor "computed under a non-overlapping windowing of the same data, indicative rather than
  protocol-matched" — good disclosure; consider whether an "indicative" number belongs in the running text at all, since
  its 12–23-point clearance invites protocol questions.
- §6.2, the freeze-probe: "we freeze the region pathway at the start of training so it can neither learn nor teach the
  category task, yet the full category lift survives" — this is one of the paper's best pieces of methodology (it
  converts "MTL synergy" into "stronger trunk") and it is buried in one sentence at three datasets. It deserves either a
  small table row or one more sentence on why AL/AZ/FL suffice.
- Fig. 3 (deltas): two metrics on one axis is disclosed in the caption, but the ±2 pp band visually applies to both bar
  families; the small "non-inferiority band" label is easy to miss at print size.
- Abstract: at ~250 words it is over the typical IEEE 150–200 comfort zone, and the sentence "with the two largest
  states measured at a single seed" is admirable honesty but syntactically bolted onto the scaling sentence. Tighten.
- Symbol load, Table III: bold + ↑ + ≈ + ° + † + ‡ in one table is at the limit; it works only because the footnote is
  complete. Keep the footnote; do not add a seventh mark.
- Missing citation: Kouam et al. (MSWiM 2025, "Beyond Aggregates") — venue-endorsed evidence that individual mobility
  modeling drives network resource outcomes; one sentence in §1 or §3 citing it strengthens the exact bridge W1 attacks,
  at near-zero page cost.
- Acronym discipline is good (TOST, MTL, POI, LBSN all expanded on first use); American English holds; I found no
  em-dashes.

5. VERDICT

Weak Accept. Confidence: 4/5.

Acceptance probability for a typical 3-person MobiWac committee: 55–70%. Reasoning: the paper's statistical and
disclosure rigor is well above the venue norm (where formal testing is rare), it lands on the human-mobility trend
visible in the 2025 program, and the closest in-venue paper (Moura et al. 2025) was accepted with less quantitative
substance and no service evaluation either. The two realistic downsides: one reviewer scoring venue-fit harshly (W1),
and one reviewer refusing to credit the scaling contribution on provisional cells (W2). Neither is likely to sink it
alone; both together could.

6. EXTRA COMMENTS — prioritized change list

The ONE change most likely to raise my verdict a full step: complete the CA/TX multi-seed joint runs (your P1). It
converts contribution 3 from "trend with two provisional points" to a confirmed monotone result across all five states,
removes the ° marks and the word "provisional" from the abstract, intro, Table III, §6.2, and §7 simultaneously, and
eliminates the strongest committee attack. Nothing in prose substitutes for it.

(a) Fixable in prose before submission (2026-07-11):

1. 02_related.tex §2.2 — replace "in that work, category and region are auxiliary signals" with "in those systems,
   category and region are auxiliary signals". Effect: removes the most visible compression scar. (Small, but reviewers
   pattern-match sloppiness.)
2. 03_problem.tex — "these are easier to learn" → "the pair is easier to learn" or "both are easier to learn than the
   exact place". Same effect.
3. 02_related.tex §2.1 — soften "We compare against CTLE directly to show that the combination... is the source of the
   gain" to something like "We compare against CTLE and a feature-concatenation control to separate the combination from
   contextualization alone and from feature injection." Effect: closes W4 without new experiments; the feature-concat
   control already does the work.
4. §1 or §3 — add one sentence citing Kouam et al. (MSWiM 2025) as venue-local evidence that individual-level mobility
   modeling drives network resource outcomes. Effect: measurably blunts W1 for the reviewer who checks whether you know
   the venue; ~2 lines.
5. §6.2 — add half a sentence explaining the tiny cross-seed sds (frozen folds ⇒ seed variance reflects initialization
   only). Effect: pre-empts the "±0.02 is implausible" margin note (W5).
6. Table III caption — move the ± semantics ("across seeds for four-seed cells, across folds for single-seed cells")
   from the footnote into the caption. Effect: kills the precision-misreading risk (W5).

(b) Response-letter material (have ready, don't add now):

1. The J1 joint-checkpoint re-score (every cell within ≤0.06 cat / ≤0.11 reg, no verdict changes) — the pre-built answer
   to "your deployment claim uses two epochs."
2. The AZ-ceiling sensitivity (+9.40 → ~+8.8 under the screened stronger arms) — the answer to "is your largest category
   gain an artifact of a weak ceiling."
3. The HMT-GRN adaptation rationale (dropped components serve the next-place head you don't predict) plus the
   observation that its in-training-region scoring convention favors the baseline.
4. STAN partial-fold defense: fold-count disclosure plus the per-fold consistency at the completed folds.

(c) Camera-ready / new-experiment material:

1. P1: CA/TX multi-seed (the verdict-changer above).
2. Extend the A4 leakage audit to CA/TX/Istanbul, so the transductivity null covers all six datasets rather than "3 of
   6" (currently the audit's scope is quietly narrower than the claim set it defends).
3. A single proxy service metric — e.g., simulated edge-prefetch or capacity-pre-positioning hit-rate at tract
   granularity driven by the model's top-10 regions, against a Markov-driven and a static-popularity policy. Even one
   figure of this converts W1 from a concession into a contribution and, at this venue, is worth more per page than any
   additional ML ablation.

Bottom line: this would not survive a hostile committee unscarred — W1 and W2 will both be said out loud — but it is
honest, unusually rigorous for MobiWac, novel at the venue, and defensible on every disclosed weakness. Weak Accept as
it stands; a solid Accept with the CA/TX seeds done or a proxy service metric added.