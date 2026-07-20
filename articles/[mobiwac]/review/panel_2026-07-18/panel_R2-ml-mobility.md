Now I have read the full paper text and inspected the figures (Fig. 1 p.3, Fig. 2 p.4, Table I p.6, Table II + Fig. 3 p.7, Table III p.8). Note: the paper contains Figures 1–3 only; there is no Figure 4. Here is my review.

---

# Review — "Predicting the Next Category and Region of a Visit: A Check-in-Level Multi-Task Study on Mobility Data" (MobiWac 2026, R2)

## 1. SUMMARY

The paper studies joint prediction of two coarse properties of a user's next visit — its category (7 classes) and its region (census tract / mahalle, 520–8,501 classes) — on five Gowalla U.S. states and Istanbul (Massive-STEPS). It first proposes Check2HGI, a check-in-level extension of hierarchical graph infomax that adds an individual-visit level below the POI/region/city hierarchy, and reports very large next-category gains (+28 to +40 macro-F1) over an HGI place-level embedding under a controlled protocol. It then trains a single cross-attention multi-task model (semantic stream + private spatial path, fixed 0.75/0.25 loss weighting) and claims it beats a per-dataset-tuned dedicated category model everywhere (+5.3 to +9.3 macro-F1) and beats or is TOST-non-inferior (±2 pp) to a dedicated region model, with the region gain growing with region count. Modified/re-implemented external baselines (HMT-GRN, STAN, ReHDM, POI-RGNN, Markov) are all below the joint model, and a cascade rewiring of their own model performs identically to the parallel form.

## 2. SCORE

**Overall: Weak Accept**

| Criterion | Score (1–5) |
|---|---|
| Relevance to venue | 4 |
| Technical soundness | 3 |
| Novelty | 3 |
| Clarity | 4 |
| Reproducibility | 4 |

## 3. STRENGTHS

1. **Unusually disciplined statistical protocol.** Pre-assigned superiority vs. non-inferiority tests per dataset, TOST with a pre-fixed ±2 pp margin justified by the use case, Holm correction, 4 seeds × 5 user-disjoint folds, and paired per-seed means (Section V-C). This is well above the norm for this literature, and the honest reporting of the Alabama result as a "small but statistically significant deficit" (VI-B) is commendable.
2. **Honest disclosure of baseline caveats.** Table III footnotes plainly state HMT-GRN is "not a reproduction of the complete published system," STAN ran partial folds (TX 4/5, CA 2/5), and ReHDM is under its own protocol. Most papers bury this.
3. **A genuine mechanism control.** Freezing the region pathway to show the category gain is not task transfer but trunk/input strengthening (VI-B), and reporting it "as a finding, not a hypothesis," is intellectually honest — even though it undermines the MTL framing (see Weakness 2).
4. **The leakage self-audit exists.** Per-fold representation rebuild (≤0.33 pp movement at AL/AZ/FL) and the disclosure that an earlier whole-dataset transition prior inflated region accuracy by 13–27 points (V-B) show the authors take leakage seriously.
5. **Clear, plain-language writing** with reference floors given for every metric (majority-class macro-F1 ≈ 7%, random top-10 ≤ 2%), and released code/data.

## 4. WEAKNESSES (ranked)

1. **Potential backward (future-visit) leakage in the representation is not ruled out (IV-A, V-B).** The graph "link[s] a user's consecutive check-ins," and each check-in node's *input features include its category*. If these edges are undirected (the paper never says) and the encoder does ≥1 hop of message passing, the per-visit vector of the last window visit aggregates the features of the *next* visit — i.e., the label. The three integrity checks in V-B do not cover this case: the per-fold rebuild addresses cross-*user* leakage, not within-user future→past propagation, and it was run only at AL/AZ/FL. A +40 macro-F1 jump over a published next-category model (POI-RGNN, Table III) is an extraordinary result that makes this the paper's central unresolved soundness question.
2. **The headline "joint beats dedicated category everywhere" conflates MTL with input enrichment and capacity (IV-B, VI-B).** The joint model's category prediction attends to the spatial stream through a 4.2M-parameter trunk; the dedicated category model (1.1M combined for both dedicated models) apparently reads the semantic stream alone. The authors' own freeze control proves the gain is not from the region *task*. The fair dedicated baseline — a single-task category model given both input streams at matched capacity — is absent, so the +5.3 to +9.3 gain cannot be attributed to "one model, two tasks."
3. **Asymmetric feature access in the representation comparison (VI-A, Table II).** Check2HGI's node features include each visit's category and time; the HGI place embedding and CTLE (which "pretrains on place identifiers and timestamps alone") have no access to the category vocabulary. The paper acknowledges this for CTLE, but the control designed to close exactly this gap — the feature-concat control — is reported only qualitatively ("does not close the gap either," VI-A) **with no numbers anywhere in the paper**. As written, the +28 to +40 gain cannot be decomposed between "hierarchical per-visit graph" and "category features in the input."
4. **Protocol asymmetry between the two dedicated ceilings (VI-B).** The dedicated *category* model is "tuned per dataset over batch size and learning rate," but the dedicated *region* models "use the strongest fixed configuration." The region claims are precisely the thin ones (+0.19 Istanbul, +0.71 FL, +2.1/+2.2 TX/CA vs. a ±2 pp margin); an under-tuned region ceiling makes them easy. The one task where superiority is marginal is the one task whose baseline was not tuned.
5. **Internal inconsistency: Fig. 3 caption says "(TX and CA: one seed)" while Section VI-B and the Table III footnote say all six datasets use four seeds** and even quote 90% CIs at TX/CA (+2.10 to +2.13; +2.19 to +2.21). One of these is wrong; either the figure is built from different runs than the table, or the caption is stale. This must be reconciled.
6. **Overclaiming "outperforms" against the paper's own relevance margin (Abstract, VI-B, Table III).** The ±2 pp margin is justified as "below the granularity at which a service would behave differently" (V-C) — yet Istanbul (+0.19) and Florida (+0.71) are counted as datasets where the joint model "outperforms" and are bolded with ↑ in Table III. By the paper's own logic these are matches, and the honest headline is "equivalent at 4 of 6, superior at TX and CA," not "outperforms at 4 of 6."
7. **The cascade comparison is architecturally confounded and clashes with the trunk story (V-D, VI-B).** The cascade variant *removes the shared trunk entirely* and adds a ~1,000-parameter conditioning projection, so "chain vs. parallel" is confounded with "with vs. without cross-attention." Worse: if the trunk-less cascade matches the parallel model on both tasks within 0.25 pp, then the shared trunk contributes nothing measurable — directly contradicting VI-B's attribution of the +5–9 category gain "to a stronger shared trunk." Something in this pair of results is under-described or inconsistent.
8. **Method under-specification (IV-A).** The graph encoder is never described: GNN type, depth, edge direction, how per-visit vectors are extracted at inference, the form of the two auxiliary losses. These details live only in the code; a paper claiming a representation contribution should specify the representation.

## 5. MUST-FIX BEFORE ACCEPTANCE

1. **Resolve the future-edge leakage question:** state edge directionality and encoder receptive field; add (or report, if already run) a control in which consecutive-visit edges from the target and later visits into the window are removed at embedding time. If the gain survives, weakness 1 dissolves.
2. **Report the feature-concat control numbers** (currently a claim with no data) and add a dedicated category model that receives both streams at comparable capacity, or explicitly re-scope the category claim from "multi-task gain" to "joint architecture with richer inputs matches/beats a narrower dedicated model."
3. **Fix the Fig. 3 caption vs. Table III / VI-B seed-count contradiction** (one seed vs. four seeds at TX/CA).
4. **Align claim language with the pre-registered margin:** Istanbul and Florida region results should be described as within-margin (matches), consistent with the ±2 pp rationale of V-C; reserve "outperforms" for TX/CA.

## 6. DETAILED COMMENTS

- Table III: HMT-GRN "scored on visits whose region appears in training" while your model counts unseen-region visits as errors — this favors HMT-GRN, so your win is conservative; worth one sentence in the main text, since it currently reads as a footnote asymmetry.
- The Markov *region* floor (51–72 Acc@10, within 4.9–10.3 pp of the joint model) appears only in prose (VI-B) and not in Table III, whereas the far weaker external baselines do. A first-order transition prior capturing most of Acc@10 is important context for how much headroom the region task has; put it in the table.
- CTLE: "Fine-tuned together with the task model at Florida at its authors' defaults, at the same 64 dimensions" — 64-d is *your* setting, not CTLE's default (128-d in the original paper); the sentence contradicts itself. Also, CTLE landing *below* the place embedding on next-category is unsurprising given it never sees categories; as a control it mostly proves category features matter (see Weakness 3). A CTLE variant with category tokens would be the informative cell.
- Abstract says "+28 to +40"; Table II's smallest delta is +27.63 (AZ). Round honestly or say "about."
- "measured during development, not shown" (II-C, gradient correlation): either show it or cut it; unverifiable claims weaken an otherwise careful paper.
- Table II is seed 0 only, unlike everything else at 4 seeds — say why, or run the seeds; the note that AL and Istanbul's identical 26.56 is "a coincidence of two independent runs" is appreciated but would be moot with multi-seed means.
- V-B integrity check covers AL/AZ/FL only; TX, CA, and Istanbul — including both datasets where you claim >2 pp region wins — are unaudited. State this limitation explicitly.
- Seed sds of ±0.01–0.02 pp at TX/CA (Table III) are extremely tight; plausible given millions of windows, but with n=4 paired seeds the TOST normality assumption is doing real work — a sentence on robustness (e.g., sign consistency across seeds) would help.
- "to our knowledge, the first to treat fine-grained region as an end target of equal standing" (II-B) — hedged and defensible given the DRRGNN/[19] discussion, but grid-cell next-location prediction (e.g., DeepMove-style setups) is close enough that I would soften "first."
- Istanbul (fewest regions) being *ahead* on region (+0.19) actually breaks the "gain grows with region count" trend line; restricting the trend to U.S. states (Fig. 3) is legitimate but the abstract's "Istanbul... is also slightly ahead" reads like having it both ways.
- Fig. 2 is clear; Fig. 1's "region vectors (one per region)" box would benefit from noting these feed the *spatial* stream of Section IV-B — the connection is currently only in prose.
- Venue fit: the caching/provisioning motivation (refs [3], [4] — a MobiWac paper) is appropriate, and scoping radio-level decisions out (Section III) is honest; but note the paper builds and evaluates no networking component, which caps its relevance score.

## 7. QUESTIONS TO THE AUTHORS

1. Are the consecutive-check-in edges directed (past→future only) or undirected, and how many message-passing hops does the encoder use? Concretely: can the embedding of the last visit in a window aggregate any feature (including the category) of the *target* visit? If undirected, please report the representation's category gain with future edges masked.
2. How can the trunk-less cascade match the parallel model's category macro-F1 within 0.25 pp (VI-B) when the +5.3 to +9.3 category gain over the dedicated model is attributed to the shared trunk? What, architecturally, distinguishes the cascade's category branch from the dedicated category model?
3. What are the numbers for the feature-concat control, and does the joint model's category advantage survive against a single-task category model given both the semantic and spatial streams at matched parameter count?

---
*Reviewed from the PDF only (9 pages; note the paper contains Figures 1–3 — there is no Figure 4). Would I cite it? For the region-as-equal-target framing and the statistical protocol, possibly; for the representation result, not until Question 1 is answered.*