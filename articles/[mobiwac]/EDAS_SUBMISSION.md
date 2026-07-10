# MobiWac 2026 — EDAS submission record

> The exact answers entered into the EDAS paper-registration form, kept so we can reproduce or
> edit them later. Title, abstract, and keywords stay editable in EDAS up to the deadline.

## Conference / track

- **Venue:** The 23rd International Symposium on Mobility Management and Wireless Access (MobiWac 2026)
- **Track:** Regular Paper
- **EDAS form:** https://edas.info/newPaper.php?c=35168&track=137247
- **Submitting account:** vitor.h.oliveira@ufv.br (EDAS user 2563350)
- **EDAS paper ID:** #1571313639
- **Manage / edit the paper:** https://edas.info/showPaper.php?m=1571313639 (title, abstract, keywords, authors, and manuscript upload are all editable here up to the deadline)
- **Status (2026-06-25):** Step 1 (Register paper) submitted; Step 2 (Add authors) done (3 authors confirmed below). Step 3 (Upload manuscript) still pending.

## Step 1 — Register paper

### Title
Predicting the Next Category and Region of a Visit: A Check-in-Level Multi-Task Study on Mobility Data

### Keywords (6) — paper order as of 2026-07-10 ("mobility data" first); update EDAS to match
1. mobility data
2. next-category prediction
3. next-region prediction
4. multi-task learning
5. check-in-level representation
6. location-based social networks

### Abstract (plain text, ~180 words)
> ⚠ **UPDATE AT UPLOAD TIME (found 2026-07-08 by the closing audit; paste block refreshed 2026-07-10
> after the glossary sweep + review-audit edits):** the abstract REGISTERED in EDAS at Step 1 is the
> OLD pre-review, pre-v17 draft. EDAS Step 1 is editable until the deadline (2026-07-11).
> **Paste the block below (the current paper abstract, LaTeX stripped) into the EDAS abstract field
> when doing the Step-3 upload. Also reorder the EDAS keywords to match the paper ("mobility data"
> first — see Keywords below).**

Location-based social networks (LBSNs) record where people go and what they do, one check-in at a time. If we can anticipate the next visit, mobile and urban services can cache content or reserve capacity ahead of demand. Two coarse questions about the next visit are usually enough: its category (the type of place) and its region (which part of the city). These are normally handled by separate models. We therefore test whether one model can learn both tasks, and what sharing a single model costs in per-task accuracy. We first build a check-in-level representation that describes each visit in its own context, instead of giving every place one fixed vector. Across six datasets, five U.S. states and one non-U.S. city (Istanbul), this lifts next-category prediction over a standard place embedding (about +28 to +40 macro-averaged F1), and most of the gain comes from the per-visit context. We then train one multi-task model for both tasks. At every dataset, it outperforms a dedicated category model (about +5 to +9 macro-F1), and on the next-region task it outperforms the dedicated region model at four of the six datasets, while matching it (statistically, within two points) at the other two. Across the five U.S. states, the gain on region grows with the number of regions; the two largest states are measured at a single random initialization and remain provisional. On the non-U.S. city, the result holds: the joint model outperforms on category and is slightly ahead on region.

> **SUPERSEDED registered text (do not reuse — provenance only):** "Location-based social networks
> record where people go and what they do... it beats a dedicated category model... One model wins
> both tasks, and the spatial win grows with scale... it beats on category and stays within two
> points on region." (Full old text in git history.)

### Topics (form allows 1–3; selected 3)
- AI-based mobility management
- Mobility models, control and management
- Social mobile networks and applications

## Step 2 — Authors

Order and identity match the published predecessors (CBIC `silva2025mtlnet`, CoUrb `paiva2026courb`);
all at NESPeD-LAB, Universidade Federal de Viçosa (UFV).

| # | Name (as matched in EDAS) | email | Role | Status |
|---|------|-------|------|--------|
| 1 | Vitor Hugo De Oliveira Silva | vitor.h.oliveira@ufv.br | submitting author | ✅ added |
| 2 | Germano dos Santos | germano.santos@ufv.br | co-author | ✅ added |
| 3 | Fabrício Aguiar Silva | fabricio.asilva@ufv.br | co-author (advisor) | ✅ added |

> All three matched their existing UFV/EDAS profiles. Order on EDAS is Vitor → Germano → Fabrício
> (student first, advisor last); it can be dragged to reorder ("Drag to change order" column) and
> edited later. Other CBIC/CoUrb collaborators (Ingred F. Almeida, Tarik S. Paiva, Felipe T.
> Sousa) are not added here; add them in EDAS if the author list changes.

## Step 3 — Upload review manuscript

- **Status 2026-07-10:** the manuscript is a complete, compiling **8-page** IEEE draft (`src/main.pdf`,
  the standard free budget — no extra-page fee), through the glossary v3 sweep, the review-audit
  package (Eq. 1, page-1 code/data footnote, Holm+TOST cites), the 10-persona panel, and the
  compliance audit; 33 rendered refs, 0 undefined, clean build. Remaining before upload: (1) upload
  the PDF (8 pages, no fee), (2) **replace the registered Step-1 abstract with the block above and
  reorder the keywords**, (3) final visual proofread of the compiled PDF. **Deadline: 2026-07-11.**
- Reminder shown on the form: "Manuscripts should not contain page numbers, headers or footers."
