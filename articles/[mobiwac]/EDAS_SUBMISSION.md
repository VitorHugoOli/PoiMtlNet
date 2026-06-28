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

### Keywords (6)
1. next-category prediction
2. next-region prediction
3. multi-task learning
4. check-in-level representation
5. location-based social networks
6. mobility data

### Abstract (plain text, ~180 words)
> Note: this is the PAPER_PLAN §4 draft with the citation marker removed (EDAS abstract is plain
> text). It uses qualitative wording, not hard numbers; swap in concrete deltas later if preferred.

Location-based social networks record where people go and what they do, one check-in at a time, and if we can anticipate the next visit then mobile and urban services can act ahead of demand. Two questions are usually enough: the next category (the kind of place) and the next region (which part of the city). Usually they are handled by separate models, so we ask whether one model can learn both, and what it costs to share one representation. We first build a check-in-level representation that describes each visit in its own context, instead of giving every place one fixed vector. Across five U.S. states, this representation improves next-category prediction by a wide margin over a standard place embedding, and most of the gain comes from the per-visit context. We then train one model for next-category and next-region together. On every state we measure, it beats a dedicated category model, and on region it beats the dedicated model where the region space is large and stays within a two-point margin where the region space is small. One model wins both tasks, and the spatial win grows with scale. On a non-U.S. city (Istanbul) the result is consistent: it beats on category and stays within two points on region.

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

- The manuscript is now a complete, compiling 9-page IEEE draft (`src/main.pdf`); the earlier submission blockers
  are closed (see `CLAUDE.md` for current state). Remaining before upload: apply the accepted Germano edits
  (`REVIEW_GERMANO.md`), restore `IEEEtran.bst`, and a final proofread. Registration reserves the slot; the
  manuscript is uploaded separately before the manuscript deadline.
- Reminder shown on the form: "Manuscripts should not contain page numbers, headers or footers."
