# Application-scope breadth: are we using examples beyond mobility?

> **Why this file exists.** In `noth_star_consideration.md` point 3, the author asks: the
> Introduction's problem statement should motivate next category and next region with **application
> examples beyond mobility**. CBIC and CoUrb cite broader domains; MobiWac stayed mobility-only
> because it was a mobility conference, but the dissertation can broaden the scope. This file answers
> "are we doing this?" and gives the verified material to do it, fail-closed.

---

## 1. Direct answer: no, not yet — the drafts are mobility-framed only

Checked the drafted Fundamentals prose (`2.1_poi_prediction_tasks.tex`, `2.5_relevance.tex`) and the
storyline. Both frame the stakes **exclusively** through mobility and mobility-aware services. The
only application words that appear are "recommenders" in passing (`2.1` L65, describing CatDM). There
is **no** mention of the broader domains the author has in mind. So the author's instinct is correct:
the dissertation is currently inheriting MobiWac's deliberately narrow, venue-driven framing, and has
not yet exercised the wider scope that the coletânea frame permits.

This matters for the arc because MobiWac's stakes paragraph is the single strongest "why care" in the
corpus, and pass-1 recommended lifting it into Chapter 1 (G-9). If that lift copies MobiWac's
mobility-only framing verbatim, the dissertation misses the chance the author is pointing at: a
dissertation Introduction can motivate the two tasks across the *full* range of applications the
component papers already gestured at, then narrow to mobility as the setting where they are
evaluated.

## 2. What CBIC and CoUrb actually cite — the verified raw material

All keys below were confirmed to resolve in the component papers' `.bib` files this session. This is
the material the Introduction can draw on without any new citation.

**From CBIC** (`sections/intro.tex` L5, currently **commented out** in the source; `sections/basis.tex`
L25, uncommented):

| Domain | How CBIC frames it | Cite key (resolves in `CBIC___MTL/references.bib`) |
|---|---|---|
| Computer vision | MTL for joint object detection and segmentation | `kokkinos2016ubernet` |
| Natural language processing | MTL for joint POS tagging + named-entity recognition | `wei2022finetuned` |
| Healthcare | MTL for simultaneous diagnosis of multiple conditions | `lipton2015learning` |
| Recommendation systems | MTL modeling user preferences + item attributes together | `zhang2020interactive` |
| Urban planning | POI prediction/classification as an urban-planning challenge | `Xu2023` (also names recommendation) |

> Note: the four-domain sentence lives in a **commented block** in CBIC's `intro.tex`. It is real,
> cited material the author wrote, but it is not in the compiled CBIC paper. For the dissertation
> Introduction it can be revived and re-verified; treat each of the four as **[VERIFY at adaptation]**
> — open the cited work and confirm it supports the one clause attributed to it (AGENT_GUARDRAILS R3),
> because a commented-out line never passed a citation gate.

**From CoUrb** (`sections/related.tex` L1, L19):

- POI prediction/classification framed as challenges in **location-based recommendation** *and*
  **urban mobility analysis** (L1).
- The spatial encoders the paper adopts (SIREN, Sphere2Vec-M) originate in **species prediction and
  population estimation** (ecology / remote sensing), cited via `wu2024torchspatial` and
  `mai2023sphere2vec` (L19). This is a genuine cross-domain provenance: the geospatial machinery came
  from ecology before it reached POIs.

## 3. How to use it, honestly (the frame move)

**The shape:** open the Introduction on the two tasks as *general* prediction problems whose value is
not limited to LBSN navigation, name two or three concrete non-mobility uses with their citations,
then narrow to the mobility setting where the dissertation evaluates them. This widens the felt
stakes (pass-1 D-MISSING-3) without diluting the honest scope statement that the *experiments* are on
mobility check-in data.

**Concrete, grounded uses the two tasks map onto** (each traceable to a cited domain above; phrase as
illustration, not as measured capability):

- **Next category** -> *recommendation and content preparation* (what kind of place / item the user
  turns to next -- `zhang2020interactive`, `Xu2023`); *demand and staffing* by activity type.
- **Next region** -> *urban planning and resource placement* (which part of the city to provision --
  `Xu2023`, `Lim2022`); the map-partition target is the standard mobility formulation
  (`luca2021mobilitysurvey`).
- **The method's transferability** -> the geospatial encoders came from *ecology / remote sensing*
  (`wu2024torchspatial`), so the representation ideas are not mobility-specific; this is an honest
  breadth note, not a claim that the dissertation tested those domains.

**Honesty guardrails on this move (fail-closed):**
1. **Illustration, not evaluation.** The dissertation evaluates on mobility data only. Non-mobility
   uses are *motivating examples*; never phrase them as things the dissertation demonstrated. A
   sentence like "our model improves urban planning" would be an unlicensed scope widening -- the
   honesty law's exact failure mode. Write "next region is the kind of prediction that supports
   resource placement in urban planning [cite]," not "we improve urban planning."
2. **Re-verify the revived CBIC citations.** The four-domain sentence is commented out; opening and
   confirming each of `kokkinos2016ubernet`, `wei2022finetuned`, `lipton2015learning`,
   `zhang2020interactive` is required before any enters compiled dissertation prose (R1--R3). Mark
   **[VERIFY at adaptation]**.
3. **Keep the narrowing explicit.** After broadening, the scope statement (§1.4: mobility check-ins,
   Gowalla + Istanbul, next place not predicted) must still land plainly, so the widened motivation
   never reads as a widened claim.

## 4. Flags

- **[NEEDS SIGN-OFF]** -- the Introduction beat "these two tasks matter beyond mobility (recommendation,
  urban planning, and by encoder-provenance even ecology), and we evaluate them in the mobility
  setting." New connective framing of the dissertation's scope; strongly grounded in CBIC/CoUrb's own
  citations but assembled here for the first time. Route through personas 07 + 14.
- **[VERIFY at adaptation]** -- the four CBIC domain-breadth citations, because their source sentence is
  commented out and never passed a citation gate.
- **Fail-closed note** -- no new reference is proposed. Every domain named above is already cited in a
  component paper's bib; the work is revival + re-verification + assembly, not new literature search.
