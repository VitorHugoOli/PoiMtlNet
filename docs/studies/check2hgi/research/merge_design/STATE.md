# Current state — merge_design study (2026-05-06 14:10)

This is the single big-picture document. Other files in this folder
are **per-design audits** (`DESIGN_*.md`), the **HGI-gap diagnostic**
(`AUDIT_HGI_GAP.md`), and **historical narrative** (`MERGE_DESIGN_NOTES.md`,
`SUMMARY.md` — kept for archaeology, not authoritative).

## The research goal

Merge HGI's POI semantics into Check2HGI without breaking next-category,
and **overcome HGI** on next-region (and eventually next-POI). Two axes:

- **Cat (next-category)**: c2hgi excels (F1 ≈ 40-43% AL/AZ); HGI fails
  (F1 ≈ 25-29%). Preserve canonical c2hgi cat performance.
- **Reg (next-region)**: HGI beats canonical c2hgi by 2-3 pp Acc@10.
  Close that gap; ideally beat HGI.

## Where we are now (leak-free, all states)

| Substrate | AL cat F1 | AL reg | AZ cat F1 | AZ reg | FL reg | fclass probe |
|---|---:|---:|---:|---:|---:|---:|
| canonical c2hgi | **40.76** | 59.15 | **43.21** | 50.24 | 0.6922 | 4 % |
| HGI | 25.26 | 61.86 | 28.69 | 53.37 | **0.7134** | 98 % |
| B (POI2Vec @ pool) | 41.51 | 61.49 | 43.91 | 52.59 | 0.6993 | 98 % |
| H (learnable POI table) | 40.97 | 62.35 | 44.14 | 52.30 | **0.7041 ✓** | 98 % |
| I (LoRA r=8) | 41.62 | 61.35 | 43.71 | 52.55 | 0.7002 | 98 % |
| J (H + λ-anchor) | 41.81 | **61.95** | 43.74 | 52.16 | **0.7034 ✓** | 98 % |
| M (B + cosine distill) | 41.31 | 61.56 | 43.67 | 52.45 | 0.7011 | 98 % |
| K (J + Delaunay GCN) | _not run_* | 0.6193† | _not run_* | 0.5209† | _skip_ | _not run_ |

*K cat F1 not measured — focus is reg axis. Build is c2hgi-pipeline-compatible (cat path detached). Estimated cat F1 ≈ J ≈ canonical.
†K AL/AZ reg numbers reported on the same leak-free protocol as the rest of the family.

✓ = strict Wilcoxon win at p=0.0312 vs canonical (5/5 folds).

### Headline reads

- **All B/H/I/J/M dominate canonical on both axes at AL+AZ** (cat
  non-inferior at TOST p<0.01, reg superior; J nominally beats HGI on
  AL by +0.10 pp).
- **At FL, only H and J are Wilcoxon-strict over canonical** at
  p=0.0312, both at +1.1 to +1.2 pp Acc@10. Both still 0.9-1.0 pp
  below HGI.
- **fclass probe goes from 4 → 98 %** for all merge designs — HGI-grade
  POI semantic recovery at no cat cost. This is a generality property,
  not a next-task gain.
- **K = J empirically** at AL+AZ. Adding HGI's Delaunay POI-POI graph
  contributes zero reg lift over J's anchor mechanism.

## What we've learned

### Settled (positive)

1. **POI2Vec at the pool boundary preserves cat by construction**
   (B/H/I/J/M all detach the cat path from the reg-side residual). Not
   one design regresses cat. The "shared encoder collapse" worry from
   the early POI2Vec-input probe (Design E) does not apply when the
   injection is post-pool.
2. **At AL/AZ, the merge mechanism gives ~+2 pp reg over canonical**,
   with J/M crossing the strict-Wilcoxon gate.
3. **At FL, the merge mechanism gives ~+1 pp reg over canonical**,
   with H/J crossing the strict-Wilcoxon gate. Cat preserved.
4. **fclass POI semantic recovery is robust** across states and designs.
   Free generality side-benefit independent of next-task results.

### Settled (negative — falsified hypotheses)

1. **Late-fusion (Design A) breaks both axes** (cat −9 pp, reg −10 pp at
   AL/AZ). Concat at the head is dead.
2. **Stop-gradient projector heads (Design E) on POI2Vec-augmented input
   do not save cat**. The failure is input-side, not gradient-side.
3. **Heterograph (Design D) leaks the cat label** through reverse
   visit-edges + 2-hop GCN (last-step linear-probe diagnostic confirmed
   +20 pp leak). Disqualified despite numerical dominance.
4. **λ-anchor sweep on J is inactive** because the table is warm-started
   to POI2Vec exactly: ‖E−POI2Vec‖² ≈ 0 throughout training, λ pulls
   nothing. Confirmed empirically: AZ Δ(λ=0.3 vs λ=0.1) = +0.11 pp.
5. **Delaunay POI-POI GCN (Design K) does not close the HGI gap**.
   K = J at AL (Δ=−0.02 pp) and AZ (Δ=−0.06 pp). The audit's
   structural-residual hypothesis is **falsified**: the residual ~1 pp
   to HGI on AZ/FL is *not* spatial-graph topology.

### Open (where the residual gap lives)

The HGI-gap residual at AZ/FL is **not** features (POI2Vec is already
injected) and **not** structure (Delaunay GCN is already added in K).
It must therefore be in the **training recipe**:

- HGI's POI2Vec is trained with a **hierarchical fclass L2 regulariser**
  during pretraining. The merge family imports the *output* of that
  training as a frozen prior; it never re-trains POI2Vec under a
  hierarchical objective alongside c2hgi's contrastive losses.
- HGI has a **POI↔POI contrastive boundary** (4 boundaries total). The
  c2hgi merge family has 3 (c2p, p2r, r2c). No POI↔POI contrastive
  signal — even K's Delaunay GCN is a passive message-passing layer,
  not a contrastive objective.
- HGI's **`cross_region_weight=0.7`** scaling on intra-vs-cross-region
  Delaunay edges is tuned on Alabama. K uses HGI's edges with their
  weights but the GCN may interact with c2hgi's loss landscape
  differently than HGI's.

## Reframed direction (per [ADVISOR_REFRAMING.md](ADVISOR_REFRAMING.md))

The advisor's review (2026-05-06 14:35) flagged that six merge variants
all converge within ±0.1 pp on FL reg — that's not "hard residual",
that's **saturated this whole class of intervention**. Adding more
feature-side levers (4 and 5) will not move the needle.

**Lever 6 also CLOSED 2026-05-07 01:45 — falsified.** Full results in
[LEVER_6_FINDINGS.md](LEVER_6_FINDINGS.md). α ∈ {0.1, 0.3, 1.0} swept;
no α overcomes HGI. L6 ≈ J empirically. The 4th contrastive boundary
adds no measurable lift.

**All structural research questions in this study are now closed.** The
residual ~1 pp gap to HGI on AZ reg is below this study's
architectural resolution. The realised contribution of the merge family
is a Pareto improvement over canonical c2hgi on cat + reg + POI semantic
generality at the strict Wilcoxon gate, settling at a small gap to HGI
on the reg axis only. See LEVER_6_FINDINGS.md §"Realised contribution".

**Phase 11 update (2026-05-07)**: substrate + methodology audit
complete. **S1, S4, S3-a, and S3-b V1 all ✗ falsified.** S1 (c2p
hard-neg) and S4 (DGI corrupted-feature c2p neg): both regress.
S3-a (auxiliary Checkin2Region 4th boundary): both v1+v2 collapse
to uniform attention; supervision redundant with 3-boundary path.
**S3-b V1** (Checkin2Region as *primary* region pathway, no POI2Vec):
AL Δ vs J = −2.91 pp, AZ Δ vs J = −2.54 pp; lands at canonical c2hgi
levels — pool-source replacement buys ~0.1 pp; the J→canonical gap
is the POI2Vec prior, not the pooler architecture. S3-b V2-a (with
POI2Vec residual on check-ins) not escalated per advisor — expected
outcome ≈J based on Phase 8 convergence-saturation. **S3-b V2-c**
(replacement + per-check-in POI2Vec anchor): user-requested redo
under one-redo rule. Falsified — AL Δ vs canonical = −9.95 pp
(catastrophic), AZ Δ vs canonical = −0.28 pp (mild). State-asymmetric:
anchor collapses small per-region pools (AL avg 10.7 POIs/region) to
mean(POI2Vec) but larger AZ pools absorb it. Six new findings:
(1) AZ-f3 architectural floor across all variants; (2) contrastive
attention pooling needs non-redundant supervision; (3) c2hgi/HGI
POI2Region's PMA is structurally a learnable mean-pool (PR_norm=1.0
throughout); (4) pool-source choice (POIs vs check-ins) accounts for
at most ~0.1 pp; (5) Phase 8 convergence-saturation re-framed —
"merge variants converge" because they all share POI2Vec at the
POI-pool boundary; remove that and lift disappears regardless of
pooler. Phase 10 Pareto verdict unchanged. See `HISTORY.md` Phase 11
and `PHASE_11_PLAN.md`.

---

**Tests 1, 2, 2½, 2¾, 3 all CLOSED 2026-05-06 19:35.** Full results in
[T1_T2_T2quarter_FINDINGS.md](T1_T2_T2quarter_FINDINGS.md). Headline:
every cheap diagnostic has been falsified. The single live candidate is
**reframed Lever 6** (POI↔POI contrastive boundary added to c2hgi
pretraining) — see [LEVER_6_TWO_OUTPUT.md](LEVER_6_TWO_OUTPUT.md).

Settled negatives (in addition to the prior K and λ-anchor falsifications):
- ✗ Test 1 — Next-POI: J closes 77 % of canonical→HGI gap at AL+AZ but
  does NOT overcome HGI (advisor's hypothesis falsified at both states)
- ✗ Test 2 — Reg-head ablation (no log_T): gap *widens* to 1.64 pp
  Wilcoxon-strict p=0.0312, falsifying "Markov-prior masking"
- ✗ Test 2½ — Seed reroll: Δ=+0.10 pp, falsifying seed-noise
- ✗ Test 3 — POI2Region `num_heads ∈ {2,8,16}`: adding heads hurts,
  halving doesn't lift, falsifying user's "consumer is undersized"

Test 4 (alpha sweep) deprioritised — Test 3 already shows the consumer
side is at/past optimal; tuning loss-weights inside the 3-boundary loss
is unlikely to find what the 4th boundary brings.

**Original calibrated test plan retained below for the historical record:**



| # | Test | Build needed? | Cost | Decision rule |
|---|---|---|---|---|
| 1 | **Next-POI probe** AL+AZ × {canonical, HGI, J} | No (reuse parquets) | ~3 h | If merge wins next-POI, half the research goal is already met |
| 2 | **Reg-head ablation** J FL with `next_gru` (no log_T) | No (reuse J FL emb) | ~1 h | Tells if 1 pp gap is real or masked by Markov prior |
| 2½ | **Single-seed reroll of J at FL** (seed=43) | Yes (1 build) | ~1 h | If gap shrinks to <0.5 pp, residual was seed-noise — collapses the whole open question |
| 2¾ | **200 ep vs 500 ep calibration** for J at AL | Yes (1 build) | ~30 min | If reg Acc@10 within 0.4 pp at 200 ep, all subsequent sweeps run at 200 (2.5× faster) |
| 3 | **POI2Region hyperparam sweep** on J at AL — `num_heads ∈ {2,4,8,16}`, region GCN layers ∈ {1,2} (+ log PMA attention entropy) | Yes (8 builds) | ~2-6 h depending on calibration | The user's exact hypothesis. If ≥0.5 pp lift on any axis, POI2Region is the residual |
| 4 | **Boundary-weight sweep** on J at AL — `alpha_p2r ∈ {0.2,0.4,0.5}`, `alpha_c2p ∈ {0.3,0.5}` | Yes (~6 builds) | ~3-7 h | Backup if (3) is null |

Tests 1 and 2 reuse already-built parquets; ~4 h covers them and may
make Tests 3-4 unnecessary. Tests 2½ and 2¾ are the advisor's added
sanity checks — both run before committing to the heavy sweeps.

**Add when implementing Test 3**: log per-region PMA attention entropy
`(α · log(α)).sum(0).mean()` after the `pyg_softmax` call in
`research/embeddings/hgi/model/RegionEncoder.py:134`. Direct evidence
of whether the PMA seed query is collapsing onto a small set of POIs
at `num_heads=4`.

**Levers 4 and 5 are deprioritised.** They remain documented but the
convergence pattern says they will not move the residual. Lever 6 is
**reframed**: not a "second output head" but **add a POI↔POI
contrastive boundary to the existing 3-boundary c2hgi loss**, on top
of the merge POI vectors. Only commit Lever 6 if Tests 1-4 do not
close the gap.

### Why this re-ordering

- Test 1 may make the question moot. c2hgi's per-visit modelling has
  structural reasons to beat HGI on next-POI (HGI literally cannot
  distinguish two visits to the same place). If the merge family
  already wins next-POI, the user's stated goal is half-met and the
  remaining 1 pp on reg becomes a paper footnote.
- Test 2 calibrates whether the 1 pp gap is real or diluted by the
  Markov-1-step floor that `next_getnext_hard` adds.
- Test 3 is the user's intuition: maybe POI2Region is undersized for
  the new mixture distribution that merge POI vectors carry. The
  cheapest single test that could confirm or rule out the residual
  being in the consumer rather than the input.
- Test 4 is the secondary tuning lever; the merge regime never
  retuned the boundary weights inherited from canonical c2hgi.

## What's running / pending

Pipeline currently **idle**. Last run: K AL+AZ at λ=0.5 finished
2026-05-06 14:01. All four candidate JSONs in
`docs/studies/check2hgi/results/P1/`.

## Files to read in this folder, in order

1. **STATE.md** — this document. Big picture + verdicts to date.
2. **AUDIT_HGI_GAP.md** — the diagnostic that identified what's
   missing in the merge family vs HGI, with empirical falsifications
   of Levers 1 and 3.
3. **DESIGN_*.md** — per-design audits (A, B, D, E, H, I, J, M, K)
   each with aim, mechanism, leak-free numbers, and verdict.
4. **LEVER_*.md** — proposals for the remaining live mechanisms.
5. **SPEEDUP_AUDIT.md** — orthogonal: how to make builds faster
   without changing quality. Tier 1 wins (~15-25 % MPS) ready to apply.
6. **MERGE_DESIGN_NOTES.md**, **SUMMARY.md** — historical narrative.
   Useful for archaeology, not authoritative for current numbers.
7. **INDEX.md** — table of contents.
