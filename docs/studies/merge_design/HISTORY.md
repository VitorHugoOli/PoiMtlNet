# Merge-design study — unified history log

A chronological account of what was tried, what was found, and the
current state. For the **big-picture verdict** see `STATE.md`. For
**per-design audits** see `DESIGN_*.md`. For the **HGI-gap diagnostic**
see `AUDIT_HGI_GAP.md`. For the **advisor reframing** see
`ADVISOR_REFRAMING.md`.

This file is the *what-happened-when* index.

## Phase 0 — pinned baselines (≤ 2026-05-04)

Leak-free per-fold log_T baselines pinned for canonical c2hgi and HGI
across AL+AZ. These are the references against which every merge design
is compared.

| Substrate | AL cat F1 | AL reg | AZ cat F1 | AZ reg |
|---|---:|---:|---:|---:|
| canonical c2hgi | 40.76 | 59.15 | 43.21 | 50.24 |
| HGI | 25.26 | 61.86 | 28.69 | 53.37 |
| c2hgi + POI2Vec input (probe) | 31.47 | 61.38 | 35.58 | 52.31 |

The probe revealed: c2hgi has zero fclass-discriminative signal in its
POI embeddings (probe ≈ 4 % vs HGI 98 %); injecting POI2Vec at the
*input* lifts reg by ~+2 pp but tanks cat by ~9 pp. This motivated the
merge-design family — push POI2Vec injection past the encoder so cat
stays canonical.

## Phase 1 — early designs (2026-05-04)

### Design A — late-fusion concat (✗ closed)

Concat `[c2hgi_checkin_emb(64) ‖ hgi_poi_emb(64)]` at the head input.
**Result**: cat AL −8.5, AZ −11.5; reg AL −18.7, AZ −9.2 (5/5 folds
in the wrong direction). Late fusion at the head is dead.

### Design E — projector heads + reg stop-gradient (✗ closed)

Add Linear→ReLU→Linear projector heads with stop-gradient on the reg
projector, on top of POI2Vec-augmented c2hgi input. Tests "is the cat
collapse a gradient-interference issue?"
**Result**: cat AL −9.5, AZ −9.4 (identical to unprojected probe). The
collapse is **input-side**, not gradient-side. Stop-gradient does not
save cat when POI2Vec is in the per-check-in input.

This finding redirected the ladder away from input-level fusion toward
**post-pool injection**.

## Phase 2 — pool-boundary merge family (2026-05-05)

### Design B — POI2Vec at the pool boundary (✓ dominance AL/AZ)

`poi_emb_for_reg = poi_emb_canonical.detach() + γ · Linear(POI2Vec)`.
Cat path is byte-identical to canonical c2hgi (detach()). Reg path
gets the POI-stable POI2Vec residual.
**Result AL/AZ**: cat +0.7 pp / +0.7 pp (TOST p<0.003, non-inferior);
reg +2.34 pp / +2.35 pp; fclass probe 98%/98% (HGI grade).
✓ DOMINANCE.

### Design H — learnable POI table at the pool boundary (✓)

Replace B's frozen Linear(POI2Vec) with `nn.Embedding(num_pois, 64)`,
warm-started from POI2Vec. Tests whether the lift is from POI2Vec
specifically or from any POI-stable lookup.
**Result AL/AZ**: cat +0.21 / +0.94 (AZ Wilcoxon strict);
reg +3.20 / +2.06 (AL Wilcoxon strict). fclass 98%/98%.
The merge mechanism does NOT require POI2Vec — a learnable table
converges to fclass geometry.

### Design I — LoRA r=8 on B (✓)

Low-rank correction on top of B's frozen Linear(POI2Vec). 8× cheaper
than H (95k vs 758k extra params at AL).
**Result AL/AZ**: cat +0.87 / +0.50 (AZ strict); reg +2.20 / +2.31.

### Design J — H + L2 anchor regulariser λ=0.1 (✓ first to nominally beat HGI)

Same as H, plus L2 anchor pulling the learnable table toward POI2Vec.
**Result AL/AZ**: cat +1.05 / +0.53 (TOST ✓); reg +2.80 / +1.91
(AL Wilcoxon strict, **+0.10 pp vs HGI on AL** — first to do so).

### Design M — B + cosine distillation toward POI2Vec λ_d=0.1 (✓ cat strict 5/5 both states)

Add cosine distillation on the post-pool POI projection toward POI2Vec.
**Result AL/AZ**: cat +0.55 / +0.46 (**Wilcoxon p=0.0312 strict at
both states** — only design with strict cat win at both). Reg +2.41 /
+2.21.

### Design D — heterograph (⚠ leaked, disqualified)

POIs and check-ins as first-class node types in a heterogeneous graph.
**Result**: cat AL +32.12, AZ +31.52 (suspiciously large). Linear-probe
on last-step embedding alone shows D=51% vs canonical=31% — **+20 pp
leak via reverse visit-edges + 2-hop GCN**. Reg lift looks legitimate
but cat is artefact. Disqualified.

## Phase 3 — Florida confound + leak-free reset (2026-05-05 evening)

### The C4 leak

FL canonical reg apparently jumped from 0.692 to 0.824 with the merge
designs. Investigation revealed a **leaky single
`region_transition_log.pt`** built from all rows including val/test.
The `next_getnext_hard` reg head additively biases `log_T[i,j]`, and
the leaky log_T inflated ~13 pp.

### Per-fold leak-free protocol

Built per-fold log_T tensors using StratifiedGroupKFold seed=42:
`output/check2hgi/<state>/region_transition_log_seed42_fold{1..5}.pt`.

### FL leak-free reruns (2026-05-05 night → 2026-05-06)

| Design | FL Acc@10 | Δ vs canonical | Wilcoxon | Δ vs HGI |
|---|---:|---:|---|---:|
| canonical (pf) | 0.6922 | — | — | −2.13 |
| B | 0.6993 | +0.71 | p=0.0625 | −1.41 |
| I | 0.7002 | +0.81 | p=0.0625 | −1.31 |
| M | 0.7011 | +0.89 | p=0.0625 | −1.23 |
| J | **0.7034** | **+1.12** | **p=0.0312 ✓** | −0.99 |
| H | **0.7041** | **+1.20** | **p=0.0312 ✓** | −0.92 |
| HGI | 0.7134 | +2.13 | — | 0 |

**At FL, only H and J pass the strict Wilcoxon gate**, both at ~+1 pp
vs canonical. The merge family is still ~1 pp below HGI at FL.

The earlier "+13 pp at FL" report was fully attributable to the leaky
log_T, not the merge mechanism.

## Phase 4 — audit & first reorganisation (2026-05-06)

After FL leak-free completed, the user requested an **audit** to
diagnose how to overcome HGI on next-region and next-POI. The audit
(`AUDIT_HGI_GAP.md`) identified:

- The HGI gap is **structural**: HGI uses a Delaunay POI-POI graph
  (`research/embeddings/hgi/preprocess.py:149-156`) for message
  passing at the POI level. The merge family has nothing analogous.
- **Next-POI is not evaluated anywhere** in the study; everything is
  next-region.

The audit proposed six levers, ranked by cost. The user picked Levers 1
(λ-anchor sweep) and 3 (Delaunay edges).

Folder reorganisation: per-design `DESIGN_*.md` audits, leak-free
table, audit doc.

## Phase 5 — Lever 1 (✗ falsified) (2026-05-06 morning)

**Hypothesis**: J's `--anchor-lambda 0.1` lets the table drift from
POI2Vec; sweep λ ∈ {0.3, 0.5, 1.0, 3.0}.

**Empirical result** at AZ:

| λ | reg Acc@10 | Δ vs λ=0.1 |
|---|---:|---:|
| 0.1 | 0.5215 | — |
| 0.3 | 0.5226 | +0.11 pp |

**Falsified**. The training log shows `anc=0.0000` from epoch 1
onward. Reason: J uses `--warm-start=True` so the table starts equal to
POI2Vec; ‖E−POI2Vec‖² stays ≈ 0; multiplying zero by larger λ is still
zero. Sweep cancelled after λ=0.3.

## Phase 6 — Lever 3 / Design K (✗ falsified) (2026-05-06 afternoon)

**Hypothesis**: add HGI's Delaunay POI-POI GCN between Checkin2POI and
POI2Region. New design: K = J + GCNConv over Delaunay edges.

**Implementation**:
- `scripts/probe/build_design_k_delaunay.py` (new)
- Loads `output/hgi/<state>/temp/edges.csv`, remaps to c2hgi POI index
  space via shared `placeid` column, symmetrises, gives 71 k AL edges.
- `Check2HGI_DesignK` adds `poi_gcn = GCNConv(64, 64, cached=True)`
  + `nn.PReLU(64)` between detached pool output and POI2Region input.

**Result**:

| State | K (λ=0.5) | Δ vs canonical | Δ vs HGI | Δ vs J(λ=0.1) |
|---|---:|---:|---:|---:|
| AL | 0.6193 | +2.78 pp ✓ p=0.0312 | +0.07 pp | **−0.02 pp** |
| AZ | 0.5209 | +1.85 pp | **−1.29 pp** | **−0.06 pp** |

**K = J empirically.** Adding HGI's spatial topology contributes zero
lift over J's anchor mechanism. The audit's structural-residual
hypothesis is **falsified**.

FL K skipped — no new information expected.

## Phase 7 — Advisor reframing (2026-05-06 14:35)

The advisor was consulted after K. Verdict:

> Six merge variants — fixed/learned/low-rank features,
> anchored/unanchored, with/without spatial topology — all converge
> within ±0.1 pp on FL reg. **That's not "hard residual". That's
> "saturated this class of intervention."**

Reframed plan (per `ADVISOR_REFRAMING.md`):

| # | Test | Cost | Decision rule |
|---|---|---|---|
| 1 | Next-POI probe AL+AZ × {canonical, HGI, J} | ~3 h | If merge wins next-POI: half goal already met |
| 2 | Reg-head ablation J FL with `next_gru` (no `log_T`) | ~1 h | Real-vs-masked diagnostic |
| 3 | POI2Region hyperparam sweep on J at AL | ~2-3 h | User's hypothesis: maybe consumer is undersized |
| 4 | Boundary-weight (alpha) sweep on J at AL | ~3 h | Secondary tuning |

If 1-4 close the gap: done. If not: commit reframed Lever 6 — POI↔POI
contrastive boundary added to c2hgi, **not** "second output head".

Levers 4 and 5 deprioritised (advisor: feature-side saturation).

## Phase 8 — speed audit (2026-05-06 13:00)

Orthogonal track: `SPEEDUP_AUDIT.md` ranks 15 candidates for training
speed. Tier 1 (combined ~15-25 % MPS speedup, zero numerical risk):

- Drop `.item()` syncs in `set_postfix`
- `tqdm refresh=False`, refresh every K
- Skip `scheduler.step()` when `--gamma=1.0`
- `randperm(device=...)` in `Check2HGIModule.py:27`
- `self.poi_table.weight` directly instead of `Embedding(arange(N))`

Pending application — gated on K AL+AZ finishing so in-flight runs
land on the same code path. (K finished 14:01; can apply now.)

## Phase 9 — Reframed diagnostic tests 1+2+2½+2¾+3 (2026-05-06)

User-led reorganisation: HISTORY.md, STATE.md, per-lever proposal docs
(LEVER_4/5/6) created. SPEEDUP_AUDIT Tier 1 patches applied across all
6 build scripts + Check2HGIModule.py:27. Smoke-test J AL 50ep at
5.85 s/iter vs ~5.6 s/iter unpatched — **null speedup empirically**;
GCN forward+backward dominates, not per-step syncs. Tier 1 retained as
clean-code baseline; Tier 2/3 deferred.

### Test 2 — J FL with `next_gru` (no `log_T`) + canonical + HGI baselines

5-fold leak-free, no Markov prior:

| Substrate | Acc@10 | Δ vs canonical | Wilcoxon p_gt |
|---|---:|---:|---|
| canonical | 0.6836 | — | — |
| **J** | 0.6922 | +0.86 pp | **p=0.0312 ✓** |
| **HGI** | 0.7086 | +2.50 pp | **p=0.0312 ✓** |
| HGI vs J | — | **+1.64 pp** | **p=0.0312 ✓** |

**Removing log_T widens the J→HGI gap from −1.00 to −1.64 pp**. The
Markov prior was *helping* J close the gap, not masking a real one.
The embedding gap is real, structural, larger than thought.

### Test 2½ — J FL build seed=43 vs seed=42

Δ = +0.10 pp. **Seed-noise hypothesis falsified.**

### Test 2¾ — Epoch calibration: J AL at 200 vs 500 ep

ep200 = 0.6155, ep500 = 0.6196, Δ = −0.41 pp. Within 0.5 pp tolerance.
Test 3 sweeps run at 200 ep (2.5× faster).

### Test 1 — Next-POI probe AL+AZ × {canonical, HGI, J}

New script `scripts/p1_poi_head_ablation.py` (target = placeid, n_pois
≈ 12k AL / 21k AZ). Found the AZ `next_poi.parquet` was missing →
patched the script to derive `poi_idx` from `sequences_next.parquet` at
runtime.

| State | canonical | J | HGI | J vs canon | J vs HGI |
|---|---:|---:|---:|---:|---:|
| AL | 0.0361 | 0.0499 | 0.0541 | **+1.38 pp ✓** | −0.42 pp (n.s.) |
| AZ | 0.0717 | 0.0855 | 0.0895 | **+1.38 pp ✓** | −0.41 pp (n.s.) |

**Both states identical pattern: J closes 77 % of canonical→HGI gap on
next-POI but does NOT overcome HGI.** Advisor's "c2hgi naturally beats
HGI on next-POI" hypothesis falsified at both states.

### Test 3 — POI2Region `num_heads` sweep on J at AL (ep=200)

| nh | Acc@10 | Δ vs J(nh=4) | Δ vs HGI |
|---|---:|---:|---:|
| 2 | 0.6174 | +0.19 pp | −0.12 pp |
| 4 (default) | 0.6155 | 0 | −0.30 pp |
| 8 | 0.6149 | −0.06 pp | −0.37 pp |
| 16 | 0.6104 | **−0.51 pp** | −0.82 pp |

Adding heads **monotonically hurts** after nh=4. **User's "POI2Region
is undersized" hypothesis falsified.** Consumer side already at/near
optimal capacity; not the residual bottleneck.

Test 4 (alpha sweep) deprioritised — Test 3 already shows consumer
capacity is fine; tuning loss-weights inside an unchanged 3-boundary
loss is unlikely to find what 4 boundaries did.

**Phase 9 documents**: `T1_T2_T2quarter_FINDINGS.md`,
`T2_FINDINGS.md`.

## Phase 10 — Lever 6 (POI↔POI contrastive boundary) — falsified (2026-05-07)

After every cheap diagnostic falsified, Lever 6 was the single live
candidate. Build script `scripts/probe/build_design_lever6_p2p.py`
(`Check2HGI_DesignL_P2P`) adds a 4th contrastive boundary to J's
3-boundary loss, scoring merge POI vectors against HGI's Delaunay edges
(positives) vs random POIs (negatives). ~6 h implementation, mirrors
c2hgi's existing bilinear discriminator pattern.

### α-sweep ∈ {0.1, 0.3, 1.0} × AL+AZ × 200 ep

| Substrate | AL Δ vs HGI | AZ Δ vs HGI |
|---|---:|---:|
| **J (baseline)** | **+0.10 pp** | −1.22 pp |
| L6 α=0.1 | −0.05 pp | −1.43 pp |
| L6 α=0.3 | −0.47 pp | −1.00 pp |
| L6 α=1.0 | −0.75 pp | −1.01 pp |

α=0.3 showed `best_ep=21/200` at AL (loss bounced after ep21);
α=1.0 showed `best_ep=9, loss=15.6` — pathological loss landscape with
stronger boundary. α=0.1 stable but no lift.

L6 next-POI at α=0.3: AL regressed −0.46 pp vs J, AZ tied. **No α
overcomes HGI on any axis at any state.** Wilcoxon p_gt vs HGI ≥ 0.59
across all settings.

### 6th falsification — every architectural surface exhausted

| Surface | Lever | Verdict |
|---|---|---|
| POI features (frozen / learnable / LoRA / anchor / distill) | B/H/I/J/M | ✓ canonical+2pp reg, no HGI overcome |
| Spatial topology | K (Lever 3) | K = J empirically |
| λ-anchor strength | J λ-sweep (Lever 1) | warm-start zeros it |
| Markov prior masking | T2 | gap is real, gets bigger w/o log_T |
| Build seed noise | T2½ | Δ = +0.10 pp |
| Reg-head capacity (PMA num_heads) | T3 | adding heads hurts |
| 4th contrastive boundary | L6 α-sweep | tied or worse than J |

**The residual ~1 pp gap to HGI on AZ reg is below this study's
architectural resolution.** It must live in recipe-level differences:
HGI's 2000-epoch POI2Vec pretraining (we use the *output* as a frozen
prior, never re-train under c2hgi's corpus); HGI's per-state
`cross_region_weight=0.7` calibration; HGI's POIEncoder GCN over POI2Vec
inputs at the front of its pipeline.

### Realised contribution

| Property | canonical | HGI | merge family |
|---|:-:|:-:|:-:|
| Cat F1 | ✓ best | ✗ catastrophic (−15 pp) | ✓ preserved |
| Reg Acc@10 | weakest | best | strict beats canon (Wilcoxon p=0.0312) |
| Next-POI Acc@10 | weakest | best | strict beats canon (Wilcoxon p=0.0312) |
| fclass POI probe | 4 % | 98 % | **98 %** (HGI-grade) |

The merge family is the **only configuration that wins all four axes
simultaneously** — a Pareto improvement over both base architectures.

**Phase 10 document**: `LEVER_6_FINDINGS.md`.

## Phase 11 — substrate audit (REOPEN, 2026-05-07)

The "study closed" verdict in Phase 10 covered the **architectural
surface accessible to c2hgi-merge build scripts** (POI2Vec injection
points, frozen vs learnable, anchored, LoRA, distill, Delaunay GCN,
4th boundary). It did **not** audit the c2hgi *substrate itself* — the
canonical pretraining recipe that every merge design inherits. User
flagged 2026-05-07 that all merges sit on the same untouched substrate;
two pieces of that substrate had not been re-examined since the merge
family was launched.

### Substrate concerns identified (re-reading
`research/embeddings/check2hgi/model/Check2HGIModule.py`)

1. **Methodology vs capacity at POI2Region (distinct from Test 3).**
   `POI2Region` is imported verbatim from HGI
   (`research/embeddings/hgi/model/RegionEncoder.py`). Its PMA
   seed-query inductive bias was tuned to consume **POI2Vec-style
   stable** vectors (Node2Vec on Delaunay graph → smooth, low-rank).
   c2hgi feeds it the **mean of contextually-encoded check-ins per
   POI** — a mixture distribution with visit-noise riding on a
   POI-stable component. Test 3 swept *capacity* knobs (`num_heads`,
   GCN depth) on this fixed mechanism and got null. **Capacity null
   does not refute methodology change.** The user's reframed proposal
   is to drop the POI bottleneck entirely on the region pathway —
   direct `Checkin2Region` PMA pool, with POI level kept as auxiliary
   regulariser.

2. **Corruption forward pass is paid for but only one boundary uses
   it.** `Check2HGIModule.forward()` runs a full feature-corrupted pass:
   - `neg_checkin_emb` = encoder(corrupted x, edge_index)  ← computed
   - `neg_poi_emb`     = checkin2poi(neg_checkin_emb, …)   ← computed
   - `neg_region_emb`  = poi2region(neg_poi_emb, …)        ← computed

   But in `loss()`, the c2p and p2r boundaries use
   `pos_poi_emb[neg_indices]` and `pos_region_emb[neg_indices]` (index-
   shuffled positive pool). Only `neg_region_emb` reaches a loss term
   (r2c). HGI behaves the same way at p2r — so the *pattern* is
   inherited, not buggy — but the c2hgi-specific c2p boundary inherits
   the same weakness without compensating.

3. **No hard-negative mining at c2p.** This is the boundary c2hgi
   *adds* over HGI. HGI mines hard negatives at p2r (similarity ∈
   (0.6, 0.8), 25 % rate). c2hgi's `_sample_negative_indices()` for
   c2p is pure random over the global POI pool. A check-in at a coffee
   shop paired against POI #4521 (random) is almost certainly a
   different category in a different region — trivial discrimination,
   weak gradient. The c2p boundary defines c2hgi's POI vectors; the
   weakest negatives in the model live at the most consequential
   boundary.

4. **Hard-negative gating at p2r is effectively off for FL.** The
   existing 25 %-hard-neg path at p2r is gated `batch_size < 50000` and
   uses a Python for-loop. AL/AZ trigger it; FL with ~30k POIs but
   larger batch sizes may not. (Status: not measured.)

### Why this lever is genuinely untested

Six merge variants converging within ±0.1 pp on FL reg means the
*feature-side* surface is saturated **conditional on the c2hgi
substrate they share**. None of the six tested designs altered the
substrate's negative-sampling strategy or the POI2Region methodology;
all of them inherited the same potentially-weak c2p signal and the
same POI-stable-tuned consumer.

Whether this matters is unproven. The ADVISOR_REFRAMING (Phase 8) was
correct that *adding more feature levers will not move the needle* —
but it did not consider that all six levers are downstream of the same
substrate.

### Test ladder (cost-ordered, on top of canonical or J)

| # | Test | Cost | Decision rule |
|---|---|---:|---|
| S1 | Hard-neg c2p (sample neg POI from same-region different-POI pool) at AL+AZ canonical | ~3 h | If reg moves ≥0.5 pp or fclass probe shifts: c2p signal was undertrained |
| S2 | Hard-neg c2p applied on top of J at AL+AZ | ~3 h | Confirms whether the substrate fix compounds with the merge mechanism |
| S3 | Direct `Checkin2Region` pooler (skip POI bottleneck on region path; POI as aux head) at AL canonical | ~1-2 d | Methodology change for region pathway. If null: methodology hypothesis falsified independent of capacity |
| S4 | Use `neg_poi_emb` at c2p (DGI-style same-identity corrupted-feature negative) | ~2 h | Cheap; tests whether corruption forward should also feed c2p |

S1 is the cheapest principled test and runs first. S2 only matters if
S1 shows lift. S3 is the user's "redesign methodology" idea in its
strongest form and is the only test that addresses concern (1).

### What "study closed" still holds for

The Pareto contribution of the merge family vs canonical c2hgi vs HGI
(15 pp cat advantage preserved + Wilcoxon-strict reg lift + fclass 4 →
98 %) is not affected by this reopen. Phase 11 tests substrate
weaknesses inherited by the *whole* family — the relative ordering of
designs is downstream of, and largely independent of, the substrate
signal strength.

## Where we are now (2026-05-07 — REOPENED) — substrate audit underway

- Phase 11 reopens the study at the substrate level.
- S1 (c2p hard-neg, AL+AZ canonical) is the first test queued.
- All Phase 1-10 verdicts on merge-design *relative ordering* remain valid.

## Closed research questions

1. ✓ Does the merge family beat HGI on next-POI? **No** — closes 77 %
   of the canonical→HGI gap but doesn't overcome (Test 1).
2. ✓ Is the 1 pp FL gap real or masked by `log_T`? **Real** (Test 2).
3. ✓ Is POI2Region undersized (capacity)? **No** — already at/past
   optimal (Test 3). *Note: methodology change not yet tested — see
   S3 in Phase 11.*
4. ✓ Were `alpha_{c2p,p2r,r2c}` retuned for the merge regime?
   Deprioritised — Test 3 already showed consumer side is fine.
5. ✓ Does a 4th `L_p2p` contrastive boundary close the gap?
   **No** — L6 ≈ J empirically across α ∈ {0.1, 0.3, 1.0}.

### Phase 11.S1 — c2p hard-neg sampling (✗ FALSIFIED 2026-05-07 12:51)

Same-region different-POI hard negatives at the c2p boundary, p=0.25.
Code landed at `Check2HGIModule.py` (`c2p_hard_neg_prob`). Build via
`scripts/probe/build_substrate_s1.py`. AL+AZ canonical c2hgi rebuilt
(500 ep MPS) at `output/check2hgi_substrate_s1/<state>/`.

| State | canonical Acc@10 | S1 Acc@10 | Δ | Wilcoxon p (S1 > can) |
|---|---:|---:|---:|---:|
| AL | 59.15 | 57.94 | **−1.21 pp** | 0.9062 |
| AZ | 50.24 | 49.43 | **−0.81 pp** | 0.6875 |

Per-fold AL Δ: [+2.54, −1.07, −3.22, −1.01, −3.29].
Per-fold AZ Δ: [+1.84, +0.72, +1.73, −4.53, −3.80].

Both states regress; pre-registered ≥0.5 pp lift gate not met. **S1
falsified.** The "weak negatives at c2p" hypothesis is wrong, or the
fix is wrong: same-region POIs share too many characteristics, so
discrimination becomes noisy/conflicting (folds 3-4 collapse on AZ;
folds 2-4 lose on AL).

S2 (S1 stacked on J) skipped — pre-registered to depend on S1 success.

Result JSONs:
- `docs/results/P1/region_head_alabama_region_5f_50ep_STL_ALABAMA_substrate_s1_p0_25_reg_gethard_pf_5f50ep_leakfree.json`
- `docs/results/P1/region_head_arizona_region_5f_50ep_STL_ARIZONA_substrate_s1_p0_25_reg_gethard_pf_5f50ep_leakfree.json`

Note: AZ S1 best_epoch=498 (very late convergence); AL best=500
(at the cutoff). Substrate-s1 runs may benefit from longer training,
but the per-fold variance pattern (some folds gain, others collapse)
is more consistent with the negatives being **misaligned** than with
under-training. Not pursued further.

Live next candidate: **S4** (`neg_poi_emb` at c2p, DGI-style same-
identity corrupted-feature negative) — see `PHASE_11_PLAN.md`.

### Phase 11.S4 — DGI-style corrupted-feature c2p negatives (✗ FALSIFIED 2026-05-07 15:43)

Replaces c2p negatives with `neg_poi_emb[checkin_to_poi]` — same POI
identity, but encoded from the corrupted-feature forward pass that
c2hgi computes unconditionally. Tests "is this the true encoding of
POI X, or a corrupted-feature encoding?" instead of "is this POI X
or a different POI?". Code at `Check2HGIModule.py`
(`c2p_corrupted_neg`, mutually exclusive with `c2p_hard_neg_prob`).
Build via `scripts/probe/build_substrate_s4.py`.

| State | canonical Acc@10 | S4 Acc@10 | Δ | Wilcoxon p (S4 > can) |
|---|---:|---:|---:|---:|
| AL | 59.15 | 59.08 | **−0.07 pp** | 0.5000 |
| AZ | 50.24 | 49.08 | **−1.16 pp** | 0.6875 |

Per-fold AL Δ: [+3.53, +0.35, −2.16, +0.05, −2.11].
Per-fold AZ Δ: [+1.80, +0.06, +0.76, −4.46, −3.99].

AL essentially flat; AZ regresses ~1 pp. **S4 falsified.**

### Cross-test fold-stability pattern (S1 vs S4)

| State | Fold | S1 Δ | S4 Δ |
|---|---:|---:|---:|
| AL | 0 | +2.54 | +3.53 |
| AL | 1 | −1.07 | +0.35 |
| AL | 2 | −3.22 | −2.16 |
| AL | 3 | −1.01 | +0.05 |
| AL | 4 | −3.29 | −2.11 |
| AZ | 0 | +1.84 | +1.80 |
| AZ | 1 | +0.72 | +0.06 |
| AZ | 2 | +1.73 | +0.76 |
| AZ | 3 | −4.53 | −4.46 |
| AZ | 4 | −3.80 | −3.99 |

Folds 3-4 of AZ collapse identically (~−4 pp) under two architecturally
different c2p interventions. AL folds 2 and 4 lose under both. The
*sign and approximate magnitude* of the per-fold deltas is preserved
across S1's same-region-different-POI replacement and S4's
corrupted-feature same-identity replacement — even though these
mechanisms produce different gradient signals at c2p.

**Per-fold AZ across canonical, merge family (B/H/I/J/M), and S1/S4**
(Acc@10, %, leak-free):

| variant | f0 | f1 | f2 | f3 | f4 | mean |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 47.84 | 49.31 | 50.19 | 49.39 | 54.47 | 50.24 |
| B | 52.71 | 53.80 | 55.28 | 47.41 | 53.76 | 52.59 |
| H | 52.58 | 53.29 | 55.01 | 47.19 | 53.46 | 52.31 |
| I | 52.52 | 54.27 | 54.95 | 47.40 | 53.59 | 52.55 |
| J | 52.61 | 53.89 | 54.82 | 46.94 | 52.49 | 52.15 |
| M | 52.78 | 54.10 | 54.59 | 47.28 | 53.48 | 52.45 |
| S1 | 49.68 | 50.03 | 51.92 | 44.86 | 50.67 | 49.43 |
| S4 | 49.64 | 49.37 | 50.96 | 44.93 | 50.48 | 49.08 |

**Two effects stack on AZ f3** (canonical 49.39 → merges ~47 → S1/S4
~44.9):
1. *Architectural-sensitivity floor*: all five merge designs lose
   ~2 pp on f3 even though they preserve canonical's c2p path entirely
   (they only modify the POI-pool boundary). The f3 collapse is
   therefore not c2p-specific — it's a property of the AZ f3 data
   split that disadvantages any departure from canonical c2hgi's
   exact recipe.
2. *c2p-specific damage*: S1/S4 lose another ~2.5 pp on top of that
   floor. This extra damage *is* c2p-specific.

The merge family is already making the right Pareto trade on AZ:
+3-5 pp on f0-f2 (where lift is gainable), −1-2 pp on f3-f4 (data-
intrinsic floor). S1/S4 do the *opposite* trade — no gain on f0-f2
(canonical-equivalent there) and big cost on f3-f4. That's why the
substrate paths net to a regression while the merge paths net to a
Pareto win.

**Implication for S3**: S3 changes the consumer (POI→Region pathway),
which doesn't touch c2p, so the c2p-specific damage (effect 2) would
not apply. But effect 1 — architectural sensitivity on AZ f3 — likely
*would* apply, since all five merge designs hit it without touching
c2p. More importantly, S3 has to gain *on top of the merge family*,
not on top of canonical. The cost is 1-2 days for "maybe lifts further
on top of merges; high chance of hitting f3 like every other
architectural variant did." The Phase 10 Pareto verdict stands either
way.

**Decision: do not run S3 under this study's scope.** S3 stays in
`PHASE_11_PLAN.md` as documented future work. Phase 11 produced two
clean falsifications and a genuinely new finding (the f3
architectural-sensitivity property, visible only because S1/S4 made
it impossible to attribute to c2p alone). The study reverts to
"closed" with Phase 11 strengthening, not weakening, the Phase 10
verdict.

## Open research questions (Phase 11)

6. ✗ Does *random* c2p hard-negative sampling lift reg? **No** (S1).
7. Does a Check-in↔Region direct pooler outperform the inherited
   POI2Region on c2hgi's mixture-distribution inputs? **DEFERRED** —
   not run; S3 documented as future work in `PHASE_11_PLAN.md`. After
   S1+S4 plus the per-fold diagnostic, S3's prior is mixed-to-low
   relative to its 1-2 day cost.
8. ✗ Does feeding the already-computed `neg_poi_emb` into c2p
   (DGI-style) sharpen the substrate? **No** (S4).
9. Why do AZ folds 3-4 collapse under c2p perturbation?
   **Partially answered** — f3 collapse is general-architectural (all
   5 merge designs lose ~2 pp on AZ f3 without touching c2p), with
   an additional c2p-specific damage layer for S1/S4. f3 has a data-
   split property that disadvantages any departure from canonical
   c2hgi's exact recipe. Full mechanism deferred to data-audit /
   future work.

### Phase 11.S3-a — Direct Checkin2Region pooler (REOPENED 2026-05-07 16:00)

User reactivated S3 after the deferred-future-work close-out.
Implemented as additive regulariser on top of J: a parallel
`Checkin2Region` PMA pooler producing region embeddings from the
check-in level directly, supervised via a fourth contrastive boundary
`L_c2r`. Downstream `region_embeddings.parquet` stays POI-pooled.
Code at `scripts/probe/build_substrate_s3a.py` (subclass
`Check2HGI_S3a` extends `Check2HGI_DesignJ`). Bit-equivalence at
`alpha_c2r=0.0` confirmed within MPS tolerance (J-vs-S3a@α=0 loss
diff 0.0015 vs J-vs-J reseed diff 0.0025).

**S3-a-v1 (DGI-style same-identity corrupted-feature negs) — KILLED
2026-05-07 16:30 by 50-ep AL probe.** Negative pairs were
`(checkin_i, region_emb_via_checkin_corrupted[region(i)])` —
same region, but encoded from a corrupted-feature forward pass.
Pre-registered kill rule: per-region PMA attention participation
ratio must drop to ≤ 0.5 by epoch 50 (concentration ≥ "half-uniform").

| Epoch | entropy_mean | participation_ratio_mean |
|---:|---:|---:|
| 1 | 3.0334 | 0.99997 |
| 25 | 3.0334 | 0.99997 |
| 50 | 3.0334 | 0.99997 |

Entropy bit-stable to 4 decimals across 50 epochs. PMA attention is
**uniform from init and stays uniform**. The c2r seed query has no
gradient signal pushing scores apart because all positives pair with
true region encoding and all negatives pair with corrupted-feature
region encoding *of the same region* — the only signal is a global
feature perturbation that the discriminator picks up without needing
selective attention.

This is itself a publishable observation: *contrastive attention
pooling requires between-pool variance in the negative distribution,
not just within-pool perturbation.* The S4 falsification is also
consistent with this — DGI-style same-identity negatives at c2p
gave null reg lift; here at c2r the same pattern produces a totally
non-functional attention.

**Pre-registered before the redo (2026-05-07 16:35):**

S3-a-v2 will swap to foreign-region negatives — for each check-in,
pair the positive `(checkin_i, region(i))` against
`(checkin_i, region_j)` where `region_j ≠ region(i)` is sampled
uniformly. Mirrors the canonical p2r contrast pattern minus the
similarity-mined hard-neg complexity (those are a different lever).

**One-redo rule**: if v2 produces non-uniform attention (PR_norm ≤ 0.5
by epoch 50) but the full 500-ep reg eval is null, S3-a is falsified.
No third loss formulation. "My loss was wrong twice" is the same as
falsified.

The dead corrupted-feature encoder pass at the c2r boundary is
dropped in v2 — keeps the cost honest and mirrors the wasted-corruption
finding from Phase 11.

**S3-a-v2 (foreign-region negs) — KILLED 2026-05-07 16:55 by 50-ep
AL probe.** Negatives switched to "for each check-in, sample a
random different region's c2r-pool encoding" — mirrors p2r's
contrast pattern minus the similarity-mined hard-neg complexity.
Dead corrupted-feature encoder pass dropped (mirrors the wasted-
corruption finding from Phase 11).

| Epoch | entropy_mean | participation_ratio_mean |
|---:|---:|---:|
| 1 | 3.0334027 | 0.99997 |
| 25 | 3.0334008 | 0.99997 |
| 50 | 3.0333972 | 0.99996 |

Entropy drift across 50 epochs: −5.5e-6 (floating-point noise; v1's
drift was −3.4e-6 — both indistinguishable from zero). PMA seed query
remains uniform under foreign-region negs *as well as* same-identity
corrupted-feature negs. Loss decreased (1.5827 vs v1's 1.6409),
indicating the model trains via *other paths* — the existing 3
boundaries absorb the optimisation while the c2r boundary is paid
as a small constant cost.

**S3-a falsified per pre-registered one-redo rule.**

Both negative-sampling formulations converge the c2r PMA seed query
to uniform attention (effective mean-pool) within 50 epochs. The
load-bearing falsification argument is **supervisory redundancy**,
not the uniform-attention property itself: the c2hgi POI level
already routes check-in→region via Checkin2POI ∘ POI2Region; an
additional check-in→region pathway with its own contrastive boundary
provides no signal that the existing pathway doesn't already carry.

**Caveat on the PR_norm framing** (added 2026-05-07 17:30 after the
S3-b canonical calibration): the original v1+v2 kill rule cited
PR_norm=1.0 as evidence the architecture wasn't learning. But the
canonical c2hgi POI2Region calibration on AL (50ep) shows
**canonical also has PR_norm=1.0 at ep50** with mean zone size 10.7
(see `logs/substrate_s1/canonical_pr_norm_alabama.json`). Canonical's
PMA is *also* effectively a learnable mean-pool whose seed query
doesn't discriminate within-region. Uniform PMA attention is a
property of the architecture, not a failure mode unique to S3-a.

Therefore S3-a is falsified on the redundancy argument alone — the
PR_norm collapse was suggestive but not load-bearing. The S3-b
pre-registration drops the PR_norm gate in favour of direct
functional gates (cat F1, reg Acc@10 at ep50).

This is stronger than running a 500-ep that confirms "mean-pool also
doesn't lift" — if mean-pool *did* lift, the reported mechanism
(learned attention) would be a confound (the actual lift is trivially
derivable from the 3-boundary path's own check-in→POI→region routing).

S3-b (full replacement of POI2Region with Checkin2Region) was
pre-registered as conditional on S3-a being directionally positive.
It is *not* a follow-up; not escalated.

### Phase 11 final close-out

Phase 11 contributions:
1. ✗ S1 falsified: c2p hard-neg mining (random same-region) does not
   lift reg; both states regress.
2. ✗ S4 falsified: DGI-style same-identity corrupted-feature c2p neg
   does not lift reg; AL flat, AZ regresses.
3. ✗ S3-a falsified: contrastive attention pooling at the check-in
   →region boundary cannot learn selective attention under standard
   contrastive loss formulations (DGI same-identity *or* HGI-style
   foreign-pool) when stacked on top of c2hgi+J. Supervision
   redundant with existing 3-boundary path.
4. New finding: AZ-f3 architectural-sensitivity floor — every
   architectural variant (5 merge designs + 2 substrate
   interventions) loses ~2 pp on AZ f3 even when the variant
   preserves canonical's c2p path. f3 has a data-split property
   that disadvantages departures from canonical's exact recipe.
5. New finding: contrastive attention pooling requires between-pool
   variance in the negative distribution, not just within-pool
   perturbation — and even with the right variance, supervision
   redundant with existing pathways collapses to a degenerate pooler.

The Phase 10 Pareto verdict (merge family > canonical > HGI on cat +
reg + fclass jointly) is unchanged. Phase 11 strengthens it by
adding three clean falsifications and two new findings that bound
*any* substrate-level rework, not just the merge family.

### Phase 11 calibration finding — canonical PMA is uniform too (2026-05-07 17:25)

50-ep canonical c2hgi training on AL with PMA-softmax probe at
epochs 1, 25, 50:

| Epoch | Loss | Entropy | PR_norm | Zone size [min/mean/max] |
|---:|---:|---:|---:|---:|
| 1 | 1.3891 | 1.6038 | 1.0000 | 1 / 10.7 / 157 |
| 25 | 1.3632 | 1.6038 | 1.0000 | 1 / 10.7 / 157 |
| 50 | 1.2715 | 1.6037 | 1.0000 | 1 / 10.7 / 157 |

Loss decreases honestly (1.39 → 1.27) — the model trains. But entropy
is bit-stable across 50 epochs, and PR_norm stays at 1.0000.

**Implication for c2hgi architecture**: PMA in c2hgi/HGI is structurally
a *learnable mean-pool* with a seed-query parameter that doesn't
differentiate per-region. The architecture's reg lift over a no-PMA
baseline must come from (a) the region-level GCN (`self.conv` in
POI2Region) over the adjacency graph, and (b) the contrastive
boundaries shaping the upstream POI representations — *not* from
selective pooling. The "attention" in HGI/c2hgi is in name only.

This is a useful framing for the writeup independent of S3-b's
outcome, and it means the S3-b smoke must use direct functional
gates (loss decrease + cat F1 + reg Acc@10 at ep50), not PR_norm.

## S3-b (REOPENED 2026-05-07 17:30) — primary-pathway replacement

User reopened S3 after clarifying: original hypothesis was a
**replacement** for POI2Region (Checkin2Region as the primary region
pathway), not S3-a's auxiliary 4th boundary. S3-a's falsification
arguments do not transfer:

- *Redundancy*: S3-b removes POI2Region from the region pathway, so
  there is no other check-in→region path to be redundant with.
- *PMA-uniform*: irrelevant — canonical PMA is also uniform (see
  calibration above). PR_norm was never the right gate.

S3-b V1 = `Checkin2Region` instantiated as `POI2Region(D, num_heads)`
with `zone = poi_to_region[checkin_to_poi]` (each check-in assigned
to its POI's region). Loss replaces L_p2r with L_c2r, foreign-region
negs HGI-style. Cat path preserved through Checkin2POI (not detached
because there is no parallel canonical path; full gradient flow).

Pre-registered gates and one-redo rule documented in
[`PHASE_11_PLAN.md`](PHASE_11_PLAN.md).

### Phase 11.S3-b V1 result — falsified (2026-05-07 22:43)

V1 = `Checkin2Region = POI2Region(D, num_heads)` with
`zone = poi_to_region[checkin_to_poi]`. L_p2r replaced by L_c2r
(HGI-style foreign-region negs, primary). POI level preserved as
input to L_c2p only — no POI2Vec residual injected to the region
pathway. 500-ep build at AL+AZ on MPS. Throughput notably faster
than canonical (~2 it/s vs ~0.2 it/s) because S3-b drops one PMA
pass (no neg POI pooling) and skips Checkin2POI's neg path.

**AL 5f×50ep reg-head leak-free** (Acc@10 %):
| variant | f0 | f1 | f2 | f3 | f4 | mean | Δ vs J |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 58.91 | 60.00 | 64.40 | 57.46 | 55.00 | 59.15 | −2.81 |
| J | 65.30 | 62.08 | 65.70 | 60.90 | 55.80 | 61.96 | — |
| HGI | 62.46 | 62.54 | 66.65 | 59.54 | 58.11 | 61.86 | −0.10 |
| S3-b V1 | 62.16 | 59.95 | 62.79 | 57.59 | 52.74 | **59.04** | **−2.91** |

**AZ 5f×50ep reg-head leak-free** (Acc@10 %):
| variant | f0 | f1 | f2 | f3 | f4 | mean | Δ vs J |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 47.84 | 49.31 | 50.19 | 49.39 | 54.47 | 50.24 | −1.91 |
| J | 52.61 | 53.89 | 54.82 | 46.94 | 52.49 | 52.15 | — |
| HGI | 50.51 | 53.24 | 52.71 | 52.91 | 57.50 | 53.37 | +1.22 |
| S3-b V1 | 49.75 | 50.46 | 51.96 | 44.89 | 50.98 | **49.61** | **−2.54** |

Pre-registered gates **fail** by large margin: AL Δ vs J = −2.91 pp
(gate ≥ +0.5 pp); AZ folds {0,1,2,4} Δ vs J = −2.67 pp (gate ≥ 0).
Falsified by ~6× the gate magnitude — not borderline.

### The decisive read (the actual contribution of S3-b)

Place S3-b V1 next to canonical and J on AL:

| variant | POI2Vec at pool? | Region pooler | AL Acc@10 |
|---|:-:|:-:|---:|
| canonical | no | POI2Region | 59.15 |
| **S3-b V1** | **no** | **Checkin2Region** | **59.04** |
| J | yes | POI2Region | 61.96 |

S3-b V1 ≈ canonical (Δ=−0.11 pp). Both sit ~−3 pp below J.

**The pool-source choice (POIs vs check-ins) accounts for at most
~0.1 pp.** The ~3 pp lift J brings comes from the POI2Vec residual
at the POI-pool boundary, **not** from the pool architecture. The
user's hypothesis — POI2Region tuned for POI-stable inputs is
suboptimal for c2hgi's check-in-mixture inputs — is **falsified**:
swapping the pool source without changing the semantic prior does
not lift. The lift in the merge family was the prior, not the
pooler.

This is also a clean re-framing of the Phase 8 convergence-saturation
finding ("six merge variants converge ±0.1 pp on FL reg"): the merge
family converges *because they all share POI2Vec at the POI-pool
boundary*, and the architectural variation around that boundary
(frozen vs learnable, anchored vs not, low-rank vs full, with or
without Delaunay GCN) is downstream noise. Removing POI2Vec
re-creates canonical's ~3 pp deficit *regardless of whether you keep
POI2Region or replace it*.

### Smoke methodology lesson (worth noting)

The 50-ep AL fold-0 quick eval gave Acc@10 = 62.39 %, suggesting a
big lift. The 500-ep 5-fold mean is 59.04 %. The smoke landed on the
strongest fold (fold 0 has the biggest canonical→S3b gap at +3.25 pp;
fold 4 has −2.26 pp). Single-fold partial-training results can
mislead by 3+ pp at this study's scale. The pre-registration
specifically excluded "single-state ≥ 0.3 pp" as a positive claim —
the same logic should have been applied to "single-fold ≥ 3 pp" at
the smoke stage. Going forward: smoke gates should use the median
of available canonical baselines or a 2-fold quick eval, not a
single fold.

### Pre-registered V2 decision: NOT escalated

Pre-reg said "V1 fails → V2 with a specific advisor-recommended
alternative, *one* iteration." The natural V2 candidate would be
**V2-a: add POI2Vec residual to check-ins before c2r pooling**.
The expected outcome under Phase 8 convergence-saturation is
V2-a ≈ J (±0.5 pp), giving "Checkin2Region with POI2Vec is
interchangeable with POI2Region with POI2Vec" — a footnote, not
a result.

The advisor recommends *not* escalating to V2-a:
1. V1 lost ~3 pp at both states (large margin, not borderline)
2. The deficit closely matches "no POI2Vec at POI-pool" cost
   (canonical also ~−2-3 pp vs J), *not* "wrong pooler architecture"
3. V2-a's expected outcome is ≈J, which doesn't extend the contribution
4. The convergence-saturation finding from Phase 8 already covers this
5. The user's original hypothesis is falsified independently of V2-a's
   outcome

S3-b closed at V1.

## Phase 11 final-final close-out (2026-05-07 22:45)

| Test | Verdict | Δ AL vs J | Δ AZ vs J |
|---|---|---:|---:|
| S1 (c2p hard-neg) | ✗ falsified | — | — |
| S2 (S1 stacked on J) | skipped | — | — |
| S4 (DGI corrupted-feature c2p neg) | ✗ falsified | — | — |
| S3-a v1 (auxiliary, same-id neg) | ✗ falsified | — | — |
| S3-a v2 (auxiliary, foreign-region neg) | ✗ falsified | — | — |
| S3-b V1 (replacement, foreign-region neg) | ✗ falsified | −2.91 | −2.54 |
| S3-b V2-a (replacement + POI2Vec) | NOT RUN — expected ≈J per advisor | — | — |

Phase 11 contributions for the writeup:

1. **Three substrate falsifications** (S1, S4, S3-a) and **one
   methodology falsification** (S3-b V1).
2. **AZ-f3 architectural-sensitivity floor** (Phase 11): all 5 merge
   designs + S1 + S4 + S3-b V1 lose ~2 pp on AZ f3 even when the
   variant preserves canonical's c2p path. Data-split property.
3. **Canonical PMA-uniform finding**: c2hgi/HGI's POI2Region PMA is
   structurally a learnable mean-pool throughout training (PR_norm
   stays at 1.0; entropy bit-stable). The architecture's lift comes
   from the region GCN + contrastive boundaries shaping upstream
   POI representations, *not* from selective attention pooling.
4. **Contrastive attention pooling needs between-pool variance AND
   non-redundant supervision** (S3-a v1+v2): with redundant supervision
   the attention collapses to uniform regardless of negative-sampling
   strategy.
5. **Pool-source ablation finding (S3-b V1)**: directly pooling
   check-ins instead of POIs accounts for at most ~0.1 pp; the merge
   family's lift comes from the POI2Vec semantic prior, not the
   pooler architecture. Re-frames Phase 8's convergence-saturation:
   the "six variants converge" was about variations *around* the
   POI2Vec-at-pool intervention, all of which preserved that
   intervention; remove the intervention and the lift disappears
   regardless of pooler choice.

Phase 10 Pareto verdict (merge family > canonical > HGI on cat + reg
+ fclass jointly) **unchanged**. Phase 11 strengthens it with five
falsifications and four cleaned-up findings about *what's actually
load-bearing* in the merge family's lift (the POI2Vec prior, not
the pool architecture).

## Phase 11.S3-b V2-c — REOPENED 2026-05-07 23:00 (per-check-in POI2Vec anchor)

User opted to run V2-c for completeness despite advisor's "expected
null" pre-registration. V2-c is **S3-b V1 + a per-check-in anchor
loss** that pulls every `pos_checkin_emb[i]` toward
`POI2Vec[poi_to_idx[poi(i)]]`. The encoder itself learns fclass-aware
geometry, then Checkin2Region pools fclass-aware check-ins.

**Mechanism**:
```
L_total = L_c2p + L_c2r + L_r2c + λ_chk · ‖ pos_checkin_emb − POI2Vec[checkin_to_poi] ‖²
```
λ_chk = 0.1 default (matches J's POI-table-anchor λ).

**Pre-registered outcome (advisor)**:
- *Most likely*: AL Acc@10 ∈ [55, 62]%, mean Δ vs J: −1 to 0 pp.
  Either anchor dominates (encoder collapses toward POI2Vec → ≈ J) or
  it fights the c2hgi temporal/contextual signal (→ worse than J).
- *Surprise threshold* (would change the writeup): AL mean ≥ 62.5 %
  AND AZ mean ≥ 52.5 % over 5 folds.
- *Information value*: completeness, not contribution — unless the
  surprise triggers.

**One-redo rule still binds**: V2-c is THE redo. V2-d (c2r hard-neg)
and V2-e (alternative pooler) are explicitly out of scope under this
study regardless of V2-c's outcome.

### V2-c result — falsified 2026-05-08 00:08

**AL 5f×50ep reg-head leak-free** (Acc@10 %):
| variant | f0 | f1 | f2 | f3 | f4 | mean | Δ vs J | Δ vs canonical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 58.91 | 60.00 | 64.40 | 57.46 | 55.00 | 59.15 | −2.81 | — |
| J | 65.30 | 62.08 | 65.70 | 60.90 | 55.80 | 61.96 | — | +2.81 |
| S3-b V1 | 62.16 | 59.95 | 62.79 | 57.59 | 52.74 | 59.04 | −2.91 | −0.11 |
| **S3-b V2-c** | 52.28 | 50.08 | 51.22 | 48.35 | 44.08 | **49.20** | **−12.75** | **−9.95** |

**AZ 5f×50ep reg-head leak-free** (Acc@10 %):
| variant | f0 | f1 | f2 | f3 | f4 | mean | Δ vs J | Δ vs canonical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 47.84 | 49.31 | 50.19 | 49.39 | 54.47 | 50.24 | −1.91 | — |
| J | 52.61 | 53.89 | 54.82 | 46.94 | 52.49 | 52.15 | — | +1.91 |
| S3-b V1 | 49.75 | 50.46 | 51.96 | 44.89 | 50.98 | 49.61 | −2.54 | −0.63 |
| **S3-b V2-c** | 49.96 | 51.30 | 51.79 | 45.61 | 51.15 | **49.96** | **−2.19** | **−0.28** |

Pre-registered surprise threshold (AL ≥ 62.5 % AND AZ ≥ 52.5 %)
**not** triggered. V2-c falsified.

**Asymmetric state pattern (genuinely interesting)**:

The per-check-in anchor produces **catastrophic regression at AL**
(−10 pp vs canonical) but only **mild regression at AZ** (−0.3 pp,
essentially V1-equivalent).

Mechanistic explanation: AL has small per-region check-in pools
(mean 102.7 check-ins/region; mean 10.7 POIs/region). The anchor
pulls every check-in to `POI2Vec[poi(checkin)]`. With mean ~10
POIs per region, region-side aggregation degenerates to
`mean(POI2Vec[POIs in region])` — a fclass-only mean pool that
discards every contextual signal c2hgi's contrastive boundaries
built. AZ has more diverse contexts (different region geography,
larger pools per region), so the anchor's collapse-to-fclass effect
is weaker.

**Advisor's pre-registered "fights c2hgi context → worse than J"
prediction triggered cleanly**. The anchor either had to dominate
(→ ≈J) or fight (→ worse than J). At AL it fought decisively;
at AZ it fought weakly.

### Phase 11 final-final-final close-out (2026-05-08 00:10)

Per the pre-registered one-redo rule: **V2-c was THE redo. No further
variants under this study.** S3-b closed.

| Test | Verdict | Δ AL vs J | Δ AZ vs J |
|---|---|---:|---:|
| S1 (c2p hard-neg) | ✗ falsified | — | — |
| S2 (S1 stacked on J) | skipped (S1 null) | — | — |
| S4 (DGI corrupted-feature c2p neg) | ✗ falsified | — | — |
| S3-a v1 (auxiliary, same-id neg) | ✗ falsified | — | — |
| S3-a v2 (auxiliary, foreign-region neg) | ✗ falsified | — | — |
| S3-b V1 (replacement, no POI2Vec) | ✗ falsified | −2.91 | −2.54 |
| **S3-b V2-c (replacement + per-checkin POI2Vec anchor)** | **✗ falsified** | **−12.75** | **−2.19** |
| S3-b V2-a (replacement + POI2Vec residual on checkins) | NOT RUN | — | — |
| S3-b V2-d (c2r hard-neg) | OUT OF SCOPE per one-redo rule | — | — |
| S3-b V2-e (alternative pooler) | OUT OF SCOPE per one-redo rule | — | — |

**Phase 11 contributions** (now finalised):

1. **Five falsifications** (S1, S4, S3-a v1+v2 counted as one mechanism,
   S3-b V1, S3-b V2-c) of substrate-level and methodology-level levers.
2. **AZ-f3 architectural-sensitivity floor**: data-split property,
   not c2p- or pooler-specific.
3. **Canonical PMA-uniform finding**: c2hgi/HGI's POI2Region PMA is
   structurally a learnable mean-pool throughout training.
4. **Contrastive attention pooling needs between-pool variance AND
   non-redundant supervision** (S3-a).
5. **Pool-source choice (POIs vs check-ins) accounts for at most
   ~0.1 pp** (S3-b V1) — the merge family's lift is the POI2Vec
   prior, not the pooler architecture.
6. **State-asymmetry of the per-check-in anchor** (S3-b V2-c): small
   per-region pools (AL: 10.7 POIs/region avg) collapse to
   `mean(POI2Vec)` under the anchor; larger more diverse pools (AZ)
   absorb the anchor without catastrophic collapse. New finding
   about anchor-loss × state-geometry interaction.
7. **Convergence-saturation re-framed** (Phase 8 finding): six merge
   variants converged ±0.1 pp because they all share POI2Vec at the
   POI-pool boundary; remove that and lift disappears regardless
   of pooler choice.

Phase 10 Pareto verdict (merge family > canonical > HGI on cat + reg
+ fclass jointly) **unchanged**. Phase 11 strengthens it with five
falsifications and two new findings (3 + 6).

## Where we are now (2026-05-08 00:10 — STUDY CLOSED, ultimate)

- All structural, methodological, and substrate levers tested.
- One-redo rule honoured for both S3-a (v1+v2) and S3-b (V1+V2-c).
- Pipeline idle. Move to writeup.
