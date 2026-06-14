# canonical_improvement Coverage Audit vs the substrate-protocol-cleanup Baseline

**Date**: 2026-05-29
**Author**: read-only audit (no GPU, no edits to code/recipes)
**Question**: Is there a validated `canonical_improvement` improvement that is MISSING from the B9/H3-alt baseline recipe used by the recently-finished substrate-protocol-cleanup Tier B / FL comparisons?

**Bottom line**: **Baseline is COMPLETE w.r.t. canonical_improvement on the recipe axis.** The study promoted exactly **two** findings to its shipping stack — `v3c` (AdamW WD=5e-2) and `T3.2 ResidualLN encoder` — and **both are substrate/encoder-side** (they change how the Check2HGI *embeddings* are trained), **not recipe-side** (the downstream MTL recipe is unchanged B9full). Neither is in the *default* shipped substrate (both are opt-in CLI flags; the engine still defaults to `encoder='gcn'` + plain Adam, no WD), but per the substrate-protocol-cleanup regime finding that is **immaterial to the MTL reg verdict**: substrate gains live on the STL/cat axis, and MTL reg is anchor/regime-limited regardless of substrate. The user's gut ("we're missing something") points at real promoted findings — but they are substrate-side cat micro-improvements, not a missing MTL recipe flag.

---

## 1. The B9 recipe (what's IN the baseline)

Per `docs/NORTH_STAR.md §Champion` and the substrate-protocol-cleanup Tier B/FL cells:

```
--mtl-loss static_weight --category-weight 0.75
--scheduler cosine (B9, FL) / constant (H3-alt, AL/AZ) --max-lr 3e-3
--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
--alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5   (B9 only)
--cat-head next_gru --reg-head next_getnext_hard
--task-a-input-type checkin --task-b-input-type region
--per-fold-transition-dir ... --batch-size 2048 --epochs 50 --folds 5
NO --log-t-kd-weight
```

`next_getnext_hard` uses `alpha_init=0.1` on the α·log_T anchor.

---

## 2. Every canonical_improvement experiment — verdict + classification

| Exp (INDEX/log) | Mechanism | Verdict | Side | In B9 baseline? | Evidence |
|---|---|---|---|---|---|
| **T1.1** leak audit | linear-probe floor establishment | PASS (gate, not an improvement) | n/a (protocol) | n/a | log.md:219 — floor reproduces historical AL within σ |
| **T1.3** α-ratio sweep (c/p/r boundary weights) | embedding loss boundary-weight triple | **FALSIFIED** at n=4–6 grids | substrate-side | n/a (no winner) | log.md:259–276, INDEX:873 — max \|Δreg\| ≤0.30 pp, all \|Δcat\| ≤1.18 pp (negative); explains <3% of HGI gap. **No α value promoted.** Canonical α=(0.4,0.3,0.3) unchanged. |
| **T1.4** best-epoch by fclass probe | selector | CLOSED — deferred-closed (never run) | recipe-side | no (intentionally; self-leak risk) | log.md:1428 |
| **T1.5** optimizer hygiene (warmup/cosine/AdamW WD) | **embedding-training optimizer** | **v3c promoted (modest, provisional)** | **substrate-side** | partial — opt-in `--weight-decay 0.05` on `check2hgi.py`; absorbed by T3.2 | log.md:354–385 — v3c=AdamW WD=5e-2 on the *embedding* trainer; AL cat +0.38, reg +0.09; all within σ at n=5. Carried as provisional canonical. |
| **T1.6** epoch budget | embedding training length | non-improvement (ep300 = exploration default; ep500 = shipping) | n/a (budget) | yes (FL keeps ep500) | log.md:387–414 — reg plateaus by ep50; no metric change |
| **T1.2** multi-seed | protocol | CLOSED — done-in-passing | n/a | n/a | log.md:1429 — covered by STACKING_ABLATION 5-seed |
| **T2.1** p2r hard-neg rate | embedding loss negatives | **FALSIFIED (reg-axis)**; cat inconclusive at n=5 | substrate-side | no (canonical 25% rate kept) | log.md:455–537, INDEX:1105 — reg flat 0.09–0.11 pp across rates {0,0.25,0.5,0.75} |
| **T2.2** InfoNCE at p2r | embedding loss shape | CLOSED — deferred (never run) | substrate-side | no | log.md:1430 |
| **T2.3** two-pass corruption | embedding loss | CLOSED — deferred (never run) | substrate-side | no | log.md:1431 |
| **T2.4** DropEdge (asym + sym) | embedding graph regulariser | **FALSIFIED — does NOT stack with v3c** | substrate-side | no | log.md:580–614 — budget-rescue trick only; over-regularizes at ep500 |
| **T3.1** GATv2 encoder (+/− edge_attr) | encoder architecture | **FALSIFIED — structural label leak** (cat→99%) | substrate-side | no | log.md:683–789, 1195–1196 — attention × temporal × user_seq triangle |
| **T3.2** ResidualLN encoder | encoder architecture | **PROMOTED — paper-grade cat micro-improvement** | **substrate-side** | partial — opt-in `--encoder resln`; engine default still `gcn` | log.md:884–998, 1376–1386 — cat +0.86 FL / +1.48 AL / +1.70 AZ, 5/5 seeds p=0.03125; reg ≈0 small states, +0.71 FL (mostly v3c); leak +2.24 IJM-verified honest |
| **T3.3** R-GCN (num_bases 2/1) | encoder architecture | **FALSIFIED** (K=2 leak +27.85; K=1 capability collapse) | substrate-side | no | log.md:1022–1106 |
| **T3.4** Time2Vec encoder (rand/warm) | encoder input feature | **FALSIFIED — reg-only trade-off, loses cat** | substrate-side | no | log.md:1064–1146 — substitute-not-stack for v3c; cat −0.56 5/5 negative |
| **T4.1** GraphMAE (masked check-in feat) | embedding aux loss | **FALSIFIED — no cat lift any λ** | substrate-side | no | log.md:1263, 1331 |
| **T4.3** POI side-features (popularity/hours/co-visit) | embedding input feature | **FALSIFIED — AL cat +0.63 doesn't replicate at FL** | substrate-side | no | log.md:1267–1333; Hyp A multi-seed reg-kill, STACKING §6.2 |
| **T4.4** Delaunay POI-POI spatial edges (uniform GCN) | embedding graph | **FALSIFIED — over-smoothing, cat −11.30 AL** | substrate-side | no | log.md:1312–1333 |
| **T4.2** SwAV prototypes | embedding aux loss | SKIPPED (never run; ≤20% prior) | substrate-side | no | log.md:1351, 1433 |
| **T5.1** native POI-ID embedding | per-POI identity params | **DEAD — V2-c pool collapse** (AL Δreg −6.37) | substrate-side | no | log.md:1707, INDEX:460 |
| **T5.2a** Node2Vec POI-POI + alignment | POI-POI co-occurrence | **§Discussion-only** — cat regression both states | substrate-side | no | log.md:1758, INDEX:460 |
| **T5.2b** masked POI feature recon | POI feature self-supervision | **§Discussion-only** — 13/15 cat+ p=0.0074 but fails Bonferroni m=28; reg flat | substrate-side | no | log.md:1833, INDEX:460 |
| **T5.3** multi-view co-training | POI cross-view alignment | **§Discussion-only** — AZ reg d=+0.85 p=0.065, sub-Bonferroni | substrate-side | no | log.md:1900, INDEX:460 |
| **T6.1** log_T knowledge-distillation λ sweep | reg-head aux loss | **FALSIFIED** (joint_geom_simple gate failed every cell) | recipe-side-adjacent | n/a | log.md tail (Phase-3); INDEX:2100 |
| **T6.2** α / w_r grid (anchor weight + reg loss weight) | reg-head + loss weighting | **FALSIFIED** — diagnostic cell +0.76 reg at −3.55 cat (Pareto trade) | recipe-side | no | INDEX:2100, log.md tail |
| **T6.3** low-rank POI side-channel | substrate | **FALSIFIED at G3 kill-check** (AZ r=8 trips) | substrate-side | no | log.md:2545–2613 |
| **T6.4** substrate variants (both/infonce/two-pass) | embedding loss | **FALSIFIED** — "+11 pp" was a cross-selector artefact | substrate-side | no | log.md:2237, INDEX:2237 |

**Tier-6 closed FALSIFIED across all four pre-registered mechanism families** (commit `bc7fc27`).

> Note on T6.2: the only place a *recipe-side* knob (α anchor weight, w_r reg loss weight) was swept. It **FALSIFIED** — the one diagnostic cell that lifted reg (+0.76 pp at α=2.0, w_r=0.3) did so only on the per-task-disjoint front and at an unambiguous −3.55 pp cat cost; the deployable `joint_geom_simple` gate failed at every cell. So no recipe-side α/weight change was promoted. This directly answers the "is the α anchor in B9 the T1.3 winner?" question: **B9's `alpha_init=0.1` is unchanged — T1.3 falsified the α-boundary sweep and T6.2 falsified the α-anchor/loss-weight sweep.**

---

## 3. The shipping stack and where it actually lives

`canonical_improvement` final shipping stack (log.md:1376, STACKING_ABLATION.md §5):

```
canonical Check2HGI + v3c (AdamW WD=5e-2) + T3.2 ResidualLN encoder
```

Both components are **embedding-training (substrate) options**, applied when *generating the Check2HGI embeddings*, then fed to the **unchanged B9full MTL recipe** downstream (log.md:431, 675: "Downstream MTL: B9full (canonical NORTH_STAR §Champion ...)"). The study never proposed a single MTL recipe-flag change.

**Are they in the shipped engine default?** No (verified in code):
- `research/embeddings/check2hgi/check2hgi.py:267` — `_encoder_name = getattr(args, 'encoder', 'gcn')` → default encoder is the **old 2-layer `CheckinEncoder` (GCN)**, not `ResidualLNEncoder`. ResLN is opt-in via `--encoder resln`.
- `research/embeddings/check2hgi/check2hgi.py:673,680–685` — `weight_decay = getattr(args, 'weight_decay', 0.0)` → default is **plain Adam, no WD**. v3c is opt-in via `--weight-decay 0.05`.

So unless the substrate-protocol-cleanup baseline embeddings were regenerated with `--encoder resln --weight-decay 0.05`, the baseline ran on the **canonical (gcn, no-WD) substrate**, i.e. WITHOUT the two promoted improvements.

---

## 4. WHAT WE'RE MISSING (if anything)

### 4a. Recipe-side (would be MATERIAL to the MTL verdict) — **NOTHING MISSING**

There is **no promoted recipe-side improvement** in canonical_improvement:
- **T1.3 α-sweep**: FALSIFIED. B9's α stays canonical. ✔ matches.
- **T1.5 optimizer hygiene**: v3c is the *embedding-trainer* optimizer (AdamW WD on `check2hgi.py`), **not** the downstream MTL optimizer. The MTL optimizer (AdamW per-head LR, alt-SGD, alpha-no-WD) was never touched by the study. No MTL LR/WD/eps/accumulation/grad-clip change promoted. ✔
- **T6.2 α-anchor / reg-loss-weight grid**: FALSIFIED (deployable gate failed; reg lift only at a cat cost). ✔
- **T1.4 / T1.6**: selector/budget — falsified-closed or non-improvement; B9's `--min-best-epoch 5`, ep50 (small)/ep500 are unaffected. ✔

**Conclusion: the B9/H3-alt recipe used by substrate-protocol-cleanup is complete w.r.t. every recipe-side lever canonical_improvement examined.** No missing flag.

### 4b. Substrate-side (STL/cat-axis per regime finding → IMMATERIAL to the MTL reg verdict) — two opt-in items

Ranked by effect size:

1. **T3.2 ResidualLN encoder** (`--encoder resln`) — cat F1 **+0.86 FL / +1.48 AL / +1.70 AZ** (5/5 seeds, p=0.03125), reg ≈0 at small states / +0.71 at FL (mostly v3c). Validated downstream via STL `next_gru`-style cat probe AND B9full MTL cat. **Not in the shipped engine default.**
2. **v3c (AdamW WD=5e-2)** (`--weight-decay 0.05`) — FL reg +0.63 standalone, **fully absorbed by T3.2 in the stack** (STACKING_ABLATION §6.1: Hyp D = drop-v3c is statistically equivalent; v3c retained only for "protocol inertia + 5-state coverage", NOT mean contribution). **Not in the shipped engine default.**

**Why these are immaterial to the substrate-protocol-cleanup MTL reg verdict.** The substrate-protocol-cleanup regime finding (`CLOSURE.md §isolation cell`, log.md D1) is decisive: under B9 joint training the **MTL reg head is α·log_T-anchor-dominated** — with α frozen to 0, reg floors at chance for BOTH designs AND canonical; the substrate-carrying encoder branch contributes ~nothing to MTL reg. Even **HGI** (the STL reg ceiling) gives **no MTL reg advantage** (Δ+0.51, p=0.41 NS). So a *better cat-axis substrate* (T3.2 ResLN, +0.86–1.70 pp cat) cannot change the MTL **reg** verdict, and its cat lift is on the same axis substrate-protocol-cleanup already treats as a "free upgrade any architectural champion can adopt" (considerations.md:16). It does NOT undermine any Tier B reg-NULL conclusion.

**Caveat worth flagging**: if the substrate-protocol-cleanup baseline was meant to represent the *best-known canonical substrate*, it arguably should have used `--encoder resln` (the only paper-grade canonical_improvement substrate win, +0.86–1.70 pp **cat**). Since Tier B's cat deltas (−1.7 to −1.9 pp) were diagnosed as a build-scope CheckinEncoder-reinit confound (CLOSURE.md:61, log.md D3) rather than a substrate effect, and the study's verdict is a **reg** verdict, the missing ResLN does not change any conclusion. But the absolute cat baseline is ~1–1.7 pp below the canonical_improvement-best substrate. This is a documentation nuance, not a verdict error.

---

## 5. Verdict line

> **Baseline is COMPLETE w.r.t. canonical_improvement on the recipe axis** (no promoted recipe-side flag exists to be missing; T1.3, T1.5-MTL, T6.2 all falsified or substrate-scoped). The only two promoted findings (v3c, T3.2 ResLN) are **substrate/encoder-side**, are absent from the *default* shipped engine (opt-in flags), and are **immaterial to the MTL reg verdict** by the study's own anchor/regime finding. The user's intuition correctly identifies two real promoted findings — but they are substrate-side cat micro-improvements, not a missing MTL recipe element.
