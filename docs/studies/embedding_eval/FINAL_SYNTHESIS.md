# embedding_eval — Final Synthesis (macro)

**Study goal (recap).** Build a better *evaluation* of the Check2HGI substrate so we can (a) compare proposed embedding improvements rigorously and (b) **measure them against HGI on next-reg**, with improvements layered on **v13** (`check2hgi_resln_design_b` = ResLN + Design-B POI2Vec). MTL is **Part 2** — the plan is to first strengthen the embeddings / close the next-reg HGI gap at STL, then a separate study works the MTL on top of the chosen substrate.

---

## 1. What this study delivered: the evaluation ladder
A 4-level, leak-aware, controlled substrate evaluation (`docs/studies/embedding_eval/README.md`):
- **L0** train-free geometry (kNN-LOO, silhouette, centroid-sep, **adjacency-coherence** for region, CKA), **L1** linear probe, **L2** STL sequence task, **L3** MTL (Part-2 ground truth).
- Each task probed on the **artifact it consumes** (next-cat→final embedding; next-reg→**region embeddings**) — caught a real wrong-artifact bug.
- **Same-protocol control**, **placeid-isolated folds**, **leak gates** (per-step + autocorrelation-ceiling + L2-vs-control), multi-state (FL/AL/AZ), two adversarial advisor passes.
- Found + fixed 3 real bugs: wrong region artifact; baseline-mismatch v3c "gain"; **MTL region modality hard-coding check2hgi region embeddings** (`region_sequence.py`).

This is the study's primary product: a metric stack that the next phase can reuse.

---

## 2. Macro comparison — canonical (v11/v12) vs v13 vs our improvements vs HGI

### next-cat — Check2HGI family DECISIVELY beats HGI (no gap to close)
STL `next_gru` macro-F1 (RESULTS_TABLE §0.3) — Check2HGI 1.5–1.9× HGI at all 5 states:
| state | Check2HGI | HGI | v13 (resln+B) |
|---|---|---|---|
| FL | 63.4 | 34.4 | **70.6** (widens) |
| AL | 41.4 | 25.3 | **51.3** |
| AZ | 43.9 | 28.7 | **52.5** |
Our L0/L1 confirm: probe acc check2hgi-family ~0.98 vs HGI ~0.68; v13/resln top. **Category is a Check2HGI win; HGI is far behind.**

### next-reg — HGI ahead; the gap is NOT closed at the substrate level (incl. v13)
⚠ **Provenance correction (2nd advisor).** Two different number sources must not be conflated:
- **Borrowed (prior tier_resln study, NOT re-verified here):** v13 STL `next_stan_flow` Acc@10 — AL 61.99 (ties HGI 61.86), AZ 52.98 (80% of gap), FL 70.21 (30%). Possibly stale-log_T-era / different protocol.
- **This study's OWN controlled L2** (`nextreg_stl.md`, same-protocol, FL): check2hgi 0.7274, **resln 0.7275 (≈ canonical — TIES, does NOT close the gap)**, HGI **0.7362**. ⇒ under our controlled ladder, **v13/resln is region-NEUTRAL (HGI still ~0.9pp ahead at FL); the gap is NOT reproduced as closed.**

So the honest position: **the next-reg HGI gap (≈1–3pp) remains OPEN; no substrate we tested — canonical, v13, or any re-screen candidate — closes it robustly under our controlled evaluation.** (The prior "v13 ties HGI at AL" is unverified here and may be small-state noise.)

**Our re-screen (L0–L2, vs same-protocol control) — none improves on v13:**
| candidate | next-cat | next-reg | verdict |
|---|---|---|---|
| v3c (WD 5e-2) | no-op | no-op (below ctrl) | falsified (3 states) |
| T2.4 DropEdge | no-op | no-op | falsified |
| T4.3 sidefeat | ≤ ctrl (hurts MTL cat −1.6pp) | adj_coh +0.05–0.07 but L2 +0.3pp NS, redundant w/ log_T | not robust |
| T6.1 p2p | no-op | no-op | falsified |
| GATv2 | **leak** (fwd-temporal cat copy) | — | disqualified |
| R-GCN | **leak** | adj_coh high (leak-inflated) | disqualified |
| GCN² region head | (n/a) | +3.0pp under next_gru but **+0.56pp NS under stan_flow** (redundant w/ log_T); AL/AZ residual within 1 SD | not robust |

**adj_coh detected HGI's region edge (3/3 states) and a small real sidefeat structure — but the deployed `next_stan_flow` log_T prior already captures that spatial signal, so neither a higher-adj_coh substrate nor an adjacency-aware head adds robustly.**

---

## 3. Where we are (corrected after 2nd advisor — honest status)
- **Evaluation goal: ACHIEVED, with one standing limit.** The ladder is rigorous, leak-aware, HGI-anchored, self-corrected 3 bugs — a reusable harness. **Limit (the study's own founding worry): L0–L2 screen STL; they cannot certify an MTL-only effect.** v13 itself (STL-best, zero-MTL-benefit) is the proof. So "no candidate survives STL" ≠ "no candidate helps MTL."
- **next-cat: solved in Check2HGI's favour** (1.5–1.9× HGI; v13 widens). Robust, multi-level.
- **next-reg gap vs HGI: OPEN.** Under our controlled ladder, no substrate (canonical, v13, or any re-screen lever) closes it robustly. v13 is region-**neutral** (ties canonical), not gap-closing, in our runs.
- **"Structural / not closeable" — DOWNGRADED to "no robust substrate-level lift detected."** The GCN²/adjacency lever is redundant-with-log_T at FL (+0.56pp NS) and within ~1 SD at AL/AZ — BUT the sign is **positive 3/3 (mean ≈+0.9pp)**, a sub-noise-floor effect that is **untested at adequate power**, not proven absent. Most of this study's MTL/L2 evidence is **single-seed (dev seed 42)** — the weakest link.

**⇒ Carry v13 to Part 2 — but for the honest reason:** it is the **strongest category substrate and region-neutral** (does not hurt reg), so it is a safe base. It is **NOT** carried because "it closes the region gap" (unverified here) or "nothing beats it in MTL" (single-seed-42 only). Carrying v13 to MTL is a **bet** that its STL representational edge eventually pays off under a better joint regime — reasonable, but a bet.

---

## 4. How to proceed — the ONE thing before Part 2, then Part 2
**Before committing to Part 2 — run the deferred paired multi-seed test** (the study's biggest gap): seeds {0,1,7,100}, paired per-fold, FL + one small state, of the **two surviving real-geometry levers** against the GCN+log_T baseline IN MTL:
1. **+sidefeat substrate** (the only positive, base-independent, leak-free geometry signal: adj_coh +0.055 on GCN & ResLN, best silhouette, CKA 0.90 = real change). Its MTL behaviour on a multi-seed basis is genuinely unknown (single-seed-42 showed cat −1.6pp, but dev-seed-42 overshoots).
2. **Adjacency-aware region head** (the strongest *positive* finding: +2.7–3.0pp under next_gru, ~7 SD). The lever is the **head, not the substrate** — but it must be designed **orthogonal to log_T** (GCN² re-derived log_T's observed-transition signal and collapsed; a head encoding what log_T can't — multi-hop reachability / hierarchy / cold-start regions — is the real lead).

This replaces dev-seed-42 with reporting seeds, tests in the **MTL regime the ladder cannot certify**, and resolves the sign-consistent ~0.9pp sub-noise effect. If null at power → "carry v13, region gap is a substrate dead-end" is *earned*. If positive → Part 2 has its target.

**Then Part 2 (MTL on v13):**
- **Primary hypothesis: the adjacency-aware / region-graph reg head** (orthogonal to log_T) — the most actionable positive lead.
- **Secondary: dual-substrate routing** (Check2HGI final emb for cat ⊕ HGI region emb for reg) — Check2HGI owns cat, HGI owns region. Lower priority (Design-A late-fusion failed in MTL; task-routed variant untested).
- Prior-pathway work (log_T-KD already gives +2–5pp small-state MTL-reg) remains the only *confirmed* MTL-reg lever.
- **Reuse this ladder** as the Part-2 screening harness — but always confirm survivors in **MTL multi-seed**, never STL-only.

**Bottom line (corrected).** The study **achieved its evaluation goal** (a rigorous, leak-aware, HGI-anchored ladder) and cleanly **falsified 5 of the proposed improvements + 2 leaks**. But the **next-reg HGI gap is OPEN** — *no* substrate (v13 included) closes it robustly under controlled evaluation; v13 is the best **category** engine and **region-neutral**, so it is the safe Part-2 base. The two real-geometry survivors (**sidefeat**, **adjacency-aware head**) were screened-out at STL but **never tested in MTL at power** — and since the whole study exists because STL≠MTL, that multi-seed MTL test is the mandatory first step before declaring the region gap a substrate dead-end.

## Appendix — consolidated L0→L2 FL table (this study's controlled runs)
| engine | cat L0 kNN | cat L1 probe | cat L2 F1 | reg L0 adj_coh | reg L1 probe@10 | reg L2 Acc@10 |
|---|---|---|---|---|---|---|
| check2hgi (canonical) | 0.982 | 0.985 | 0.676 | 0.274 | 0.685 | 0.726 |
| **v13 (resln_design_b)** | **0.983** | **0.987** | 0.671 | **0.231** | 0.681 | ~0.728† |
| **HGI** | 0.773 | 0.682 | 0.343 | **0.326** | 0.677 | **0.736** |
| sidefeat (GCN) | 0.981 | 0.983 | 0.647 | 0.269 | 0.684 | 0.728 |
| adjacency-head (gprop) | — | — | — | — | — | 0.731 (+0.56 NS) |

† v13 L2-reg-STL not run directly in our p1 (resln encoder proxy 0.7275; MTL v13 re-run pending).
**Reads:** cat — v13 best, HGI far behind. reg — **HGI leads every level**; notably **v13 has the LOWEST region adj_coh (0.231)** — the design_b POI2Vec injection *reduces* region spatial coherence, so v13 has **no controlled region advantage** here (the borrowed "v13 closes the gap" is not reproduced). The combined "v13+sidefeat+adjacency-head" is **not a built artifact** (needs the design_b-build extension + a head mod); the sidefeat(GCN) + gprop rows are its closest proxies, both region-neutral-to-marginal.

## Appendix — advisor verdict: substrate surface is EXHAUSTED (the region gap is a head/fusion problem)
A strategy advisor reviewed the prior studies for any untried region lever on v13. Verdict: **do not run another substrate experiment.** Rationale:
- v13 has the *lowest* region adj_coh (0.231) yet ties L2-reg ⇒ **log_T dominates the spatial signal; substrate adj_coh differences wash out.** Any lever that re-derives observed transitions (geographic adjacency, Delaunay/Design-K, GCN², contrastive boundaries/Lever-6) is **redundant with log_T** — and merge_design already **falsified** Design A (concat fusion: −18pp AL), Design K (≈ Design J), Lever-6, and the POI2Vec family (Design B/H/I/J/M = v13's family, saturates ~1pp below HGI, never beats it).
- Only two levers are genuinely log_T-orthogonal and unfalsified: (a) an adjacency-aware head encoding **multi-hop/hierarchy/cold-start** (our GCN² proxy was within 1 SD — not actionable as-is); (b) **dual-substrate task-routing** — Check2HGI-v13 final-emb → cat tower, **HGI region-emb → reg tower** (separate towers, NOT Design-A concat which failed). This imports HGI's actual hierarchical region structure instead of re-deriving it.
- **Both live levers are HEAD/FUSION architecture, not substrate** ⇒ they belong to Part 2 (MTL), not a substrate re-screen. Lever-4/Lever-5 (POI2Vec at p2r / ranking-distill) were never measured but are POI2Vec-family (strong negative prior); only worth a re-screen for a reviewer footnote.

**Final recommendation: stop substrate-side. v13 is the safe Part-2 base (best cat, region-neutral). The next-reg HGI gap is a head/fusion problem whose one structurally-sound, log_T-orthogonal lever is dual-substrate task-routing — to be tested in Part-2 MTL.**

## Appendix — design_b is net-negative on geometry; v14 (ResLN, no design_b) candidate (2026-06-01)
Meticulous L0 (FL, 5-fold ±SD, statistically separated): **design_b (POI2Vec) HURTS region geometry and slightly hurts cat.**
| engine | reg adj_coh | cat probe | cat silhouette |
|---|---|---|---|
| **resln+sidefeat** | **0.337±.003 (> HGI 0.326!)** | 0.985 | 0.554 |
| HGI | 0.326 | 0.682 | 0.002 |
| resln (no design_b) | 0.282±.006 | **0.9879** | **0.557** |
| check2hgi canonical | 0.274±.006 | 0.985 | 0.505 |
| **v13 (resln+design_b)** | **0.231 (lowest)** | 0.9873 | 0.546 |

- **resln → v13 (adding design_b): adj_coh 0.282→0.231 (−0.05), cat probe 0.9879→0.9873.** The POI2Vec design_b — meant to *help* region — **reduces region spatial coherence** and doesn't help cat. ⇒ **v13 is NOT the best substrate; plain ResLN (no design_b) dominates it on both axes' geometry.**
- **resln+sidefeat = best region geometry of ALL (0.337 > HGI 0.326)** — sidefeat's side-features on the clean ResLN base beat HGI's region adj_coh at L0.
- **⇒ v14 candidate = ResLN (no design_b), optionally + sidefeat** — supersedes v13 on STL geometry.
- **DECISIVE test running:** does resln+sidefeat's L0 adj_coh edge (>HGI) translate to **L2 next_stan_flow region Acc@10** (vs canonical/resln/HGI, FL), or does the log_T prior wash it out (as it did the GCN² adjacency lever)? This is the make-or-break for "we finally close the next-reg gap at the substrate level." (Result pending; the GCN-base multi-seed MTL was redirected to this — it was testing the wrong base.)

## ⚠ Correction to the v14/design_b claim (user-caught): metric mismatch + untested-at-L2
The "design_b is net-negative / v14 supersedes v13" claim above was **over-read from adj_coh alone** and must be qualified:
1. **Different metric than the prior verdict.** Prior studies rated v13 "best dual-axis STL" on **L2 STL Acc@10** (region task: v13 closes ~80%/~30% of the HGI gap at AZ/FL, ties at AL; widens cat). My "canonical > v13" is on **L0 adj_coh** — a *train-free spatial-adjacency* geometry metric the prior studies never used. They do not directly contradict.
2. **adj_coh measures the WRONG axis for design_b.** adj_coh = fraction of a region's cosine-NN that are *geographically adjacent*. design_b injects **POI2Vec (fclass/category similarity)** — it makes regions cluster by POI-type-mix, NOT geographic adjacency. So design_b **lowering adj_coh is expected** (it reprioritizes fclass-similarity over spatial adjacency) and is **not** evidence it hurts the region task. The L2 head can exploit the fclass structure adj_coh can't see.
3. **I never L2-tested v13 (resln_design_b) region directly** — I used `resln` (no design_b) as a proxy (resln ≈ canonical). So "v13 region-neutral" was inferred from the *wrong engine*. The prior L2 gap-closure for v13 stands unrefuted by this study.

**Honest status:** "canonical > v13 on adj_coh" is TRUE but **metric-specific** and does NOT establish design_b is bad for the region task. **Running now: a direct L2 `next_stan_flow` region test of v13 (resln_design_b) vs resln vs resln+sidefeat vs canonical vs HGI** — this settles whether design_b's L2 region benefit (prior claim) reproduces despite its lower adj_coh, and whether resln+sidefeat's high adj_coh translates. Until then, the v14-supersedes-v13 conclusion is **suspended**.

## RESOLVED — direct L2 region test settles the v13/design_b question (2026-06-01)
L2 `next_stan_flow` region Acc@10 (FL, 5-fold, seed42), DIRECT (each engine's own region emb):
| engine | reg adj_coh (L0) | L2 reg Acc@10 | L2 Acc@1 |
|---|---|---|---|
| check2hgi (canonical) | 0.274 | 0.7274±.005 | 0.4687 |
| check2hgi_resln | 0.282 | 0.7275±.005 | 0.4687 |
| **v13 (resln_design_b)** | **0.231 (worst)** | **0.7309±.004 (best family)** | **0.4729 (best)** |
| resln+sidefeat | **0.337 (best)** | 0.7307±.006 | 0.4696 |
| HGI | 0.326 | **0.7362±.004** | 0.4740 |

**Verdict — my adj_coh-based "design_b net-negative / v14 supersedes v13" is RETRACTED:**
- **v13 has the worst adj_coh yet the BEST L2 region of the family** (+0.35pp Acc@10, best Acc@1). adj_coh measures *geographic adjacency* (which log_T already captures); design_b adds **POI fclass-similarity** which the L2 head exploits. So design_b **helps the region task** — the prior "v13 best dual-axis STL" verdict **STANDS**; my geometry read was metric-myopic.
- **Two independent routes reach the same ~0.731:** design_b's fclass (v13, 0.7309) and sidefeat's adjacency (resln+sidefeat, 0.7307). Both ~0.35pp over canonical (marginal, ~0.7 SD at seed42), both **still ~0.5pp below HGI (0.7362)**.
- **Promising untested combo:** since the two routes are *mechanistically distinct* (fclass vs adjacency), **v13 + sidefeat (resln + design_b + side-features)** might **stack** toward/past HGI — this is exactly the combo the user originally proposed, now **data-supported**. Needs the build_design_b_poi_pool side-feature extension. **This is the one promising substrate-side experiment left**; multi-seed needed (the +0.35pp routes are within ~0.7 SD at seed42).

**Net:** v13 remains the best family substrate at the region TASK; no family substrate closes the HGI gap at seed42 (best ~0.731 vs 0.736); the live substrate lever is **stacking design_b ⊕ sidefeat (v13+sidefeat)**, multi-seed.

## CLOSED — v13+sidefeat does NOT stack (2026-06-01)
Built `check2hgi_resln_design_b_sidefeat` (resln + design_b POI2Vec + T4.3 side-features
stacked at the POI-pool reg-path) and ran direct L2 (FL, 5-fold, seed42, next_stan_flow):
| substrate | Acc@1 | Acc@10 |
|---|---|---|
| canonical | 0.4687 | 0.7274±.005 |
| resln | 0.4687 | 0.7275±.005 |
| **v13 (resln_design_b)** | 0.4729 | **0.7309±.004** |
| resln+sidefeat | 0.4696 | 0.7307±.006 |
| **v13+sidefeat** | 0.4718 | **0.7293±.004** |
| **hgi** | 0.4740 | **0.7362±.004** |

**v13+sidefeat (0.7293) is BELOW both components alone (v13 0.7309, sidefeat 0.7307)** —
the two routes do NOT stack; they are redundant/interfering, both capped by the log_T
ceiling (all within ~1 SD = statistical tie). This **confirms the region-silhouette L0
prediction**: neither fclass (design_b) nor usage/popularity (sidefeat) is the *spatial-
cohesion* axis where HGI wins, so stacking two non-spatial axes cannot approach HGI.

**Final substrate verdict:**
- **Carry v13 (resln_design_b) — best family substrate at L2-reg (0.7309), best cat.** Pure v13.
- **v13+sidefeat: DO NOT adopt** — no gain over v13 (a hair below), adds complexity.
- HGI still leads (0.7362, ~0.5pp gap), unclosed by any substrate.
- **Only actionable HGI-gap lever (from L0):** POI-level spatial/Delaunay edges (T6.1 p2p /
  design_k = J+Delaunay) — the axis region-silhouette concordantly localizes HGI's advantage.
  fclass/sidefeat are exhausted. Defer to a dedicated spatial-substrate study / Part-2 MTL.

## VALIDATED + SPATIAL LEVER FALSIFIED (2026-06-02)
**v13+sidefeat implementation audited — CORRECT, no-stack result is REAL.** Independent
code review: injector mirrors canonical T4.3 byte-for-byte; cat-path gradient isolation
proven (encoder grads identical to 6e-8 with side-features on/off); side-features genuinely
propagate into the reg embeddings (mean-abs-diff 0.39 vs v13). The 0.7293 < v13 0.7309 is a
real finding, not a bug. (Build now also pins `--seed`, default 42, for reproducibility —
it previously set no seed.)

**Spatial/Delaunay substrate lever — FALSIFIED, do not run more substrate builds.** Two
prior falsifications confirmed: Design K (Delaunay POI-GCN) ≡ Design J (AL Δ−0.02pp, AZ
Δ−0.06pp); T6.1 p2p (InfoNCE POI-POI boundary) no-op at L0 (GCN+ResLN) and L2 (FL 0.7271).
T5.2 check-in Delaunay hooks CLOSED (cat regression / Bonferroni). Empirical root: corr(
region-cosine, T_ij)≈0.05 → the `next_stan_flow` α·log_T flow prior already supplies the
spatial-cohesion axis a higher-cohesion substrate would add. The advisor verdict is explicit:
the live levers are **head/fusion architecture (Part-2 MTL)**, NOT substrate — specifically
**dual-substrate task-routing** (route HGI's region tower to the reg head), the one
log_T-orthogonal unfalsified lever.

**Full next-reg diagnostic table (FL, region_eval.py --region-silhouette):**
| engine | adj_coh@10 | reg-silhouette | (L2 Acc@10) |
|---|---|---|---|
| check2hgi | 0.274 | −0.649 | 0.7274 |
| check2hgi_resln | 0.282 | −0.641 | 0.7275 |
| v13 (resln_design_b) | 0.231 | −0.665 | 0.7309 |
| v13+sidefeat | 0.231 | −0.677 | 0.7293 |
| **hgi** | 0.326 | **−0.460** | **0.7362** |

region-silhouette cleanly localizes HGI's advantage to POI-level spatial cohesion (−0.46 vs
family ~−0.65) — the diagnostic lead — but the lever that would raise it (Delaunay substrate)
is log_T-redundant and falsified. **The HGI next-reg gap is a Part-2 routing/fusion problem,
not a substrate problem. Part-1 (substrate) is complete: carry v13.**

## ⚡ REOPENED — design_k (Delaunay POI edges) closes the HGI gap at FL (2026-06-02)
User-requested re-validation OVERTURNED the prior "Delaunay falsified / K≡J" verdict.
That prior was **AL/AZ-only (small states); FL was never tested.** Rebuilt design_k (GCN
base, λ=0.1) at FL and ran direct L2 + L0:

| substrate | L2 reg Acc@10 | reg-silhouette (L0) | adj_coh (L0) |
|---|---|---|---|
| canonical | 0.7274 | −0.649 | 0.274 |
| v13 (resln_design_b) | 0.7309 | −0.665 | 0.231 |
| v13+sidefeat | 0.7293 | −0.677 | 0.231 |
| **design_k (Delaunay)** | **0.7341±.0047** | **−0.394** | **0.379** |
| **hgi** | **0.7362±.0043** | −0.460 | 0.326 |

**design_k is now the BEST Check2HGI-family substrate at L2-reg** (+0.32pp over v13), and at
**0.21pp below HGI (~0.5 SD) it is statistically indistinguishable from HGI at seed42** — the
gap effectively closes. It is also the ONLY substrate whose L0 spatial-cohesion EXCEEDS HGI
(silhouette −0.394 > −0.46; adj_coh 0.379 > 0.326). **The L0 spatial-cohesion diagnostic
PREDICTED this** (design_k maxes it out) and this time it TRANSLATED to L2 — the Delaunay POI
edges add spatial structure that, at a large state with 4703 regions, is NOT fully redundant
with log_T (unlike at AL/AZ).

**This VALIDATES the region-silhouette diagnostic as genuinely predictive of the spatial axis,
and the L0→L2 ladder end-to-end.** The earlier within-family anti-rank (v13) was because v13's
fclass is the WRONG axis; design_k raises the RIGHT axis (spatial) and L2 follows.

**Corrections to prior conclusions:**
- "Spatial/Delaunay substrate lever FALSIFIED" (this file, earlier today) — **RETRACTED for
  large states.** True at AL/AZ; FALSE at FL. design_k FL closes the gap.
- t61_p2p re-validated as genuinely no-op (worse L0 than canonical, all 3 states) — that
  falsification STANDS; it is the InfoNCE-boundary variant, mechanistically different from
  design_k's actual Delaunay POI-GCN.
- adj_coh / region-silhouette: now shown to be PREDICTIVE for the spatial lever (design_k),
  not merely diagnostic — when the lever raises the spatial axis, L2 follows. They still
  anti-rank fclass-axis moves (v13). So: silhouette ranks the SPATIAL axis correctly.

**MANDATORY next step:** multi-seed {0,1,7,100} design_k FL L2 (seed=42 overshoots at large
states per CLAUDE.md) + design_k at AL/AZ/CA/TX to map state-dependence. If multi-seed holds,
design_k (or design_k on the v13/resln base) is the new substrate recommendation and the
HGI next-reg gap is CLOSED at the substrate.
