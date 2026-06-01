# Embedding-eval ladder — lab log

## 2026-05-31 — Study created; L0+L1 built, advisor-reviewed, run

### What was done
- Formalized the 4-level ladder (`README.md`): L0 geometry (train-free), L1 linear probe, L2 capacity ladder (STL), L3 MTL (deployment). L0/L1 automated in `scripts/embedding_eval/`; L2/L3 reuse `scripts/train.py`.
- Adversarial methodology+code review (advisor subagent). Applied fixes: standardize probe inputs on train split (scale fairness); minibatch+early-stop probe (convergence fairness across 7 vs 4703 classes); similarity-weighted kNN (removes tie bias); post-mask subsampling + `n_eval`/`label_coverage`/`test_class_coverage`/`train_acc` provenance; ddof=1 + "SD≠CI" labelling; tempered doc claims (pooling-bias + self-prediction caveats; ρ-gate "not yet satisfied").
- Ran L0+L1 on `{hgi, check2hgi, check2hgi_design_b, check2hgi_resln, check2hgi_resln_design_b}`.
  - FL POI-pooled (all 5; HGI only exists at FL) → `docs/results/embedding_eval/fl_poi/`
  - FL check-in granularity, 150k items (4 check-in engines) → `fl_checkin/`
  - AL+AZ POI-pooled (4 check-in engines, ref=check2hgi) → `smallstates_poi/`

### Findings (L0/L1 — screening only; ρ-vs-L3 gate NOT yet measured)

**F1 — Category is saturated across the Check2HGI family; HGI is far behind.**
FL pooled probe acc: check2hgi 0.986, design_b 0.984, resln 0.988, resln_design_b 0.988, **HGI 0.681**. kNN agrees. Category differences *within* the family are ≤0.5 pp (noise). Check2HGI's contextual substrate encodes category near-perfectly; HGI does not.

**F2 — Pooling bias is real and large (validates advisor BLOCKER B1).**
check2hgi next-reg probe acc **0.051 (POI-pooled) → 0.232 (check-in)** — a 4.5× jump, top5 0.118→0.339. Pooling per-visit vectors to one POI-mean destroys exactly the contextual region signal Check2HGI is built to carry. **Corollary: the pooled "HGI (0.074) beats Check2HGI (0.051) on region" result is a pooling artifact and must NOT be read as a substrate verdict.** HGI has no check-in-level representation, so the two cannot be compared at the granularity where Check2HGI's region signal lives.

**F3 — Pooling helps category, hurts region (mirror image).**
check2hgi cat probe 0.986 (pooled) → 0.887 (check-in); reg 0.051 → 0.232. Category is a POI-constant property, so averaging check-ins denoises it; region depends on per-visit context, so averaging erases it. Granularity is not neutral — always declare it.

**F4 — ResLN is the strongest substrate on the geometry/probe axis; small but consistent.**
At FL pooled, FL check-in, and AL+AZ, `resln`/`resln_design_b` edge `check2hgi` on *both* cat (~+0.3 pp) and reg (~+0.2 pp), while barely changing the space (CKA(resln, canonical)=0.97). This is consistent with the documented v12/v13 "ResLN is STL-only, no MTL benefit" regime — the proxy sees a representational gain whose transfer to MTL is exactly the open ρ-gate question.

**F5 — design_b slightly *hurts* static region geometry, despite being the v13 recommended STL base.**
design_b is the worst family member on reg (FL pooled 0.044 vs 0.051 canonical) and perturbs the space most (CKA 0.85). Reading: design_b's STL win comes through the POI2Vec *teacher in the sequence head*, not through the static embedding geometry — so L0/L1 (which only see the frozen substrate) cannot see it. A clean illustration of why L0/L1 are screening-only.

### Caveats locked in
- L0/L1 measure *own-item* recoverability, not the *transition* task. No ranking authority until Spearman ρ(L0/L1, L3) is measured across engines×states. **Next step: run L2/L3 for these engines and compute ρ.**
- HGI absent outside FL → 5-engine comparison is FL-only.
- next-poi axis: meaningful only at check-in granularity (recoverability), not yet a forecasting metric.

### 2026-05-31 (later) — second adversarial audit + plots
- **Second audit verdict:** all 7 first-round fixes verified *correct* in code; zero blockers. Applied remaining items: kNN weight `(cos+1)/2` (kills the all-negative-row class-0 bias), canonical lru_cache key, documented per-metric normalization (README §8).
- **next-reg granularity — decisive (answers a standing question):** POI-pooled is the WRONG granularity for next-reg. It is a per-visit-contextual, **Check2HGI-only** task; HGI has no check-in-level representation, so the pooled HGI(0.074) > Check2HGI(0.051) result is an artifact comparing a lossless POI vector to a lossy POI-mean for a task HGI cannot run. **Retired as a substrate verdict** (README §7). next-reg cross-substrate verdicts come from L2/L3 only; the pooled axis is kept for next-cat (where pooling denoises a POI-constant label).
- **ρ-gate retired:** with ~5 engines (HGI@FL only) and a near-identical Check2HGI family, a Spearman ρ vs L3 has near-zero power + restricted range + state confound — it can never legitimately unlock "ranking." Reframed: **L3 is the sole ranking authority; L0/L1 are permanent screens/explainers** reporting concordant/discordant *calls* descriptively (README §6).
- **Plots:** `scripts/embedding_eval/plot.py` → bar charts (kNN/silhouette/sep_ratio/probe-acc) + PCA-2D scatters, stored in `docs/results/embedding_eval/*/plots/`. Visual: Check2HGI shows separated category clusters; HGI a mushy blob — matches the 0.98 vs 0.77 kNN gap.

### 2026-05-31 (later 2) — INPUT-ARTIFACT correction for next-reg (user-caught)
**Two errors fixed:**
1. **next-reg was probed on the wrong artifact.** next-cat consumes the final per-item embedding (correct in run.py); next-reg consumes the **region embedding** (`region_embeddings.parquet`, via `--task-b-input-type region`). The original run.py next-reg numbers labelled the *final* embedding by region — wrong input. Corrected in `scripts/embedding_eval/region_eval.py` (probes region embeddings; L0 adjacency-coherence + L1 1-step transition probe).
2. **"next-reg is Check2HGI-only / HGI can't run it" was false.** Only the `next_region` *label* builder is Check2HGI-only. Both engines produce region embeddings and both run next-reg STL (`scripts/p1_region_head_ablation.py --region-emb-source {hgi,check2hgi}`). **Real STL has HGI winning next-reg at all 5 states** (RESULTS_TABLE §0.3, FL 71.3 vs 69.2 Acc@10). So my pooled "HGI > C2HGI region" pointed the RIGHT direction; I wrongly dismissed it as a pooling artifact. §7 rewritten.

**Corrected region-embedding results (FL, region_eval.py):**
| engine | adj_coh@10 (L0) | transition acc@1 | acc@10 |
|---|---|---|---|
| **hgi** | **0.326** | 0.482 | 0.676 |
| check2hgi | 0.274 | 0.479 | 0.685 |
| check2hgi_design_b | 0.240 | 0.483 | 0.681 |
| check2hgi_resln | 0.282 | 0.480 | 0.687 |
| check2hgi_resln_design_b | 0.231 | 0.482 | 0.680 |

- **L0 adjacency-coherence reproduces HGI's region win** (0.326 vs 0.274) — HGI region embeddings are the most spatially coherent; design_b variants are the *least* (0.23–0.24), concordant with design_b's documented no-MTL-region-benefit. This is the cheap proxy correctly recovering the §0.3 ranking.
- **The 1-step transition probe is a near-tie** (acc@10 ~0.68 all engines; self-transition rate 0.495). Too crude — no 9-window, no log_T prior — to resolve the real ~2 pp gap. ⇒ region's substrate ranking stays an **L2/L3 verdict**; among L0/L1, only the *structural* (adjacency) metric carries signal for region.
- F2/F3 (run.py pooled/check-in region numbers) are **superseded** for next-reg — they measured the final embedding, not the region embedding. They remain valid only as a demonstration of pooling bias on the final embedding.

### 2026-05-31 (later 3) — HGI AL/AZ regenerated; region_eval across FL+AL+AZ
- **Regenerated HGI embeddings for AL+AZ** (`scripts/embedding_eval/regen_hgi.py`, canonical CONFIG lr 0.006/warmup 40/2000 ep/dim 64, CPU-only to avoid GPU-sweep contention). Both OK. Region partitions verified aligned with check2hgi (counts match AL 1109 / AZ 1547 / FL 4703; adj_coh ≫ random confirms positional indexing aligns). Resolves the AL/AZ HGI gap (only FL existed before; AL/AZ were also affected by the 2026-05-20 byte-identical bug).
- **region_eval across all 3 states** → `docs/results/embedding_eval/region_eval/summary.md`. **L0 adjacency-coherence reproduces HGI's region win at ALL 3 states** (HGI tops adj_coh: FL 0.326 / AL 0.375 / AZ 0.393, clearly above the check2hgi family), concordant with real STL §0.3 (HGI wins next-reg Acc@10 everywhere). The 1-step transition probe does NOT track it (mixed/near-tie) — too crude (self-transition 0.30–0.50, no 9-window/log_T). **Concordance call: adj_coh is a valid cheap SCREEN for region embeddings (3/3 states); the transition probe is not.** HGI's region edge is spatial-structural (hierarchical region graph) — exactly what adj_coh measures.

### 2026-05-31 (later 4) — L2/L3 sweep done (28/28 OK); concordance calls
Full numbers: `docs/results/embedding_eval/l2l3/summary.md`. Seed 42, FL/AL/AZ.

**L2 STL next-cat (next_gru, macro-F1):** HGI 0.343 (FL) ≪ Check2HGI family 0.65–0.67. Within family: design_b weakest (FL 0.646), resln/resln_design_b/canonical tied top (~0.67). Same ordering at AL/AZ.
**L2 capacity ladder (check2hgi):** next_gru BEATS next_single at all 3 states (FL 0.673>0.656, AL 0.470>0.384, AZ 0.470>0.429) — a heavier transformer head does *not* help next-cat.
**L3 MTL (family):** cat F1 check2hgi 0.703 ≥ resln/resln_design_b 0.701/0.702 > design_b 0.687; region top10(indist) all ~0.598–0.602 (within-family tie). HGI not in MTL (region label is check2hgi-only + needs check-in emb).

**Concordance calls (descriptive — L0/L1 proxy vs L2/L3 ground truth; NOT a ρ):**
- ✅✅ **HGI ≪ Check2HGI on next-cat** — proxy (kNN 0.77 vs 0.98; probe 0.68 vs 0.98) → L2 (0.34 vs 0.66). Strong.
- ✅ **design_b is the weakest family member on cat** — proxy (lowest sep_ratio/probe) → L2 & L3 (lowest cat F1). 
- ✅ **resln ≈ canonical on top for cat** — proxy → L2/L3. 
- ✅✅ **HGI wins next-reg** — region_eval **adj_coh** (HGI top at FL/AL/AZ) → real STL §0.3 (HGI wins all 5). The corrected-artifact structural proxy is concordant.
- ✅ **Category signal is "easy"/near-linear** — L1 linear-probe saturation (~0.98) → capacity ladder gru ≥ single (heavy head doesn't help). The ladder shape confirms the proxy's reading.
- ❌ **1-step transition probe (L1 on region emb) does NOT predict region ranking** — near-tie, missed HGI's win; only the structural L0 (adj_coh) tracked it.
- ❌ **Within-family region ordering is below proxy resolution** — adj_coh ranked design_b lowest, but MTL region top10 differs <0.5 pp (tie). Proxy can't rank near-identical substrates on region.

**Bottom line for the original question ("is STL-head eval the best way?"):** No single number suffices. The graded ladder works *when each level uses the task's real input artifact*. For **next-cat** the cheap L0/L1 (final embedding) is strongly predictive of L2/L3 and even diagnoses *why* (linear-easy → heavy heads don't help). For **next-reg** the cheap proxy only works as the *structural* metric (adjacency-coherence on region embeddings); the probe and the sequence ranking need L2/L3. L0/L1 = screen + explain; L3 = rank. The HGI-vs-Check2HGI split (HGI better region-structure, Check2HGI better check-in-category) is robust across L0→L3.

### 2026-06-01 — HGI AL/AZ L2 completed + re-screen candidates mined
- **HGI next-cat STL run at AL/AZ** (now that HGI embeddings exist there): macro-F1 HGI AL 0.259 / AZ 0.282 — far below the Check2HGI family (~0.47), completing the L2 next-cat table at all 3 states. Confirms HGI ≪ Check2HGI on category everywhere (concordant with L0/L1). Table refreshed in `docs/results/embedding_eval/l2l3/summary.md`.
- **Mined other studies for dropped improvements** → `CANDIDATES.md`. Headline: **v3c weight-decay 5e-2** (canonical_improvement T1.5) is an *embedding-trainer* WD that was absorbed by ResLN in the stack and **never isolated on the region-embedding axis** — top re-screen candidate. Others: T2.4 DropEdge, T4.3 POI side-features, T3.1/T3.3 encoder swaps (leak-sniff), T6.1 log_T-KD-λ, T3.4 Time2Vec. The point: several were judged on a single MTL metric / final-embedding only — the ladder can check whether a real signal on the *region axis* was missed before deciding they don't help a future MTL.

### 2026-06-01 (later) — L0/L1 converted to 5-fold CV; on-disk variants re-screened
- **Protocol fix (user-requested):** L0/L1 were multi-seed random splits (L1) / full-data point estimates (L0); now both use a **shared 5-fold StratifiedKFold** (KFold for sparse region labels), matching L2. kNN is now train→test (per-fold reference set, no LOO); silhouette/sep computed per held-out fold; probe trains 4 folds / evals 1. All metrics report **mean±SD over the 5 folds**. Code: `geometry.knn_predict`, `linear_probe.fit_probe`/`make_folds`/`cv_probe`, `run.py` shared-fold orchestration, `region_eval` (KFold transition probe + per-region-fold adj_coh). Plots get fold error bars; PCA scatters use full data (deterministic = "best execution").
- **Re-screened all on-disk variants** (added design_j, design_l, resln_design_j, lever4_canonical) on next-cat (run.py) + next-reg (region_eval), FL/AL/AZ, 5-fold. Results: `fl_poi/`, `smallstates_poi/`, `region_eval/summary.md`.
- **Verdict (5-fold, multi-variant):** next-cat — resln variants top, HGI ≪ family (unchanged, tighter SDs). next-reg — **HGI tops adj_coh at FL/AZ and is top-tier at AL**; design_j/resln_design_j approach HGI at AL only (state-specific, collapse at FL/AZ → not robust); **no dropped variant robustly beats HGI/canonical on region** — validates their falsification on the region axis with proper CV. lever4 weakest. Transition probe stays a near-tie.
- **Note:** run.py's old "reg" task (final embedding labelled by region) is retired from re-runs — next-reg is screened only via region_eval (region embeddings). The check-in-granularity demo (`fl_checkin/`) was not re-run (secondary caveat artifact).
- **Pending:** v3c (WD 5e-2) re-screen needs a safe non-clobbering embedding rebuild (regen script writes to the frozen `output/check2hgi/`).

### 2026-06-01 (later 2) — fold-isolation fix (user-caught confound)
- **Confound:** run.py's per-engine StratifiedKFold split on each engine's own row order. check2hgi & HGI share the placeid SET but not row ORDER → the same POI landed in different folds per engine → next-cat L0/L1 was NOT comparing identical held-out sets (worst at small N). region_eval was already isolated (shared pairs + index-based KFold).
- **Fix:** run.py now assigns each item's fold by its **placeid in canonical (sorted) order** → same POI → same fold for every engine (seed 42) → byte-identical held-out sets → isolates the substrate.
- **Effect:** AL/AZ next-cat orderings became consistent with FL. **resln family (resln_design_j ≥ resln ≥ resln_design_b) now leads next-cat at all 3 states** (~+0.3 pp probe over canonical, tight ±SD); design_b/j/l and lever4 below canonical; HGI far below everywhere. The earlier AL "divergence" was the fold confound (now gone) + genuine small-N variance (SD ~0.003 AL/AZ vs ~0.0005 FL) + AL≠AZ being different cities.
- Residual variance at AL/AZ is real (small N), not a protocol artifact.

### 2026-06-01 (later 3) — resln at L2 next-reg (user-requested)
Ran the real STL next-reg (`p1_region_head_ablation.py next_stan_flow --input-type region`, 5-fold, FL) for resln vs canonical vs hgi → `docs/results/embedding_eval/l2l3/nextreg_stl.md`. **resln ties canonical (Acc@10 0.7275 vs 0.7274)** — its small L0 adj_coh edge does NOT translate; **resln's value is the category axis, neutral on region**. HGI wins region (+0.9 pp Acc@10), concordant with adj_coh + §0.3/§0.5 (and the run reproduces §0.5 canonical numbers, validating the pipeline). Confirms: the proxy resolves the big HGI-vs-family region gap but not within-family. Dual-axis MTL gain would need to combine check-in-category strength (resln) with HGI-style region-graph structure — design_b/j injections didn't robustly deliver either.

### 2026-06-01 (later 4) — re-screen of dropped candidates (v3c + 4 others)
Rebuilt 7 variants via `rescreen_build.sh` (OUTPUT_DIR-scratch, frozen substrate md5-verified untouched; region partitions all aligned). Screened L0/L1 at AL+AZ vs a same-protocol `gcn_ctrl` (fresh GCN wd=0). Full table: `docs/results/embedding_eval/rescreen_cat/RESCREEN.md`.
- **v3c (WD 5e-2): FALSIFIED** — no gain over the clean control on cat OR region at AL+AZ (region adj_coh slightly below ctrl). Its prior "region benefit" was a baseline-mismatch artifact; with the control it vanishes. Answers the original v3c question: no.
- **T4.3 POI side-features: consistent small REGION gain** (adj_coh +0.05–0.07, probe@10 +0.9–1.6pp, both states; cat neutral). Originally falsified on a single-state *cat* cell — the region axis was never measured. → L2 confirmation launched.
- **T3.3 R-GCN: biggest REGION gain** (adj_coh +0.07–0.09, probe@10 +0.7–3pp) without the catastrophic cat-leak in the static probe → needs leak-audit + L2.
- DropEdge / T6.1 p2p: no gain. GATv2: hurts cat. ✗
- Confirmed a ~+0.5pp **fresh-vs-frozen offset** (frozen check2hgi > fresh gcn_ctrl) — validates the same-protocol control.
- **Methodology win:** the ladder on the correct artifact (region embeddings) + a same-protocol control surfaced a region-axis signal (sidefeat, rgcn) the single-MTL-metric eval missed, and killed v3c cleanly. L0/L1 = screen; L2/L3 = rank (sidefeat/rgcn L2 next-reg running; FL control+v3c building).

### Next steps
1. **L2/L3** for next-cat across all 5 engines (FL) — the legitimate cross-substrate cat verdict; and next-reg L2/L3 within the Check2HGI family (check-in-level) — the only valid reg verdict. Emit commands via `run.py --emit-l2l3`; pull metrics from `results/`.
2. Report L0/L1 vs L3 as concordant/discordant *calls* (descriptive), NOT a pooled ρ.
3. Optional: a faithful check-in-vs-check-in cross-engine axis once a check-in-level HGI baseline exists.
