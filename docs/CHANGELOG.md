# CHANGELOG — Check2HGI Study

> **Why this file exists.** During paper preparation we hit the multiple-sources-of-truth confusion trap several times: numbers diverging across PAPER_CLOSURE_RESULTS, FINAL_SURVEY, NORTH_STAR, RESULTS_TABLE, and intermediate handoffs. Two external Codex audits (commits `7a60e1c` → `ed90e8a` → `6de13ca` → `8a95e92` → `8444b31`) caught load-bearing errors that traced back to "which version of the table am I citing?". This CHANGELOG is the single chronological source of truth for *what changed when, why, and which numbers to trust now*.
>
> **Rules of use.**
> - When in doubt about a number, the canonical source is the **latest dated row** below pointing to `results/RESULTS_TABLE.md §0`.
> - Any document elsewhere in the repo that contradicts the latest row is **stale** and should either be archived (under `archive/`) or fixed in the same commit that introduces the contradiction.
> - Workflow / prompt artefacts go to `archive/` once the work they coordinated has landed; only canonical / current docs live at top level.
> - **Do not edit historic rows.** Add new dated rows; the historic record stays.

---

## Canonical sources of truth (current)

| What | Where | Last updated |
|---|---|---|
| **Canonical version registry (v11/v12 + v13/v14 opt-in bases + reproduction map)** | **`results/CANONICAL_VERSIONS.md`** | **2026-06-02 (v14 blessed, opt-in)** |
| Five-state architectural-Δ + cat-Δ Wilcoxon | `results/RESULTS_TABLE.md §0.1` (**v11 paper canon — no-KD/GCN**) | **v11, 2026-05-02** (FL upgraded to n=20; all five states paper-grade on §0.1) |
| Δm joint score (CH22 leak-free) | `results/RESULTS_TABLE.md §0.2` | v6 (leak-free) — unchanged in v7/v8/v9/v10 |
| Substrate axis (CH16 cat + CH15 reg reframing) | `results/RESULTS_TABLE.md §0.3` + `FINAL_SURVEY.md §2-§4` | v6 — unchanged in v7/v8/v9/v10 |
| Recipe selection (B9 vs H3-alt) | `results/RESULTS_TABLE.md §0.4` | **v9, 2026-05-02** (TX upgraded to n=20 multi-seed) |
| External baselines | `results/RESULTS_TABLE.md §0.5–§0.6` | unchanged |
| Champion config + recipe | `NORTH_STAR.md` | v10-aligned entry banner; historical derivation preserved below |
| Claim catalogue | `CLAIMS_AND_HYPOTHESES.md` (whitelist banner) | CH16 / CH18-cat / CH15 reframing / CH19 / CH22 are paper-facing safe |
| Article-side paper docs | `articles/[BRACIS]_Beyond_Cross_Task/` | v10-aligned 2026-05-02 (BRACIS submission) |

**Rule of single-source.** Paper tables are sourced from `RESULTS_TABLE.md §0` and only that. `PAPER_CLOSURE_RESULTS_2026-05-01.md` is background provenance — it has been moved to `archive/post_paper_closure_2026-05-01/` because numerous numbers there were superseded by later `RESULTS_TABLE` updates (e.g., FL Δ_reg simple mean-diff −7.28 vs paired Δ −7.99; AL/AZ STL cat means refreshed from single-seed to multi-seed; CA/TX §0.1 upgraded to n=20 in v10).

---

## Timeline of findings (most recent first)

### 2026-06-19 — Pre-freeze A40 session (`study/pre-freeze-a40`): all lane gates closed, §0 STOP lifted, recipe stays v16, baselines landed, compile+TF32 pinned

The `closing_data` **pre-freeze gates** ran on the A40. **Recipe UNCHANGED (still v16)** + new production code is **BYTE-IDENTICAL** (FL non-overlap MTL scan = **73.0116 / 73.5414** exact) → nothing in the frozen number-set moves.
- **Lane 1 (recipe gates):** G0.1 aligned-pairing → **ADVISORY NULL** (FL null; AL aligned *hurts* cat); loss-scale lever **EXCLUDED**. No v17. (`pre_freeze_gates/LANE1_G01_VERDICT.md`, `LANE1_LOSSSCALE_VERDICT.md`)
- **Lane 2 (overlap):** overlapping-windows **ADOPT supported** (AL cat +8.12 / FL cat +3.64; both heads positive, scale-saturated). Stride-1 leak re-audit **CLOSED CLEAN** — all 4 paths incl. (d) E2-chrono. (`LANE2_OVERLAP_VALIDATION.md`, `STRIDE1_LEAK_REAUDIT.md`)
- **Lane 3 (substrate):** CA + TX v14 built → **all 6 states hash-manifested** (`V14_HASH_MANIFEST.json`) → **§0 STOP-condition lifted**.
- **New production code (memory/OOM fixes, all byte-identical):** S1 streaming train-metric, S2 chunked val-metric (scored path), dataset-on-GPU auto-fit, `<U32` builder fix.
- **Baselines:** 7 INCLUDE externals (B1 CTLE / B2a faithful POI2Vec / B2b skipgram / B2c onehot64 / B3 HMT-GRN / B4 cascade / B5 Flashback) implemented + adversarially audited + cheap-fixed + enum-registered. **POI2Vec resolved:** unfaithful B2a renamed `GeoPOI2Vec→GeoTreeSkipGram` (kept honest) + **faithful AAAI'17 POI2Vec built** (`scripts/baselines/poi2vec_lib/`). SC baselines run via `train.py --engine`; `--only-fold k` added (default-inert). (`closing_data/BASELINES_IMPL_AUDIT.md`)
- **Speed levers (the one re-baseline knob):** controlled probe → dataset-on-GPU ~0%, TF32 ~0%, `torch.compile` ~15% (model is launch-bound). **`--compile --tf32` ADOPTED + pinned for the P3 board** — result-neutral (+0.05 pp, within noise). **torch stays 2.11** (NO-GO: `torch_cluster` cu128 wheel + topk re-baseline); workers skipped. (`pre_freeze_gates/SPEED_LEVERS.md`)
- **Numbers to trust:** unchanged — `RESULTS_TABLE.md §0` / `CANONICAL_VERSIONS.md` (v16 = champion G). The P3 board re-baselines absolutes by ~0.05 pp (compile pin) with an identical recipe. **Deferred to P3:** the n=20 board build, B1/B2b per-fold scoring via `--only-fold`, the B2a/GeoTreeSkipGram include/exclude decision, the reg-ceiling **Acc@10**-under-overlap verdict (T3 matched scorer — not matchable in the FL probe: STL-full vs MTL-indist).

### 2026-06-12 — `mtl_improvement` FINAL CLOSE (4th-pass review endorsed) — `FINAL_SYNTHESIS.md` published; `closing_data` study scaffolded

The closure drop (row below) was critically reviewed (4th pass) and **ENDORSED with one verdict-wording correction**: the X1 roll probe earns "numbers pairing-safe + the deployed model performs no per-sample cross-modal mixing" — NOT "mixing intrinsically dead" (the probe is circular against the aligned-training counterfactual; X3's β→0 and F52 P5 inherit the same conditioning). The **aligned-pairing training test** is inherited by `closing_data` as a **PRE-FREEZE gate** (it must run before the recipe freezes for the CA/TX majors). Paper wording: "the architecture wins *without* per-sample cross-modal mixing (pairing-invariance verified)".
- **The study's one-stop closure doc**: [`studies/archive/mtl_improvement/FINAL_SYNTHESIS.md`](studies/archive/mtl_improvement/FINAL_SYNTHESIS.md) — headline R0 table, the six findings (C25 / dual-tower / orthogonality / no-optimizer / cat decomposition / rising-tide rule), **the corrections-and-retractions registry (cite the RIGHT claims)**, process lessons (C25–C28), shipped code, inherited items.
- **`closing_data` scaffolded (NOT launched)**: [`studies/closing_data/`](studies/closing_data/) — phases P0 pre-freeze gates (G0.1 aligned-pairing) → P1 cross-study re-eval → P2 recipe FREEZE → P3 CA/TX majors (once) → P4 final tables. Launch pending user sign-off on `PLAN.md`.
- Registry updates: `studies/log.md` (mtl_improvement CLOSED + closing_data SCAFFOLDED rows), `studies/README.md` (both rows). Numbers to trust: unchanged — `RESULTS_TABLE.md §0` (v11 paper canon) + `R0_matched_metric_bar.json` (G matched-metric bar) + `CANONICAL_VERSIONS.md` (v16 = G = train.py MTL default).

### 2026-06-12 — `mtl_improvement` HANDOFF_AUDIT punch list CLOSED — study CLOSED; P0 data-integrity fix + X-series all NULL; champion G unchanged
Pre-closure deep code-audit (`CODE_AUDIT_2026-06-12.md`) findings executed on the A40 (`docs/studies/archive/mtl_improvement/{HANDOFF_AUDIT.md,X_SERIES_FINDINGS.md → docs/results/mtl_improvement/X_SERIES_FINDINGS.md}`, `log.md` 2026-06-12 third pass).
- **P0 (data integrity):** the cat-transfer manifest's FL `s1/s7/s100` rows had mis-pointed (via an `ls -dt|head` capture race) to the FL *fully-shared* intrinsic-test run, not a cat-transfer run — FL cat-transfer at {1,7,100} had never run. Re-ran genuine reg-OFF: FL cat+trunk **72.24 ± 0.03** (was 72.09); decomposition **architecture +2.13→+2.27, region-transfer +1.08→+0.93** (sign held, −0.15pp, now closer to the seed0 +0.89). One paper-bound number corrected; verdict (architecture-dominated) unchanged.
- **X-series — all NULL** (every MTL-only lever the audit surfaced + two claim stress-tests): X1 cross-attn mixing genuinely dead (roll Δcat −0.004); X2 aux-gate fixed + first *real* KD-on-G test null (FL reg +0.05 / AL −0.13, cat −0.57; the old "KD adds nothing" was a dead codepath); X3 β decays to ≈0 by gradient even with WD removed (not a WD artifact, metrics null); X4 the "matches" −0.31pp gap is fp32-precision-clean (Δ −0.005). The "matches reg, beats cat / can't beat reg" verdict is now earned at a strictly higher standard.
- **H1/H2:** cos≈0 figure widened from 2 → 16 G runs (4 states × 4 seeds, pooled +0.0008, n=3,797); `T4_corrected_rerun.json` committed.
- **Code (all env-gated, G defaults unchanged):** aux-gate fix (`folds.py`), β logging (`mtl_cv.py`), `MTL_DISABLE_AMP_EVAL`, `MTL_ROLL_TASKB_EVAL`, `MTL_BETA_NO_WD`. New concern **C28** (aux-gate dead codepath + rundir-race). **Champion G unchanged; CA/TX deferred to `closing-data`.** Numbers to trust: `RESULTS_TABLE.md §0` unchanged; cat-transfer decomposition → `cat_transfer_and_T53.md`.

### 2026-06-08 — `mtl_improvement` Tiers 3 + 4 + 5 CLOSED; gradient-orthogonality is the unifying mechanism; champion G unchanged

A long execution session ran the re-scoped Tiers 3 (reg-input pathway) and 4 (loss/optimization) + closed Tier 5. **No champion change — G stands.** Key outcomes:
- **Tier 3 CLOSED** (reg-input axis exhausted): **R0** pinned the matched-metric G−ceiling bar multi-state (G *matches* reg Δ −0.09…−0.31, *beats* cat +2.6…+4.1, all 4 states); **R1** (overlap/data-scale) + **R2** (HGI substrate routing) = clean **rising-tide nulls** — both reg-input levers *transfer* into G's private reg tower post-C25 (falsifying "data/substrate washes out in MTL" twice) but neither beats the moving ceiling. **R1b** corrected an R1 over-read (the overlap absorption is C25-unweighting, not the dual-tower). T3-richer not run (trigger didn't fire).
- **Tier 4 CLOSED** (loss/optimization — clean, bug-free, fairness-checked NEGATIVE): no balancer (full `src/losses` registry, per-method-tuned + arch-wired), no loss-scale-norm, and no alternative static weight Pareto-beats G's `static_weight cw=0.75`. Six convergent lines: RLW litmus, full registry screen, corrected re-run (gradnorm/nash retuned), static cw-sweep (0.75 on the Pareto front), **scale-norm FALSIFIED** (starves the high-card reg head), and **gradient cosine(cat,reg)≈0** (FL +0.0007/AL +0.0026 — the mechanism; matches Kurin/Xin NeurIPS'22). An audit found + fixed a latent preflight bug + documented a gradient-surgery×dual-tower wiring limitation (`CONCERNS §C27`). New gated `--loss-scale-norm` flag (default off, falsified).
- **Tier 5 CLOSED:** T5.1/T5.2 already done; **T5.3 HSM reg head FALSIFIED** (HSM=flat at FL 4.7k, 73.21 vs 73.22). Cat-transfer ablation: the +3pp MTL-cat gain is **architecture-dominated** (cross-attn trunk +2.3…+3.1pp); genuine region→cat transfer only **+0.89 FL / −0.71 AL** (refines CH30 — not "region teaches category").
- **Unifying mechanism (new):** the two tasks have **orthogonal gradients** — explains why balancers can't help (Tier 4), why more-sharing failed (Tier 2), and why the dual-tower wins (Tier 2/G). Conceptual write-up + 3 figures: [`studies/archive/mtl_improvement/WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`](studies/archive/mtl_improvement/WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md). Claims: `CLAIMS_AND_HYPOTHESES.md CH31`. Detail: `studies/archive/mtl_improvement/{log.md,INDEX.html}`, `results/mtl_improvement/{T4_audit_and_verdict.md, cat_transfer_and_T53.md, R0_matched_metric_bar.json, R1_overlap_under_g.md, R2_dual_substrate_routing.md}`.
- **Net:** the experimental track is at a comprehensive close; remaining work is completeness only (CA/TX) + the BRACIS paper-doc restatement.

### 2026-06-07 — `mtl_improvement` B-A3 / G′ (cat-private dual-tower) explored & CLOSED — FL-only dead-end; G stays champion

A both-private variant **G′** (`mtlnet_crossattn_dualtower_catpriv`, the cat head ALSO `next_stan_flow_dualtower`) looked like a Pareto cat win **at FL only** (cat 74.77, +1.61 vs G; reg flat). It was briefly **over-promoted to "new champion" in 5 claim docs (CHAMPION/NORTH_STAR/CANONICAL_VERSIONS/CLAIMS/HANDOFF) then REVERTED** when the multi-state confirm (AL/AZ/GE × {0,1,7,100}, `gprime_multistate.sh`) **FALSIFIED** it: cat **CRATERS** at small states — AL 37.66 (−15.25 vs G), AZ −12.45, GE −3.59; reg flat everywhere.
- **Mechanism CORRECTED** from "overfits" → **UNDERFITS** (advisor root-cause): AL G′ cat **train**-F1 caps at 0.45 vs the `next_gru` head's 0.98 (tiny train–val gap = textbook underfit). No wiring bug (α·log_T prior correctly OFF).
- **Rescue screen CLOSED with NO rescue** (`gprime_rescue_screen.sh`, 1-seed AL+FL, 6 levers: priv_dropout↓ / cat-lr↓ / smaller tower / combos): best AL lever still −14.5pp vs G; the FL gain survives **only** at the original `priv_dropout=0.3` (lowering it to 0.1 erases the gain, 74.74→73.17). Refined mechanism = the STAN flow/attention head is **architecturally mismatched for a 7-class target at small data** (head↔task-cardinality mismatch), not a tunable.
- **Verdict: G′ DEMOTED — FL-only experimental dead-end, NOT a champion. G (cat-SHARED `next_gru`) remains THE multi-state champion. The B-A3/G′ line is CLOSED.** Lesson logged as `CONCERNS §C26`. Trail: `studies/archive/mtl_improvement/{log.md 2026-06-07, CHAMPION.md §G′, INDEX.html #T2V-5}`; drivers `gprime_{multistate,rescue_screen}.sh`.

### 2026-06-07 — `mtl_improvement` Tier 2V: champion G VALIDATED against a skeptical critique (paper-safe)

The 2026-06-06 G result (below) was stress-tested via `CRITIQUE_TIER2_C25_2026-06-06` and **held on every axis** — so the Pareto-positive headline is now critique-hardened, not just a first result:
- **Seed-matched ceilings (the #1 gate):** the (c)/(d) STL ceilings were re-run at G's own seeds {0,1,7,100} (they were seed-42 only). The ceilings are stable (σ≤0.7) — the feared "lucky seed-42 ceiling" did not materialize; on the in-distribution metric G's margins held (AL +1.80reg/+2.56cat … FL +0.30/+3.20).
- **⚠ REG verb CORRECTED (B-A2 independent re-eval, 2026-06-07):** the reg "beat" compared G's *in-distribution* `top10_acc_indist` to the (c) ceiling's *full* `top10_acc` (the p1 ceiling harness has no indist/OOD split). On a MATCHED metric G is **~0.35pp BELOW** the (c) reg ceiling (FL: G-full 72.93 vs ceiling-full 73.31). **So the honest reg claim is "MATCHES the STL ceiling" (Pareto-non-inferior, within ~0.4pp), NOT "beats".** The independent re-eval also forecloses harness inflation (the −0.6 vs the indist number is the mechanical OOD penalty, ood_fraction 0.83%). The **cat +3pp beat is exact** (single metric). The Pareto-positive / inverted-tradeoff headline STANDS (matches reg + beats cat dissolves the −7..−17pp tension); only the reg verb tempers. Also closed: the STAN private tower is **load-bearing** (lighter GRU/TCN towers all lose 1.8–3.4pp); plain CE is the MTL cat optimum (logit-adjust/focal/CB all lose); static_weight confirmed (FAMO/uncertainty/CAGrad/Nash ≈ G).
- **Architecture falsification is now FAIR:** standalone hard-share/CrossStitch/MMoE/CGC were re-ranked post-C25, each at its own `category-weight` — **all lose to G by 1.6–2.1pp**, so "architecture-capacity is not the reg lever" is un-confounded and paper-safe (was previously confounded by the class-weighting bug + a single un-swept loss-weight).
- **No tail regression:** G's prior-OFF ≈ prior-ON on the macro/tail metric → prior-OFF is a free choice.
- **No hypertuning lever beats G:** logit-adjust HURTS the MTL cat (plain CE is the MTL cat optimum — the joint rising-tide regime ≠ STL), the private STAN is right-sized, FAMO ≈ G. The "½ params vs 2 models" claim was corrected (G = base_a +4.9%, one model).
- **Deferred:** CA/TX (T2V.9, scale-conditional completeness) → documented future-work (`future_works/mtl_improvement_catx_scale_conditional.md`); the 4-state result is already paper-grade.

**Numbers to trust:** `studies/archive/mtl_improvement/{INDEX.html #tier2v, log.md 2026-06-06/07, CRITIQUE_TIER2_C25_2026-06-06.md §7, CHAMPION.md}`. **G remains a STUDY champion, NOT yet paper §0 canon** — the BRACIS restatement is the lone open author item.

### 2026-06-06 — ⭐⭐⭐ `mtl_improvement` SUCCEEDED: champion "G" makes MTL Pareto-POSITIVE (single model matches reg + beats cat ceiling, 4 states) [reg "beats"→"matches" corrected 2026-06-07, see entry above]

**The headline result of the whole branch.** A single jointly-trained MTL model ("G") **beats BOTH single-task STL ceilings (next-region Acc@10 AND next-category macro-F1) at all 4 available states, 4-seed**: AL reg 64.47/cat 52.91; AZ 55.75/54.48; GE 59.37/61.43; FL 73.57/73.16 — every Δ positive vs the (c) ceilings; FL also ties the (d) 2-model composite while winning cat → composite strictly dominated. **The classic MTL tradeoff is INVERTED, not just dissolved.**

**This OVERTURNS the 2026-06-03 entry below ("NO MTL benefit").** That entry — and the whole "regime finding / architecture-negative / ship-the-composite" line — was an artifact of a **class-weighting confound (C25)**: the MTL reg head trained on class-WEIGHTED CE while the reported Acc@10 metric + STL ceiling are unweighted, depressing MTL reg ~10-14pp. Fixed (both heads unweighted, `default_mtl` default). On top of the fix, the champion architecture is the reg-private **dual-tower + `aux` fusion + α·log_T-prior-OFF** (`mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower`). Architecture *capacity* is NOT the lever — falsified 5 independent ways (MoE, SwiGLU, MulT, crossstitch→crossattn, more cross-attn blocks).

**Numbers to trust:** `studies/archive/mtl_improvement/CHAMPION.md` (config + reproduction), `CANONICAL_VERSIONS.md §v16` (pin), `CLAIMS_AND_HYPOTHESES.md` (CH30 + the C25 banner overturning CH25/CH28), `CONCERNS.md §C25` (the confound), `studies/archive/mtl_improvement/{log.md,HANDOFF.md,PAPER_UPDATE.md}` (chronology). **NOT yet on the paper §0 whitelist** — G is a study champion on the v14 substrate / unweighted recipe; the BRACIS §0 restatement (paper still v11) is the lone open author decision. CA/TX deferred (no v14 substrate).

### 2026-06-03 — v14 MTL multi-seed confirms NO MTL benefit (upgrades the 2-fold pilot) [⛔ SUPERSEDED 2026-06-06 — see entry above; this was the class-weighting confound]

Full paper-grade MTL eval of **v14** (`check2hgi_design_k_resln_mae_l0_1`) vs **matched canonical**
(`check2hgi`, frozen v11 GCN), FL/AL/AZ, seeds {0,1,7,100}, 5-fold, leak-free, KD off, corrected
geom_simple selector. **v14 ≈ matched canonical** (FL tie both tasks; AL/AZ mixed within noise) →
the v14 STL dual-axis gains do **not** survive MTL. This **upgrades the prior 2-fold seed42 FL
pilot** (`embedding_eval/FINAL_SYNTHESIS.md`) from "(pilot)" to **confirmed multi-seed** and is a
**confirmation** of the documented STL-only / regime finding — not a contradiction. Full doc +
3-basis tables + audit reconciliation: [`results/v14_mtl_vs_canonical.md`](results/v14_mtl_vs_canonical.md).
- FL geom_simple: v14 reg 61.21 vs canon 61.54 (Δ−0.33); cat 66.73 vs 66.77 (Δ−0.04).
- The v14 "closes ~69% of the HGI next-reg gap / cat ≫ HGI" headline is **STL**, a different regime;
  not tested by this MTL run. A separate STL verification sweep (v11/v14/HGI, FL) replicates it.
- Matched-canonical (this harness) is the valid Δ; frozen §0.1 differs by a documented harness offset.

### 2026-06-03 — MTL joint-selector fix (C21) PROMOTED TO CODE DEFAULT

The C21 selector fix (`joint_geom_simple`), validated 2026-05-24 but never actually wired as the
live default, is now the **default checkpoint selector**. Default = `joint_geom_simple = sqrt(cat_macroF1 ·
reg_Acc@10)` — geometric mean of each head's REPORTED headline metric (cat `f1`, reg `top10_acc_indist`),
**no majority normalization** (F1 ~0.7 and Acc@10 ~0.6 are already comparable [0,1] scales; the old
acc1-lift majority denominator would be cardinality-wrong for Acc@10).
- **Why:** the live default had silently remained the broken v11 `joint_score = 0.5*(cat_f1+reg_f1)`
  (the geom path was opt-in only, and `--save-task-best-snapshots` used the interim acc1-`geom_lift`).
  The broken selector discarded ~10.7 pp of reg Acc@10 capacity at FL (CONCERNS §C21); geom_simple
  recovers it (+5.62 pp deployable, ~95% of capacity).
- **Code:** new `ExperimentConfig.checkpoint_selector` + CLI `--checkpoint-selector
  {geom_simple,joint_f1_mean,geom_lift}` (default `geom_simple`). All three selection sites in
  `src/training/runners/mtl_cv.py` (gate, `model_task.log_val`, `MultiTaskBestTracker` joint slot) now
  use the selected scalar; min_best_epoch is honored. Files: `mtl_cv.py`, `configs/experiment.py`,
  `scripts/train.py`.
- **Reproducibility:** §0.1 (per-task diagnostic-best) is UNCHANGED. **v11 paper canon now requires
  `--checkpoint-selector joint_f1_mean` explicitly** — see [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md).
- **Empirical (post-hoc, seeds {0,1,7,100}, KD off):** under geom_simple the v14 MTL JOINT reg jumps to
  FL 61.2 / AL 50.1 / AZ 37.8 (≈ per-task diagnostic-best ceiling and ≈ matched canonical) — vs the broken
  selector's noisy FL 56.0±2.3. Confirms the fix recovers the deployable single-checkpoint capacity.

### 2026-06-02 — embedding_eval Part-1 CLOSED: v14 dual-axis champion base (opt-in)

The `embedding_eval` study (leak-aware L0→L3 substrate ladder) closed Part-1 (substrate) and blessed **v14 = `check2hgi_design_k_resln_mae_l0_1`** — ResLN+mae cat lever ⊕ Delaunay-POI-GCN (design_k) reg lever, an orthogonal stack — as the recommended STL / forward-MTL base (supersedes v13). **Opt-in, same posture as v13; the canonical `check2hgi` engine + v11/v12 paper-canon are untouched.** Full record: [`studies/archive/embedding_eval/FINAL_SYNTHESIS.md`](studies/archive/embedding_eval/FINAL_SYNTHESIS.md); version registry: [`results/CANONICAL_VERSIONS.md §v14`](results/CANONICAL_VERSIONS.md).
- **design_k (Delaunay) was wrongly discarded** by a prior AL/AZ-only study (K≡J); FL re-validation overturned it — the spatial axis is the one that moves L2-reg.
- **Leak-free multi-seed FL** (seeded `--per-fold-transition-dir`; p1's default log_T leaks ~+3pp): design_k reg +0.9-1.1pp over canonical, closes 54% (AL) / 78% (FL) of the canon→HGI gap; **HGI keeps a small significant edge** (−0.26pp FL). v14 (resln+mae) reg 0.7024 closes ~69% at next-cat 67.36 (≈ frozen-canon, ≫ HGI). resln=cat lever, mae=+0.4pp cat, Delaunay=reg lever — orthogonal.
- **STL-only — NO MTL benefit** from v14 OR dual-substrate routing (2-fold seed42 pilots) — reproduces the v13 regime finding; the MTL cross-attn regime is the binding constraint (Part-2 = regime work).
- Methodology corrections: always seed log_T; compare to FRESH canonical (not frozen v11); L0 ranks cat only (diagnostic for reg; adj_coh demoted; region-silhouette is the spatial diagnostic). v13/v14 mechanisms graduated into `Check2HGIModule` (`reg_poi_mode`).

### 2026-05-30 — v11 → v12 default flip (log_T-KD ON + ResLN encoder) + version registry

The two validated `substrate-protocol-cleanup` findings were settled into the code **defaults**, with the prior paper-canonical pinned first so nothing is lost. New version registry: [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md) (v11 = BRACIS paper canon, FROZEN; v12 = new default).

- **Code defaults flipped (reproduction-preserving):**
  - **log_T-KD default → W=0.2, τ=1.0**, in `scripts/train.py`, **scoped** to `--task mtl --task-set check2hgi_next_region` only (the only task-set whose reg head reads the per-fold log_T). The `ExperimentConfig` dataclass field stays 0.0 (task-agnostic); the v12 default is applied at the CLI layer. Category-only / non-region / non-MTL runs are untouched. Emits `log_T-KD default ON (W=0.2) — v12 default; pass --log-t-kd-weight 0.0 for v11 paper-canon`. **v11 reproduction: `--log-t-kd-weight 0.0`.**
  - **Check2HGI encoder default → `resln`** (ResidualLNEncoder), in `research/embeddings/check2hgi/check2hgi.py` + `scripts/canonical_improvement/regen_emb_t3.py`. Affects **FUTURE builds only**. The frozen v11 GCN substrate at `output/check2hgi/<state>/` was **NOT** rebuilt/overwritten. **v11 reproduction: rebuild with `--encoder gcn`, or use the existing frozen GCN substrate.**
- **Evidence + grade (honest):** log_T-KD is **paper-grade at AL/AZ** (n=20, +2.27/+4.91 pp, p=9.54e-07, leak-clean), **single-seed PILOT at FL/CA/TX**. ResLN is **STL-only** (+0.86–1.70 pp cat, 5/5 seeds) with **NO MTL benefit** — never implies an MTL improvement.
- **The investigation synthesis (regime bottleneck):** the cross-attn MTL joint-training regime washes out encoder/substrate gains on BOTH axes; only the log_T prior pathway moves MTL reg. Triply confirmed: design substrates null in MTL (0 % gap closed) + **HGI ceiling** (STL reg winner gives no MTL reg gain, Δ+0.51 p=0.41 NS) + STL-α=0 (~73 %) vs MTL-α=0 (~0.03 %) isolation cell. MTL reg bottleneck = architectural → `mtl_improvement`. New one-stop synthesis: [`findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md`](findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md); numbers in [`results/RESULTS_TABLE.md §0.9`](results/RESULTS_TABLE.md).
- **Tests:** full suite re-run; the v11/v12 scoped-default behaviour is covered by new `TestLogTKDCLIDefault` (5 cases) in `tests/test_substrate_protocol_cleanup_flags.py`; the dataclass default stays 0.0 (`test_config_field_default_zero` unchanged). No new failures introduced (6 pre-existing failures are in unrelated working-tree files: paths.py enum, research-variant metadata, hgi/sphere2vec equivalence — all reproduce with the v12 changes stashed).
- **Docs settled:** NORTH_STAR, RESULTS_TABLE (§0.1 v11-labeled, §0.8 v12-default, new §0.9 substrate-null/HGI-ceiling/ResLN), CLAIMS (CH26 v12 + new CH28 regime / CH29 ResLN), CONCERNS C15 (architectural-and-localized, not closed), CLAUDE.md, study CLOSURE + log.
- **Reproduction safety:** v11 paper §0.1 remains fully reproducible (pass `--log-t-kd-weight 0.0` + frozen GCN substrate). No GPU runs; substrate not rebuilt.

---

### 2026-05-29 — `substrate-protocol-cleanup` study CLOSED

The study launched 2026-05-28 (entry below) is now closed. Closure synthesis: [`studies/archive/substrate-protocol-cleanup/CLOSURE.md`](studies/archive/substrate-protocol-cleanup/CLOSURE.md). Five Tiers landed; **one promotion** (Tier A1), the rest nulls/closures. ~10–11 GPU-h total (well under the ~40–45 GPU-h budget). Small-states only (AL/AZ) for paper-grade; FL/CA/TX as pilots only.

- **Tier A1 — log_T-KD (W=0.2) → PROMOTED multi-seed n=20 at AL/AZ.** Disjoint reg Acc@10 +2.27 pp (AL) / +4.91 pp (AZ), seeds {0,1,7,100}, 20/20 folds positive, paired Wilcoxon p=9.54e-07 each. Cat untouched at disjoint (AL −0.20, AZ +0.08). Reproduces the Phase-3 single-seed=42 effect within 0.15 pp at both states (no dev-seed bias at small states). **Leak-audited clean** ([`findings/F_TIER_A1_LEAK_AUDIT.md`](findings/F_TIER_A1_LEAK_AUDIT.md), 7 vectors). **Large-state pilot TRANSFERS** (FL +2.40 5-fold p=0.031; CA +1.42, TX +1.71 1-fold) but is **seed=42, NOT paper-grade** (W=0.0 baselines overshoot §0.1 per C23). Cite as the isolated single-MTL-artefact reg lift — smaller than and distinct from the §4.2 composite headline (+7–12 pp). Implementation provenance: the `--log-t-kd-weight`/`--log-t-kd-tau` flags + KL term were implemented in this study (Phase-3 Rank-1 numbers came from uncommitted code); mechanism preserved verbatim. Source: [`results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md`](results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md) + [`tier_a1_largestate/phase_a1_largestate_addendum.md`](results/substrate_protocol_cleanup/tier_a1_largestate/phase_a1_largestate_addendum.md). New finding: [`findings/F_TIER_A1_PROMOTION.md`](findings/F_TIER_A1_PROMOTION.md).
- **Tier B — substrate axis CLOSED: no design promoted under MTL+F1.** All four variants NOT PROMOTED at AL/AZ: Design B (B1), Design J (B2), Lever 5 (B4), canonical+Lever 4 (B3). Reg flat (|Δ|≤0.38 pp disjoint, all p≥0.44) — and the NULL holds on BOTH fronts: the deployable joint/geom_simple selector agrees (|Δreg|≤1.22 pp, every p≥0.21; `tier_b/phase_b_two_front.md`). **[RE-AUDIT 2026-05-29 — framing softened from "doesn't transfer"; hedged per independent verification, `tier_b/phase_b_reaudit.md` + log.md]** The reg NULL is REAL but the mechanism is **anchor dominance**, not non-transferring substrate: under the joint config the MTL reg head is dominated by its α·log_T transition anchor — D1 (α frozen to 0 at AL, an OOD ablation) leaves reg at near-floor (all-epoch mean ~1.1% vs ~0.9% pure chance) for BOTH design and canonical, so the substrate-carrying encoder branch contributes ~nothing beyond the prior under MTL. The substrate's STL reg advantage is **real and reproduces** (AL +2.34 pp, Wilcoxon p=0.0312, WITH the prior; a separate FL no-prior ablation gives J−canonical +0.86 pp). The ~−2.4 pp cat is a **build-scope confound** (every design build re-trains a fresh-init CheckinEncoder, drifting the cat-path checkin vectors 100 % vs canonical; at α=0 cat Δ = +0.19 pp), NOT a substrate property. Accurate claim: Tier B measured "no reg gain BEYOND the canonical log_T anchor under the joint config." Avoid the absolute "encoder is inert / α=0 floors at chance" (chance ≈0.9% not 5.5%; the 5.5% was a best-epoch readout; α=0 is OOD). No leak signature. Source: [`results/substrate_protocol_cleanup/tier_b/phase_b_reaudit.md`](results/substrate_protocol_cleanup/tier_b/phase_b_reaudit.md) + [`phase_b_two_front.md`](results/substrate_protocol_cleanup/tier_b/phase_b_two_front.md) + [`phase_b1b2b4_verdict.md`](results/substrate_protocol_cleanup/tier_b/phase_b1b2b4_verdict.md) + [`phase_b3_verdict.md`](results/substrate_protocol_cleanup/tier_b/phase_b3_verdict.md).
- **Tier C — closes §4.4 + the P4 residual hole.** C2 (`--reg-freeze-at-epoch`, the last unfalsified curriculum form) → ARCHIVE, no N improves cat without reg regression > σ_fold. C3 (`--zero-cat-kv`) → P4 FULLY CLOSED: zeroing the cat→reg cross-attention K/V channel does not recover MTL reg or delay its peak. C1 (3-snapshot routing, variant A) → **§Discussion footnote (one-state pass)**: AZ clears +2 pp (Δreg +2.54, p=0.031, 5/5) but AL fails on one genuine degenerate Acc@1-selected reg-best snapshot (fold3 reg Acc@10 ≈ 0.1 % on the correct region modality vs 48 % joint-best) — not a promotion; conditional follow-up (Acc@10-aligned selector + degenerate guard) before any §0.x. An earlier C1 reg-modality scoring bug was found+fixed by advisor; the numbers above are the corrected re-score. Source: [`results/substrate_protocol_cleanup/tier_c/phase_c_verdict.md`](results/substrate_protocol_cleanup/tier_c/phase_c_verdict.md).
- **Tier D1 — window/causal-mask audit CLEAN** (no leak; shared artefact with `mtl_improvement` T0.2). Surfaced C24 (STAN bidirectional watch-item) + the Tier C4 mtime guard recommendation. Source: [`studies/archive/substrate-protocol-cleanup/window_mask_audit.md`](studies/archive/substrate-protocol-cleanup/window_mask_audit.md).
- **Tier C4 (added) — C22 stale-log_T mtime preflight guard** landed in `src/training/runners/mtl_cv.py` (refuses to start if log_T mtime < `next_region.parquet` mtime). Defensive, near-zero compute.
- **FL EXTENSION (2026-05-29, append-only) — THREE-WAY substrate comparison at Florida (B9), framed as gap-closure (canonical vs designs B/J/M/L vs HGI).** Source: [`results/substrate_protocol_cleanup/tier_b_fl/phase_b_fl_3way.md`](results/substrate_protocol_cleanup/tier_b_fl/phase_b_fl_3way.md) + [`tier_c_fl/phase_c_fl_verdict.md`](results/substrate_protocol_cleanup/tier_c_fl/phase_c_fl_verdict.md). **STL three-way (gethard):** canon→HGI gap +2.12 pp; only **J** Wilcoxon-strict over canonical (+1.12 pp p=0.0312, closes **53 %** of the gap); B/M +0.71/+0.89 pp ns; all designs strictly < HGI. **MTL three-way (B9):** designs give NO reg gain over canonical on either front (disjoint |Δ|≤0.16 pp ns) → designs close **0 %** of any MTL gap; cat −1.7 to −1.9 pp is the build-scope CheckinEncoder-reinit confound (AL D3). **ISOLATION cell:** STL `next_stan_flow` α=0 (log_T off) LEARNS FL region at **~73 % Acc@10** (canon 72.74 / design_b 73.12, Δ+0.37 p=0.0312) while the IDENTICAL head/config under MTL floors at ~0.03 % (chance 0.21 %) → **the joint-training REGIME kills the MTL reg encoder, not the head** (single-cell apples-to-apples STL↔MTL; hedged: α=0 is OOD → regime/config-scoped). **Corrects** the prior FL claim "STL→MTL collapse REPEATS at FL" (over-claimed a large STL advantage): accurate = the small FL STL design advantage does not survive MTL, and the MTL reg encoder is anchor/regime-limited. **C2 §4.4-closed + C3 P4-closed both HOLD at FL** (two-front). **Un-evaluated leg: HGI MTL+F1 at FL was NOT built** (expensive, not approved); HGI numbers are STL-only reference ceiling, MTL HGI gap unmeasured (irrelevant — designs don't move off canonical in MTL regardless). ⚠ A separate concurrent agent session was observed building HGI FL embeddings during this work (outside this task's scope, left untouched).

**Mechanism convergence (residual reg gap now TRIPLY confirmed architectural):** P4 cat-params + C3 cat-activations (K/V) + B substrate axis all exonerated. Combined with the earlier §4.6 sampler and Tier-6 substrate falsifications, the residual MTL-vs-STL reg gap is, by elimination, the **shared-backbone architecture** → `mtl_improvement` T2 owns the fix. log_T-KD (Tier A1) is an orthogonal small-state free upgrade that stacks onto whatever champion lands.

---

### 2026-05-28 — `substrate-protocol-cleanup` study LAUNCHED; future_works re-routed

New study at [`studies/archive/substrate-protocol-cleanup/`](studies/archive/substrate-protocol-cleanup/) (main worktree) owns substrate + protocol items orthogonal to backbone, splitting cleanly from the parallel active `mtl_improvement` study (architectural axis on branch `mtl-improve`).

**Scope (small states only — AL/AZ; FL/CA/TX only as 1-fold pilots):**
- **Tier A:** §4.5 log_T-KD multi-seed n=20 promotion (Phase 3 Rank 1 already PROMOTED at single-seed).
- **Tier B:** Designs B/J + Lever 4 + Lever 5 (orphan rescue) MTL re-eval under F1 selector.
- **Tier C:** §4.1 per-task 3-snapshot routing (variant A, internally coherent); §4.4 freeze-reg-after-peak (last unfalsified curriculum variant); P4 K/V capacity-stealing pilot.
- **Tier D:** window/causal-mask audit (no GPU; first-to-claim handoff with `mtl_improvement` T0.2).

**Decisions captured:** §4.1 variant A (3 internally-consistent MTL snapshots routed by task at deploy) over variant C (mixed-epoch heads + backbone — incoherent). Cost budget ~40-45 GPU-h + ~4-5 days code at small states. Advisor pass closed 5 critical gaps before launch (Lever 5 absorption, D1↔T0.2 handoff, BestTracker rebase cadence, variant C-prime as deferred trigger, P4 K/V capacity-stealing pilot).

**Future-works re-routing:** `mtl_architecture_revisit.md` execution moved into `mtl_improvement`; §4.1 moved into `substrate-protocol-cleanup` Tier C. `substrate_adaptive_mtl_balancing.md` priority lowered post-P4. `head_window_batch_audit.md` §A → `mtl_improvement` T7, §B → `substrate-protocol-cleanup` Tier D, §C sampler form FALSIFIED via Phase 3 Rank 2. `reg_head_architecture_sweep.md` log_T-KD PROMOTED banner. `composite_two_substrate_engine.md` AL/AZ DONE, FL/CA/TX held until champion lands. See [`future_works/README.md`](future_works/README.md) §"2026-05-28 re-routing" for the full table.

**Outcomes-only cross-study log** introduced at [`studies/log.md`](studies/log.md).

---

### 2026-05-24 — `mtl-protocol-fix` Phase 3 post-closure execution

Three deferred items from `DEFERRED_WORK.md` executed at single-seed=42 small states (FL where cheap):

- **§4.5 log_T as supervisory KD signal → PROMOTED.** Wilcoxon-strict (p=0.0312) at all 9 cells: +2.40/+5.06/+2.32 pp disjoint reg at AL/AZ/FL @ W=0.2; cat untouched. The head already consumes `log_T` as additive prior; the KD term forces the output distribution to also match `log_T` — a second pressure that accelerates prior-alignment and stabilises the deployable selector. Multi-seed n=20 promotion deferred to `substrate-protocol-cleanup` Tier A. Source: [`results/mtl_protocol_fix/phase3_rank1_findings.md`](results/mtl_protocol_fix/phase3_rank1_findings.md).
- **§4.6 class-balanced reg sampler → FALSIFIED.** `WeightedRandomSampler` regresses disjoint reg by −30.46/−18.49 pp at AL/AZ (p=1.0000); FL skipped. Closes the long-tail-undersampling hypothesis: the existing weighted-CE is enough; layering a sampler creates dual-prior conflict. Source: [`results/mtl_protocol_fix/phase3_rank2_findings.md`](results/mtl_protocol_fix/phase3_rank2_findings.md).
- **§4.2 composite (STL c2hgi-cat + STL HGI-reg, routed by task at deploy) → ESTABLISHED.** Reg lift vs MTL@disjoint: AL +11.04 / AZ +12.04 / CA +7.16 / TX +9.64 / FL +7.43 pp. **Current project headline on the reg axis.** Cat untouched (same MTL c2hgi cat checkpoint). Pure inference-side recipe, zero retraining. Source: [`results/mtl_protocol_fix/phase3_rank4_composite_analysis.md`](results/mtl_protocol_fix/phase3_rank4_composite_analysis.md). Memo: [`future_works/composite_two_substrate_engine.md`](future_works/composite_two_substrate_engine.md).

**Mechanism convergence (residual MTL-vs-STL reg gap is architectural):** three independent strands now converge — (1) Phase 2 P4 frozen-cat ⇒ cat is not the bottleneck; (2) Phase 3 Rank 2 sampler ⇒ long-tail is not the bottleneck; (3) canonical_improvement Tier 6 ⇒ substrate is not the bottleneck. By elimination, the residual is the **shared-backbone architecture itself**, motivating the parallel `mtl_improvement` study.

**Caveats and follow-ups** captured in [`results/mtl_protocol_fix/phase3_summary.md`](results/mtl_protocol_fix/phase3_summary.md) + the post-closure deferred inventory at [`studies/archive/mtl-protocol-fix/DEFERRED_WORK.md`](studies/archive/mtl-protocol-fix/DEFERRED_WORK.md).

---

### 2026-05-20 (close-of-day) — `mtl-protocol-fix` study CLOSED (v6 final verdict); C22 stale log_T + C23 dev-seed bug discoveries documented

The full one-day execution of the mtl_protocol_fix study closed with a v6 final verdict at [`docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md`](results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md).

**Headline findings:**

1. **C21 selector bug is REAL and the F1 fix recovers most capacity (paper-bearing).** At FL multi-seed (n=4 seeds × 5 folds, fresh log_T): MTL @ disjoint = 63.91 ± 0.16 (matches §0.1 v11's 63.27 ± 0.10), MTL @ `joint_geom_simple` (F1 fix) = 61.54 ± 4.54, MTL @ `joint_canonical_b9` (legacy production) = 55.92 ± 3.40. **Selector bug (geom − b9) = +5.62 pp at multi-seed.** The F1 fix recovers ~95% of substrate capacity at FL (capacity gap disjoint − geom = 2.37 pp).

2. **C22 — stale log_T bug DISCOVERED 2026-05-20.** `scripts/canonical_improvement/regen_emb_t3.py` regenerates embeddings + `next_region.parquet` but does NOT rebuild `region_transition_log_seed{S}_fold{N}.pt`. FL seed=42 log_T was 2 weeks stale (mtime 2026-05-06), surviving the canonical_improvement Tier 6 FL-MTL sweeps. Empirical impact: **STL Acc@10 inflated by +8.02 pp; MTL @ disjoint inflated by ~+12 pp; MTL @ geom_simple inflated by +11.74 pp.** Tier 5/6 RELATIVE falsifications still HOLD (baseline + variants used same stale log_T), but ABSOLUTE Acc@10 in canonical_improvement Tier 6 FL-MTL artefacts is biased by unknown sign-and-magnitude. Documented at [`docs/CONCERNS.md` C22](CONCERNS.md#c22). Code fixes: `regen_emb_t3.py` now auto-calls `compute_region_transition.py`; `scripts/train.py` preflight raises on stale log_T.

3. **C23 — development-seed bias at large states.** §0.1 v11 uses seeds {0, 1, 7, 100} explicitly excluding seed=42 (the development seed). At small states (AL/AZ) and FL (post-stale-log_T fix), seed=42 matches multi-seed within σ. At large states (CA: 8501 regions, TX: 6553 regions), single-seed numbers (both seed=42 and seed=0 from this study) overshoot §0.1's n=20 multi-seed by +3 pp (CA) / +7 pp (TX) — likely methodology delta (pooled mean vs per-seed mean) or recipe parameter difference. **§0.1 v11 remains the paper canon**; this study adds the F1-fix selector axis as a NEW column without contradicting v11. Documented at [`docs/CONCERNS.md` C23](CONCERNS.md#c23).

4. **Tier 5/6 §Discussion candidates re-eval under F1 selector: NO winners found.**
   - T6.2 a2.0_0.3 at geom_simple = 57.64 vs shipping 61.14 (Δ −3.50, FALSIFIED at deployable axis).
   - T5.3 multi-view at geom_simple = 62.08 vs shipping 61.14 (Δ +0.94, within σ — sub-Bonferroni, NOT a winner).
   - T5.2b masked POI (cat-side only, F1 fix is reg-axis only) skipped for parsimony.
   - **Substrate axis genuinely exhausted** (consistent with canonical_improvement Tier-6 closure).

5. **Mechanism diagnosis (P4 frozen-cat horizon test) FALSIFIES the cat-task-interference hypothesis.** With cat fully frozen + zero weight at CA, MTL reg STILL peaks at ep 2 and crashes by ep 11 (identical pattern to regular MTL). **The MTL backbone architecture itself caps reg learning, NOT cat-task interference.** STL reg head consumes region embeddings DIRECTLY; MTL reg head reads from the shared backbone. Different pathways → different ceilings. This redirects the next-tier study from loss-balancing → architecture revisit.

**Residual gap brief for next-tier study (highest-EV):**

| State | n_regions | MTL @ disjoint (fresh) | STL on shipping (fresh) | Δ (MTL − STL) |
|---|---:|---:|---:|---:|
| AL | 1,109 | 50.82 ± 3.21 | 62.10 ± 4.63 | −11.28 |
| AZ | 1,547 | 41.33 ± 2.73 | 53.60 ± 3.33 | −12.27 |
| FL (multi-seed n=20) | 4,703 | 63.91 ± 0.16 | 70.92 ± 0.10 | −7.01 |
| CA (seed=42 + seed=0) | 8,501 | 50.61 ± 1.23 | 57.19 ± 0.96 | −6.58 |
| TX (seed=42 + seed=0) | 6,553 | 50.83 ± 1.89 | 59.81 ± 0.36 | −8.98 |

**Next-tier study priority (per P4 mechanism finding):**
1. [`docs/future_works/mtl_architecture_revisit.md`](future_works/mtl_architecture_revisit.md) — **HIGHEST-EV**. Give MTL reg head direct region-embedding access (bypass shared backbone) or implement faithful MMoE/CGC/DSelect-K. The −7 to −12 pp residual is architectural.
2. [`docs/future_works/substrate_adaptive_mtl_balancing.md`](future_works/substrate_adaptive_mtl_balancing.md) — LOWER-EV. P4 horizon test already falsified the loss-balancing hypothesis.
3. [`docs/future_works/paper_canon_reevaluation.md`](future_works/paper_canon_reevaluation.md) — multi-seed CA + TX on shipping substrate to resolve the +3/+7 pp overshoot vs §0.1 v11 at CA/TX.

**Docs updated this entry:**
- `docs/CONCERNS.md` — new C22 (stale log_T) + C23 (dev-seed bias)
- `docs/studies/archive/mtl-protocol-fix/log.md` — Phase 2 P5+P6 closure + v6 final verdict entry
- `docs/studies/archive/canonical_improvement/log.md` — retroactive caveat re Tier 6 FL-MTL stale log_T
- `CLAUDE.md` (project root) — stale-log_T preflight + dev-seed convention warnings
- `scripts/canonical_improvement/regen_emb_t3.py` — auto-rebuild log_T after regen (C22 fix item 2)
- `scripts/train.py` (worktree branch) — preflight raises if log_T mtime < next_region.parquet mtime (C22 fix item 3)
- `docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md` — study closure verdict

**§0.1 v11 paper canon is UNCHANGED.** This study reproduces §0.1 v11's FL multi-seed disjoint within σ and adds the F1-fix selector axis (`joint_geom_simple`) as a new deployable-checkpoint column.

### 2026-05-20 — `mtl-protocol-fix` study launched; canonical_improvement post-closure pivot

After full deep-dive review of the closed canonical_improvement study (Tier 1-6, 26 mechanism families, ~40 GPU-h), the user-directed pivot is to **shift the next research track from the substrate axis to the protocol axis**. The canonical_improvement substrate exhaustion + the C21 finding (production `joint_canonical_b9` selector throws away ~10.7 pp of reg-top10 capacity at FL on the canonical shipping recipe alone) together indicate the next-reg gap is **70%+ protocol-side, not substrate-side**.

**New study launched**: [`docs/studies/archive/mtl-protocol-fix/`](studies/archive/mtl-protocol-fix/) (branch `mtl-protocol-fix`). Single-sentence goal: *"Fix the production `joint_canonical_b9` selector bug (C21), instrument the three-frontier MTL evaluation protocol (MTL @ best joint + MTL @ best disjoint + STL ceiling), and characterise the residual MTL-vs-STL reg gap that survives the protocol fix at all five states."*

**Five future-work memos created** to document deferred work the study deliberately scoped OUT:

- [`docs/future_works/paper_canon_reevaluation.md`](future_works/paper_canon_reevaluation.md) — §0.1 n=20 multi-seed re-evaluation under new selector + arch (sequenced AFTER `mtl_architecture_revisit.md`)
- [`docs/future_works/substrate_adaptive_mtl_balancing.md`](future_works/substrate_adaptive_mtl_balancing.md) — NashMTL revival, GradNorm, PCGrad, FAMO, Aligned-MTL, per-task LR decay
- [`docs/future_works/mtl_architecture_revisit.md`](future_works/mtl_architecture_revisit.md) — faithful MMoE / CGC / DSelect-K / cross-stitch / hybrid implementations, per-task evaluation
- [`docs/future_works/head_window_batch_audit.md`](future_works/head_window_batch_audit.md) — head re-design + window/mask audit + batch class-balance experiment
- [`docs/future_works/reg_head_architecture_sweep.md`](future_works/reg_head_architecture_sweep.md) — focused reg-head sweep (variant of head-audit §A)

**User-flagged conceptual clarifications captured in the new study's considerations**:
1. STL substrate-Δ on reg (HGI > c2hgi by 1.6-3.1 pp in §0.3) reflects HGI's POI-stable spatial inductive bias (Delaunay POI-POI graph) which c2hgi's per-visit context substrate does not encode. The cat substrate is c2hgi-load-bearing (per-visit context); the reg substrate is HGI-marginally-load-bearing (spatial adjacency). They are not contradictions but two different inductive biases.
2. **Three-frontier MTL evaluation** (best joint + best disjoint + STL ceiling) is the new paper-grade reporting protocol; replaces the single-selector reporting that obscured C21.
3. Coverage beyond substrate axis: protocol/selector (partially tested), MTL loss balancing (untested under leak-free), MTL architecture (very-simple-variants only), head architecture (grandfathered), window/mask discipline (never audited), batch class-balance (never tested).

**Docs updated**:
- `docs/future_works/README.md` — 5-row addition
- `docs/CONCERNS.md` C21 — closure path now points to `mtl-protocol-fix`
- `docs/NORTH_STAR.md` — selector-limitation banner updated
- `docs/studies/archive/canonical_improvement/log.md` — final post-closure entry references new study
- `docs/studies/mtl-exploration/README.md` — urgent banner updated to point to new study

### 2026-05-19 (final) — canonical_improvement Tier 6 CLOSED — full hypothesis falsified across all four pre-registered mechanism families

After the earlier 2026-05-19 entry below (T6.4 falsified + selector bug surfaced), the remaining Tier-6 candidates were run with locked pre-registered criteria. **Tier 6 closes operationally falsified across all four mechanism families.**

| ID | Mechanism | Per-task disjoint Δ_reg | Verdict |
|---|---|---:|---|
| T6.4 ×2 | Loss-shape reform (InfoNCE @ p2r, two-pass corruption) | +0.08 to +0.17 | FALSIFIED |
| T6.1 ×5 (4 orig + 1 robust) | POI↔POI co-visit InfoNCE 4th boundary | +0.05 to +0.20 | FALSIFIED |
| T6.2 ×4 | HGI-inspired composite edge weights (Delaunay + cross-region penalty) | +0.23 to +0.76 | §Discussion (cat -1.3 to -3.6 pp Pareto trade) |
| T6.3 ×2 (AL/AZ stage-1 only) | Low-rank per-POI bias at Checkin2POI attention-logit | halted | FALSIFIED at G3 hi/lo ratio gate (AZ r=8: 3.58× → 3.09×) |

**Per-task-disjoint reg-top10 ceiling at FL clusters at ~76.1-76.9 pp across all 11 cells with σ ~0.3 pp.** No Tier-6 intervention delivers a deployable single-checkpoint improvement under `joint_geom_simple`. The cleanest paper-claim phrasing: *"Tier 6 falsifies the POI-internal-supervision hypothesis under matched protocol at FL across all four pre-registered mechanism families. No cell delivers a deployable single-checkpoint improvement; per-task-disjoint reg-capacity stays bounded at ~76.1 pp under canonical static_weight w_cat=0.75 MTL balancing."*

**The load-bearing Tier-6 finding is NOT a substrate result.** It is `CONCERNS.md` C21 / `CLAIMS_AND_HYPOTHESES.md` CH23-B: the production `joint_canonical_b9` selector throws away ~10.7 pp of reg-top10 capacity from the canonical Check2HGI substrate itself. Substrate-axis effect: ±0.8 pp. Protocol-axis effect: +10.7 pp. The natural next study is the mtl-exploration F1/F2/F3 workstream.

**Pre-registration discipline held.** Two advisor consults (2026-05-19, both general-purpose independent agents) locked criteria before the T6.1/T6.2/T6.3 sweeps; in all three cases the locked criteria fired correctly and the closure narrative did not drift post-hoc. Specifically: (1) the T6.1 "+11-13 pp reg lift" claim from the morning of 2026-05-19 was a cross-selector comparison artefact corrected after matched-protocol analysis; (2) the T6.2 +0.76 pp per-task-disjoint reg lift was kept as §Discussion-only per advisor warning ("don't let it grow into a substrate claim"); (3) the T6.3 AL/AZ-first kill-check fired automatically on AZ r=8 hi/lo ratio compression, halting before FL stage 2.

**Artefacts**:
- T6.1: `docs/results/canonical_improvement/T6_1_lambda{0_05,0_1,0_2,0_3}/florida_mtl/` + `T6_1_dual_selector.{json,md}` + `T6_1_robustness_lambda0_2/florida_mtl/` + `T6_1_robustness_dual_selector.{json,md}`
- T6.2: `docs/results/canonical_improvement/T6_2_a{1_5_0_3,1_5_0_5,2_0_0_3,2_0_0_5}/florida_mtl/` + `T6_2_dual_selector.{json,md}`
- T6.3: `docs/results/canonical_improvement/T6_3_r{4,8}/{alabama,arizona}/` + `G3_{alabama,arizona}_T6_3_r{4,8}.json` (no FL — gate aborted)
- Code: `research/embeddings/check2hgi/model/Checkin2POI.py` (T6.3 attention-logit bias), `research/embeddings/check2hgi/model/Check2HGIModule.py` (T6.1/T6.4), `research/embeddings/check2hgi/preprocess.py` (T6.1 covisit_pairs, T6.2 composite C3), `scripts/canonical_improvement/t6{1,2,3}_sweep.sh` (per-mechanism sweep scripts)
- Closure log: `docs/studies/archive/canonical_improvement/log.md` 2026-05-19 final entry (the authoritative one — supersedes the earlier same-day entries in framing while keeping their numerical results)

**Doc-correction sweep** (all updated 2026-05-19):
- `docs/studies/archive/canonical_improvement/INDEX.html` Tier 6 — closure callout box + T6.2/T6.3 results blocks (rewrite, not append)
- `docs/CONCERNS.md` C21 — unchanged from earlier 2026-05-19 update (still load-bearing, scope confirmed)
- `docs/CLAIMS_AND_HYPOTHESES.md` CH23-A/B — extended to include T6.1, T6.2, T6.3 falsifications

**Closure path: mtl-exploration F1/F2/F3.** All canonical_improvement substrate work formally closed; future substrate interventions pre-route through C21 reading.

---

### 2026-05-19 — canonical_improvement Tier-6 / T6.4 FALSIFIED at matched protocol; `joint_canonical_b9` selector bug surfaces as the real finding

**Tier 6 was reopened 2026-05-18** to re-attempt the POI-level supervision hypothesis the user felt was under-explored in Tier 5. Built G3 (per-POI hold-out leak probe, calibrated against T5.1's known leak Δ_low = +3.82 pp), implemented **T6.4** (InfoNCE @ p2r + two-pass corruption) as opt-in default-off code paths in `Check2HGIModule.py`, swept the variants × {AL, AZ, FL} at ep=500, ran FL MTL under canonical B9 ep=50.

An initial ep=15 protocol-cap attempt was attacked by advisor consult #1 as post-hoc val-leak; advisor consult #2 supported a full-ep=50 dual-selector framing instead. A shipping FL ep=50 single-seed=42 n=5 baseline was added for matched-protocol comparison. The matched-protocol comparison then **falsified the Tier-6 substrate hypothesis** and surfaced a separate, more important finding about the production B9 selector itself.

**Matched-protocol dual-selector results (FL, single-seed=42, n=5 folds, ep=50):**

| Selector | shipping | T6.4 two_pass | T6.4 infonce τ=0.5 | Δ T6.4 vs shipping |
|---|---:|---:|---:|---|
| Per-task disjoint best: cat F1 | 70.49 ± 0.86 | 70.55 ± 0.85 | 70.49 ± 0.95 | **+0.00 to +0.06** |
| Per-task disjoint best: reg top10 | **76.12 ± 0.33** | 76.20 ± 0.27 | 76.29 ± 0.29 | **+0.08 to +0.17** |
| `joint_geom_simple`: cat F1 | 67.93 ± 1.74 | 67.33 ± 2.06 | 67.12 ± 2.45 | −0.60 to −0.81 |
| `joint_geom_simple`: reg top10 | 72.38 ± 2.20 | 73.33 ± 2.28 | 73.48 ± 2.48 | +0.95 to +1.10 |
| `joint_canonical_b9` (production): cat F1 | 69.99 ± 1.13 | 70.13 ± 1.06 | 70.28 ± 0.82 | +0.14 to +0.29 |
| `joint_canonical_b9` (production): reg top10 | 65.38 ± **9.10** | 61.19 ± **11.86** | 56.78 ± **11.79** | **−4.19 to −8.60** |

Reference: shipping FL §0.1 multi-seed n=20 reports reg top10 = 63.27 ± 0.10 — matches the matched-protocol `joint_canonical_b9` single-seed value (65.38 ± 9.10) within single-seed variance. §0.1 reports joint-best, not reg-best.

**Finding 1 — Tier-6 T6.4 substrate hypothesis FALSIFIED at matched protocol.** T6.4 variants add Δ_reg = +0.08-0.17 pp over shipping at per-task disjoint best — well within fold σ (~0.3) and not statistically meaningful at n=5. The InfoNCE-and-two-pass code paths land as opt-in infrastructure (default-off, byte-identical, useful for future studies that pair them with other interventions), but the variants alone are §Discussion-only and the paper claim for T6.4 is "falsified at matched protocol." The original "+11 pp reg lift" claim from 2026-05-19 was a cross-selector comparison artefact (T6.4 reg-best ep vs shipping §0.1 joint-best ep) — not a substrate effect. See `CLAIMS_AND_HYPOTHESES.md` CH23-A.

**Finding 2 — `joint_canonical_b9` selector throws away ~+11 pp of reg-top10 capacity from the canonical Check2HGI substrate itself.** Per-task disjoint best on shipping reaches reg top10 = 76.12; production selector reaches 65.38. Gap = ~10.7 pp on the shipping substrate, with no substrate change. The bug is **not Tier-6-specific** — it exists in the production B9 recipe AS-IS. Root cause: `reg_macro_f1` over ~4 700 sparse FL regions is dominated by rare-class noise (stays ~16-18 % across full ep=1-50 trajectory) and is blind to reg_top10's collapse from ~76 % at ep ~5 to ~65 % at ep ~30. The mean-of-F1s formula is scale-incoherent when one head has 7 well-supported classes (cat_macro_f1 ≈ 0.70) and the other has 4 700 sparse classes (reg_macro_f1 ≈ 0.17). See `CLAIMS_AND_HYPOTHESES.md` CH23-B and `CONCERNS.md` C21.

**Locked decisions (2026-05-19):**
- T6.4 does not promote to multi-seed or to AL/AZ MTL evaluation. Falsified.
- AL G3 gate violation (T6.4 low-visit Δ +1.05-1.41 pp vs +1 pp budget) is moot since T6.4 has no path to shipping under any selector.
- §0.1 reg numbers are reported under a known-broken selector. Until the F1 fix (below) is applied to the shipping baseline, **reg-side conclusions drawn from §0.1 multi-seed numbers under-report the substrate's reg capacity by ~10 pp**. The current paper canon stands as-is (it's internally consistent) but any future MTL paper should pair the §0.1-style numbers with the F1-fix numbers.
- All canonical_improvement Tier 1-6 candidate runs on disk can be re-analysed under any new selector **without retraining** via `scripts/canonical_improvement/analyze_t64_selectors.py` (reads per-epoch val CSVs).

**mtl-exploration F1 fix is URGENT — for shipping itself, not just for substrate variants.** Workstreams (`docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`):
- **F1** — substrate-aware joint_score (`reg_top10_acc_indist` instead of `reg_macro_f1`, or wire in the already-coded `joint_geom_lift` at `mtl_cv.py:710`). One-line code change. Re-evaluate **shipping AND all Tier 1-6 candidates** under the new selector without retraining; expose the ~11 pp reg-top10 capacity that the production selector currently hides.
- **F2** — substrate-adaptive MTL balancing (NashMTL revival on FL where the cvxpy solver is well-conditioned; per-task LR decay after reg peak; gradient masking after reg plateau). Goal: prevent reg destabilisation past its early peak so a single checkpoint near ep ~10-15 captures both heads near peak with low σ.
- **F3** — substrate × protocol 2×2 ablation as paper headline: (shipping, T6.4 substrate) × (B9 selector, F1-fix selector). Likely outcome based on this study: the protocol-axis effect dominates the substrate-axis effect on reg.

**Cross-references updated.** `CONCERNS.md` C21 (rewritten — not T6.4-specific; the bug is in shipping); `CLAIMS_AND_HYPOTHESES.md` CH23-A/CH23-B (locked claims with falsified-and-corrected framing); `AGENT_CONTEXT.md` blocker callout (rewritten); `NORTH_STAR.md` (B9 selector limitation warning rewritten to flag the bug as applying to the shipping recipe itself); `docs/studies/archive/canonical_improvement/log.md` 2026-05-19 entry; `docs/studies/archive/canonical_improvement/INDEX.html` T6.4 Results; `docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md` (rewritten with matched-protocol numbers); `docs/studies/mtl-exploration/README.md` URGENT banner (rewritten).

**Artefacts.** `scripts/canonical_improvement/analyze_t64_selectors.py` (dual-selector tool, reads per-fold val CSVs); `docs/results/canonical_improvement/T6_4_dual_selector_final.{json,md}` (all 3 arms at matched single-seed=42 n=5 ep=50 — replaces the earlier `T6_4_dual_selector_preliminary.{json,md}` whose numbers were §0.1-vs-single-seed cross-selector comparisons).

---

### 2026-05-18 follow-up — canonical_improvement Tier-5 Phase-3 closed (no shipping change)

**Tier-5 Phase-3 closed; canonical+v3c+T3.2 remains the shipping stack.** After the 2026-05-18 first-pass Tier-5 close (`docs/results/canonical_improvement/STACKING_ABLATION.md §7.1-§7.5`), two further multi-seed cells landed in Phase 3:
- **T5.2b multi-seed extended to FL** (5 seeds × FL; `T5_2b_maePoi_FL_seed{42,0,1,7,100}.json`). 4/5 paired-positive on FL cat (mean +0.234 pp); FL reg flat at −0.069 pp. Closes 3-state coverage. **3-state cross-state cat sign-test 13/15 paired-positive, p = 0.0074** — strongest single piece of Tier-5 evidence.
- **T5.3 multi-seed ran** (AL+AZ × 5 seeds; `T5_3_multiview_{alabama,arizona}_seed42.json` + `T5_3_multiview_alaz_seed{0,1,7,100}.json`). §7.1 had T5.3 marked SKIPPED → §Future Work; Phase 3 un-skipped it. All four (AL+AZ × cat+reg) cells mean-positive; AZ reg Cohen d ≈ +0.85 (strongest Tier-5 effect size), p_one = 0.065 — sub-Bonferroni at m=28.

**Multiple-testing posture (Phase-3 update):** family count tightens from m = 26 (§7.3) to **m = 28** (Tier 1–4 + Phase 1 Hyp A/B/C/D + Tier 5 T5.1/T5.2a + T5.2b 3-state + T5.3 AL+AZ multi-seed). Bonferroni α* = 0.05/28 ≈ 0.00179. **No Tier-5 cell clears it.** T5.2b pooled cat sign-test (p=0.0074) misses by ~4× — closest to threshold.

**Shipping stack unchanged:** `canonical Check2HGI + v3c (AdamW WD=5e-2) + T3.2 ResLN encoder`. §5 paper headlines stand. Tier 5 closes as §Discussion-only in the BRACIS draft (Beats 5/6/7/8 in `PAPER_DRAFT.md §7`).

**Artefacts.** `docs/results/canonical_improvement/STACKING_ABLATION.md §7.6` (Phase-3 closeout); `docs/studies/archive/canonical_improvement/log.md` (2026-05-18 follow-up entry); `docs/studies/archive/canonical_improvement/INDEX.html` (T5.x pills updated; Phase-3 callout); `docs/findings/F62_T5_2b_implementation.md` (FL multi-seed section); `docs/findings/F63_T5_3_implementation.md` (multi-seed results replace SKIPPED placeholder); `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md §7` (Beats 5/6/7/8); `articles/[BRACIS]_Beyond_Cross_Task/AUDIT_LOG.md §7` (record of Beats 5/6/7 → 5/6/7/8 replacement).

---

### 2026-05-02 — RESULTS_TABLE v11 (FL §0.1 arch-Δ upgraded to n=20 — all five states paper-grade)

**FL §0.1 architectural-Δ row upgraded from n=5 (single seed=42) to n=20 (seeds {0,1,7,100} × 5 folds).**
- Δ_reg = −7.34 pp, p = 1.9e-06, 0/20 fold-pairs positive (sign-consistent negative).
- Δ_cat = +1.40 pp, p = 2e-06, 20/20 fold-pairs positive.
- MTL B9 cat F1 = 68.56 ± 0.79 % (matches seed=42 reference 68.51 %); reg Acc@10 = 63.27 ± 0.10 %.
- **The last remaining headline asymmetry is closed.** All five states (AL/AZ/CA/TX/FL) are now n=20 multi-seed on §0.1 with paper-grade significance on the cat axis (AL small-significantly negative; AZ/CA/TX/FL paper-grade positive) and all five paper-grade significant on reg.
- Recipe used the canonical B9 invocation (matches `scripts/run_f51_multiseed_fl.sh` and CA/TX `run_h100_camera_ready_gaps.sh`): `--cat-head next_gru --reg-head next_getnext_hard --task-a-input-type checkin --task-b-input-type region --category-weight 0.75 --alternating-optimizer-step --scheduler cosine --max-lr 3e-3 --alpha-no-weight-decay`.

**Artefacts.** `research/FL_CAT_DELTA_WILCOXON.json` (new); `scripts/analysis/fl_cat_delta_wilcoxon.py`; `scripts/run_h100_fl_mtl_b9_multiseed.sh` (4-way H100 launcher with canonical recipe).

---

### 2026-05-02 — RESULTS_TABLE v10 (CA+TX §0.1 arch-Δ upgraded to n=20)

**CA and TX §0.1 architectural-Δ rows upgraded from n=5 (single seed=0) to n=20 (seeds {0,1,7,100} × 5 folds).**
- CA: Δ_reg = −9.50 pp p=2e-06; Δ_cat = +1.68 pp p=2e-06. Both axes paper-grade significant.
- TX: Δ_reg = −16.59 pp p=2e-06; Δ_cat = +1.89 pp p=2e-06. Both axes paper-grade significant.
- **All five states now have at minimum n=20 arch-Δ evidence** (AL/AZ n=20, FL n=5 ceiling, CA/TX n=20).
- Classic MTL tradeoff confirmed paper-grade at all large-scale states: reg trails STL by 7–17 pp; cat leads STL by 1.2–1.9 pp.

**Artefacts.** `research/ARCH_DELTA_WILCOXON.json` (new); `scripts/analysis/arch_delta_wilcoxon.py` (new); `scripts/run_h100_arch_delta_stl_ca_tx.sh` (16-run launcher).

---

### 2026-05-02 — RESULTS_TABLE v9 (TX recipe multi-seed landed; commit `928bdad`)

**TX B9 vs H3-alt upgraded from n=5 (single-seed=42) to n=20 (seeds {0,1,7,100} × 5 folds).**
- Δ_reg = +1.87 pp, p = 7e-04. Δ_cat = +0.52 pp, p = 2e-04. Both axes paper-grade significant.
- TX joins FL and CA as a large-scale state where B9 is paper-grade superior to H3-alt.
- **Recipe-selection narrative strengthened:** B9 is paper-grade at FL/CA/TX (all three large-scale states, n=20); H3-alt remains better at small scale (AL/AZ). Scale-conditional claim is now symmetric across all five states.

**Camera-ready audit item now fully closed.** The last remaining single-seed n=5 entry in §0.4 is resolved.

**Artefacts.** `research/GAP_FILL_WILCOXON.json` (TX section added); `scripts/analysis/gap_fill_wilcoxon.py` (Analysis C block).

---

### 2026-05-01 — RESULTS_TABLE v8 (Gap 1 + Gap 2 Wilcoxon landed; commit `bd707e8`)

**A) Cat-Δ Wilcoxon at AL/AZ/FL against multi-seed STL ceiling** (was "pending re-run" for 2 weeks).
- AL: Δ_cat = −0.78 pp, paired Wilcoxon p = 0.036 (n = 20 multi-seed; 14/20 fold-pairs negative). Statistically small-significantly negative. Magnitude small (~1.9 % relative on 41 % F1 scale).
- AZ: Δ_cat = +1.20 pp, p < 1e-04 (n = 20; 18/20 positive). Paper-grade.
- FL: Δ_cat = +1.52 pp (refined from +1.43 mean-diff to paired Δ), p = 0.0625 (n = 5 ceiling; 5/5 folds positive at seed = 42). Sign-consistent positive at single-seed ceiling.

**B) FL MTL B9 cat F1 refined 68.59 → 68.51 ± 0.51** (multi-seed pooled).

**C) CA recipe-selection (B9 vs H3-alt) upgraded to n = 20 multi-seed.**
- Δ_reg = +4.18 pp, p < 1e-04. Δ_cat = +0.51 pp, p < 1e-04. Paper-grade significant on both tasks.
- TX still pending multi-seed (single-seed n = 5 ceiling at submission).

**Lessons.**
- The "pending re-run" framing in the paper draft was a real BRACIS-rigour gap; running `gap_fill_wilcoxon.py` against the existing per-fold JSONs took ~5 minutes and resolved it.
- AL's "≈ tied" framing was generous given the formal stat (p = 0.036 in the negative direction). Honest framing now reports both significance and magnitude.

**Artefacts.** `research/GAP_FILL_WILCOXON.json`; `scripts/analysis/gap_fill_wilcoxon.py`.

---

### 2026-05-01 — Paper closure (Phases 1-3; commit `03af55c`)

**Five-state cross-state P3 + multi-seed at AL/AZ/FL + recipe ablation.**
- 28 paper-grade runs (5f × 50ep, leak-free per-fold log_T).
- AL/AZ B9 multi-seed at {0, 1, 7, 100}; FL B9 multi-seed at {42, 0, 1, 7, 100}; CA/TX seed = 42 single-seed.
- STL ceilings landed at all 5 states with multi-seed at AL/AZ/FL.

**Architectural-Δ picture (the classic MTL tradeoff, sign-consistent across 5 states):**
- Reg: MTL B9 < STL `next_stan_flow` at every state by 7–17 pp.
- Cat: MTL B9 ≥ STL `next_gru` at every state by 0 to +2 pp (refined to four-of-five-states-positive in v8 once cat-Δ Wilcoxon landed at AL).

**Reframe vs F49 (the leak that misled us).** F49's "AL +6.48 pp MTL > STL on reg" headline was a leak artefact of pre-F50 measurements (full-data `region_transition_log.pt` leaks ~13–27 pp). Under leak-free symmetric comparison, AL's reg pattern matches every other state. The headline "scale-conditional architecture-dominant state" framing from F49 is **superseded**.

**Lessons.**
- Leak detection (F44, F50 T4) caused a paper-reshaping reframe twice. Always run leak-free comparisons before declaring a champion.
- The B9 vs H3-alt recipe split is **scale-conditional**: B9 is FL-tuned; H3-alt is small-state universal. No single recipe wins on both axes across all states.

**Artefacts.** `PAPER_CLOSURE_RESULTS_2026-05-01.md` (now in `archive/post_paper_closure_2026-05-01/`); `research/PAPER_CLOSURE_WILCOXON.json`; `research/PAPER_CLOSURE_RECIPE_WILCOXON.json`.

---

### 2026-04-30 — F51 multi-seed validation (commit `f87321f`)

**B9 vs H3-alt across 5 seeds.** Pooled paired Wilcoxon (5 × 5 = 25 fold-pairs): Δ_reg = +3.48 ± 0.12 pp; p_reg = 2.98 × 10⁻⁸ (25/25 positive); p_cat = 1.33 × 10⁻⁵ (19/25 positive). Recipe is essentially deterministic across seeds (σ_across_seeds = 0.11 pp).

**F51 Tier 2 capacity sweep.** 21 capacity smokes confirm B9 is locally optimal in 5/7 capacity dimensions. Cat width-stability cliff at `shared_layer_size 384/512`. F52's "mixing is dead at FL" is depth-conditional.

**Per-seed log_T leak (caught + fixed mid-sweep 2026-04-30).** The original C4 fix wrote per-fold log_T as `region_transition_log_fold{N}.pt` with no seed in the filename, but the trainer loaded that file regardless of `--seed`. At any seed ≠ 42, ~80 % of val users live in seed = 42's fold-N TRAIN set → ~80 % val transition leak → reg inflated ~9 pp. **Fix:** filename is now `region_transition_log_seed{S}_fold{N}.pt`; trainer hard-fails if missing.

**Lesson.** Per-seed leakage is subtle. Always seed-tag prior files; hard-fail on missing or unseeded ones.

---

### 2026-04-29 to 2026-04-30 — Phase 3 leakage closure (F50 T4; commit `473dd41`)

**C4 leakage diagnosed and fixed.** Legacy full-data `region_transition_log.pt` leaked val transitions into training; ~13–17 pp inflation propagated through 5 heads. **Fix:** `--per-fold-transition-dir` builds log_T from train-fold-only edges per fold.

**Substrate-asymmetric leak.** C2HGI exploited the leaky log_T more than HGI (α grew more aggressively in C2HGI runs). This **inverted CH18-reg sign at every state** under leak-free measurement: HGI ≥ Check2HGI on reg by 1.6–3.1 pp under matched-head STL (vs the pre-leak-free framing where Check2HGI > HGI on reg).

**Lessons.**
- Leak symmetry is not guaranteed. Check whether different substrates exploit the leak differently before declaring a finding.
- The CH18-reg "MTL substrate-specific" claim was a leak artefact at the reg-side; only the cat-side substrate finding (CH16, CH18-cat) survives leak-free.

**Artefacts.** `research/F50_T4_C4_LEAK_DIAGNOSIS.md`; `research/F50_T4_BROADER_LEAKAGE_AUDIT.md`; `research/F50_T4_PRIOR_RUNS_VALIDITY.md`.

---

### 2026-04-29 — sklearn version reproducibility caveat (commit `4f2a982`)

`StratifiedGroupKFold(shuffle=True)` produces **different fold splits** across sklearn 1.3.2 → 1.8.0 (PR #32540). Within-phase paired tests are unaffected (both arms in each comparison ran in the same env on the same folds), but absolute leak-magnitude attribution across phases mixes leak removal with fold-shift. Disclosure: `FINAL_SURVEY.md §8`. **Fix-forward:** freeze fold indices via `scripts/study/freeze_folds.py` for any future runs.

---

### 2026-04-28 — F37 FL closing + F50 audit (commit `76f2443`)

**F37 STL `next_gru` cat 5f on FL = 66.98 ± 0.61** (pre-multi-seed; seed = 42). Δ_cat = +0.94 pp at FL — cat-side claim survives at scale. **STL `next_getnext_hard` reg 5f on FL = 82.44 ± 0.38** (legacy leaky). At the time, FL Δ_reg was reported as −8.78 pp paired Wilcoxon p = 0.0312, 5/5 folds negative. *(Post-leak-free: this leaky number was inflated; the leak-free FL Δ_reg is −7.99 in v8.)*

**F50 audit.** External-critic-driven audit of the MTL proposal — tiered plan T0/T1/T2/T3.
- T0 Δm joint score (Maninis 2019): backed CH22 — Pareto-positive at AL/AZ on MRR; Pareto-negative at FL on Acc@10. (Sign-flipped after leak-free reframe; final v8 has Δm-MRR positive at FL only.)
- T1 drop-in fixes: FAMO, Aligned-MTL, HSM-reg-head — none reach paired-Wilcoxon significance at FL against H3-alt.

**Artefacts.** `research/F37_FL_RESULTS.md`; `research/F50_DELTA_M_FINDINGS.md`; `research/F50_T1_RESULTS_SYNTHESIS.md`.

---

### 2026-04-27 — F49 attribution analysis (3-way decomposition; commit `a1996e9`)

**3-way decomposition (encoder-frozen λ = 0 / loss-side λ = 0 / Full MTL).**
- Cat-supervision transfer through `L_cat` is null on AL/AZ/FL n = 5 (≤ |0.75| pp); refuted the legacy "+14.2 pp transfer at FL" claim by ≥ 9σ on FL alone.
- AL: H3-alt reg lift = +6.48 pp from architecture alone (frozen-random cat features). *Post-leak-free, this turned out to be a leak artefact (asymmetric leak inflated MTL more than STL); see 2026-05-01 paper-closure entry.*

**Layer 2 methodological contribution (survives leak-free).** Loss-side `task_weight = 0` ablation is **unsound under cross-attention MTL**: the silenced encoder co-adapts via attention K/V projections; encoder-frozen isolation is the only clean architectural decomposition. Generalises to MulT, InvPT, and any cross-task interaction MTL. Regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`.

**Lesson.** The cross-attn ablation pitfall is the most general methodological survival of this study. The substantive AL "architecture-dominant" finding did not survive leak-free re-measurement.

**Artefacts.** `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`; `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`.

---

### 2026-04-27 — Phase 1 substrate validation (commit `f0b2c95`)

**Five-leg substrate study at AL+AZ.**
- CH16 head-invariant: 8/8 head-state probes positive at p = 0.0312 (linear / `next_gru` / `next_single` / `next_lstm`); cat-side substrate Δ +11.58 to +15.50 pp.
- CH15 reframing (initial): under matched MTL reg head `next_getnext_hard`, C2HGI ≥ HGI on reg. *Later leak-free re-run sign-flipped this — see 2026-04-29 entry.*
- CH18 (initial): MTL B3 substrate-specific (HGI substitution breaks reg by 30 pp). *The reg-side of this claim was a leak artefact; only cat-side (CH18-cat) survives.*
- CH19: per-visit context = ~72 % of cat substrate gap at AL (POI-pooled counterfactual). Single-state mechanism evidence; survives all subsequent re-measurements.

**Lesson.** Phase 1 substrate findings on the *cat side* (CH16, CH18-cat, CH19) were robust to leak-free re-measurement. Reg-side findings (CH15 reframing, CH18-reg) were leak-dependent and sign-flipped.

---

### 2026-04-26 — F48-H3-alt champion (commit before `a1996e9`)

**Per-head LR recipe.** `cat_lr = 1e-3, reg_lr = 3e-3, shared_lr = 1e-3` (constant). At the time, claimed +6.25 pp MTL > STL on AL reg. Three orthogonal negative controls (F40, F48-H1, F48-H2) bracketed H3-alt as unique in its design space.

*Post-leak-free reframe.* The "+6.25 pp MTL > STL on AL reg" was a leak artefact. The H3-alt recipe survives as the small-state recipe (paper-grade better than B9 on cat at AL/AZ in `RESULTS_TABLE §0.4`); the original "AL architecture-dominant" lift narrative is superseded.

**Artefacts.** `research/F48_H3_PER_HEAD_LR_FINDINGS.md`; `MTL_ARCHITECTURE_JOURNEY.md` (preserved as supplementary material narrative).

---

### 2026-04-24 — F21c gap discovery; F27 cat-head refinement

**F21c.** Matched-head STL `next_getnext_hard` beat MTL B3 on reg by 12–14 pp at AL/AZ. Filed CH18 Tier B (methodological limitation). *Triggered the F38–F48 attribution chain and eventually the H3-alt recipe.*

**F27.** Cat head `NextHeadMTL` → `next_gru` (+3.43 pp AL 5f, +2.37 pp AZ 5f at p = 0.0312). FL flipped sign at n = 1 (later resolved by H3-alt FL 5f).

**Lessons.**
- Always run matched-head STL ceilings before declaring an MTL win. Comparing MTL against unmatched STL heads is over-claiming.
- Single-fold sign-flips are noise; resolve via 5-fold or multi-seed before reframing.

**Artefacts.** `research/F21C_FINDINGS.md`; `research/F27_CATHEAD_FINDINGS.md`.

---

### 2026-04-22 to 2026-04-23 — B3 champion identified

**B3 = `mtlnet_crossattn + static_weight(cat=0.75) + next_mtl (later next_gru) + next_getnext_hard`.** Validated 5-fold on AL + AZ + FL-1f. Beats baselines on cat F1; beats Markov-1-region on AL/AZ (FL Markov-saturated).

**Mechanism.** Late-stage handover under unbalanced static weighting: cat head converges fast in early epochs under high cat weight, then the shared backbone becomes available to reg in the remaining epochs (cat training extends to ep ~42 vs ≤ 10 for soft/equal-weight).

---

### 2026-04-16 to 2026-04-22 — Phase B-M architecture search (Tier B → B3)

Iterated through MTL backbone variants (`mtlnet`, `mtlnet_cgc`, `mtlnet_ple`, `mtlnet_mmoe`, `mtlnet_dselectk`, `mtlnet_crossattn`) and MTL loss variants (NashMTL, PCGrad, EqualWeight, StaticWeight, UncertaintyWeighting). Convergent choices: cross-attn backbone + static_weight(0.75) + GETNext-hard reg head.

---

### 2026-04-13 to 2026-04-16 — Initial study setup

Branch and study scope established. Phase 0 simple baselines (Markov, Majority). P1 head ablation. P1.5b user-disjoint fair-folds reset (after discovering CH16 measurement was leaky under non-grouped StratifiedKFold).

**Lesson.** User-disjoint cross-validation matters: under non-grouped folds, HGI memorises user-POI co-visit structure and over-performs; under user-disjoint folds, the substrate-asymmetry on cat (CH16) sharpens.

---

## Lessons learned (paper-prep meta)

These are the rules that came out of this study. Apply them to the **next** study.

### Single-source-of-truth discipline

1. **One canonical numerical source per paper.** For check2hgi this is `results/RESULTS_TABLE.md §0`. Every other doc references it; numbers in other docs that diverge are either stale (mark, fix, or archive) or audit-historical (clearly framed as such).
2. **Date-stamp the canonical source.** RESULTS_TABLE now has v6 → v7 → v8 → v9 → v10 stamps. Article-side cites the version explicitly. When the canonical updates, downstream docs must update in the same commit.
3. **Background provenance vs. canon.** Files like PAPER_CLOSURE_RESULTS that record the lab-trail of how a number was computed are valuable as audit but **not** as paper canon. Move them to `archive/` with a deprecation banner once the number lands in the canonical source.
4. **Use a CHANGELOG (this file).** Timeline-organised, dated, with what changed and why. Future readers (including yourself in 6 months) will not remember the F-trail; they will read this CHANGELOG.

### Leak detection discipline

1. **Run leak-free comparisons before declaring a champion.** Two of our biggest narrative reversals (F49 architecture-dominant; CH18-reg substrate-specific) were leak artefacts.
2. **Check leak symmetry.** Different substrates / methods can exploit a leak differently — symmetric removal is what reveals the true ordering.
3. **Seed-tag prior files.** A subtle per-seed log_T leak almost slipped through F51 multi-seed; only a hard-fail on missing seeded files saved us.

### Paired-test ceiling discipline

1. **n = 5 paired Wilcoxon has a ceiling at p = 0.0312 one-sided / 0.0625 two-sided.** State this once in §Experimental Setup; do not let reviewers think p = 0.0312 is a coincidence.
2. **Multi-seed pooling breaks the n = 5 ceiling.** Always pool fold-pairs across seeds where computable. n = 20 (4 seeds × 5 folds) reaches sub-1e-4 p-values. Without it, claims sit at the ceiling regardless of effect size.
3. **Honest small-significance framing.** AL's Δ_cat = −0.78 pp at p = 0.036 (n = 20) is *small-significantly negative*. Don't call it "tied" (that hides the significance) or "MTL trails STL on cat" (that overstates the magnitude); state both axes.

### Story-spine discipline

1. **Reviewer-facing ≠ workflow-facing.** "In flight on H100", "ETA ~1 h", "must check before T3 commits" belong in working notes, never in paper-prep docs that sub-agents will inherit.
2. **External critics are worth their weight.** Two Codex audit passes caught story-level overclaims (scale-sensitive title with TX outlier; substrate +33 pp conflating STL with MTL counterfactual; AL "≈ tied" understating significance). The cost of paying for a critical review is far less than the cost of a desk-rejection.
3. **Honest framing wins at BRACIS.** The 2023 best paper (*Embracing Data Irregularities*) led with "low computational cost", not peak F1. Our paper leads with "the substrate carries; the architecture pays" — also honest.

### Code-as-source-of-truth discipline

1. **The Wilcoxon should be a script, not a manual computation.** `scripts/analysis/gap_fill_wilcoxon.py` is reproducible and re-runnable. JSON artefacts are versioned and citable.
2. **Always emit a JSON.** `GAP_FILL_WILCOXON.json` (n = 20 fold-vectors per state per axis) is what lets a reviewer verify our numbers without re-running 28 paper-grade jobs.

---

## Pointers (where things live now)

> ⚠ **Updated 2026-05-14:** check2hgi study promoted from `docs/studies/check2hgi/` to `docs/` root. The tree below is the **historical (pre-2026-05-14)** snapshot — kept as the v10/v11 reference. For the current layout see [`docs/README.md`](README.md). Mapping:
> - `docs/studies/check2hgi/<file>` → `docs/<file>` (top-level docs)
> - `docs/studies/check2hgi/results/` → `docs/results/`
> - `docs/studies/check2hgi/research/` (F-trail) → `docs/findings/`
> - `docs/studies/check2hgi/research/{canonical_improvement,merge_design,hgi_category_injection}/` → `docs/studies/<name>/` (now active follow-up studies)
> - `docs/studies/check2hgi/baselines/` → `docs/baselines/` (merged with existing BASELINE.md)
> - `docs/studies/check2hgi/{paper,scope,review,launch_plans}/` → `docs/<name>/`
> - `docs/studies/check2hgi/issues/` → `docs/issues/check2hgi/`
> - `docs/studies/check2hgi/archive/<subdir>/` → `docs/archive/check2hgi-<subdir>/`

Historical (pre-2026-05-14) tree:

```
docs/studies/check2hgi/
├── README.md                              ← navigation hub (canonical-source-aware)
├── CHANGELOG.md                           ← THIS FILE (timeline + lessons)
├── AGENT_CONTEXT.md                       ← study briefing (post-v10)
├── NORTH_STAR.md                          ← champion config (post-v10)
├── CLAIMS_AND_HYPOTHESES.md               ← claim catalogue with whitelist banner
├── FINAL_SURVEY.md                        ← substrate panel (canonical)
├── CONCERNS.md                            ← acknowledged risks audit log
├── MTL_ARCHITECTURE_JOURNEY.md            ← supplementary material narrative (F-trail)
├── PAPER_BASELINES_STRATEGY.md            ← which baselines in which paper table
├── results/
│   ├── RESULTS_TABLE.md §0                ← THE canonical numerical source (v10)
│   ├── paired_tests/, P0/, P1/, ...       ← raw JSON artefacts
├── research/
│   ├── GAP_FILL_WILCOXON.json             ← v9 Wilcoxon JSON (cat-Δ + TX recipe landed)
│   ├── PAPER_CLOSURE_WILCOXON.json
│   ├── PAPER_CLOSURE_RECIPE_WILCOXON.json
│   ├── F49_LAMBDA0_DECOMPOSITION_GAP.md   ← cross-attn methodology contribution
│   ├── F50_DELTA_M_FINDINGS_LEAKFREE.md
│   ├── F51_MULTI_SEED_FINDINGS.md
│   ├── SUBSTRATE_COMPARISON_FINDINGS.md
│   └── ...                                ← per-experiment findings
├── baselines/                             ← faithful baseline ports + audits
├── paper/                                 ← paper-prep artefacts (methodology, results, limitations)
├── review/                                ← dated critical reviews
├── issues/, scope/, launch_plans/         ← audit / planning sub-dirs
└── archive/
    ├── post_paper_closure_2026-05-01/     ← stale paper-closure docs (this cleanup)
    ├── 2026-04-20_status_reports/
    ├── pre_b3_framing/
    ├── research_pre_b3/
    ├── research_pre_b5/
    ├── phases_original/
    └── v1_wip_mixed_scope/
```

**Article-side (BRACIS submission):** `articles/[BRACIS]_Beyond_Cross_Task/` — the working paper folder. Sub-agent fan-out plan in `PAPER_STRUCTURE.md` there; per-paragraph beats in `PAPER_DRAFT.md` there; numerical / statistical contracts in `STATISTICAL_AUDIT.md` and `TABLES_FIGURES.md` there.

---

## Maintenance

This file is the chronological record. Append new dated rows at the top (most recent first); never edit historic rows. When a doc is moved to `archive/`, log the move here under that day's row.
