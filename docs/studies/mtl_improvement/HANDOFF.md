# HANDOFF — MTL Improvement track (read this FIRST, then `log.md` + `INDEX.html`)

> ## ✅ STUDY CLOSED (2026-06-12) — HANDOFF_AUDIT punch list DONE; champion G unchanged; CA/TX → `closing-data`.
> The audit punch list (`HANDOFF_AUDIT.md`) is complete: **P0 fixed** (FL cat-transfer re-run — the
> `s1/s7/s100` rows had mis-pointed to the fully-shared intrinsic run; true FL transfer +1.08→**+0.93**,
> sign held), **X1–X4 all NULL** (mixing dead Δcat −0.004; KD-on-G null after the aux-gate fix; β→0 is
> gradient-driven not WD; "matches" fp32-precision-clean Δ −0.005), **H1** cos≈0 widened 2→16 runs
> (+0.0008, n=3,797), **H2** `T4_corrected_rerun.json` committed. New concern **C28** (aux-gate dead
> codepath + rundir race). Evidence: `X_SERIES_FINDINGS.md`, `CODE_AUDIT_2026-06-12.md`, `log.md` (third
> pass), `INDEX.html #tier7`. **No X-probe promoted → no scope re-open.** Only open item: the BRACIS
> paper-doc restatement (author decision). Below is the historical close state.
>
> Tiers 1–5 + the Tier-3 probes (R0/R1/R1b/R2) + Tier 4 are ALL closed; champion **G** stands:
> **matches the STL reg ceiling on the matched metric (Δreg −0.09…−0.31) + beats the STL cat ceiling
> (+2.6…+4.1), 4 states × 4 seeds** (R0 bar, `results/mtl_improvement/R0_matched_metric_bar.json`).
> The 2026-06-12 local re-audit of the Tier-3/4/5 drop found the science sound and self-correcting
> (R1b, B-A2) but flagged: **(P0) the FL cat-transfer multi-seed manifest points 3 seeds at ONE rundir**
> (paper-bound +1.08 FL transfer number at risk — verify/re-run), (H1) the cos≈0 evidence is n=2 runs —
> widen for free over the 16 multistate rundirs, (H2) corrected-rerun/wgrid numbers are markdown-only —
> commit JSONs. It also corrected overclaims in place: Tier-4 is a **convergent-evidence negative, NOT
> "per-method-tuned + arch-wired"** (see `T4_audit_and_verdict.md` banner); "composite strictly
> dominated" → **dominated on the joint reading only** (composite keeps ~0.5pp reg-only FL edge, R0);
> CHAMPION.md/PAPER_UPDATE stale "+Δreg beats-both" tables replaced with the R0 matched values. The
> partition pre-flight T4.1 mandated now exists as pytest (`test_mtl_param_partition.py`, dualtower
> family, 33 green).
>
> **SECOND-PASS DEEP CODE AUDIT (same day, user-requested) — `CODE_AUDIT_2026-06-12.md` + INDEX Tier 7.**
> Three verified structural finds: **(P0-A/X1)** cross-attn trains on RANDOMLY-PAIRED windows (two
> independent shuffled loaders) but evaluates aligned — per-sample cross-modal transfer was never
> trainable (may itself explain "mixing is dead" + caps the cat-transfer reading); **(P0-B/X2)** the
> dualtower head is missing from the aux gate → the `g_kd0.1/0.2` arms were NO-OPS — "KD adds nothing
> on the dual-tower" RETRACTED, KD-on-G genuinely untested; **(P1-C/X3)** the aux fusion scalar β is
> weight-decayed and never logged; **(P1-D/X4)** MTL eval is fp16 vs the fp32 ceiling harness inside a
> −0.09…−0.31pp verdict. X1/X2/X3 are **MTL-only levers (no rising tide — the first since C25)** = the
> last honest in-study shots at "reg beats STL". Also: v11/B9 `--category-weight` is a DEAD FLAG under
> alternating-opt (doc note in CANONICAL_VERSIONS §v11); CLI `KEY=False` bool-inversion FIXED
> (`train.py` + `test_cli_param_coercion.py`); literature verdict = parity is the expected outcome in
> the weak-auxiliary regime (citations in the audit doc Part 2; future-work mechanisms in INDEX `#T7-FW`).
>
> **Remaining work: `HANDOFF_AUDIT.md` (A40 punch list: P0 + X1–X4 + H1/H2) → then the study CLOSES**
> (if an X-probe promotes ≥0.3pp, STOP and report — it re-opens the champion question). **T6.1 CA/TX is
> DEFERRED OUT** to the upcoming **`closing-data` study** (user decision 2026-06-12 — the major
> large-state runs happen ONCE, against the final frozen recipe, after all improvement studies close);
> T6.2 paper-canon restatement = author-side.

**As of 2026-06-06 (⭐⭐⭐ THE UNWEIGHTING FINDING + RE-VALIDATION + CEILING BROKEN — the reg narrative is reframed AND the FL champion (G) `dual aux + prior-OFF` matches the STL reg ceiling + beats the cat ceiling; verbs per B-A2/R0).** Branch `mtl-improve`, pushed.
This is the organized "you are here". Full chronology: `log.md` (newest-first 2026-06-05 entries). Design/cards: `INDEX.html`. Paper-facing: `PAPER_UPDATE.md`. Confound write-up: `docs/CONCERNS.md §C25`.

> ## ✅ TIER 4 CLOSED (2026-06-08) — loss/optimization axis exhausted; G's static_weight is Pareto-optimal
> Audit (user-flagged the suspicious "all balancers cluster at equal_weight"): found gradient-surgery
> (cagrad/pcgrad/aligned_mtl) is mis-wired under G's dual-tower (private reg tower trains at unit weight →
> collapses to equal) + gradnorm/dwa/fairgrad misconfigured + a latent preflight bug (FIXED). BUT the
> negative is real + airtight: **gradient cosine(cat,reg)≈0 at AL+FL** (no conflict for any balancer to
> resolve) + corrected re-runs (gradnorm lr=0.05, nash max_norm=2.2 still trade cat for reg) + static
> cw-sweep (0.75 on the Pareto front) + **scale-norm FALSIFIED** (starves the high-card reg head) + RLW
> litmus + literature (Kurin/Xin NeurIPS'22: tuned scalarization wins at k=2). **Six convergent lines.**
> Mechanistic payoff: orthogonal task gradients unify the study (balancers can't help; more-sharing failed
> in Tier 2; dual-tower wins). New: `--loss-scale-norm` flag (gated, default off). 3 figures for the talk.
> Write-up: `docs/results/mtl_improvement/T4_audit_and_verdict.md`. **Champion G unchanged.**
>
> ## ✅ TIER 5 CLOSED + follow-ups DONE (2026-06-08) — G unchanged; only completeness remains
> - **(a) cat-transfer ablation DONE:** the +3pp MTL-cat gain is **architecture-dominated** (cross-attn trunk +2.27 FL/+3.11 AL); genuine region→cat transfer only +0.89 FL / −0.71 AL. Refines CH30 (not "region teaches category"). `cat_transfer_and_T53.md §a`.
> - **(b) T5.3 HSM reg head FALSIFIED:** HSM = flat softmax at FL 4.7k (73.21 vs 73.22). No dual-tower-HSM build motivated. With T5.1/T5.2 already done (STAN load-bearing; next_gru+plain-CE), **Tier 5 is CLOSED.** `cat_transfer_and_T53.md §b`.
> - **Conceptual write-up created** (the orthogonality / why-no-modern-optimizers narrative + the 3 figures): `WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`.
> - **Cross-doc settled:** CHANGELOG 2026-06-08, CONCERNS §C27, CLAIMS CH31 + CH30 cat-refinement, MTL_FLAWS §5.6, CANONICAL_VERSIONS v16 re-confirm.
> **→ NEXT AGENT: see `HANDOFF_TIER5.md` (Tier 5 is closed; the live work is Tier 6 — CA/TX completeness + the BRACIS paper-doc restatement).**
>
> ## ✅ TIER 3 CLOSED (2026-06-08) — reg-input axis exhausted; G unchanged; NEXT = Tier 4 (T4.0)
> **R0/R1/R2 done + advisor-signed-off CLOSE.** (1) **R0** (free re-score) pinned the matched G−ceiling bar
> multi-state: G **matches** reg (Δ −0.09/−0.12/−0.09/−0.31 at AL/AZ/GE/FL, Pareto-non-inferior) + **beats**
> cat (+2.6..+4.1) — the "matches" verb now holds at all 4 states, not FL-only. (2) **R1** (overlap/data-scale)
> + **R2** (HGI/substrate) = clean **rising-tide nulls** on the matched bar: both reg-input levers *transfer*
> into G's private reg tower post-C25 (falsifying "data/substrate washes out in MTL" TWICE) but neither beats
> the achievable ceiling (the ceiling moves with them — magnitude rule). (3) **R1b** de-confound: the overlap
> absorption is **C25 unweighting, not the dual-tower** (advisor-caught over-read, corrected). T3-richer NOT run
> (trigger "R1/R2 stall" didn't fire; it'd be a 3rd rising tide). **No Tier-3 probe changes champion G.**
> Results: `docs/results/mtl_improvement/{R0_matched_metric_bar.json,R1_overlap_under_g.md,R2_dual_substrate_routing.md}`;
> drivers `scripts/mtl_improvement/{r0_matched_rescore.py,r1_overlap_under_g.sh,r1b_shared_overlap_deconfound.sh,r2_dual_substrate_routing.sh}`.
> **→ NEXT: Tier 4 (loss/optimization) — the ONE lever class with no STL analogue → the ceiling is FIXED for it
> (unlike every reg-input lever), so it is the only place a lever can structurally move G−ceiling. Start T4.0
> (loss-scale-norm + RLW litmus, cheap/ungated), then T4.1 (per-method-tuned registry, arch-wired to G).**
>
> ## ⭐ CURRENT STATE & NEXT (2026-06-07) — read this first
> **The experimental study is at a paper-grade CLOSE.** Tier 2V + the whole `CRITIQUE_TIER2_C25` are CLOSED
> (all `[x]` except T2V.9 CA/TX = `[~]` deferred). **Champion = G** (`mtlnet_crossattn_dualtower` +
> `next_stan_flow_dualtower`, aux+prior-OFF, v14, unweighted onecycle KD-OFF): a single MTL model that
> **MATCHES the STL reg ceiling** (Pareto-non-inferior within ~0.4pp matched-metric — the B-A2 "matches not
> beats" correction) **and BEATS the STL cat ceiling by +3pp**, at all 4 available states, 4-seed. **G′
> (cat-private) is DEMOTED** — an FL-only dead-end (craters small-state cat); G (cat-shared) is THE champion.
>
> **⛔→♻ Tiers 3–6 RE-SCOPED IN PLACE (2026-06-07, user-reopened + advisor-calibrated) — INDEX `#tier36`→`#tier6`.**
> The original chain's HEADLINE levers ARE moot (log_T-KD = identical to G; T5 heads settled, B-A1). **BUT the
> C25/G reversal RE-OPENED the cards whose prior negative/moot verdict depended on the now-removed regime**
> (class-weighted, shared-backbone). Re-scoped into the REAL tiers (no parallel umbrella):
> - **Tier 3 — Reg-input pathway** (prior · supervision · substrate): **R1 ★** overlap-under-G (RE-OPENED — the
>   dense-supervision MTL-negative was the *shared backbone*; G's private tower may absorb it = dual-tower
>   mechanism test) · **R2 ★** dual-substrate routing HGI→reg, FL (REFRAMED — C25 falsified "washes out"; hook+HGI
>   on disk) · T3.1 KD = done/moot · richer-priors = parked.
> - **Tier 4 — OPTIMIZATION ★ (user-prioritised, expanded with `src/losses`):** T4.0 loss-scale-norm + RLW
>   (untested, distinct from balancers) · **T4.1 the FULL registry incl. NashMTL** — T2V.6 swept only 4 of ~20
>   AT DEFAULT PARAMS; the rest (db_mtl/dwa/gradnorm/aligned_mtl/uw_so/stch/scheduled_static/RLW/…) untested
>   under G. ⚠ **Each balancer (a) at its OWN best params** (Nash optim_niter/max_norm — watch cvxpy collapse;
>   CAGrad c; GradNorm α; FAMO γ/w_lr) **and (b) arch-wired to G's dual-tower partition** (gradient-surgery
>   methods enumerate `shared_parameters()`; the private tower ∈ `reg_specific` — run the param-partition
>   unit-test on G per balancer before its result counts; O(k) Nash/CAGrad/PCGrad may be slow) · T4.2 optimizer
>   (per-head LR + AdamW knobs; Lion/Prodigy = optional new dep).
> - **Tier 5 — Heads:** T5.1/T5.2 DONE (STAN load-bearing, next_gru+plain-CE); **T5.3 ★** HSM high-cardinality
>   reg head = the one live residual (never GPU-run, large-state lever).
> - **Tier 6 — Ship+completeness:** **R0/T6.0** (FREE, FIRST) multi-state matched-metric re-score = the
>   G−ceiling "matches" bar (FL-only today; gates every probe) · T6.1 CA/TX (heavy, parallel) · T6.2 paper-canon.
> - PARKED (only if the probes stall): T3 richer priors, T4.2 Lion/Prodigy.
> ⚠ **Magnitude rule (C25-trap):** every probe measures whether it moves **G−ceiling on the MATCHED metric**
> (a lever lifting STL reg lifts the ceiling too) — NOT "a +5pp lever for a 0.35pp gap." Report the **mechanism**.
> **Sequencing:** T6.0/R0 (free, pins the bar) FIRST → R1+R2 (Tier 3) + T4.0/T4.1 (Tier 4) cheap probes → T5.3 →
> contingencies only if all stall. CA/TX + paper-canon = PARALLEL track.
>
> **DEFAULTS LANDED (2026-06-07): G is now the train.py default via `--canon` (v16).** `train.py --task mtl
> --state X --seed S` runs G. Traceback with ONE flag: `--canon v11|v12|v15|none` (explicit flags override the
> bundle; `src/configs/canon.py`, guarded by `tests/test_configs/test_canon.py`). **Contract: pin `--canon` in
> EVERY script** (partial flags merge with the default bundle). See `CANONICAL_VERSIONS.md §The --canon selector`.
>
> **COMPLETENESS (parallel, won't change the champion):** **CA/TX (C-A/T2V.9)** — v14 build at CA(8.5k)/TX(6.5k),
> the scale-conditional reviewer challenge (heavy); **paper-canon restatement (C-B/T6.2, author)** — CH25/CH28/§0.1
> → matches-reg + beats-cat Pareto-positive. (R0 also hardens the "matches" verb multi-state — do it first.)

> ## ⭐⭐⭐ THE FINDING (the "unweighting problem") — read this whole block
> **What it was.** The MTL reg head was silently trained on **class-WEIGHTED CrossEntropyLoss** (`default_mtl use_class_weights=True`, `src/configs/experiment.py:235,364` → `mtl_cv.py:1283-1291`), while the STL reg ceiling (`p1_region_head_ablation.py`, unweighted) AND the reported metric (**`top10_acc_indist` = Acc@10**) are frequency-weighted. Class-balancing optimises *macro* accuracy and **away from top-K**, depressing MTL reg **~10-14pp** (scales with class count / imbalance → bigger at FL). **This — not architecture, not the joint loop, not the substrate — was the entire "MTL→STL reg gap."** The cat head was also weighted; unweighting it gains **+3-5pp macro-F1** too. (The α=0 OOD floor finding is distinct and NOT dissolved by this.)
>
> **The FIX (committed).** **Per-task class-weighting**, BOTH heads UNWEIGHTED by default (`use_class_weights_{reg,cat}=False` in `default_mtl`; `mtl_cv.py:1283-1291`). reg-unweighted matches Acc@10 + the STL ceiling; cat-unweighted was **empirically validated** (+5.1pp macro-F1 AL — the "balancing helps macro-F1" assumption was tested and FALSE). CLI `--[no-]reg-class-weights` / `--[no-]cat-class-weights`. **Reproduction of pre-C25 numbers:** `--reg-class-weights --cat-class-weights`. (Re-audit verdict: this is the SINGLE dominant confound — no second equal bug; two smaller real secondaries flagged: wd=0.05-vs-ceiling-0.01 [tested, NOT the cause]; the Acc@1 reg checkpoint monitor [deployable-only ~2-3pp].)
>
> **RE-VALIDATION RESULTS (all multi-seed {0,1,7,100}, unweighted, real-joint, AL/GE/FL — the new evidence base):**
> | claim (OLD, class-weighted) | NEW (unweighted) | verdict |
> |---|---|---|
> | Regime finding "STL substrate gain washes out in MTL" (CH28) | Δreg(v14−canon) **+1.92 AL / +1.49 GE / +0.81 FL** (σ~0.1) | **OVERTURNED — substrate TRANSFERS to MTL** |
> | "MTL sacrifices reg −7..−17pp" (§0.1) | MTL reg ≈ STL ceiling (AL +1.6 / GE −0.6 / FL −1.8 vs (c)); §0.1 MTL reg **+10-13pp** | **DISSOLVED** |
> | "Composite = +7-12pp over MTL; ship 2 models" (CH25) | single MTL model **≥ composite** (AL +0.94, GE −0.92, FL −2.07) | **DISSOLVED — 1 model ≈/beats composite** |
> | Tier-2 "dual-tower LOSES, irreducibly architectural" | dual-tower went WORST→**BEST**: FL +1.51 vs base_a, −0.25 vs ceiling (orderings FLIPPED) | **OVERTURNED** |
> | MTL cat (§0.1) | MTL cat **EXCEEDS** the STL cat ceiling everywhere (+2-3.5pp) | re-stated UP |
>
> **The narrative flip:** OLD = *"MTL sacrifices reg; the gap is irreducibly architectural; ship the 2-model composite."* → NEW (as of 2026-06-06) = **"a single MTL model BEATS both STL ceilings (reg AND cat) and matches the 2-model composite — the MTL tradeoff is INVERTED, not just dissolved."** The strongest possible paper.
>
> **Residual: CLOSED (2026-06-06; reg verb CORRECTED 2026-06-07).** The FL gap was the **α·log_T prior + shared-pathway dilution** (NOT architecture — MoE & SwiGLU both null on reg). The **(G) champion `dual aux + prior-OFF` MATCHES the (c) STL reg ceiling** (Pareto-non-inferior) AND substantially **beats the cat ceiling** (73.16 vs 69.97 = **+3.19**). ⚠ **B-A2 correction:** the earlier "BEATS reg +0.26" compared G's *indist* Acc@10 (73.57) to the (c) ceiling's *full* `top10_acc` (73.31); on a matched metric G is ~0.35pp BELOW (FL-full 72.93 vs 73.31) → reg = "matches", not "beats" (see INDEX `#T2V-3` + `log.md` 2026-06-07). The cat +3.19 beat is exact. The inverted-tradeoff headline STANDS (matches reg + beats cat). ⚠ **G′ (cat-private, FL-ONLY — DEMOTED 2026-06-07):** giving the CAT head a private tower too (both-private dual-tower, `mtlnet_crossattn_dualtower_catpriv`) gained cat at FL only (74.77, +1.61); the multi-state confirm (AL/AZ/GE × 4 seeds) **FALSIFIED** it — cat CRATERS at small states (AL 37.66 = −15.25 vs G, AZ −12.45, GE −3.59; reg flat). The cat-private tower **UNDERFITS** small-state cat (NOT overfit — the off-label STAN-flow head is over-regularized: AL train-F1 caps ~0.45 vs the GRU head's 0.98; a rescue screen of lower dropout / softer LR / smaller tower **CLOSED 2026-06-07 with NO rescue** — best AL lever still −14.5pp vs G, and the FL gain survives ONLY at the original `priv_dropout=0.3` (lowering it erases the gain, 74.74→73.17); the STAN flow/attention head is architecturally mismatched for a 7-class target at small data) → G′ is a **CLOSED FL-only experimental dead-end**, **NOT a champion**; **G (cat-SHARED) remains the multi-state champion**. See CHAMPION.md / INDEX `#T2V-5`. `aux` fusion adds the shared pathway WITHOUT diluting the private tower (vs `gated` competition); prior-OFF removes the biased logit term. Config + trail: `log.md` 2026-06-06 "CEILING BROKEN".
>
> **CLOSED 2026-06-05 (the three remaining gaps + the optional stretch — see `log.md` 2026-06-05 "STRETCH CLOSED" entry):**
> - ✅ **#17 CANONICAL_VERSIONS v15** pinned (C25 unweighted recipe + reproduction map `--reg/cat-class-weights`); per-task class-weighting in `NORTH_STAR.md`.
> - ✅ **#18 Acc@1→Acc@10 reg checkpoint monitor** fixed (`PrimaryMetric.TOP10` + reg `primary_metric`→TOP10; selector-independent reads unaffected).
> - ✅ **#19 FL-B9 §0.1 continuity** (canon GCN, exact B9 recipe). **Same-harness A/B (only the class-weight flag differs):** the C25 fix lifts §0.1 FL MTL **+3.15 reg (63.91→67.06) / +3.52 cat (70.34→73.86)**; architectural Δ_reg (vs §0.1 STL 70.62) **−6.71→−3.56 (~halved)**, Δ_cat +3.18→+6.70. Harness drift vs *published* §0.1 quantified (+0.64 reg / +1.78 cat). → RESULTS_TABLE §0.1 gets a continuity ANNOTATION (author sign-off; no table rewrite).
> - ✅ **T2.3 MoE (mmoe/cgc) + T2.4 SwiGLU (`mtlnet_crossattn_swiglu`, pre-norm+SwiGLU, gate GREEN) = both NULL on reg** (+0.13..+0.22, within σ). Two independent capacity/quality interventions null → the gap is NOT architecture capacity. SwiGLU's only effect: a cat bump.
> - ✅ **T2.4 COMBO SCREEN → CEILING BROKEN (2026-06-06).** Advisor-ranked 1-seed screen of dual-tower reg-lever combos, promoted the 2 ceiling-breakers to 4 seeds. **(G) `dual aux + prior-OFF` = CHAMPION: FL reg 73.57±0.06 (+0.26 over the (c) ceiling) / cat 73.16 (+3.19) — a single MTL model that BEATS both STL ceilings + ties the composite.** (H) private_only+prior-OFF also clears (73.42). Config: `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (`raw_embed_dim=64 fusion_mode=aux freeze_alpha=True alpha_init=0.0`). New model `mtlnet_crossattn_dualtower_swiglu` (combo F) built+gated but NULL on reg (cat play). Drivers: `c25_combos_{screen,promote}.sh`.
> - ✅ **G GENERALIZES + sweep VALIDATES it (2026-06-06, `c25_gv2.sh`, seed-0).** **(III) G sweep found NO Pareto gain** — category-weight/priv_dropout/log_T-KD/fp32 all ≈ or worse than G (fp32 +0.13 reg but −1.11 cat = a trade, not a win; KD identical; G is well-tuned). **(I) Speculative hybrids NULL/NEGATIVE on reg** — MulT-faithful 71.28, crossstitch→crossattn 71.13 (both < base_a 71.55); architecture-capacity hypothesis now falsified **5 independent ways** (MoE, SwiGLU, MulT, xstitch, more-blocks).
> - ✅ **G CONFIRMED MULTI-STATE @ 4 SEEDS (2026-06-06, `c25_g_multistate.sh`).** G beats **BOTH (c) STL ceilings at ALL 4 available states**, 4 seeds each (tight σ): **AL reg 64.47±0.11 (+1.59)/cat 52.91 (+2.94); AZ 55.75±0.21 (+0.64)/54.48 (+3.47); GE 59.37±0.04 (+0.92)/61.43 (+3.31); FL 73.57±0.06 (+0.26)/73.16 (+3.19)**. The Pareto-positive headline is now **multi-state, multi-seed, paper-grade** — the MTL tradeoff is INVERTED at every state, not just FL.
> - ✅ **TIER 2V — CRITIQUE CLOSE-OUT (2026-06-06/07, `CRITIQUE_TIER2_C25` §7).** Every critique concern resolved: **T2V.1** reg headline HOLDS seed-matched (multi-seed ceilings stable σ≤0.7, G beats 4/4); **T2V.2** no tail regression (prior-OFF ≈ prior-ON on macro); **T2V.3** reproduced + artifact-foreclosed (independent p1 harness) + "½ params" corrected (G=base_a +4.9%); **T2V.4** alt-arch FAIR re-rank (per-arch cat-weight, post-C25, standalone) → **G holds, all alts −1.6 to −2.1pp** → "architecture-capacity falsified" now un-confounded/paper-safe; **T2V.5/6/7** no hypertuning lever beats G (logit-adjust HURTS MTL cat — plain CE is the optimum; STAN right-sized; FAMO ≈ G) → **T2V.8 combine MOOT**. **G is validated, robust, paper-safe.** Lone open card: **T2V.9 (CA/TX)** — heavy `design_k` build on the 2 largest states (user go/no-go). Cards: INDEX `#tier2v`.
>
> **STILL OPEN (paper-doc restatement ONLY — author decision, no GPU work left):** re-state CH25/CH28/§0.1 + the Tier-2 verdict (Pareto-POSITIVE, multi-state, 4-seed) in the BRACIS paper docs. Optional/non-blocking: re-run (c)/(d) ceilings multi-seed to seed-match (G margins robust to it); G at CA/TX after a v14 substrate build. T2P.1/2/3 = UNNEEDED.
>
> **How it was found (provenance — 3 retractions; read so you don't re-walk them):** T2P.0 isolated the gap to "the joint loop" → a wrong "input-artifact" hypothesis (RETRACTED: MTL reg input byte-identical to (c)) → an agent's "wrapper preamble" localization (also a RED HERRING) → a deletion-bisect + verification pinned it to the class-weighted reg CE. (My own earlier "loss-weighting ruled out" was WRONG — I checked the `getattr` fallback, not the `default_mtl` factory value True.) Hunt scripts: `scripts/mtl_improvement/{t2p0_*, c25_*}.{sh,py}`; diagnostic env-gate `MTL_DISABLE_AMP=1` (fp32, NON-causal). **Lesson: test loss-vs-metric interactions, don't reason about them (it bit me twice — `use_class_weights` itself + the cat default).**

---

## 0. ⛔ SUPERSEDED (2026-06-05 by the C25 unweighting finding — see the top block). [Kept for the trail.]
> **This §0 "Tier-2 architecture NEGATIVE" close-out is OVERTURNED.** It was measured under the class-weighting confound. Under the unweighted fix the dual-tower goes WORST→BEST and the gap closes (see top + `log.md` 2026-06-05 TIER-2 RE-RUN entry). Everything below §0 (T2P.0, the joint-loop thread, etc.) is the historical hunt that LED to the C25 root cause — read as provenance, not current verdict.

**(1) [SUPERSEDED] Architecture: clean, multi-seed-hardened NEGATIVE.** No single-model MTL architecture closes the
MTL→STL reg gap. The reg-private **dual-tower LOSES** to the matched baseline (FL multi-seed −3.35); a
5-point sharing dose-response (`CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower`) shows **more sharing
helps reg** — refuting the §6.4 "missing private backbone" hypothesis; 3 mechanism cells (cat-weight=0,
prior-OFF+wd0.01 — but that wd0.01 cell was the SHARED config base_a, NOT the private tower) localize the
SHARED-config gap to the **joint cross-attn pathway** (not interference/prior/wd). **→ the composite
(two-model) is the deployable reg answer for now. ⚠ "irreducibly architectural" is SUPERSEDED by the
REDIRECT below — the private-tower (STL-topology) collapse points the residual at the training PROTOCOL.**
The sharing-topology axis is near-exhausted — **T2.3 (MoE) + T2.4 (hybrids) remain as the confirmatory close of
the card set (§0b; expected negative per §6.3).** CrossStitch = a real-but-small partial (+1pp reg multi-seed, mixed cat,
−5..−10 below ceiling, NOT a closer). Implementation (`next_stan_flow_dualtower`,
`mtlnet_crossattn_dualtower`) + unit gate + all drivers are committed; capstone advisor verified the
code is correct and the decisions sound.

**(2) Recipe WIN: `onecycle` is the new recommended SMALL-STATE recipe.** onecycle (aggressive schedule,
NO alt-opt) dominates H3-alt at AL/AZ (v14 multi-seed +6–9pp reg / +1–2pp cat) and beats B9 on the v11
paper substrate (AL reg +2.98 / cat +7.36; AZ reg +0.76 / cat +4.69). **alt-opt flips sign by scale** →
keep B9 at large states (FL/CA). Adopted in `NORTH_STAR.md`. **§0.1 small-state arch-Δ annotated in
`results/RESULTS_TABLE.md §0.1` — author sign-off needed** (it reshapes a central claim; reg shrink is
modest on v11, the cat-flip is mostly a B9→deployable-recipe fix — read the nuance in `PAPER_UPDATE.md`).

**⭐ 2026-06-04 REDIRECT (user-approved) — NEW Tier 2P (joint-training protocol). This supersedes "ship
the composite."** Independent review found the close-out's "irreducibly architectural" headline is
**contradicted by its own data**: the `private_only` dual-tower arm IS the STL reg topology by
construction, ran under the GOOD (onecycle) recipe, and still lost ~10pp to STL-standalone (AL 52.41 vs
62.88). An identical topology failing only when trained jointly ⟹ the residual is **the joint training
PROTOCOL, not the topology** — which Tier 2 never varied (onecycle, a pure protocol change, already
recovered +5–9pp). **New `Tier 2P` (INDEX `#tier2p`): T2P.0 linchpin → T2P.1 staged / T2P.2 asymmetric-
recipe / T2P.3 distillation, goal = composite-quality reg in ONE model.**

**THE IMMEDIATE NEXT STEP is `T2P.0` (the linchpin, ~few GPU-h, DECISIVE) — see §0c below.** Then:
(a) finish T2.3+T2.4 as the confirmatory close of the *topology* card-set (§0b — no longer the headline,
just completeness); (b) run Tier 2P per the T2P.0 verdict; (c) the §0.1 paper re-statement decision
(author); (d) Tier 3 / close only after Tier 2P; (e) optional onecycle CA/TX.

---

## 0b. ⛔ SUPERSEDED (2026-06-06 by the Tier 2V close-out — see top block). [Kept for the trail.]
> T2.3 (-lite MoE) + T2.4 (SwiGLU/MulT/xstitch) **were run and are NULL on reg** (closed in INDEX); the
> "run after/alongside T2P.0" framing below is obsolete (T2P is MOOT). **Do NOT start here.** Live residuals
> + the A40 queue = `CRITIQUE_TIER2_C25_2026-06-06.md §8`.

### [obsolete] T2.3 + T2.4 — confirmatory close of the TOPOLOGY card-set

**You are running T2.3 (faithful MoE family) + T2.4 (per-task-input mixers/hybrids)** — the two Tier-2
cards not yet executed (INDEX `#T2-3`, `#T2-4`). Tier 2's verdict so far is a hardened NEGATIVE (§0); the
prior (§6.3) says MMoE/CGC lose ~2.7pp reg and PLE collapses, so **these are expected to be confirmatory
negatives** that complete the architecture card-set for the paper. Promote only on a genuine surprise
(≥1pp on the targeted axis, cat non-inferior TOST δ=2, at ≥2 of {AL,AZ,FL}).

**The comparand (CRITICAL — use the matched baseline, not the landed (a)):** score Δ vs **`base_a` @
onecycle MULTI-SEED**, which already exists as the **`onecyc_val`** rows in
`scripts/mtl_improvement/t21_harden_manifest.tsv` (= `mtlnet_crossattn + next_getnext_hard` @ onecycle,
{0,1,7,100}, AL/AZ/FL). Also position each arm vs the frozen (c)/(d) ceilings (§2) and **fold the new
arms into the sharing dose-response** (`scripts/mtl_improvement/t21_doseresp.py` +
`docs/results/mtl_improvement/T21_dose_response_50ep_seed42.txt`) — MoE/hybrids are new points on the
`CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower` curve.

**Recipe (the now-adopted one):** `onecycle` for AL/AZ (`--scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3
--reg-lr 3e-3 --shared-lr 1e-3`), `mtlnet_crossattn`-style COMMON: `--mtl-loss static_weight
--category-weight 0.75 --cat-head next_gru --reg-head next_getnext_hard --task-a-input-type checkin
--task-b-input-type region --log-t-kd-weight 0.0 --engine check2hgi_design_k_resln_mae_l0_1
--per-fold-transition-dir output/.../<state> --no-checkpoints`, 5f×50ep, seeded per-fold log_T. AL+AZ
first; FL only for a promoted arm.

**T2.3 — faithful MoE.** Models `mtlnet_mmoe`, `mtlnet_cgc` are REGISTERED and (verified 2026-06-04)
their `cat/reg/shared` partition is already **bijective+exhaustive (0 uncovered params)** → they run
under the per-head onecycle recipe with NO code change. **Skip PLE** (robust collapse, §6.3/F50).
**Fidelity caveat:** the on-disk MMoE/CGC are "-lite" per-task-input adaptations (not canonical; DSelect-K
is misnamed — dense convex combo), see `MTL_FLAWS_AND_FIXES.md §3.1`. Pragmatic path given the §6.3
prior + exhausted axis: run the existing -lite MMoE+CGC as the confirmatory multi-seed check + document
the fidelity caveat; only build faithful versions if -lite surprises. DSelect-K only if MMoE/CGC surprise.

**T2.4 — 3 hybrids (must BUILD — not in registry).** (i) MulT-faithful (intra-task self-attn BEFORE the
cross-attn block — extend `mtlnet_crossattn`); (ii) cross-stitch→cross-attn series (compose
`mtlnet_crossstitch` then `mtlnet_crossattn`); (iii) pre-norm + SwiGLU FFN (a `_CrossAttnBlock` variant).
Each = new `@register_model` subclass + the gates below. (The card says "compose with T2.1 winner" — but
T2.1 was negative, so the comparand is `base_a`; reframe T2.4 as "does per-task-input mixing beat plain
cross-attn?".)

**HARD GATES (do not skip — same as T2.1):**
1. **Unit-test gate (hard rule 10)** before any multi-fold launch — adapt `scripts/mtl_improvement/
   t21_unit_gate.py`: forward/backward on a synthetic 100-user batch, loss-finite, param-count within
   ~10% of B9 at D=256, and `shared/cat_specific/reg_specific_parameters()` **bijective+exhaustive**
   (for any NEW T2.4 module, wire it into the right group — experts/mixers → `shared`, reg-private bits
   → `reg_specific`; the base `MTLnet.shared_parameters()` is a NAME-SUBSTRING match, so a new module
   whose name lacks `shared_layers/film/task_embedding` will be SILENTLY dropped from `shared` → fix by
   overriding the partition like `mtlnet_crossattn` does). MMoE/CGC already pass; verify anyway.
2. **Per-arch LR mini-sweep (hard rule 7)** for each T2.4 hybrid (new arch): 5 regimes × AL+AZ × 5f×40ep
   × seed42, then full-protocol at the winner. (Reuse `t21_lr_sweep.sh` pattern.) MoE may start at
   onecycle (existing arch) + mini-sweep only if it surprises.
3. Stay at `shared_layer_size=256`; no fclass-as-feature; log_T-KD OFF; **stale-log_T preflight**
   (`stat` log_T vs next_region.parquet) + `freeze_folds.py --check` before each sweep.

**Reusable assets (this session):** `t21_harden.sh` (copy the `harden2` stage pattern — add a `t23`/`t24`
arms function + STAGE case; it has the **PID-safe rundir capture** + the **process-substitution wait-fix**
+ idempotent manifest skip), `t21_doseresp.py` / `t21_agg.py` (aggregation), `t21_unit_gate.py` (gate
template). **CONC discipline:** small states ~5GB (CONC=4 ok), FL ~14GB (CONC≤2), CA ~31GB (CONC=1 only).
**Rundir-race trap:** never capture rundirs via `ls -dt | head -1` under concurrency — use the `$!`
PID-suffix (the driver already does; see memory `ref-concurrent-rundir-race`).

**Tier-2 close after T2.3+T2.4:** if nothing recovers a meaningful fraction of the composite gap
(expected), the architecture axis is CLOSED with the complete card set (T2.0–T2.4) → the paper's negative
("MTL reg gap irreducibly architectural; ship composite") is final. If a hybrid surprises → promote,
re-judge under MTL + HGI sanity probe (2 seeds × AL+AZ × 5f×30ep), compose with base_a. Then advisor pass
→ update `PAPER_UPDATE.md` + this HANDOFF → surface to user (tier-boundary cadence).

---

## 0c. ⛔ SUPERSEDED (2026-06-06 — T2P is MOOT; do NOT start here). [Kept for the trail.]
> **This said "NEXT AGENT STARTS HERE — T2P.0".** That is obsolete: the "joint loop poisons reg" hypothesis
> T2P.0 was built to test WAS the C25 class-weighting confound (top block). T2P.0/.1/.2/.3 are superseded.
> **The live next work is `CRITIQUE_TIER2_C25_2026-06-06.md §8` (A40 queue) — start with B-A1, the lighter
> GRU/TCN private tower.** The §0c body below is kept only for the provenance trail.

### [obsolete] T2P.0 — the linchpin (kept for the trail)

**Run T2P.0 FIRST** (INDEX `#T2P-0`) — it decides whether Tier 2P's primary lever is staged training
(T2P.1) or asymmetric per-task recipe (T2P.2). It is one clean knob change on an existing cell.

**Why this is NOT already done (read — the mechanism cells are close but don't cover it).** The mechanism
probe (`t21_mech.sh`, log §2026-06-04) ran `base_clean` = the **SHARED** config (base_a, cross-attn) at
prior-OFF+wd0.01+cat0 → AL 57.07, ≈ unchanged from wd0.05 → so prior/wd do NOT explain the **shared**
config's −5.8pp gap (that gap is the cross-attn pathway). BUT the **private tower** (`dtpriv_cat0`, the
STL-topology arm) was tested ONLY at wd=0.05+prior-ON (AL 52.98) — it was **never** matched to (c) on wd
(the team noted a per-head `reg-head-wd` "needs a new flag"). T2P.0 closes exactly that cell, and it does
NOT need a new flag: in `private_only`+`cat-weight=0`, a GLOBAL `--weight-decay 0.01` IS the private
tower's wd. T2P.0 is the clean test of "does the joint LOOP alone poison the STL topology."

**The experiment.** Re-run the `private_only prior-OFF` arm at **wd=0.01** (STL-matched), everything else
identical to (c). Compare vs (c) 62.88 and vs the existing `dtpriv_cat0` wd0.05 cell (AL 52.98):
```
train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_design_k_resln_mae_l0_1 \
  --model mtlnet_crossattn_dualtower --reg-head next_stan_flow_dualtower \
  --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=private_only \
  --reg-head-param alpha_init=0.0 --reg-head-param freeze_alpha=True \
  --category-weight 0.0 --weight-decay 0.01 --scheduler onecycle --max-lr 3e-3 \
  --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 --log-t-kd-weight 0.0 \
  --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state> --no-checkpoints
```
AL+AZ+FL, 5f×50ep seed42, seeded per-fold log_T. **Pre-check (advisor):** confirm in `mtl_cv.py` that
`--category-weight 0.0` zeroes cat's gradient into the shared/cat params (not just the loss scalar) — in
`private_only` mode reg never touches the cross-attn, so with cat-weight 0 the only residual difference
from STL is the joint loop (`max_size_cycle` mixed-batch iteration + the shared optimizer/scheduler step).

**Decision.** Recovers to ≈62 (STL-level) → the collapse was wd/recipe mismatch, the joint loop is NOT the
poison → **T2P.2 (asymmetric per-task recipe) primary.** Still ≈52 → the joint LOOP caps reg even with
identical arch+HP → **T2P.1 (staged / sequential) primary.** Then STOP + surface to the user (tier-boundary
cadence) before launching the chosen lever. Honest-framing reminder: staged reg→freeze→cat gives reg≈STL by
construction — the real question is whether CAT survives the frozen-reg trunk (composite-quality in one
model); the 2-model composite is the null every Tier-2P arm must beat. See INDEX `#tier2p`.

---

## 1. Where we are (pre-Tier-2 context, all still true)
- **Tier 0 + Tier 1 are COMPLETE and FROZEN.** The (c)/(d) STL ceilings are the immutable track yardstick
  (UNTOUCHED by Tier 2; `t14_freeze_sanity.py` GREEN).
- **Tier S (STL head search) is COMPLETE — a reviewer-proof NEGATIVE**: the head is NOT the lever.
- **Tier 2 sharing-topology axis COMPLETE (NEGATIVE)** — see §0; **redirected to Tier 2P (protocol), next = §0c.**
- A major out-of-band finding (overlapping windows) was validated + documented as future-work; **the
  non-overlapping canon is deliberately KEPT** for whole-study consistency.

## 2. The FROZEN ceilings (immutable — T2-T5 Δ are measured against these)
Recipe: **reg** = `next_stan_flow` α=0 (log_T prior OFF); **cat** = `next_gru` logit-adjust τ=0.5.
| | AL | AZ | GE | FL |
|---|---|---|---|---|
| (c) cat macro-F1 | 49.97 | 51.01 | 58.12 | 69.97 |
| (c) reg Acc@10 | 62.88 | 55.11 | 58.45 | 73.31 |
| (d) composite reg (max v14/HGI α0) | 63.58 | 55.11 | 58.76 | 73.62 |
Guard: run `scripts/mtl_improvement/t14_freeze_sanity.py` after ANY ceiling change (asserts cat-ceiling
arch == NextHeadGRU + every ceiling ≥ the MTL it bounds). Currently GREEN. **Caveat:** these are
seed=42 single-seed; §0.1 paper numbers use seeds {0,1,7,100}.

## 3. What got done this session (newest first) + pointers
1. **Overlap-window study** (`future_works/overlapping_windows.md`, `results/.../overlap_window_probe.md`):
   non-overlapping windows cap ceilings ~+5–9.8pp at small-state STL (head-independent). Validated harness
   + real STL + real MTL. **Documented as future-work; canon kept.** See §4.
2. **Pipeline deep-dive audit** (`PIPELINE_AUDIT_2026-06-03.md`): the windowing cap (HIGH) + secondary levers
   (58% short-user drop, cat fp16-autocast-no-scaler, 50ep overfit). Everything else CLEAN.
3. **Critical advisor on heads + per-task tuning + STAN** → STAN-attention for cat FALSIFIED even tuned
   (−7.4pp); next_lstm tuned ties but never wins. Mechanistic: recurrence wins cat, attention wins reg —
   the frozen ceilings already use the right head per task.
4. **Tier S Prong A + B**: all coded heads + new SSM (`next_mamba`) + SimGCL aux (`next_gru_simgcl`) — none
   beats the incumbent. (INDEX S.1/S.2/S.3/S.4.)
5. **CAT CEILING BUG (caught + fixed)**: `train.py --cat-head` is silently ignored on `--task next` (MTL-track
   only) → the cat ceiling had run `next_single` not `next_gru`. Re-pinned with `--model next_gru` (the +8pp
   AL correction in §2). Two advisor passes confirmed the re-pin sound + added the freeze-sanity guard.
6. **T1.4** (the tuning that set the frozen ceilings): new leak-free loss code `src/losses/calibrated.py`
   (logit-adjust/focal/tail-loss; 19 tests). Full per-task tune → reg α=0 + cat logit-adjust τ=0.5 win.

## 4. The overlap finding — what the next agent MUST understand
Non-overlapping windows (stride=9) train on ~8.5× less data than overlap (stride=1). Validated at AL (real
pipeline): **cat rising-tide** (STL +9.77 / MTL +8.92), **reg gap WIDENS** (STL +5.13 / MTL +0.50; the
STL→MTL reg gap 8.34→12.96). The MTL reg shared-backbone bottleneck *can't absorb the extra data* → this
**STRENGTHENS the regime/dual-tower thesis**. HGI behaves like v14 (+4.89), stays tied. **Decision: keep
non-overlap canon; it's future-work.** All probe code is isolated in engine `check2hgi_dk_ovl` — the
canonical substrate is UNTOUCHED. If ever adopted, see the rebuild checklist in the future-work memo.

## 5. Traps / gotchas (don't repeat these)
- **`--cat-head`/`--reg-head` are MTL-track-only.** For STL `--task next`, use `--model <head>`. Always
  verify the arch that actually ran (`results/.../model/arch.txt`).
- **Sanity-check any STL ceiling ≥ the MTL it bounds** (the cat bug was visible as ceiling < MTL). Use
  `t14_freeze_sanity.py`.
- **`--per-fold-transition-dir` must be the seeded log_T** (`region_transition_log_seed{S}_fold{N}.pt`);
  default log_T leaks ~+3pp. Stale-log_T guard: mtime(log_T) > mtime(next_region.parquet).
- **The registry silently drops unknown kwargs** — a head can ignore a param you think you set.
- **Frozen folds + the moving-baseline guard**: Tier-S/T5 winners feed candidates, NEVER re-open (c)/(d).
- Repo pre-stages unrelated `articles/*` — always `git add` with explicit pathspec + check `git show --stat`.

## 6. New reusable assets (this session)
- `src/losses/calibrated.py` (+ test) — leak-free logit-adjust/focal/balanced/CB/LDAM; wired into
  `next_cv.py` via `ExperimentConfig.loss_calibration` + `p1` + `train.py` flags.
- Heads: `next_mamba` (selective-SSM), `next_gru_simgcl` (SimGCL aux + `model.aux_loss` trainer hook in
  `_single_task_train.py`). Both sound but lose/tie — keep as tested assets, not in use.
- Overlap infra: backward-compatible `stride` param (`core.py`/`builders.py`), engine-aware region-seq
  (`region_sequence.py` + `folds.py` + `p1` `seq_engine`), isolated engine `check2hgi_dk_ovl`,
  `build_overlap_probe_engine.py`, `overlap_probe.py`.
- Scripts: `scripts/mtl_improvement/` — `t14_*` (T1.4 sweep/validate/repin/agg/sanity), `tierS_*`
  (screen/confirm/unit), `stan_for_cat.sh`, `overlap_*`.

## 7. THE NEXT STEP — ⚠ SUPERSEDED (this section described T2.1, now DONE). See §0c (T2P.0).
<s>The clear next is Tier 2 — T2.1 dual-tower.</s> **T2.1 is complete (NEGATIVE); the live next step is the
Tier 2P redirect — §0c (T2P.0 linchpin).** Other still-open/optional items (unchanged): T4.0 loss-scale/RLW
litmus (cheap, ungated); cheap training levers (cat fp32 vs fp16-autocast, shorter 25ep schedule — audit
MED); the deferred overlap/dense-supervision follow-up study (`future_works/overlapping_windows.md`).

## 7b. Audit close-out (O1–O5) — ✅ ALL CLOSED 2026-06-04 (`archaive/AUDIT_TIER1_TIERS_2026-06-03.md §6`)
The 5 audit items are closed + advisor-reviewed (leak audit: NONE). Full write-ups: `TIER01_RESULTS.md
§Audit close-out`. Frozen (c)/(d) UNCHANGED; `t14_freeze_sanity.py` GREEN. Commits `4fba15b` → `b94b29f`
→ `87e3f62`.
- **O1 (α=0 "prior is a drag") — both audit hypotheses FALSIFIED.** A faithful re-run (`o1_alpha_probe.py`;
  reproduces 62.32/70.28/52.87/55.81) shows the learnable α converges **large** (AL +0.45 / AZ +0.79 / GE
  +0.94 / FL +1.09 — larger at higher-coverage states, n=4 suggestive), i.e. the model *leans into* the
  prior, yet prior-ON stays 0.56–3.03pp BELOW α=0. The prior carries real signal (standalone Acc@10 50.86/66.15
  ≈ Markov-1-region floors 47.01/65.05). **Reframed claim: "the fixed additive log_T prior is a net drag on
  the STL-reg ceiling"** — NOT "embeddings subsume transitions," NOT a stuck-α bug; mechanism (train/val gap
  vs additive scale-mismatch vs double-counting) NOT isolated. Strengthens the §2c HGI-prior-artifact corollary.
- **O2 (Tier-S cat crack) — multi-band negative HOLDS.** Multi-seed {0,1,7,100}: next_lstm's single-seed wins
  evaporate → tie at all 4 states. next_single GE +1.54±0.17 (robust) but GE-SPECIFIC (AL −8.11) → fails the
  ≥2-band gate → a **T5.2 candidate** (re-judged under MTL), does NOT re-open (c). NB the per-state GE-cat STL
  ceiling is next_single 59.66 > (c) 58.12; (c) is the scale-robust incumbent, not the per-state max.
- **O3 (FL (c)-cat inversion).** Multi-seed 69.96±0.08 validates seed42 69.97; the −0.30pp inversion vs MTL
  diag-best 70.26 PERSISTS multi-seed (not a seed artifact) but is tiny + explained (oracle epoch + small FL
  cat transfer); (c) validly bounds the *deployable* MTL cat (≫66.73). Not a bug. CAT-side, orthogonal to T2.
- **O4** next_hybrid accounted (AL cat 49.34 < floor; reporting omission) + `*_hsm` deferral noted.
- **O5** paper limitation (vi) (non-overlap windows + AL rebuttal) added to `PAPER_DRAFT.md §7`; dense-rebuild
  deferred to `future_works/overlapping_windows.md`.

## 8. How to resume
1. Read this (esp. §0, §0c) → `log.md` (the 2026-06-04 entries incl. the REDIRECT) → `INDEX.html` §Tier 2P
   (`#tier2p`) + the Tier-2 final-decision callout → `TIER01_RESULTS.md`.
2. `git pull`; confirm `t14_freeze_sanity.py` is GREEN.
3. **Run §0c (T2P.0 linchpin) FIRST**, then T2.3/T2.4 (§0b) confirmatory.
4. STOP + surface at the tier boundary (advisor pass → summary → user decision).

## 9. ⚠ SUPERSEDED — T2.1 onboarding (kept for the gates/yardstick reference; T2.1 itself is DONE/NEGATIVE)
**This section onboarded T2.1 (now complete — see §0). Its hard-gate discipline + the FROZEN yardstick table
below still apply to Tier 2P; the "one experiment: T2.1" framing is obsolete (T2.1 lost; the live work is §0c).**
The headline (the regime finding) is confirmed at AL/AZ/GE/FL (`v14_mtl_vs_canonical.md`): v14 ≈ matched
canonical in MTL — the STL substrate gains wash out jointly. **The locus is the joint-training architecture, not
the substrate or the per-task head** (Tier-S proved the head is not the lever; T1.3 proved the upstream encoder
is not the residual). Tier 2 attacks that locus.

**The one experiment: T2.1 — reg-private dual-tower** (INDEX `#tier2`). Build a reg-private full-STAN backbone
(the §6.4 decomposition says ~75% of the MTL→STL reg residual is the *missing private backbone*) so the reg head
stops sharing the cross-attn/shared trunk with cat. Primary arm = gated-fusion (b); + a PCGrad-off arm.

**Hard gates BEFORE any multi-fold launch (do NOT skip — these are why prior arch swaps collapsed):**
1. **Unit-test gate** (hard rule 10): forward/backward shapes on a synthetic 100-user batch, loss-finite,
   param-count within ~5% of B9 at D=256, and `shared/cat_specific/reg_specific_parameters()` partition
   bijective+exhaustive — **the dual-tower's private backbone is a NEW param group; wire it into the partition**
   (a silent omission here = the F49 class of bug).
2. **Per-arch LR mini-sweep** (hard rule 7): 5 regimes × 5f × 40ep × seed42 × AL+AZ, then full-protocol at the
   winner. (The B9_STL_STAN_SWAP collapse = B9 recipe blindly applied to a non-α head — don't repeat it.)
3. Stay at `shared_layer_size=256` (F51 widening falsified). No fclass-as-feature. log_T-KD ON, seeded per-fold
   log_T mandatory.

**Design discipline:** frozen-fold paired (hard rule 2b) — score **Δ vs the frozen (c)/(d)**, not bare absolutes.
Run the regime×substrate 2×2: {v14-fresh, canonical-fresh `gcn_ctrl`} × {B9, dual-tower}. HGI sanity probe per
promoted arch (2 seeds × AL+AZ × 5f × 30ep; escalate if |MTL+HGI − STL+HGI| ≥ 2pp).

**The yardstick you measure against (FROZEN — do NOT recompute or re-pin; see §2):**
| | AL | AZ | GE | FL |
|---|---|---|---|---|
| (c) cat macro-F1 | 49.97 | 51.01 | 58.12 | 69.97 |
| (c) reg Acc@10 | 62.88 | 55.11 | 58.45 | 73.31 |
| (d) composite reg | 63.58 | 55.11 | 58.76 | 73.62 |
| MTL deployable reg (v14, the (a) baseline T2 must beat) | 50.14 | 37.78 | 42.64 | 61.21 |

The composite (d) beats single-model MTL reg by **+12.4 to +17.3pp** — that is the gap the dual-tower must close
*inside one model*. If nothing recovers a meaningful fraction → composite is the deploy fallback (a paper-grade
negative). **Do NOT add AdaShare/Learning-to-Branch** (collapse to branch-depth at 2 tasks, already spanned).

**Carry-overs from the close-out into Tier 2/5:** (i) the O1 reframe — log_T is a KD loss in MTL (helps) vs an
additive bias in STL (hurts); T3.1 will re-sweep log_T-KD on the new stack, so do not assume the prior behaves
the same. (ii) `next_single@GE` is a logged T5.2 cat candidate (state-conditional; re-judge under MTL, do not
auto-pick). (iii) Tier S is an OPEN sandbox running parallel to Tiers 2-4 (must not starve the regime headline).

**Files to read for the build:** `docs/findings/B9_STL_STAN_SWAP_AZ_FL.md §6.4` (gap decomposition + the residual-skip falsification),
`future_works/mtl_architecture_revisit.md`, `src/models/mtl/mtlnet_crossattn/model.py` (current backbone),
`src/models/mtl/mtlnet_crossstitch/model.py` (scaffolded for T2.2), `src/training/runners/mtl_cv.py` +
`src/training/helpers.py` (`setup_per_head_optimizer`). Drivers template: `scripts/_v14_run/` (currently SERIAL —
that's the parallelization headroom; MPS-collocate small states per AGENT_PROMPT §14).
