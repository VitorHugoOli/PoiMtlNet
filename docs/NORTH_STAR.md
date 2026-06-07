# North-Star MTL Configuration

> ⭐⭐⭐ **2026-06-06 — CEILING BROKEN: the FL MTL champion is a SINGLE model that beats both STL ceilings.** Building on the C25 fix (banner below), the `mtl_improvement` combo screen found **(G) `dual aux + prior-OFF`**: `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (`raw_embed_dim=64 fusion_mode=aux freeze_alpha=True alpha_init=0.0`), v14 substrate, unweighted onecycle KD-OFF. **CONFIRMED MULTI-STATE @ 4 seeds {0,1,7,100} — MATCHES the STL reg ceiling + BEATS the cat ceiling (+3pp) at ALL 4 states: AL reg 64.47±0.11/cat 52.91 (+2.94); AZ 55.75±0.21/54.48 (+3.47); GE 59.37±0.04/61.43 (+3.31); FL 73.57±0.06/73.16 (+3.19).** ⚠ **REG CORRECTED 2026-06-07 (B-A2):** the reg "+Δ vs ceiling" used G's in-distribution Acc@10 vs the (c) ceiling's FULL `top10_acc`; on a matched metric G is ~0.35pp BELOW the (c) reg ceiling (FL 72.93 vs 73.31) → reg = "matches" (Pareto-non-inferior), NOT "beats". Cat +Δ is exact. ⚠ **G′ (cat-private, FL-ONLY — DEMOTED 2026-06-07):** giving the CAT head a private tower too (both-private dual-tower, `mtlnet_crossattn_dualtower_catpriv`) gained cat at FL only (74.77, +1.61); the multi-state confirm (AL/AZ/GE × 4 seeds) **FALSIFIED** it — cat CRATERS at small states (AL 37.66 = −15.25 vs G, AZ −12.45, GE −3.59; reg flat). The cat-private tower **UNDERFITS** small-state cat (NOT overfit — the off-label STAN-flow head is over-regularized: AL train-F1 caps ~0.45 vs the GRU head's 0.98; a rescue screen of lower dropout / softer LR / smaller tower is testing recoverability) → G′ is (for now) an FL-only experimental variant, **NOT a champion**; **G (cat-SHARED) remains the multi-state champion**. See CHAMPION.md / INDEX `#T2V-5`. FL also TIES the (d) 2-model composite reg (73.62) while winning cat. **The MTL tradeoff is INVERTED at every state — a single joint model Pareto-dominates single-task AND the composite (same/better reg, better cat, half the params).** Mechanism: `aux` fusion adds the shared pathway WITHOUT diluting the private reg tower (vs `gated` competition); prior-OFF removes the biased α·log_T logit term. Architecture capacity is NOT the lever (MoE, SwiGLU, MulT, crossstitch→crossattn all null on reg — falsified 5 ways). **Status: multi-state paper-grade** (4/4 available states, 4-seed) **+ VALIDATED 2026-06-07 (Tier 2V).** A skeptical critique close-out (`studies/mtl_improvement/CRITIQUE_TIER2_C25_2026-06-06.md` §7) confirmed it on every axis: seed-matched (c)/(d) ceilings (G still beats both 4/4), FAIR alt-arch re-rank (all lose 1.6–2.1pp → un-confounded), no tail regression, no hypertuning lever beats G (plain CE is the MTL cat optimum), param-honest (+4.9%, one model). **CA/TX deferred** to future-work (`future_works/mtl_improvement_catx_scale_conditional.md`). The paper §0 canon is NOT yet re-run — only the BRACIS paper-doc restatement remains (author decision). Full trail: `studies/mtl_improvement/{log.md 2026-06-06/07, HANDOFF.md, PAPER_UPDATE.md, INDEX.html #tier2v, CHAMPION.md}`; drivers `scripts/mtl_improvement/{c25_combos_{screen,promote},c25_g_multistate,t2v1_ceilings_multiseed,t2v4_altarch_rerank,t2v567_hypertune}.sh`.

> ⭐⭐⭐ **2026-06-05 — C25 UNWEIGHTING FIX (a recipe WIN) + the 2026-06-04 Tier-2 NEGATIVE below is OVERTURNED.** The MTL heads were silently trained on **class-weighted CE** (`default_mtl use_class_weights=True`) while the reported metrics (reg Acc@10, the STL ceiling) are **unweighted** — depressing MTL reg ~10-14pp and cat ~3-5pp. **Adopt the fix: BOTH heads UNWEIGHTED** (per-task `use_class_weights_{reg,cat}=False`, now the `default_mtl` default; reproduce pre-C25 with `--reg-class-weights --cat-class-weights`). Re-validated multi-seed {0,1,7,100} AL/GE/FL: **MTL reg ≈/> the STL ceiling AND the composite; the substrate gain transfers to MTL; the dual-tower CLOSES the FL gap; cat exceeds its ceiling.** So the **2026-06-04 banner below ("architecture NEGATIVE; composite remains the deployable reg answer; irreducibly architectural") is SUPERSEDED** — a single jointly-trained MTL model matches/beats the composite once the loss matches the metric. Full reframe: `studies/mtl_improvement/{HANDOFF.md,PAPER_UPDATE.md,log.md}` + `CONCERNS.md §C25`. (Re-validation used onecycle; FL-B9 §0.1-continuity follow-up pending.)

> 🔬 **2026-05-30 — v11→v12 default flip.** [`studies/substrate-protocol-cleanup/`](studies/substrate-protocol-cleanup/) **CLOSED 2026-05-29** and produced the **v12 default** = the recipe below + **log_T-KD W=0.2** (deployable reg lift; paper-grade AL/AZ, pilot FL/CA/TX) + **ResLN encoder** (STL-best, no MTL benefit). The committed champion recipe below is unchanged in structure; v12 stacks log_T-KD on it as the new default. The **BRACIS paper cites v11** (reproduce: `--log-t-kd-weight 0.0 --encoder gcn`, frozen `output/check2hgi/`). Version registry: [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md). **Recommended STL / forward-MTL base (v13, opt-in, blessed 2026-05-30):** engine **`check2hgi_resln_design_b`** (ResLN + POI2Vec@pool) — the best STL dual-axis engine (equalises HGI reg at AL, keeps/widens cat 2–3× over HGI); **STL-only, NO MTL benefit today** — it is the strongest *base* for future MTL work once `mtl_improvement` fixes the joint-training regime. Built at all five states (AL/AZ/FL/CA/TX) as of 2026-05-30; see CANONICAL_VERSIONS §v13. **⭐ 2026-06-02 — v14 SUPERSEDES v13 as the recommended STL / forward-MTL base** (`embedding_eval` Part-1 CLOSED): engine **`check2hgi_design_k_resln_mae_l0_1`** (ResLN+mae cat lever ⊕ Delaunay-POI-GCN reg lever, orthogonal stack). Leak-free multi-seed FL: next-cat 67.36 (≈ frozen-canon ≫ HGI) + next-reg 0.7024 (closes ~69% of the canon→HGI gap; HGI keeps a −0.36pp edge). **Same opt-in posture; STL-only, NO MTL benefit (pilot) — the MTL cross-attn regime is the wall.** Mechanisms graduated into `Check2HGIModule` (`reg_poi_mode`); canonical `check2hgi` untouched. See CANONICAL_VERSIONS §v14 + [`studies/embedding_eval/FINAL_SYNTHESIS.md`](studies/embedding_eval/FINAL_SYNTHESIS.md). [`studies/mtl_improvement/`](studies/mtl_improvement/) (architectural axis, branch `mtl-improve`) remains **ACTIVE**; when it lands a champion the §0 pipeline re-runs. Known v11 follow-on results:
> - **C21 RESOLVED** — F1 selector fix (`joint_geom_simple`) lands +5.6 pp deployable lift at FL multi-seed (`mtl-protocol-fix` v6 final 2026-05-24).
> - **Composite ceiling on reg** (Phase 3 §4.2 ESTABLISHED) — STL c2hgi-cat + STL HGI-reg routed by task at deploy = +7 to +12 pp vs MTL@disjoint at every state. Current project headline on the reg axis. See [`future_works/composite_two_substrate_engine.md`](future_works/composite_two_substrate_engine.md).
> - **log_T-KD** (Phase 3 §4.5; `substrate-protocol-cleanup` Tier A1 CLOSED 2026-05-29; **NOW v12 DEFAULT 2026-05-30**) — **PROMOTED multi-seed n=20 at small states** (AL +2.27 / AZ +4.91 pp disjoint reg, p=9.54e-07, leak-audited clean), large-state pilot TRANSFERS (FL/CA/TX, seed=42, NOT paper-grade). This is a **validated small-state reg lift orthogonal to the B9 recipe** — as of the **v12 default flip it is ON by default** (`--log-t-kd-weight 0.2 --log-t-kd-tau 1.0`, scoped to MTL `check2hgi_next_region`); pass `--log-t-kd-weight 0.0` for the **v11 paper-canon** (no-KD) reproduction. It stacks on B9 and does NOT change the architectural champion (owned by `mtl_improvement`). It is the isolated single-MTL-artefact lift, distinct from and smaller than the §4.2 composite (+7–12 pp). See [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md) (v11 vs v12), [`results/RESULTS_TABLE.md §0.8`](results/RESULTS_TABLE.md), [`findings/F_TIER_A1_PROMOTION.md`](findings/F_TIER_A1_PROMOTION.md).
> - **ResLN encoder** (`canonical_improvement` T3.2; **NOW v12 DEFAULT encoder 2026-05-30**) — `ResidualLNEncoder` is the **best STL cat encoder** (+0.86 FL / +1.48 AL / +1.70 AZ cat F1, 5/5 seeds, p=0.03125) and is now the default for FUTURE Check2HGI builds (`scripts/canonical_improvement/regen_emb_t3.py` defaults to `--encoder resln`). **CRITICAL HONESTY: ResLN gives NO MTL benefit** — under the cross-attn MTL joint-training regime the substrate axis is washed out (the regime finding below). Its value is **STL / representation-quality / generality**, never MTL reg or cat. The frozen v11 paper substrate (`output/check2hgi/<state>/`, GCN) was **NOT** rebuilt — pass/rebuild `--encoder gcn` for v11. See [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md).
> - **The regime finding (central)** (`substrate-protocol-cleanup` CLOSURE) — the cross-attn MTL joint-training regime washes out encoder/substrate improvements on BOTH axes. Even **HGI** (the STL reg winner, +2.12 pp STL at FL) is ≈ canonical in MTL reg (Δ+0.51, p=0.41 NS); the four design substrates (B/J/L/M) close 0 % of any MTL gap; the STL-α=0 (~73 % Acc@10) vs MTL-α=0 (~0.03 %) isolation cell shows it is the **regime, not the head/substrate, that kills the MTL reg encoder**. **Only prior-pathway work (log_T-KD) moves MTL reg.** The MTL reg bottleneck is **architectural** → handed to `mtl_improvement`. The dual-axis STL champions are **ResLN+design_b** (general) / **ResLN+design_j** (AL-specific), but these are **STL-only research variants** (kept registered/opt-in, NOT defaulted). **Deployable conclusion: ship canonical + log_T-KD; substrate/encoder = STL/generality only.**
> - **P4 verdict** — residual MTL-vs-STL reg gap is architectural (not cat-interference, not long-tail, not substrate). `mtl_improvement` T2 owns the fix.
>
> 🏁 **2026-06-04 — `mtl_improvement` Tier 2 COMPLETE (branch `mtl-improve`).** Two results. **(1) Architecture NEGATIVE (multi-seed-hardened):** no single-model MTL architecture closes the MTL→STL reg gap — the reg-private dual-tower LOSES to the matched baseline (FL multi-seed −3.35), a 5-point sharing dose-response shows more-sharing-helps-reg (refutes the §6.4 private-backbone hypothesis), and 3 mechanism cells localize the gap to the **joint cross-attn harness itself** (not interference/prior/wd). **→ the composite remains the deployable reg answer; the gap is irreducibly architectural.** The architecture axis is now exhausted (dual-tower + CrossStitch[weak +1pp partial, not a closer] + hard-share ≈ soft-share). **(2) Recipe WIN — `onecycle` (aggressive schedule, NO alt-opt) is the new recommended SMALL-STATE recipe** (AL/AZ): dominates H3-alt (v14 multi-seed +6–9pp reg / +1–2pp cat) and beats B9 on the v11 paper substrate (AL reg +2.98 / cat +7.36; AZ reg +0.76 / cat +4.69). **alt-opt flips sign by scale** → **keep B9 at large states** (FL/CA: onecycle doesn't dominate). **§0.1 small-state arch-Δ should be re-stated under the deployable recipe** (annotation in `results/RESULTS_TABLE.md §0.1`; author sign-off needed — it reshapes a central claim). Close-out: [`studies/mtl_improvement/PAPER_UPDATE.md`](studies/mtl_improvement/PAPER_UPDATE.md).

> ⚠ **2026-05-02 v10 CURRENT NORTH STAR — use `results/RESULTS_TABLE.md §0` as the only paper-canonical numerical source.**
> B9 is paper-grade at **FL/CA/TX**; H3-alt remains the small-state recipe at **AL/AZ**.
> The paper story is **substrate task-asymmetry first, classic MTL tradeoff second**.
> Historical derivation remains below; when a lower section disagrees with the current tables here, trust `RESULTS_TABLE.md §0`.
>
> **Current recipe-selection headline (v9/v10 canonical):**
>
> | State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat | Verdict |
> |---|---:|---:|---:|---:|---:|---|
> | AL | 20 | **−0.35** | **1.9e-03** | **−2.22** | **1.9e-06** | **H3-alt > B9 on cat; reg tied** |
> | AZ | 20 | −0.09 | 0.23 (n.s.) | **−0.96** | **7.1e-04** | **H3-alt > B9 on cat; reg tied** |
> | FL | 25 | **+3.48** | **3.0e-08** | +0.42 | 1.3e-05 | **B9 > H3-alt on both** |
> | CA | 20 | **+4.18** | **<1e-04** | **+0.51** | **<1e-04** | **B9 > H3-alt on both** |
> | TX | 20 | **+1.87** | **7.0e-04** | **+0.52** | **2.0e-04** | **B9 > H3-alt on both** |
>
> **Current architectural-Δ headline (v10 canonical):**
>
> | State | Δ_reg pp (MTL−STL) | p_reg | Δ_cat pp (MTL−STL) | p_cat |
> |---|---:|---:|---:|---:|
> | AL (n=20) | **−11.04** | **1.9e-06** | **−0.78** (small-significantly negative) | **0.036** |
> | AZ (n=20) | **−12.27** | **1.9e-06** | **+1.20** | **<1e-04** |
> | FL (n=20) | **−7.34** | **1.9e-06** | **+1.40** | **2e-06** |
> | CA (n=20) | **−9.50** | **2e-06** | **+1.68** | **2e-06** |
> | TX (n=20) | **−16.59** | **2e-06** | **+1.89** | **2e-06** |
>
> **Current reading:** MTL is positive on cat at four of five states, small-significantly negative at AL, and sign-consistently negative on reg at all five states. Recipe selection is scale-conditional; architectural cost varies non-monotonically across states.
>
> Background provenance only: `archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md`.
>
> ---
>
> ⚠ **2026-05-01 SCALE-CONDITIONAL CHAMPION FINDING — historical banner preserved below.**
> Cross-state B9 vs H3-alt comparison (28 paper-closure runs + 8 AL/AZ H3-alt
> gap-fill = 36 total) reveals B9's recipe lift is FL-scale-specific:
>
> | State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat | Verdict |
> |---|---:|---:|---:|---:|---:|---|
> | AL | 20 | **−0.35** | **1.9e-03** | **−2.22** | **1.9e-06** | **H3-alt > B9 on cat** |
> | AZ | 20 | −0.09 | 0.23 (n.s.) | **−0.96** | **7.1e-04** | **H3-alt > B9 on cat** |
> | FL | 25 | **+3.48** | **3.0e-08** | +0.42 | 1.3e-05 | B9 > H3-alt (F51) |
> | CA | 20 | **+4.18** | **<1e-04** | **+0.51** | **<1e-04** | **B9 > H3-alt on both** |
> | TX | 20 | **+1.87** | **7.0e-04** | **+0.52** | **2.0e-04** | **B9 > H3-alt on both** |
>
> **B9's three additions over H3-alt (alt-SGD + cosine + α-no-WD) help on FL but
> hurt cat at small states (AL/AZ).** Mechanism hypothesis: the additions target
> FL's reg-saturation problem (D5 finding); at smaller transition graphs the
> reg saturation is less severe AND alt-SGD's per-step temporal gradient separation
> costs cat-side signal that small states can't afford to lose. The recipe-selection
> claim reframes from "B9 is the universal champion" to **"B9 is FL-scale champion;
> H3-alt is the universal recipe at small scale; the optimal MTL recipe is
> scale-conditional"**. Full doc: `archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md §4a-bis` (background provenance) or `results/RESULTS_TABLE.md §0.4` (v10 canonical).
> Wilcoxon JSON: `research/PAPER_CLOSURE_RECIPE_WILCOXON.json`.

> 🎯 **PAPER CLOSURE — historical 2026-05-01 banner; current canon is v10 above.**
> Cross-state P3 (CA + TX), STL ceilings at all 5 states with multi-seed at AL/AZ/FL,
> AL/AZ MTL B9 multi-seed, FL STL reg multi-seed extension. The architectural-Δ
> picture is now multi-seed at AL+AZ, multi-seed STL + single-seed B9 at FL,
> single-seed at CA+TX (P1 multi-seed extension deferred to camera-ready).
> Full results: **`results/RESULTS_TABLE.md §0` (v10, canonical)**; background
> provenance in `archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md`.
> Wilcoxon JSONs: `research/PAPER_CLOSURE_WILCOXON.json`,
> `research/GAP_FILL_WILCOXON.json` (v8 cat-Δ).
>
> **Headline (historical banner, superseded where it disagrees with v10 current canon above):**
> | State | Δ_reg pp (MTL−STL) | p_reg | Δ_cat pp (MTL−STL) | p_cat |
> |---|---:|---:|---:|---:|
> | AL (n=20) | **−11.04** | **1.9e-06** | **−0.78** (small-significantly negative) | **0.036** |
> | AZ (n=20) | **−12.27** | **1.9e-06** | **+1.20** | **<1e-04** |
> | FL (n=20) | **−7.34** | **1.9e-06** | **+1.40** | **2e-06** |
> | CA (n=20) | **−9.50** | **2e-06** | **+1.68** | **2e-06** |
> | TX (n=20) | **−16.59** | **2e-06** | **+1.89** | **2e-06** |
>
> **v8 update (2026-05-01):** cat-Δ Wilcoxon landed for AL/AZ/FL via `gap_fill_wilcoxon.py` →
> `research/GAP_FILL_WILCOXON.json`. AL is now **small-significantly negative** at p=0.036
> across n=20 multi-seed fold-pairs (14/20 fold-pairs negative; magnitude small at
> ~1.9% relative on a 41% F1 scale). FL Δ_cat refined from +1.43 (mean-diff) to
> **+1.52** (paired Δ); FL MTL B9 cat F1 refined from 68.59 → **68.51 ± 0.51**
> (multi-seed pooled). CA recipe-selection upgraded to n=20 multi-seed, paper-grade
> on both tasks (see §0.4 below).
>
> n=20 = 4 seeds × 5 folds (pooled multi-seed). n=5 = single seed paired.
> p=0.0625 is the minimum for n=5 paired Wilcoxon — single-seed numbers are
> signed-consistent (5/5 in the claimed direction) but not formally significant.
> Reg metric: per-fold max `top10_acc_indist` for epoch ≥ 5 (F51 canonical).
> Cat metric: per-fold max unweighted `f1` for epoch ≥ 5.
>
> **Reg:** MTL B9 < STL `next_getnext_hard` at every state by 7-17 pp.
> **Cat:** MTL > STL `next_gru` at four of five states. AL is the only state where
> the cat delta is significantly negative, but small in magnitude.
>
> **Reframe vs F49:** F49's "AL +6.48 pp MTL>STL on reg" was a leak artifact of
> pre-F50 measurements (full-data `region_transition_log.pt`, leaks ~13-27 pp).
> Under leak-free symmetric comparison, AL's reg pattern matches every other state.
> The headline "scale-conditional architecture-dominant state" framing in F49 is
> superseded; the leak-free framing is **classic MTL tradeoff: hard task pays,
> easy task gains**.
>
> 🎉 **F51 MULTI-SEED (2026-04-30):** B9 vs H3-alt validated across 5 seeds {42, 0, 1, 7, 100}. **Δreg = +3.48 ± 0.12 pp across seeds; pooled paired Wilcoxon (5 × 5 = 25 fold-pairs): p_reg = 2.98×10⁻⁸ (25/25 positive); p_cat = 1.33×10⁻⁵ (19/25 positive).** Cat reaches paper-grade once seeds pool. Absolute B9 reg σ_across_seeds = 0.11 pp — recipe is essentially deterministic in the partition-difficulty axis. The seed=42 +3.34 pp number was the worst-case seed; cross-seed mean is slightly larger. Full doc: `research/F51_MULTI_SEED_FINDINGS.md`.
>
> 📊 **F51 TIER 2 CAPACITY SWEEP (2026-04-30):** 21 capacity smokes across 7 dimensions (5f×30ep) confirm **B9 is locally optimal in 5/7 capacity dimensions**. No paper-grade lift available via capacity scaling. Two NEW negative findings: (a) **cat width-stability cliff** — wider shared backbone (`shared_layer_size` 384/512, `num_crossattn_blocks=4`, `crossattn_ffn_dim=1024`) breaks cat without affecting reg; (b) **F52's "mixing is dead at FL" is depth-conditional** — alive at `num_crossattn_blocks=3` (Pareto-trade: +0.75 reg / -2.62 cat), breaks cat at depth=4. Full doc: `research/F51_TIER2_CAPACITY_FINDINGS.md`.
>
> ⚠ **PER-SEED log_T LEAK (caught + fixed 2026-04-30 mid-F51-sweep):** the original C4 fix wrote per-fold log_T as `region_transition_log_fold{N}.pt` with no seed in the filename, but the trainer loaded that file regardless of its own `--seed`. At any seed != 42, ~80% of val users live in seed=42's fold-N TRAIN set → ~80% of val transitions leaked back into the prior, inflating absolute reg by ~9 pp. Fix: filename is now `region_transition_log_seed{S}_fold{N}.pt`; trainer hard-fails if missing or if a legacy unseeded file is present. Paired Δs from earlier runs survive (uniform-leak property — both arms read the same wrong prior on the same val set), but absolute numbers from the v1 multi-seed sweep are wrong; v2 (clean) is in `F51_MULTI_SEED_FINDINGS.md`.
>
> ⚠ **C4 LEAKAGE CAVEAT (added 2026-04-29 19:50, F50 T4):** All absolute `next_region` numbers below were measured under the legacy full-data `region_transition_log.pt` graph prior, which leaked val transitions into training. Direct measurement: ~13-17 pp inflation at convergence, propagating through 5 heads (`next_getnext_hard*`, `next_getnext`, `next_tgstan`, `next_stahyper`, `next_getnext_hard_hsm`). **Use `--per-fold-transition-dir` for any future run.** Under leak-free conditions, the committed champion is **B9 (P4 + Cosine + alpha-no-WD)**, headline numbers in the F51 banner above. See `research/F50_T4_C4_LEAK_DIAGNOSIS.md` and `research/F50_T4_BROADER_LEAKAGE_AUDIT.md`. The numbers below are KEPT for historical comparison and method derivation; the absolute targets (e.g. "73.61", "76.07") need a "−15 pp footnote" mental model when read.

**Status (2026-04-29 17:30 UTC, Pareto-corrected):** Champion is **P4 alternating-SGD + Cosine (max_lr=3e-3) + delayed-min selector (`min_epoch=10`)**. Earlier today P4+OneCycle was promoted as champion based on reg-only metrics; closer inspection of the cat-side data shows OneCycle DEGRADES cat F1 by −1.84 pp with one fold collapsing to 62.68 (vs 67-68 in others). **P4+Cosine is the Pareto-dominant variant**: reg +4.63 pp paper-grade (paired Wilcoxon p=0.0312, 5/5 positive), cat tied/slightly improved (+0.15 pp, no fold collapse). P4-alone is also Pareto-dominant (+4.04 reg, cat tied) but P4+Cosine is +0.59 pp stronger on reg.

P4+OneCycle (+6.08 reg / −1.84 cat) is documented as the **reg-only-optimal alternative** for ablation studies that don't constrain cat preservation. See `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §6.3-§6.4`.

**Status (2026-04-27):** Two complementary tracks now confirm the MTL story from different angles. **The previous recipe (F48-H3-alt per-head LR) is the committed champion**; substrate validation and architecture attribution both back it.

**Architecture-side (F48-H3-alt + F49, 2026-04-26 → 04-27):** Per-head LR recipe validated 5-fold on AL/AZ/FL — cat preserved within ~2 pp of B3, reg Acc@10 lifts by 6.7-15 pp over B3. AL **exceeds** STL F21c ceiling by +6.25 pp; AZ closes 75%; FL is most stable (σ=0.68). Three orthogonal negative controls (F40, F48-H1, F48-H2) bracket H3-alt as the unique design. **F49 attribution (2026-04-27):** the H3-alt reg lift on AL is *purely architectural* (+6.48 pp from architecture alone, F49c 5f × 50ep); cat-supervision transfer is null on all 3 states (≤|0.75| pp), refuting the legacy "+14.2 pp transfer" claim by ≥9σ on FL n=5. CH18 Tier A; CH19 Tier A. See `research/F48_H3_PER_HEAD_LR_FINDINGS.md` + `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`.

**Substrate-side (Phase 1 substrate validation, 2026-04-27):** Five-leg study on AL+AZ confirms the substrate side of the MTL claim:

1. **CH16 confirmed under matched-head, head-invariant** at AL+AZ (8/8 head-state probes positive, p=0.0312 each; ranges +11.58 to +15.50 pp).
2. **CH15 reframed** — under the matched MTL reg head (`next_getnext_hard`), C2HGI ≥ HGI (was "HGI > C2HGI" under STAN). The previous CH15 was head-coupled.
3. **CH18 — MTL B3 is substrate-specific.** Substituting HGI breaks the joint signal (cat −17 pp, reg −30 pp Acc@10_indist at both states; MTL+HGI is *worse than STL+HGI* on reg by ~37 pp).

These findings **do not** change the committed config — they explain *why* it works. See `research/SUBSTRATE_COMPARISON_FINDINGS.md` for the full Phase 1 verdict + `PHASE2_TRACKER.md` for FL/CA/TX replication queue.

**Status (2026-04-24):** Cat head refined via F27 from `NextHeadMTL` (Transformer) → `next_gru` (GRU). Paper-reshaping F21c finding noted in §§Caveats. See §Committed config below.

> **⚠ B9 joint-selector bug (added 2026-05-19, applies to the canonical shipping recipe AS-IS).** The `selector` row below describes the per-task best tracker. The **primary checkpoint** is selected by `joint_score = 0.5 * (cat_macro_f1 + reg_macro_f1)` at `src/training/runners/mtl_cv.py:679`. This formula is **structurally broken on the canonical Check2HGI MTL setup itself** — `reg_macro_f1` over ~4 700 sparse FL regions is dominated by rare-class noise (stays ~16-18 % across full ep=1-50 trajectory) and is blind to `reg_top10_acc_indist`'s peak-and-collapse trajectory.
>
> Matched-protocol measurement on canonical shipping FL ep=50 single-seed=42 n=5 (NO substrate changes):
>
> | Selector | cat F1 | reg top10 |
> |---|---:|---:|
> | Per-task disjoint best | 70.49 ± 0.86 | **76.12 ± 0.33** |
> | `joint_canonical_b9` (production — what §0.1 reports) | 69.99 ± 1.13 | **65.38 ± 9.10** |
>
> **Capacity gap: ~10.7 pp reg-top10 thrown away by the production selector**, on the shipping recipe itself, with no substrate change. §0.1 multi-seed reg = 63.27 ± 0.10 matches the matched-protocol `joint_canonical_b9` value within single-seed variance — confirming §0.1 reports joint-best, not the substrate's reg-best capacity.
>
> See `CONCERNS.md` C21 (diagnosis) and the active **[`docs/studies/mtl-protocol-fix/`](studies/mtl-protocol-fix/)** study (Phase 1 = F1 selector fix + three-frontier evaluation across all 5 states) for the closure path. The predecessor memo at `docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md` is superseded by the active study. Until F1 lands, **reg-side conclusions drawn from §0.1 numbers under-report the substrate's actual reg capacity at FL by ~10 pp**. Substrate-axis orderings under the production selector are unreliable — they reflect the selector's choice of destabilised epoch, not the substrate's reg potential.

## Champion — F50 B9 (P4 + Cosine + α-no-WD) — multi-seed validated (2026-04-30 F51)

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)
task_a head (cat)    : next_gru
task_b head (reg)    : next_getnext_hard                # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, batch=2048, 50 epochs
                       seeds {42, 0, 1, 7, 100} all paper-grade ✅
LR scheduler         : Cosine(max_lr=3e-3)              # decay from peak
LR per param group   : cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3
optimizer step       : ALTERNATING per-batch (P4) — cat batch then reg batch, separate optimizer.step()
selector             : per-fold-best top10_acc_indist with --min-best-epoch 5
α-no-WD              : alpha scalar peeled out of AdamW weight_decay group (B9 refinement)
per-fold log_T       : MUST be seed-tagged: region_transition_log_seed{S}_fold{N}.pt
                       built via: scripts/compute_region_transition.py --state STATE --per-fold --seed S
```

### ⚠ Full canonical CLI invocation (use this verbatim; do NOT rely on defaults)

> Three `scripts/train.py` defaults are silently wrong for this recipe and each one alone drops the corresponding head by 10–30 pp on AL/AZ (verified 2026-05-14, A40):
> - `--mtl-loss` defaults to `nash_mtl` → must override to `static_weight` (with `--category-weight 0.75`).
> - `--cat-head` / `--reg-head` default to the preset values (`next_mtl`/`next_gru`) → must override to `next_gru`/`next_getnext_hard`.
> - `--task-b-input-type` defaults to `checkin` → must override to `region` (the reg head consumes region-sequence input). Cat head correctly defaults to `checkin`.

```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state {state} --engine check2hgi --seed 42 \
    --epochs 50 --folds 5 --batch-size 2048 \
    --model mtlnet_crossattn \
    --mtl-loss static_weight --category-weight 0.75 \
    --scheduler cosine --max-lr 3e-3 \
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --task-a-input-type checkin --task-b-input-type region \
    --per-fold-transition-dir output/check2hgi/{state}
```

A40 reproduces canonical §0.1 numbers within σ at AL (cat 40.57, reg 49.92) and AZ (cat 45.14, reg 40.69) with this exact invocation in ~3 min per state.

> 🔁 **v12 DEFAULT FLIP (2026-05-30) — the invocation above is now v12+ unless you add `--log-t-kd-weight 0.0`.** As of 2026-05-30 the code defaults `--log-t-kd-weight 0.2 --log-t-kd-tau 1.0` ON for `--task mtl --task-set check2hgi_next_region` (the validated Tier A1 reg lift). The block above (without `--log-t-kd-weight`) therefore now runs **v12** (canonical recipe + log_T-KD on the GCN substrate). **To reproduce the v11 BRACIS paper §0.1 numbers (no-KD), add `--log-t-kd-weight 0.0`.** The encoder default also flipped to ResLN for FUTURE builds, but the on-disk `output/check2hgi/<state>/` substrate is still the frozen v11 GCN artifact (not rebuilt). Full reproduction map: [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md).

**Single-line additive recipe vs H3-alt** (small-state recipe at AL/AZ — see §0.4 RESULTS_TABLE):
```bash
--alternating-optimizer-step \
--scheduler cosine --max-lr 3e-3 \
--alpha-no-weight-decay \
--min-best-epoch 5 \
--per-fold-transition-dir output/check2hgi/STATE
```

**Multi-seed headline (FL 5f×50ep, leak-free per-seed log_T, ≥ep5):**

| seed | B9 reg ± σ | H3-alt reg ± σ | Δreg | p_reg | n+/n |
|---:|---:|---:|---:|:---:|:---:|
| 42 | 63.47 ± 0.75 | 60.12 ± 1.15 | **+3.34** | 0.0312 | 5/5 |
| 0 | 63.24 ± 0.89 | 59.58 ± 0.95 | **+3.65** | 0.0312 | 5/5 |
| 1 | 63.41 ± 1.16 | 60.02 ± 1.03 | **+3.39** | 0.0312 | 5/5 |
| 7 | 63.21 ± 0.50 | 59.72 ± 0.54 | **+3.49** | 0.0312 | 5/5 |
| 100 | 63.38 ± 0.93 | 59.87 ± 1.17 | **+3.51** | 0.0312 | 5/5 |
| **mean** | **63.34 ± 0.11** (across seeds) | 59.86 ± 0.22 | **+3.48 ± 0.12** | — | — |

**Pooled paired Wilcoxon (25 fold-pairs):** Δreg = +3.48 pp, **p = 2.98×10⁻⁸**, 25/25 positive. Δcat = +0.42 pp, **p = 1.33×10⁻⁵**, 19/25 positive.

Full doc: `research/F51_MULTI_SEED_FINDINGS.md`.

**⚠ Historical numbers below — kept for method derivation. The Pareto picture below was measured under the LEAKY full-data log_T at seed=42 only.** Under leak-free per-fold log_T (the current C4 fix) the absolute reg drops by ~13 pp uniformly; under multi-seed averaging the +3.48 pp Δ is stronger evidence than the single-seed +4.63 pp shown below. See the F51 multi-seed table above for the current paper-grade numbers.

**Pareto picture — paired Wilcoxon vs H3-alt (FL 5f × 50ep, seed 42, LEAKY):**

| metric | H3-alt | P4 alone | **P4 + Cosine** ⭐ | P4 + OneCycle |
|---|---:|---:|---:|---:|
| reg top10 @ ≥ep10 | 71.44 ± 0.76 | 75.48 ± 0.75 | **76.07 ± 0.62** | 77.52 ± 0.53 |
| Δreg vs H3-alt | — | +4.04 | **+4.63** | +6.08 |
| p(reg > H3-alt) | — | **0.0312** ✅ | **0.0312** ✅ | **0.0312** ✅ |
| cat F1 @ best | 68.36 ± 0.74 | 68.20 ± 0.69 | **68.51 ± 0.88** | 66.52 ± 2.29 ⚠ |
| Δcat vs H3-alt | — | −0.16 | **+0.15** | **−1.84** |
| Pareto verdict | predecessor | dominant ✅ | **dominant 🏆** | **TRADE** ⚠ |

**Per-fold reg @ ≥ep10:**
- P4 + Cosine: `[75.88, 75.10, 76.18, 76.60, 76.59]` mean 76.07, best_eps `{10,10,10,10,11}`
- vs H3-alt deltas: `[+5.02, +4.43, +4.98, +4.17, +4.54]` (5/5 positive, σ_Δ = 0.36)

**Per-fold cat F1:** P4+Cosine `[69.58, 67.23, 68.11, 68.84, 68.80]` — stable, no outlier. Compare to P4+OneCycle `[68.60, 66.38, 67.35, 62.68, 67.60]` — fold 4 collapses ~5 pp below others, blowing σ from 0.74 → 2.29.

**Mechanism (Pareto-aware):** P4 alternating-SGD is the necessary substrate — it gives reg its own optimizer step on its own batch, preventing post-ep-5 cat-dominance collapse. The scheduler choice then trades reg strength against cat stability:
- **Cosine** (decay from peak) — α gets early boost (best_ep ~ep 4-6 in greedy view), P4's separation preserves it through ep 10+, then graceful decay protects cat throughout. Net: +4.63 reg, +0.15 cat → Pareto-dominant.
- **OneCycle** (warmup → peak at ep 19-20) — second growth window for reg, but the high LR at ep 19-20 destabilises cat in some folds. Net: +6.08 reg, −1.84 cat → Pareto-trade.

**Tier-A negative controls confirm P4 is necessary** — every non-P4 config either fails reg significance or trades cat (or both). OneCycle without P4 (A1) underperforms H3-alt by 9 pp; cosine without P4 (A2) collapses to 67.59 ± 8.99 at ≥ep10. See `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §6.4` for the full table.

**Reg-only-optimal alternative — P4 + OneCycle** (run `_1636`): when cat preservation isn't a constraint (e.g. ablation studies focused on reg), `--scheduler onecycle --max-lr 3e-3 --pct-start 0.4` lifts reg by an additional +1.45 pp (77.52 vs 76.07) at the cost of a single-fold cat collapse. Documented but **not the committed default for joint MTL**.

---

## Predecessor — F50 P4 alone + delayed-min (2026-04-29, superseded same day)

P4 alternating-SGD with constant scheduler — the first paper-grade fix for the FL gap. Headline: 75.48 @ ≥ep10 (vs H3-alt's 71.44; +4.04 pp, paired Wilcoxon p=0.0312, 5/5 positive). Superseded by P4+OneCycle which is +2.04 pp stronger by composing with OneCycle's late-LR peak. Run dir: `_0520`.

---

## Predecessor — F48-H3-alt (2026-04-26)

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)
task_a head (cat)    : next_gru
task_b head (reg)    : next_getnext_hard                # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, batch=2048, 50 epochs, seed 42
LR scheduler         : constant (no OneCycleLR / no annealing)
LR per param group   : cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3   # ← new
```

**Single-line additive recipe vs B3:**
```bash
--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
```

**Full canonical CLI invocation for H3-alt** (heads + input modalities still apply — same caveats as B9 above):
```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state {state} --engine check2hgi --seed 42 \
    --epochs 50 --folds 5 --batch-size 2048 \
    --model mtlnet_crossattn \
    --mtl-loss static_weight --category-weight 0.75 \
    --scheduler constant \
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --task-a-input-type checkin --task-b-input-type region \
    --per-fold-transition-dir output/check2hgi/{state}
```

**Headline numbers (5-fold × 50ep, seed 42):**

| State | cat F1 (B3 → H3-alt) | reg Acc@10 (B3 → H3-alt) | vs STL F21c |
|---|---|---|---|
| **AL** | 42.71 → **42.22 ± 1.00** (-0.49) | 59.60 → **74.62 ± 3.11** (+15.02) | **+6.25 EXCEEDS** ✓ |
| **AZ** | 45.81 → **45.11 ± 0.32** (-0.70) | 53.82 → **63.45 ± 2.49** (+9.63) | -3.29 (closes 75% of gap) |
| **FL** | 65.72† → **67.92 ± 0.72** (+2.20) | 65.26† → **71.96 ± 0.68** (+6.70) | TBD (F37 4050-assigned) |

†FL B3 ref is F32 1-fold n=1.

**Mechanism (single sentence):** α (graph-prior weight in `next_getnext_hard.head`) needs sustained 3e-3 to grow → reg lift; `shared_lr=1e-3` keeps cross-attn gentle so the cat path stays stable; `cat_lr=1e-3` keeps the cat encoder/head from diverging. The earlier monolithic-LR family (F44-F48-H2) couldn't satisfy both simultaneously because it forced α and the cat path to share an LR.

**Attribution refinement (F49, 2026-04-27):** the H3-alt mechanism above is the *operational* story — it explains why the optimizer recipe works. F49's 3-way decomposition asked the *causal* question — "what does the resulting MTL model do that STL `next_getnext_hard` doesn't?" — and showed the answer is **architecture, not cat-supervision transfer**. On AL the architecture alone (encoder-frozen λ=0, frozen-random cat features) lifts reg by +6.48 pp over STL F21c; cat-supervision via L_cat adds ≈ 0; cross-attn-mediated cat-encoder co-adaptation also adds ≈ 0. AZ shows the classical "architectural overhead, multi-task wrap rescues" pattern. FL's frozen variant is unstable (separate caveat). Operationally H3-alt is unchanged; the *paper claim* is sharpened from "joint MTL transfers cat→reg signal" to "the cross-attention architecture under the per-head LR regime extracts more reg signal from the same input than STL can." See `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` and `CLAIMS_AND_HYPOTHESES.md §CH19`.

**Cross-checks landed:**
- F40 (loss-side cat_weight ramp 0.75→0.25) — cat OK, reg only +1 pp → loss balance is not the lever
- F48-H1 (constant 1e-3 everywhere) — cat OK, reg flat → α needs LR ≥ 2e-3 to grow
- F48-H2 (warmup_constant 50→3e-3 plateau, single LR) — cat OK, reg WORSE → cat-vs-reg compete for shared cross-attn capacity at plateau LR
- F48-H3 (per-head with `shared_lr=3e-3`) — cat collapsed → shared cross-attn at 3e-3 destabilises cat path

H3-alt is the unique configuration in this design space. See `research/F48_H3_PER_HEAD_LR_FINDINGS.md` for the full derivation, `research/F48_H2_WARMUP_CONSTANT_FINDINGS.md` and `research/F40_SCHEDULED_HANDOVER_FINDINGS.md` for the negative controls, and `MTL_ARCHITECTURE_JOURNEY.md` for the end-to-end narrative from initial design to the current recipe.

**Note on `experiments/check2hgi_up/run_mtl_b3.py` and `docs/COLAB_GUIDE.md`:** both use the **predecessor B3 recipe** (`--max-lr 0.003`, no per-head LR), not H3-alt. This is deliberate for the check2hgi-up embedding-variant study (B3 is the established fair-comparison harness for downstream MTL, so embedding-variant-vs-baseline deltas are interpretable). For new MTL claims against STL, use the H3-alt recipe above. To extend `experiments/check2hgi_up` to H3-alt, append `--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3` to the `run_mtl_b3.py` command and rename the script accordingly. Tracked in `PAPER_PREP_TRACKER.md §2.3` as a camera-ready follow-up.

## Predecessor — B3 50ep (2026-04-24, kept for reference)

**B3 champion (the predecessor recipe):**

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)   # reg weight = 0.25
task_a head (cat)    : next_gru                                 # ← updated 2026-04-24 (F27)
task_b head (reg)    : next_getnext_hard                        # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, max_lr=0.003, batch=2048, 50 epochs, seed 42
LR scheduler         : OneCycleLR (PyTorch default)
```

**F27 cat-head refinement — 5-fold AZ paired Wilcoxon** (B3 default `next_mtl` cat-head vs B3 `next_gru` cat-head):
| Metric | Δ mean | p_greater | Verdict |
|---|---:|---:|---|
| cat F1 | +2.37 pp | **0.0312** ✅ | significant |
| cat Acc@1 | +4.69 pp | **0.0312** ✅ | significant |
| reg MRR_indist | +1.50 pp | **0.0312** ✅ | significant |
| reg Acc@10_indist | +1.98 pp | 0.0625 | marginal |
| reg Acc@5_indist | +1.69 pp | 0.0625 | marginal |

Per-fold, all 5 cat folds positive on both cat F1 and cat Acc@1. See `research/F27_CATHEAD_FINDINGS.md` and `scripts/analysis/az_b3_cathead_wilcoxon.py`.

## AZ headline numbers under the new B3 (2026-04-24)

| Metric | Value |
|---|---:|
| Cat F1 | **0.4581 ± 0.0130** |
| Cat Acc@1 | **0.4930 ± 0.0067** |
| Reg Acc@10_indist | 0.5382 ± 0.0311 |
| Reg Acc@5_indist | 0.4054 ± 0.0340 |
| Reg MRR_indist | 0.2766 ± 0.0241 |

**vs STL Check2HGI cat (matched-class):** cat F1 0.4208 ± 0.0089 → **Δ = +3.73 pp** (much stronger than the pre-F27 +1.65 pp).
**vs STL STAN (reg ceiling):** reg Acc@10 0.5224 ± 0.0238 → Δ = +1.58 pp (tied within σ).
**vs STL GETNext-hard (F21c matched-head reg baseline):** reg Acc@10 0.6674 ± 0.0211 → Δ = **−12.92 pp** (MTL still trails on reg — F21c finding persists).

## Caveats — Phase-1 substrate-specific addendum (2026-04-27, ⚠ RETRACTED 2026-05-16)

> ⚠ **2026-05-16 RETRACTION — the "MTL+HGI catastrophic 37 pp break" claim below is dominantly a leak artefact.** The 2026-04-27 measurements used the pre-F50-T4 legacy `region_transition_log.pt` (full-data, val transitions in the prior — C4 leak documented in `findings/MTL_FLAWS_AND_FIXES.md §2.12`). Under leak-free per-fold log_T at the SAME B3 recipe, the catastrophic break almost entirely disappears.
>
> **Leak-free B3+HGI re-run (mtl-exploration, 5f × 50ep × seed=42, AL+AZ+FL):**
>
> | State | Pre-leak B3+HGI reg (2026-04-27) | **Leak-free B3+HGI reg (2026-05-16)** | STL+HGI ceiling (v11 §0.3) | MTL Δ vs STL |
> |---|---:|---:|---:|---:|
> | AL | 29.95 | **57.38 ± 4.64** | 61.86 ± 3.29 | **−4.48 pp** |
> | AZ | 22.10 | **46.95 ± 2.60** | 53.37 ± 2.55 | **−6.42 pp** |
> | FL | (not measured pre-leak) | **70.47 ± 1.68** | 73.58 ± 0.43 | **−3.11 pp** |
>
> **Attribution decomposition (AL, full 31.91 pp gap from 29.95 to STL ceiling 61.86):**
> - Leak-fix contribution: **+27.43 pp (86 %)** — *dominantly the leak*.
> - Recipe upgrade B3 → B9: +4.10 pp (13 %).
> - Residual to STL ceiling: +0.38 pp (1 %).
>
> AZ: leak fix +24.85 pp (79 %), recipe upgrade +6.93 pp (22 %).
>
> **Cat F1 under HGI substrate is essentially constant** across all four recipes (AL: B3 pre-leak 25.96 / B3 leak-free 25.95 / B9 leak-free 25.23 / STL 25.26). HGI substrate's cat capacity is structurally capped — recipe-invariant.
>
> **Refined framing**: under leak-free measurement, MTL+HGI does NOT catastrophically break the reg head. It trails the STL+HGI ceiling by a normal MTL-cost margin (3–6 pp on reg, ≤1 pp on cat). The "MTL is substrate-specific" original framing should be re-read in light of v11 §0.3 (which already shows reg substrate axis HGI > C2HGI by 1.6–3.1 pp at STL).
>
> Sources for the retraction: `docs/studies/mtl-exploration/EXPERIMENT_HGI_SUBSTRATE.md`, `docs/studies/mtl-exploration/run_b3_hgi.sh`, run dirs `results/hgi/{alabama,arizona,florida}/mtlnet_lr1.0e-04_bs{2048,2048,1024}_ep50_20260516_*`.

**Historical pre-retraction text** preserved below (do not cite for paper without the retraction context):

**MTL B3 only works with Check2HGI substrate.** Phase-1 Leg III (MTL counterfactual with HGI substituted, 5f × 50ep, seed 42 each at AL+AZ):

| State | MTL+C2HGI cat F1 | MTL+HGI cat F1 | MTL+C2HGI reg Acc@10_indist | MTL+HGI reg Acc@10_indist |
|---|---:|---:|---:|---:|
| AL | **42.71** | 25.96 | **59.60** | 29.95 |
| AZ | **45.81** | 28.70 | **53.82** | 22.10 |

The MTL configuration was tuned around Check2HGI's per-visit context. Substituting POI-stable HGI embeddings into the same B3 setup actively **breaks the reg head** (MTL+HGI Acc@10 = 29.95 < STL+HGI gethard Acc@10 = 67.52 at AL — a 37 pp regression vs the standalone HGI baseline). Paper framing implication: the MTL win is **interactional** with the substrate; F49's architectural attribution further qualifies it as **architecture interacts with substrate** (not "transfer happens").

Source: `results/hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_20260427_*`. Full table in `CLAIMS_AND_HYPOTHESES.md §CH18`, `OBJECTIVES_STATUS_TABLE.md §2.4`, and the Phase-1 verdict at `research/SUBSTRATE_COMPARISON_FINDINGS.md`.

## Caveats — the F21c finding (SCALE-CONDITIONAL, reframed 2026-04-28 after F37 FL)

**F21c (2026-04-24):** STL-with-the-graph-prior (`next_getnext_hard` single-task) outperformed MTL-B3 on region by 12–14 pp Acc@10 at AL + AZ. Full analysis: `research/F21C_FINDINGS.md`.

**First-pass resolution (F48-H3-alt, 2026-04-26):** the gap was NOT structural to MTL on AL+AZ — it was a single confound in the LR schedule. Per-head LR (cat=1e-3, reg=3e-3, shared=1e-3) closed/exceeded the STL ceiling on AL (+6.25 pp, paired Wilcoxon p=0.0312) and closed 75% of the gap on AZ. Full analysis: `research/F48_H3_PER_HEAD_LR_FINDINGS.md`.

**FL closure (F37, 2026-04-28) — scale-conditional reframing:** F37 STL `next_getnext_hard` FL 5f Acc@10 = **82.44 ± 0.38**, far above MTL-H3-alt FL 73.65 (per-task best, top10_acc_indist) / 71.96 (joint best). Paired Wilcoxon **−8.78 pp p=0.0312, 5/5 folds negative**. The matched-head STL ceiling exceeds MTL-H3-alt at FL. The H3-alt recipe **does not lift reg above STL at FL scale** — at 4,702 regions, the cross-attention architecture pays an architectural cost (F49 Layer 3: architectural Δ vs STL = **−16.16 pp**) that the per-head LR cannot recover.

**Per-state pattern (architectural Δ from F49 + MTL vs STL gap):**

| State | Regions | Architectural Δ (frozen − STL) | MTL H3-alt vs STL F21c | Verdict |
|-------|--------:|------------------------------:|----------------------:|---------|
| AL | 1,109 | **+6.48 pp** (architecture wins) | **+6.25 pp** | MTL exceeds STL ✓ |
| AZ | 1,547 | −6.02 pp | −3.29 pp (75% closed) | classical pattern |
| FL | 4,702 | **−16.16 pp** (heavy cost) | **−8.78 pp p=0.0312** | STL ceiling above MTL ✗ |

**Implication for paper framing.** CH18/CH21 are reframed as scale-conditional: AL is the architecture-dominant state where MTL H3-alt > STL on reg. FL's headline reg ceiling is STL `next_getnext_hard` (the matched-head single-task baseline). The H3-alt recipe is still the recommended joint-deployment config — and at FL the **substrate-side cat advantage** (CH16 + CH18-substrate) carries the contribution; the architecture-side reg lift is AL-only.

Full analysis: `research/F37_FL_RESULTS.md`. Concern tracker: `CONCERNS.md §C15` (re-opened 2026-04-28 with FL caveat).

The B3-vs-STL framing below is preserved for the predecessor recipe (still relevant when the per-head LR mode is not used):

- **Joint-task single-model deployment:** B3 gives both `next_category` and `next_region` predictions in one forward pass. Two STL models (one GETNext-hard for region + one matched STL cat head) would beat B3 on region by 12 pp but require running two separate models.
- **Cat F1 lift over STL:** MTL-B3 does lift STL cat F1 (AZ +3.73 pp, p=0.0312). This contribution survives F21c.
- **FL-scale PCGrad pathology:** F2's late-stage-handover finding is independent and paper-worthy.

## Validation status across states (post-F27)

| State | Protocol | cat F1 | reg Acc@10 | Status |
|---|---|---:|---:|---|
| AL | 5f × 50ep (pre-F27) | 0.3928 ± 0.0080 | 0.5633 ± 0.0816 | superseded by F31 |
| AL | **5f × 50ep (post-F27, next_gru)** | **0.4271 ± 0.0137** | **0.5960 ± 0.0409** | ✅ **F31 validated, +3.43 cat / +3.27 reg Acc@10** |
| AZ | 5f × 50ep (pre-F27) | 0.4362 ± 0.0074 | 0.5276 ± 0.0392 | superseded |
| AZ | **5f × 50ep (post-F27, next_gru)** ⭐ | **0.4581 ± 0.0130** | **0.5382 ± 0.0311** | ✅ **committed; Wilcoxon p=0.0312 on 3 metrics** |
| FL | 1f × 50ep (pre-F27, F2 + F17 fold 1 ×2) | 0.6623 / 0.6706 | 0.6582 / 0.6655 | prior n=1 |
| FL | **1f × 50ep (post-F27, next_gru)** | 0.6572 | 0.6526 | ⚠️ **F32 — cat F1 −0.93 vs pre-F27 mean**; within n=1 noise but direction flipped |
| FL | **5f × 50ep H3-alt MTL** (F48-H3-alt, 2026-04-26) | **0.6792 ± 0.0072** | 0.7196 ± 0.0068 (top10_acc_indist) | ✅ MTL champion FL run |
| FL | **5f × 50ep STL `next_gru` cat** (F37 P1, 2026-04-28) | **0.6698 ± 0.0061** | — | ✅ matched-head cat ceiling. MTL > STL by **+0.94 pp** at FL ✓ |
| FL | **5f × 50ep STL `next_getnext_hard` reg** (F37 P2, 2026-04-28) | — | **0.8244 ± 0.0038** | ⚠️ matched-head reg ceiling **exceeds MTL-H3-alt by −8.78 pp p=0.0312, 5/5 folds**. CH18 reframes scale-conditional. |

## ⚠ F27 scale-dependence flag (2026-04-24)

The cat-head swap `NextHeadMTL → next_gru` **helps AL (+3.43 pp cat F1) and AZ (+2.37 pp, p=0.0312) but slightly hurts FL at n=1 (−0.93 pp cat F1)**. Three paths documented in `research/F27_CATHEAD_FINDINGS.md §Decision`:

- **A:** Commit `next_gru` universally (accept small FL cost for simpler narrative).
- **B:** Scale-dependent — `next_gru` for AL/AZ, `next_mtl` for FL/CA/TX.
- **C:** Run FL 5f B3+gru (~6 h MPS) to resolve definitively.

The NORTH_STAR config above currently reflects **A** pending user decision. If the user picks **B**, the cat head for FL/CA/TX reverts to `next_mtl` (MTLnet's historical default).

## History

### Post-F2 update (2026-04-23 evening)

F2 (`research/B5_FL_TASKWEIGHT.md`) completed all four phases. The Phase B3 configuration **`mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8h`** at n=1 fold on FL delivers:

| Metric | Soft B-M13 (prior north-star) | B3 | Δ |
|---|---:|---:|---:|
| cat F1 | 0.6601 | **0.6623** | **+0.22 pp** |
| reg Acc@10_indist | 0.6062 | **0.6582** | **+5.20 pp** |
| reg Acc@5_indist | 0.3601 | **0.3988** | **+3.87 pp** |
| reg MRR_indist | 0.2555 | **0.2794** | **+2.39 pp** |

B3 Pareto-dominates soft at n=1 on every joint-score metric. The mechanism: cat-heavy weighting triggers a **late-stage handover** — cat head converges fast in early epochs, then the shared backbone becomes available to the region head for the remaining epochs (cat training extends to epoch 42 vs ≤10 for soft/pcgrad/equal-weight).

**Interim policy:** soft remains the reported north-star until F2's follow-up validation lands (§Re-evaluation triggers). If both checks hold, B3 becomes the new universal north-star.

### Follow-up required before committing B3

| Check | Cost | Pass criterion |
|---|:-:|---|
| B3 on FL, 5-fold | ~5–6 h MPS | σ on cat F1 does not pull B3 below soft (soft cat = 66.01 n=1; B3 cat = 66.23 n=1 — σ could be decisive) |
| B3 on AL, 5-fold | ~1 h MPS | B3 does not break the AL cat head (current AL-hard+pcgrad is 38.50 cat F1) |
| B3 on AZ, 5-fold | ~1–1.5 h MPS | B3 preserves the AZ region lift from B-M9d (53.25 Acc@10 with pcgrad) |

Note: static_weight is a simpler optimizer than PCGrad, and AL/AZ already work under the harder PCGrad; low-risk that B3 breaks them.

## Interim choice (still current, pre-B3-validation)

**`mtlnet_crossattn + pcgrad + next_getnext (soft probe) d=256, 8 heads`** (B-M6b on AL, B-M9b on AZ, B-M13 on FL). All paper tables currently reference this config.

If F2 follow-up passes and B3 replaces soft, the migration is a single-string swap in every paper-facing table plus a 5-fold re-run on each state — same wall-clock cost as any scientific revision.

## Why (short version)

| State | soft joint Acc@10 / cat F1 | hard joint Acc@10 / cat F1 | Winner |
|:-:|:-:|:-:|:-:|
| AL 5f | 56.49 / 38.56 | 57.96 / 38.50 | tied within σ |
| AZ 5f | 46.66 / 42.82 | 53.25 / 42.22 | hard on reg (+6.59 outside σ), cat σ-tied |
| FL 1f | **60.62 / 66.01** | 58.88 / **55.43** | **soft** — hard's cat head fails to train |

- **FL is the headline state** (per `CONCERNS.md §C01` — the paper's primary table is FL + CA + TX).
- Hard has a **diagnosed training failure at FL scale** (see `research/B5_FL_SCALING.md` + the 2026-04-23 JSON comparison in `archive/check2hgi-reviews-2026-04/2026-04-23_critical_review.md`): cat head's best-val F1 over 50 epochs is 55.43 vs soft's 66.01 under the identical fold split. Not noise — gradient imbalance.
- Soft scales uniformly across AL / AZ / FL. Cat F1 is within σ of the cross-attn + GRU champion at every state.

## What this choice costs us

- The **AZ +1.01 pp MTL-over-STL-STAN** result (53.25 vs 52.24) that currently sits in hard. Soft on AZ lands at 46.66 Acc@10, which is +3.70 pp above Markov-1 but −5.58 below STL STAN. Under soft, AZ reg is framed as "MTL beats Markov" rather than "MTL beats STL".
- The "faithful Yang 2022 SIGIR" framing. Soft is an adaptation (learned probe) rather than the original hard-index formulation.

## What hard is still used for

Hard remains a **reported ablation row**, not retired. In the paper:

> We propose MTL-GETNext-soft as the joint-task model. We report a faithful hard-index variant as an ablation: at region-cardinality ≤ 1.5 K it matches (AL, within σ) or dominates (AZ, +6.59 pp Acc@10, +3.08 pp MRR) the soft adaptation. At 4.7 K-region scale (FL), hard over-dominates the MTL gradient through PCGrad and the category head fails to train (best-val cat F1 0.554 across 50 epochs vs soft's 0.660). We analyse the mechanism in §X and recommend soft as the scale-robust default.

## Re-evaluation triggers

This choice is revisited if **any** of the following lands:

1. **F2 (FL task-weight sweep).** If `task_b_weight < 1` restores FL-hard cat F1 to ≥ 60 while keeping reg Acc@5 lift, hard becomes scale-uniform and is re-promoted as north.
2. **F12 (FL 5-fold hard) with σ showing cat F1 within σ of soft.** Would argue the 10 pp cat gap was n=1 amplification, not training pathology — low likelihood given the `diagnostic_task_best` analysis but empirically checkable.
3. **A new MTL variant** (e.g., per-task weight clipping, prior-magnitude normalisation) that rescues FL-hard without a task-weight hack. Post-paper research direction.

Until one of those lands: **soft is the headline MTL config**.

## Pointers

- Joint-execution comparison: `OBJECTIVES_STATUS_TABLE.md §2`
- Cross-state deltas: `research/B5_MACRO_ANALYSIS.md`
- FL failure-mode diagnosis: `research/B5_FL_SCALING.md` + `archive/check2hgi-reviews-2026-04/2026-04-23_critical_review.md §FL-hard training pathology`
- Open follow-ups that can change this: `FOLLOWUPS_TRACKER.md` F2, F12
