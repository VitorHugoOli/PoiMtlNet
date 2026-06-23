# RUN_MATRIX — DRAFT for P2-freeze sign-off (closing_data)

> **STATUS: DRAFT, NOT FINAL.** This is the P1b inventory reconciled with the B1–B5 baseline triage. The
> *real* RUN_MATRIX is signed off by the user at the **P2 freeze** (PLAN §P2 / §Remaining sign-off). Do not
> treat any disposition here as committed. Authoritative inputs:
> [`PHASE1_VERDICT.md`](PHASE1_VERDICT.md) §3 (P1a dispositions, honored row-for-row),
> [`PLAN.md`](PLAN.md) §P1b/§P2/§P3, [`PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md) gate ledger,
> [`TABLES_FIGURES.md`](../../../articles/[BRACIS]_Beyond_Cross_Task/TABLES_FIGURES.md) T1–T5/F1–F2/F-arch,
> [`RESULTS_TABLE.md`](../../results/RESULTS_TABLE.md) §0.1–§0.9,
> [`PAPER_BASELINES_STRATEGY.md`](../../PAPER_BASELINES_STRATEGY.md),
> [`CANONICAL_VERSIONS.md`](../../results/CANONICAL_VERSIONS.md) §v14/§v16, baseline triage (B1–B5).

## 0 · Frozen-recipe constant (the yardstick for every RE-RUN cell)

**Champion G (v16) on v14 substrate.** Engine `check2hgi_design_k_resln_mae_l0_1`; model
`mtlnet_crossattn_dualtower` + reg-head `next_stan_flow_dualtower` (`raw_embed_dim=64`, `fusion_mode=aux`,
`freeze_alpha=True`, `alpha_init=0.0` → **additive prior OFF**); cat-head `next_gru`; `--mtl-loss
static_weight --category-weight 0.75`; **UNWEIGHTED CE both heads** (v15 / C25 fix); `--scheduler onecycle
--max-lr 3e-3`; per-head LR cat 1e-3 / reg 3e-3 / shared 1e-3; `--log-t-kd-weight 0.0`;
`--checkpoint-selector geom_simple`. **Protocol:** ALL 6 states {AL, AZ, FL, CA, TX, GE} × seeds
{0,1,7,100} × 5 folds = **n=20/cell**. **Matched scorer:** `r0_matched_rescore.py` — cat **macro-F1**; reg
**FULL `top10_acc` fp32 on BOTH MTL and STL sides** (the B-A2 correction — never compare MTL-indist to
STL-full). **Provenance (C28, every RE-RUN):** PID-suffixed rundirs + per-run seed echo; seed-tagged
per-fold `log_T` freshness preflight (log_T mtime > `next_region.parquet` mtime) before any
`--per-fold-transition-dir` run; `--canon` pinned in every driver.
**Execution config (PINNED 2026-06-19):** every P3 cell runs with **`--compile --tf32`** + GPU-resident
dataset (auto-fit) — ~15% faster, empirically result-neutral (cat +0.046 / reg +0.065 pp vs the no-knobs
byte-identical anchor, ≪ noise; `pre_freeze_gates/SPEED_LEVERS.md`). Run the WHOLE board compiled (never mix
compiled/non-compiled cells); reviewers reproduce *with* them. torch **2.11** (no upgrade); `num_workers=0`.

**Precision (PINNED 2026-06-23 — supersedes the implicit fp16 autocast; root cause `CA_MTL_DIVERGENCE.md`).**
Every MTL training cell runs **bf16 autocast** (`MTL_AUTOCAST_BF16=1`, `mtl_cv.py:321-326`) — NOT the trainer
default fp16. fp16 autocast (no GradScaler) overflows the 65504 ceiling at CA/TX's wide reg logits → a
deterministic ep30 NaN that poisons the shared backbone → both heads collapse (CA −5.23 / TX −2.41 reg were
**collapse artifacts, VOID**), and it *understated* MTL reg even where it didn't crash (AL fp32 closes ½ the
gap −0.38→−0.18; **FL fp32 reg 77.71 BEATS the STL ceiling 76.71**). bf16 has fp32's exponent range (no
overflow) at tensor-core speed. **Eval/scoring stays fp32** — the matched scorer `r0_matched_rescore.py`
re-forwards in fp32, and any in-trainer val metric uses `MTL_DISABLE_AMP_EVAL=1`; the STL **reg** ceiling is
already true fp32 (`p1_region_head_ablation.py:83`), so MTL-vs-ceiling is now precision-matched (it was a
fp16-MTL-vs-fp32-ceiling MISMATCH before). **bf16 is the default PENDING the A40 equivalence gate (§0a)** — if
bf16 ≢ fp32 there, the board falls back to full fp32 (`MTL_DISABLE_AMP=1`, ~2–3× slower). STL ceilings need NO
re-run (already fp32); STL cat ceilings are precision-insensitive (AL 63.44→63.48, optional). Re-baseline scope
= **MTL cells only**.

## 0a · Precision-equivalence gate (A40 — PRECONDITION for the §0 precision pin)
Before the board standardizes on bf16, the A40 lane runs a **bf16-vs-fp32 A/B** (mirrors the A100-equivalence
A/B, same ±0.05 pp rule). State **FL** (representative scale; completes in both precisions; cross-checks the
H100 fp32 fold-1 anchor reg 77.71 / cat 79.43), champion-G MTL, seed 0, 5 folds, `--compile --tf32` on
`check2hgi_dk_ovl`, **compile+tf32 held FIXED on both arms** (isolates the autocast dtype):
- **Arm X (bf16):** `MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1` → bf16 train, fp32 eval.
- **Arm Y (fp32):** `MTL_DISABLE_AMP=1` → fp32 train + fp32 eval (still tf32-matmul, board-consistent).
- **Compare** matched cat macro-F1 + reg FULL `top10_acc`, per-fold and 5f-mean, **4 dp**.
- **Decision rule:** `|Δcat| ≤ 0.05 pp AND |Δreg| ≤ 0.05 pp` ⇒ **bf16 ≡ fp32 → standardize bf16 board-wide**
  (fast). Else ⇒ **fp32 board-wide** (`MTL_DISABLE_AMP=1`) for every MTL cell. Either way, report Arm-X reg vs
  the FL STL reg ceiling 76.71 (expect Δreg ≳ 0, confirming the gap closed/reversed). STOP for the user with
  the table; the chosen precision is then frozen into §0 + the shared recipe before the full board launches.

**Story consequence (load-bearing):** the frozen recipe **INVERTS** the BRACIS headline. §0.1/T3 reported
"MTL sacrifices region −7…−17 pp"; under G the gap **dissolves** (MTL matches the STL reg ceiling, matched
Δ −0.09…−0.31, and beats the STL cat ceiling +2.6…+4.1). Restating §0.1 / the BRACIS tables is an AUTHOR
decision (PAPER_UPDATE rule) — closing_data regenerates the base story-agnostically.
> ⚠ The "matched Δ −0.09…−0.31" reg parity above was measured under the **fp16 harness** (MTL reg
> fp16-understated). Under the §0 bf16/fp32 pin the gap is expected to **close further or reverse** (AL fp32
> −0.18; FL fp32 +1.0 over the ceiling) — these numbers are being re-baselined by the precision re-run; do not
> cite −0.09…−0.31 as final.

## 1 · BRACIS-suite cells (T1–T5, F1–F2, F-arch, §0.x supporting)

Comparability class — **SC** = substrate-column (inherits the frozen base through the matched-head
pipeline); **E2E** = end-to-end (builds its own sequences, must mirror windowing/stride + user-disjoint
folds + label spaces); **n/a** = descriptive/diagram/derived.

| Cell | Backing today | Disposition | Run-spec (frozen recipe unless noted) | Prereqs |
|---|---|---|---|---|
| **T1** dataset stats (§4.1) | `data/<state>` + `output/check2hgi/<state>/regions`; some placeholders | **REUSE** (recompute placeholders) | No training. pandas counts over corpus (users/check-ins/POIs/mean-traj-len) + region count. Add **GE** row. *Conditional RE-RUN:* if overlapping-windows ADOPTED, sequence/mean-traj-len-derived rows recompute (raw users/POIs/check-ins unchanged). | Corpus parquet 6 states + `regions.parquet`. |
| **T2** substrate ablation C2HGI vs HGI, both tasks (§5.1 / §0.3) | STL `next_gru`+`next_stan_flow`, GCN substrate, seed-42 | **STORY-DEPENDENT → RE-RUN if kept** | Single-v14 board drops the comparison by default. IF kept: STL matched-head per substrate. cat: `train.py --task next --engine {v14|hgi} --cat-head next_gru`; reg: `p1_region_head_ablation.py --region-emb-source {v14|hgi} --region-head next_stan_flow --per-fold-transition-dir …`. 6 states × {0,1,7,100} × 5f. | HGI embeddings at all 6 states (absent CA/TX/GE → build); seeded per-fold log_T (reg). |
| **T3** MTL vs STL both tasks (§5.2 / **§0.1**) | B9/H3-alt, class-weighted CE, GCN, n=20 (AL/AZ cat n=4 seeds); 5 states | **RE-RUN ★ (highest priority)** | 3 families: (1) **MTL champion G** (full frozen recipe, `--per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/{S}`); (2) **STL cat ceiling** `train.py --task next --engine v14 --cat-head next_gru`; (3) **STL reg ceiling** `p1_region_head_ablation.py --region-emb-source v14 --region-head next_stan_flow`. 6 states × {0,1,7,100} × 5f. Matched scorer both sides. | **M0:** v14 substrate at CA/TX (H100 build), GE (sync/build), FL/AL/AZ present; seeded per-fold log_T all 6 states (built AFTER windowing gate); frozen folds; staleness preflight. |
| **T4** Δm joint score (§5.2 / §0.2) | leak-free CH22, mostly seed-42, FL n=25; broken selector, class-weighted CE | **RE-RUN** (derived, no new training) | Compute Δm from T3 outputs: cat F1 + reg MRR (primary), cat F1 + reg Acc@10 (secondary) per Maninis 2019/Vandenhende 2022; per-fold paired Wilcoxon over (seed,fold), n=20, 6 states. Needs MRR + Acc@10 extracted at the geom_simple checkpoint. | Same as T3 (MTL-G + STL-ceiling rundirs with MRR+Acc@10 per fold). |
| **T5** external baselines per state (§5.3 / §0.5–0.6) | STAN/ReHDM/POI-RGNN/MHA+PE faithful + substrate probes, BRACIS-era lighter protocol, seed-42 | **RE-RUN at full n=20 (USER-DECIDED)**; per-engine INCLUSION story-dependent | See §2 (the baseline block — this is where B1–B5 merge in). Run families split by SC vs E2E. ReHDM CA/TX deferred (compute caveat, footnote). | v14 at 6 states; per-engine substrate artifacts where absent (HGI/DGI/HMRM/Time2Vec/Space2Vec/CTLE/POI2Vec at CA/TX/GE — many absent → build); seeded log_T (reg); frozen folds; **E2E blocked on windowing**. |
| **F1** per-visit mechanism @ AL (§6.1, REQUIRED) | STL linear-probe + `next_gru`, canonical/POI-pooled/HGI (§0.7), seed-42 | **STORY-DEPENDENT → RE-RUN if kept** | IF kept: STL matched-head `next_gru` + linear probe on {canonical C2HGI, POI-pooled, HGI} (or v14 + v14-pooled + HGI if re-anchored) @ AL, {0,1,7,100}, 5f. NOTE: **A2 (RESOLVED, ON NULL)** already strengthens the substrate-cat mechanism — F1 may be backed by `A2_RESULTS.md` rather than re-run. | AL substrate artifacts (present) + pooled-variant builder. |
| **F2** scale-progression scatter (optional, cut-first) | §0.1 Δ_reg column (the OLD gap) | **STORY-DEPENDENT (likely obsolete)** | Visualizes the −7…−17 pp gap that DISSOLVES under G → likely DROP. Re-derive from T3 Δ_reg only if a residual-gap story survives (no training). | T3 outputs. |
| **F-arch** architecture schematic (optional, cut-second) | none (textual spec = single cross-attn = v11 arch) | **REUSE (redraw, no compute)** | Redraw for champion G **DUAL-TOWER**: add reg-private tower + aux-fusion path, remove additive log_T prior arrow (α frozen 0). d_model=256, 8 heads, 4 backbone blocks. | Frozen G arch spec (CANONICAL_VERSIONS §v16). |
| **§0.4** recipe selection B9 vs H3-alt | n=20 multi-seed, pre-C25 class-weighted CE, GCN | **SUPERSEDED → RE-RUN/reframe** | New question (P2): ONE onecycle recipe vs a documented small/large-state split. G + onecycle vs candidate per-state variants at AL/AZ (small) and FL/CA/TX/GE, {0,1,7,100}, 5f. May collapse to one recipe if onecycle dominates everywhere (§0.1 annotation suggests it does). | Same as T3. |
| **§0.8** log_T-KD reg lift (W=0.2 vs 0.0) | B9/v12 single-pathway, AL/AZ n=20, FL/CA/TX pilot, GCN | **STORY-DEPENDENT (standalone B9-panel only)** | **NULL on G** (X2: FL reg +0.05 / AL reg −0.13 ≪ 0.3 pp gate; FL cat −0.57). Do NOT run on G. Reuse existing §0.8 numbers AS-IS only if the story wants the "pre-G prior channel" narrative. | None new. |
| **§0.9** substrate-null-in-MTL (regime finding) | designs B/J/L/M + HGI ceiling + STL↔MTL isolation + ResLN, pre-C25, seed-42 | **RE-RUN (lightweight) / re-read** | Mechanism HOLDS on G. **Option B (recommended):** cite `mtl_improvement` R0 bar + X2/X3 (substrate/optimizer/prior all null on G) — no compute. **Option A:** one FL STL↔MTL α=0 isolation run at G, {0,1,7,100}. | Option A: v14+log_T at FL (present). Option B: none. |
| **T6** drop-in MTL ablation (FAMO/Aligned-MTL/HSM) — cut to prose | §2.3 prose, single-seed n=5, pre-C25, H3-alt | **STORY-DEPENDENT (prose)** | Already cut to prose. Covered by the regime finding (optimizer levers null on G = T4 convergent negative). If kept, re-anchor to G's convergent-negative optimizer result; do NOT re-run FAMO/Aligned-MTL/HSM. | None. |
| **C1-panel** 3-snapshot per-task routing (deploy headroom) | PLAN G0.2 C1 (CLOSED 2026-06-17), G/v14, FL+AL {0,1,7,100} n=20 | **STORY-DEPENDENT (SUPPORTIVE diagnostic only)** | Gate CLOSED → PROMOTE as supportive deploy-headroom panel ONLY; the single geom_simple checkpoint stays the headline (single-model property). No new training: `route_task_best.py` over existing T3 MTL-G rundirs (re-select per-task-best from `--save-task-best-snapshots` snapshots). To extend to 6 states, save snapshots during the T3/M2 G runs. | Snapshots saved during T3/M2 G runs at panel states. |

## 2 · External-baseline block (T5 / §0.5–0.6) with B1–B5 merged in

T5 is RE-RUN at full n=20 under the frozen regime (protocol settled, PLAN §Resolved-3). Per-engine
INCLUSION is the live user decision (§Decisions). **SC** baselines inherit the frozen base automatically;
**E2E** baselines build their own sequences and are **BLOCKED on (a) the overlapping-windows ADOPT/KEEP
decision and (b) the P2 freeze** — they fold into P3/M1+M3 and budget ONE re-run if overlap is adopted.

### 2a · next-CATEGORY block (macro-F1)

| Row | Source | Class | Disposition | Run-spec | Prereqs / notes |
|---|---|---|---|---|---|
| Majority class | existing floor | n/a | RE-RUN (recompute on frozen folds) | descriptive recompute | frozen folds |
| Markov-1-POI | existing floor | n/a | RE-RUN (recompute on frozen folds) | `P0/simple_baselines` recompute | frozen folds |
| POI-RGNN faithful (Capanema 2022) | §0.6 v11 | E2E | RE-RUN n=20 | native trainer, mirror windowing/folds/labels; region/POI head→category | non-user-disjoint published caveat (T5 caption); blocked on windowing |
| MHA+PE faithful (Zeng 2019) | §0.6 v11 | E2E | RE-RUN n=20 | native trainer, mirror windowing/folds/labels | blocked on windowing |
| Substrate linear probe × C2HGI/HGI | §0.6 v11 | SC | RE-RUN n=20 | `p1_poi_head_ablation.py` linear probe on frozen base | HGI at CA/TX/GE → build |
| STL `next_gru` × v14 (cat ceiling) | = T3 fam (2) | SC | RE-RUN n=20 | shared with T3 STL cat ceiling | — |
| **B1 CTLE** (AAAI 2021) | NET-NEW | SC | **INCLUDE (Tier-1)** | Adapt Logan-Lin/CTLE → Gowalla state corpora; emit **64-d** per-visit embedding routed as a substrate column under `next_gru`; matched head, 6 states × {0,1,7,100} × 5f. **CTLE PRE-TRAINS on TRAIN-PORTION-ONLY per fold** (its MLM objective) → no transductive full-corpus advantage. | Coordinate transductive-fairness statement with **pre_freeze_gates/A4**; record tuning budget in fairness ledger (baseline_gap §1.4); re-run pretrain if overlap ADOPTED (re-windowed inputs). |
| **B2a POI2Vec** (AAAI 2017) | NET-NEW | SC | **INCLUDE (Tier-1)** | Emit **per-POI** (not per-fclass) **64-d** POI2Vec substrate parquet, register as substrate engine; matched `next_gru`, 6 states × {0,1,7,100} × 5f. **Pretrain TRAIN-PORTION-ONLY per fold** (parity with B1 CTLE + B2b skip-gram; a full-corpus POI2Vec would carry the same transductive asymmetry CTLE is held to). | **NET-NEW caveat:** in-repo POI2Vec (`research/embeddings/hgi/poi2vec.py`) is an **fclass-level HGI input feature**, NOT a standalone POI-level column — must be emitted fresh. Record tuning budget in the fairness ledger (`baseline_gap/TRIAGE.md` + baseline_gap §1.4). |
| **B2b skip-gram** (word2vec over check-in seqs) | NET-NEW | SC | **INCLUDE (Tier-1)** | Train skip-gram on check-in sequences **train-portion-only per fold** (parity with CTLE); emit 64-d; matched `next_gru`, 6 states × {0,1,7,100} × 5f. | Cheap (minutes/state). |
| **B2c one-hot-POI 64-d** (zero-training floor) | NET-NEW | SC | **INCLUDE (Tier-1 floor) — recommend ADD** | Emit a fixed **64-d** one-hot/hashed-POI-id (deterministic random projection) substrate column — **NO training**; matched `next_gru`, 6 states × {0,1,7,100} × 5f. Completes CTLE's canonical substrate-floor triplet (**one-hot / skip-gram / POI2Vec**) — the trivial absolute-zero below every learned substrate. | No training cost; deterministic across folds (fixed seed). |
| MTL × v14 (our model, cat) | = T3 fam (1) | SC | RE-RUN n=20 | shared with T3 MTL-G | — |

### 2b · next-REGION block (Acc@10)

| Row | Source | Class | Disposition | Run-spec | Prereqs / notes |
|---|---|---|---|---|---|
| Majority / Markov-1-region | existing floors | n/a | RE-RUN (recompute on frozen folds) | descriptive recompute | frozen folds |
| STL GRU × v14 | §0.5 v11 | SC | RE-RUN n=20 | matched-head region GRU on frozen base | seeded log_T |
| STL STAN × {v14, HGI} | §0.5 v11 | SC | RE-RUN n=20 | `p1_region_head_ablation.py --region-head next_stan` | HGI at CA/TX/GE → build; seeded log_T |
| STL STAN-Flow × {v14, HGI} (reg ceiling) | §0.5 v11 (= T3 fam (3)) | SC | RE-RUN n=20 | `p1_region_head_ablation.py --region-head next_stan_flow --per-fold-transition-dir …` | seeded per-fold log_T; staleness preflight |
| STAN faithful (Luo 2021) | §0.5 v11 | E2E | RE-RUN n=20 | native trainer, swap POI head → region head (~1.1k–8.5k tracts), report Acc@10; mirror windowing/folds | repo has the STAN region-adaptation pattern (`next_stan/`); blocked on windowing |
| ReHDM faithful (Li 2025) | §0.5 v11 (AL/AZ/FL only) | E2E | RE-RUN n=20; **CA/TX DEFERRED** | native trainer, mirror frozen protocol; CA/TX marked "—" (collaborator pool scales quadratically with region cardinality, exceeded H100 budget — honest-framing footnote, TABLES_FIGURES §7) | blocked on windowing; CA/TX deferral is a USER decision |
| **B3 HMT-GRN-style MTL** (SIGIR 2022) | NET-NEW | **E2E** | **INCLUDE (sole external MTL row)** | Shared LSTM/GRN hidden state + per-task softmax heads for next-**category** + next-**region** (TIGER tracts), equal-weight CE; mirror windowing/folds/seeds/labels, per-fold train-only priors. **Documented deviations:** beam-search/selectivity DROPPED (regions are headline, no next-POI head); geohash→TIGER tract; native per-user 80/20 split → our fold regime. Label **"HMT-GRN-STYLE"**. 6 states × {0,1,7,100} × 5f. | blocked on windowing + P2 freeze; budget one re-run if overlap ADOPTED. |
| **B4 cascade (CSLSL/CatDM pattern)** | NET-NEW | **SC (pinned)** | **INCLUDE — pin SC cascade** | **Pinned SC variant:** cascade OVER the frozen substrate — predicted-category signal wired into the region head, reusing `next_gru` (cat) + `next_stan_flow`/`next_lstm` (reg). Inherits windowing/folds/labels; isolates "cascade vs parallel" as the only varying factor. 6 states × {0,1,7,100} × 5f. **This is a controlled cascade-vs-parallel isolation reusing our heads, NOT a faithful CSLSL/CatDM reproduction** (pre-empts the faithfulness objection; see `baseline_gap/TRIAGE.md §B4 framing`). Faithful CSLSL/CatDM E2E (Tier-3) **DEFERRED to camera-ready**. No next-POI head; 7-root cat + TIGER tract. | SC variant: none new (reuses heads). Faithful E2E deferred (camera-ready). |
| **B5 Flashback** (IJCAI 2020) | NET-NEW | **E2E** | **INCLUDE — pin Flashback-only** | Swap POI head → region head (~1.1k–8.5k tracts), report Acc@10 (mirrors repo's STAN region adaptation); keep Flashback's spatiotemporal hidden-state weighting faithful. **Pinned Flashback ONLY** — well-justified for the **sparse AL/AZ traces** (the next-region axis is already multi-state anchored by STAN + ReHDM, so this is the defensible single add). **DeepMove DEFERRED to camera-ready** (Tier-3). mirror windowing/folds/seeds/labels. 6 states × {0,1,7,100} × 5f. | blocked on windowing + P2 freeze; one re-run if overlap ADOPTED. |
| MTL × v14 (our model, reg) | = T3 fam (1) | SC | RE-RUN n=20 | shared with T3 MTL-G | — |

## 2.5 · Baseline comparison design (the comparison ladder + the "original way" rule)

> Defines WHAT each baseline run answers and the non-negotiable rule that keeps every run comparable. The
> statistical tests for these comparisons are pre-registered in
> [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md) (family B = baselines-vs-STL / baselines-vs-MTL; paired
> Wilcoxon for "we beat X", Holm-Bonferroni across the baseline × state × task grid; pairing per the SC/E2E
> split). Grounding: [`../baseline_gap/TRIAGE.md`](../baseline_gap/TRIAGE.md) (B1–B5 + comparability classes),
> `PAPER_BASELINES_STRATEGY.md`, §2 above (the row inventory).

### The 4 run types (the comparison ladder)

Every baseline appears as one or both of these forms. The ladder is designed so each rung isolates exactly one
factor versus our champion, and the two baseline rungs (3, 4) answer two *different* questions — both reported.

| # | Run type | Architecture | Embedding / substrate | Heads | Isolates | Answers |
|---|---|---|---|---|---|---|
| **1** | **MTL champion G** (ours) | our dual-tower `mtlnet_crossattn_dualtower` | **our Check2HGI v14** (`check2hgi_design_k_resln_mae_l0_1`) | `next_gru` (cat) + `next_stan_flow_dualtower` (reg) | — (the proposed system) | the headline number |
| **2** | **STL ceiling** (ours) | our single-task trainers | **our Check2HGI v14** | our matched head per task (`next_gru` cat / `next_stan_flow` reg) | the MTL coupling (vs run 1) | does joint training beat our own per-task ceiling? (cat superiority / reg non-inferiority) |
| **3** | **Substrate-column baseline** | **OURS** (our `next_gru` / `next_stan_flow` heads) | **BASELINE embedding** (CTLE / POI2Vec / skip-gram / one-hot, 64-d, dropped into the v14 slot) | our matched head per task | the **EMBEDDING** (only the substrate slot changes; heads/folds/windowing held to ours) | **does our Check2HGI embedding beat theirs at matched capacity?** |
| **4** | **End-to-end native baseline** | **BASELINE arch** (HMT-GRN-style / Flashback / cascade / STAN / ReHDM / POI-RGNN / MHA+PE) | baseline's own embedding | baseline's own heads, region-adapted where needed | the **SYSTEM** (the whole published method) | **do we beat the published method end-to-end?** |

Run 3 is a **substrate-column (SC)** form: it inherits the frozen base through the matched-head pipeline, so it
is **paired** vs runs 1 and 2 by construction (same folds/seeds/windowing/labels — only the embedding slot
swaps). Run 4 is an **end-to-end (E2E)** form: it builds its own sequences and must be re-windowed onto the
adopted base; it is paired vs ours only if it ran on the same user-disjoint splits + same windowing, else
unpaired (see `STATISTICAL_PROTOCOL.md §4`).

### The non-negotiable rule — "original way" = original ARCH/EMBEDDING, NOT original DATA-PROTOCOL

> A baseline's **"original way"** means its original **architecture / embedding** — NOT its original
> data-protocol. **ALL** baselines run on **OUR data + OUR user-disjoint splits + OUR seeds {0,1,7,100} + OUR
> label spaces (7-root next-category macro-F1 + TIGER-tract next-region Acc@10) + the ADOPTED gated-overlap
> windowing (min_seq=10, stride-1).** Only the **embedding (run 3)** or the **architecture (run 4)** varies;
> everything in the data/eval harness is held to ours.

Running a baseline under its **own** windowing / splits / label space / metric breaks comparability — the
resulting Δ would confound the method difference with a data-protocol difference, and no paired test on it
would be valid. This is the same trap the B-A2 correction (RUN_MATRIX §0) and the windowing-matched-ceiling
condition (`../pre_freeze_gates/BOARD_ADOPTION_DECISION.md` Condition 3) guard against on our own arms: the
ceiling/baseline must be built on the **same windowing** as what it is compared to, or the comparison is void.
Concretely: a faithful E2E baseline keeps its backbone, hidden-state weighting, and auxiliary objectives, but
its **inputs are re-built from our windowed, user-disjoint, seeded folds** and it predicts into **our** label
spaces — documented deviations (beam-search dropped, geohash→TIGER tract, native split→our fold regime) are
recorded per baseline in `../baseline_gap/TRIAGE.md` and labelled (e.g. "HMT-GRN-STYLE", "Flashback-only").

### Baseline → form map (which rung(s) each baseline occupies)

| Baseline | Form(s) | Class | Question answered | Notes |
|---|---|---|---|---|
| **CTLE** (Lin 2021) | Substrate-column (run 3) | SC | does our embedding beat theirs at matched capacity? | CTLE's *native* usage IS a per-visit contextual substrate → its faithful form is exactly the SC column. Pretrain train-portion-only per fold. |
| **POI2Vec** (Feng 2017) | Substrate-column (run 3) | SC | " | static-embedding floor; emit fresh per-POI 64-d (in-repo POI2Vec is fclass-level — net-new). Train-portion-only per fold. |
| **skip-gram** (word2vec / check-in seqs) | Substrate-column (run 3) | SC | " | completes CTLE's canonical floor triplet. Train-portion-only per fold. |
| **one-hot-POI 64-d** | Substrate-column (run 3) | SC | " (absolute-zero floor) | zero-training; deterministic random projection / hashed POI id. |
| **STAN** (Luo 2021) faithful | End-to-end (run 4) | E2E | do we beat the published method? | re-windowed to overlap base; POI head → region head (Acc@10). |
| **ReHDM** (Li 2025) faithful | End-to-end (run 4) | E2E | " | re-windowed; CA/TX deferred (quadratic collaborator-pool scaling — footnote). |
| **POI-RGNN** (Capanema 2022) faithful | End-to-end (run 4) | E2E | " (next-category) | re-windowed; non-user-disjoint published caveat preserved. |
| **MHA+PE** (Zeng 2019) faithful | End-to-end (run 4) | E2E | " (next-category) | re-windowed. |
| **HMT-GRN-style** (Lim 2022) | End-to-end (run 4) | E2E | the sole external **MTL** row — beats the published shared-LSTM MTL design? | re-windowed; "HMT-GRN-STYLE" (beam-search/selectivity dropped, geohash→TIGER). |
| **Flashback** (Yang 2020) | End-to-end (run 4) | E2E | do we beat the published method on the sparse-trace reg axis? | re-windowed; Flashback-only (DeepMove deferred). |
| **cascade** (CSLSL/CatDM pattern) | **Substrate-column (run 3, pinned)** + faithful E2E deferred | SC (pinned) | does the cascade ordering beat our parallel dual-tower? | pinned SC variant reuses our heads, changes ONE factor (region head conditioned on predicted category); faithful CSLSL/CatDM E2E deferred to camera-ready. |

**Why both run 3 and run 4 are reported (they answer different questions).** Run 3 (substrate-column) isolates
the **embedding** at matched architectural capacity — it tells a reviewer that any advantage attributable to
the Check2HGI *substrate* is not merely "any contextualisation" (CTLE is the head-on contextual competitor;
POI2Vec / skip-gram / one-hot are the floor). Run 4 (end-to-end native) isolates the **whole system** —
whether the proposed model, embedding and heads together beat the published SOTA-equivalent method as a unit.
A baseline can lose at run 3 (our embedding wins at matched capacity) yet be a different story at run 4 (its
native architecture closes part of the gap), or vice-versa; reporting only one would let a reviewer dissolve
or inflate the contribution. Both are carried, each tested per `STATISTICAL_PROTOCOL.md` (paired Wilcoxon vs
our STL/MTL where folds match; Holm-corrected across the baseline × state × task grid).

## 3 · Counts

- **BRACIS-suite cells inventoried: 12** — RE-RUN 4 (T3★, T4, §0.4, §0.9) + T5 (RE-RUN, per-engine
  inclusion story-dependent) = **5 RE-RUN**; REUSE 2 (T1, F-arch); STORY-DEPENDENT 6 (T2, F1, F2, §0.8,
  T6, C1-panel). Cross-checked against PHASE1_VERDICT §3 — zero contradictions.
- **Paper-facing REQUIRED cells:** T1 REUSE · T2 STORY-DEP · T3 RE-RUN★ · T4 RE-RUN · T5 RE-RUN · F1 STORY-DEP.
- **External-baseline rows after the B1–B5 merge:** existing T5 rows (Majority, Markov-1-POI/region,
  POI-RGNN, MHA+PE, substrate probes, STL ceilings, STAN/ReHDM faithful, MTL rows) **+ NET-NEW: B1 CTLE,
  B2a POI2Vec, B2b skip-gram, B2c one-hot-POI 64-d (4 SC substrate-floor columns, Tier-1 INCLUDE);
  B3 HMT-GRN-style MTL (1 E2E row, INCLUDE); B4 cascade (1 row, INCLUDE, pinned SC); B5 Flashback
  (1 E2E row, INCLUDE, pinned Flashback-only)** = **7 net-new baseline rows/columns, all INCLUDE**
  (faithful-B4 CSLSL/CatDM E2E + DeepMove DEFERRED to camera-ready). B1/B2a/B2b/B2c are the CTLE
  canonical substrate-floor suite (one-hot/skip-gram/POI2Vec) measured against the contextual substrates.
- **Comparability split:** SC (auto-inherit frozen base) = B1, B2a, B2b, B2c, substrate probes, STL
  ceilings, pinned-SC-B4. E2E (mirror windowing) = B3, B5, STAN/ReHDM/POI-RGNN/MHA+PE faithful,
  faithful-B4 (deferred).

## 4 · Blocked-on note (what gates this RUN_MATRIX before sign-off)

1. **Overlapping-windows ADOPT/KEEP** (pre_freeze_gates) — base change. If ADOPTED: all sequences rebuild
   → every per-fold log_T rebuilds, T1 derived counts change, leak-audit re-runs, and **every E2E baseline
   (B3, B5, STAN/ReHDM/POI-RGNN/MHA+PE faithful, faithful-B4) must mirror the new windowing + budget ONE
   re-run.** Gates M0b (seeded log_T) and all reg-touching RE-RUN cells. **Must resolve before P2.** OPEN.
2. **G0.1 aligned-pairing** — the lone *recipe-changing* P0 gate; ≥0.3 pp either head → recipe→v17 (would
   re-pin the whole frozen constant). Freeze cannot commit while OPEN.
3. **B1–B5 per-engine INCLUSION** — which net-new baselines the board carries (this matrix's §2).
   Inventory decision, must pin at P1b. Triage now pins all 7 net-new rows as INCLUDE (B4 = SC cascade;
   B5 = Flashback-only; faithful-B4 + DeepMove deferred to camera-ready); user confirmation at P2.
4. **M0 artifact prerequisite** (does not block sign-off, blocks P3 execution): v14 substrate at CA/TX
   (H100 build) + GE (sync/verify) — FL/AL/AZ present; seeded per-fold log_T for {0,1,7,100} at every state
   (FL multi-seed to consolidate from A40; CA/TX/GE to build — built AFTER the windowing gate); HGI/DGI/
   HMRM/Time2Vec/Space2Vec/CTLE/POI2Vec substrate artifacts at states that lack them.
5. **CLOSED gates (no longer blocking, recorded for provenance):** C1 (PROMOTE, supportive diagnostic
   panel), A2 (substrate cat claim STRENGTHENED, ON NULL), A4 (ON NULL both axes, one-paragraph defusal),
   mtl_frontier R1/R2/R3/R10 (all null, champion G unchanged).
6. **AUTHOR-side, not a closing_data block:** restating §0.1 / the BRACIS paper-canon tables under the
   inverted thesis (PAPER_UPDATE rule — do not silently rewrite).
