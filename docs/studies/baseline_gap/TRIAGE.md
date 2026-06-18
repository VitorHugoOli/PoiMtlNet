# External-Baseline Triage (B1–B5) — DRAFT for P1b/P2 sign-off

> **STATUS: DRAFT, NOT FINAL.** Triage for the `baseline_gap` P1b decision (which external baselines the new
> paper's board carries). The kept set becomes rows/columns in
> [`RUN_MATRIX.md`](../closing_data/RUN_MATRIX.md) §2 and is signed off by the user at the **P2 freeze**. Grounded in
> `docs/research/baseline_gap_analysis.md` + `literature_review.md` + `project_positioning.md`,
> `docs/baselines/{BASELINE.md,README.md}`, [`PAPER_BASELINES_STRATEGY.md`](../../PAPER_BASELINES_STRATEGY.md),
> the `EmbeddingEngine` enum (`src/configs/paths.py`), `research/embeddings/`. Each candidate paper web-verified.
> Terminology held throughout: **next-category** (macro-F1) vs **next-region** (Acc@10), distinct from
> next-POI (the project deliberately has NO next-POI head). External literature baselines = SOTA-equivalent.

## Triage table

| ID | Baseline | Targets | Integration | Recommendation | Role in matrix | Tier | Comparability |
|---|---|---|---|---|---|---|---|
| **B1** | CTLE (Lin et al., AAAI 2021) | embedding/substrate | substrate-column | **INCLUDE** | The head-on novelty competitor — the only OTHER per-visit contextual location embedding in the comparison; answers "why is a hierarchical-infomax check-in substrate better than CTLE's contextual-transformer embeddings?" One cat column + one reg column. | **Tier-1** (paper-blocking) | **SC** — inherits frozen base via matched-head pipeline (same v14 slot REPLACED by CTLE 64-d vectors; same folds/seeds/windowing/labels). Drop-in, no sequence-building. **One obligation:** CTLE pre-trains on **train-portion-only per fold** (its MLM objective) → no transductive advantage; coordinate the fairness statement with pre_freeze_gates/A4. |
| **B2a** | POI2Vec (Feng et al., AAAI 2017) | embedding/substrate | substrate-column | **INCLUDE** | Static-embedding substrate floor against which BOTH contextual substrates (Check2HGI, CTLE) are measured. One cat + one reg column. | **Tier-1** | **SC**. **Pretrain protocol: TRAIN-PORTION-ONLY per fold** (parity with B1 CTLE + B2b skip-gram — both train-only for transductive fairness; a full-corpus POI2Vec would carry the same transductive asymmetry CTLE is held to). **NET-NEW caveat:** in-repo POI2Vec is an **fclass-level HGI input feature** (`research/embeddings/hgi/poi2vec.py`), NOT a standalone per-POI substrate column. B2a funds emitting a fresh **per-POI 64-d** column. |
| **B2b** | skip-gram / word2vec over check-in seqs | embedding/substrate | substrate-column | **INCLUDE** | Completes the canonical location-embedding baseline suite (CTLE's own one-hot/skip-gram/POI2Vec/TALE canon). One cat + one reg column. | **Tier-1** | **SC**. Train **train-portion-only per fold** (parity with CTLE); 64-d. Cheap (minutes/state). |
| **B2c** | one-hot-POI 64-d (random-projected / hashed) | embedding/substrate | substrate-column | **INCLUDE (recommend ADD)** | The **trivial zero-training absolute-zero** that completes CTLE's canonical substrate-floor triplet (**one-hot / skip-gram / POI2Vec**). With B2a+B2b it bounds the floor below every learned substrate; its absence is the only gap in the canonical floor suite. One cat + one reg column. | **Tier-1** (floor) | **SC**. NO training cost — a fixed 64-d one-hot (random projection / feature-hash of POI id) emitted per fold; matched `next_gru`, 6 states × {0,1,7,100} × 5f. Identical across folds (deterministic seed). |
| **B3** | HMT-GRN-style MTL (Lim et al., SIGIR 2022 / TORS 2023) | multi-task | end-to-end | **INCLUDE** | The **sole external MTL row** — calibrates "is the cross-attn dual-tower champion better than the canonical published shared-LSTM MTL design?" Today the MTL table has ZERO external MTL rows. | **Tier-2** (paper-blocking for the MTL story) | **E2E** — builds its own sequences, MUST mirror windowing/stride + user-disjoint folds + seeds + train-only priors + label spaces. **Documented deviations** ("HMT-GRN-STYLE", not faithful): beam-search/selectivity DROPPED (its regions are geohash auxiliaries to a next-POI MAIN task; here category+region are headline, no next-POI head), geohash→TIGER tract, native per-user 80/20 split → our fold regime. Blocked on windowing + freeze. |
| **B4** | Cascade category→region (CSLSL / CatDM pattern) | multi-task | substrate-column (SC) | **INCLUDE — pin SC cascade** | The dominant PUBLISHED alternative to parallel MTL — predict category, condition region on predicted category. One row testing "does the cascade beat the parallel dual-tower?" | **Tier-2** (SC variant); faithful E2E = Tier-3 deferred to camera-ready | **SC (pinned).** **SC cascade** over the frozen substrate reusing `next_gru` (cat) + `next_stan_flow`/`next_lstm` (reg) — wire predicted-category into the region head → inherits windowing/folds/labels, isolates "cascade vs parallel" cleanly. This is a **controlled cascade-vs-parallel isolation, NOT a faithful CSLSL/CatDM reproduction** (see framing below). Faithful CSLSL/CatDM reimpl = **E2E**, DEFERRED to camera-ready. No next-POI head. |
| **B5** | Flashback (IJCAI 2020) / DeepMove (WWW 2018) | next-region | end-to-end | **CONDITIONAL** (recommend Flashback-only) | Additional sequential next-region STL reference beyond STAN + ReHDM. Flashback targets the SPARSE-TRACE regime (AL/AZ) → the most defensible single add. | **Tier-2** (lowest marginal value — reg axis already best-covered) | **E2E** — swap POI head → region head (Acc@10), mirroring the repo's existing STAN region adaptation; keep Flashback's spatiotemporal hidden-state weighting / DeepMove's historical attention faithful; only the target space changes. Blocked on windowing + freeze. Recommend **Flashback ONLY**; DEFER DeepMove to Tier-3. |

## Summary / verdict

- **INCLUDE B1 (CTLE) and B2 (POI2Vec + skip-gram)** as **substrate columns** — both drop-in 64-d
  embeddings that inherit the frozen v14 base through the matched-head pipeline (`next_gru` cat /
  `next_stan_flow` reg), comparability guaranteed by construction, both **Tier-1 paper-blocking**. Without
  B1, the headline substrate claim ("per-visit contextual embeddings carry next-category") cannot be
  attributed to the hierarchical-infomax design vs *any* contextualization — a reviewer can dissolve the
  contribution in one line. B2 completes the canonical static-substrate floor — **B2a POI2Vec + B2b
  skip-gram + B2c one-hot-POI 64-d** together are CTLE's canonical substrate-floor triplet
  (one-hot/skip-gram/POI2Vec). B2c is the trivial zero-training absolute-zero; **recommend ADD** (no
  training cost — a fixed 64-d one-hot/hashed-id projection emitted per fold).
- **INCLUDE B3 (HMT-GRN-style)** as the **sole external MTL row** — end-to-end, must mirror
  windowing/folds/labels, **documented as an adaptation** (regions-as-tools dropped; "HMT-GRN-STYLE"),
  mirroring the existing STAN-Flow/GETNext naming-conflict precedent in PAPER_BASELINES_STRATEGY.md.
- **INCLUDE B4 (cascade) — pinned as the cheap SC cascade** reusing existing heads (`next_gru` cat +
  `next_stan_flow`/`next_lstm` reg) for a clean "cascade vs parallel" isolation. Framed explicitly as a
  **controlled isolation, NOT a faithful CSLSL/CatDM reproduction** (see the framing paragraph below);
  the costly faithful CSLSL/CatDM E2E reimpl is **DEFERRED to camera-ready** (Tier-3).
- **INCLUDE B5 — pinned as Flashback-only.** Lowest marginal value (STAN + ReHDM already anchor the
  next-region axis multi-state), but Flashback is well-justified for the **sparse AL/AZ traces** (its
  spatiotemporal hidden-state weighting is designed for sparse regimes). **DeepMove DEFERRED to
  camera-ready.**
- **KEY COMPARABILITY SPLIT:** SC baselines (B1, B2a, B2b, B2c, pinned-SC-B4) inherit the base
  automatically; **E2E baselines (B3, B5, faithful-B4)** build their own sequences and are **BLOCKED on
  the overlapping-windows ADOPT/KEEP decision + the P2 freeze**, with ONE budgeted re-run if overlap is
  adopted.
- **B2 net-new caveat** (load-bearing): in-repo POI2Vec is fclass-level (an HGI input), NOT a standalone
  POI-level column — B2 is genuinely net-new, a separate RUN_MATRIX decision, not "already present."
- **Gap-analysis cross-walk (Tier-1 #2 = feature-concat control):** `baseline_gap_analysis.md §2`
  Tier-1 #2 (HGI ⊕ raw per-visit features → same heads) is **`pre_freeze_gates/A2`, RESOLVED ON NULL
  (substrate claim STRENGTHENED)** — concat lifts cat ≤2 pp / closes <10% of the gap, inert on reg
  (`pre_freeze_gates/A2_RESULTS.md`). It is an **interpretation gate, not a baseline row**, so it
  correctly does NOT appear as a B1–B5 entry; this note makes the cross-walk clean (it otherwise reads
  as a possible omission). F1 may be backed by `A2_RESULTS.md` rather than re-run (RUN_MATRIX F1).
- **B1 transductive coupling:** CTLE pre-trains train-portion-only per its protocol while the Check2HGI
  substrate currently trains transductively — coordinate with pre_freeze_gates/A4 to report the asymmetry
  as a measured fairness gap (A4 RESOLVED ON NULL: downstream inflation ≈0 both axes — so the asymmetry is
  small, but the fairness statement must still be coherent).
- **Second-dataset scope:** the full B1–B5 suite is OUT of scope for the Massive-STEPS (NYC/Istanbul)
  validation (champion G + STL ceilings + Markov floor only) unless the user explicitly pulls one in.
- All five candidates target embedding / MTL / next-region; **none targets next-POI**. Decision is
  cheap and pre-freeze; runs trail the P2 freeze and fold into P3 (M1/M3).

## B4 framing — controlled cascade-vs-parallel isolation (NOT a faithful reproduction)

The **pinned B4** is a **controlled cascade-vs-parallel isolation**, not a faithful CSLSL/CatDM
reproduction — stated up front to pre-empt the faithfulness objection. It reuses our own heads
(`next_gru` for the category prediction, `next_stan_flow`/`next_lstm` for the region prediction over the
frozen v14 substrate) and changes exactly ONE factor versus the champion-G parallel dual-tower: the
region head is conditioned on the **predicted category** (predict next-category → wire that signal into
the next-region head) instead of being trained in parallel. Everything else — substrate, windowing,
folds, seeds, label spaces (7-root next-category macro-F1 + TIGER-tract next-region Acc@10) — is held
identical, so the comparison isolates "does the cascade ordering beat the parallel ordering?" cleanly.

This is deliberately **not** an end-to-end CSLSL/CatDM reimplementation (those carry their own
backbones, auxiliary objectives, and hyperparameters that would confound the cascade-vs-parallel
contrast). A **faithful CSLSL/CatDM E2E reproduction is DEFERRED to camera-ready** (Tier-3, blocked on
the overlapping-windows decision); the controlled SC isolation is what the board carries now.

## Fairness ledger (stub) — pretrain protocol × tuning budget × parity

Pre-registers the fairness statement each baseline is held to. The headline **substrate claim** rests on
HGI being tuned at parity with Check2HGI (`baseline_gap_analysis.md §1.4`: "HGI had historical per-state
tuning; an explicit tuning-budget statement is needed for the fairness claim") — so the HGI-substrate
comparator row is INCLUDED here. To be completed at P2 with the actual HP-search counts / epochs.

| Baseline | Pretrain protocol | Tuning budget (HP-search · epochs · per-state vs global) | Parity note |
|---|---|---|---|
| **Check2HGI v14** (ours) | transductive (full-corpus substrate) | frozen recipe; champion G; {0,1,7,100} × 5f | the reference; transductive asymmetry vs train-only baselines bounded NULL by `pre_freeze_gates/A4` |
| **HGI** (substrate comparator) | transductive (full-corpus) | **historical per-state tuning — MUST be stated at P2** (§1.4 fairness gap) | headline substrate claim rests on HGI tuning parity; document the budget explicitly |
| **B1 CTLE** | **train-portion-only per fold** (MLM objective) | TBD (per-state vs global at P2) | coordinate transductive statement with `pre_freeze_gates/A4` |
| **B2a POI2Vec** | **train-portion-only per fold** | TBD | parity with B1/B2b (train-only) |
| **B2b skip-gram** | **train-portion-only per fold** | TBD | parity with B1/B2a (train-only) |
| **B2c one-hot-POI 64-d** | none (zero-training floor) | n/a | deterministic; absolute-zero floor |
| **B3 HMT-GRN-style** | train-only priors per fold | TBD (equal-weight CE) | E2E; mirror windowing/folds/seeds/labels |
| **B4 SC cascade** | inherits frozen substrate | reuses tuned `next_gru` + `next_stan_flow`/`next_lstm` | controlled isolation (one factor changed) |
| **B5 Flashback** | train-only per fold | TBD | E2E; sparse-trace fit (AL/AZ) |
