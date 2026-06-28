> ⛔ **ARCHIVED 2026-06-28 — the work this document drove is DONE.** Kept for provenance, not live
> state. Current status and pointers: [`../CLAUDE.md`](../CLAUDE.md). (Relative links below may be off by one
> directory level after the move to `archive/`.)

# MobiWac 2026: Baseline Selection Audit (2026-06-24)

> ⚠ **UPDATE 2026-06-25 (supersedes the "we beat the cascade" framing in the body):** the CSLSL cascade was run
> (AL/AZ/FL, canonical dk_ovl 5f) and is a **DEAD TIE** with our parallel model (Δjoint ≈ 0). So the cascade is
> **not** "the number that converts 'we beat our own configs' into 'we beat the published design'" as the body
> says; it is a **defense** ("a cheaper cascade would not have matched our STL-ceiling lift"), a **tie at equal
> cost**, never a win. Read every "beat the cascade / dominant published design" line below as "match it at equal
> cost." Contribution 2 is anchored to the §1 STL-ceiling lift, not the cascade cell.

> Critical evaluation of the proposed baseline set (next-category: CTLE + POI-RGNN; next-region: HMT-GRN +
> STAN + Markov-1). Method: 4 readers (status/fairness, claim->baseline map, web field-expectation,
> beat-margins) + 2 adversarial advisors (minimum-credible-set, tough-reviewer) + synthesis. Internal
> decision-support; numbers read from committed JSONs and RESULTS_BOARD.

---

## Synthesis (decision-ready recommendation)

# VERDICT
The proposed set is **directionally right but incomplete and mis-framed**: keep all five, but it under-guards the two seams reviewers reject on (feature-injection confound, all-internal MTL) and ships CTLE's +37.8 in its weakest possible form — it needs 3 cheap additions and one mandatory reframe before it is safe.

# KEEP (load-bearing, all four reports concur)
- **CTLE** — non-negotiable W3 novelty gate (gap_controls, tough #1). But it is load-bearing *only* re-cast as a substrate column ("CTLE-emb → matched head"), never as a "next-category model." Web_field confirms the field (AdaptGOT 2025) already documents CTLE underperforms category by design; status confirms Check2HGI-SC (55.59) is the correct matched comparand, not the MTL champion.
- **POI-RGNN** — strongest-covered baseline (faithful, all 6 states, stricter SGKF folds → conservative floor); the external category anchor (status Q4, margins). Cite canonical 34.49/31.78/33.03, not the superseded comparison.md snapshot (margins #6).
- **HMT-GRN** — the *only* external MTL/region precedent; sole defense against "all comparisons internal" (gap_controls, tough #4). Keep — **all 6 states done incl. TX (PR #38: reg 53.85 / cat 25.81)**; the Mac device-confound is **RESOLVED** (PR #38: MPS == deterministic CPU within 0.06 pp; the old 62.37 was the unreproducible anomaly). ⚠ It is **HMT-GRN-*style*** (own LSTM trunk + train-only region-transition prior from raw; graph module + hierarchical beam search dropped, no next-POI head) — frame it as "region-native E2E", **not "faithful HMT-GRN"** (deviation ledger: `docs/baselines/next_region/hmt_grn.md`).
- **STAN** — keep; but its `stl_hgi` variant EXISTS at CA/TX (CA 60.45, TX 62.70) and is the *tightest* large-state competitor (margins #3). Do NOT label STAN "infeasible at CA/TX" — that is true only for `faithful`.
- **Markov-1 region** — correct universal floor, beaten +12..23 pp everywhere (status Q3). Keep.

# ADD (ranked; claim protected + cost)
- **BLOCKER — SC control ladder (one-hot / skip-gram / POI2Vec → matched head).** Protects C1-novelty; converts CTLE's +37.8 from "rigged strawman" into a principled ladder (one-hot 19.5 < CTLE 17.8 < POI2Vec 22.5 < skip-gram 24.2 ≪ Check2HGI 55.6). Cost: near-zero — already computed at AL; finish AZ+. (status Q1, gap_controls #3, margins #5, both advisors #1.)
- **BLOCKER — feature-concat control (HGI ⊕ raw per-visit features → same head).** Protects C1-*mechanism* — the one control CTLE cannot substitute for, since Check2HGI's node features literally *are* category one-hot + sin/cos time. Without it the +36..48 pp lift could be feature injection, not the hierarchy. Cost: no new embedding training (concat existing vectors). Named "the biggest internal-logic hole" by the gap analysis. (gap_controls #1, tough #5, minimal #2.)
- **IMPORTANT — ReHDM-faithful (AL/AZ/FL) back in the region table.** Protects C3 region-cost-vs-SOTA and kills "why is the 2025 SOTA missing?" It is run, validated, leak-clean, and is your *tightest* small-state competitor (AL 66.06, only +3.75). Cost: zero new compute — already in repo. Dropping a passing 2025 SOTA while keeping weaker 2021 STAN is indefensible. (all four reports; tough #3, web_field #1.)
- **IMPORTANT — cascade category→region baseline (CSLSL/CatDM pattern).** Protects C2 — the only number that converts "we beat our own configs" into "we beat the dominant published joint design." Cost: one new from-scratch baseline (moderate). HMT-GRN is region-only E2E and does not rebut the cascade attack. (gap_controls #2, web_field #3, tough #4.)
- **IMPORTANT — CTLE-SC at FL/CA/TX (CUDA) + CTLE-E2E companion note.** Protects C1 where the region win actually lives — the gate is currently unmeasured at the 3 largest states. CTLE-E2E (~21.2) beside CTLE-SC (17.8) disarms the frozen-vs-finetuned confound. Cost: low (CUDA runs already queued, `HANDOFF_A40_CTLE_SC.md`). (status, margins #2.)
- **NICE — joint-vs-Markov rows + majority-class category floor.** Protects the "beats trivial baselines" claim for the *joint* model the paper headlines (currently shown only for the dedicated model) and restores category/region floor symmetry. Cost: tabulation only, numbers exist. (gap_controls #4, tough #6.)
- **NICE — one learned category-native sequential baseline (GETNext-cat-head / SGRec / CSLSL).** Protects against "thin category set for a 2026 venue" once CTLE is demoted to a substrate column. Cost: moderate. Lower priority than the cascade, which double-counts for C2. (web_field #2, tough #8.)

# DROP or SKIP (safe given the deadline)
- **MHA+PE** — SKIP as a headline row (drop), but it is already run (FL 32.06) so retain as a one-line second category point if free; redundant with POI-RGNN (gap_controls, minimal).
- **STAN/ReHDM *faithful* at CA/TX** — SKIP; legitimately "infeasible at scale," footnote it and use STAN `stl_hgi` for the CA/TX region point (status, margins #3).
- **CTLE-E2E full tabulation** — SKIP; one companion note row beside SC suffices (minimal).
- **LLM zero-shot row, full 2022-25 graph wave (GETNext/STHGCN as run baselines)** — SKIP; ReHDM + a one-line lineage acknowledgment answers the datedness jab (web_field, tough #10).
- **New from-scratch region SOTA training** — SKIP; nothing not already run.

# THE CTLE FAIRNESS QUESTION
**The +37.8 is real arithmetic but NOT defensible as currently framed — all four reports agree.** The kill-shot evidence is internal: CTLE-SC (AL 17.77 / AZ 19.30) scores *below* the count-based Markov-K bigram floor (AL 20.50 / AZ 23.92), below MHA+PE (18.95), and below the one-hot-64 control (19.50) (margins #1, status Q1). A frozen AAAI'21 contextual embedding losing to a stupid-backoff bigram reads as a crippled harness, not genuine inferiority — and CTLE is a next-POI method probed on next-category through a foreign head on cold-user folds (a 4-way protocol shift). Web_field confirms this is *expected direction* per the literature (AdaptGOT: "CTLE biases towards the next-POI task") but the *magnitude* invites the strawman charge.

To make a reviewer believe it, do all four:
1. **Present CTLE as one substrate column, never a "next-category model"** — the comparison is CTLE-SC vs Check2HGI-SC (55.59), matched head, matched folds, matched rows at min_seq=10 (the row-parity fix `b195b89f` is documented). Both are frozen → the protocol is symmetric, which is the actual fairness defense.
2. **Show the SC control ladder** so CTLE's number reads as "frozen contextual embeddings are uniformly weak here, ours is the lone exception," not "we crushed CTLE."
3. **Quote the honest headline as ~+31 pp over the strongest control (skip-gram 24.2)**, not the bare +37.8 over crippled-frozen-CTLE (status, minimal).
4. **Report CTLE-E2E (~21.2) beside CTLE-SC (17.8)** to make the frozen-vs-finetuned confound visible and disarmed, and footnote that CTLE-SC < Markov-K is a frozen-substrate/task-mismatch artifact.

# THE SINGLE BIGGEST BASELINE RISK
**The all-internal MTL story (no external/cascade comparator for C2).** Resolving the advisor conflict explicitly: the **minimalist** advisor ranks the CTLE strawman #1 and treats the cascade as second-tier ("footnote otherwise"); the **tough** advisor flags two reject-grade holes — #1 CTLE strawman AND #4 no cascade — and I side with tough here. The CTLE strawman is *cheap to fully defuse* (the SC ladder is already computed; it is a framing fix), so it will be closed before submission. The cascade hole is *structural and cannot be reframed away*: every comparison that matters is champion-G vs the authors' own STL ceilings and own ablations, HMT-GRN is run region-only (not joint cat+region), and the field's dominant alternative to parallel cross-attn MTL — the CSLSL/CatDM category→region cascade — has no number. For an MTL paper that is the first seam a reviewer pulls, and no presentation trick fixes it. **Fix: run the cascade category→region baseline** (one moderate-cost run; it double-counts as the "external/recent category model" the web report also wants). Second-priority within this same risk: get one device-consistent external region baseline at CA/TX (re-run HMT-GRN on CUDA) so the headline large-state region win is not anchored on device-confounded Mac numbers plus a Markov floor.

Key files: `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/next_category/ctle.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/next_region/comparison.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/research/baseline_gap_analysis.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/alabama_{ctle,check2hgi_sc,b2b,b2c,poi2vec}.json`, `/Users/vitor/Desktop/mestrado/ingred/docs/studies/closing_data/RESULTS_BOARD.md`, `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/PAPER_PLAN.md` (§7 feature-concat checkbox), `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/REVIEW_PANEL.md` (W2/W3).
---

## Beat-margins (per state, per task) + critical assessment

All numbers verified against source JSONs. I have everything needed to build the two tables and the critical assessment.

---

# Baseline verification — MobiWac 2026, per state, per task

All numbers below are read directly from the committed JSONs/docs (paths inline). "Ours-MTL" = champion-G from `RESULTS_BOARD.md §1`; "Ours-STL" = the dedicated single-task ceiling. **All ours-numbers are n=5 (seed 0) provisional**; the §0.1 canonical (n=20, seeds {0,1,7,100}, frozen GCN substrate, no-overlap) gives different absolute values — flagged where it changes a verdict.

## TABLE A — next-CATEGORY (macro-F1)

| State | Ours-STL cat | Ours-MTL cat | CTLE-SC | POI-RGNN | MHA+PE | Δ vs CTLE | Δ vs POI-RGNN | Δ vs MHA+PE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AL | 55.87 | **63.56** | 17.77 | 23.80 | 18.95 | **+45.8** | **+39.8** | **+44.6** |
| AZ | 57.13 | **63.39** | 19.30 | 27.64 | 24.99 | **+44.1** | **+35.8** | **+38.4** |
| FL | 75.15 | **79.82** | _→CUDA_ | 33.35 | 32.06 | _pending_ | **+46.5** | **+47.8** |
| CA | 70.26 | **77.33** | _→CUDA_ | 30.71 | 29.13 | _pending_ | **+46.6** | **+48.2** |
| TX | 69.95 | **77.50** ²ᐟ⁵ | _→CUDA_ | 32.08 | 29.91 | **+45.4** ²ᐟ⁵ | **+47.6** | _provisional, TX MTL 2/5 folds_ |

We win category **everywhere, by enormous margins (+36..+48 pp)**. Sources: `comparison.md` + per-state JSONs (POI-RGNN/MHA+PE byte-match), `ctle.md` (CTLE-SC). All margins are comfortable to the point of being a problem (see critique).

## TABLE B — next-REGION (Acc@10)

| State | regions | Ours-STL reg | Ours-MTL reg | HMT-GRN | STAN (best variant) | ReHDM (best) | Markov-1 | Δ vs HMT-GRN | Δ vs STAN | Δ vs Markov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AL | 1109 | 69.99 | 69.81 | 57.05 | 62.88 (stl_hgi) | 66.06 (faithful) | 47.01 | **+12.8** | **+6.9** | +22.8 |
| AZ | 1547 | 59.40 | 59.34 | 43.70 | 54.86 (stl_hgi) | 54.65 (faithful) | 42.96 | **+15.6** | **+4.5** | +16.4 |
| FL | 4703 | 76.71 | 77.28 | 63.74 | 73.58 (stl_hgi) | 65.68 (faithful) | 65.05 | **+13.5** | **+3.7** | +12.2 |
| CA | 8501 | 63.48 | 65.66 | 49.61 | **60.45 (stl_hgi)** | — (deferred) | 52.09 | **+16.1** | **+5.2** | +13.6 |
| TX | 6553 | 64.96 | 67.13 ²ᐟ⁵ | 53.85 | **62.70 (stl_hgi)** | — (deferred) | 54.94 | **+13.3** ²ᐟ⁵ | **+4.4** ²ᐟ⁵ | +12.2 |
| Istanbul | 520 | 70.37 | 69.79 | 56.56 | — | — | 52.5 | **+13.2** | n/a | +17.3 |

We win region against the **native-E2E** baselines (HMT-GRN, Markov) everywhere. Sources: `comparison.md`/JSONs (STAN, Markov), `hmt_grn.md` (HMT-GRN).

---

# CRITICAL ASSESSMENT — is this the best baseline call?

## 1. The CTLE +37 pp gap is a FAIRNESS LANDMINE, not a win — and it is your headline novelty gate

This is the single most dangerous cell in the whole table. CTLE-SC cat = **17.77 (AL) / 19.30 (AZ)**. Cross-check against your own floors at the *same state*:
- **Markov-K-cat floor (count-based, no learning): AL 20.50, AZ 23.92** — CTLE scores **BELOW the N-gram floor**.
- **MHA+PE faithful: AL 18.95, AZ 24.99** — CTLE is below MHA+PE too.
- **Majority-class floor: AL 7.28** — CTLE is above only the trivial floor.

A contextual location-embedding model from AAAI 2021 scoring *below a stupid-backoff bigram* on next-category is a screaming sign the baseline is **crippled, not genuinely worse.** CTLE was designed for next-POI; you've frozen its 64-d embedding and bolted on your head for next-category. The prompt's worry ("is CTLE crippled on next-category?") is confirmed by the data: it loses to a bigram. A reviewer will not accept "+37 pp" as evidence that *the hierarchy* helps — they'll read it as "the authors handicapped the competitor." Your own `ctle.md` even notes CTLE-E2E (21.2) > CTLE-SC frozen (17.75), i.e. **the frozen-SC protocol itself depresses CTLE by ~3.5 pp**, and that is the protocol you're comparing against. The +37 is partly a protocol artifact.

What makes this worse: CTLE is the **non-negotiable W3 novelty gate** (`REVIEW_PANEL.md` #2). You can't drop it. So the action is not "remove it" but "**de-risk it**": report CTLE-E2E alongside CTLE-SC, add a per-visit feature-concat control so the gap is attributable to the *hierarchy* and not to *any* contextualization, and explicitly footnote that CTLE-SC < Markov-K is a frozen-substrate/task-mismatch artifact. Note also you compare CTLE-SC against **Check2HGI-SC (55.59), not the MTL champion (63.56)** — that's the correct matched-capacity comparand, so keep the novelty claim on the SC row, not the headline MTL row.

## 2. CTLE / POI-RGNN are ABSENT exactly where your strongest claim lives (CA/TX region — and category too)

Your load-bearing new result is "**CA, the largest region state, beats both ceilings**" (retires the cardinality story). But:
- **CTLE-SC**: only AL/AZ leak-clean. **FL/CA/TX = "→ CUDA, pending."** The novelty gate is unmeasured at 3 of 6 states, including the two biggest.
- **POI-RGNN / MHA+PE** do cover CA/TX for *category* (CA 30.71 / TX 32.08) — fine. But you have **no external category baseline run under the new gated-overlap board** — POI-RGNN/MHA+PE are the older category-native study, a *different protocol* (the comparison.md banner says so explicitly). So Table A mixes protocols: ours = stride-1 overlap board, baselines = older study. That cross-protocol gap will be challenged.

If CA/TX is where you plant the flag, the baselines must be present *at CA/TX under the matched protocol*. Right now they are not for CTLE (the gate) at all, and POI-RGNN/MHA+PE are protocol-mismatched.

## 3. You are dropping STAN where it is your STRONGEST region competitor — this is a self-inflicted wound

Look at the region table. **STAN `stl_hgi` is by far the closest baseline to us on region**, and it is the only one that scales:
- FL: STAN 73.58 vs ours 77.28 → **only +3.7**.
- TX: STAN 62.70 vs ours 67.13 → **+4.4** (and TX MTL is only 2/5 folds — the margin could move).
- CA: STAN 60.45 vs ours 65.66 → **+5.2**.
- AZ: STAN 54.86 vs ours 59.34 → **+4.5**.

The proposed selection keeps STAN — good — **but the proposal flags "STAN infeasible at CA/TX"**, which is WRONG for the variant that matters. STAN `faithful` is infeasible at scale, but **STAN `stl_check2hgi`/`stl_hgi` are computed at all 5 states** (JSON-confirmed: CA 60.45, TX 62.70). So STAN IS available at CA/TX via the STL variants. Do **not** describe STAN as absent at CA/TX — it's your tightest comparison there and it exists. The thin margins (FL +3.7) live entirely in the STAN row; that's where the paper is won or lost on region.

## 4. ReHDM (IJCAI 2025 SOTA) is being dropped, and at AL it BEATS your STL ceiling and ties your MTL

This is the proposal's biggest risk. ReHDM `faithful`:
- **AL: 66.06.** Your STL reg ceiling = 69.99, MTL = 69.81. Margin = **only +3.75 (MTL) / +3.93 (STL)**, *and* ReHDM is on its own paper protocol (chronological, 5 seeds) while you're on n=5 seed-0 folds — within plausible noise overlap. ReHDM **beats** the closest other baseline (STAN stl_hgi 62.88) at AL.
- **AZ: 54.65** vs ours 59.34 → +4.7.
- **FL: 65.68** vs ours 77.28 → +11.6 (comfortable).

Dropping the **most recent SOTA (IJCAI 2025)** is the kind of omission a reviewer flags immediately — "why is the 2025 region SOTA missing?" And it's your *thinnest* region competitor at AL. ReHDM is deferred at CA/TX (compute), but it's faithful at AL/AZ/FL. **Recommendation: keep ReHDM in the region table, not drop it.** Replacing STAN-or-ReHDM with the other is a false economy; you want both, because together they bracket the small-state (ReHDM strong) and large-state (STAN strong) regimes.

## 5. The "biggest internal-logic hole" (feature-concat control) is still absent

The gap analysis (`baseline_gap_analysis.md §2 Tier-1`) names the **HGI ⊕ raw per-visit features → same head** control as "the biggest internal-logic hole in CH16," plus skip-gram/POI2Vec/one-hot standalone substrates as the canonical embedding-paper control set. None are in the proposed selection. The +37 CTLE gap (critique #1) is *exactly* the claim this control defends. Without it, a reviewer says "Check2HGI's win is just feature injection, not the hierarchy." A `poi2vec` compare JSON already exists (`baseline_compare/alabama_poi2vec.json`) — at least partially runnable cheaply.

## 6. POI-RGNN provenance caveat

`comparison.md` carries a SUPERSEDED banner: the values in that file (33.35/30.71/32.08) are pre-bugfix May-2 snapshots; the canonical POI-RGNN is **34.49/31.78/33.03** (`RESULTS_TABLE §0.6` / `POI_RGNN_AUDIT.md`). The JSONs in `results/` match the older file, not the canonical. **Use the canonical POI-RGNN numbers** in the paper; even at 34.49 (FL) the margin vs ours (79.82) is +45, so the win is unaffected — but cite the right number.

---

# Bottom line

**Category:** we win everywhere by +36..+48 pp. The margins are *too large to be credible* on the CTLE row (CTLE < bigram floor) — that's a fairness liability, not a strength. CTLE must stay (it's the W3 gate) but needs the feature-concat + skip-gram/POI2Vec controls and a CTLE-E2E companion, and must be measured at FL/CA/TX. POI-RGNN/MHA+PE wins are real and comfortable but protocol-mismatched to the new board.

**Region:** we win against native-E2E (HMT-GRN +13..+16, Markov +12..+23) comfortably. But the **thin margins all live in STAN (FL +3.7, TX +4.4) and ReHDM (AL +3.75)** — the two baselines the proposal under-rates. STAN is NOT infeasible at CA/TX (the STL variants exist: CA 60.45, TX 62.70) and is your tightest large-state competitor; ReHDM is the 2025 SOTA and your tightest small-state competitor (AL within ~3.8 pp, overlapping protocols).

**Verdict on the proposed selection:** Category set (CTLE + POI-RGNN) is acceptable *only if* CTLE is de-risked with controls and extended to large states; add MHA+PE (it's already run, cheap, and a second category point). Region set should be **HMT-GRN + STAN + ReHDM + Markov-1 — do NOT drop ReHDM**, and do NOT label STAN absent at CA/TX. The riskiest individual cells: **CTLE (+37, implausibly large, below floor, AL/AZ only)**, **ReHDM-AL (only +3.75, dropping the 2025 SOTA)**, **STAN-FL (+3.7, thin)**, and **TX MTL (2/5 folds — every TX margin is provisional)**.

Key files: `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/next_category/ctle.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/next_region/comparison.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/research/baseline_gap_analysis.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/alabama_ctle.json`.
---

## Baseline status, fidelity & fairness

I have everything needed. The MTL champion region beats Markov-1 at every state by +12.2 to +22.8 pp. I have the full fairness picture on CTLE, including the decisive control-baseline evidence (CTLE-SC scores below one-hot). Let me compose the final report.

# BASELINE AUDIT — MobiWac 2026 (multi-task POI: next-category macro-F1 + next-region Acc@10)

## Per-baseline status table

| Baseline | Task | Fidelity | Current numbers (per state, mean±σ) | State coverage GAPS | Leak status | Fairness concern |
|---|---|---|---|---|---|---|
| **CTLE-SC** (Lin AAAI'21) | next-cat (substrate-isolation gate) | On-substrate, **frozen** CTLE emb → our `next_gru` head | AL 17.77±1.47 · AZ 19.30±0.95 (macro-F1) | **FL/CA/TX missing** (→ CUDA, `HANDOFF_A40_CTLE_SC.md`); Istanbul → Mac | Leak-clean AL/AZ only (per-fold train-only pretrain; the original `--folds 1` leaked 81.8% val users — closed) | **SEVERE — see Q1.** Frozen-SC scores *below a one-hot control*; reviewers will read it as crippled. |
| **CTLE-E2E** | next-cat | Native end-to-end (CTLE's own transformer fine-tuned) | AL ≈ 21.2 (note only, not tabulated) | All except AL | same as SC | The honest read of CTLE's true strength; NOT in proposed selection. |
| **POI-RGNN** (Capanema AdHoc'23) | next-cat | Faithful reimpl (RNN+GNN, paper arch from raw inputs) | AL 23.80±1.12 · AZ 27.64±2.34 · FL 33.35±1.14 · CA 30.71±0.82 · TX 32.08±0.70 · GA 30.24±0.87 | **5-fold complete, all states** ✅ | User-disjoint `StratifiedGroupKFold` (stricter than paper's per-user KFold) | Fair; conservative (our folds stricter → our Δ is a floor). But it's a 9-class category transition baseline, NOT an embedding baseline. |
| **MHA+PE** (Zeng IDEAL'19) | next-cat | Faithful reimpl (GRU+self-attn) | AL 18.95 · AZ 24.99 · FL 32.06±0.23 · CA 29.13 · TX 29.91 · GA 27.62 | **5-fold complete, all states** ✅ | User-disjoint SGKF | Fair. Weaker than POI-RGNN; NOT in proposed selection. |
| **HMT-GRN** (Lim SIGIR'22) | next-region (native E2E) | On-substrate-free native E2E; adapted {next-POI,region}→{cat,region} | AL 57.05 · AZ 43.70 · FL 63.74 · CA 49.61 · Istanbul 56.56 · **TX in-flight** | TX pending; numbers are **[M4/MPS], device-confounded vs CUDA champion** | Per-fold train-only region prior; leak-safe | Device mismatch (Mac HMT vs CUDA champion). Old AL 62.37 was an anomaly; current 57.05 ≈ CPU within 0.06pp. |
| **STAN** (Luo WWW'21) | next-region | `faithful` reimpl + `stl_check2hgi`/`stl_hgi` substrate variants | faithful: AL 34.46 · AZ 38.96 · FL 65.36±0.69 · GA 40.68; stl_hgi: AL 62.88 · AZ 54.86 · FL 73.58 · CA 60.45 · TX 62.70 | **faithful MISSING at CA/TX** (⚪ "infeasible at scale", `_pending_faithful` in JSONs) | Substrate variants leak-free; faithful leak-free | faithful is below Markov-1 at AL/AZ — a weak from-scratch number. Substrate variants are head-coupled, not headline. |
| **ReHDM** (Li IJCAI'25, region SOTA) | next-region | `faithful` (paper protocol: chrono 80/10/10, 5 seeds) + substrate variants | faithful: AL 66.06±0.98 · AZ 54.65±0.77 · FL 65.68±0.26 · GA 55.82 | **CA/TX deferred** (⚪ ~75–120 h/state); σ is inter-seed (not fold-comparable) | Faithful leak-clean (3 bugs fixed incl. target leakage); substrate variants leak-free | **NOT in proposed selection** — but it is the strongest region number at AL (66.06) and the only 2025 SOTA. Its absence is a hole (see below). |
| **Markov-1-region** (floor) | next-region | Exact count-based | AL 47.01 · AZ 42.96 · FL 65.05 · CA 52.09 · TX 54.94 · Istanbul 52.5 · GA 48.19 | All states ✅ | n/a | Fair floor. |
| **Markov-K-cat** (floor) | next-cat | Exact, same 9-step window | best-K: AL 20.50 · AZ 23.92 · FL 29.74 · CA 27.59 · TX 28.67 | All states ✅ | n/a | The honest apples-to-apples cat floor (POI-RGNN only beats it +3.1–3.7pp). |
| SC controls: **skip-gram (b2b)** 24.20 · **POI2Vec (b2a)** 22.53 · **one-hot-64 (b2c)** 19.50 (AL macro-F1) | next-cat | On-substrate, frozen → our head | AL only | AL only (AZ+ pending) | leak-clean | These are the embedding-paper control set the gap analysis demanded. **Decisive context for CTLE — see Q1.** |
| HAVANA / PGC | POI *labeling* (not sequential) | numbers-only / faithful | FL ~62.9 / 40.79 | n/a | n/a | Different task (labeling). Not a sequential baseline. Correctly out of scope. |

---

## Load-bearing answers

### Q1 — WHY does CTLE lose by +37.8 macro-F1, and would a reviewer believe it?

**The +37.82 (AL) / +37.01 (AZ) gap is real arithmetic but the comparison is NOT clean, and a sharp reviewer will reject it as currently framed.** The CTLE side is crippled by the **frozen substrate-column protocol**, not by genuine inferiority. Three pieces of internal evidence, all from your own JSONs:

1. **CTLE-SC (17.77) scores BELOW a one-hot-64 control.** The AL board has SC controls the prompt's gap analysis demanded: skip-gram `b2b` = **24.20**, POI2Vec `b2a` = **22.53**, one-hot-64 `b2c` = **19.50**, CTLE = **17.77**. A pre-trained AAAI'21 contextual embedding landing *below a one-hot category encoding* and *below random-ish skip-gram* is a screaming signal that the frozen-CTLE pipeline is not extracting CTLE's signal — it is essentially noise to your head. `HANDOFF_A40_CTLE_SC.md §5` even states the expected CTLE-SC cat is *"below the random floor."* That is an admission the baseline is being run in a regime where it cannot perform.

2. **CTLE's own E2E variant scores ~21.2 > frozen-SC 17.75** (`ctle.md §e2e note`). Even keeping CTLE entirely on its own terms, fine-tuning beats freezing by ~3.4 pp. So part of the 37.8 is "frozen undersells deep models," which `ctle.md` itself flags as NOT a substrate-superiority result.

3. **CTLE was designed for next-POI with a learned task head and warm-user temporal eval.** Here it's frozen, fed to a foreign head, on cold-user SGKF folds, scored on next-*category*. That is a 4-way protocol shift away from CTLE's home turf.

**Would a reviewer believe +37.8?** No — they will say *"you froze the strongest competitor and fine-tuned yours, then compared."* But the comparison is NOT actually rigged that way: Check2HGI-SC is ALSO frozen (it's a substrate column too, matched head, matched folds, matched rows at min_seq=10 — the row-parity fix `b195b89f` is documented). The genuine problem is that the **W3 gate is being carried by frozen-CTLE's collapse below a one-hot baseline**, which makes the headline look like a measurement artifact even though the matched-capacity logic is sound.

**Recommendation:** Do not lead with +37.8 alone. To survive review you MUST (a) report the SC control ladder (one-hot 19.5 < CTLE 17.8 < POI2Vec 22.5 < skip-gram 24.2 ≪ Check2HGI 55.6) so CTLE's number is contextualized as "frozen contextual embeddings are weak here, ours is not" rather than "we crush CTLE"; and (b) report CTLE-E2E (~21.2) alongside SC so the frozen-vs-finetuned confound is visible and disarmed. The Check2HGI advantage over the *strongest* control (skip-gram 24.2) is still ~+31 pp, which is the honest, defensible number. The bare CTLE-only +37.8 is the weakest framing of your strongest result.

### Q2 — Is STAN actually missing at CA/TX, and is that exactly where the region win is strongest?

**Yes on both, and this is a genuine exposure.** STAN `faithful` is confirmed absent at CA/TX — the JSONs carry literal `"_pending_faithful": "STAN faithful train still running"` and the README marks CA/TX as ⚪ "infeasible at scale" (`GAP_A_CLOSURE_20260430.md`). The proposed selection uses STAN as a *region* baseline, but STAN-faithful only exists at AL/AZ/FL/GA.

The danger: the **region win is strongest precisely at the large states** — RESULTS_BOARD §1 shows MTL beats the region *ceiling* only at FL (+0.57), CA (+2.18), TX (+2.17) and merely matches at AL/AZ. So the headline "MTL beats region at large cardinality / CA-8501-retires-the-cardinality-story" rests on CA and TX, which are **exactly the two states with NO faithful STAN and NO faithful ReHDM**. A reviewer can say: *"your region story lives at CA/TX, but your only external region baselines there are HMT-GRN (device-confounded Mac numbers) and Markov-1."* That is thin. Note also STAN-faithful is a weak baseline anyway (below Markov-1 at AL/AZ), so even where present it does not anchor much.

### Q3 — Is Markov-1 region beaten by the JOINT model at every state? 

**Yes, decisively, at every state.** Joint MTL champion region Acc@10 over Markov-1 floor:

| State | Markov-1 | MTL reg | Δ |
|---|---:|---:|---:|
| AL | 47.01 | 69.81 | **+22.80** |
| AZ | 42.96 | 59.34 | **+16.38** |
| FL | 65.05 | 77.28 | **+12.23** |
| CA | 52.09 | 65.66 | **+13.57** |
| TX | 54.94 | 67.13 | **+12.19** (2/5 folds) |
| Istanbul | 52.50 | 69.79 | **+17.29** |

Clears the floor by +12.2 to +22.8 pp everywhere. The prompt's "+15..25pp" is slightly optimistic at the low end — FL/CA/TX are +12.2/+13.6/+12.2, so the accurate claim is **"+12..23 pp."** Caveat that travels: TX is 2/5 clean folds; AL/AZ/Istanbul *match* the STL ceiling (not beat) but still crush the Markov floor.

### Q4 — POI-RGNN status/coverage, and is it a strong enough next-category anchor?

**Status: the strongest-covered baseline you have.** Faithful reimpl, 5-fold complete at ALL six states (AL 23.80 / AZ 27.64 / FL 33.35 / CA 30.71 / TX 32.08 / GA 30.24), user-disjoint SGKF (stricter than the paper → the +28–32pp delta is a conservative floor, per `POI_RGNN_AUDIT.md`). It beats MHA+PE everywhere (+2.6–4.9pp) and beats Markov-K-cat by +3.1–3.7pp. Leak status clean; the only soft spot is the fold-protocol axis (paper used per-user KFold; you used cold-user SGKF), which `POI_RGNN_AUDIT.md` recommends footnoting — but that asymmetry *helps* you, not hurts.

**Is it a strong enough anchor? For category-as-a-task, yes; for the substrate/representation novelty claim, no.** POI-RGNN is a category-*transition* model (7-class graph + RNN); it does not test "does a contextual embedding carry category." That job belongs to CTLE-SC and the SC control ladder. So POI-RGNN + CTLE is a sensible pairing only if you are clear that they answer different questions: POI-RGNN = "do we beat the published category-prediction architecture" (yes, by a huge margin — but note your matched-head Check2HGI cat is 40–63 vs POI-RGNN 24–33, a +15..30pp gap that is so large a reviewer will ask whether the tasks are truly identical), CTLE-SC = "is the win attributable to the hierarchy."

---

## Verdict on the proposed selection — what's being left out (BE CRITICAL)

The proposed set (cat: CTLE + POI-RGNN; region: HMT-GRN + STAN + Markov-1) has **four holes**, in priority order:

1. **The SC control ladder (one-hot / skip-gram / POI2Vec) is omitted but ALREADY COMPUTED at AL.** This is the single most important fix and it's nearly free. Without it, CTLE-SC's +37.8 looks rigged (CTLE below one-hot). With it, the result reads as a principled embedding-paper comparison. The gap analysis explicitly called these "the canonical embedding-paper control set" and the "biggest internal-logic hole." Including them turns your weakest-framed headline into your most defensible. **Add these; finish them at AZ+ states.**

2. **ReHDM (IJCAI'25 region SOTA) is dropped, but it is your strongest region number at AL (66.06 > STAN 62.88).** Dropping the most recent SOTA while keeping the weaker STAN-faithful (which sits below Markov-1 at AL/AZ) is exactly the move a reviewer flags. If you keep only STAN as the faithful-region anchor, expect "why is your region baseline a 2021 method that loses to a count-based Markov floor, when a 2025 region-SOTA exists in your own repo?" Keep ReHDM-faithful at AL/AZ/FL at minimum.

3. **STAN AND ReHDM are both missing at CA/TX — exactly where the region win lives.** Your region headline (CA/TX beat the ceiling, retiring the cardinality story) is anchored at the two states with only HMT-GRN (device-confounded Mac) + Markov-1 as external region baselines. This is the most attackable seam. Either (a) finish faithful STAN/ReHDM at CA/TX, or (b) re-run HMT-GRN on CUDA (device-matched to the champion) so the large-state region claim has at least one trustworthy, device-consistent external comparand. As-is, the strongest region claim has the weakest external support.

4. **No external MTL baseline and no cascade baseline.** Every MTL comparison is internal (champion-G vs your own STL ceilings). HMT-GRN is run as a region-only E2E comparand, not as a joint cat+region multi-task system. A reviewer of an MTL paper will say *"all comparisons that matter are between the authors' own configurations"* (gap analysis §3 says this verbatim). The cheap fix is a **cascade category→region baseline** (the dominant published alternative to parallel MTL); the stronger fix is adapting HMT-GRN to the joint cat+region pairing rather than region-only.

**Bottom line:** The proposed selection is defensible for *category-as-architecture* (POI-RGNN is solid, well-covered) but is the weakest-possible framing for your *representation-novelty* headline (CTLE alone, no controls) and is *under-anchored exactly where your region win is strongest* (CA/TX have no faithful STAN/ReHDM; HMT-GRN there is device-confounded). Minimum changes to survive a critical MobiWac review: (1) add the SC control ladder — it's already half-computed; (2) keep ReHDM-faithful (don't drop the 2025 SOTA); (3) get one device-consistent external region baseline at CA/TX (re-run HMT-GRN on CUDA); (4) add a cascade baseline for the MTL claim.

**Numbers to correct in the proposal text:** region floor-beating is **+12..23 pp**, not +15..25 (FL/CA/TX are ~+12–14); and the headline CTLE gap should be quoted against the strongest control (~+31 pp over skip-gram), not the bare +37.8 over crippled-frozen-CTLE.

Relevant files: `/Users/vitor/Desktop/mestrado/ingred/docs/results/closing_data/baseline_compare/alabama_{ctle,check2hgi_sc,b2b,b2c,poi2vec}.json` (the decisive SC control ladder), `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/next_category/ctle.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/next_region/{comparison.md,hmt_grn.md,stan.md,rehdm.md}`, `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/next_region/results/{california,texas}.json` (STAN-faithful `_pending` proof), `/Users/vitor/Desktop/mestrado/ingred/docs/studies/closing_data/RESULTS_BOARD.md`, `/Users/vitor/Desktop/mestrado/ingred/docs/research/baseline_gap_analysis.md`.
---

## Claim -> baseline mapping & missing controls

I now have everything needed. Here is my critical assessment.

---

# BASELINE DECISION REVIEW — Verdict: the proposed selection is INCOMPLETE on the load-bearing axes, and over-invests in one optional baseline.

The proposed set (CTLE + POI-RGNN for category; HMT-GRN + STAN + Markov-1 for region) is *defensible* but it leaves the paper's **two weakest seams unguarded** — the seams the simulated panel converged on (W3 novelty collapse, W2 "all comparisons internal"). It also picks STAN over the strictly stronger ReHDM and routes CTLE into the role it is *weakest* at (next-category), where its suspicious +37 gap is a liability, not an asset.

---

## (1) CLAIM → BASELINE/CONTROL MATRIX

Contributions (PAPER_PLAN §1 ¶5): **C1** = check-in-level representation carries next-category (novel as a combination); **C2** = single joint model beats dedicated category, bounds region cost (empirical, NOT architectural — per REVIEW_PANEL W7 ruling); **C3** = cross-scale when-sharing-helps account.

| Claim | What it asserts | Baseline / control that is DECISIVE | In proposed set? | Verdict |
|---|---|---|---|---|
| **C1-novelty** | per-visit context is the *active ingredient*, attributable to hierarchical-graph-infomax, not "any contextualization" | **CTLE** as a substrate under matched heads | ✅ (but mis-routed, see below) | LOAD-BEARING. Without it C1 evaporates (panel W3). |
| **C1-mechanism** | the gain is the *graph/infomax machinery*, not "the embedding just carries category one-hot + time features" | **Feature-concat control: HGI ⊕ raw per-visit features → same head** | ❌ **MISSING** | LOAD-BEARING and ABSENT. This is the single most decisive missing control. CTLE does NOT substitute for it (see §2). |
| **C1-canon** | Check2HGI is a credible *embedding* contribution vs the standard location-embedding baseline canon | **POI2Vec / skip-gram / one-hot standalone substrates** | ❌ **MISSING** | DECISIVE for the embedding-paper framing; cheap (POI2Vec already in-repo as an HGI input). |
| **C2 / C3 — joint model is real, not internal-only** | one model beats dedicated category + bounds region vs *the field*, not vs the authors' own configs | **HMT-GRN** (external MTL/region precedent) + **a cascade category→region baseline** | ⚠️ HMT-GRN ✅; **cascade ❌ MISSING** | HMT-GRN load-bearing for "not all-internal"; cascade load-bearing for "is parallel-MTL better than the dominant published alternative?" |
| **C3 — region cost vs SOTA region model** | the joint region head is competitive with a strong dedicated region model | **STAN and/or ReHDM** (region SOTA) | ⚠️ STAN proposed; **ReHDM dropped** | STAN is the WEAKER choice; ReHDM (IJCAI 2025, region-native) is strictly stronger and answers "why no recent baseline?" |
| **Region floor (both tasks)** | the *joint* model beats trivial memory/frequency | **Markov-1 region** + **majority-class category** + **joint-vs-Markov rows** | ⚠️ Markov-1 ✅; majority-class & joint-vs-Markov ❌ | Markov present and beaten by MTL +15..25pp. But the panel (W2, fix #2) flags the floor is currently shown only for the *dedicated* model — the **joint-vs-Markov rows are a to-do**, and there is **no majority-class category floor** in the proposed set. |
| **C2 not a capacity artifact** | category +3..+8 is *sharing*, not a bigger trunk | **STL ceiling at matched head capacity** (internal) | ✅ STL ceilings | Internal control, already have it; Reviewer 2's caveat is per-head *parameter* parity, not class parity. |

---

## (2) THE DECISIVE MISSING CONTROLS, RANKED

**#1 — Feature-concat control: HGI (or POI2Vec) ⊕ raw per-visit features (category one-hot + hour/dow sin/cos) → same head.** This is *the* internal-logic hole and it is NOT in the proposed selection.
- **What it proves that CTLE does NOT:** CTLE is a *different external embedding method* — beating it shows "we beat one published contextual embedding." It does NOT isolate *why* Check2HGI wins. Check2HGI's node features literally ARE category one-hot + sin/cos time. The headline +14..29pp category lift could be substantially "the embedding carries explicit category/time features," not "hierarchical infomax learned something." The feature-concat control is the ONLY thing that separates "the graph/infomax machinery adds signal" from "you injected the label-adjacent features and read them back out." `project_positioning.md §2.1` calls this "the feature-injection confound… currently unaddressed"; `baseline_gap_analysis.md` Tier-1 #2 calls its absence "the biggest internal-logic hole in CH16."
- It is **cheap** (no new embedding training — concat existing HGI/POI2Vec vectors with features you already compute) and **decisive**. CTLE-loses + feature-concat-also-loses together pin the win on the hierarchy. CTLE alone leaves C1's mechanism exposed.
- **It must be added.** This is the highest-value single addition to the whole table.

**#2 — A cascade (category→region) baseline.** The dominant *published* alternative to parallel MTL for this task pair (CSLSL/CatDM pattern: predict category, condition region on it) is absent. Without it, the strongest reviewer attack on C2 — "you compared your parallel joint model only to your own STL ceilings and your own MTL ablations; the real competitor is the cascade everyone else uses" — stands. HMT-GRN partially covers "external MTL exists," but HMT-GRN is *hierarchical-for-next-POI*, not a clean category→region cascade. The cascade tests the specific architectural choice the paper makes.

**#3 — POI2Vec / skip-gram / one-hot standalone substrate columns.** Every location-embedding paper (including CTLE's own suite) reports these. POI2Vec already exists in-repo as an HGI input; promoting it to a standalone substrate column is near-free. Their absence makes the embedding half look like it skipped the canonical baseline set. One-hot is the trivial-substrate floor for C1 (analogous to Markov for region).

**#4 — Joint-vs-Markov-1 rows + a majority-class category floor.** The panel (W2, required-change #2) is explicit: the Markov comparison currently cited is the *dedicated* model vs Markov, so R1's "loses to Markov" is **not yet rebutted for the joint model the paper headlines**. The board says MTL clears Markov-1 by +15..25pp — that number must appear as an explicit *joint-vs-Markov* row, not be asserted. Add the majority-class macro-F1 floor for category symmetrically (it is in PAPER_BASELINES_STRATEGY's category-table plan but not in the proposed selection).

**#5 — ReHDM should replace or join STAN as the region SOTA point** (see §3).

---

## (3) WHICH PROPOSED BASELINES ARE LOAD-BEARING vs OPTIONAL, AND THE MIS-ROUTINGS

**Load-bearing (keep, non-negotiable):**
- **CTLE** — but **re-scope its role.** CTLE is the existential gate for C1 *novelty* (panel W3). Its decisive job is **next-category as a substrate under matched heads, scored leak-clean at n=20** (currently deferred to P3, and the one run it had carried 81.8% transductive overlap — `BASELINES_IMPL_AUDIT`). ⚠️ **The +37.8/+37.0 category gap is a RED FLAG, not a trophy.** A gap that large at AL/AZ (the two *smallest* states, exactly where the panel notes the substrate effect is least stressed) strongly suggests CTLE is *crippled on next-category*, not "genuinely 37 points worse." CTLE is a next-POI method; if it was run with a next-POI objective/readout and probed on category, the comparison is unfair and a reviewer will say so. **Before this number ships: confirm CTLE is fed to the SAME matched category head with the SAME capacity as Check2HGI, and sanity-check it on its home turf (region/next-POI) to show it is not simply broken in your harness.** A fair CTLE gap of, say, +8..+15 is a *stronger* paper than an unfair +37 a reviewer will reject.
- **Markov-1 region** — floor, present, beaten. Keep, but extend to joint-vs-Markov rows (#4 above).

**Load-bearing but UNDER-SPECIFIED:**
- **HMT-GRN** — this is the *only* thing standing between the paper and "all MTL comparisons are internal" (panel W2; `baseline_gap_analysis` §3 "A reviewer can currently say…"). Keep it, but ⚠️ the board notes the old 62.37 was an unreproducible anomaly (PR #38) and current faithful HMT-GRN is FL 63.74 / CA 49.61 — **well below** MTL ~65-69. That is fine (you beat it) but means HMT-GRN's value is "external precedent we surpass," not "close competitor." It does NOT by itself rebut the cascade attack — pair it with the cascade baseline (#2).

**Optional / the weaker choice:**
- **STAN for region** — STAN is faithful only at AL/AZ/FL and is flagged **"infeasible at scale" at CA/TX** in the gap analysis. CA/TX are *the exact states the W1 headline claim is about*. A region SOTA baseline that is missing precisely where the contribution lives is weak. **ReHDM (IJCAI 2025) is strictly the better pick:** it is region-native, more recent (answers "why no recent baseline?" — which is *why the plan dropped GETNext*, PAPER_BASELINES_STRATEGY), and faithful at AL/AZ/FL with CA/TX merely deferred-for-compute (same coverage limit as STAN, but a stronger, current method). Recommendation: **lead region-SOTA with ReHDM, keep STAN as a secondary 2021 reference if page budget allows.** Do NOT present ReHDM-STL (frozen-substrate) numbers — they are architecture-bound nulls, correctly demoted to a footnote.

**Optional (keep only if page budget allows):**
- **POI-RGNN for category** — a reasonable GNN category baseline (faithful, all-state: FL 34.49, CA 31.78, TX 33.03), but it is the *most* optional of the proposed set. The category claim's decisive comparators are the *substrate-axis* controls (CTLE, feature-concat, POI2Vec/skip-gram/one-hot under matched heads), not another full external category model. POI-RGNN is a "we also beat a published GNN" row — nice external anchor, not load-bearing. If page budget forces a cut (panel §11), POI-RGNN is more expendable than MHA+PE is *complementary*: MHA+PE (FL 32.06, all 5 states) is the cleaner transformer-sequence category reference. Keep one external category model; CTLE+feature-concat+POI2Vec carry the actual C1 argument.

---

## BOTTOM LINE FOR THE AUTHORS

The proposed selection answers "do we beat published methods?" but **under-answers the two questions reviewers will actually reject on:**

1. **Is the substrate win the *hierarchy* or just *feature injection*?** → needs the **feature-concat control (HGI⊕features)** + **POI2Vec/skip-gram/one-hot** columns. CTLE alone cannot close this. (Missing controls #1, #3.)
2. **Is the joint-model story anything but the authors comparing their own configs?** → HMT-GRN helps but is not enough; needs a **cascade category→region baseline** and the **joint-vs-Markov rows** stated, not asserted. (Missing controls #2, #4.)

**Concrete recommended final set:**
- **next-category:** CTLE (re-scoped, fairness-audited, n=20 leak-clean) **+ feature-concat control + POI2Vec/skip-gram/one-hot substrate columns** + one external model (keep MHA+PE *or* POI-RGNN, not necessarily both).
- **next-region:** **ReHDM (lead) + STAN (secondary if budget) + cascade category→region + Markov-1 + joint-vs-Markov rows + majority-class category floor.**
- **MTL story:** HMT-GRN (external precedent, beaten) + cascade (the real published alternative) + internal STL ceilings (capacity-matched per-head).

**Single most important action:** add the **feature-concat control** — it is cheap, it is the biggest internal-logic hole, and it is the one control CTLE explicitly does not substitute for. **Single biggest risk in the proposed set:** shipping the **+37 CTLE category gap unaudited** — it reads as a crippled baseline and will be caught; route CTLE through a fair matched-head comparison or it damages C1 instead of supporting it.

Relevant files: `/Users/vitor/Desktop/mestrado/ingred/docs/research/baseline_gap_analysis.md` (Tier-1 list, §3 minimum-credible-table), `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/PAPER_PLAN.md` §7 (the feature-concat control is already listed as an "(internal control)" checkbox at line 510 — it is in the *checklist* but absent from the proposed *selection*; that is the gap to close), `/Users/vitor/Desktop/mestrado/ingred/docs/PAPER_BASELINES_STRATEGY.md` (ReHDM-faithful vs STAN coverage; majority/Markov floors), `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/REVIEW_PANEL.md` (W2 internal-only, W3 CTLE novelty gate, required-changes #2).
---

## Field / venue expectation (web)

I have all the evidence needed. Synthesizing the final verdict.

---

# Baseline Review: MobiWac 2026 (next-category macro-F1 + next-region Acc@10, check-in data)

## VERDICT (headline)
The proposed set — next-cat: CTLE + POI-RGNN; next-region: HMT-GRN + STAN + Markov-1 — is **defensible but thin and mis-cast in one place**. CTLE belongs in the table, but **as a substrate/embedding baseline, NOT as a next-category model**; presenting it as a "next-category competitor" with a +37pp win is the single most likely thing to draw reviewer fire. The region set is dated/thin for a 2026 venue, and dropping ReHDM (your faithful 2025 SOTA, already run at AL/AZ/FL) is the second-biggest exposure. Details and the field-standard sets below.

---

## 1. Next-CATEGORY / next-activity prediction — what the field expects
The honest finding: there is **no single canonical "next-category" benchmark suite** the way there is for next-POI. Category prediction shows up in two distinct lineages, and reviewers expect baselines from the one you're claiming membership in:

- **MTL "time→activity→location" lineage (CSLSL, EPJ Data Science 2024 / arXiv 2206.05731).** This is your closest neighbor — it jointly predicts *activity (category)* and *location* via multi-task learning, exactly your framing. Its baseline set is the de-facto standard for category+location MTL: **FPMC (Markov floor), DeepMove, Flashback, LSTPM, GeoSAN, STAN, GETNext.** It reports category/activity via **Recall@1 / accuracy** (not macro-F1). GETNext "introduces location categories as inputs and prediction targets," and **SGRec** adds an auxiliary next-category head — so a 2026 reviewer of an MTL category paper will expect at least one of {CSLSL, GETNext-with-category-head, SGRec, CatDM} as a *learned sequential* category baseline.
- **Embedding-paper lineage (CTLE, AAAI 2021; AdaptGOT/GOT, 2025).** Here "next category prediction" is a *downstream probe of an embedding*, and the baseline columns are **embeddings**: one-hot, skip-gram/Word2Vec, POI2Vec, Geo-Teaser/Teaser, TALE, HIER(-CEM), SpaBERT, CTLE — each fed into a shared predictor (FPMC/GETNext/ST-LSTM). AdaptGOT (2025) explicitly runs "5.2.2 Next Category Prediction" with **CTLE and SpaBERT as baselines**, reporting Rec@k.

**So:** your *real* category baseline gap is (a) no faithful learned-sequential category model that natively predicts category — POI-RGNN/MHA+PE cover the Capanema family, but a reviewer will want the GETNext-category-head or CSLSL cascade too; and (b) the embedding-control set (skip-gram/POI2Vec/one-hot) that every CTLE-lineage paper carries. Both gaps were already flagged internally (gap-analysis Tier-1 items 2-3) — the field-research confirms they are real.

## 2. Is CTLE-on-next-category a FAIR comparison? — the load-bearing answer
**It is fair as a published protocol, but a large gap is EXPECTED, not suspicious — and framing matters enormously.**

- **CTLE never predicts category in its own paper.** Direct from the AAAI 2021 PDF: CTLE is evaluated on exactly one downstream task — *user next location prediction* (predicting the next location ID), plus a brief land-use classification mention. Its embeddings are "validated by incorporating them into downstream tasks like user location prediction." It is a **masked-LM contextual embedding tuned for next-POI**.
- **The field already treats CTLE as a weak next-category baseline — by design.** AdaptGOT (2025) states verbatim: *"CTLE bias[es] towards the next POI recommendation task"* and shows GOT beating CTLE on next-category "across all models and metrics." So the literature explicitly documents that CTLE underperforms on next-category because it optimizes location identity, not semantic class.
- **Therefore your +37.8 / +37.0 gap is consistent with the literature's direction**, but the *magnitude* is large enough that a reviewer will suspect a crippled implementation unless you (i) report CTLE's matched next-POI/next-location number to show it is *not* broken, and (ii) frame it as a **representation-isolation / substrate column** ("CTLE-emb → our head" vs "Check2HGI-emb → our head", matched capacity), which is exactly how you've scoped it internally (RESULTS_BOARD §4). **Do NOT present CTLE as a "next-category model" baseline in the headline table** — that invites "you compared a next-POI embedding on a task it was never built for and called it a 37-point win." Present it as one embedding column in a substrate comparison, alongside skip-gram/POI2Vec/one-hot. Cast that way it is fair and even favorable to you (it's the canonical contextual-embedding precedent). Cast as a next-category competitor it is a strawman.

## 3. Next-REGION / next-location — what the field expects, and is HMT-GRN+STAN credible?
The 2025 canonical next-POI/next-location baseline set (confirmed by the Massive-STEPS benchmark 2025 and the ReHDM IJCAI-2025 baseline table) is:
- **RNN/attention era:** FPMC (Markov floor), RNN, DeepMove, LSTPM, Flashback, STAN, GeoSAN.
- **Graph era (now mandatory):** **GETNext** (CVPR-of-POI, the universal 2022+ comparator), **Graph-Flashback** (KDD 2022), **STHGCN** (hypergraph, 2023), and the 2024-25 SOTA **ReHDM** (IJCAI 2025), LoTNext.
- **HMT-GRN** (SIGIR 2022) is the *region-multi-task* precedent specifically (User-Region matrices + hierarchical beam search) — perfect as your region-MTL anchor, but it is **one paper, 2022-vintage.**

**Assessment of HMT-GRN + STAN as the region set: too thin and slightly dated for 2026.** STAN (WWW 2021) + HMT-GRN (2022) skips the entire 2022-25 graph/hypergraph wave that any region/next-POI reviewer now expects. You already have a **faithful ReHDM (2025 SOTA)** at AL/AZ/FL in-repo (AL 66.06, AZ 54.65, FL 65.68) — leaving it out of the proposed set while it sits run-and-validated is indefensible. The credible region set for 2026 is **STAN + HMT-GRN + ReHDM + Markov-1**, with GETNext acknowledged (you already cite it as inspiration for `next_stan_flow`). STAN/ReHDM "infeasible at scale (CA/TX)" is a legitimate, statable limitation — report them at AL/AZ/FL and footnote the scale ceiling; that is normal and reviewers accept it.

## 4. Would dropping ReHDM and MHA+PE be noticed?
- **ReHDM: YES, loudly.** It is your strongest, most recent (IJCAI 2025) region baseline, *already faithfully reproduced* at 3 states. Omitting a run-and-passing 2025 SOTA baseline reads as cherry-picking the comparison set. Keep it. This is the #1 omission risk on the region side.
- **MHA+PE: minor.** It's a 2019 architectural reference and a weak lower bound (FL 32.06). Dropping it is fine *if* POI-RGNN stays (same paper family, stronger). But you then have only one learned next-category baseline (POI-RGNN) — see §5.

## 5. The 2-3 omissions most likely to draw reviewer fire
1. **ReHDM (2025 region SOTA) left out of the region set despite being reproduced** — pure self-inflicted exposure. Add it.
2. **CTLE mis-cast as a "next-category model" rather than an embedding/substrate column, with an unexplained +37pp gap** — reframe as a substrate row, report its matched next-POI number to prove it isn't crippled, and add the canonical embedding controls it implies (skip-gram, POI2Vec, one-hot) since you're claiming the CTLE-lineage comparison. The internally-flagged **feature-concat control (HGI ⊕ raw per-visit features → same head)** belongs here too — it is the cheapest, most decisive answer to "is the gain from the graph/infomax design or just from feature injection?"
3. **No external MTL baseline / no cascade baseline.** Every MTL comparison is currently internal (champion-G vs your own STL ceilings). A region+category MTL paper at MobiWac 2026 should run **HMT-GRN as the external region-MTL baseline** (you propose it — good) AND a **cascaded category→region baseline (CSLSL/CatDM pattern)**, the dominant published alternative to your parallel cross-attn design. Without the cascade, a reviewer asks "why parallel MTL beats the obvious cascade" and you have no number.

Secondary (expected, lower fire): a **GETNext-with-category-head or SGRec** learned-sequential category baseline (so category isn't only POI-RGNN + a strawman-risk CTLE); and a **single LLM zero-shot reference row** (LLM-Mob / LLMMove / AgentMove, NAACL 2025) — increasingly standard in 2025-26 mobility submissions, even just to show supervised models beat zero-shot prompting. The Markov-1 floor you already include is correct and expected — keep it; it's the universal sanity floor (FPMC plays this role in the next-POI literature).

## Recommended final baseline sets
- **next-category (macro-F1):** POI-RGNN (faithful) + **GETNext-category-head or SGRec or CSLSL-cascade** (a real learned next-category model) + Markov-K floor; CTLE/skip-gram/POI2Vec/one-hot **as substrate columns under matched heads** (not as "next-category models") + the HGI⊕features concat control. MHA+PE optional lower bound.
- **next-region (Acc@10):** STAN + HMT-GRN + **ReHDM (add it)** + Markov-1 floor, at AL/AZ/FL; honest "infeasible at scale" footnote for STAN/ReHDM at CA/TX; GETNext acknowledged. Optionally a cascade (category→region) row as the MTL alternative and one LLM zero-shot row.

---

## Sources
- CTLE (downstream = next-location only; baseline embeddings = one-hot/skip-gram/POI2Vec/Geo-Teaser/Teaser/TALE/HIER; eval = Accuracy/macro-Recall/macro-F1): [AAAI 2021 paper](https://ojs.aaai.org/index.php/AAAI/article/view/16548) · [code](https://github.com/Logan-Lin/CTLE)
- CSLSL (category+location MTL; baselines FPMC/DeepMove/Flashback/LSTPM/GeoSAN/STAN/GETNext; activity via Recall@1/accuracy): [arXiv 2206.05731](https://arxiv.org/abs/2206.05731) · [EPJ Data Science 2024](https://link.springer.com/article/10.1140/epjds/s13688-024-00460-7)
- AdaptGOT / GOT (runs "Next Category Prediction" with CTLE + SpaBERT baselines; "CTLE biases towards next-POI"): [arXiv 2506.21612](https://arxiv.org/pdf/2506.21612)
- GETNext (category embeddings + categories as prediction targets): [arXiv 2303.04741](https://arxiv.org/pdf/2303.04741)
- HMT-GRN (region-multi-task, hierarchical beam search): [SIGIR 2022](https://dl.acm.org/doi/abs/10.1145/3477495.3531989) · [PDF](https://bhooi.github.io/papers/hmt_sigir22.pdf) · [code](https://github.com/poi-rec/HMT-GRN)
- ReHDM (IJCAI 2025 region SOTA; baselines STAN/GETNext/STHGCN/Graph-Flashback): [IJCAI 2025 #343](https://www.ijcai.org/proceedings/2025/0343.pdf)
- Massive-STEPS benchmark (2025 canonical next-POI baseline set + LLM baselines): [arXiv 2505.11239](https://arxiv.org/pdf/2505.11239)
- LLM zero-shot next-POI references (LLM-Mob/LLMMove/AgentMove): [AgentMove NAACL 2025](https://github.com/tsinghua-fib-lab/AgentMove) · [Where to Move Next, arXiv 2404.01855](https://arxiv.org/pdf/2404.01855)

Internal grounding cross-checked: `/Users/vitor/Desktop/mestrado/ingred/docs/research/baseline_gap_analysis.md` (Tier-1/2 items independently confirmed by this web research) and `/Users/vitor/Desktop/mestrado/ingred/docs/baselines/BASELINE.md`.
---

## Advisor: minimum-credible set

This is a baseline-selection decision, not a code task. I have four detailed reports with all the numbers I need. Let me synthesize the decision directly.

Looking at the four reports, the consensus is clear and the numbers cross-check. Here is my decisive verdict.

---

# VERDICT: The proposed set is NOT sufficient. Two cheap additions are mandatory; one mis-framing must be fixed before anything ships.

The proposed set (cat: CTLE + POI-RGNN; region: HMT-GRN + STAN + Markov-1) answers "do we beat published methods" but leaves the two seams reviewers actually reject on unguarded: (1) is the substrate win the *hierarchy* or just *feature injection*, and (2) is the MTL story anything but the authors comparing their own configs. All four reports converge on this independently.

## THE SINGLE MOST EMBARRASSING RISK
**The CTLE +37.8 pp category gap, shipped as a headline "win," with CTLE scoring BELOW a count-based bigram floor (CTLE-SC 17.77 vs Markov-K 20.50 vs one-hot 19.50 at AL).** A frozen AAAI'21 contextual embedding landing under a stupid-backoff bigram is a screaming "crippled baseline" signal. A reviewer reads this as "they handicapped the competitor and called it a 37-point win," and it discredits C1. This is not a number to lead with — it must be reframed (see Add #1, Fix #1) or it actively damages the paper.

## KEEP (load-bearing, non-negotiable)
- **CTLE** — the W3 novelty gate; cannot be dropped. But RE-CAST as one substrate column ("CTLE-emb → matched head"), NOT a "next-category model." The field (AdaptGOT 2025) already documents CTLE underperforms on category by design.
- **POI-RGNN** — strongest-covered baseline (faithful, all states, stricter folds). The external category anchor. Use the *canonical* numbers (FL 34.49 / CA 31.78 / TX 33.03), not the superseded comparison.md snapshot.
- **HMT-GRN** — the *only* external MTL/region precedent; sole defense against "all comparisons internal." Keep, but re-run on CUDA to kill the Mac-vs-CUDA device confound (currently device-mismatched against the champion).
- **STAN** — keep as region competitor, but the `stl_hgi` variant EXISTS at CA/TX (CA 60.45, TX 62.70). Do NOT label it "infeasible at CA/TX" — that's only true for `faithful`. It is your tightest large-state competitor.
- **Markov-1 region** — correct universal floor, beaten everywhere +12..23 pp. Keep.

## ADD (ranked by necessity — only the top 3 are real holes)
1. **SC control ladder (one-hot / skip-gram / POI2Vec), already computed at AL.** Highest value, near-free. Turns the CTLE +37.8 from "rigged" into a principled embedding ladder (one-hot 19.5 < CTLE 17.8 < POI2Vec 22.5 < skip-gram 24.2 ≪ Check2HGI 55.6). Finish at AZ+. Without this, the headline looks like a measurement artifact.
2. **Feature-concat control (HGI ⊕ raw per-visit features → same head).** The single biggest internal-logic hole; the ONE control CTLE cannot substitute for. Check2HGI's node features ARE category one-hot + sin/cos time — this is the only thing separating "the hierarchy adds signal" from "we read back injected labels." Cheap (concat existing vectors, no new training), decisive.
3. **ReHDM (IJCAI'25 region SOTA), already faithful at AL/AZ/FL.** Dropping a run-and-passing 2025 SOTA while keeping weaker 2021/2022 methods is auto-flagged ("why no recent baseline?"). It's also your *thinnest* small-state competitor (AL only +3.75) — keeping it is honest and strengthens, not weakens. Add it; footnote CA/TX deferred-for-compute (same legit ceiling as STAN-faithful).
4. **Cascade category→region baseline (CSLSL/CatDM pattern).** Real but second-tier given the deadline. The dominant *published* alternative to parallel MTL; rebuts "why is parallel better than the obvious cascade." If time exists after 1–3, add it; otherwise footnote as future work and lean on HMT-GRN for the external-MTL coverage.
5. **CTLE at FL/CA/TX + CTLE-E2E companion.** The novelty gate is currently unmeasured at the 3 largest states (the CUDA runs are pending). At minimum get the CTLE-SC CUDA runs done so the gate exists where the region win lives; report CTLE-E2E (~21.2) beside CTLE-SC to disarm the frozen-vs-finetuned confound.

## DROP / SKIP (given the deadline)
- **MHA+PE** — drop. Weaker sibling of POI-RGNN, redundant as a category anchor. (Keep only if a second category point is free — it's already run.)
- **CTLE-E2E as a full variant** — skip the full tabulation; one note row beside SC suffices.
- **GETNext-category-head / SGRec / LLM zero-shot** — skip for this deadline. Nice-to-haves the web report flagged, not holes. Footnote GETNext as acknowledged inspiration.
- **STAN/ReHDM faithful at CA/TX** — skip (legitimately "infeasible at scale"); footnote it. Use STAN `stl_hgi` for the CA/TX region point instead.
- **Majority-class category floor** — add only if trivial (you have Markov-K-cat as the honest floor already); low priority.

## NUMBERS TO CORRECT IN THE PROPOSAL TEXT
- Region floor-beating is **+12..23 pp**, not +15..25 (FL/CA/TX are ~+12–14).
- Quote the CTLE comparison against the strongest control (~+31 pp over skip-gram), never the bare +37.8.
- POI-RGNN: cite canonical 34.49/31.78/33.03, not the superseded snapshot.
- TX MTL is **2/5 folds** — every TX margin is provisional; flag it.

## BOTTOM LINE
Minimum credible set = **KEEP all five proposed + ADD the SC control ladder (#1) + feature-concat control (#2) + ReHDM (#3)**, and **FIX the CTLE framing** (substrate column, not category model; report against skip-gram not one-hot-crippled-frozen). Items 1–3 are mostly already computed or cheap; they close the two real holes (feature-injection confound, all-internal MTL). Cascade (#4) and CTLE-at-large-states (#5) are the next tier — do them if the deadline allows, footnote otherwise. Skip MHA+PE, LLM rows, and any new from-scratch SOTA training.
---

## Advisor: tough reviewer

I have four thorough reports. The task is pure adversarial synthesis — I am the hostile reviewer. No tools needed; I have all the evidence. Let me compose the ranked attack list.

# HOSTILE REVIEWER ATTACK SHEET — MobiWac 2026 baseline set

Target under attack: **cat = CTLE + POI-RGNN; region = HMT-GRN + STAN + Markov-1.** I am Reviewer 2. Below, ranked by how badly each kills the paper, each attack states the charge, the evidence I will cite from your own repo, the damage, and the *single* addition that defuses it. I am not here to be fair; I am here to find the seam and pull.

---

## ATTACK 1 — "You froze the competitor and fine-tuned yourself: CTLE scores BELOW a bigram. The +37.8 is a handicapped strawman." 
**Damage: FATAL to the C1 novelty claim. This is the kill shot.**

The charge writes itself from your own numbers. CTLE-SC = 17.77 (AL) / 19.30 (AZ). Your own Markov-K-cat count-based floor is 20.50 (AL) / 23.92 (AZ). **A pre-trained AAAI'21 contextual embedding loses to a stupid-backoff bigram.** It also loses to your own MHA+PE (18.95) and your one-hot-64 control (19.50). Your handoff doc literally says CTLE-SC is "below the random floor." A model that scores below a 9-step count table is not "37 points worse" — it is *broken in your harness*, and every reviewer knows a frozen next-POI embedding probed on next-category through a foreign head on cold-user folds is being run off its home turf (a 4-way protocol shift). I will write: *"the headline novelty result rests on a competitor the authors crippled below a count baseline."* Because CTLE is your **non-negotiable W3 novelty gate**, this single cell can sink C1 entirely — not weaken it, sink it.

**Single defusing addition: the SC control ladder, already half-computed at AL (one-hot 19.50 < CTLE 17.77 < POI2Vec 22.53 < skip-gram 24.20 ≪ Check2HGI-SC 55.59).** This recasts the result from "we crush CTLE" to "frozen contextual embeddings are uniformly weak here, ours is the lone exception," and the honest headline becomes Check2HGI over the *strongest* control (skip-gram), ~+31 pp — still devastating, now defensible. (Pair it with the CTLE-E2E ~21.2 companion to disarm the frozen-vs-finetuned confound, but the ladder is the load-bearing fix.)

---

## ATTACK 2 — "Your strongest claim lives at CA/TX, and you have NO faithful region SOTA there. The flag is planted on empty ground."
**Damage: SEVERE. Directly undermines the headline region result.**

Your headline is "CA (8501 regions) and TX beat both ceilings — the cardinality story is retired." Region win is *only* > ceiling at FL/CA/TX (+0.57/+2.18/+2.17); AL/AZ merely match. So the contribution **lives at CA/TX**. And at CA/TX your external region baselines are: HMT-GRN (Mac/MPS numbers, **device-confounded** vs your CUDA champion) and Markov-1 (a floor). STAN-*faithful* and ReHDM are both absent there (your JSONs literally carry `"_pending_faithful"` and the README says "infeasible at scale"). I will write: *"the region contribution is anchored at exactly the two states with no trustworthy external comparand."*

**Single defusing addition: re-run HMT-GRN on CUDA at CA/TX (device-matched to the champion).** This is cheaper than finishing faithful STAN/ReHDM at 8501 classes and gives the large-state claim at least one device-consistent external region baseline. (Note: the `margins` report flags that STAN `stl_hgi` *does* exist at CA 60.45 / TX 62.70 — if so, your proposal text wrongly calls STAN absent there; either way, do not ship "STAN infeasible at CA/TX" while a usable STAN variant sits in your JSONs.)

---

## ATTACK 3 — "You dropped the 2025 region SOTA (ReHDM) that you ALREADY RAN — and it's your tightest small-state competitor. That is cherry-picking."
**Damage: SEVERE. Self-inflicted; the worst kind because there is no compute excuse.**

ReHDM (IJCAI 2025) is faithful at AL/AZ/FL in your repo: AL 66.06, AZ 54.65, FL 65.68. At AL it **beats your STL ceiling** (69.99 → margin only +3.75 to MTL, +3.93 to STL) and **beats the STAN variant you kept** (62.88). You propose keeping the *weaker, 2021* STAN and dropping the *stronger, 2025* SOTA you already reproduced. I will write the most damning sentence available: *"the authors retained a 2021 baseline that loses to a count-based Markov floor at two states, while omitting the 2025 region SOTA they had already run and which beats their own STL ceiling at Alabama. Why is the most recent and strongest competitor missing?"* There is no defensible answer to that.

**Single defusing addition: put ReHDM-faithful (AL/AZ/FL) back in the region table.** It is run, validated, leak-clean. Dropping a passing 2025 SOTA is indefensible; including it costs zero new compute.

---

## ATTACK 4 — "Every comparison that matters is you-vs-you. There is no external MTL baseline and no cascade — the dominant published alternative to your design."
**Damage: SEVERE. Hits C2 (the joint-model thesis) at its foundation.**

Your MTL story is champion-G vs *your own* STL ceilings and *your own* ablations. HMT-GRN is run region-*only* E2E, not as a joint cat+region system, so it does not rebut "all-internal." The field's dominant alternative to your parallel cross-attn MTL is the **cascade category→region** (CSLSL / CatDM pattern — predict category, condition region on it). You have *no number* for it. I will write: *"the central claim — parallel joint sharing beats the alternatives — is never tested against the alternative the field actually uses; all decisive comparisons are between the authors' own configurations."* For an MTL paper, this is the structural weakness reviewers reach for first.

**Single defusing addition: a cascade category→region baseline.** It is the one number that converts "we beat our own configs" into "we beat the dominant published joint design." Cheaper than adapting HMT-GRN to the joint pairing and more directly on-point.

---

## ATTACK 5 — "The feature-concat control is missing, so your substrate win is unattributable: it could be pure feature injection, not the hierarchy."
**Damage: SEVERE and HIGH-LEVERAGE. This is the cheapest fatal hole — your own gap analysis calls it 'the biggest internal-logic hole.'**

Check2HGI's node features *literally are* category one-hot + sin/cos time. So your +36..+48 pp category lift could be "you injected label-adjacent features and read them back out," not "hierarchical infomax learned structure." CTLE does **not** close this — CTLE is a *different* external embedding; beating it shows you beat one method, not *why* you win. The only control that pins the win on the graph/infomax machinery is **HGI (or POI2Vec) ⊕ raw per-visit features → same head**. Its absence means I write: *"the representation contribution is confounded with feature injection and the authors provide no control to separate them."* This is the exact attack Attack 1's ladder *partially* anticipates but does not fully answer.

**Single defusing addition: the HGI⊕features concat control.** No new embedding training (concat existing vectors with features you already compute), and it is *the* control that separates "the hierarchy adds signal" from "we fed the label back in." Highest value-per-effort fix in the entire set. It is already a checkbox in your PAPER_PLAN §7 — it just never made the selection.

---

## ATTACK 6 — "The category floor is missing on the joint model, and there's no majority-class floor. Your 'beats the floor' claim is asserted, not shown — for the model you actually headline."
**Damage: MODERATE-to-SEVERE. A known prior-review attack (R1 'loses to Markov') is not yet rebutted for the JOINT model.**

You cite Markov-1 region beaten by +12..23 pp — good, but the panel's own ruling is that the floor is currently shown for the *dedicated* model, not the *joint* model you headline; the joint-vs-Markov rows are a to-do. And there is **no majority-class macro-F1 category floor** in the proposed set (you have it at AL = 7.28 but it is not in the selection). Symmetry matters: region has a floor, category's floor is missing from the table. I will write: *"the headline joint model's superiority over trivial baselines is asserted in prose, not tabulated; and the category task lacks the trivial floor the region task is given."*

**Single defusing addition: explicit joint-vs-Markov-1 rows + a majority-class category floor row.** Both numbers exist or are trivial to compute; the fix is tabulating them, not running anything.

---

## ATTACK 7 — "Your category table mixes protocols: 'ours' is the stride-1 overlap board, baselines are the older category-native study. Apples to oranges."
**Damage: MODERATE. Credibility/hygiene attack that compounds Attacks 1-2.**

POI-RGNN / MHA+PE were run under the older category study (user-disjoint SGKF, different stride/min-seq), while your MTL numbers are the new gated-overlap board. The `comparison.md` banner says so explicitly. A +45 pp margin invites scrutiny, and the first thing I check is whether the rows share a protocol. They do not. I will write: *"the category comparison is cross-protocol; the reported deltas conflate method and evaluation regime."* Additionally, you are citing **superseded** POI-RGNN numbers (33.35/30.71/32.08) when the canonical post-bugfix values are 34.49/31.78/33.03 — a sloppiness that, once I find it, makes me distrust every other number.

**Single defusing addition: a one-row protocol-reconciliation footnote stating the fold/stride/min-seq deltas and citing the canonical POI-RGNN numbers.** The protocol asymmetry (cold-user SGKF is *stricter*) actually favors you — so state it; an unstated asymmetry reads as hiding, a stated one reads as conservative.

---

## ATTACK 8 — "Your only learned next-category baselines are a strawman (CTLE) and one GNN (POI-RGNN). The field expects a learned sequential category model — GETNext-cat-head / SGRec / CSLSL."
**Damage: MODERATE. A 'thin baseline set for the venue' attack, sharpened by lineage.**

For a category+location MTL paper, the closest neighbors (CSLSL, GETNext-with-category-head, SGRec) are the de-facto category baseline canon. After Attack 1 demotes CTLE to a substrate column, your *only* learned-sequential category competitor is POI-RGNN — a single 2023 GNN. I will write: *"the category contribution is benchmarked against one learned model and a mis-cast embedding; the standard category-MTL comparators (CSLSL/GETNext-cat/SGRec) are absent."*

**Single defusing addition: one learned sequential category baseline with a native category head (GETNext-category-head is the lightest lift since you already cite GETNext as inspiration for `next_stan_flow`).** Keeping MHA+PE as a second cheap category point (it is already run, FL 32.06) is a fallback but does not fully answer the "recent, category-native" expectation.

---

## ATTACK 9 — "TX is your headline large-state win and it's 2/5 folds. Every TX margin is provisional."
**Damage: LOW-to-MODERATE. Easy to footnote, fatal if hidden.**

TX MTL is 2/5 clean folds. Every TX delta in both tables (cat +45.4, region +12.2, STAN +4.4) is provisional, and STAN-FL/TX are your *thinnest* region margins (+3.7/+4.4) — exactly where 3 missing folds could move the verdict. If I find this undisclosed, I escalate to "the large-state claim is built on incomplete runs." If disclosed, it is a normal limitation.

**Single defusing addition: finish TX to 5/5 folds (or, minimum, a prominent n-flag on every TX cell).** Completing the run is the real fix; the flag is the floor.

---

## ATTACK 10 — "For a 2026 venue you skipped the entire 2022-25 graph/hypergraph wave and any LLM zero-shot reference."
**Damage: LOW. A 'datedness' jab, not a kill — but it colors the whole review.**

STAN (2021) + HMT-GRN (2022) skips GETNext / Graph-Flashback / STHGCN / LoTNext and the now-common LLM zero-shot row (LLM-Mob / AgentMove). Reviewers in 2025-26 increasingly expect at least an acknowledgment. I will write: *"the region baselines predate the graph-POI wave the venue now treats as standard."* Adding ReHDM (Attack 3) already buys you the 2025 SOTA and largely answers this; this attack is mostly residual.

**Single defusing addition: adding ReHDM (already required by Attack 3) plus a one-line acknowledgment/citation of GETNext/STHGCN as the modern lineage** (a single LLM zero-shot row is a nice-to-have, not load-bearing).

---

## Ranked summary (worst → least, with the one-line defusal)

1. **CTLE +37.8 strawman (CTLE < bigram floor)** — FATAL to C1 → **add the SC control ladder** (one-hot/skip-gram/POI2Vec, AL already computed).
2. **No faithful region SOTA at CA/TX (where the win lives; HMT-GRN there is device-confounded)** — SEVERE → **re-run HMT-GRN on CUDA at CA/TX.**
3. **Dropped ReHDM, the 2025 SOTA you already ran (tightest at AL)** — SEVERE → **put ReHDM-faithful (AL/AZ/FL) back in the table.**
4. **All-internal MTL story; no cascade baseline** — SEVERE → **add a cascade category→region baseline.**
5. **Missing feature-concat control (win could be feature injection)** — SEVERE, cheapest fix → **add HGI⊕features → same head.**
6. **No joint-vs-Markov rows / no majority-class category floor** — MODERATE+ → **tabulate joint-vs-Markov + majority-class floor.**
7. **Cross-protocol category table + superseded POI-RGNN numbers** — MODERATE → **add a protocol-reconciliation footnote + cite canonical POI-RGNN.**
8. **Only one learned category baseline after CTLE is demoted** — MODERATE → **add GETNext-category-head (or SGRec/CSLSL).**
9. **TX is 2/5 folds; all TX margins provisional** — LOW-MODERATE → **finish TX to 5/5 (or flag every TX cell).**
10. **No 2022-25 graph wave / no LLM zero-shot for a 2026 venue** — LOW → **ReHDM + a one-line GETNext/STHGCN acknowledgment.**

**The two I would actually reject on if unaddressed: #1 (CTLE strawman) and #4 (no external/cascade MTL comparator).** #1 makes your novelty claim look engineered; #4 makes your MTL thesis untested against the field. The single highest-leverage cheap fix across the whole set is **#5 (feature-concat control)** — it costs almost nothing and closes the one hole CTLE can never close.