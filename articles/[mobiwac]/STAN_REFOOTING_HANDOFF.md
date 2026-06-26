# MobiWac 2026: Region-baseline completion handoff (faithful STAN audit + ReHDM + HMT-GRN)

> **Why this exists (audit finding, 2026-06-26).** Table 3's region externals are on the **wrong footing** vs our
> headline numbers. The board (our STL/MTL region) is **seed 0, gated stride-1 overlap (`check2hgi_dk_ovl`,
> MIN_SEQ=10), fp32**. But:
> - **STAN-`stl_hgi`** (AL 62.88 / AZ 54.86 / FL 73.58 / CA 60.45 / TX 62.70) was measured at **seed 42** and on the
>   **non-overlap** windowing (source JSONs `region_head_<state>_region_5f_50ep_STAN_HGI_<state>_5f50ep.json`, dated
>   2026-04-25, pre-overlap-board). Wrong seed + wrong windowing.
> - **Istanbul STAN** (PR #51) was run on **our Check2HGI** (`stl_check2hgi`) and on **set-a** (non-overlap): its
>   70.39 lands on the set-a ceiling 70.37, not the board stride-1 ceiling 74.80. Wrong substrate + wrong windowing.
> - **ReHDM-faithful** (AL 66.06 / AZ 54.65 / FL 65.68) is on its **own** protocol (chronological 80/10/10 + 24h
>   sessions + 5 seeds), not our 5-fold user-disjoint CV. This is acceptable AS a published-method reference, but it
>   is NOT a paired/matched comparison.
> - **HMT-GRN** is board-matched (seed 0, stride-1 overlap) at all 6 states (TX completed PR #38: reg 53.85 / cat
>   25.81) — the one clean region external today, though an HMT-GRN-*style* adaptation, not a strict reproduction (see #3).
>
> **Decision (user, 2026-06-26 — UPDATED after the faithful-STAN literature + implementation audit):**
> 1. **PHASE 1 (do first): run FAITHFUL STAN** — STAN's OWN embeddings learned end-to-end **from raw**
>    (`research/baselines/stan/`), NOT fed our HGI/Check2HGI embedding (the literature norm; feeding STAN a
>    pretrained embedding is non-standard). Run it with **STAN's OWN native sequence construction** (NOT our stride
>    windowing — audit it first, see Phase 1); the shared protocol is only seed 0, user-disjoint folds, region target,
>    Acc@k. Scope **AL/AZ/FL** (CA/TX faithful-STAN is infeasible at scale → footnote infeasible, like ReHDM). **Istanbul STAN
>    is on the M4 — EXCLUDED here.**
>    ⚠ **The current faithful-STAN v4 numbers (AL 34.46 / AZ 38.96, below the Markov floor) are UNDER-TRAINED
>    ARTIFACTS — DO NOT CITE THEM.** A two-agent audit (2026-06-26) found they are confounded by under-training
>    (best-epochs at 49/50, still climbing), stride-9 data starvation (~9x too few windows), and an under-powered
>    STAN-DERIVED head. The Phase-1 re-run MUST apply the audit fixes (§"Phase 1") before any STAN number is reported.
> 2. **PHASE 2 (after Phase 1): run ReHDM-faithful** — **AL/AZ/Istanbul in parallel first, then FL/CA/TX as
>    possible** (faithful is heavy at FL/CA/TX scale; footnote-infeasible is acceptable). ReHDM-faithful is reported
>    as a **published-method reference under its own protocol** (chronological split, 5 seeds), never a paired cell.
> 3. **Comparability hierarchy:** **HMT-GRN** (region-native, multi-task, board-matched — **HMT-GRN-***style***, not
>    strictly faithful**: its own end-to-end LSTM trunk + train-only region-transition prior, learned from raw on our
>    tract targets; the GRN **graph module and hierarchical beam search are dropped** as next-POI is out of scope —
>    deviation ledger in `docs/baselines/next_region/hmt_grn.md`) is the **PRIMARY** region-native comparison. Note the
>    asymmetry vs the strict-faithful bar this handoff imposes on STAN: HMT-GRN is independent (own architecture, not our
>    substrate) but is an *adaptation*, not a reproduction. **STAN (faithful, from raw — after the fixed re-run)** and **ReHDM (own protocol)** are
>    **SECONDARY references, each labeled.** The substrate-bound **STAN-`stl_hgi`** (STAN on our HGI embedding) is now
>    ONLY an OPTIONAL, explicitly-labeled **ablation** (isolates substrate vs architecture), NEVER the headline STAN
>    cell. Until the fixed faithful re-run lands, the STAN cells in Table 3 are **PENDING — cite no STAN number.**

---

## House rules
- **The shared "common protocol" is the DATA, the SPLIT, the TARGET, and the METRIC — NOT the windowing.** Every
  faithful baseline runs on our states, our **seed-0 user-disjoint 5-fold split**, the **region target** (census
  tract / mahalle), and **Acc@k**, but **each keeps its OWN native sequence construction, embeddings, and training
  recipe.** Do NOT force our stride-1 / window-9 pipeline onto a faithful baseline — that is exactly the mistake this
  handoff corrects. (Only the OPTIONAL substrate-bound `stl_hgi` ablation uses the board stride-1 windowing + log_T.)
- **Faithful STAN = STAN's own embeddings + own sequence construction + own attention**, learned from raw
  (`research/baselines/stan/`), NOT fed our HGI/Check2HGI embedding and NOT sliced into our windows. The
  substrate-bound `stl_hgi` variant (STAN on HGI, our windowing) is an OPTIONAL labeled ablation only.
- **Train to convergence:** the per-fold best-epoch must land BEFORE the epoch cap (the prior run clipped at 49/50 —
  under-trained). Raise epochs / early-stop on a real plateau; verify macro-F1 is not ~0 and Acc@1 is sane.
- fp32; never cite a NaN-collapsed or degenerate fold. Commit one JSON per state + a finding; branch + PR, no merge
  to main (the orchestrator audits).

## Phase 1 — Faithful STAN: AUDIT against the paper FIRST, then execute (do FIRST) — ✅ DONE (AL/AZ/FL/Istanbul)
> **✅ CLOSED for AL/AZ/FL/Istanbul (PR #53 + #54, merged 2026-06-26).** The audited **v5/v6** re-implementation applied
> all 6 faithfulness fixes (STAN-native prefix-expansion sequences; restored matching layer + interval embedding;
> constant-LR convergence; etc.), passed a two-agent audit + independent GO code review, and was ~85× optimized
> (`F.embedding`+`torch.compile`, audit≈compiled within 0.1 pp). It **CONVERGED** (best-epochs 5–12, not clipped) and
> **clears the Markov floor AND stays below our joint model**: **AL 60.72 / AZ 49.86 / FL 72.99 / Istanbul 61.86**
> (reg Acc@10, seed 0 × 5f). The v4 collapse (AL 34.46 / AZ 38.96) is superseded — never cite. **FL completed (PR #54):
> 5-fold v6, Acc@10 72.99 ± 0.34** (< our joint reg 77.28; the fold-0-only 0.7307 checkpoint is superseded by the
> committed 5-fold JSON). **CA/TX footnoted infeasible-at-scale.** Table-3 STAN cells filled.
> Methodology preserved below for the record. Findings: `docs/studies/closing_data/FAITHFUL_STAN_FINDINGS.md`.

Faithful STAN is its OWN code (`research/baselines/stan/` — `model.py` + `etl.py` + `train.py`): STAN learns its own
embeddings, builds its own sequences, and applies its own spatio-temporal interval attention. It is **NOT** the
`next_stan` region-head ablation (the substrate-bound `stl_hgi` variant, now an optional ablation only).

> ⚠ **The current faithful-STAN v4 numbers (AL 34.46 / AZ 38.96, below the Markov floor) are NOT trustworthy —
> SUPERSEDE them, do not cite.** A two-agent audit (2026-06-26) found them confounded by (i) under-training
> (best-epochs clipped at 49/50, still climbing), (ii) a head that collapsed to a proximity prior (macro-F1 ~0), and
> (iii) an ETL that imposes OUR stride-9 windowing on STAN instead of STAN's native sequence construction.

### A. Audit faithful STAN against the paper + public code (do this FIRST, before any run)
Read the STAN paper (Luo et al., WWW 2021, arXiv 2102.04095) and the authors' reference repo
(`github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation` — `layers.py` / `models.py` /
`load.py` / `train.py`). Audit OUR `research/baselines/stan/{model.py,etl.py,train.py}` against it and **fix every
faithfulness gap before running**:
1. **Sequence construction — faithful STAN does NOT use our stride windows.** STAN trains with **prefix-expansion
   over each user's full trajectory** (predict position t from prefix 1..t-1, with its own max-length truncation),
   not fixed stride-9 / window-9 slices. Our `etl.py` imposes stride-9 window=9 (MIN_HISTORY=5) — a deviation that
   also starves training. **Reproduce STAN's native construction faithfully** (full-trajectory prefix expansion +
   STAN's max-len); the shared protocol is only our **data + user-disjoint fold split + region target + Acc@k**.
2. **Matching layer + interval embedding.** The v4 head replaced STAN's learned `Linear(max_len,1)` matching collapse
   with a softmax-weighted mixture, re-parameterized the interval embeddings (64-bin scalar table vs STAN's
   two-endpoint vector interpolation), and stripped residual/LayerNorm → it collapses to a proximity prior. Restore
   these toward the reference (`layers.py`: `Attn` / `Embed` / `SelfAttn` / `MultiEmbed`) so we benchmark STAN, not a
   crippled STAN-derived variant.
3. **Record the audit** in `docs/baselines/next_region/stan.md`: a short "faithfulness vs reference" table — each
   component FAITHFUL / FIXED, with before/after — so the comparison is defensible to a reviewer.

### B. Execute faithful STAN to convergence
After the audit fixes, run on our data + **seed-0 user-disjoint 5-fold split**, region target, **trained to
convergence** (raise the epoch budget / early-stop on a real Acc@10 plateau — best-epoch must land BEFORE the cap).
Scope: **AL, AZ, FL** (faithful STAN is **infeasible at CA/TX scale** → footnote "infeasible at scale", like ReHDM;
HMT-GRN + Markov carry CA/TX). **Istanbul STAN is on the M4 — not here.** Verify per fold: macro-F1 above floor,
Acc@1 sane (not the v4 collapse), no NaN. Confirm flags against `research/baselines/stan/train.py --help`.

**Acceptance (Phase 1):** faithful STAN, **audited-faithful (A) AND converged (B)**, at AL/AZ/FL (seed 0); CA/TX
footnoted infeasible. Commit one JSON per state + the `stan.md` faithfulness-audit section; supersede the v4/seed-42
numbers. **If audited-faithful + converged STAN STILL lands below the Markov floor at AL/AZ, THAT is the honest
reportable result** (STAN is built for fine next-POI, not coarse regions); if it clears Markov, report that. The
number is trustworthy only after BOTH the faithfulness fixes and convergence.

## Phase 2 — ReHDM faithful (do AFTER Phase 1)
Run ReHDM in its **faithful** form (its own architecture + raw inputs + own protocol: chronological 80/10/10 + 24h
sessions + 5 seeds). **Order: AL/AZ/Istanbul in parallel first, then FL/CA/TX as possible.**
- **AL/AZ** faithful already exist (66.06 / 54.65) — re-confirm or reuse; cheap, run in parallel with Istanbul.
- **Istanbul** faithful is NEW: it needs an **FSQ→mahalle region-assignment adapter** (the ReHDM ETL assigns
  regions via US-only geometry; map the Istanbul target to the mahalle taxonomy the board uses). Run it alongside
  AL/AZ. If the adapter proves infeasible by the deadline, footnote Istanbul ReHDM as not-available (do NOT fall
  back to `stl_check2hgi` on our substrate — that is dropped).
- **FL** faithful already exists (65.68); **CA/TX** faithful is heavy (~75–120 h/state) — run **as possible**, else
  footnote "faithful infeasible at scale" (as today).

ReHDM is reported as a **published-method reference under its own protocol**, gap-to-ceiling, NOT a paired/matched
cell. Never report it as if it were on our folds. The Istanbul ReHDM `stl_check2hgi` (running in PR #51, on our
substrate + set-a) is **dropped** — superseded by the faithful run above (or the not-available footnote).

## Phase 3 — HMT-GRN completion (the PRIMARY region-native baseline) — ✅ DONE
HMT-GRN is the **primary** region-native comparison and is board-matched (seed 0, stride-1 overlap, region-native
end-to-end) at **all 6 states: AL 57.05 / AZ 43.70 / FL 63.74 / CA 49.61 / TX 53.85 / Istanbul 60.4** (reg Acc@10;
`comparison.md` / `hmt_grn.md` reduced board). Unlike STAN/ReHDM it CAN be a matched cell (on our footing) — but it is
an **HMT-GRN-*style* adaptation, NOT a strict reproduction** (own LSTM trunk + region-transition prior from raw; graph
module + hierarchical beam search dropped — see the comparability note above and `hmt_grn.md`). Both gaps are CLOSED:
1. ✅ **TX DONE** — TX HMT-GRN finished on the board footing (seed 0 × 5 folds, MPS, PR #38): **reg Acc@10 53.85 /
   cat F1 25.81** (per-fold reg 53.5/54.2/53.1/54.5/54.0). Table-3 `--`$^{\S}$ cell filled with **53.85**; the
   `$^{\S}$` footnote carries the HMT-GRN-*style* deviation ledger.
2. ✅ **Cells re-confirmed board-matched** (seed 0, stride-1) against the MACS table; Istanbul resolved to **60.4
   (stride-1)**, not 56.56. HMT-GRN sits **below our MTL reg at every state** including TX (53.85 < 67.02):
   AL 57.05 < 69.81 / AZ 43.70 < 59.34 / FL 63.74 < 77.28 / CA 49.61 < 65.66 / TX 53.85 < 67.02 /
   Istanbul 60.4 < 74.28. ✅ all clear, the "joint beats the primary region-native at every state" claim holds.

## After the runs: paper-doc updates
- **Table 3** (`src/tables/tbl3_results.tex`): fill the STAN cells with the **audited-faithful, converged, seed-0**
  values (AL/AZ/FL; CA/TX footnoted infeasible); the STAN footnote stays "STAN run faithfully (its own embeddings and
  sequence construction, from raw) on our data/folds/region targets"; the HMT-GRN TX cell is **FILLED (53.85, PR #38)**
  and the HMT column now carries a `$^{\S}$` "HMT-GRN-*style* / deviation-ledger" footnote (NOT "faithful"); the ReHDM
  footnote stays "ReHDM under its own published protocol (reference)". The substrate-bound `stl_hgi` STAN, if kept, is a
  clearly-labeled ablation row, never the headline.
- **`docs/baselines/next_region/comparison.md`** + `stan.md`: add the faithful board-footing STAN row; mark the old
  seed-42 / v4 `FAITHFUL_STAN` numbers superseded-for-the-paper (under-trained, kept only for the v1→v4 audit trail).
- **`RESULTS_BOARD.md §4`** + **`PAPER_PLAN.md §5.4 / §7`**: faithful STAN board-footing done; ReHDM = footnoted reference.
- **Istanbul row of Table 3**: STAN comes from the **M4** run (not this handoff); POI-RGNN/Markov category cells
  from PR #51 are windowing-robust and can go in now (they sit far below our 53.20/59.89 regardless).

## Acceptance checklist
**Phase 1 (faithful STAN, Gowalla): ✅ DONE for AL/AZ/FL/Istanbul (PR #53 + #54)**
- [x] **Faithfulness audit done** vs the paper + reference repo (`stan.md` "faithfulness vs reference" section +
  `FAITHFUL_STAN_FINDINGS.md §2`: sequence construction = STAN's native prefix-expansion; matching layer + interval
  embedding restored toward the reference; two-agent audit + independent GO review).
- [x] Faithful STAN (own embeddings + own sequence construction) committed for **AL/AZ** (60.72 / 49.86), **FL**
  (72.99, 5-fold v6, PR #54) **+ Istanbul** (61.86) at **seed 0, converged**; CA/TX footnoted infeasible-at-scale.
- [x] Acc@10 clears the Markov floor (AL 60.72>47.01 / AZ 49.86>42.96 / FL 72.99>65.05 / Istanbul 61.86>52.5) and stays below our joint;
  best-epochs 5–12 land BEFORE the 200-cap; no NaN; macro-F1 above the v4 ~0 collapse.
- [x] Old v4 (AL 34.46 / AZ 38.96) struck as a collapse artifact — superseded, never cite.

**Phase 3 (HMT-GRN, the primary): ✅ COMPLETE**
- [x] TX HMT-GRN finished on the board footing (seed 0 × 5f, MPS, PR #38: reg 53.85 / cat 25.81); Istanbul value
  resolved (60.4 stride-1); all 6 cells re-confirmed board-matched and below our MTL reg. Table-3 TX cell filled.
- [x] HMT-GRN labeled **HMT-GRN-*style*** (own LSTM trunk + transition prior from raw; graph + beam search dropped),
  NOT "faithful" — deviation ledger in `hmt_grn.md`; Table-3 `$^{\S}$` footnote and the prose carry the qualifier.

**Phase 2 (ReHDM, after Phase 1):**
- [ ] ReHDM-faithful AL/AZ/Istanbul run in parallel (Istanbul via the FSQ→mahalle adapter, or footnoted not-available).
- [ ] ReHDM-faithful FL/CA/TX as possible, else footnoted infeasible-at-scale.
- [ ] ReHDM labeled own-protocol reference (never a paired cell); the Istanbul `stl_check2hgi` stop-gap dropped.
- [ ] seed-42 STAN numbers struck from the paper artifacts.
