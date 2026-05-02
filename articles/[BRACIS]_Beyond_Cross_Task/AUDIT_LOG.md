# AUDIT_LOG.md — Working notes for the meticulous-detail + macro-BRACIS audit

> **Purpose.** This is a running scratchpad for the third-pass audit (post commits 7a60e1c → ed90e8a → 6de13ca). Two parallel agents are running: a numerical audit and a macro-level BRACIS-fit audit. I record (i) the canonical-source table I'm cross-checking against, (ii) the commit-history timeline, (iii) the agent verdicts, and (iv) the synthesised fix list.
>
> **Lifetime.** This file is paper-side working memory. Delete before final commit if it bloats the repo, or keep as `audit_log_3.md` for traceability.

---

## §1 · Canonical numerical anchors (what every cited number must match)

Source: `docs/studies/check2hgi/results/RESULTS_TABLE.md §0` (v9, 2026-05-02 — TX recipe-selection upgraded to n=20 multi-seed, B9 paper-grade at FL/CA/TX; v8: AL/AZ/FL cat-Δ Wilcoxon, CA recipe multi-seed).

### §0.1 Five-state architectural-Δ (MTL B9 vs matched-head STL) — v10

| State | n_pairs | MTL B9 reg Acc@10 | STL `next_stan_flow` Acc@10 | Δ_reg pp | p_reg | MTL B9 cat F1 | STL `next_gru` cat F1 | Δ_cat pp | p_cat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AL | 20 | 50.17 ± 0.24 | 61.21 ± 0.18 | **−11.04** | **1.9e-06** | 40.57 ± 0.24 | 41.35 ± 0.17 (n=4 seeds) | **−0.78** | **0.036** (small-significant negative) |
| AZ | 20 | 40.78 ± 0.07 | 53.06 ± 0.15 | **−12.27** | **1.9e-06** | 45.10 ± 0.19 | 43.90 ± 0.17 (n=4 seeds) | **+1.20** | **<1e-04** |
| FL | 5 | 63.34 ± 0.11 | 70.62 ± 0.09 | **−7.99** | 0.0625 | **68.51 ± 0.51** (F51, n=5) | 67.16 ± 0.13 (n=4 seeds) | **+1.52** | 0.0625 |
| **CA** | **20** | **47.35 ± 0.11** | **56.85 ± 0.09** | **−9.50** | **2e-06** | **64.07 ± 0.14** | **62.39 ± 0.13** | **+1.68** | **2e-06** |
| **TX** | **20** | **42.84 ± 0.14** | **59.44 ± 0.09** | **−16.59** | **2e-06** | **65.00 ± 0.11** | **63.11 ± 0.13** | **+1.89** | **2e-06** |

**v10 changes vs v8 (2026-05-02):** CA and TX upgraded from n=5 single-seed to n=20 (seeds {0,1,7,100} × 5 folds). Source: `ARCH_DELTA_WILCOXON.json`.
- CA: Δ_reg = −9.50 pp p=2e-06; Δ_cat = +1.68 pp p=2e-06. Both paper-grade significant.
- TX: Δ_reg = −16.59 pp p=2e-06; Δ_cat = +1.89 pp p=2e-06. Both paper-grade significant.

**v8 changes vs v7:**
- AL p_cat: was "—" pending; now **0.036** (n=20 multi-seed; 14/20 fold-pairs negative; small-significantly negative).
- AZ p_cat: was "—" pending; now **<1e-04** (n=20; 18/20 fold-pairs positive; significantly positive).
- FL MTL B9 cat F1: refined from 68.59 → **68.51 ± 0.51** (multi-seed pooled). Δ_cat: refined from +1.43 → **+1.52** (paired Δ from `GAP_FILL_WILCOXON.json`).

**Rules (v10).**
- AL is **small-significantly negative** (p=0.036, n=20). Honest framing: report the significance AND the small magnitude (0.78 pp ~ 1.9% relative on 41% F1 scale).
- AL/AZ/CA/TX p_reg = 1.9–2e-06 (multi-seed n=20 pooled). FL p=0.0625 (n=5 ceiling).
- CA/TX both axes now paper-grade at n=20. FL remains n=5 ceiling (sign-consistent, not formally significant at α=0.05).
- CA recipe-selection axis (B9 vs H3-alt) is multi-seed n=20 in v8 §0.4 — paper-grade significant on both tasks.

### §0.2 Δm joint score (CH22 leak-free)

| State | n_pairs | Δm-MRR (%) | n+/n− | p_greater | p_two | Δm-Acc@10 (%) | n+/n− | p_two |
|---|---:|---:|:-:|---:|---:|---:|:-:|---:|
| AL | 5 | **−24.84** | 0/5 | 1.0000 | 0.0625 | −22.41 | 0/5 | 0.0625 |
| AZ | 5 | **−12.79** | 1/4 | 0.9688 | 0.1250 | −14.53 | 0/5 | 0.0625 |
| **FL** | **25** | **+2.33** | **25/0** | **2.98e-08** | **5.96e-08** | **−1.12** | 4/21 | **3.20e-05** |
| CA | 5 | −1.61 | 1/4 | 0.9375 | 0.1875 | −6.85 | 0/5 | 0.0625 |
| TX | 5 | −4.63 | 0/5 | 1.0000 | 0.0625 | −11.60 | 0/5 | 0.0625 |

**Rules.**
- FL multi-seed n = 25 (5 seeds × 5 folds) — only paper-grade significant Δm cell.
- All other states are at n = 5 single-seed ceiling; AL/CA/TX two-sided p = 0.0625 minimum; AZ Δm-Acc@10 also at ceiling.
- Δm-Acc@10 negative everywhere; Δm-MRR positive at FL only (FL Acc@10-vs-MRR split is itself a paper-worthy mechanism note).

### §0.3 Substrate ablation (CH16 cat side + CH15 reframing reg side)

**Cat panel (matched-head STL `next_gru`, leak-free):**

| State | C2HGI cat F1 | HGI cat F1 | Δ pp | Wilcoxon p_greater | Pos/Neg |
|---|---:|---:|---:|---:|:-:|
| AL | 41.35 ± 0.17 (multi-seed) | 25.26 ± 1.18 | **+15.50** | 0.0312 | 5/0 |
| AZ | 43.90 ± 0.17 (multi-seed) | 28.69 ± 0.79 | **+14.52** | 0.0312 | 5/0 |
| FL | 63.43 ± 0.98 | 34.41 ± 1.05 | **+29.02** | 0.0312 | 5/0 |
| CA | 59.94 ± 0.59 | 31.13 ± 1.04 | **+28.81** | 0.0312 | 5/0 |
| TX | 60.24 ± 1.84 | 31.89 ± 0.55 | **+28.34** | 0.0312 | 5/0 |

**Reg panel (matched-head STL `next_stan_flow`, leak-free, TOST):**

| State | C2HGI Acc@10 | HGI Acc@10 | Δ Acc@10 | Wilcoxon p_greater | TOST δ=2pp | TOST δ=3pp |
|---|---:|---:|---:|---:|:-:|:-:|
| AL | 59.15 ± 3.48 | **61.86 ± 3.29** | −2.71 | 1.0000 | ✗ | ✗ |
| AZ | 50.24 ± 2.51 | **53.37 ± 2.55** | −3.13 | 1.0000 | ✗ | ✗ |
| FL | 69.22 ± 0.52 | **71.34 ± 0.64** | −2.12 | 1.0000 | ✗ | ✓ |
| CA | 55.92 ± 1.20 | **57.77 ± 1.12** | −1.85 | 1.0000 | ✓ | ✓ |
| TX | 58.89 ± 1.28 | **60.47 ± 1.26** | −1.59 | 1.0000 | ✓ | ✓ |

**Rules.**
- Cat substrate Δ = +14.52 to +29.02 pp; **NOT strictly monotone** (CA +28.81 < FL +29.02). Range is "+14.5 to +29 pp" or "+14 to +29 pp"; "+15 to +33" is wrong (33 was MTL counterfactual).
- Reg substrate: HGI ≥ Check2HGI by 1.6 to 3.1 pp; TOST passes (tied) at CA/TX δ=2pp; FL δ=3pp; AL/AZ fail TOST.

### §0.4 Recipe selection (B9 vs H3-alt)

| State | n_pairs | Δ_reg pp (B9 − H3-alt) | p_reg | Δ_cat pp | p_cat | Verdict |
|---|---:|---:|---:|---:|---:|---|
| AL | 20 | −0.35 | 1.9e-03 | −2.22 | 1.9e-06 | H3-alt > B9 on cat; reg tied |
| AZ | 20 | −0.09 | 0.23 | −0.96 | 7.1e-04 | H3-alt > B9 on cat; reg tied |
| FL | 25 | +3.48 | 3.0e-08 | +0.42 | 1.3e-05 | B9 > H3-alt on both |
| CA | 5 | +4.74 | 0.062 | +0.72 | 0.125 | B9 directional |
| TX | 5 | +1.76 | 0.125 | +0.64 | 0.125 | B9 directional |

### §0.5 External literature baselines — `next_region` (Acc@10)

| Baseline | Variant | AL | AZ | FL | CA | TX |
|---|---|---:|---:|---:|---:|---:|
| Markov-1-region | — | 47.01 ± 3.55 | 42.96 ± 2.05 | 65.05 ± 0.93 | 52.09 ± 0.80 | 54.94 ± 0.46 |
| **STAN** | `faithful` | 34.46 | 38.96 | 65.36 | (deferred) | (deferred) |
| STAN | `stl_check2hgi` | 59.20 | 52.24 | 72.62 | 58.82 | 61.35 |
| STAN | `stl_hgi` | **62.88** | **54.86** | **73.58** | **60.45** | **62.70** |
| **ReHDM** ‡ | `faithful` | **66.06** | 54.65 | 65.68 | (deferred) | (deferred) |

**ReHDM CA/TX deferred** — hypergraph collaborator-pool quadratic in region cardinality.

### §0.6 External literature baselines — `next_category` (macro-F1)

- POI-RGNN faithful FL = 34.49; CA = 31.78; TX = 33.03.
- C2HGI lifts cat by **+28-33 pp over POI-RGNN** at FL/CA/TX (uses MTL counterfactual numbers; OR +28-29 if matched-head STL `next_gru`). Wording in T5/§5.3 must be precise about which axis.

---

## §2 · Commit-history timeline (study side)

(filled in by my git log walk; running list)

---

## §3 · Agent verdicts (filled in when they return)

### Agent A — numerical audit verdict (returned)

Two high-severity numerical mismatches:

1. **AL substrate Δ stated as +14.5 pp; canonical is +15.50** (RESULTS_TABLE §0.3). The minimum of the substrate cat-Δ range is **AZ at +14.52**, not AL. Files affected:
   - `AGENT.md:36` (C1 bullet) — "+14.5 at AL, +14.5 at AZ" (AL wrong)
   - `PAPER_DRAFT.md:54` (Beat 5 findings) — same
   - `PAPER_DRAFT.md:143` (§5.1 Beat 1) — same
   - `STATISTICAL_AUDIT.md:37` ("DO WRITE" example) — has "+14.5 at AL, +15.5 at AZ" — **REVERSED** from canonical
   - `PAPER_STRUCTURE.md:44, 109, 169` — uses aggregated "+14.5 pp at AL/AZ" — should clarify split is 15.5/14.5
   - **Correct cells (no fix needed):** PAPER_STRUCTURE.md:197 (claim map: lists per-state +15.50/+14.52/...); TABLES_FIGURES.md:131 (T2 column-ordered cells); TABLES_FIGURES.md:244 (F1 caption).

2. **POI-RGNN FL "~31.8 %" is ambiguous.** RESULTS_TABLE §0.6 shows our faithful FL = 34.49; the "31.78" is our faithful CA. The "~31.8 % FL" cited in PAPER_DRAFT.md (and STATISTICAL_AUDIT.md:190) appears to be the published POI-RGNN paper number, but the wording reads as if it were our reproduction. Need to clarify: *"POI-RGNN's published Gowalla state-level numbers (Capanema 2022 reports a 31.8–34.5 pp range across states)"* OR cite our faithful reproduction values (FL 34.49, CA 31.78, TX 33.03).

Source-pointer compliance: clean — no PAPER_CLOSURE_RESULTS cited as primary canon.

All other numbers (n=17 cells) round-trip correctly.

### Agent B — macro BRACIS-fit verdict (returned)

**Verdict:** ready for sub-agent fan-out conditional on three medium-risk pre-blockers.

**Story spine:** coheres consistently across all five article-side files post-Codex audit. C1/C2 framing is consistent end-to-end; voice/protocol guidance aligns with `BRACIS_GUIDE §10.2`. **One real spine wobble:** the "transfer empirically null on next-region" framing (PAPER_STRUCTURE §2 land-line) is sharper than "empirically vacuous on the harder task" (PAPER_DRAFT §7 Beat 1) and stronger than the abstract's "textbook tradeoff". Reviewer eyes will compare — pick one register.

**Pre-fan-out blockers (medium-risk):**
1. **AL/AZ/FL cat-Δ Wilcoxon "pending re-run"** against v7 multi-seed STL ceiling — surface this as a real BRACIS-rigour gap, not a TODO. Either run it before fan-out, or commit to "5/5 fold sign-pattern" framing.
2. **D6 anonymous code link** is "pending" — reviewers click; 404 is desk-rejection-adjacent.
3. **Abstract clarity edits (3):** (a) n = 5 ceiling p = 0.0312 lacks context for cold readers, (b) "head-invariant" is jargon, (c) "8 to 17 pp" needs "on next-region Acc@10" qualifier.

**Other findings (medium):**
- **CoUrb arc tension (§1 Beat 3):** per-visit-vs-per-POI is an orthogonal axis to CoUrb's per-modality decomposition; needs to be made explicit as a third bottleneck dimension.
- **C2 cat-lift register inconsistency:** abstract uses "small additional cat lift"; AGENT/PAPER_DRAFT C2 bullets just say "lifts cat" — align register.
- **POI-RGNN reproduction caveat** mentioned in §5.1 but missing from the §7 Limitations list.

**BRACIS-fit risk matrix:**
- Empirical depth: **strong** (5 states, mechanism counterfactual, drop-in ablation).
- Statistical rigour: **borderline-strong** (paired Wilcoxon + TOST + Δm + sign-consistency; soft spots are AL/AZ/FL cat Wilcoxon pending + CA/TX seed=42).
- Cross-state generalisation: **strong** (5 states, sign-consistent cost).
- External baselines: **adequate** (ReHDM CA/TX deferral honest).
- Novelty: **adequate, not strong** (empirical, not algorithmic — substrate-asymmetry framing carries it).
- Honest framing: **strong** (matches BRACIS 2023 best-paper pattern).

**Items I cannot fix unilaterally** (surface to user):
- D6 anonymous code link snapshot generation (Anonymous GitHub setup).
- AL/AZ/FL cat-Δ Wilcoxon re-run (compute task, not doc task).

---

## §4 · Synthesised fix list

**HIGH (apply now):**
1. AL substrate Δ: fix `+14.5` → `+15.5` at AL across 4 files (AGENT.md:36, PAPER_DRAFT.md:54+143, STATISTICAL_AUDIT.md:37 reversed). Aggregated "+14.5 pp at AL/AZ" in PAPER_STRUCTURE.md needs split clarification.
2. POI-RGNN ambiguity: clarify "~31.8 % FL" in PAPER_DRAFT §5.1 Beat 2 (and any other appearance) — published-paper number vs our faithful reproduction (FL = 34.49).

**MEDIUM (apply now):**
3. Abstract clarity (samplepaper.tex): 3 edits (n=5 ceiling caveat, "head-invariant" → "across four head probes", "Acc@10" qualifier).
4. Spine register match: harmonize §2 land-line / §7 Beat 1 / abstract on the transfer-claim sharpness.
5. CoUrb arc tension: make per-visit-vs-per-POI an explicit *third* bottleneck axis in §1 Beat 3.
6. C2 cat-lift register: align "lifts cat" → "small additional cat lift" across AGENT/PAPER_DRAFT bullets to match abstract.
7. POI-RGNN reproduction caveat → §7 Beat 3 limitations list.

**LOW (verify, fix only if drift):**
8. C1-mechanism ~72% — verify "at AL" inline qualifier in every appearance.

**SURFACED TO USER (cannot fix unilaterally):**
9. D6 anonymous code link snapshot.
10. AL/AZ/FL cat-Δ Wilcoxon re-run against v7 multi-seed STL ceiling.

---

## §4 · Synthesised fix list

(filled in after agents return)

---

## §5 · Final disposition

**Fixes applied this turn (8 files touched):**
- AL substrate Δ corrected from +14.5 → +15.5 in: AGENT.md C1, PAPER_DRAFT.md §1 Beat 5 + §5.1 Beat 1 + §8 conclusion, STATISTICAL_AUDIT.md §4.3 (reversed example), PAPER_STRUCTURE.md §1 banner + §1 Findings + §3 T2 inventory + §5.1 caption + §3 T2 inventory caption.
- POI-RGNN ambiguity resolved across PAPER_DRAFT.md §5.1 Beat 2, STATISTICAL_AUDIT.md §7 DO WRITE example, TABLES_FIGURES.md T5 sample + §6 audit-items list. Now consistently uses our faithful reproduction values (FL 34.49, CA 31.78, TX 33.03 per `RESULTS_TABLE §0.6`); the published 31.8–34.5 pp range is contextualised as the published (non-user-disjoint) configuration.
- Abstract clarity: rewritten to 149 words (under 150 LNCS limit) with three macro-agent edits absorbed — n=5 ceiling caveat, "head-invariant" → "across four head probes", "8 to 17 pp" → "8 to 17 percentage point cost on next-region top-10 accuracy".
- Spine register match: PAPER_STRUCTURE §2 land-line softened from "transfer empirically null on next-region" to "small additional cat lift / sign-consistent reg cost / textbook tradeoff" — matches the abstract register; the sharper "empirically vacuous" formulation is reserved for §7 Beat 1 discussion.
- CoUrb arc tension: PAPER_DRAFT §1 Beat 3 now explicitly positions per-visit-vs-per-POI as an *orthogonal* axis to CoUrb's per-modality decomposition (granularity vs. what-features) — not as a continuous follow-up.
- C2 cat-lift register: aligned across AGENT.md C2 + PAPER_DRAFT.md §1 Beat 5 + §5.2 Beat 1 + §7 Beat 1 + §8 conclusion to use "small additional cat lift" matching the abstract.
- POI-RGNN reproduction caveat added to PAPER_DRAFT §7 Limitations as new item (iv).
- §7 Limitations also expanded with AL/AZ/FL cat-Δ Wilcoxon "pending re-run" note (unchanged direction across v6→v7 STL refresh).

**Surface to user (cannot fix unilaterally):**
- **D6 anonymous code link** snapshot — needs user to set up Anonymous GitHub or equivalent. This is a real BRACIS-fit risk if not done before submission.
- ~~AL/AZ/FL cat-Δ Wilcoxon re-run~~ **RESOLVED v8 (2026-05-01)** via `gap_fill_wilcoxon.py` + `GAP_FILL_WILCOXON.json`. AL p=0.036 (small-significantly negative); AZ p<1e-04 (significantly positive); FL p=0.0625 (n=5 ceiling, sign-consistent positive). The directional reading is unchanged.
- ~~**NEW (camera-ready audit items):** TX MTL multi-seed at {0,1,7,100}~~ **RESOLVED v9 (2026-05-02)** via `gap_fill_wilcoxon.py` Analysis C. TX B9 vs H3-alt n=20: reg +1.87 pp p=7e-04, cat +0.52 pp p=2e-04 — paper-grade on both axes. B9 confirmed paper-grade at FL/CA/TX; H3-alt at AL/AZ. Scale-conditional narrative fully closed.
- ~~**Remaining open:** CA MTL-vs-STL §0.1 axis multi-seed~~ **RESOLVED v10 (2026-05-02)** via `arch_delta_wilcoxon.py`. CA and TX §0.1 both upgraded to n=20; all four tests p=2e-06. No remaining n=5 single-seed gaps in §0.1 for large-scale states.

**Macro BRACIS-fit verdict (post-consolidation):**
- Story spine: ✅ coherent across all five article-side files.
- Numerical claims: ✅ round-trip to RESULTS_TABLE v7.
- Statistical rigour: ✅ borderline-strong; soft spots honestly disclosed in §7.
- External baselines: ✅ adequate; ReHDM CA/TX deferral honest.
- Honest framing: ✅ matches BRACIS 2023 best-paper pattern.
- **Ready for sub-agent fan-out**, conditional on the two surfaced user-facing items above.

**Lifetime of this file.** Keep as `AUDIT_LOG.md` for traceability through camera-ready; can be moved to `docs/studies/check2hgi/audit_history/` post-acceptance, or deleted from the article folder if it bloats the submission package.
