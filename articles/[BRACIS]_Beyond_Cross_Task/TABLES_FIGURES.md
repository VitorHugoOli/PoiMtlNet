# TABLES_FIGURES.md — Visualization Plan

> **Purpose.** Commit the table / figure inventory, the BRACIS-style layout conventions, the per-claim mapping, and the page-budget arithmetic. Sub-agent A5 (Results) and A6 (Mechanism) implement these layouts; A4 (Setup) lands T1 + the encoding-convention block in §4.3.
>
> **Inputs absorbed.** `STATISTICAL_AUDIT.md` (rigour contract), advisor-pass 2026-05-01 (CH15-reg substrate panel, T3 merge, T6 → prose, F2 keep, encoding convention).

---

## 0 · Statistical encoding convention (lives in §4.3 caption, referenced everywhere)

Drop this once into the §4.3 metrics-and-statistics paragraph and cite the symbols in every table caption that follows.

```
Reporting convention. Cell entries are mean ± σ.
- σ over folds for single-seed runs, over seeds for multi-seed runs (indicated).
- Bold marks the proposed MTL row at each state when compared against an STL ceiling;
  italics mark the STL ceiling for reference.
- Δ columns report MTL − reference, in percentage points.
- Significance footers:
    * = at-ceiling significance for n = 5 paired Wilcoxon (5/5 folds in claimed direction; p = 0.0312 one-sided).
    † = p < 0.05 single-seed paired Wilcoxon (one-sided, claimed direction).
    ‡ = pooled multi-seed paired Wilcoxon (n_pairs declared; reaches sub-1e-4).
    n.s. = not significant.
- TOST non-inferiority at δ ∈ {2 pp, 3 pp} where the claim is equivalence-within-margin.
```

This block is cited verbatim in T1 caption and elided in subsequent captions ("see §4.3").

---

## 1 · BRACIS layout conventions (mandatory)

- **`booktabs`** — `\toprule \midrule \bottomrule`; never `\hline`. Modern Springer LNCS / 2023-2024 BRACIS norm.
- **No vertical rules.** Booktabs convention; reviewers read this as polished.
- **`\footnotesize`** for any table with > 6 columns. Saves space without losing legibility at LNCS column width.
- **One-decimal precision** for F1 / Acc / MRR. **Two-decimal** for Δm percentages and TOST margins.
- **Right-align numerics** with `S` columns from `siunitx` (`S[table-format=2.2]`) or `r` columns with manual padding. Right-align makes Δ comparisons glanceable.
- **Bold = proposed MTL row**, *italics = STL ceiling*. Caption sentence makes this explicit ("STL ceilings reported as a reference; bold marks the proposed MTL row at each state.").
- **Per-state blocks** with `\midrule` between blocks for tables that group by state (T3, T5).

---

## 2 · Inventory — final commitment

### 2.1 Required tables

| ID | Caption (working) | Rows × cols | Page | Source artefact |
|---|---|---|:-:|---|
| **T1** | Dataset statistics (FL/CA/TX headline + AL/AZ smaller-scale anchors) | 5 × 7 | 0.4 | `data/checkins/<state>.parquet`, `output/check2hgi/<state>/regions.parquet` |
| **T2** | Substrate ablation: **Check2HGI vs HGI on both tasks**, 5 states. Two-panel: (a) cat F1 STL `next_gru` (Δ +14.5 to +29.0 pp; paired Wilcoxon p = 0.0312 each, head-invariant at AL/AZ); (b) reg Acc@10 STL `next_stan_flow` (HGI ≥ Check2HGI 1.6–3.1 pp at AL/AZ/FL; TOST δ=2pp passes at CA/TX, δ=3pp passes at FL). | 5 × 5 (×2 panels) | 0.6 | **`RESULTS_TABLE.md §0.3` (v7)** + `FINAL_SURVEY.md §2, §4` |
| **T3** | MTL vs STL on both tasks, 5 states, **single merged table** with `\midrule` separating headline (FL/CA/TX) and smaller-scale anchors (AL/AZ). | 5 × 7 | 0.4 | **`RESULTS_TABLE.md §0.1` (v7, multi-seed STL ceiling, paired Δs)** |
| **T4** | Δm joint score (cat F1 + reg MRR primary; cat F1 + reg Acc@10 secondary), 5 states. **FL multi-seed (n=25) bolded as the strongest number in the paper.** | 5 × 5 | 0.4 | **`RESULTS_TABLE.md §0.2` (v7, leak-free CH22 2026-05-01)** |
| **T5** | External baselines per state — per-state block layout (CoUrb's `tabela_comparativa` pattern). cat: Majority, Markov-1-POI, POI-RGNN, MHA+PE, our STL ceiling, our MTL row. reg: Majority, Markov-1-region, STL GRU, STL STAN, STL STAN-Flow, ReHDM (AL/AZ/FL only — CA/TX deferred, see §7), our MTL row. Headline (FL/CA/TX) only; AL/AZ in T5-supp if pages allow. | 3 blocks × ~7 rows | 0.7 | **`RESULTS_TABLE.md §0.5–0.6` (v7)** + `baselines/next_*/results/<state>.json` |

**Total tables: ≈ 2.5 pp.**

### 2.2 Required figure (post-Codex audit)

| ID | Caption (working) | Page | Source |
|---|---|:-:|---|
| **F1** | **Per-visit mechanism bar — REQUIRED** (post-Codex audit promotion). At Alabama (single-state mechanism evidence), three-bar grouped chart of cat F1 for {Check2HGI canonical, Check2HGI POI-pooled, HGI} under matched-head STL `next_gru` and linear probe; ~72 % per-visit / ~28 % training-signal split (matched-head). Explains *why* the substrate is task-asymmetric — per-visit variance is what cat needs; pooling smooths it away for reg. **F1 is the only visual mechanism anchor for the substrate task-asymmetry; cannot be cut.** | 0.4 | `CLAIMS_AND_HYPOTHESES.md §CH19` (whitelisted; AL only — wording must say "at Alabama") |

**Total figures (required): ≈ 0.4 pp.**

### 2.3 Cut to prose (was T6)

The drop-in MTL ablation (FAMO, Aligned-MTL, HSM) reads stronger as a single disciplined sentence than a 3-row table. **Tables for failures-to-recover make negative results look ambiguous; one inline sentence frames it as negative-result rigour.** Drop in §6.2 verbatim:

> *"None of the three drop-in alternatives reaches paired-Wilcoxon significance against H3-alt at FL on `next_region` Acc@10 (FAMO Δ_reg = +0.62 pp, p = 0.219; Aligned-MTL = −0.11 pp, p = 0.844; hierarchical-additive softmax = −3.01 pp, p = 0.313 — all n = 5 single-seed). The architectural reg cost does not respond to balancer or head-capacity drop-ins under our compute budget."*

Saves 0.2 pp; reads better.

### 2.4 Optional — cut order if pages tight

| ID | Caption (working) | Page | Cut order | Source |
|---|---|:-:|:-:|---|
| **F2** | Scale-progression scatter — Δ_reg pp vs n_regions or log(check-ins), 5 dots labelled by state, dashed trend AL→AZ→FL with TX annotated as non-monotone outlier. Demoted from required to optional after the title was reframed away from "Scale-Sensitive". | 0.4 | **cut 1st** — descriptive observation; one sentence in §5.2 / §7 carries the same content | `RESULTS_TABLE.md §0.1` (v7) reg col |
| **F-arch** | Architecture schematic of MTLnetCrossAttn — two task-specific encoders (cat ← check-in seq, reg ← region seq), 8-head bidirectional cross-attention block, residual shared backbone, GRU cat head + STAN-Flow reg head. | 0.5 | **cut 2nd** — standard for BRACIS methods papers but not load-bearing | new figure |

---

## 3 · Page budget arithmetic (post-Codex)

| Section | Prose | Tables | Figures | Total |
|---|---:|---:|---:|---:|
| §1 Intro | 1.5 | — | — | 1.5 |
| §2 Related | 1.5 | — | — | 1.5 |
| §3 Method | 2.5 | — | (F-arch optional 0.5) | 2.5–3.0 |
| §4 Setup | 1.1 | T1 (0.4) | — | 1.5 |
| §5 Results | 2.5 | T2+T3+T4+T5 (2.1) | (F2 optional 0.4) | 5.0–5.4 |
| §6 Mechanism | 1.4 | (T6 cut) | **F1 required (0.4)** | 1.8 |
| §7 Discussion | 1.0 | — | — | 1.0 |
| §8 Conclusion | 0.5 | — | — | 0.5 |
| §9 References | 0.75 | — | — | 0.75 |
| **Total** | **12.75** | **2.5** | **0.4–1.3** | **15.65–16.55** |

**Status: tight at the lower bound (15.65 pp), over at the upper.** Cut F2 first → 15.25 pp; cut F-arch second → 14.75 pp. **F1 is non-negotiable** (carries the substrate task-asymmetry mechanism, which is the headline reading post-Codex audit) — cut F2 + F-arch before touching it.

---

## 4 · Per-table specifications

### T1 — Dataset statistics (§4.1, 0.4 pp)

```
                       Headline (B9 recipe)              Anchors (H3-alt recipe)
              ──────────────────────────────  ─────────────────────────
Statistic       FL         CA         TX        AL          AZ
─────────────────────────────────────────────────────────────────────
Users         X,XXX     X,XXX     X,XXX     X,XXX       X,XXX
Check-ins   XXX,XXX   XXX,XXX   XXX,XXX    XX,XXX      XX,XXX
POIs        XX,XXX    XX,XXX    XX,XXX     X,XXX       X,XXX
Regions      4,703     8,501     6,553     1,109       1,547
Mean traj len  XX.X      XX.X      XX.X      XX.X        XX.X
Cat balance  see Fig    see Fig   see Fig   see Fig    see Fig
```

Caption sentence: *"Five Gowalla user-disjoint state splits. Headline analysis (B9 recipe) at FL/CA/TX; smaller-scale anchors (H3-alt recipe) at AL/AZ included for completeness across the cardinality range."*

**Sub-agent A4 must compute the missing X cells from `data/<state>/` and `output/check2hgi/<state>/` — do not copy stale numbers.**

### T2 — Substrate ablation, two-panel (§5.1, 0.6 pp)

```
Panel (a) — next-category macro-F1 (matched-head STL `next_gru`, 5 folds, single-seed)
              FL          CA          TX          AL          AZ
─────────────────────────────────────────────────────────────────
Check2HGI   63.4±1.0   59.9±0.6   60.2±1.8   40.8±1.7   43.2±0.9
HGI         34.4±1.0   31.1±1.0   31.9±0.5   25.3±1.2   28.7±0.8
─────────────────────────────────────────────────────────────────
Δ (C2−HGI)  +29.0*    +28.8*    +28.3*    +15.5*    +14.5*

Panel (b) — next-region Acc@10 (matched-head STL `next_stan_flow`, leak-free per-fold log_T)
              FL          CA          TX          AL          AZ
─────────────────────────────────────────────────────────────────
Check2HGI   69.2±0.5   55.9±1.2   58.9±1.3   59.2±3.5   50.2±2.5
HGI         71.3±0.6   57.8±1.1   60.5±1.3   61.9±3.3   53.4±2.5
─────────────────────────────────────────────────────────────────
Δ (C2−HGI)  −2.1‡(δ3) −1.9 (δ2) −1.6 (δ2) −2.7      −3.1
            HGI nominally above C2HGI 1–3 pp at AL/AZ/FL; tied at CA/TX (TOST δ=2pp passes).
```

Caption sentence: *"The substrate is task-asymmetric: Check2HGI's per-visit context lifts cat F1 by 14–29 pp at every state (panel a, paired Wilcoxon p = 0.0312, 5/5 folds positive each — at-ceiling for n = 5); on the reg task, where pooling smooths per-visit variation, HGI matches or marginally exceeds Check2HGI (panel b, TOST non-inferiority δ = 2 pp passes at CA/TX, δ = 3 pp passes at FL). The substrate's value is concentrated where per-visit intent matters."*

This panel is the **substrate-vs-architecture decoupling story** — without it a reviewer will assume we cherry-picked the cat side. With it, the substrate's role is precisely scoped.

### T3 — MTL vs STL on both tasks, v7 numbers (§5.2, 0.4 pp)

```
                  Cat F1 (vs STL next_gru)        Reg Acc@10 (vs STL next_stan_flow)
                ─────────────────────────────  ──────────────────────────────────
State  Recipe   STL          MTL          Δ_cat    STL          MTL          Δ_reg
──────────────────────────────────────────────────────────────────────────────────
HEADLINE
FL     B9       67.16±0.13   68.59 (n=5)  +1.43    70.62±0.09   63.34±0.11   −7.99
CA     B9       62.29±0.31   64.23 (n=1)  +1.94    56.86        47.93        −8.92
TX     B9       63.02±0.28   65.04 (n=1)  +2.02    59.32        42.63       −16.69
──────────────────────────────────────────────────────────────────────────────────
SMALLER-SCALE ANCHORS
AL     H3-alt   41.35±0.17   40.57±0.24   −0.78    61.21±0.18   50.17±0.24  −11.04‡
AZ     H3-alt   43.90±0.17   45.10±0.19   +1.20    53.06±0.15   40.78±0.07  −12.27‡
──────────────────────────────────────────────────────────────────────────────────

n_pairs: AL/AZ = 20 (4 seeds × 5 folds);
FL = 25 (5 seeds × 5 folds, reg only — cat side n = 5 because STL multi-seed
landed but Wilcoxon pending re-run); CA/TX = 5 (seed = 42 single-seed at
submission; {0,1,7,100} multi-seed extension is a camera-ready audit item).

Source: docs/studies/check2hgi/results/RESULTS_TABLE.md §0.1 (v7,
2026-05-01 PM). MTL B9 cat numbers unchanged from v6; STL `next_gru` cat
refreshed from multi-seed runs {0,1,7,100} → seed σ replaces fold σ.
Δ_cat p-values for AL/AZ pending re-Wilcoxon against the v7 multi-seed
STL ceiling.
```

Caption sentence: *"With Check2HGI fixed as substrate, MTL vs matched-head STL ceilings on both tasks. Headline (FL/CA/TX) reports the B9 recipe (cosine + alternating-SGD + α-no-WD); smaller-scale anchors (AL/AZ) report H3-alt (per-head LR, constant). Δ_reg is sign-consistent at every state, with the magnitude varying non-monotonically (FL has the smallest cost at −7.99 pp; TX the largest at −16.69 pp). On the cat side, MTL is positive at four of five states (AZ +1.20, FL +1.43, CA +1.94, TX +2.02 pp; directional at CA/TX with 4/5 folds positive, single-seed n = 5 ceiling) and ≈ tied at AL within the multi-seed STL noise (Δ = −0.78 pp, multi-seed STL σ = 0.17 pp). Paired Wilcoxon p-values for AL/AZ/FL cat-Δ pending re-run against the v7 multi-seed STL ceiling and reported as 'pending' in the table; the directional and magnitude reading is unchanged across the v6→v7 STL refresh."*

**Voice cue:** the caption commits to *directional* not *paper-grade* on the cat side until the Wilcoxon re-runs land. Sub-agent A5 must not upgrade the wording without the rerun.

### T4 — Δm joint score, FL multi-seed bolded (§5.2, 0.4 pp)

```
State  n_pairs   Δm-MRR (%)        Δm-Acc@10 (%)
                 (PRIMARY)         (SECONDARY)
─────────────────────────────────────────────────
AL     5         −24.84            −22.41
AZ     5         −12.79            −14.53
FL     **25**    **+2.33** ‡       −1.12 ‡
                 p = 2.98 × 10⁻⁸    p = 3.20 × 10⁻⁵
                 25/25 positive    4/21 positive
CA     5          −1.61            −6.85
TX     5          −4.63           −11.60
─────────────────────────────────────────────────
```

Caption: *"Δm joint score (Maninis 2019; Vandenhende 2021), primary metric cat F1 + reg MRR; secondary cat F1 + reg Acc@10. The FL multi-seed cell (n = 25 fold-pairs) is the strongest paired result in the paper: Δm-MRR = +2.33 % at p = 2.98 × 10⁻⁸ with 25/25 fold-pairs positive, while the same FL run shows Δm-Acc@10 = −1.12 % at p = 3.2 × 10⁻⁵ with 4/21 positive — MTL produces better-ranked region predictions than STL even where raw top-K is worse. Other states are at the n = 5 single-seed ceiling."*

### T5 — External baselines per state, headline only (§5.3, 0.7 pp)

> **ReHDM coverage caveat:** ReHDM rows reported at AL/AZ/FL; **CA/TX cells will be marked "—" with a footnote** *"deferred to camera-ready; dual-level hypergraph's collaborator pool scales quadratically with region cardinality and exceeded our H100 compute budget at 8.5 K (CA) / 6.5 K (TX) regions."* This is a paper-strengthening honest-framing disclosure (per `BRACIS_GUIDE.md §10.2.7`) — silently dropping ReHDM would invite reviewers to ask why; explicitly disclosing the budget constraint frames it as compute-asymmetry rather than methodological gap.


CoUrb's `tabela_comparativa` pattern: per-state block, baselines stacked vertically, columns split between cat F1 and reg Acc@10 / MRR.

```
=== Florida ===
                          Cat F1     Reg Acc@10    Reg MRR
Majority                    —        22.25         —
Markov-1-POI               XX        —             —
Markov-1-region            —         65.05         —
POI-RGNN (Capanema 2019)  ~31.8      —             —
MHA+PE (Zeng 2019)        XX         —             —
STL GRU                    —         68.33         —
STL STAN (Luo 2021)        —         XX            XX
*STL next_stan_flow*        —        70.62         55.04
*STL next_gru (matched)*  67.16      —             —
ReHDM                      —         XX            —
**MTL B9 (ours)**         68.59     63.34         52.86

=== California ===
... (same layout)

=== Texas ===
... (same layout)
```

Caption: *"External-baseline comparison at the headline scale. STL `next_gru` Check2HGI (italics) is the matched-head cat ceiling; STL `next_stan_flow` (italics) is the matched-head reg ceiling. The MTL B9 row (bold) reports the proposed model. Published POI-RGNN numbers (Capanema 2019) are reported under the original non-user-disjoint folds; our reproduction at user-disjoint folds is reported in the supplementary baseline audit, where the gap is wider."*

**Sub-agent A4 / A5 must verify the published POI-RGNN / MHA+PE numbers from primary sources before T5 commits.**

---

## 5 · Per-figure specifications

### F1 — Per-visit mechanism (§6.1, REQUIRED, 0.4 pp)

**Promoted to required after Codex audit** — explains *why* the substrate is task-asymmetric. Without it, the substrate-asymmetry headline has prose only and no visual mechanism anchor.

3-bar grouped chart at AL:
- **Group 1 (linear probe, head-free):** Check2HGI canonical (30.84), POI-pooled (23.20), HGI (18.70). Per-visit Δ = +7.64 pp; training-signal Δ = +4.50 pp.
- **Group 2 (matched-head STL `next_gru`):** Check2HGI canonical (40.76), POI-pooled (29.57), HGI (25.26). Per-visit Δ = +11.19 pp (~72 %); training-signal Δ = +4.31 pp (~28 %).
- **Annotations:** *"per-visit context"* arrow between canonical and POI-pooled; *"training signal"* arrow between POI-pooled and HGI; percent-of-gap labels.

Caption: *"Per-visit context is the dominant mechanism behind the substrate's cat advantage. At AL, decomposing the matched-head STL substrate gap (Check2HGI − HGI = +15.5 pp) into a per-visit-context component (canonical − POI-pooled = +11.2 pp, ~72 %) and a training-signal component (POI-pooled − HGI = +4.3 pp, ~28 %) shows that the bulk of the gap comes from per-visit variance — exactly the property POI-stable embeddings cannot supply, and exactly the property that pooling to region cardinality smooths away (explaining why the substrate ties on next-region)."*

### F2 — Scale-progression scatter (§5.2 or §7, OPTIONAL, 0.4 pp; cut first)

**Demoted to optional after Codex audit** — TX (−16.69 pp) breaks the scale-shrinks-with-data pattern, so F2 lost its title-anchoring role when "Scale-Sensitive" was demoted from the title. Keep only if pages allow; one descriptive sentence in §5.2 covers the same content.

If kept:
- **x-axis:** `n_checkins` (log scale) OR `n_regions` (linear). `n_checkins` (10K, 26K, 127K, 187K, 230K) reads cleaner.
- **y-axis:** `Δ_reg` pp, range [−20, 0], grid at every 2 pp.
- **5 dots, labelled** AL, AZ, FL, CA, TX. Headline (FL/CA/TX) = filled; anchors (AL/AZ) = open.
- **Dashed trend line** through AL→AZ→FL (the broadly downward trajectory). **TX annotated as non-monotone outlier** with arrow.
- **Caption:** *"Architectural reg cost (Δ_reg = MTL − STL Acc@10, pp) varies non-monotonically across the five U.S.-state Gowalla splits. The cost is broadly downward on the AL → AZ → FL trajectory (~5 pp recovery), CA preserves the regime, and TX breaks monotonicity (state-specific factors beyond raw class count). We report this descriptively rather than as an inferential scaling claim."*

### F-arch — Architecture schematic (§3.3, optional, 0.5 pp)

Standard methods-paper schematic — boxes for {check-in encoder, region encoder, cross-attn block, shared backbone, GRU cat head, STAN-Flow reg head}; arrows between; labelled hyperparameters (`d_model = 256`, 8 heads, 4 backbone blocks). Cut second if pages bind; prose in §3.3 covers the same material.

---

## 6 · Coverage state at submission (paper-side limitations, not workflow)

Three coverage items become paper-side limitations rather than workflow notes:

1. **CA/TX MTL multi-seed.** Seed = 42 single-seed at submission; multi-seed extension at {0, 1, 7, 100} is a **camera-ready audit item**. T3/T4 CA/TX cells therefore sit at the n = 5 paired-Wilcoxon ceiling (p_min = 0.0625 two-sided). Disclosed in §7 Limitations.
2. **AL/AZ/FL cat-Δ Wilcoxon.** v7 RESULTS_TABLE.md refreshed STL `next_gru` cat F1 to multi-seed means {0, 1, 7, 100}; Wilcoxon p-values for cat-Δ vs the v7 multi-seed STL ceiling have not yet been re-computed. T3 reports the Δ values and labels p as "pending re-run". This does not change the directional reading.
3. **POI-RGNN / MHA+PE absolute baseline numbers.** Working values: POI-RGNN FL ~31.8 / CA ~34.5 (cat F1); MHA+PE values from `baselines/next_category/results/<state>.json`. The POI-RGNN reproduction caveat (non-user-disjoint folds in the published evaluation) is disclosed in T5 caption and §7 Limitations.

---

## 7 · Sub-agent acceptance criteria for tables and figures

Before A5 / A6 commit a table:

1. Caption ends with a sentence describing what the reader should glance to read — no "Table 3 shows our results" tautologies.
2. Every numerical cell has σ unless the source is single-seed-single-fold.
3. Every Δ column has a direction symbol or a paired-test footer (`*` / `†` / `‡` / `n.s.`).
4. Bold = MTL; italics = STL; STL ceilings are reference rows, not "best" rows.
5. Booktabs only; no vertical rules; `\footnotesize` if > 6 cols.
6. Captions use the encoding convention from §4.3 — do not redefine symbols inline.

Before A6 commits F1 (required):

1. The "at Alabama" qualifier is in both caption and any in-text reference — single-state mechanism evidence.
2. Bars are labelled clearly (linear-probe vs matched-head; canonical / POI-pooled / HGI).
3. The per-visit-context arrow and training-signal arrow are unambiguous; percent-of-gap labels are above the bars.

Before A5 / A6 commits F2 (optional, only if pages allow):

1. The state labels are visible (no overlap with axis tick labels).
2. The TX outlier annotation is unambiguous (arrow + text label).
3. Caption explicitly says "non-monotone" for TX and "broadly downward" for the AL→AZ→FL trajectory — matches the prose framing in `STATISTICAL_AUDIT.md §4.3`. **Do not write "scales monotonically" or "shrinks with data" anywhere.**
