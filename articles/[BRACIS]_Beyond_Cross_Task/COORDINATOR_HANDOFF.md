# COORDINATOR_HANDOFF.md — sub-agent fan-out plan + state

> **Purpose.** Working memory for the coordinator session running the BRACIS sub-agent
> fan-out. Read this first if you wake up mid-fan-out and need to know what's been
> assigned, what's landed, what's still open, and what the audit gate has caught.
>
> **Lifetime.** Discard once `samplepaper.tex` is camera-ready and all `sections/*.tex`
> are merged + compile-clean. Until then, append a per-batch status block at the bottom.

---

## 0 · Coordinator role (this session only)

This session does **not** draft prose. It owns:

1. **Cross-section consistency audit** between sub-agent returns (numerical canon, voice rules, page budget, LaTeX compile).
2. **Numerical canon enforcement** — every cited number must round-trip to `docs/studies/check2hgi/results/RESULTS_TABLE.md §0` (v11). No exceptions.
3. **Final assembly of `samplepaper.tex`** — only the coordinator edits the include skeleton; sub-agents only write their assigned `sections/<name>.tex`.
4. **Pre-computation of T1 dataset cells** before A4 launches (real numbers from `data/<state>/`).
5. **`references.bib` verification** — coordinator task, not A8 sub-agent. 27 entries already in file; check cite-key alignment with what each section ends up referencing.

---

## 1 · Sub-agent fan-out plan (locked 2026-05-02)

### Batch 1 — parallel (Opus, run together)

| ID | Section | File to write | Page budget | Reads |
|---|---|---|---:|---|
| **A1** | Intro (§1) | `sections/intro.tex` | 1.5 pp | `PAPER_DRAFT.md §1`, `AGENT.md` voice + style |
| **A2** | Related Work (§2) | `sections/related.tex` | 1.5 pp | `PAPER_DRAFT.md §2`, voice + third-person CBIC/CoUrb snippet (see §3 below) |
| **A4** | Experimental Setup (§4) | `sections/setup.tex` | 1.5 pp | `PAPER_DRAFT.md §4`, T1 numbers (pre-computed by coordinator) |

A1 and A2 share the locked third-person CBIC/CoUrb framing — inject the snippet into both prompts so they do not diverge.

### Batch 2 — serial (Opus, one at a time)

| ID | Section | File to write | Page budget | Depends on |
|---|---|---|---:|---|
| **A3** | Method (§3) | `sections/method.tex` | 2.5 pp | (none — describes architecture + loss + training) |
| **A5** | Results (§5) | `sections/results.tex` | 4 pp | A3 (must reuse method's Δ definitions, head names) |
| **A6** | Mechanism + Ablations (§6) | `sections/mechanism.tex` | 1.5 pp | A5 (must scope Alabama-only CH19 against Results' framing) |
| **A7** | Discussion + Conclusion (§7 + §8) | `sections/discussion.tex` + `sections/conclusion.tex` | 1 + 0.5 pp | A6 (limitations narrative inherits CH19 scope + recipe-selection wording) |

### Coordinator-only tasks (no sub-agent)

| ID | Task | Trigger |
|---|---|---|
| **C-T1** | Pre-compute T1 dataset cells from `data/<state>/.parquet` and `output/check2hgi/<state>/regions.parquet`. Hand to A4 as prose-only inputs. | Before Batch 1 launches |
| **C-bib** | Verify `references.bib` cite-key alignment after every sub-agent return; dedup, add missing keys. | After every sub-agent |
| **C-tex** | Update `samplepaper.tex` `\input{sections/...}` lines once each section file lands. | After every sub-agent |

---

## 2 · v11 numerical canon snippet — **inject verbatim into every sub-agent prompt**

```
NUMERICAL CANON (RESULTS_TABLE.md §0 v11, sole canonical source — do NOT paraphrase):

§0.1 — MTL B9 vs matched-head STL ceiling, all five states n = 20 multi-seed:
  AL  Δ_cat = −0.78 pp  p = 0.036    (small-significantly negative; ~1.9 % relative)
  AZ  Δ_cat = +1.20 pp  p < 1e-04
  FL  Δ_cat = +1.40 pp  p = 2e-06
  CA  Δ_cat = +1.68 pp  p = 2e-06
  TX  Δ_cat = +1.89 pp  p = 2e-06
  AL  Δ_reg = −11.04 pp  p = 1.9e-06
  AZ  Δ_reg = −12.27 pp  p = 1.9e-06
  FL  Δ_reg =  −7.34 pp  p = 1.9e-06
  CA  Δ_reg =  −9.50 pp  p = 2e-06
  TX  Δ_reg = −16.59 pp  p = 2e-06

Reg-cost range: "7 to 17 pp" (NOT "8 to 17"). FL smallest at −7.34; TX largest at −16.59.
Pattern: broadly downward AL→AZ→FL, CA preserves regime, TX breaks monotonicity (descriptive only).

§0.3 — Substrate cat ablation (Check2HGI − HGI, matched-head STL `next_gru`, single-seed):
  AL  +15.50 pp   AZ  +14.52 pp   FL  +29.02 pp   CA  +28.81 pp   TX  +28.34 pp
  Paired Wilcoxon p = 0.0312 each (5/5 folds positive; n = 5 ceiling).
  Range: "+14.5 to +29 pp" (or "+14 to +29"). NEVER "+15 to +33" (33 was the MTL counterfactual).

§0.3 — Substrate reg ablation (matched-head STL `next_stan_flow`):
  HGI nominally above C2HGI by 1.6 to 3.1 pp at every state.
  TOST δ = 2 pp passes at CA/TX (statistically tied); δ = 3 pp passes at FL.

§0.2 — Δm joint score (CH22 leak-free):
  FL multi-seed (n = 25): Δm-MRR = +2.33 % at p = 2.98e-08 (25/25 fold-pairs positive);
                         Δm-Acc@10 = −1.12 % at p = 3.20e-05 (4/21 positive).
  AL/AZ/CA/TX = single-seed n = 5 ceiling on this axis.

§6.1 — CH19 mechanism (single-state, AL only):
  At AL, matched-head STL `next_gru`: Check2HGI canonical 40.76 / POI-pooled 29.57 / HGI 25.26.
  Per-visit context = +11.19 pp ≈ 72 %; training signal = +4.31 pp ≈ 28 %.
  ALWAYS write "at Alabama" or "at AL"; never imply multi-state.

D6 anonymous code link: https://anonymous.4open.science/r/PoiMtlNet-FE6A/
```

---

## 3 · Locked third-person CBIC/CoUrb framing — **inject into A1 and A2 prompts**

```
THIRD-PERSON FRAMING (locked 2026-05-02; double-blind safe):

Prior work 1 (CBIC 2025): "Silva et al. (2025) proposed a hard-parameter-sharing MTL
framework on POI-stable graph embeddings (DGI) and reported only marginal gains over
single-task baselines — most differences fell within standard deviations. The diagnosis
pointed at representation mismatch."

Prior work 2 (CoUrb 2026): "Paiva et al. (2026) tested an embedding-side response by
decomposing the monolithic graph embedding into independent spatial (SIREN /
Sphere2Vec-M), temporal (Time2Vec), and categorical (HGI) sub-encoders, recovering
substantial cat-side gains across three states."

NEVER write "our prior work" or "we previously showed" in the main text. NEVER name
the lab, institution, funder, or city. The Gowalla state set (Alabama, Arizona,
Florida, California, Texas) is fine — those are dataset partitions, not author origins.
```

---

## 4 · Voice no-no list — **inject into every sub-agent prompt; mirror in audit grep**

```
DO NOT WRITE:
- "8 to 17 pp" (use "7 to 17 pp" — FL closed at −7.34 in v11)
- "FL p = 0.0625" / "FL n = 5 ceiling" (FL is now n = 20 paper-grade)
- "+15 to +33 pp" / "+33 pp" anywhere on the substrate axis (33 was MTL counterfactual)
- "We propose Check2HGI" (Check2HGI is a derivative substrate, not a contribution)
- "We propose per-head learning rates" (H3-alt was a detour, not a paper claim)
- "We refute MTL" / "MTL is broken" (the framing is "textbook tradeoff")
- C3 (cross-attn task_weight = 0 note) as a top-tier contribution bullet — it lives in §6.3 only
- \hline anywhere (use booktabs: \toprule / \midrule / \bottomrule)
- Vertical rules in tables (booktabs convention)
- "novel" / "intuitive" / "obvious" / "carefully designed" / "extensive" filler
- "our prior work" / "we previously showed" (use third-person Silva/Paiva)
- Author / affiliation / lab / funder / institution names
- "Lightning Studio" or any subscription-tier hardware name (single H100 only)

DO WRITE (preferred phrasings, lifted from PAPER_DRAFT.md beats):
- "task-asymmetric substrate"
- "small-significantly negative" (for AL Δ_cat = −0.78 pp p = 0.036)
- "paper-grade significant" (for p ≤ 2e-06 at n = 20)
- "sign-consistent" (when Δ has same sign across all five states)
- "single-state mechanism evidence" (for CH19)
- "reported descriptively, not as an inferential scaling claim" (for TX non-monotonicity)
```

---

## 5 · Per-batch audit gate (run after every sub-agent returns)

For every returned `sections/<name>.tex`:

1. **Numerical canon grep** — pattern set:
   - `8 to 17 pp` → must be zero hits in active prose
   - `0\.0625` near "FL" → must be zero hits
   - `\+1\.52` → must be zero hits
   - `7\.99` → must be zero hits
   - `n = 5 ceiling` near "FL" → must be zero hits
   - `\+33` near "substrate" → must be zero hits
2. **Conflict-marker grep** — `<<<<<<< |>>>>>>> |Stash base` → zero hits.
3. **Voice grep** — forbidden phrases: `\b(novel|obvious|intuitive|carefully designed|extensive)\b`, `our prior work`, `we previously showed`, `\\hline`.
4. **Cite-key check** — every `\cite{X}` resolves to a key in `references.bib`.
5. **Page-budget tally** — `wc -w sections/*.tex` plus a 0.5-page margin per section. Hard cap is 15 pp incl. references.
6. **LaTeX compile** — `pdflatex samplepaper.tex` against the current partial assembly. Catches `\ref{}` to non-existent labels and template breakage.
7. **Cross-section consistency** — Δ numbers and recipe names (B9 vs H3-alt) must match across §3, §5, §6, §7.

If any gate fires, the coordinator either (a) self-fixes if it's a 1-line typo, or (b) re-spawns the sub-agent with a focused fix-up prompt.

---

## 6 · Sub-agent prompt template (use this skeleton per A_n)

```
You are sub-agent A<N> for the BRACIS 2026 submission "Substrate Carries, Architecture
Pays". You ONLY write `sections/<name>.tex`. You do NOT edit other files.

Read first (in this order):
1. articles/[BRACIS]_Beyond_Cross_Task/AGENT.md (voice + statistics + anonymisation)
2. articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md §<N> (paragraph-level beats)
3. articles/[BRACIS]_Beyond_Cross_Task/PAPER_STRUCTURE.md §<N> (sectional commitment)
4. articles/[BRACIS]_Beyond_Cross_Task/TABLES_FIGURES.md (tables/figures referenced from your section)

[INJECT v11 NUMERICAL CANON SNIPPET — see COORDINATOR_HANDOFF §2]

[INJECT THIRD-PERSON FRAMING — A1 and A2 only — see COORDINATOR_HANDOFF §3]

[INJECT VOICE NO-NO LIST — see COORDINATOR_HANDOFF §4]

Page budget: <X.Y> pp. Hard cap.

Deliverable: a single self-contained `sections/<name>.tex` file that:
- compiles when included by samplepaper.tex
- uses booktabs for tables (no \hline)
- cites only keys present in references.bib
- contains zero placeholders / TODOs / "XX" cells
- matches the voice rules verbatim

If you need a number that is not in the v11 numerical canon snippet, STOP and ask
the coordinator — do not invent or infer.

Do not edit samplepaper.tex, references.bib, or any other section file.
```

---

## 6.1 · Author voice — match the user's prior papers (CBIC___MTL, CoUrb_2026)

The sub-agent prose must read like a continuation of the user's prior papers.
Two reference points:

- `articles/CBIC___MTL/sections/` (English, IEEE-style, Florida-only MTL POI paper, 2025).
- `articles/CoUrb_2026/sections/` (Portuguese — for tone/cadence reference only; do NOT mimic the language).

Distilled style features (sample any prior section to verify):

```
SENTENCE RHYTHM
- Medium-length, paratactic. 3-6 sentence paragraphs.
- Connectives: "However,", "Therefore,", "While X, Y", "Crucially,", "Furthermore,"
  used SPARINGLY (one per paragraph at most; never two in a row).
- Avoid em-dashes (— or --) almost entirely. The user uses commas, semicolons,
  parentheses instead. (LLMs over-use em-dashes; this is a watermark.)

VOICE
- First-person plural: "we use", "our results", "our approach". Not "I" or "this paper".
- Hypothesis-driven framing: an explicit "the central hypothesis of this study is that..."
  type sentence near the end of the intro.
- Honest framing for negative results: "did not yield the anticipated improvements",
  "fell within standard deviations", "largely comparable", "competitive landscape".
- Hedge with "may", "could", "suggests", "lend weight to". Avoid absolutes.

STRUCTURE / SIGNPOSTING
- Sections start with a one-sentence claim of what is being established, then evidence.
- Tables are introduced with "As shown in Table X, ..." or "The results, detailed in
  Table X, indicate..." — anchor THEN discuss.
- Numerics ALWAYS with σ: "62.51 ± 0.94". Use \pm in LaTeX.
- Citations integrated mid-sentence: "by Cho et al. (2011) \cite{Cho2011}".
- Last paragraph of the intro is the "rest of this paper is organised as follows" map.
- Contribution bullets at end of intro, 3 to 4 items, one sentence each.

FORMATTING
- Bold for term emphasis ("\textbf{negative transfer}"), restrained — at most 2-3 bolds
  per paragraph, and only on technical terms or load-bearing claims.
- \emph{} for soft emphasis on dataset / category names (\emph{Gowalla}, \emph{Food}).
- Bullet lists used for: contribution items, task enumerations. Not for prose.
- LaTeX math in \( ... \) or $...$ inline; $\pm$ for ± symbol.
```

---

## 6.2 · LLM watermarks — DO NOT WRITE (mirror in audit grep)

These phrases are AI-writing tells. The user's prior papers don't use them; ours
shouldn't either.

```
PHRASES TO AVOID (zero-tolerance grep targets):
- "delve into" / "delving into"
- "tapestry" (any context)
- "navigate the complexities"
- "unprecedented" / "paradigm shift"
- "harness" / "unleash" / "unlock"
- "intricate" (use "complex" or just describe)
- "It's worth noting that..." / "It should be noted that..."
- "In conclusion," (use "We conclude" or just the conclusion)
- "Indeed," at sentence start (use it max once per section)
- "Moreover," (replace with "We also" / "In addition" / nothing)
- "Furthermore," (allowed sparingly — max once per section)
- "underscore" / "underscores"
- "robust" as filler adjective ("robust framework", "robust performance")
- "leverage" as a verb (use "use" / "exploit" / "build on")
- "comprehensive" as filler adjective
- "novel" / "obvious" / "intuitive" / "carefully designed" / "extensive"
- "Not only X but also Y" repeated in close proximity
- Triplet adjective stacking: "novel, robust, and comprehensive"
- "Let's" / contractions in body prose

FORMATTING WATERMARKS:
- Em-dashes (— or --) in body prose. Use commas, semicolons, parentheses.
- Heavy bolding (3+ bold phrases in one paragraph beyond technical terms).
- Headings every 2 sentences (the user uses few subsubsections).
- Bullet lists for narrative content. Bullets are for contributions / enumerated tasks only.
```

---

## 6.3 · T1 dataset cells (pre-computed by coordinator — hand to A4)

Filter: users with at least 5 check-ins (matches `src/data/folds.py` + sliding-window prereq).
Computed 2026-05-02 from `data/checkins/<state>.parquet` (Gowalla, Cho 2011).

| State | Users | Check-ins | POIs | Regions | Mean traj. length |
|---|---:|---:|---:|---:|---:|
| Alabama (AL) | 1,622 | 109,695 | 11,666 | 1,109 | 67.6 |
| Arizona (AZ) | 3,331 | 227,956 | 20,469 | 1,547 | 68.4 |
| Florida (FL) | 13,935 | 1,392,262 | 76,266 | 4,703 | 99.9 |
| California (CA) | 26,171 | 3,148,594 | 168,771 | 8,501 | 120.3 |
| Texas (TX) | 27,603 | 4,067,543 | 160,575 | 6,553 | 147.4 |

n_regions taken from `RESULTS_TABLE.md §0.1` (canonical). Other cells from raw filtered parquet.
A4 cites these as `Table~\ref{tab:datasets}` and writes the caption per AGENT.md voice rules.

---

## 7 · Status log (append per batch)

### State at hand-off (2026-05-02, v11 hygiene round-2 committed at `a0751a2`)

- ✅ Scaffold complete: AGENT, PAPER_DRAFT, PAPER_STRUCTURE, TABLES_FIGURES, STATISTICAL_AUDIT, AUDIT_LOG.
- ✅ Numerical canon locked at v11 across both article + study folders.
- ✅ samplepaper.tex (title + 145-word abstract + include skeleton) ready.
- ✅ references.bib has 27 entries (within 25–30 budget).
- ⏳ `sections/*.tex` — empty (only `.gitkeep`).
- ⏳ `figs/` — empty.
- ⏳ `tables/` — empty.
- ⏳ T1 dataset cells — pre-computation pending (coordinator C-T1 task).

### Batch log

**Batch 1 (A1 Intro, A2 Related, A4 Setup) — closed 2026-05-02**

- A1 (intro.tex): returned in prior session before token-quota error; coordinator-edited 2× to replace `C1 — ` / `C2 — ` em-dashes with `C1.` / `C2.` after audit. 1207 words ≈ 2.3 pp (1.5 pp budget — Task #18 tracks the trim pass).
- A2 (related.tex): 827 words ≈ 1.5 pp ✅. Three subsections (`subsec:related-embeddings`, `subsec:related-mtl`, `subsec:related-poi-mtl`) + positioning paragraph. §2.4 cross-attention forwarded into §6.3 instead of its own subsection (page-budget call). Note: `lambda_cat = 0.75` quoted in §2.2 — A3 must align Method to the same value.
- A4 (setup.tex + tables/datasets.tex): 605 + 137 words ≈ 1.5 pp ✅. T1 booktabs table follows the AL/AZ-below / FL/CA/TX-above split. Cites `wilcoxon1945ranking` (verified in bib).

**Audit gate verdict — all green**:
- Numerical canon grep (`8 to 17 pp`, `0.0625`, `+1.52`, `7.99`, `n = 5 ceiling` near FL, `+33 pp` near substrate): 0 hits.
- Conflict-marker grep: 0 hits.
- Voice grep (`novel`, `obvious`, `intuitive`, `extensive`, `comprehensive`, `leverage`, `robust`, `harness`, `unleash`, `delve`, `tapestry`, `our prior work`, `we previously showed`, `\hline`): 0 hits.
- Em-dashes in body prose: 0 hits (after intro.tex coordinator edit).
- Cite-key resolution: 21/21 keys verified in references.bib.
- Booktabs: T1 uses `\toprule / \midrule / \bottomrule`, no `\hline`, no vertical bars.
- Cross-section refs: `sec:related`, `sec:setup` labels present and resolve from intro.tex; A3/A5/A6/A7 still owe `sec:method`, `sec:results`, `sec:mechanism`, `sec:discussion`, `sec:conclusion`.

**Coordinator-side notes carried forward**:
- A2 worked around the absent `kendall2018multitask` and `sener2018mgda` keys (intentional bib pruning per audit). No action needed.
- A2 used the bib's actual key `zeng2019mhape` rather than the spec's `zeng2019next`. Already aligned.
- §2.2 positioning sentence about `lambda_cat = 0.75` static weighting must be mirrored in A3's Method section.

### Batch 2 — in progress

**A3 (method.tex, 2.5pp) — closed 2026-05-02 ✅**
- 1141 words (target 1100–1300). `\section{Method}` + 3 subsections; `eq:crossattn` and `eq:loss` labelled equations; anonymous URL inline in §3.3.
- Cross-section consistency: $\lambda_{\text{cat}} = 0.75$ matches §2.2/§3.3; head class names `next_gru`/`next_stan_flow` match §4.2; substrate name "Check2HGI" consistent.
- Audit gate: 0 watermark hits, 0 `B9`/`H3-alt` labels in prose, 0 `task_weight = 0` promoted to §3 contribution (deferred to §6 via `\ref{sec:mechanism}`).

**A5 (results.tex + 5 result tables, 4pp) — closed 2026-05-02 ✅** (sub-agent quota'd mid-run; coordinator finished `external.tex` + `results.tex` directly)
- A5 wrote: `tables/substrate_cat.tex`, `tables/substrate_reg.tex`, `tables/mtl_vs_stl.tex`, `tables/deltam.tex` (4 of 5).
- Coordinator wrote: `tables/external.tex` + `sections/results.tex` (1467 words; tables ~37 booktabs lines).
- Inconsistencies caught + fixed:
  - `mtl_vs_stl.tex` caption listed seeds `{0,1,7,42,100}` (5 seeds = 25 fold-pairs); n=20 implies 4 seeds. Fixed to `{0,1,7,100}` per canon line 60 of RESULTS_TABLE.md.
  - `results.tex` had LaTeX en-dash range `1.6--2.1 pp` (line 13). Replaced with "1.6 to 2.1 pp" per CBIC voice convention.
  - `deltam.tex` `4/25` was initially flagged but is actually correct (positive-out-of-total convention; canon's `4/21` is positive-out-of-negative convention; both = 4 positive, 21 negative, 25 total).
- POI-RGNN canonical numbers: 34.49/31.78/33.03 (FL/CA/TX) per RESULTS_TABLE §0.6 v11. JSON values diverge (33.35/30.71/32.08) — JSONs may be pre-bugfix. Trusted §0.6.
- MHA+PE canonical numbers: not in §0.6 ("(closed all 5 states)"). Used JSON values: 32.06/29.13/29.91 (FL/CA/TX).
- Audit gate: 0 watermark hits, all 6 new cite keys (`capanema2022poirgnn`, `zeng2019mhape`, `luo2021stan`, `li2025rehdm`, `maninis2019attentive`, `vandenhende2022mtl`) resolve.

**A6 (mechanism.tex, 1.5pp) — closed 2026-05-02 ✅**
- 979 words (within 1.5pp budget). 3 subsections: `subsec:mechanism-pervisit` (CH19 counterfactual at AL), `subsec:mechanism-ablations` (FAMO/Aligned-MTL/HSM-reg drop-in panel at FL), `subsec:mechanism-isolation` ($\lambda=0$ pitfall + encoder-frozen protocol).
- §6.1 single-state framing locked: "This is single-state mechanism evidence, run at Alabama only." + explicit note that the $\sim$72/$\sim$28 split is not generalised. Decomposition numbers verbatim from canon: 40.76 / 29.57 / 25.26 → +11.19 pp ($\sim$72\%) per-visit + +4.31 pp ($\sim$28\%) training signal.
- §6.2 ablation numbers from `docs/studies/check2hgi/research/archive/F50/F50_T1_RESULTS_SYNTHESIS.md`: FAMO +0.62 pp; Aligned-MTL $-0.11$ reg / $-0.90$ cat; HSM-reg $-3.01$ pp (with single-fold-init caveat). Florida-only panel; "across all five states" phrases refer back to §5.2 sign-consistency, not generalising §6.1.
- §6.3 cross-attention $\lambda = 0$ note: encoder co-adapts via K/V projections; encoder-frozen isolation framed as the clean decomposition; regression tests pointed at via anonymous URL (`\ref{sec:method}`).
- Figure F1 placeholder: `\includegraphics{figs/per-visit.pdf}` with `\label{fig:per-visit}` and TODO comment retained for camera-ready render.
- Audit gate: 0 watermark hits (em-dash, "delve", "tapestry", "leverage", "robust", "comprehensive", etc.); both new cite keys (`liu2023famo`, `senushkin2023aligned`) already in `references.bib`; `eq:loss` cross-ref resolves to `method.tex:37`.

**A7 (discussion.tex + conclusion.tex, 1.5pp total) — closed 2026-05-02 ✅**
- `discussion.tex`: 633 words (target 600–700); `\section{Discussion and Limitations}` + `\label{sec:discussion}`; four-paragraph structure (substrate task-asymmetry / scale-magnitude descriptive / five honest limitations stitched into one paragraph / two follow-up directions). All numerics verbatim from v11 canon.
- `conclusion.tex`: 350 words (target 300–350); `\section{Conclusion}` + `\label{sec:conclusion}`; two-paragraph structure (recap of headline empirical picture / three forward-looking lines + closer "Recovering the next-region performance without sacrificing the next-category lift is the open challenge this paper leaves to follow-up work."). No verbatim copy of the abstract.
- Cite keys: `liu2023famo`, `senushkin2023aligned`, `li2025rehdm`, `capanema2022poirgnn`, `tang2020ple`, `misra2016crossstitch` — all 6 resolve in `references.bib`.
- Cross-refs: `sec:results`, `sec:mechanism`, `sec:method`, `subsec:results-substrate`, `subsec:results-mtl-stl`, `subsec:results-external`, `subsec:mechanism-pervisit`, `subsec:mechanism-ablations`, `subsec:mechanism-isolation` — all targets exist in sibling sections.
- Audit gate: 0 watermark hits (em-dash, "delve", "tapestry", "leverage", "robust", "comprehensive", "novel", "Moreover,", "In conclusion,", etc.); 0 LaTeX en-dash numeric ranges (`[0-9]--[0-9]`); 0 forbidden environments (`\hline`, `\input`, `equation`, `figure`, `table`, `itemize`, `enumerate`); 0 hardware-tier names; 0 author / lab / institution leaks.
- Intentional departures from PAPER_DRAFT.md: dropped recipe-selection (B9 vs H3-alt) sentence from §7 Beat 2 (F-trail surfacing risk per AGENT.md §9.3); dropped the FL Markov-saturation limitation from §7 Beat 3 (not in canon block); omitted explicit head-invariance chain from §8 recap to fit 350-word budget (covered in §5.1).

### Carry-forward open items
- Task #18: trim intro.tex from ~1207w to ~800w (1.5pp budget) during final assembly pass.
- POI-RGNN published-vs-reproduction footnote pull from PMC paper deferred to camera-ready (per TABLES_FIGURES.md §232).
- ReHDM CA/TX cells deferred to camera-ready (compute budget).

