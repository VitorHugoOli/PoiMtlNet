# review — running log + cross-reference ledger

> **Purpose.** One place to record decisions that ripple, so a later edit can see what it touches. When you change
> something with a downstream effect, add a row to **Ripple ledger** and an entry to **Timeline**. Keep numbers'
> provenance (which JSON / which run) next to the number.

## Timeline

- **2026-06-30 (OP1 + Tier A + Tier B APPLIED + reviewed + clean build).** Applied the author-approved set: **OP1**
  paragraph added to §2.3 (our-evidence balancer/correlation finding, plain-language, tied to §6.2, board-backed
  ~0.001 = CH31 0.0007–0.0026; avoids the banned word "orthogonal"). **Tier A** acronyms: dropped the §1 LBSN
  re-expansion (abstract keeps it), §4 POI re-expansion, the orphan §5 "(CV)", the early §4 "Acc@10" token, the §5.3
  "(TOST)" re-expansion; **Tier A** gloss trims: redundant macro-F1 gloss (§6.1) + Acc@10 re-expansion (§6.2).
  **Tier B**: glossed "hard parameter sharing" (§2.3) + named "nearest-neighbor category purity" (§6.1); census tract
  left (already glossed). A **review agent** audited all of it: glossary-compliant + regression-free; one nit it
  surfaced (Acc@10 used at §5.2 before its §5.3 def — traced to the earlier S-5 edit) was FIXED ("region Acc@10" →
  "region accuracy"). Build clean: 9 pages, 0 undefined, 0 non-font warnings. **Deferred:** TOST letter-tie (ledger
  mandates "(TOST)" in §1; acceptable per review), the §7/§8 `%`-comment "beats" drift (cosmetic, → final cleanup),
  Tier C consolidations, and Phase 4 `WF-ENGLISH`. NOT yet committed (checkpoint commit was 4be0b35f, pre-this-round).
- **2026-06-30 (OP1 + Phase 2 WF-CONSISTENCY ran — discovery; edits PENDING author sign-off).** Workflow
  `op1-and-consistency-sweep` (16 agents: OP1 investigate+advise + 11 section readers + 3 reducers). **OP1: CAN
  claim, grounded in OUR experiments** — full balancer registry (PCGrad/GradNorm/Nash-MTL/CAGrad/FAMO/… ~18 methods)
  none beat tuned static cw=0.75; task-gradient cosine ≈ 0.001 across measured states/seeds (intrinsic, holds even
  in a fully-shared model); ties to §6.2 "stronger trunk, not transfer". Sources: `WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`,
  `T4_audit_and_verdict.md`, `orthogonality_intrinsic_test.md`, `T1-5_v2_cosine.json`, `W6_ENCODER_ISOLATION.md`.
  Advisor: agree, with wording fixes (don't re-expand PCGrad/GradNorm; drop the "four datasets" scope — measured set
  includes Georgia, excludes CA/TX). **OP2/OP3/OP4 reducers** returned the full-paper matrices. **No `src/` edits
  yet** — the edit list is presented for sign-off; Phase 4 (`WF-ENGLISH`) still to run.
- **2026-06-30 (CORRECTION + checkpoint commit).** Caught an overclaim in PLAN/README: they had marked the OP
  discovery sweeps "done via the §4–§8 audit." NOT true. The dedicated **WF-CONSISTENCY** (full-paper OP2/OP3/OP4
  matrices, incl. abstract + §1–§3) and **WF-ENGLISH** (OP5) were never run, and **OP1** was never pursued. The
  §4–§8 Germano-lens audit + the CC1–CC7 sweeps only *incidentally* touched those themes. The Germano per-comment
  edits + CC1–CC7 ARE applied and the paper builds clean (9 pages), but **OP1–OP5 still owe their proper separate
  phases** (`review_v2.md` correctly had them as TODO all along). PLAN/README/this log corrected; committed the
  work-so-far as a checkpoint. Next: run Phase 2 (`WF-CONSISTENCY`) + Phase 4 (`WF-ENGLISH`) + OP1 over the WHOLE
  paper, now that the prose has settled.
- **2026-06-28** — Re-ran the Germano review against `GLOSSARY.md` + `CLAUDE.md §3` ledger and folded in the author's
  inline considerations. Added the **Final decision pass (v2)** to [`../REVIEW_GERMANO.md`](../REVIEW_GERMANO.md)
  (cross-cutting CC1–CC5 + per-item rulings + changed-verdict index). Created `review/` (this folder):
  [`review_v2.md`](review_v2.md) (OP1–OP5 / INV1–INV6), [`PLAN.md`](PLAN.md) (phasing + workflow shapes), this log.
  **No `src/` edits yet** (per author instruction). Phase 0 done.
- **2026-06-28 (workflow `germano-author-notes-eval`)** — investigated each author note with an agent, an adversarial
  advisor, and two critics. Added the **Evidence-backed update** subsection to `REVIEW_GERMANO.md`. Material changes:
  **#67 +5% is FALSE** (INV2 done), **#66.1 loss justification is backwards** (INV3 mostly done), **#38 author
  incorrect** (code-backed), **CC3 downgraded FINAL→PENDING** (verb family + scope). INV1/INV2/INV4/INV5 effectively
  closed; INV3 leaves only the optional per-state category-distribution table. Still no `src/` edits.
- **2026-06-29 (Germano-lens audit §4–§8)** — ran the pitfall-taxonomy audit over Method→Conclusion (5 auditors + 5
  advisors). Cataloged ~35 verified findings in `REVIEW_GERMANO.md` ("Germano-lens proactive audit" section). New
  cross-chapter sweeps **CC6** (spatial gain) + **CC7** (region space) — both are accepted Germano fixes (#13/#11)
  that were only patched once. Findings route into OP3/OP4/OP5 + the prose pass.
- **2026-06-30 (Istanbul correction + Fig 3 + distribution + 7% audit)** —
  - **Istanbul: the article was WRONG, now fixed.** `istanbul_stride1_multiseed_summary.json` confirms windowing =
    "stride-1 overlap (min_seq=10)", n=20 (4 seeds × 5 folds), fp32 — the SAME protocol as the Gowalla board. The
    article's "the overlapping-window representation was not rebuilt for it / cross-setting check / earlier
    representation" was false. Fixed both mentions (§6.2, §6.3): Istanbul is now stated as the same overlapping-window,
    multi-seed protocol, reported as a lift (region counts differ), like-for-like within Istanbul. **R21-Istanbul
    dissolved** (abstract "result holds" is now justified). The only real residual (an internal substrate-build
    variant, GCN vs design_k) is invisible to the reader and neutralized by the within-Istanbul lift, so not exposed.
    ⚠ NOTE: internal docs (`[mobiwac]/CLAUDE.md §2`, `PAPER_PLAN §10.4`) still carry the old "cross-substrate / overlap
    substrate not built" framing — reconcile if desired (not done; user scoped this to the article).
  - **Fig 3 regenerated** (agent): `fig3_embquality.py` y-axis → "Score (0–1)", tick → "kNN purity (by category,
    k=10)", honest docstring; `fig3_embquality.tex` caption → "Class separability ... not on discovered clusters";
    PDF re-rendered. ⚠ Chrome extension was OFFLINE — audited via 200/600-dpi PNG + column-width sim instead (verdict:
    publication-quality, clustering-free). Literal Chrome screenshot still pending if the extension is reconnected.
  - **Per-state category distribution** (agent): wrote `docs/studies/closing_data/CATEGORY_DISTRIBUTION.md`; added a
    §5.2 sentence (stratified on next-category + the imbalance: Food majority everywhere, ~25% FL to 34% AL) — closes
    #66.1's "write about stratification". Source: `output/check2hgi/<state>/input/next.parquet` `next_category`.
  - **R15 "7%→6%": AUDITED, KEEP 7%.** Per-state macro-F1 floors AL 7.28 / AZ 7.25 / FL 5.66 / TX 6.76 / CA 7.04 /
    Ist 7.15 → Gowalla mean 6.80%, 5/6 round to ~7%. "about 7%" is correct; my earlier "6%" optional tweak was wrong
    and is withdrawn. No change.
  - **Build re-run clean: 9 pages, 0 undefined, 0 non-font warnings.** #57 (motivation citation) reclassified by the
    author as FUTURE WORK — dropped from the pending list.
- **2026-06-29 (PROSE PASS applied + clean build)** — applied the accepted Germano edits + audit fixes across the
  abstract, §1–§8, Table 1, Table 3, and the Fig.1/Fig.2 captions: the per-comment Accepts/Partials (#2,#9,#16,#18,
  #20,#21,#22,#23,#24,#26,#27,#28,#29,#30,#31,#34,#36,#37,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#49,#50,#51,#52,
  #53,#55,#56,#58,#59,#60,#61,#62,#63,#64,#65,#66,#68); the §4 loss rewrite (#66.1); #67 Option A; the cascade fix
  (R18); tbl1 density/next-POI (R19); and the cross-cutting sweeps **CC1** (em-dash), **CC4** (coarser→2 our-target
  uses), **CC5** (scope statement ×1, in §3), **CC6** (gain on region), **CC7** (number of regions, incl. abstract +
  Tbl 3). **Build: `pdflatex→bibtex→pdflatex×2` clean — 9 pages, 0 undefined refs/citations, 0 non-font warnings, no
  large overfull boxes.** Verified: 0 prose em-dashes, 0 verdict beats/wins, 0 "region space"/"spatial gain".
  **DEFERRED (author judgment / out of mechanical scope):** R21 (Istanbul-substrate + scaling-confound hedge in the
  abstract/conclusion — conflicts with the #14 "keep current" decision); #57 (motivation citation, author-owned);
  the Fig.3 figure regeneration (INV4 — relabel axis + "kNN purity" tick, needs running `fig3_embquality.py`); the
  optional per-state category-distribution table; the optional "about 7%"→"about 6%" precision tweak.
- **2026-06-29 (final advisor pass, 2 advisors)** — audited the whole review. **Corrections:** R15 (7% floor) is NOT
  an error — closed, no fix (the audit over-flagged; advisor computed ≈6.2%). **CC3 was incomplete** (figs/ never
  swept): fixed `fig4_deltas.tex:10` "win"→"gain" (R20, the project's 2nd `src/` edit); CC3 now truly 0-remaining.
  **New HIGH:** R18 cascade self-contradiction (`06:113-115`). **New:** R19 tbl1 "next-POI"; R14 extended to the
  abstract + Table 3 caption (the two highest-read CC7 misses). **New whole-paper risks:** R21 (Istanbul-substrate
  caveat + scaling confound dropped in the abstract/conclusion). Apply-safety notes added (treat CC3 as applied;
  don't paste #61/#64/#66 dashes; #50 no new bib). Verdict: review is safe to start applying.
- **2026-06-28 (CC3 applied)** — author confirmed "outperforms", global. **First `src/` edits of the project:** swept
  the superiority verb across the paper (abstract, §1, §5, §6, §7, §8, Tbl 3) + the law (`GLOSSARY.md`,
  `[mobiwac]/CLAUDE.md §3`). 0 verdict beats/wins remain. This is an isolated, self-contained normalization — the
  broader prose pass (other accepted edits, OP sweeps) is still NOT started. **Paper not yet recompiled** (no TeX run
  this session); a `pdflatex` build is the obvious next check before relying on the PDF.

## Cross-cutting decisions (the load-bearing ones)

- **CC1 — no em-dash.** Banned construction (`--` / `---`). LIVE violation to fix: `06_results.tex:57-58`. Also
  scrub it from every v1 edit when applied (#29, #61, #62, #64, #66). Owner: OP5 (+ Phase 3 as edits land).
- **CC2 — CTLE not in the intro.** Name/cite CTLE only in §2.1 + baseline list. Changes the #29 edit.
- **CC3 — superiority verb = "outperforms", global. ✅ APPLIED 2026-06-28 (author confirmed).** One verb
  "outperforms" (paired Wilcoxon superiority), "matches/non-inferior (TOST)" kept for equivalence. Whole family
  swept ({beats, beat, wins, win, "spatial win"}); `05_setup:68` "meant to win"→"meant to outperform"; Table III
  legend reconciled ("win"/"beat" → "improvement"/"outperforms"). Generic `02_related:73` "beat"→"improve on" (not
  a verdict). GLOSSARY + `[mobiwac]/CLAUDE.md §3` updated; repo-root `/CLAUDE.md` had no such rule. See R1.
- **CC4 — "coarser" once, glossed.** 8 occurrences → 1. Do NOT use "simpler" (feeds #48). See R2.
- **CC5 — "we do not predict the exact next place" once.** Keep the §3 instance; delete intro (#21) + related (#47).
  See R3.
- **CC6 — "spatial gain"/"spatial benefit" → "the gain on region" PAPER-WIDE** (Germano #13, accepted but only
  patched at the flagged line). Recurs: abstract `main.tex:70`, §6 `06:51`, §7 `07:17`, §8 `08:20`. See R13.
- **CC7 — "region space"/"space of regions" → "number of regions"/"many regions" PAPER-WIDE** (Germano #11, accepted
  but only patched once). Recurs: §1 `01:65`, §5 `05:18`, §6 `06:51/63/67`, §7 `07:14/16-17`, §8 `08:14`. See R14.
- **Process lesson:** every *accepted* Germano edit needs a paper-wide sweep, not a single-line patch.

## Ripple ledger (one change → what it touches)

> **STATUS BANNER (2026-06-30): R1–R20 are ALL APPLIED** in the 2026-06-29 prose pass + the 2026-06-30 follow-up
> (Istanbul / Fig 3 / distribution), and the paper builds clean. Exceptions: **R15** (7%-floor) is CLOSED with **no
> fix** (audited correct); **R21** Istanbul part is DONE, its scaling-confound-hedge part is **left as-is per the
> author**. The per-row "pending Phase 3 / pending OP4 / BLOCKED" labels below predate execution and are superseded
> by this banner and by the Timeline above (the Timeline is the authoritative as-built record).

| ID | Change | Touches / must update together | Status |
|---|---|---|---|
| R1 | CC3: superiority verb → "outperforms" (whole family {beats, beat, wins, win, "spatial win"}) | DONE: law (`GLOSSARY` honesty rule + §6 checklist; `[mobiwac]/CLAUDE.md §3` Verdict-verb row + §2 narrative) + prose (abstract anti-triple preserved; §1; `05_setup:68` test-binding sentence; §6 ×12 incl. the hidden `:138`; §7; §8; Tbl 3 caption "win"+"beat"→"improvement"/"outperforms"). `02_related:73` generic "beat"→"improve on". Repo-root `/CLAUDE.md` had NO verdict-verb rule (nothing to change there). "matches"/"TOST" untouched. Verified: 0 verdict beats/wins left. | **✅ DONE 2026-06-28 (author confirmed global "outperforms")** |
| R2 | CC4: collapse "coarser" to one glossed use | abstract:58, `01:23`, `02:3,40,51`, `03:16,31`, `05:73`; keep #48 rebuttal but reword it (don't re-add the word) | pending Phase 3 |
| R3 | CC5: single scope statement | delete `01_introduction.tex:26` (#21) + `02_related.tex:44` (#47); keep `03_problem.tex:16`; re-verify after OP4 | pending Phase 3 |
| R4 | #29 intro reword (drop CTLE) | §1 ¶4; ensure §2.1 still carries the CTLE comparison (it does); coordinate with OP2 (CTLE acronym count) | pending Phase 3 |
| R5 | #46 split the §2.1 novelty paragraph | §2.1; must keep #40 (`emit`→`produce`) and #45 (name DGI/HGI) copyedits; don't duplicate the intro (#29) or method (#61) versions | pending Phase 3 |
| R6 | #59+#60 radio-cell gloss + split | §3 (`03_problem.tex:29-34`); keep the tract-level motivation scope (PAPER_PLAN venue-bridge decision) | pending Phase 3 |
| R7 | #26+#64 actually define MTL | §1 ¶3 (`01:35`) gloss + §4.2 first-use gloss; trim the §4.2 line-65 repeat so the definition lands once; CC1 on the dashes | pending INV5/OP3 |
| R8 | #67 "+5%" is FALSE (INV2 done) | replace per Option A: drop the magnitude, keep "cheaper than two models"; change "price of **one**"→"well under the price of **two**" at `04:69-72`; fix header `04:5`, `PAPER_PLAN:393/745`, and the v1 #67 rationale (called +5% "a selling point"). Measured: vs ONE model +38..90% params / +22..74% FLOPs; vs TWO models −17..20% params / −28% FLOPs | **RESOLVED (apply in Phase 3)** |
| R9 | #66.1 loss justification (INV3 mostly done) | Acc@10 is frequency-weighted (`mtl_eval:57-61`); §4.2 reason is backwards for category but the unweighted *fact* is correct (board `--no-*-class-weights`); replace with the per-task+empirical version (C25, QUALITATIVE — not the board recipe so no pp); add stratification sentence (`05`). Optional: per-state category dist table = light groupby, unsourced | **RESOLVED qualitatively (per-state table optional)** |
| R10 | #50 "little precedent" (INV1 done) | coarse-cell-as-target IS precedented; keep the method contrast (auxiliary vs co-equal), soften to "to our knowledge … underexplored"; NO TrajLearn bib (locked 26); "underexplored" is outside `PAPER_PLAN §3` CAN-say → add with hedge or drop | **RESOLVED (apply + §3 decision)** |
| R11 | #69 Fig 3 (INV4 done: cosmetic) | no clustering bug; relabel y-axis + "kNN purity (k=10)" + caption "no clustering" + **regenerate PDF**; HOLD the numeric caption lift inside the regen (numbers must match the .py) | **RESOLVED (figure regen in Phase 3)** |
| R12 | #31 "trunk" wording | settle one phrasing; then OP4 verifies intro/§4/§6/§7 agree | pending OP4 |
| R13 | CC6: "spatial gain"→"the gain on region" paper-wide | `main.tex:70`, `06:51`, `07:17`, `08:20` (+ §8 comment `08:5`) | pending Phase 3 |
| R14 | CC7: "region space"→"number of regions" paper-wide | **abstract `main.tex:69`** (high-read), **tbl3 caption `tbl3_results.tex:21`** (high-read), `01:65`, `05:18`, `06:51/63/67`, `07:14/16-17`, `08:14` (+ §8 comment). Do abstract+tbl3 in the SAME pass, never half. | pending Phase 3 |
| R15 | §6 `06:53` "7% floor" | **NOT an error (final advisor): majority Food≈27.8% → macro-F1≈6.2%≈"about 7%"; §5.3 def agrees.** Drop the proposed "one in seven"(14%) fix — it injects an accuracy rate as a macro-F1 reference. Optional: "about 7%"→"about 6%". | **CLOSED — no fix** |
| R18 | §6 `06:113-115` cascade self-contradiction (HIGH, claim discipline) | ties the cascade ("same combined accuracy ... about zero") then "does not match our gains" (a win-claim, breaks "never claim a win over the cascade"). Fix: drop "does not match our gains"; keep the tie framing. | pending Phase 3 |
| R19 | `tbl1_datasets.tex:21` "next-POI" (GLOSSARY-banned) + density-vs-sparsity | → "standard sparsity measure for next-place recommendation"; fold into S-8 | pending Phase 3 |
| R20 | fig4 `fig4_deltas.tex:10` "largest win" (CC3 miss — figs/ not swept) | → "largest gain" | **✅ DONE 2026-06-29 (CC3 completion)** |
| R21 | whole-paper: Istanbul substrate caveat dropped in abstract/conclusion; scaling-confound hedge dropped outside §6.2 | add a half-clause caveat ("is consistent"/note the different representation) + carry the n=6 confound hedge into one of abstract/§1/§7/§8 | pending Phase 3 (claim discipline) |
| R16 | CC3 residue `07:18` "meets or outperforms" → "matches or outperforms" | one-word verb-discipline fix | pending Phase 3 |
| R17 | §6 `06:40` CTLE fairness: add "same windowing + same single-task head" (verified true) | matched-protocol clause | pending Phase 3 |

## Facts (sourced 2026-06-28 by the workflow)

- **+5% params/compute** (#67) — ✅ MEASURED, and it is FALSE. vs ONE dedicated model: +38..90% params / +22..74%
  FLOPs; vs TWO models: −17..20% params / −28% FLOPs. Use Option A wording (R8).
- **Acc@10 weighting / stratification / class-weighting↔macro-F1** (#66.1) — ✅ SOURCED. Acc@10 frequency-weighted
  (`mtl_eval:57-61`); folds StratifiedGroupKFold (user-grouped, category-stratified, `folds.py`); class-weighting
  tested + hurt both tasks (C25, `docs/CONCERNS.md:590`) — keep qualitative on the board recipe.
- **census-tract region precedent** (#50) — ✅ CHECKED (web). Coarse-cell-as-target precedented (TrajLearn etc.);
  owned combo underexplored. Method-contrast framing; no bib add.

## Still open / author-owned

- **per-state 7-category distribution table** (#66.1, optional) — needs a light groupby on the category column; not
  in docs today. Only if the paper wants a printed distribution.
- **orthogonal-transfer / why-no-balancer claim** (OP1) — `docs/` regime finding + W6 probe + the C25 result;
  plain-words + honest (no "Pareto", no "orthogonal gradients" repo jargon). Not yet investigated.
- **mobility-aware-service citation** (#57) — author-owned (INV6); attach to concrete examples only, `%verified`.

## Already-sourced (do NOT re-open as "needs a number")

- §5.2 leak-Δ numbers (AL −0.33 / AZ +0.01 / FL −0.12 region; AL +0.29 / AZ +0.27 / FL +0.00 category) — sourced
  from the A4 audit (`docs/studies/pre_freeze_gates/A4_RESULTS.md`); already in `05_setup.tex:46-48` with a code note.
- "region-transition prior inflated region by 13–27 pp" — sourced (`AGENT_CONTEXT.md`); in `05_setup.tex:54`.
- All Tbl 2 / Tbl 3 cells — board JSONs via `RESULTS_BOARD.md §3`; the table comment blocks carry the values.
