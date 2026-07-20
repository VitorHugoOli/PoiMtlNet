VERDICT: **minor pass needed.** The paper is structurally sound, the narrative arc (Part 1 representation → Part 2 one model, two tasks) carries, the register is clean (zero hits on the GLOSSARY §7 banned-word/template lists, zero em-dashes, idiom sweep clean), and Table III renders legibly with markers matching the honesty rules. One factual self-contradiction (Fig. 3 caption vs. Table III/§6.2) and one GLOSSARY-rule breach (§6.1 lead sentence) must be fixed before submission; the rest is polish.

---

## Ranked edits

**1. [BLOCKER] Fig. 3 caption contradicts Table III and §6.2 on seeds.** `figs/fig4_deltas.tex:7` still carries the pre-A1 disclosure, but Table III's footnote says "All joint and dedicated entries average four seeds over five folds" and §6.2 says "All six datasets are measured at four seeds for both models." Since the CA/TX n=20 top-up landed (2026-07-11, per the working-folder ledger), the caption under-reports and self-contradicts.
- Current: "the region gain (Acc@10) rises across the five U.S. states (TX and CA: one seed) and is also positive at Istanbul."
- Replace: "the region gain (Acc@10) rises across the five U.S. states and is also positive at Istanbul."

**2. [GLOSSARY §4 breach] §6.1 opens with a topic sentence, not a takeaway.** `sections/06_results.tex:15-17`. Rule: "every table and results subsection opens with a lead takeaway sentence." §6.2 and §6.3 comply; §6.1 does not (the takeaway arrives four sentences in).
- Current: "Table~\ref{tab:substrate} compares our check-in-level representation against a standard place embedding (HGI~\cite{huang2023hgi}) on next-category macro-F1."
- Replace: "The check-in-level representation outperforms a standard place embedding (HGI~\cite{huang2023hgi}) on next-category macro-F1 by a wide gap at every dataset (Table~\ref{tab:substrate})." Then continue with the controlled-comparison sentences, dropping the now-redundant "The check-in-level representation outperforms the place embedding by a wide gap at every state:" down to just the number list.

**3. Table III floats to page 8, after the conclusion, inside the references.** `main.tex:124`. The paper's main result table is two pages past the prose that reads it (§6.2, pages 6-7). Move `\input{tables/tbl3_results}` earlier (e.g., immediately after `\input{sections/05_setup}` or before `06_results`) so the `table*[!t]` lands on page 7.

**4. Elliptical selector definition in the §6.2 convention sentence.** `sections/06_results.tex:46`. "the joint model at the epoch its joint validation score selects" never says what the joint score is, yet the cascade paragraph later uses "the same combined score, the geometric mean of the two task metrics" — the same quantity, unconnected.
- Current: "and the joint model at the epoch its joint validation score selects, with both tasks read from that one model."
- Replace: "and the joint model at the epoch selected by its joint validation score (the geometric mean of the two task metrics), with both tasks read from that one model." (The cascade paragraph can then say "the same combined score used for model selection".)

**5. §6.1 CTLE sentence is ~80 words with a stacked "at ... at ... at" and a dangling "our 75.15".** `sections/06_results.tex:27`.
- Current: "Fine-tuned together with the task model at Florida at its authors' defaults, at the same 64 dimensions and windowing as ours, CTLE reaches $33.45$ macro-F1 at its best epoch and $29.69$ at its final one (an epoch is one pass over the training data), about two points below the place embedding under the same best-epoch rule and below our $75.15$; frozen and fed to the same single-task model, it repeats the ordering at Alabama, Arizona, and Istanbul, with gaps of $+37.8$, $+37.0$, and $+28.7$ macro-F1."
- Replace (split, name the referent): "Fine-tuned together with the task model at Florida, at its authors' defaults and at the same 64 dimensions and windowing as ours, CTLE reaches $33.45$ macro-F1 at its best epoch and $29.69$ at its final one (an epoch is one pass over the training data), about two points below the place embedding under the same best-epoch rule and far below the check-in-level representation's $75.15$. With fixed weights, fed to the same single-task model, it repeats the ordering at Alabama, Arizona, and Istanbul, with gaps of $+37.8$, $+37.0$, and $+28.7$ macro-F1." (Also replaces the bare "frozen" with "with fixed weights", matching §5.4's own wording for the same runs; note "far" budget rises to 3 — alternatively drop "far".)

**6. Abstract: missing unit and a stronger plain verb.** `main.tex:71-73`.
- Current: "this lifts next-category prediction over a standard place embedding (about $+28$ to $+40$ macro-averaged F1)"
- Replace: "this improves next-category prediction over a standard place embedding by about 28 to 40 points of macro-averaged F1" ("lifts" then survives once, in §7, within the GLOSSARY budget).

**7. Fig. 1 internal label "POI" vs. caption "place".** `figs/fig1_dataflow.tex:58` labels the level `POI` while the caption (`main.tex:97`) enumerates "(check-in, place, region, city)" and all §4.1 prose says "place". Rename the box to `place` (or `place (POI)`).

**8. "not from extra training signal" appears three times** (abstract `main.tex:74`, intro bullet `01_introduction.tex:27-28`, §6.1 `06_results.tex:24`) — the same negative-parallelism fingerprint the GLOSSARY §7 density rule warns about. Keep the abstract and intro instances (they scope the claim at first statement); in §6.1 end the sentence early: "...removes roughly 64 to 90 percent of the gain (state-dependent), so most of the gain is the context that each visit carries." (Also drops one metaphorical "carries" collision — currently 2, at budget.)

**9. Markov baseline lacks its plain gloss** (GLOSSARY §3: "keep, gloss once"). `sections/05_setup.tex:47`.
- Current: "and a Markov baseline over recent categories (Markov-K, best order per dataset)."
- Replace: "and a Markov baseline that predicts the category that most often follows the recent ones (Markov-K, best order per dataset)." (Low priority for this audience, but the glossary says gloss.)

**10. Unused acronym "LBSNs"** (`main.tex:62`): defined in the abstract, never used again anywhere. IEEE practice and the glossary's "acronym count as low as possible" both favor dropping "(LBSNs)".

**11. §7/§8 adjacent "First, ... Second, ..." enumerations** (`07_discussion.tex:21-22`, `08_conclusion.tex:9-11`) on top of §5.2's "First/Second/Third" — the exact density pattern GLOSSARY §7 flags. Suggest de-numbering the conclusion: "A check-in-level representation, one vector per visit rather than one per place, makes the next-category task far more learnable than the standard place-level one. On that representation, a single multi-task model outperforms..."

**12. [nit] AZ category Δ rounding**: prose "+9.35" (`06_results.tex:61`) vs. Table III displayed cells 65.79 − 56.43 = 9.36. If the Δ comes from unrounded means, no change is required, but a reviewer recomputing from the table will get 9.36; consider a table-note-free fix by quoting +9.4-level precision only, or leave as is knowingly.

**13. [nit] §5.3 never names the superiority test**; the reader learns it is a paired t only in §6.2, and region superiority is carried by the 90% CI. One clause in §5.3 ("superiority with a paired $t$ on the per-seed means, reported with the 90\% confidence interval of the paired difference") would close it.

---

## GLOSSARY §6 checklist, item by item

1. **Acronyms expanded at first use, count low** — PASS with nit: LBSNs (and arguably MTL) defined but barely reused (edit 10).
2. **category/region throughout; never activity/area** — PASS. "activity" occurs only describing MCARNN and DRRGNN (allowed); "area" zero.
3. **next category / next region / next place distinct; "we do not predict the exact next place" stated once** — PASS (§3, plus intro).
4. **Region wording (v17 board)** — PASS on verbs and scope: "outperforms" at Ist/FL/TX/CA, "matches"/non-inferior at AL/AZ; AZ explicitly "a match, not a gain" (never upgraded); scaling claim scoped to the five U.S. states in abstract, §1, §6.2, §7, §8; the formal "statistically non-inferior within a two-point margin (TOST)" appears (§1 bullet 2); no "ties"/"Pareto"/"beats"/"wins". **Exception:** the glossary's "CA/TX disclosed as a single seed" clause is now stale on the glossary side (all six cells are n=20 since 2026-07-11); the only remaining single-seed text in the paper is the Fig. 3 caption, which is now wrong (edit 1). Update GLOSSARY.md §1/§6 to the n=20 state; do not re-add single-seed wording to the paper. The §7 shortlist analysis correctly keeps its own "single seed over five folds" tag.
5. **No recipe/version codenames** — PASS (prose clean; codenames confined to comments).
6. **No bare substrate/engine/head/cross-attention** — PASS. "cross-attention" appears twice, both binding the "shared trunk" gloss; "head" only for HMT-GRN's next-place head (allowed for other systems); "substrate" only in invisible labels/comments.
7. **Every table has a lead takeaway sentence** — FAIL at Table II/§6.1 (edit 2); Table III/§6.2 and §6.3 pass; Table I's "summarizes" opener is acceptable for a data-summary table.
8. **"state of the art", never "SOTA"** — PASS (§5.4).
9. **American English** — PASS.
10. **No em-dash** — PASS (zero in prose, captions, tables).
11. **AI-tell sweep (§7)** — PASS: banned words/templates zero; boosters down to "far" ×2, "wide gap" ×1; -ly density normal; no true semicolon braids (multi-semicolon strings are CI notation and caption column definitions). Residual risk: the "X, not Y" density is still ~20 (edit 8 removes one decorative repeat); §5.2 + §7 + §8 enumeration stacking (edit 11).
12. **Idiom sweep (§8), captions included** — PASS: edges past/buys/ships/trail/staging/folds in/clears by all zero; "deliberately" zero; metaphorical "carries" ×2 (≤3); noun-"lift" zero ("lifts" verb ×2, within budget; edit 6 reduces to 1); "frozen" only for weights (§6.1 CTLE; edit 5 replaces it with §5.4's own "fixed weights"); "checkpoint" zero; "epoch" glossed at first use (§6.1).
13. **"seed" glossed at first use; abstract/intro say "random initialization"; arm(s)/result-"cell(s)" nowhere** — PASS: §5.3 carries the definition + the two-axes + 4×5=20 arithmetic verbatim; intro says "four random initializations"; "cell" appears only as "grid cell" and "radio cell" (both intended senses).
14. **"margin" only for the TOST margin; Part-1 difference is a "gap"** — PASS: all 5 "margin" uses are TOST-bound; §6.1 uses "wide gap"/"gaps".
15. **One name for the shared component** — PASS: "shared trunk" everywhere, glossed once in §1 as "a cross-attention stack"; "exchange stack" zero.
16. **§9 sweep** — PASS: constants carry provenance (0.75/0.25 "tuned once on validation... held fixed"; 2-pp margin "fixed in advance" with the service rationale; window-9 under the blanket fixed-in-development sentence); tuning asymmetry stated twice; one self-delta sentence in §2.2; §5.2 leak audit intact at full evidence floor (three grounds + numbers + per-fold prior + coverage caveat); Holm and TOST cited; digits for data quantities; relative pronouns written; zero contractions; "no change (0.00)" form used. Residual ellipses: "our 75.15" (edit 5) and the unnamed joint selector (edit 4).

Files: `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/figs/fig4_deltas.tex` (blocker), `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/sections/06_results.tex`, `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/main.tex`, `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/figs/fig1_dataflow.tex`, `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/src/sections/05_setup.tex`, `/Users/vitor/Desktop/mestrado/ingred/articles/[mobiwac]/GLOSSARY.md` (stale CA/TX single-seed rule, needs its own update).