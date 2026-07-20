# TRANSLATION_NOTES.md — CoUrb 2026 faithful EN translation (`src_en/`)

**What this folder is.** A **faithful American-English translation** of the published CoUrb 2026
paper *ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado
Multitarefa* (Tarik S. Paiva, Vitor H. O. Silva, Germano B. dos Santos, Fabrício A. Silva; Anais
do CoUrb 2026, SBRC, pp. 323–336, DOI `10.5753/courb.2026.22960`). The Portuguese original is in
[`../src/`](../src/); this folder is the English mirror.

**Scope discipline (read before touching this).** This is a *translation*, produced under the
dissertation's translation-fidelity gate (`articles/dissertacao/AGENT_GUARDRAILS.md` L5,
`WRITING_LAW.md` §6). It is **not** the dissertation Ch.4 adaptation. The Ch.4 step — which applies
the audited errata below, renames tasks to the dissertation-canonical terms, and adds the
authorship/contribution note — is separate and needs the author's per-claim sign-off
(`NORTH_STAR.md` §4 Ch.4, decision #7). **Do not read `src_en/` as the audited/corrected version.**

---

## 1 · Faithfulness guarantee (what was verified)

- **Nothing added, nothing dropped.** Two independent fresh-eyes audits (hallucination/addition and
  omission) over every file pair returned PASS with zero findings: no sentence, claim, number,
  citation, equation, or figure/table exists in `src_en` that is absent from `src`, and vice versa.
- **Numbers map 1:1.** A machine round-trip over the nine file pairs confirmed identical sets of
  **30 cite keys, 13 `\label`s, 9 `\ref`/`\eqref`s**, and an identical multiset of every numeric
  value after locale normalization. All F1 cells, standard deviations, dataset counts, and
  hyperparameters ($d_{shared}=256$, 7 classes, 8 heads / 4 layers, $L_h=9$, 192 / 1728 / 576 dims,
  ≤10 km / ≥70 km, $\tau=0.15$, $\alpha=0.5$, $S=16$, 10 km–10,000 km) match exactly.
- **Claim strength preserved.** Quantifiers, hedges, and comparatives map 1:1; nothing was
  strengthened, softened, or "corrected."
- **Builds.** `pdflatex` + `bibtex` produce a complete 14-page PDF with resolved citations and zero
  `!` errors (see §5 for the one inherited asset caveat).

## 2 · Deliberate translation decisions (format changes, not content changes)

1. **Bilingual layout kept.** The English `\abstract` (already the authors' own English) is preserved
   verbatim; the Portuguese `\resumo` is kept verbatim (SBC template + author decision). Only the
   body, `\title`, section titles, table captions/headers were translated.
2. **Number *format* → American English, *values* unchanged.** Decimal comma → point
   (`62,44` → `62.44`; `0{,}15` → `0.15`), thousands point → comma (`990.518` → `990,518`;
   `10.000` → `10,000`). Every value is identical; only the locale glyphs differ. Set notation
   `\{0,1\}` and RGB color triples keep their commas (not decimals).
3. **`babel[brazil]` retained** (lowest build risk; needed by `\resumo`). Cosmetic consequence: the
   English body uses Brazilian hyphenation patterns. Switching to `[main=english,brazil]` is deferred
   (it touches `\resumo` rendering) and is not needed to build.
4. **Commented-out LaTeX kept byte-identical** (the HAVANA and Space2Vec/Space2Vec-grid `%` blocks
   remain in Portuguese; they do not render).
5. **Portuguese completed inside math** (`\text{}` labels are human-readable): `\text{se}` → `\text{if}`
   in the Time2Vec equation; loss subscripts `POI--região`/`região--cidade` → `POI--region`/`region--city`;
   symbol subscript `\mathbf{E}_{\text{POI\_categoria}}` → `POI\_category` (all occurrences, consistently).
6. **Three redundant PT→EN glosses collapsed:** the source glossed a Portuguese term with its English
   name, which duplicates once the body is English, so `Location-Based Social Networks (\textit{Location-Based
   Social Networks}, LBSNs)` → `Location-Based Social Networks (LBSNs)` and, twice,
   `multi-task learning (\textit{Multitask Learning}, MTL)` → `multi-task learning (MTL)`. No information
   lost (term + acronym retained).
7. **Three awkward literal carryovers reworded** (meaning-preserving; flagged by the style audit):
   "follows the following format" → "is organized as follows"; "Differently from what occurs in
   classification" → "Unlike classification"; "the adoption itself of" → "the very adoption of".
8. **English caption/reference labels.** `babel[brazil]` (kept for the `\resumo`) auto-generates
   Portuguese labels, so the PDF rendered "Figura", "Tabela", "Referências". Added
   `\addto\captionsbrazil{\renewcommand{\figurename}{Figure}...\tablename{Table}...\refname{References}}`
   after the babel load so figures/tables/bibliography render in English. Rendered text only; the
   `\resumo` stays Portuguese.

Style audit passed: em-dash count 0, contractions 0, banned AI-tell words 0, American spelling
throughout, terminology consistent with the paper's own English abstract.

## 3 · CARRIED-OVER ERRATA — reproduced verbatim, NOT fixed (for the Ch.4 step)

These are in the **published** paper and are preserved faithfully here. The dissertation's audit
recorded corrected values; apply them only in the Ch.4 adaptation, with author approval
(`NORTH_STAR.md` §4 Ch.4; source of audited numbers: `../slides/judge_feedback.md`).

| In `src_en` (verbatim) | Audited value (do NOT put in `src_en`) | Where |
|---|---|---|
| next-POI: **"16 of the 21"** (76%) wins | **15/21 strict wins + 1 technical tie** — the tie is FL Outdoors (baseline 21.61 vs Sphere2Vec-M 21.59, a 0.02 gap within σ) | results.tex, conclusion.tex, abstract |
| category: **"20 to 24 percentage points"** average gains | per-state means **+20.2…+22.0 pp** (the 20–24 range is best-of-two-encoders per row; SIREN-alone at TX is ≈ +17.9 pp) | intro.tex, results.tex, conclusion.tex, abstract |
| `\cite{silva2025mtlnet}` bib entry | venue name is wrong ("Brazilian Conference on Intelligent Systems (CBIC)") and marked "Submetido"; the real work is CBIC 2025, DOI `10.21528/CBIC2025-1191324` | `references.bib` |

The FL-Outdoors next-POI baseline-bold cell (`\textbf{21.61 ± 0.99}`) is preserved exactly as
published — it was **not** re-bolded or reclassified.

## 4 · Other honesty items to raise in Ch.4 (not translation issues)

Carried over from `NORTH_STAR.md` §4 Ch.4, for the frame narrative (not to be edited into `src_en`):
the evaluation split is stratified by sample, not user-disjoint; no external baselines; the Nash-MTL
"works" assumption predates the repo's NashMTL solver-bug history. These are time-indexed framing
points for the dissertation, orthogonal to the translation.

## 5 · Assets (updated 2026-07-20)

- **Figure 2 image supplied.** `imagens/subáreas/distribuicao_estados.png` (plus `florida/california/
  texas.png`) was added to `src/` by the author and copied into `src_en/`; Figure 2 now renders in the
  build. The figure is already in English (axis labels Latitude/Longitude, legend Food/Shopping, state
  titles), so there is no in-figure Portuguese to fix.
- **Figure 1** (`arquitetura_modelo.png`) uses English labels (Category Output / Next POI Output).

## 6 · How this was produced (provenance)

Section-bounded translation (7 units) via a dynamic workflow, each unit **translated then verified by
a separate fresh-eyes agent**, followed by **4 parallel cross-document reviewers** (hallucination,
omission, number+citation round-trip, AI-tell/American-English) — all PASS. The author then
independently re-read every file, ran a machine number/citation round-trip and a decimal-leak scan,
applied the §2.5–§2.7 completions, and confirmed the `pdflatex`+`bibtex` build. Nothing entered the
text from model memory; every number traces to `../src/`.

## 7 · Adversarial red-team (2026-07-20)

A second, adversarial workflow attacked the translation: **7 attacker lenses** (semantic fidelity,
claim strength, numbers/tables, citations/refs, omission/addition, LaTeX-structure-render,
native-English register) each trying to prove the translation guilty, then **independent
adjudicators** that defaulted to REJECT unless a defect could be proven against both files. Result:
**13 findings raised → 2 CONFIRMED (both minor), 11 rejected, 0 uncertain** — no critical or major
defect survived; numbers, claim strength, citations, and structure all held. The two confirmed
defects were fixed:

- **metodology.tex**: removed a broken self-gloss `hard parameter sharing (\textit{hard parameter
  sharing})` (the redundant-gloss collapse of §2.6 that had missed this one instance).
- **results.tex**: `expressive` → `substantial` / `largest` (a false-friend calque of PT *expressivo*,
  which means substantial in magnitude, not "expressive").

## 8 · Reading a compiled PDF

`src_en/` builds with `pdflatex` + `bibtex`, and a compiled `src_en/main.pdf` (14 pp) is committed
alongside the sources. On a full TeX install it renders the proper SBC look with no changes. On a
**BasicTeX** install (missing the URW Helvetica/Courier metrics) the fonts must be substituted, e.g.
inject on the command line (so the source stays unchanged):

```
pdflatex -jobname=main '\AtBeginDocument{\renewcommand{\ttdefault}{\rmdefault}\renewcommand{\sfdefault}{\rmdefault}}\input{main.tex}'
bibtex main ; pdflatex ... ; pdflatex ...
```

or once with `sudo tlmgr install helvetic courier` to get the real fonts. Figure 2 is blank until
its image (§5) is supplied. The committed `main.pdf` was built with the font substitution above
(headings/URLs in Times, not Helvetica/Courier); it is a reading proof, not the camera-ready look.

## 9 · Revision sync (2026-07-20)

The author revised `src/` after the initial translation; the deltas were spotted via `git diff` and
ported to `src_en/`, then verified by a fresh-eyes fidelity pass (PASS, no defects):

- `references.bib`: the uncited MTL/optimizer/head/benchmark entries were removed in `src/`; `src_en`
  synced to be identical.
- Figure 2 images added (see §5); Figure 2 now renders.
- New content translated faithfully: `\section*{Acknowledgments}` (MCTI, Manna Team, Araucária
  Foundation, Softex, CNPq project 421548/2022-3, FAPEMIG, CAPES); a Gowalla-vintage limitation
  paragraph (2009–2010 collection window, caution on generalization to current mobility); a
  dimensionality-confound caveat in Embedding Integration; and a Travel/complementarity paragraph in
  Results. A new commented-out alternative future-work block is kept byte-identical in Portuguese.
