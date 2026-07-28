# Round 6 agent brief — read this before you touch anything

**Repository root:** `/Users/vitor/Desktop/mestrado/ingred`
**Dissertation folder:** `articles/dissertacao/` (referred to as DISS below)
**LaTeX source:** `DISS/src/` — this is the ONLY working copy. `storyline/` and `fundamentals/` are frozen.
**Support material:** `DISS/src_utils/` (a SIBLING of `src/`, never inside it)

## 0 · The law you obey, in read order

1. `DISS/CLAUDE.md` — the landing and the decisions ledger.
2. `DISS/WRITING_LAW.md` — the word-level law. Register, canonical names, honesty rules, AI-tell bans.
3. `DISS/GLOSSARY.md` — the term registry. **Fail-closed: a term not in the registry may not be used.**
   You PROPOSE new entries; the author approves; the entry lands BEFORE the term does.
4. `DISS/AGENT_GUARDRAILS.md` §1–§3 — citation protocol, number protocol, claim registry. Obey exactly.
5. `DISS/NORTH_STAR.md` §3–§4 + §6 — chapter map, per-chapter errata, story spine.
6. `DISS/science/AGENT_HANDOFF.md` — the failure modes agents actually hit in this repository.

For **Chapter 5** prose the MobiWac paper's own glossary wins: `articles/[mobiwac]/GLOSSARY.md`
(393 lines, a 26-row jargon substitution table and a never-use list). It was missed in an earlier
round; do not miss it.

## 1 · Building the document (this WILL fail if you skip it)

```bash
source articles/dissertacao/src_utils/texenv.sh   # sets PATH, TEXMFHOME, TEXMFVAR, TEXMFCONFIG
cd articles/dissertacao/src
make defense       # -> build/main.pdf, copied to dissertacao.pdf   (104 pp)
make final         # -> build/main_final.pdf                        (99 pp)
make check         # the lint gate
bash ../src_utils/build.sh . both    # the honest build report
```

**Why texenv.sh exists.** Without `TEXMFHOME` you get "abntex2.cls not found" (honest). With the
WRONG `TEXMFVAR` you get `!pdfTeX error: Font ntx-Regular-tlf-ot1r at 657 not found ==> Fatal
error`, which is NOT a missing font (both .tfm and .pfb are in the home tree) but a missing font
MAP. `kpsewhich -var-value TEXMFVAR` reports an unreadable path on this machine, so it cannot be
probed. Source the script.

## 2 · The failure mode that governs this round

**From commit `6d780b58` to `a880632b` the source did not compile, and the gate could not see it.**
A missing `{` in `tables/frame/bib_errata.tex`. The two build paths disagreed:

- `make` passes `-halt-on-error` → died, produced NO PDF.
- `build.sh` passed `-interaction=nonstopmode` → pdflatex RECOVERED and still wrote a 104-page PDF,
  which build.sh then measured and reported as `pages=['104'] overfull_hbox=0 undef_cite=0`.

Six commit messages carried "104/99 pp, 0 overfull, 0 undefined" from a PDF built out of a source
tree containing a LaTeX error. Fixed in `ba90aa6d`; `build.sh` now reports `tex_errors=N` and fails.

**What this means for you:** `tex_errors=0` is now part of every build claim. A PDF existing is not
evidence the source is correct. And **a check that has never failed is not a check** — validate
your gate in both directions before you trust it.

## 3 · Non-negotiables

- **Nothing from model memory.** Every citation, number, name, date traces to a file in this repo or
  a source opened in your session. A reference is usable only with (a) a resolvable identifier
  checked against the source of record, (b) the landing page or PDF actually opened by you, and
  (c) the specific claim located in the source. If any of the three fails: DROP it or flag
  `[VERIFY: ...]`. Never smooth over.
- **Quote numbers, never compute them.** No mental arithmetic, rounding, aggregation, percentage
  conversion or delta-taking in prose. Derived quantities come from a committed script, then are
  quoted. Every number carries its reference point and its convention (metric, selection rule,
  n = seeds × folds).
- **Verbs bound to tests.** "outperforms" only with a paired superiority test; "matches" only with
  TOST non-inferiority within a two-point margin; never upgrade a non-inferior result to a win;
  never "beats", "wins", "ties".
- **Measure before you opine.** This project's standing failure is a plausible claim about its own
  state. If you assert the document does or does not do something, you measured it.
- **Verify the RENDER, not only the source.** The prior audit called Portuguese figure labels
  "RESOLVED" because PDF text extraction found no Portuguese — but those figures are raster PNGs
  whose text is not extractable. The labels are still there. Look at the image.
- **No em-dash anywhere in prose. No contractions. American English.**
- **No repo codenames in prose**: B9, v11-v17, champion-G, H3-alt, log_T (write "region-transition
  prior"), "substrate", "engine", "board", "recipe", "frozen", `mtlnet_crossattn_dualtower`.
- **Self-reported success is not trusted.** The author audits independently. State what you
  measured and how, and name what you could not confirm.

## 4 · Errata regime (differs by chapter, this matters)

- **Ch.3 (CBIC) and Ch.4 (CoUrb) are PUBLISHED.** A correction to their reproduced prose is applied
  in the dissertation and listed in Appendix B. The published article records are not edited.
- **Ch.5 (MobiWac) is UNDER REVIEW.** A correction is applied to the dissertation AND to the
  submitted source at `articles/[mobiwac]/src/`, so the two texts stay identical; it is then named
  in that article's own errata record rather than in Appendix B. Author instruction, 2026-07-27.
- **Frame chapters (1, 2, 6) and appendices** are the author's own text: no errata mechanism, but
  claim changes are `[NEEDS SIGN-OFF]`-class.

## 5 · What you deliver

Write your report to `DISS/src_utils/_round6/<NN>_<name>.md`. It must carry:

1. **What you did**, with the file and line of every edit.
2. **A source ledger**: every reference → identifier → where you opened it → the claim it supports.
   Every number → source file → field → the convention it is on.
3. **`[VERIFY]` flags** for anything you could not close.
4. **What you could not confirm**, stated plainly.
5. **If you edited `src/`: the build result** from `bash ../src_utils/build.sh . both` —
   pages, `tex_errors`, overfull, undefined, oversized floats — plus `make check`.

Commit your own work with a message that says what changed and why, in the style of this
repository's history (the defect first, then the fix, then the measurement). Do not commit a
broken build.
