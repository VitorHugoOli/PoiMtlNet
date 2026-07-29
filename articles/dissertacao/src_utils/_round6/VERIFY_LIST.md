# VERIFY_LIST.md — the twenty checks worth your own eyes, in order of consequence

**Built 2026-07-28 by the ledger track**, against `4e84cf7a` (108 / 105 / 109 pp; the per-section
chapter split, whose render is byte-identical to `01915ba7`). Companion to
[`SOURCE_LEDGER.md`](SOURCE_LEDGER.md), which carries the full trail; this file is the short list.

**How to read the ordering.** Consequence, not effort. Items 1-6 are things that would mislead the
banca or the advisor if wrong. Items 7-13 are claims a pass verified about **its own work**, where no
fresh eyes have looked. Items 14-20 are traceability and hygiene. Every row gives the phrase to
anchor on, the command or the page, and **what the answer should be if all is well** — so a check
that comes back different is the finding.

**If you only do three, do these:** item 0 (a numeric bound in your submitted paper is false for the
first state it names), item 6 (a frame sentence on p. 23 now contradicts a chapter sentence on
p. 36, and the repair was drafted but never applied) and item 1 (a gate reported green across the
whole round is red, and has been red since before the round started).

---

**0. The `±0.003` gradient-cosine bound is false, in the dissertation AND in the submitted paper.**
`chapters/5_mobiwac/02_related.tex:161` and `articles/[mobiwac]/src/sections/02_related.tex:99`,
both reading "per-dataset means within $\pm0.003$".
```bash
sed -n '29,31p' ../../docs/studies/archive/mtl_improvement/WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md
```
(from the repository root). *If all is well:* the per-state means read FL +0.0007, **AL +0.0032**,
AZ −0.0005, GE −0.0004 — and `0.0032 > 0.003`, so the bound is false for **Alabama, the first state
the sentence names**. The round rescoped this sentence's pool from three datasets to four and carried
the bound over unchanged; the bound is right only against the superseded two-run figures (AL
+0.0026). The pooled `+0.001` is correct and the orthogonality conclusion is untouched — this is a
false bound, not a false finding. It is first on the list because it is in a manuscript under review
and it needs the two-file change plus the `ERRATA.md` line. Raised as N-1 by the number/claim pass;
**I re-derived it at the source and confirm it**, and I note that my own ledger row is where it slipped
past, because I recorded the cosine as inherited rather than re-deriving it. Ledger finding L-8.

Before anything else, one command reproduces the build state every other row assumes:

```bash
cd ~/Desktop/mestrado/ingred/articles/dissertacao && source src_utils/texenv.sh \
  && (cd src && make defense && make academico && make ppgc)
grep -h 'Output written' src/build/main.log src/build/main_academico.log src/build/main_ppgc.log
```
Expect **108, 105, 109 pages**. (This rewrites the tracked `src/dissertacao.pdf`; `git checkout --
articles/dissertacao/src/dissertacao.pdf` afterwards if you do not intend to commit it.)

---

## Where to run these commands

Unless a block says otherwise with its own `cd`, **run from `articles/dissertacao/`**:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
```

Paths that reach outside the dissertation folder are written `../../` from there. Blocks that begin
`cd articles/dissertacao` are meant to run from the **repository root** and say so in their own first
line.

**What has and has not been verified about these commands.** Run
`python3 src_utils/check_verify_list.py` from `articles/dissertacao/`; as of 2026-07-28 it reports:

```
15 documented command(s) executed; 7 carried a machine-checkable expectation; 0 failed.
2 skipped to avoid recursion (they invoke this gate's own caller)
1 build block: cwd checked, build NOT run here
8 were executed but NOT asserted against their prose expectation.
```

Read that as four categories, because they are not the same thing:

- **7 verified.** Output compared against a stated expectation (`# EXPECT: lines=N` / `contains=` /
  `equals=` inside the block).
- **8 run, not verified.** They execute and produce output, but their "if all is well" text is a human
  judgment or too discursive to encode. **Do not describe these as verified.**
- **2 skipped, deliberately.** They invoke `make check`, which invokes this harness. Running them here
  does not terminate; they are exercised every time `make check` itself runs.
- **1 build block.** Its working directory is checked but the three-target build is not re-run here —
  that takes four minutes and `build.sh` is the tool for it. Verified separately: 108/105/109 pages,
  `tex_errors 0`, `make check` RC=0.

**Greps over `.tex` files strip comment lines first** (`grep -vn '^[[:space:]]*%'`). This source carries
dense provenance comments that quote the very strings being searched for, so an unfiltered sweep
reports more hits than the reader sees. Three commands in this file were wrong for exactly that reason
on 2026-07-28 and were corrected: a `\path{}` count annotated 13 that returned 15, a "four of six"
sweep promising 3 prose hits that returned 4, and a "three of our six" sweep promising **zero** that
returned 5, every one an audit comment. Six paths across four commands also resolved from neither
working directory. The harness above exists so the next such defect is caught by running the file
rather than by reading it.

## Tier 1 — would mislead a reader or the banca (items 1-6)

**1. `make check` is red, and the round's build claim says it is green.**
The round state and the split commit both assert "`make check` all gates pass". It exits 1, and it
does so at `870f882c`, `01915ba7` and `4e84cf7a` alike, on the `'this paper' / 'this article'`
gate.
```bash
cd src && bash ../src_utils/check.sh; echo "EXIT=$?"
```
*If all is well:* you see exactly one hit,
`chapters/apx_b_errata.tex:307: This article differs from the other two…`, `EXIT=1`, and you decide
whether that sentence (which refers to the MobiWac manuscript, not to the dissertation) earns the
same documented exemption `apx_b_errata` already has in the banned-words gate. **What must not
stand is a durable record claiming the gate passes while it does not** — that is the failure mode
`AGENT_GUARDRAILS` §7 names, and it is the reason this item is first. Ledger finding L-1.

**2. The Standley correction changes a published claim against the chapter's own interest.**
Page 34 of the defense PDF, the `Empirical Performance` bullet plus its footnote.
*Check:* read p. 34 and the footnote, then read the Appendix B row (p. 93). *If all is well:* the
bullet claims **only** reduced inference cost; the footnote reproduces the published sentence, says
the cited work names accuracy and reduced training time among benefits joint training may have "in
theory", and quotes it arguing the other way. I re-read `arXiv:1905.07553` (v3 and v4) this session
and confirm all nine quotations verbatim. **This is `[NEEDS SIGN-OFF]` and it removes a stated
advantage of the architecture Chapter 3 adopts — it is your call, not the reviewer's.**

**3. The Nash-MTL guarantee, narrowed in published co-authored prose.**
`chapters/4_courb/methodology.tex:36`, rendered p. 47.
*If all is well:* the sentence reads "Away from a Pareto-stationary point … and under the method's
assumption that the gradients are linearly independent there, that direction is a descent direction
for every task". Both conditions are in the paper: p.1 "Under certain as-sumptions", p.3 "if θ is
not Pareto stationary then the gradients are linearly independent", p.6 "our update rule is a
descent direction for all tasks". I verified all three in the 19-page PDF. Also `[NEEDS SIGN-OFF]`.

**4. Two glossary terms are in the rendered document and not in the registry.**
The registry is fail-closed, so this **blocks** the new Ch.2 paragraph rather than merely awaiting
wording.
```bash
grep -c 'bilinear discriminator\|logistic function' GLOSSARY.md   # expect 0 today
```
*If all is well:* you approve (or reject) the two proposed entries in `16_frame_numbers.md` §4 and
the entry lands **before** p. 19 ships. Same question for **Pareto-stationary point** at p. 47
(`15_claim_scoping_applied.md` §9). Three entries, one decision.

**5. Chapter 5 hedges the region result and the frame does not.**
`chapters/5_mobiwac/05_setup.tex:76` (p. 66) states that the analysis plan "did not cover
next-region superiority, so the four next-region gains … are secondary results outside it". The
Resumo (p. 2), the Abstract (p. 3), Chapter 1 (p. 13) and Chapter 6 all say the joint model
outperforms on region "at four of six" with no such qualifier.
```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
for f in src/0_main.tex src/chapters/1_introduction.tex src/chapters/6_conclusion.tex; do
  grep -vn '^[[:space:]]*%' "$f" | grep 'four of six\|four of the six' | sed "s|^|$f:|"
done
# EXPECT: lines=3
```
Comment lines are dropped **before** the search (`grep -vn` keeps the original line numbers). Without
that this returns 4 hits, one being an indented provenance comment in `0_main.tex` rather than prose
the reader sees: **3 prose hits is the answer**, not 4.
*If all is well:* you rule either that the frame adds "as a secondary result" once, or that the
asymmetry is deliberate and goes in `LEFT_OUT.md`. The statistics record's own 2026-07-27 correction
is unambiguous that the registered primary test for **every** region cell is TOST non-inferiority.
No round-6 track owned this. Ledger finding L-5.

**6. The Ch.2 sentence that the Ch.3 protocol addition just falsified.**
Chapter 2 (`chapters/2_fundamentals.tex:601-602`, **rendered p. 23**) says Chapter 3 "reports
five-fold cross-validation without identifying the split axis". The Ch.3 addition landed and does
identify it (`chapters/3_cbic/results.tex:30`, rendered p. 36). I confirmed both in the PDF, so this
is a live contradiction the reader can see, thirteen pages apart.
```bash
# the clause wraps across two source lines, so grep the file as one string:
python3 -c "print('without identifying the split axis' in open('src/chapters/2_fundamentals.tex').read().replace(chr(10),' '))"
```
*If all is well:* that prints `False`, because the clause has been replaced by the repair drafted in
the comment at the Ch.3 site ("Chapters 3 and 4 both stratify by sample rather than by user … and
only Chapter 5 splits by user"). It prints `True` today. `[VERIFY]` V-8.

---

**6b. A printed count in the errata appendix does not sum.**
Page 95 says the MTLnet spelling was normalized "at all 25 places where the name appears in the
printed chapter: 21 in prose, one in a subsection heading, one in a figure caption, and two in table
headings".
```bash
grep -rn 'subsection{.*MTLnet' src/chapters/4_courb/    # expect TWO hits
# EXPECT: lines=2
```
*If all is well:* the appendix says 26 with "two in subsection headings" — because there are two
(`methodology.tex:87` "Baseline: MTLnet with DGI" and `related.tex:42` "The MTLnet framework"), and
21 + 2 + 1 + 2 = 26. The chapter's own source comment (`4_courb.tex:7`) already says **26 sites**,
and `12_figures.md` calls it the 26-site normalization, so three records give three counts and the
wrong one is the one that prints. No result depends on it; it is in the appendix whose only job is to
be exactly right about what changed. Ledger finding L-9.

---

## Tier 2 — a pass verified its own work and no fresh eyes have looked (items 7-13)

**7. The Appendix A protocol numbers, which describe your own conduct.**
`chapters/apx_a_contributions.tex:107,113` (rendered p. 91): `StratifiedGroupKFold`, five splits,
partition seed **42**, grouping by user, seeds **0, 1, 7, 100**.
```bash
grep -n 'random_state=42\|Seeds {0, 1, 7, 100}' ../../docs/context/DATA_SPLITS.md
```
*If all is well:* `DATA_SPLITS.md:16` and `:65` say exactly that. I reproduced both. The reason to
look yourself is that this appendix is new this round and is a claim about how the experiments were
run, which is the class of claim you personally answer for at the defense.

**8. The Check2HGI loss equation (three numbered equations, new to Ch.2, p. 19).**
The weights `0.4 / 0.3 / 0.3` appear in two independent places and agree:
`docs/context/check2hgi_overview.tex:215` and
`research/embeddings/check2hgi/model/Check2HGIModule.py:51-53`, summed at `:1192-1195`. I checked
both. *If all is well:* they still agree, **and** you settle the open question the pass could not:
the code carries two further auxiliary terms whose defaults are `0.0` and which the equation omits.
`[VERIFY]` V-14's sibling; the pass says settling it needs the run configuration of the shipped
representation.

**9. The joint model's descent from MTLnet (new frame prose, p. 20).**
The claim is that the joint model is a *specialization* of the MTLnet class overriding exactly one
component.
```bash
sed -n '42p'  ../../src/models/mtl/mtlnet_crossattn_dualtower/model.py  # class …DualTower(MTLnetCrossAttn)
sed -n '207p' ../../src/models/mtl/mtlnet_crossattn/model.py            # class MTLnetCrossAttn(MTLnet)
sed -n '368p' ../../src/models/mtl/mtlnet_crossattn/model.py            # "Override MTLnet's FiLM + shared_layers…"
```
(from the repository root). *If all is well:* all three lines read as above — I confirmed each of the
six coordinates the comment cites. This one is worth your eyes because it is the sentence that
licenses reading Chapter 3's null against Chapter 5's positive result, which is the arc of the
dissertation.

**10. The Resumo and Abstract word counts, on the rendered page.**
Reported as 310 / 271. I measure **310 and 272** with the report's own instrument.
```bash
python3 src_utils/_round6/_measure_abs.py <(printf '[{"pdf":"src/build/main.pdf","pages":[2],"label":"Resumo"},{"pdf":"src/build/main.pdf","pages":[3],"label":"Abstract"}]')
```
*If all is well:* Resumo 310 words / 11 sentences / mean 28.2 exactly, Abstract 272 / 11 / 24.7. The
one-word gap is two soft-hyphen breaks on p. 3; the instrument does not apply the hyphenation
normalization its own documentation declares. Trivial in size, but it is a number in a durable
record that does not reproduce. Ledger finding L-4.

**11. The near-blank page the Resumo cut was meant to remove.**
*Check:* open p. 2 of the defense PDF. *If all is well:* the `Palavras-chave` block is on **p. 2
with the Resumo** (I confirmed: keywords appear on p. 2 only, and the old orphan page is gone; front
matter word counts are p.1 = 54, p.2 = 363, p.3 = 317). Worth a glance because the pagination has
moved three times since that fix.

**12. The paper/dissertation parity divergence at the trunk attribution.**
The round softened the attribution in **both** texts and declared one deliberate divergence:
Chapter 5 states the disconfirming ablation with its numbers, the paper does not.
```bash
cd /Users/vitor/Desktop/mestrado/ingred
for f in articles/dissertacao/src/chapters/5_mobiwac/07_discussion.tex \
         'articles/[mobiwac]/src/sections/07_discussion.tex'; do
  grep -vn '^[[:space:]]*%' "$f" | grep 'One model serves both tasks' | sed "s|^|$f:|"
done
# EXPECT: lines=2
```
Two fixes to this command: the paper path was written relative to `articles/dissertacao/` and did not
resolve from where the rest of this list is run, and without the comment filter the paper file returns
an extra hit that is a section banner. Filtered, **one prose hit per file** is the answer.
*If all is well:* the same sentence opens both (dissertation p. 73), neither names a component as
the source of the category gain, and `articles/[mobiwac]/ERRATA.md` carries the four dated entries.
The declared divergence is a judgment you should endorse or reject, since it is your submitted
paper.

**13. The `+0.001` gradient-cosine sentence, fixed for parity in both texts.**
```bash
for f in $(grep -rl 'three of our six\|three of six' src/ '../[mobiwac]/src/' 2>/dev/null); do
  grep -vn '^[[:space:]]*%' "$f" | grep 'three of our six\|three of six' | sed "s|^|$f:|"
done
# EXPECT: lines=0
```
Comment lines are dropped before the search. Unfiltered this returns **5 hits, every one an audit
comment** recording the old wording, which is the opposite of the "zero" the expectation states — the
filtered form returns nothing, which is what "zero prose hits" means.
*If all is well:* **zero prose hits** (only audit comments mention the old wording — I verified
this), and both texts read "four Gowalla states … Alabama, Arizona and Florida, which are three of
the five United States datasets reported here, and Georgia, which this study does not otherwise
use". I did **not** re-derive the cosine value itself; that number remains on the protocol pass's
authority.

---

## Tier 3 — traceability and hygiene (items 14-20)

**14. The `nash` page range, the one identifier nobody could resolve.**
`references.bib` gives `pages = {16428--16446}`. Crossref has no DOI for the ICML version, OpenAlex
returns only the preprint with null pages, Semantic Scholar confirms ICML 2022 but no pages, and
`proceedings.mlr.press` and `dblp.org` are both outside the sandbox allowlist. *One click closes
it:* `proceedings.mlr.press/v162/navon22a.html`. *If all is well:* the range matches, or you drop
the field — which is the precedent this same bibliography set for `standley2020tasks`. `[VERIFY]` V-5.

**15. `ruder2017sluice` is the third preprint entry, and it was not upgraded.**
The entry's title is the superseded preprint title ("Sluice Networks…"); the arXiv title of record
is "Latent Multi-task Architecture Learning" and the version of record is AAAI 2019
(`10.1609/aaai.v33i01.33014822`, v.33 pp. 4822-4829). I resolved both. *If all is well:* you take
the metadata decision **together with** the claim decision at the same key (it carries the round's
highest-load NOT-SUPPORTED verdict, `chapters/3_cbic/method.tex:91`) so the entry is touched once.
Ledger finding L-3.

**16. Three claim-support verdicts on published prose you have not yet ruled on.**
`chapters/4_courb/methodology.tex:126` (`sun2020go` cited for temporal cycles revealing place
*function*), `:184` (`belkin2003laplacian` cited for a hierarchical embedding regularizer), and
`chapters/3_cbic/method.tex:91` (`ruder2017sluice` cited for hard-sharing regularization). All three
are NOT-SUPPORTED at high load, all three are in reproduced published prose, so all three are
errata decisions rather than free edits. *If all is well:* each gets a ruling and, if changed, an
Appendix B row. The suggested swaps (`baxter2000model`, `Xu2023`) are already in the bibliography
and already cited for those claims elsewhere.

**17. The Appendix B reconciliation count.**
The header now claims `8 + 13 + 4 + 18 = 43` itemized rows, replacing a stale `= 36`.
```bash
python3 - <<'PY'
import re
for f in ["src/tables/cbic/errata.tex","src/tables/cbic/errata_wording.tex",
          "src/tables/courb/errata.tex","src/tables/frame/bib_errata.tex"]:
    t="\n".join(l for l in open(f).read().splitlines() if not l.lstrip().startswith('%'))
    m=re.search(r'\\endlastfoot(.*?)\\end\{longtable\}',t,re.S) or re.search(r'\\midrule(.*?)(?:\\bottomrule|\\end\{longtable\}|\\end\{tabular\})',t,re.S)
    rows=[r for r in re.split(r'\\\\\s*',m.group(1)) if r.count('&')>=1 and r.strip() and 'multicolumn' not in r]
    print(f, len(rows))
PY
```
*If all is well:* 8, 13, 4, 18. I reproduced exactly this.

**18. The two errata rows and the claim they carry, as the reader sees them.**
Pages 93 (B.1, the Standley narrowing) and 96 (B.3, the Nash guarantee). *If all is well:* both rows
name the cited work's own position; the B.4 Sphere2Vec row **names the work rather than printing the
52-character key** (printing it produced the round's only overfull box, `113.58371pt`); and there
are **0 overfull boxes and 0 oversized floats** in all three builds — which I confirmed.

**19. The Appendix B static-scope section, which makes a public statement about a published
co-authored result.**
Rendered p. 99, and **suppressible by commenting one `\input` line** at
`chapters/apx_b_errata.tex:407`, per your own condition. Its numbers reproduce: I recomputed the
fine-class counts from `data/checkins_by_state/*.parquet` and get 284 / 305 / 324 / 333 / 365 across
AL/AZ/FL/CA/TX with **zero** values spanning more than one category — exactly the range the section
states. *If all is well:* the numbers are right and the only open question is the one you reserved,
the advisor conversation. `[NEEDS SIGN-OFF]`.

**20. Sixty-three percent of this round's report coordinates now point past the end of their file.**
279 of 443 `file:line` references across the fifteen `_round6/*.md` reports land past EOF, because
the split reduced `3_cbic.tex`, `4_courb.tex` and `5_mobiwac.tex` to 55, 42 and 50 lines.
*If all is well:* you do not fix those reports. Use `SOURCE_LEDGER.md` tables A and B and this file
as the current address book — every load-bearing coordinate there was re-resolved by phrase against
the split tree on 2026-07-28 — and hold future reports to `ANCHORS.md` §5: cite the phrase, date the
line number. Ledger finding L-6.

---

### Two things deliberately NOT on this list

- **The 43 `[NEEDS SIGN-OFF]` markers as a set.** Six are this round's and are covered above (items
  2, 3, 7, 12, 19, plus the "identically" narrowing at p. 74). Reading all 43 is a separate pass,
  not a spot check.
- **The 25-row citation failure table as a whole.** Ten of its rows are low-load PARTIALs in
  reproduced prose, dispositioned "leave and record". Items 16 and 15 above pull out the four that
  carry real weight. The full table, with every identifier resolved and every site re-anchored, is
  `SOURCE_LEDGER.md` §A.3.

### One flag you should not re-raise

"The CA and TX category cells are provisional and the frame does not say so" was raised, checked,
and **withdrawn** — correctly. `stats_n20/RESULTS.md` is at rev 4 (2026-07-13) and reports all six
datasets rejecting at α = 0.05 (CA +6.45, TX +7.45, Holm-adjusted p = 8.9e-07); the provisional
material sits under a heading that literally begins `## 1b · … (✅ A1 n=20 now COMPLETE`. I
re-verified this independently. **"At all six" is correct.** The lesson worth keeping: that record
retains its superseded revisions inline, so anchor on the revision header, not on the first matching
line.

---

# Addendum: the seven items added after this list was written

**Appended 2026-07-28.** This list was written at `c5c6789d`, before the eight review tracks landed.
These seven come from what they found and what was changed in response. Same ordering rule as above:
by consequence, not by chapter order.

### A1. The corrected Appendix B paragraph on Chapter 3 — highest consequence in the document

**What to check.** That the paragraph says what you want said about Chapter 3, because it no longer
says what your 2026-07-27 ruling assumed.

**Where.** Defense PDF **p. 99**, section B.5, the paragraph beginning "The second is that the two
chapters differ in how direct the channel is". Source `src/chapters/apx_b_static_scope.tex`.

**How.** Read the paragraph. Then, if you want the mechanism checked rather than taken:

```bash
sed -n '114,131p' ../../research/embeddings/dgi/preprocess.py    # the feature: neighbours' mean, self excluded
sed -n '28,30p'  ../../research/embeddings/hgi/model/POIEncoder.py  # a single GCNConv, self-loops on by default
```

**What the answer should be if all is well.** The paragraph should say the two chapters differ in
**degree**: CoUrb's channel is an exact deterministic lookup, CBIC's is a neighbourhood average that
returns diluted through one convolution. It should **not** say Chapter 3 is unaffected. Your ruling
said "esse nao se aplica ao DGI que usamos no cbic"; the measurement says the channel there is
indirect, not absent. If you want the stronger exculpation, it cannot be supported as written.

### A2. The bounded Ch.4 number in the conclusion

**What to check.** That the two added sentences say what you would say.

**Where.** Defense PDF **p. 76**, from "Two qualifications bound what that number licenses".

**What the answer should be.** The 20.2 to 22.0 point figure stays (it is the published chapter's own
audited number), now labelled as the **static task's** and pointing at Appendix B; and the arc's
diagnosis should rest on the sequential task, naming Chapter 5 as what tests it.

### A3. The weakened reproducibility sentence

**What to check.** Whether you would rather publish the **nine** missing files than weaken the
sentence. Appendix A cites **thirteen** paths; four are already public.

**Where.** Defense PDF **p. 88**; source `src/chapters/apx_a_contributions.tex`.

**How.** Check all thirteen at once, so the four that are already there are visible too:

```bash
cd /Users/vitor/Desktop/mestrado/ingred
S=docs/studies/closing_data/v17_completion/stats_n20
for p in src/data/folds.py \
         scripts/closing_data/score_joint_best.py \
         scripts/closing_data/superiority_wilcoxon.py \
         scripts/closing_data/region_match_tost.py \
         docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md \
         scripts/build_phase3_per_fold_transitions.sh \
         docs/studies/closing_data/joint_best/JOINT_BEST_RESULTS.md \
         "$S" "$S/m1_stats_n20.py" "$S/m2_prereg_perfold.py" \
         "$S/m1_full_output.txt" "$S/m2_prereg_output.txt" \
         scripts/embedding_eval/autocorrelation_ceiling.py; do
  printf "%-70s " "$(basename "$p")"
  git cat-file -e "mobiwac:$p" 2>/dev/null && echo PRESENT || echo ABSENT
done
# EXPECT: lines=13
```

**What the answer should be.** The **first four PRESENT**, the **remaining nine ABSENT**. Push the
nine and the strong sentence comes back; the instruction is in the `[round6, F-01]` comment at the
site.

> **Two corrections to this item, 2026-07-28 (`c6e62c62`).** It first said "the eight files" and its
> command listed only eight, omitting `m1_full_output.txt` — a line-based grep of the appendix missed
> it, because it shares a source line with another `\path{}`. The command was also wrong in a way
> that would have looked right: it queried the four bare filenames at the repository root, where they
> do not exist in any branch, so they would have reported ABSENT for the wrong reason. They live
> inside `stats_n20/`, and the command above gives their full paths.

### A4. The deposit build's page numbering

**What to check.** That each build prints its own physical page number.

**How.** This compares the printed number against the physical position for you, rather than asking
you to eyeball three PDFs:

```bash
python3 - <<'PY'
import pypdfium2 as pdfium, re
for stem in ("main", "main_academico", "main_ppgc"):
    d = pdfium.PdfDocument(f"src/build/{stem}.pdf")
    for i in range(min(20, len(d))):
        t = d[i].get_textpage().get_text_range()
        m = re.match(r'\s*(\d{1,3})\s', t) or re.search(r'\n\s*(\d{1,3})\s*$', t)
        if m:
            got = int(m.group(1))
            print(f"{stem:11s} first numbered page: physical {i+1:3d} prints {got:3d}  "
                  f"{'OK' if got == i+1 else 'MISMATCH'}")
            break
PY
# EXPECT: lines=3
# EXPECT: contains=main        first numbered page: physical  11 prints  11  OK
```

**What the answer should be.** Three `OK` lines: `main` physical 11 prints 11, `main_academico`
physical 8 prints **8**, `main_ppgc` physical 12 prints 12. Before this round the deposit build
(then `main_final`) printed 11 on physical page 8, and every page after it inherited that three-page
error. Run `make defense && make academico && make ppgc` first if `src/build/` is stale.
(The deposit target was renamed `final` -> `academico` on 2026-07-29, LATEX_UPGRADE.md §4 A-1; the
command above is executed by `src_utils/check_verify_list.py`, so the stem here is live tooling and
not a frozen record.)

### A5. The footnote links

**What to check.** That clicking a footnote mark no longer jumps to page 1.

**How.** `grep -c Hfootnote src/build/main.log src/build/main_academico.log src/build/main_ppgc.log`,
then click a footnote mark in the PDF.

**What the answer should be.** **0** in all three logs, and the mark should be plain text with no link.

### A6. The gate suite, including the four new gates

**How.**

```bash
cd articles/dissertacao && source src_utils/texenv.sh && (cd src && make check); echo "RC=$?"
# EXPECT: contains=RC=0
```

**What the answer should be.** **RC=0** — for the first time this round; it exited 2 throughout while
six commit messages said otherwise. You should see `OK: 49 .tex files, every root directive present and
resolving`, `negative parallelism: ... 3.19 per 1k (ceiling 3.60)`, `OK: no doubled reference macros in
49 files`, and `trapped-prose suspects: 0`. Each of the four new gates self-tests in both directions
before it reports; if one prints only OK and no self-test line, distrust it.

### A7. The 46 sign-off markers, three of them first

**How.** `grep -rn "NEEDS SIGN-OFF" src/ | wc -l` should give 46. The by-file inventory is in
`PENDENCIAS.md` §2.1.

**What the answer should be.** Read A1, A3 and A2 above before the other 43. Those three are the ones
where the round changed what the document claims rather than how it says it.
