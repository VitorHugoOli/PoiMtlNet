# 43 - closing the liu2019dwa [VERIFY] against the PDF on disk

Baseline commit: **8f17f294**. Item: close the open `[VERIFY]` on `liu2019dwa` by locating the claim
in the paper body. Files touched: `src/chapters/2_fundamentals.tex` (the liu2019dwa clause and its
comment blocks only), `src/references.bib` (the `liu2019dwa` note and its provenance comment),
`src_utils/check_audit_claims.py` (three probes).

## 1 - The author's instruction, verbatim

> liu2019dwa -- o PDF esta em science/articles/1803.10704v2.pdf. Localize a alegacao em pagina e
> feche o [VERIFY], ou corrija a frase se o artigo disser outra coisa. Confira dois pontos no corpo,
> nao no abstract: o nome e a definicao do esquema de pesos, e se ele e apresentado como parte do
> MTAN ou como esquema separavel -- a prosa diz 'alongside their attention architecture' e e essa
> cautela que a mantem defensavel.

## 2 - The LIVE text before the edit (quoted from the file, not from the brief)

Prose, `chapters/2_fundamentals.tex` at the time of reading (lines 922-927):

```
uncertainty~\cite{kendall2018uncertainty}. Chen et al.\ introduce GradNorm, which
balances gradient magnitudes and training rates~\cite{chen2018gradnorm}. S. Liu et
al.\ introduce dynamic weight averaging alongside their attention architecture, using
recent changes in the task losses~\cite{liu2019dwa}, and B. Liu et al.\ propose FAMO,
which tracks loss changes without computing every task gradient at each
step~\cite{liu2023famo}.
```

The flag it carried (lines 948-953):

```
% [VERIFY] liu2019dwa, the neighbouring clause, is NOT corrected and is flagged instead. Its abstract
% (arXiv:1803.10704v2) is entirely about the MTAN architecture and never names dynamic weight
% averaging, which appears in the paper body. The chapter's phrasing is careful about exactly that ("
% introduce dynamic weight averaging ALONGSIDE their attention architecture"), so it is very likely
% right, but only the abstract was reachable this session and the claim is therefore not located in a
% source. Fail-closed: flag, do not reword a sentence that is probably correct.
```

The bib note before the edit, `src/references.bib:570`: `note = {Introduces Dynamic Weight
Averaging (DWA)},`.

## 3 - The source, and what was located in it

**Source opened this session:** `science/articles/1803.10704v2.pdf`, ten pages, text layer extracted
with pypdfium2. Identifier **arXiv:1803.10704v2** (the arXiv stamp is on p.1, left margin: "arXiv:
1803.10704v2 [cs.CV] 5 Apr 2019"). Bib DOI 10.1109/CVPR.2019.00197, unchanged and not re-checked
this session. The term "Dynamic Weight" occurs on p.2 (once) and p.5 (twice); "DWA" occurs on pp.
2, 5, 6, 7. It does **not** occur in the abstract, which is what the previous round reported.

### 3.1 The NAME. The chapter was WRONG, and this is a correction, not a confirmation

The paper's own name is **"Dynamic Weight Average (DWA)"**, singular *Average*, in both places it is
introduced:

- **p.2**, last paragraph of Sec. 1 (Introduction): "we also propose a novel weighting scheme,
  Dynamic Weight Average (DWA), which adapts the task weighting over time by considering the rate
  of change of the loss for each task."
- **p.5**, the heading of **Sec. 4.1.3** is literally "Dynamic Weight Average", and its text: "we
  propose a simple yet effective adaptive weighting method, named Dynamic Weight Average (DWA)."

The chapter said "dynamic weight averaging". That is the participle, not the authors' noun phrase,
so it is an R2 defect of the same class as the round-11 Aligned-MTL "principal/orthogonal" fix:
describe a system as its own authors describe it. **Corrected to "Dynamic Weight Average".** Note the
divergence this creates and why it is deliberate: `chapters/3_cbic/basis.tex:63` carries "Dynamic
Weight Averaging (DWA)", and Chapter 3 is a version of record under NORTH_STAR, so it keeps CBIC's
published wording and was not touched.

### 3.2 The DEFINITION. The old wording was fair but vaguer than the paper

**p.5, Sec. 4.1.3, Eq. 7 and the text around it.** The weight for task $k$ is a softmax over
$w_k(t-1)/T$ scaled by $K$, with

> $w_k(t-1) = \frac{L_k(t-1)}{L_k(t-2)}$

and the paper's gloss: "Here, $w_k(\cdot)$ calculates the relative descending rate in the range
$(0,+\infty)$, $t$ is an iteration index, and $T$ represents a temperature which controls the
softness of task weighting". On the window, same page: "the loss value $L_k(t)$ is calculated as the
average loss in each epoch over several iterations."

So the basis is the **ratio of the two previous loss values** (a rate of change), averaged per epoch,
not a general "recent changes in the losses". The old wording was not false, but it was looser than
the source, so the clause now names the rate of change and the ratio. The temperature and the softmax
are NOT in the prose: the chapter's sentence is a one-clause attribution, and adding $T$ would be
detail the surrounding sentences do not carry for any other method.

### 3.3 "ALONGSIDE" HOLDS. This was the word at issue, and the body confirms it

DWA is presented as a **separable weighting scheme, not as part of MTAN**. Three independent pieces
of body evidence:

1. **Placement.** It is introduced in the *experimental* section, **Sec. 4.1.3 (p.5)**, under the
   framing "To test our method across a range of weighting schemes, we propose a simple yet
   effective adaptive weighting method" -- that is, as an instrument for evaluating the
   architecture's robustness, not as a component of it. The architecture itself is Sec. 3 (pp. 2-4)
   and DWA appears nowhere in it.
2. **It is run on other architectures.** **p.6, Table 2** and **p.7, Table 3** list "DWA, T = 2" as
   one of three weighting rows under *each* of Split-Wide, Split-Deep, Dense, Cross-Stitch **and**
   MTAN. Training text, p.6: "we ran experiments with three types of weighting methods: equal
   weighting, weight uncertainty [14], and our proposed DWA (with hyper-parameter temperature
   T = 2...)". A scheme applied to Cross-Stitch Networks is not part of MTAN.
3. **The authors separate the two contributions explicitly.** p.1, Sec. 1 frames the paper's two
   challenges as "(i) Network Architecture (how to share)" and "(ii) Loss Function (how to balance
   tasks)"; DWA answers (ii), MTAN answers (i). And p.5 distinguishes DWA from GradNorm on exactly
   the separability axis: "whilst GradNorm requires access to the network's internal gradients, our
   DWA proposal only requires the numerical task loss, and therefore its implementation is far
   simpler."

**Verdict on the author's question:** the caution was correct and is kept. "alongside their attention
architecture" is the right framing and was NOT changed. A bare "introduce DWA" would drop the true
fact that the paper's headline contribution is MTAN; a "part of MTAN" phrasing would be false.

### 3.4 ATTRIBUTION

**p.1**, author line: "Shikun Liu   Edward Johns   Andrew J. Davison / Department of Computing,
Imperial College London". So **"S. Liu" is correct**, and the initial stays load-bearing: B. Liu
(Bo Liu, CAGrad and FAMO) appears twice in the same paragraph and is a different person. The bib
entry's author list already matched the paper and was not altered.

## 4 - What changed and where

Line numbers are POST-EDIT and the file was moving under me (another agent held the HGI subsection
and the 0.7 sentence in the same file), so each edit was made by matching the live string, not by
line number.

| # | File:line | Change |
|---|---|---|
| 1 | `src/chapters/2_fundamentals.tex:987-991` | The clause. "introduce dynamic weight averaging alongside their attention architecture, using recent changes in the task losses" -> "introduce Dynamic Weight Average alongside their attention architecture, deriving each weight from the rate of change of that task's loss, measured as the ratio of its two previous loss values". "alongside" untouched. |
| 2 | `src/chapters/2_fundamentals.tex:965-984` | NEW provenance block above the clause, page-level, in the style of the existing round-11 block below it: name (p.2 + Sec. 4.1.3 p.5), definition (p.2 and Eq. 7 p.5), separability (Sec. 4.1.3 p.5 + Tables 2-3 pp. 6-7 + the GradNorm contrast), attribution (p.1). Records the name correction as a correction. |
| 3 | `src/chapters/2_fundamentals.tex:1012-1014` | The six-line `[VERIFY]` block REPLACED by a three-line closure note pointing at the provenance block. |
| 4 | `src/chapters/2_fundamentals.tex:323-324` | Section ledger line: "DWA: loss-rate schedule weights" -> the paper's name plus the pages read. |
| 5 | `src/chapters/2_fundamentals.tex:335-340` | The round-10 lineage comment said "The paper's own contribution is the MTAN attention architecture; DWA is introduced alongside it". Updated to "headline contribution" and "a separable weighting scheme", which is what the body supports, plus the p.1 author-line read. |
| 6 | `src/references.bib:570` | `note` field: "Introduces Dynamic Weight Averaging (DWA)" -> "Introduces Dynamic Weight Average (DWA)". Renders in the bibliography (confirmed in the PDF, entry [53]). |
| 7 | `src/references.bib:565-570` | Provenance comment above the entry: PDF path, date, p.1 author line, the name correction, the separability finding with table pages. |
| 8 | `src_utils/check_audit_claims.py:494-521` | Three probes, `R12-dwa`, `R12-dwa2`, `R12-dwa3`, with the rationale block. |

Nothing else in the file was touched. No glossary term was added: "Dynamic Weight Average" is a
proper name of a cited system, in the same position as GradNorm, PCGrad and CAGrad, none of which
hold registry rows either (`Nash-MTL` does, and I did not touch it).

## 5 - The probes, and the sabotage that validated them

Added in the same edit as the fix, per the file's own "HOW TO ADD A PROBE".

- **`R12-dwa`** - `r"Dynamic Weight Average alongside their attention\s+architecture"`, must be
  PRESENT. Pins the name AND the separability in one collocation, because both are what the body
  established and either can drift alone.
- **`R12-dwa2`** - `r"dynamic weight averaging"`, must be ABSENT. Bans the superseded gloss from live
  prose. Matching runs on `live_text()`, so the comments that quote the old wording do not trip it.
- **`R12-dwa3`** - `r"rate of change of that task's loss,\s*measured as the ratio of its two previous loss values"`,
  must be PRESENT. Pins the definition to the paper's own basis.

**Sabotage, run one at a time, result read BEFORE restoring, and each replacement applied to every
occurrence of its target (`str.replace` with no count), not just the first:**

| Sabotage | rc | Which probes fired |
|---|---|---|
| "alongside" -> "as part of" in the clause | **1** | `R12-dwa` NOT APPLIED; other two hold |
| the name -> "dynamic weight averaging" in live prose | **1** | `R12-dwa` and `R12-dwa2` NOT APPLIED; `R12-dwa3` holds |
| definition -> the old "using recent changes in the task losses" | **1** | `R12-dwa3` NOT APPLIED; other two hold |
| restored (byte-compared against the original) | **0** | all three hold |

Suite after restore: **91 of 91 probes hold**, 0 not applied (88 -> 91 string probes; the total line
in the script reconciles automatically).

## 6 - The six exit codes (rule 6), each run as its own invocation from `src/`, rc read directly

| command | rc |
|---|---|
| `make defense` | **0** |
| `make academico` | **0** |
| `make ppgc` | **0** |
| `make extra` | **0** |
| `make check` | **2** (first run) -> **2** (after sync; see 6.1) |
| `make selftest` | **0** (run both before and after the sync) |

### 6.1 `make check` is RED, and the red is NOT mine. I did not touch it.

The first run reported TWO things. The first was the expected page-count staleness and I cleared it:

```
STALE CLAUDE.md: records 108 for the defense build, measured 104
STALE CLAUDE.md: records 105 for the academico build, measured 101
STALE CLAUDE.md: records 109 for the ppgc build, measured 105
STALE PLAN.md / src_utils/codex_reviewer.md: same three
```

I ran `python3 ../src_utils/sync_page_counts.py --write` from `src/` (rc 0, "all recorded page counts
agree with the build"), which the brief names as correct and expected under the suspended page
budget. That gate is now green: "all recorded page counts agree with the build".

The second is a **pre-existing failure in another agent's file**, still red after the sync:

```
FAIL     VERIFY_LIST.md: python3 -c "
         output does not contain 'repair_in_prose: True'
```

That block (`src_utils/_round6/VERIFY_LIST.md:455-469`) asserts the sentence "stratified its folds by
sample rather than by user" lives in `src/chapters/apx_a_contributions.tex`. **It has been moved out
of that file into `src/chapters/apx_extra_platform.tex:88`** by whoever is working on the appendix
split this round. Measured, not assumed:

```
baseline 8f17f294 apx_a repair_in_prose: True
live              apx_a repair_in_prose: False
grep -rn 'stratified its folds by sample' src/  ->  src/chapters/apx_extra_platform.tex:88
mtime apx_a_contributions.tex  Aug 3 10:51:12   (my first edit to 2_fundamentals.tex: 10:58)
```

The claim itself still holds (the sentence exists, once, in live prose); only its **address** moved,
exactly as the block's own 2026-08-02 repointing comment describes happening once before. The fix is
to repoint the `VERIFY_LIST.md` block at `apx_extra_platform.tex`. **I did not make it:**
`VERIFY_LIST.md` and the appendix files are not in my item's file list, another agent is live in
them, and repointing a probe I do not own is the shape of the "never reword a probe to make it pass"
prohibition even when the repoint is legitimate. It is flagged under UNFINISHED for the appendix
agent or the author.

So: **`check` rc 2 both before and after my edit, for a cause my edit did not create and does not
touch.** My three new probes pass inside the audit gate, which is itself green.

## 7 - Page counts

| build | before (recorded at baseline) | after (measured this session) |
|---|---|---|
| defense | 108 | **104** |
| academico | 105 | **101** |
| ppgc | 109 | **105** |
| extra | 22 | **26** |

The four-page drop and the extra-build growth are NOT from my edit: my prose change is within one
line of the original and the rest is comments. They are the appendix material moving into the extra
build, which is another agent's work in progress in this same tree. `--write` was run once, from
`src/`, for the reason in 6.1: the page budget is suspended, so the counts moved and the gate asks
for a sync. `extra` is not one of the three counts that script tracks.

## 8 - Verified in the RENDERED PDF, both directions (rule 7)

Text layer extracted with pypdfium2 from all four builds. `\s+`-normalized, the running header
`Chapter 2. Fundamentals 27` stripped (the clause straddles the page break between pp. 26 and 27),
and the typographic apostrophe in "task's" mapped to ASCII before matching, since a raw ASCII
assertion would have missed it.

| build | pages | new clause PRESENT | "dynamic weight averaging alongside" ABSENT | "using recent changes in the task losses" ABSENT |
|---|---|---|---|---|
| `build/main.pdf` (defense) | 104 | yes | yes | yes |
| `build/main_academico.pdf` | 101 | yes | yes | yes |
| `build/main_ppgc.pdf` | 105 | yes | yes | yes |
| `build/main_extra.pdf` | 26 | n/a (Ch. 2 not in this build) | yes | yes |

The rendered clause, copied out of `dissertacao.pdf`: "S. Liu et al. introduce Dynamic Weight Average
alongside their attention architecture, deriving each weight from the rate of change of that task's
loss, measured as the ratio of its two previous loss values [53]".

The bib note renders too, entry **[53]**: "LIU, S.; JOHNS, E.; DAVISON, A. J. End-to-end multi-task
learning with attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR). [S.l.: s.n.], 2019. Introduces Dynamic Weight Average (DWA)." Singular
*Average*, as the paper has it.

Two surviving occurrences of "Dynamic Weight Averaging" in the PDF were checked and are correct:
one is Chapter 3's published CBIC wording (version of record, not mine to touch), and the other was
the bib note, now fixed.

## 9 - `[VERIFY]` tokens: what remains

In `chapters/2_fundamentals.tex`, three remain and NONE concerns `liu2019dwa`:

- `:190` the averaging convention of the swept "Cat F1"
- `:472` whether Chapter 5's two runs share one loss form
- `:550` sources for a clause in the HGI/0.7 area (another agent's live item this round)

`grep -n "liu2019dwa" chapters/2_fundamentals.tex` returns four hits, none of them a flag: the
ledger line, the lineage attribution, the `\cite`, and the closure note. In `src/references.bib`,
no `[VERIFY]` sits on or near the `liu2019dwa` entry (the four `[VERIFY]` strings in that file
belong to other keys).

**Repository-wide count of `[VERIFY]` flags:** I searched for a tracker that claims one and found
none to update. `src_utils/CONSIDERATIONS.md` §5 is titled "Bandeiras `[VERIFY]` desta rodada" but
enumerates that round's flags without asserting a repository total; `PENDENCIAS.md`,
`CODEX_VS_PERSONAS.md`, `BIB_MERGE_REPORT.md` and `LEFT_OUT.md` mention individual flags, not a
count. So no number moved and nothing was edited outside my file list. If the author knows of a
tracker that does carry a total, it needs a decrement of one.

## 10 - Sources opened this session

| source | identifier | opened as | what was located |
|---|---|---|---|
| Liu, Johns, Davison, "End-to-End Multi-Task Learning with Attention" | arXiv:1803.10704v2; bib DOI 10.1109/CVPR.2019.00197 | `science/articles/1803.10704v2.pdf`, full text layer, 10 pp. | p.1 author line; p.2 Sec. 1 name + "rate of change of the loss for each task"; p.5 Sec. 4.1.3 heading, name, Eq. 7 $w_k(t-1)=L_k(t-1)/L_k(t-2)$, epoch-average window, the GradNorm contrast; pp. 6-7 Tables 2-3 with DWA on four non-MTAN architectures |

No other external source was consulted. Everything else in this report is a read of a file in this
repository.

## UNFINISHED

1. **`make check` is rc 2** on a failure I did not cause and did not touch:
   `src_utils/_round6/VERIFY_LIST.md:455-469` points its `repair_in_prose` assertion at
   `src/chapters/apx_a_contributions.tex`, but the sentence now lives at
   `src/chapters/apx_extra_platform.tex:88`. Repointing that block (one path string) turns the gate
   green; it belongs to whoever is splitting the appendices this round, not to this item.
2. **No commit was made.** The tree holds several agents' concurrent work and staging only my four
   files would still commit a tree whose `check` is red for item 1. Left for the author.
3. The **repository-wide `[VERIFY]` count** was searched for and not found as a claimed number
   (section 9). If one exists somewhere I did not look, it is now one too high.
