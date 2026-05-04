# H100 FL+CA+TX Per-Visit Integration Prompt

Paste this to the next H100 / local agent session. The compute is **already
done**; this prompt covers only the integration step: pulling the FL+CA+TX
result artifacts from the compute branch into `worktree-check2hgi-mtl` and
closing the D2 deferred item.

---

You are working on the **Check2HGI MTL study** on branch
`worktree-check2hgi-mtl`.

## 1 · What is already done

The FL+CA+TX per-visit counterfactual (CH19) ran successfully on branch
`h100/pervisit-fl-ca-tx-results` (commit `a858177`, 2026-05-04). Nine cells
(3 states × {canonical, POI-pooled, HGI}), matched-head `next_gru` STL,
seed=42, 5f × 50ep, bs=1024, H100 80 GB. Per-fold JSONs, a 5-state summary
doc, and an updated `per-visit.png` are all committed to that branch.

**Results (5-state confirmed CH19):**

| State | C2HGI | Pooled | HGI | total gap | per-visit pp | training-sig pp | per-visit % |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL | 40.76 ± 1.50 | 29.57 | 25.26 ± 1.06 | +15.50 | +11.19 | +4.31 | **72%** |
| AZ | 43.17 ± 0.28 | 34.09 ± 0.63 | 28.99 ± 0.51 | +14.18 | +9.08 | +5.10 | **64%** |
| FL | 63.48 ± 1.04 | 37.42 ± 0.76 | 34.46 ± 0.97 | +29.02 | +26.06 | +2.96 | **90%** |
| CA | 60.55 ± 0.81 | 34.47 ± 0.44 | 31.14 ± 1.00 | +29.41 | +26.08 | +3.33 | **89%** |
| TX | 60.35 ± 0.30 | 34.93 ± 0.71 | 32.19 ± 0.61 | +28.16 | +25.42 | +2.74 | **90%** |

AL/AZ from prior runs. Per-visit share ~89–90% at large states (FL/CA/TX)
vs 64–72% at small states (AL/AZ): **two-band pattern**.

**D3 (AZ Wilcoxon)** is already closed. `mechanism.tex §6.1` already reads
"Both per-visit and training-signal deltas are paired-Wilcoxon positive at
each state ($p=0.0312$, $5/5$ folds, $n=5$ ceiling), so AZ replicates AL
inferentially rather than descriptively." No changes needed to D3.

## 2 · Read first

Before touching any file, read:

1. `docs/studies/check2hgi/results/CH19_PERVISIT_5STATE_SUMMARY.md` — on
   branch `h100/pervisit-fl-ca-tx-results`. Use `git show` to read it.
2. `articles/[BRACIS]_Beyond_Cross_Task/REVIEW.MD` (current branch) —
   D2 section starting at line ~129.
3. `articles/[BRACIS]_Beyond_Cross_Task/src/sections/mechanism.tex` —
   the `\subsection{Per-visit context counterfactual (AL + AZ)}` block.

## 3 · Scope — exactly four steps

Do **only** these four steps. Do not re-run any training, do not touch
table files (T1–T4), and do not rewrite sections other than `mechanism.tex
§6.1`.

### Step 1 — Cherry-pick the 9 per-fold JSONs

```bash
git checkout origin/h100/pervisit-fl-ca-tx-results -- \
  docs/studies/check2hgi/results/phase1_perfold/FL_check2hgi_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/FL_check2hgi_pooled_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/FL_hgi_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/CA_check2hgi_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/CA_check2hgi_pooled_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/CA_hgi_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/TX_check2hgi_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/TX_check2hgi_pooled_cat_gru_5f50ep_20260504.json \
  docs/studies/check2hgi/results/phase1_perfold/TX_hgi_cat_gru_5f50ep_20260504.json \
  scripts/probe/extract_pervisit_perfold.py
```

The `extract_pervisit_perfold.py` pick-up carries a `RUNDIR_RE` bugfix
committed on the h100 branch.

### Step 2 — Render the 5-panel figure

```bash
python3 "articles/[BRACIS]_Beyond_Cross_Task/src/figs/render_per_visit_5state.py" \
  --shared-y \
  --out "articles/[BRACIS]_Beyond_Cross_Task/src/figs/per-visit.png"
```

The renderer auto-discovers the JSONs in
`docs/studies/check2hgi/results/phase1_perfold/` by prefix. It falls back to
AL+AZ if FL/CA/TX files are missing — the output filename will tell you
which mode fired. If the script errors, do **not** touch the existing
`per-visit.png`; stop and report.

After rendering, visually verify the output PNG has 5 panels (AL, AZ, FL,
CA, TX) with three bars each (HGI / POI-pooled / canonical) and per-visit /
training-signal brackets. The per-visit share annotations should read ~72%,
~64%, ~90%, ~89%, ~90%.

### Step 3 — Update `mechanism.tex §6.1`

The subsection currently covers only AL+AZ. Extend it to report the
five-state two-band pattern. Exact changes:

**3a. Subsection title (line 6 of `mechanism.tex`):**

```latex
% Before:
\subsection{Per-visit context counterfactual (AL + AZ)}

% After:
\subsection{Per-visit context counterfactual (five states)}
```

**3b. Body paragraph** — replace the current single paragraph (lines 9–18
in `mechanism.tex`) with the expanded five-state version below. Keep the
`\begin{figure}` block unchanged:

```latex
To isolate per-visit context, we compare canonical Check2HGI against
POI-pooled Check2HGI (canonical vectors mean-pooled per POI and applied
uniformly to every visit, killing per-visit variation while preserving the
training signal). Canonical vs.\ pooled isolates per-visit context;
pooled vs.\ HGI isolates the contrastive training signal.

Under the matched-head GRU single-task ceiling, the substrate gap
(canonical $-$ HGI) splits consistently at all five states
(Figure~\ref{fig:per-visit}): per-visit context dominates at every state,
with both components positive at every state. Paired Wilcoxon signed-rank
tests reach the $n=5$ ceiling at AL and AZ ($p=0.0312$, $5/5$ folds
positive each); the FL/CA/TX pattern is consistent with the same
monotone ordering at all five per-fold pairs. A two-band pattern emerges
across state scale: at small states (AL ${\sim}72\,\%$, $+11.19$ pp of
$+15.50$; AZ ${\sim}64\,\%$, $+9.08$ of $+14.18$) the training-signal
residual is 28--36\,\%; at large states (FL ${\sim}90\,\%$, $+26.06$ pp
of $+29.02$; CA ${\sim}89\,\%$, $+26.08$ of $+29.41$; TX ${\sim}90\,\%$,
$+25.42$ of $+28.16$) the pooled embeddings collapse almost entirely to
the HGI level, leaving ${\sim}10\,\%$ to the training signal. The pattern
is consistent with large transition graphs (4.7k--8.5k regions) providing
weaker per-POI contrastive signal, so Check2HGI's advantage at large
states is almost entirely in the per-visit contextual variation it
introduces. A head-free linear probe at AL gives a comparable per-visit
share (${\sim}63\,\%$; the GRU head amplifies the share rather than
creating it).
```

**3c. Figure caption** — update to describe all five panels:

```latex
\caption{Per-visit context decomposition at all five states under the
matched-head GRU single-task ceiling. The substrate gap (canonical
Check2HGI $-$ HGI) splits into a per-visit component (canonical $-$
POI-pooled) and a training-signal component (POI-pooled $-$ HGI).
At small states (AL, AZ) per-visit accounts for ${\sim}64$--$72\,\%$
of the gap; at large states (FL, CA, TX) it accounts for
${\sim}89$--$90\,\%$, reflecting reduced marginal value of
graph-contrastive signal when the transition graph is sparse.}
```

**3d. `\includegraphics` width** — the 5-panel figure needs more horizontal
space. Change `0.72\linewidth` to `\linewidth`:

```latex
\includegraphics[width=\linewidth]{figs/per-visit.png}
```

After editing, run `pdflatex` once to confirm the paper compiles and the
figure renders correctly. Check the page count — the paper must stay at or
under 15 pages. If it overflows, reduce `\includegraphics` back to
`0.9\linewidth` and re-check; if still over 15 pages, stop and report.

### Step 4 — Close D2 in REVIEW.MD

In `articles/[BRACIS]_Beyond_Cross_Task/REVIEW.MD`, find the D2 section
(currently starting with `## D2. 5-state per-visit decomposition figure`)
and replace its heading with:

```markdown
## D2. 5-state per-visit decomposition figure — **CLOSED (2026-05-04)**
```

Then replace the body of D2 with a one-sentence closure note:

```markdown
Compute landed on `h100/pervisit-fl-ca-tx-results` (commit `a858177`). Nine
cells cherry-picked into `worktree-check2hgi-mtl`; 5-panel `per-visit.png`
regenerated; `mechanism.tex §6.1` updated to cover all five states with the
two-band pattern (~72/64% at AL/AZ, ~89–90% at FL/CA/TX). See
`docs/studies/check2hgi/results/CH19_PERVISIT_5STATE_SUMMARY.md` for the
full audit trail.
```

## 4 · Study docs to update (optional, low priority)

If the paper compiles cleanly and D2 is closed, update:

- `docs/studies/check2hgi/results/RESULTS_TABLE.md` — add the §0.7 block
  from `h100/pervisit-fl-ca-tx-results` (available via `git show
  origin/h100/pervisit-fl-ca-tx-results:docs/studies/check2hgi/results/RESULTS_TABLE.md`
  — copy the `## 0.7 · CH19 per-visit 5-state` section only; do not
  overwrite §0.1–§0.6).
- `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md` — update the CH19 entry
  status line to read "five-state confirmed 2026-05-04". Do not touch any
  other CH entries.

These updates are not blocking the paper commit. Skip them if they add risk
or scope.

## 5 · Commit

Stage only:

```
articles/[BRACIS]_Beyond_Cross_Task/src/figs/per-visit.png
articles/[BRACIS]_Beyond_Cross_Task/src/sections/mechanism.tex
articles/[BRACIS]_Beyond_Cross_Task/REVIEW.MD
docs/studies/check2hgi/results/phase1_perfold/FL_*.json
docs/studies/check2hgi/results/phase1_perfold/CA_*.json
docs/studies/check2hgi/results/phase1_perfold/TX_*.json
scripts/probe/extract_pervisit_perfold.py
```

Plus the optional study-doc files if updated. Use commit message:

```
feat(check2hgi): D2 closed — 5-state per-visit figure + §6.1 two-band update
```

Do not commit the PDF (`out/samplepaper.pdf`). Do not commit any training
result dirs from `results/`.

Push to `origin/worktree-check2hgi-mtl`.

## 6 · Hard stops

Stop and report before proceeding if:

- The 5-panel PNG renders with fewer than 5 panels (fall-back triggered)
- `pdflatex` fails or the PDF exceeds 15 pages after Step 3
- Any per-fold JSON from the h100 branch is missing or has `null` fold values
- The per-visit shares in the rendered figure deviate from the table above
  by more than 2 pp (suggests a wrong JSON was picked up)

## 7 · Deliverable

Return with:

- Whether all 9 JSONs cherry-picked cleanly
- Whether the 5-panel PNG rendered with the expected per-visit shares
- Whether `pdflatex` compiled successfully and the page count
- Whether D2 is closed in REVIEW.MD
- Any hard-stop conditions encountered
