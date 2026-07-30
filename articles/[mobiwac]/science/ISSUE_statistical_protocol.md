# ISSUE — the statistical protocol claim does not match the executed analysis

> **What this file is.** A grounded problem report for the **MobiWac 2026 — Codebase & Results
> Audit** Claude Science project (set up per [`science.md`](science.md)), plus the prompt to launch
> the audit-and-remedy pass. The finding below was raised by the dissertation's external review as
> **REV-007 — "Statistical protocol and artifact trail are not internally synchronized"** (Major,
> Open; reported by reviewers 04, 06, 07, 09, 10, and 12) in
> [`articles/dissertacao/src_utils/dissertation_review.md`](../../dissertacao/src_utils/dissertation_review.md)
> — that entry is the **source of record** for the finding (it also carries the author response and
> the related REV-001 leakage finding). Independently re-verified against this repo on 2026-07-25,
> it lands on **both** the MobiWac paper and dissertation Chapter 5.
>
> **Read this as a lead, not as settled fact.** The project's fail-closed rule applies to this file
> too: every claim below names its artifact so the audit can re-derive it. Where this report and
> the repo disagree, the repo wins and the report is the finding.
>
> Meta-documentation: exempt from the writing law itself; any edit it produces to
> `articles/[mobiwac]/src/` is not.

---

## 1 · TL;DR

The paper tells the reader that its statistical plan was fixed in advance. The pre-registered
protocol in this repo fixed **less** than the paper implies, and the executed analysis departed
from it in two documented ways. No number and no verdict is in question; the **epistemic labels**
are.

Three distinct defects, in increasing severity:

| ID | Defect | Where | Severity |
|----|--------|-------|----------|
| **D1** | Pairing level and test family deviate from the pre-registration (per-fold n=20 Wilcoxon → per-seed n=4 paired t). Logged in the repo; **not disclosed in the paper**. | `05_setup.tex` §5.3 | Low |
| **D2** | **Region superiority was never pre-registered.** The registered region test is non-inferiority (TOST) only; the four "outperforms" region claims are post-hoc and sit in **no multiplicity family**. | `05_setup.tex` §5.3, `06_results.tex` §6.2, abstract, `08_conclusion.tex` | **Real** |
| **D3** | The sentence "fixed in an analysis plan during development" **overstates the record**: the plan fixed the assignment *per task*, not per dataset, and did not cover region superiority. | `05_setup.tex` §5.3 (and the mirrored text in dissertation Ch.5) | **Real** |

---

## 2 · Evidence trail (verified 2026-07-25; re-verify, do not trust this file)

**2.1 What the pre-registration actually fixed** —
`docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md`

- §1, family table, family (A) (the headline family):
  > `cat: superiority (MTL > STL); reg: **non-inferiority** (MTL not worse than STL by more than δ_reg)`
  > `cat → paired Wilcoxon (§2); reg → **TOST** (§3)`
- §2 (superiority test):
  > "Paired Wilcoxon signed-rank on the matched per-fold Δs, **multi-seed pooled** (n=20 = 4 seeds × 5 folds)"
- §5.2 (multiplicity):
  > "The headline family is small and fixed: {6 states} × {cat superiority, reg non-inferiority}. Apply
  > Holm-Bonferroni **within the cat-superiority set** (6 states) … **TOST cells are equivalence tests,
  > not superiority tests, and are not pooled into the cat Holm family.**"

⇒ **There is no pre-registered region-superiority family anywhere in the protocol.**
⚠ Related lead: `m1_stats_n20.py` prints the phrase *"the pre-registered reg-'beats' family"* — that
label appears unsupported by the protocol document and should itself be audited.
⚠ Path note: REV-007 cites `docs/studies/closing_data/STATISTICAL_PROTOCOL.md`; the file found on
2026-07-25 is under `v17_completion/`. Confirm there is exactly one protocol of record.

**2.2 What was executed** —
`docs/studies/closing_data/v17_completion/stats_n20/RESULTS.md` §"Deviation log (protocol §8)"

1. *"Seed-level pairing (n=4) instead of the pre-registered per-fold n=20."* Reason: the MTL
   per-fold matched-score sidecars were not in the committed tree (A40 rundirs gitignored), so only
   per-seed means were available to pair.
2. *"Paired t reported alongside the pre-registered Wilcoxon."* Reason: at n=4 the exact one-sided
   Wilcoxon's minimum attainable p is 1/2⁴ = **0.0625 > α**; all four category cells sat exactly at
   that floor with 4/4 positive.
3. *"Holm applied to the paired-t family"* (the Wilcoxon footing is floor-locked).

**2.3 What the paper currently says** — `articles/[mobiwac]/src/sections/05_setup.tex:42`

> "We fix the assignment in advance, before reading any results: where the joint model was expected
> to outperform, we test *superiority*; where it was expected only to match the dedicated model, we
> test *non-inferiority*. The assignment and the margin were fixed in an analysis plan during
> development and are released with the code; superiority is tested with a paired $t$ on the
> per-seed means …"

The first clause implies a **per-dataset** a-priori assignment (which the protocol does not record
for region); the third asserts the plan is released (so a reviewer can read §2/§5.2 and see the
mismatch). The Wilcoxon→t departure is not mentioned in the paper.

**2.4 Provenance of the finding**

- **REV-007 — "Statistical protocol and artifact trail are not internally synchronized"**
  (Severity: Major; Status: Open; Classification: Confirmed inconsistency; reported by reviewers 04,
  06, 07, 09, 10, 12), in
  [`articles/dissertacao/src_utils/dissertation_review.md`](../../dissertacao/src_utils/dissertation_review.md)
  (§3 Major Issues; the heading is at line 583 as of 2026-07-25 — locate it by title, line numbers
  drift). Its finding, verbatim: *"The dissertation describes a Wilcoxon-based protocol fixed in
  advance, but the final n=4 seed-level analysis uses paired t-tests after noting that the minimum
  exact Wilcoxon p-value is 0.0625. This deviation is documented in a results note but not candidly
  explained in the dissertation. Region superiority is also post-hoc relative to the registered
  non-inferiority framing."* Its why-it-matters: *"The point estimates can be traced, but the
  confirmatory status, multiplicity family, and authoritative statistical artifact cannot be
  reconstructed cleanly. 'Fixed in advance' overstates the record."* Its recommended action is R4
  below. The entry also carries the **author response** (the authority chain: `RESULTS_BOARD.md` →
  `v17_completion/stats_n20/RESULTS.md` → `joint_best/`), which the audit should read before
  proposing anything.
- The same concern was raised earlier, internally, by the MobiWac review panel's stats-skeptic
  persona (must-fix: *"provide evidence for the a priori test assignment, and explain why Istanbul
  — fewest regions — was assigned superiority"*). That item was never ruled on, and the
  "analysis plan" clause was subsequently added to §5.3, strengthening the claim the panel had
  questioned.

---

## 3 · What is NOT in question (calibration — do not over-correct)

The region gains are **statistically supported on the reported footing**: 90 % CIs of the paired
difference lie entirely above zero (Istanbul +0.15…+0.23, FL +0.67…+0.76, TX +2.10…+2.13,
CA +2.19…+2.21), with 20/20 folds positive, and TOST non-inferiority also passes at all six.
Category superiority clears Holm at every dataset (worst adjusted p ≈ 1e-06). Verify these against
the board before relying on them.

The seed-level (n=4) pairing is arguably the **more conservative and defensible** footing, not a
weaker one: the five folds are one fixed partition shared across seeds, so pooling 20 fold-level Δs
as independent pairs risks pseudo-replication. The deviation was forced by artifact availability
but landed on the safer choice. **Any remedy must not read as an admission that the numbers are
wrong — they are not.**

---

## 4 · Constraints any remedy must respect

- **Verb law** (`GLOSSARY.md` §1 honesty rule): "outperforms" bound to a superiority result,
  "matches"/non-inferior bound to TOST; **never upgrade Arizona**; the scaling claim stays scoped
  to the five U.S. states.
- **Decisions ledger** (`articles/[mobiwac]/CLAUDE.md` §3): the abstract's softened TOST wording,
  the verdict-verb ruling, the FL "no materiality caveat" ruling. Reopening a ledger row needs the
  author, explicitly.
- **Page budget: 8 pages, no fee, currently flush.** Any added sentence must be funded by a trim in
  the same pass, or proposed as a swap.
- **Cross-artifact consistency:** whatever is decided must apply to the paper §5.3/§6.2 *and*
  dissertation Chapter 5, which mirrors the wording.
- The paper is **under review** (EDAS #1571313639); edits land in the revision/camera-ready, so the
  remedy should also be stated in a form usable in a response letter.

---

## 5 · Candidate remedies (for the audit to evaluate, extend, and cost — not a decision)

- **R1 — Minimal honesty repair (current author-side lean).** Rewrite §5.3 to state what the plan
  actually fixed (task-level assignment: category → superiority, region → non-inferiority; plus the
  margin), disclose the Wilcoxon→t departure with its one-line reason, and label the region gains
  as secondary observations outside the registered family. ~1.5–3 lines; needs funding.
- **R2 — Run the pre-registered test.** Re-run the per-fold n=20 paired Wilcoxon on the joint-best
  arrays and report it (it clears α on its own where it has been run: p ≈ 9.5e-07, 20/20 positive).
  Blocked at ≥1 cell: Istanbul's per-fold category ceiling is not in the committed tree. Trades a
  documented conservative deviation for a pseudo-replication objection.
- **R3 — Report both footings.** Seed-level as headline, per-fold as corroboration in one clause.
- **R4 — Full REV-007 remedy** (the action REV-007 itself recommends, quoted): *"Create one
  immutable analysis manifest that names the exact input files, seeds, fold aggregation, tests,
  multiplicity families, deviations, and generated tables. Regenerate all reported intervals and
  p-values from that manifest. Label deviations and post-hoc tests explicitly."* Correct answer for
  the dissertation and camera-ready; a work item, not a text patch.
- **R5 — Drop the region "outperforms" claims.** Not recommended: understates a real result and
  collides with the verb law; listed for completeness.

---

## 6 · Prompt for Claude Science (paste as a message in the MobiWac audit project)

```
Audit and remedy a statistical-protocol integrity finding in the MobiWac 2026 paper. Obey the
project custom instructions (fail-closed number protocol, writing law, canonical names, decisions
ledger). Everything you need is internal to this repo — do not search external literature.

CONTEXT: articles/[mobiwac]/science/ISSUE_statistical_protocol.md states the finding. Treat that
file as a LEAD, not as evidence: re-derive every one of its claims from the artifacts yourself, and
report any place where the file is wrong, overstated, or incomplete. It was written from a
conversation, so it is exactly the kind of source this project is built to distrust.

The finding originates as REV-007 ("Statistical protocol and artifact trail are not internally
synchronized", Major/Open) in articles/dissertacao/src_utils/dissertation_review.md, §3 Major
Issues — read that entry in full first, including its author response (which names the authority
chain) and the review's own "where the review is strongest and weakest" section, which warns that
its file-level assertions are leads to verify, not established facts. Also read REV-001 there for
context on what else that review disputes about Chapter 5, but do NOT act on REV-001 in this pass.

STEP 0 — Orient: articles/[mobiwac]/CLAUDE.md (§2b especially), PAPER_PLAN.md §3, GLOSSARY.md,
ERRATA.md, docs/studies/closing_data/RESULTS_BOARD.md (§1/§3), joint_best/JOINT_BEST_SCORING.md.

STEP 1 — Establish what was PRE-REGISTERED. Read the statistical protocol of record
(docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md; first confirm whether any other
protocol file exists and which is authoritative). Answer, quoting the document:
  (a) exactly which hypothesis families were registered, per task and per dataset;
  (b) the registered test, pairing level, and n for each;
  (c) the registered multiplicity families and what is explicitly excluded from them;
  (d) whether region superiority is registered anywhere;
  (e) what the deviation log records, where, and whether the log is part of the released bundle
      the paper points at ("released with the code").

STEP 2 — Establish what was EXECUTED. From stats_n20/RESULTS.md, the joint-best records, and the
generator scripts (superiority_wilcoxon.py, region_match_tost.py, m1_stats_n20.py,
score_joint_best.py): the actual test, pairing level, n, and correction applied to every cell the
paper reports, on the joint-best convention that Table III now uses. Flag any script text that
asserts a registration the protocol does not contain (see the m1_stats_n20.py lead in the issue
file).

STEP 3 — Establish what the PAPER claims. From articles/[mobiwac]/src/: every sentence asserting a
test, a pre-registration, a correction, or a significance verdict (abstract, §5.3, §6.2, §6.3, §7,
§8, Table III caption/footnote). Quote each with file:line.

STEP 4 — Reconcile. For each paper sentence: MATCH / MISMATCH / STALE / UNSUPPORTED / CANNOT
VERIFY, with the artifact path and key that decides it. Answer explicitly:
  - Is any reported number affected? (Expected: no — confirm or refute.)
  - Are the four region "outperforms" claims inside any registered or applied correction family?
  - Does the paper's "fixed in advance" sentence overstate what the protocol fixed?
  - Is the Wilcoxon→paired-t departure disclosed anywhere the reader can see?

STEP 5 — Feasibility of the pre-registered test today. Determine, from the committed tree only,
for which datasets a per-fold n=20 paired Wilcoxon can be run on BOTH arms under the joint-best
convention (the MTL side and the dedicated-ceiling side). Name the exact missing artifacts for any
dataset where it cannot, and say what it would cost to obtain them. Do not fabricate a partial
family: if it cannot be run at all six, say so.

STEP 6 — Remedy options. Evaluate R1-R5 in the issue file, add any option they missed, and for the
one you recommend produce EXACT replacement text for the affected paper sentences, obeying the
writing law (canonical names, verbs bound to tests, no em-dash, no contractions, every number with
its reference point and convention) and the decisions ledger. State the line cost of each option
against the 8-page budget, and name the trim that funds it. Provide the parallel wording for
dissertation Chapter 5 and a two-to-four-sentence version usable in a response letter.

CONSTRAINTS: do not weaken a supported finding into speculation; do not upgrade Arizona; do not
reopen a ledger row silently (flag it for the author instead); numbers are quoted from artifacts,
never recomputed by hand.

DELIVERABLES: (a) the pre-registered-vs-executed-vs-claimed reconciliation table; (b) verdicts on
the four questions in STEP 4; (c) the STEP 5 feasibility statement; (d) the recommended remedy with
exact text, line cost, and funding trim; (e) a [VERIFY] list of anything you could not resolve from
the artifacts. Do NOT edit any file in this pass — deliver the report and wait for approval.
```

---

## 7 · Acceptance criteria for the fix (whatever remedy is chosen)

1. Every statement in the paper about pre-registration is true of the protocol document as written.
2. The confirmatory status of each claim is legible: which tests were registered, which are
   secondary/post-hoc, and what correction (if any) applies to each family.
3. Any departure from the registered test is visible to the reader, with its reason, in one clause.
4. No number, verdict verb, or ledger ruling changes without the author's explicit approval.
5. The paper stays at 8 pages, and dissertation Chapter 5 carries the same account.
6. The record itself is reconstructible: the released bundle the paper points at contains the
   protocol **and** its deviation log.
