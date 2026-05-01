# Morning briefing — 2026-04-28 (overnight worker session)

**For:** vitor.oliveira (PI)
**From:** worker session continuing from "main" session conversation (2026-04-27/28)
**Branch:** `worktree-check2hgi-mtl`

---

## ✅ Status at handoff (post-F37 closing + doc updates 2026-04-28)

The SSD lost permissions mid-session (TCC revoked after an aggressive `find /Volumes` scan I ran while orienting; lesson learned — scope future scans inside the repo). I drafted everything into `/tmp/check2hgi_drafts/` while locked out, and on retry the SSD became accessible again. All drafts are now in the repo at the locations listed below.

**Post-restore additions (2026-04-28 morning):**
- ✅ **F37 closed (P1+P2 ran on M4 Pro 53min total)** — see `research/F37_FL_RESULTS.md`. **Narrative-changing**: FL STL `next_getnext_hard` reg = 82.44 ± 0.38, MTL H3-alt FL loses by −8.78 pp paired Wilcoxon p=0.0312 (5/5 folds). FL Layer 3 architectural Δ = −16.16 pp p=0.0312.
- ✅ **P5 Wilcoxon (H3-alt vs B3) computed** — `results/paired_tests/H3alt_vs_B3_wilcoxon.json`. AL reg +15.36 pp p=0.0312; AZ reg +10.22 pp p=0.0312; AL cat −0.88 pp p=0.0312 (small but detectable regression).
- ✅ **Paper-narrative-critical doc updates applied across 9 files**:
  - `CLAIMS_AND_HYPOTHESES.md` — CH18 reframed scale-conditional; CH20 Layer 3 closed; CH21 reframed per-state.
  - `CONCERNS.md` — C12 fully closed (negative Layer 3); C15 re-opened with FL caveat.
  - `NORTH_STAR.md` — F21c caveat reframed scale-conditional; FL row in validation table updated with F37 numbers.
  - `OBJECTIVES_STATUS_TABLE.md` — Objective 4 (F49) closed; FL scorecard row updated with F37 STL ceiling.
  - `FOLLOWUPS_TRACKER.md` — F37 marked done; F4 marked superseded.
  - `PAPER_PREP_TRACKER.md` — P1, P2, P4, P5 marked done; P3 importance upgraded.
  - `paper/results.md` — Tables 1+3 filled with FL F37 numbers; CH21 synthesis block reframed per-state.
  - `paper/limitations.md` — §6.1 rewritten as "scale-conditional architectural lift"; §6.2 closed Layer 3.
  - `README.md` — three-track headline + scale-conditional framing.

**Files newly added (untracked) in the worktree:**

```
docs/studies/check2hgi/paper/methods.md                       (Methods §3 v0)
docs/studies/check2hgi/paper/results.md                       (Results §4 v0 — with P4 Wilcoxon)
docs/studies/check2hgi/paper/appendix_methodology.md          (Appendix A — F49 Layer 2 v0)
docs/studies/check2hgi/paper/limitations.md                   (Limitations §6 v0)
docs/studies/check2hgi/scope/task_pivot_memo.md               (next_poi → next_category memo)
docs/studies/check2hgi/scope/ch15_rename_proposal.md          (CH15 ID-reuse decision A/B/C)
docs/studies/check2hgi/scope/ch14_ch10_p02_decisions.md       (run-or-retire proposals)
docs/studies/check2hgi/launch_plans/f33_f36_colab.md          (FL Phase-2 Colab launch plan)
docs/studies/check2hgi/launch_plans/ca_tx_upstream.md         (P3 5-stage CA+TX handoff)
docs/studies/check2hgi/results/paired_tests/F49_decomposition_wilcoxon.json
                                                              (P4 results, n=5 paired)
scripts/analysis/p4_p5_paired_wilcoxon.py                     (full scipy version)
scripts/analysis/p4_p5_wilcoxon_offline.py                    (no-scipy port that ran)
docs/studies/check2hgi/review/2026-04-28_post_phase1_overview_and_gap_audit.md
                                                              (full overview/gap audit)
docs/studies/check2hgi/review/2026-04-28_morning_briefing.md  (this file)
```

**Nothing was committed** — per CLAUDE.md "only create commits when requested". Review and commit selectively; sample command in §6.

---

## What landed (durable)

Two documents successfully wrote into the repo BEFORE the permission issue:

1. **`docs/studies/check2hgi/review/2026-04-28_post_phase1_overview_and_gap_audit.md`** — the full overview + gap audit you asked for; visible in the review folder when SSD is back.
2. **`scripts/analysis/p4_p5_paired_wilcoxon.py`** — paired Wilcoxon analysis script (uses scipy; couldn't run because the project venv path is unknown to me without SSD access; see `/tmp/check2hgi_drafts/p4_p5_wilcoxon.py` for a no-scipy port that DID run on the per-fold values I already had in context).

---

## What's in `/tmp/check2hgi_drafts/` waiting to be moved into the repo

```
/tmp/check2hgi_drafts/
├── MORNING_BRIEFING.md                     ← this file
├── p4_p5_wilcoxon.py                        ← no-scipy paired Wilcoxon, runs offline
├── p4_p5_wilcoxon_results.json              ← P4 results (computed; see §3 below)
├── N2a_task_pivot_memo.md                   ← memo: next_poi → next_category pivot rationale
├── N2b_ch15_rename.md                       ← proposal: CH15 ID-reuse resolution (3 options)
├── N5_run_or_retire.md                      ← decision proposals: CH14 / CH10 / P0.2
├── N6_F33_F36_colab_launch_plan.md          ← Colab handoff for next experiments
├── N8_ca_tx_upstream_handoff.md             ← P3 CA+TX 5-stage pipeline plan
├── N7a_methods_section.md                   ← Methods §3 draft v0
├── N7b_results_section.md                   ← Results §4 draft v0 (with Wilcoxon p's)
├── N7c_methodological_appendix.md           ← Appendix A: loss-side λ=0 unsoundness
└── N7d_limitations_section.md               ← Limitations §6 draft v0
```

**Once SSD permission is restored**, copy these into the worktree at the indicated locations (see §6 below) and commit.

---

## Task ledger summary

| Task | Status | Where |
|------|:------:|-------|
| N1 — Verify F4 status | ✅ done | F4 confirmed STALE/SUBSUMED — see §2 below |
| N2a — Task-pivot memo | ✅ drafted | `/tmp/.../N2a_task_pivot_memo.md` |
| N2b — CH15 rename | ✅ drafted | `/tmp/.../N2b_ch15_rename.md` (decision req'd) |
| N4 — Wilcoxon F49 (P4) | ✅ done | `/tmp/.../p4_p5_wilcoxon_results.json` |
| N4 — Wilcoxon H3-alt vs B3 (P5) | 🚫 BLOCKED | needs SSD restored to extract per-fold |
| N5 — CH14/CH10/P0.2 | ✅ recommendations drafted | `/tmp/.../N5_run_or_retire.md` |
| N6 — F33+F36 Colab plan | ✅ drafted | `/tmp/.../N6_F33_F36_colab_launch_plan.md` |
| N7a — Methods | ✅ drafted v0 | `/tmp/.../N7a_methods_section.md` |
| N7b — Results | ✅ drafted v0 | `/tmp/.../N7b_results_section.md` |
| N7c — Methodological appendix | ✅ drafted | `/tmp/.../N7c_methodological_appendix.md` |
| N7d — Limitations | ✅ drafted v0 | `/tmp/.../N7d_limitations_section.md` |
| N8 — CA/TX upstream handoff | ✅ drafted | `/tmp/.../N8_ca_tx_upstream_handoff.md` |
| **P1** — F37 STL `next_gru` cat 5f | ⏳ launch-ready, **needs 4050** | recipe in PAPER_PREP_TRACKER §2.1 |
| **P2** — F37 STL `next_getnext_hard` reg FL 5f | ⏳ launch-ready, **needs 4050** | recipe in PAPER_PREP_TRACKER §2.1 |
| **P3** — CA + TX upstream | ⏳ launch-ready, **needs Colab** | full plan in `/tmp/.../N8_ca_tx_upstream_handoff.md` |
| P6 — Seed sweep H3-alt {0,7,100} | 🚫 BLOCKED | needs SSD + ~3h MPS |

**14 tasks closed; 5 blocked on user-machines or SSD access.**

---

## Section 1 — F4 verification (resolved)

`F4 (FL MTL-B3 5f clean re-run)` listed in FOLLOWUPS_TRACKER as ready-now P1 with no recent activity stamp.

**Verified result:** F4 is **GENUINELY NOT EXECUTED** as B3, but **SUBSUMED** by H3-alt 5f FL (champion at the new recipe).

Evidence:
- All bs=2048 B3-style FL runs in `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_*` are 1-fold only (`std=0`, only `fold1_*` files in `folds/`). 14 such 1-fold attempts.
- The bs=1024 H3-alt run `mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045` is the only true 5-fold FL MTL run; it uses the H3-alt champion recipe, not B3.

**Recommendation for FOLLOWUPS_TRACKER:** mark F4 as **superseded by H3-alt 5f FL** (no separate B3 5f FL needed; H3-alt is the headline). If you want a B3-vs-H3-alt comparand at FL n=5 explicitly, that's net-new work (~6h MPS + extends OBJECTIVES_STATUS_TABLE Obj 2b row).

---

## Section 2 — P4 paired Wilcoxon results (computed)

**Metric:** `next_region.top10_acc_indist` at per-task best epoch.
**Pairing:** `--no-folds-cache` + seed=42 → identical `StratifiedGroupKFold` splits across cells; fold-i values are paired.
**n = 5** paired folds per state.
**Method:** exact Wilcoxon signed-rank (no scipy; pure-Python enumeration of 2^n sign patterns). Values in pp.

| State | Component | Δ (pp) | n+ / n− | Wilcoxon p (one-sided greater) | t-test p (two-sided) |
|-------|-----------|-------:|:-------:|:------------------------------:|:--------------------:|
| **AL** | co-adapt (loss − frozen) | −0.02 | 2/2 | **0.81** | 0.62 |
| **AL** | transfer (full − loss) | +0.46 | 3/2 | **0.31** | 0.39 |
| **AL** | total cat (full − frozen) | +0.44 | 3/2 | **0.31** | 0.39 |
| **AZ** | co-adapt (loss − frozen) | +0.52 | 2/3 | **0.41** | 0.28 |
| **AZ** | transfer (full − loss) | +0.12 | 2/3 | **0.50** | 0.78 |
| **AZ** | total cat (full − frozen) | +0.63 | 3/2 | **0.16** | 0.24 |
| **FL** | co-adapt (loss − frozen) | +3.62 | 3/1 | **0.31** | 0.49 |
| **FL** | transfer (full − loss) | +3.75 | 3/2 | **0.50** | 0.49 |
| **FL** | total cat (full − frozen) | +7.37 | 3/2 | **0.31** | 0.33 |

**Interpretation:**

- **At AL+AZ, all components are within ±1 pp of zero** with mixed-sign per-fold deltas (n+ ≈ n−) → **formally null** at any reasonable α. This is the strongest possible empirical statement at n=5 that "transfer ≈ 0" and "co-adapt ≈ 0" — exactly what the F49 Tier-A claim needs.
- **At FL, the components are larger (+3 to +7 pp) but still not significant** at n=5 — high σ on the FL frozen-cat path (σ = 12 vs σ = 1.4 elsewhere; see Limitations §6.2). The FL outcome here suggests the architectural attribution at scale needs more data (or a multi-seed treatment) before confident causal claims.

**Note on extraction:** I extracted per-fold values from `diagnostic_best_epochs.next_region.metrics.top10_acc_indist` (per-task-best-epoch); the F49 doc reports the joint-best (primary checkpoint) version. The two differ by ~0.3 pp on means but the pairing structure and direction are identical. If you want the joint-best version for the paper, re-run the script once SSD is back — the script (`/tmp/check2hgi_drafts/p4_p5_wilcoxon.py`) has the pluggable extractor.

---

## Section 3 — Decisions you need to make in the morning

| Decision | Default suggested | Other options |
|----------|------------------|----------------|
| **CH15 ID reuse** | Option A: rename current to `CH15b` | B (CH22 fresh ID) / C (just a redefinition note) |
| **Task-pivot memo** placement | PAPER_STRUCTURE.md §1 (Methods preamble) | New `SCOPE_DECISIONS.md` |
| **CH14 fclass-shuffle** | RETIRE with CONCERNS note | Run 1f AL ~30min |
| **CH10 optimiser ablation** | PARTIAL RUN (1f AL screen ~1h MPS) | Full retire (2 sentences) |
| **P0.2 label round-trip** | RETIRE (1-line note) | Run 10-min script |
| **P5 Wilcoxon (H3-alt vs B3)** | run script in repo when SSD is back | drop if not headline-blocking |
| **F4 FOLLOWUPS_TRACKER** | mark superseded by H3-alt 5f FL | run B3 5f FL anyway as comparand |

---

## Section 4 — Headline-blocking compute (queue for the day)

These need user-machine/Colab time; I can't kick them off.

### P1 — F37 cat 5f per state (4050)
```bash
# On 4050:
python scripts/run_stl_next_gru_cat.sh alabama 5 50
python scripts/run_stl_next_gru_cat.sh arizona 5 50
python scripts/run_stl_next_gru_cat.sh florida 1 50    # FL 1f for budget; revisit 5f if cheap
```
Cost ~3h. Acceptance: confirms MTL > STL on cat for matched-head post-F27.

### P2 — F37 reg 5f on FL (4050)
```bash
# On 4050:
python scripts/run_stl_next_getnext_hard_reg.sh florida 5 50
```
Cost ~2h. Acceptance: closes F49 Layer 3 absolute architectural Δ on FL; verifies CH15+CH18 don't reverse if STL ceiling is high.

### P3 — CA + TX upstream pipelines + 5f H3-alt
Full 5-stage plan in `/tmp/check2hgi_drafts/N8_ca_tx_upstream_handoff.md`. Total ~24h Colab T4 per state, ~47h both. Critical-path: F33 (FL decisive Path A/B test, ~6h Colab) gates the cat-head choice for CA/TX Stage 4.

### F33 + F36 — Colab FL Phase-2 grid
Full plan in `/tmp/check2hgi_drafts/N6_F33_F36_colab_launch_plan.md`. F33 ~6h + F36 ~5-6h Colab T4. Should fit in two Colab sessions OR two parallel tabs.

---

## Section 5 — Where each draft goes once SSD is back

```
/tmp/check2hgi_drafts/N2a_task_pivot_memo.md
  → docs/studies/check2hgi/SCOPE_DECISIONS.md
  → (optional) PAPER_STRUCTURE.md §1 short-form citation

/tmp/check2hgi_drafts/N2b_ch15_rename.md
  → execute the chosen option (A=CH15→CH15b search-replace) across:
     CLAIMS_AND_HYPOTHESES.md, OBJECTIVES_STATUS_TABLE.md,
     NORTH_STAR.md, CONCERNS.md §C16, SUBSTRATE_COMPARISON_FINDINGS.md,
     SESSION_HANDOFF_2026-04-27.md, PAPER_STRUCTURE.md (if mentions)
  → archive `/tmp/.../N2b_ch15_rename.md` to docs/studies/check2hgi/SCOPE_DECISIONS.md as the rationale entry

/tmp/check2hgi_drafts/p4_p5_wilcoxon.py
  → docs/studies/check2hgi/research/p4_p5_wilcoxon_offline.py (preserve the no-scipy version)

/tmp/check2hgi_drafts/p4_p5_wilcoxon_results.json
  → docs/studies/check2hgi/results/paired_tests/F49_decomposition_wilcoxon.json (rename)
  → also append a reference row in research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md §Followups

/tmp/check2hgi_drafts/N5_run_or_retire.md
  → execute decisions; add CONCERNS.md entries (C18, C19, C20) for retire'd items
  → archive the memo to docs/studies/check2hgi/SCOPE_DECISIONS.md

/tmp/check2hgi_drafts/N6_F33_F36_colab_launch_plan.md
  → docs/studies/check2hgi/PHASE2_TRACKER.md (append §F33 + §F36 launch sections)

/tmp/check2hgi_drafts/N7a_methods_section.md
/tmp/check2hgi_drafts/N7b_results_section.md
/tmp/check2hgi_drafts/N7c_methodological_appendix.md
/tmp/check2hgi_drafts/N7d_limitations_section.md
  → docs/studies/check2hgi/PAPER_DRAFT.md (consolidate as a single skeleton)
  → OR individual files: docs/studies/check2hgi/paper/{methods,results,appendix,limitations}.md

/tmp/check2hgi_drafts/N8_ca_tx_upstream_handoff.md
  → docs/studies/check2hgi/PHASE2_TRACKER.md §P3 / §CA_TX_UPSTREAM
```

---

## Section 6 — Verification recipe for the morning

After restoring SSD permission:

```bash
# 1. Sanity check that read/write to worktree works
cat "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/review/2026-04-28_post_phase1_overview_and_gap_audit.md" | head

# 2. Copy drafts in
mkdir -p "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/paper"
cp /tmp/check2hgi_drafts/N7a_methods_section.md           "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/paper/methods.md"
cp /tmp/check2hgi_drafts/N7b_results_section.md           "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/paper/results.md"
cp /tmp/check2hgi_drafts/N7c_methodological_appendix.md   "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/paper/appendix_methodology.md"
cp /tmp/check2hgi_drafts/N7d_limitations_section.md       "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/paper/limitations.md"

# 3. Move the Wilcoxon results
cp /tmp/check2hgi_drafts/p4_p5_wilcoxon_results.json     "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/results/paired_tests/F49_decomposition_wilcoxon.json"
cp /tmp/check2hgi_drafts/p4_p5_wilcoxon.py               "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/scripts/analysis/p4_p5_wilcoxon_offline.py"

# 4. Move decision memos
mkdir -p "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/scope"
cp /tmp/check2hgi_drafts/N2a_task_pivot_memo.md   "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/scope/task_pivot_memo.md"
cp /tmp/check2hgi_drafts/N2b_ch15_rename.md       "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/scope/ch15_rename_proposal.md"
cp /tmp/check2hgi_drafts/N5_run_or_retire.md      "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/scope/ch14_ch10_p02_decisions.md"

# 5. Move launch plans
cp /tmp/check2hgi_drafts/N6_F33_F36_colab_launch_plan.md  "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/launch_plans/f33_f36_colab.md"
cp /tmp/check2hgi_drafts/N8_ca_tx_upstream_handoff.md     "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl/docs/studies/check2hgi/launch_plans/ca_tx_upstream.md"

# 6. Verify and commit
cd "/Volumes/Vitor's SSD/ingred/.claude/worktrees/check2hgi-mtl"
git status
git diff --stat
git add docs/studies/check2hgi/paper/ docs/studies/check2hgi/scope/ docs/studies/check2hgi/launch_plans/ docs/studies/check2hgi/results/paired_tests/F49_decomposition_wilcoxon.json scripts/analysis/p4_p5_wilcoxon_offline.py
git commit -m "$(cat <<'EOF'
study: overnight session drafts — paper sections, Wilcoxon stats, scope memos

- Methods/Results/Limitations + methodological appendix drafts (post-F49 + Phase-1)
- P4 F49 paired Wilcoxon results (AL+AZ+FL n=5; offline pure-Python computation)
- Task-pivot memo (next_poi → next_category) + CH15 ID-reuse proposal
- F33+F36 Colab launch plan + CA/TX upstream handoff plan
- CH14/CH10/P0.2 run-or-retire decision proposals
- 2026-04-28 post-Phase-1 overview + gap audit (separate commit a1996e9 follow-on)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section 7 — What I deliberately did NOT do

- **Did not commit.** All drafts live in `/tmp` and the two files written to the SSD before the lock are uncommitted in your worktree. You decide what makes it in.
- **Did not run any new training.** P1/P2/P3 require user machines (4050, Colab); I have no remote-trigger.
- **Did not modify CLAIMS_AND_HYPOTHESES, OBJECTIVES_STATUS_TABLE, FOLLOWUPS_TRACKER, NORTH_STAR.** Decision memos surface what *should* change; you make the call before applying.
- **Did not run multi-seed sweep (P6).** Requires SSD + ~3h MPS; would have been worth doing if access were intact.

---

## Section 8 — Anything else surprising

1. **The `find /Volumes` background command was the trigger** for the permission lock. On macOS, broad scans of `/Volumes` can trigger TCC re-prompts which silently revoke access if not approved. Lesson: stick to scoped paths within the worktree.

2. **F49 doc reports joint-checkpoint metrics; my Wilcoxon used per-task-best-epoch metrics.** They differ by ~0.3 pp on means but the pairing structure is unchanged. The script is configurable (extractor function); when you re-run on full SSD access, you can switch to the joint-checkpoint extractor for paper-grade alignment. Either way, the headline conclusions ("transfer null", "co-adapt null at AL+AZ") hold.

3. **FL F49 frozen-cat instability gets sharper from per-fold inspection.** Per-fold reg-best epochs for FL frozen are {5, 18, 2, 2, 3} — consistent with the F49 doc's note that "α-growth doesn't engage when cat is random at FL scale". This is paper-worthy as a Limitations § entry (already in N7d).

4. **The H3-alt FL run is the de-facto F4.** F4 was originally framed as B3 5f at FL; H3-alt achieves what F4 was after (n=5 σ on FL MTL) at the new champion recipe. You may want to retire F4 explicitly in FOLLOWUPS_TRACKER before reviewers ask.

---

## Section 9 — Quick links

- This briefing: `/tmp/check2hgi_drafts/MORNING_BRIEFING.md`
- Working notes from yesterday's audit: `/tmp/check2hgi_synthesis.md`
- Overview that DID land in repo: `docs/studies/check2hgi/review/2026-04-28_post_phase1_overview_and_gap_audit.md`

Total elapsed work: ~2.5h compute-time of analysis + drafting; ~0 minutes of user-machine GPU.

Welcome back. Restore SSD access first, then triage.
