# P1 Methodology Flaws — Tracker

**Opened:** 2026-04-17 (post-P1-drain critical review).
**Scope:** Flaws discovered after the full 180-test P1 grid completed. Some are data-analysis artifacts (resolvable without new runs), some require ≤4 h of new compute. **All must be acknowledged before P2 begins** — P2's C06 MTL-vs-single-task claim inherits whatever champion we freeze from P1.

Status column legend: `open` / `investigating` / `mitigated` / `resolved` / `wontfix`.

---

## Summary table

| ID | Severity | Status | Short description |
|----|----------|--------|-------------------|
| **F1** | HIGH | **mitigated** | `joint_f1_taskbest` now in state.json for all 181 tests; `archive_result.py` computes it for future runs. Analysis discipline: report both joint@J and joint@T. |
| **F2** | HIGH | open | No multi-seed — P1 rankings within top-5 are inside fold-noise; champion is seed-42-specific. |
| **F3** | HIGH | **proxy-mitigated** | Fclass shortcut survives on fusion (linear probe: HGI half 0.688 F1 vs Sphere2Vec half 0.111). Full arm-C retrain pending for MTL-level quantification. |
| **F4** | MED  | **resolved (with surprise)** | base × eq AL 5f × 50ep = 0.4070 joint@J (joint@T 0.4217). At confirm protocol, base is **TIED with expert cells** (spread 0.0014 at joint@T across 6 configs). C05 downgraded to `partial`. |
| **F5** | MED  | open | NextHead best-epoch is 10–20 even at 50-epoch schedules — LR schedule mismatched for next task. |
| **F6** | MED  | open | `bayesagg_mtl × cgc22` AZ screen collapsed (joint=0.271 vs peers ~0.40); analyzer did not flag (verdict band too wide). |
| **F7** | LOW  | open | `expected.joint_range = [0.1, 0.65]` is so wide every test verdicts `matches_hypothesis`. Auto-analyzer is effectively a no-op. |
| **F8** | LOW  | open | Claim-evidence bookkeeping drifted (different optim buckets labeled "static" vs "grad" inconsistently across summary passes). |
| **F9** | LOW  | open | C02 was labeled `partial` on +0.0051 AL gap; Z = 0.39 vs fold-noise → honest status is `partially_refuted`. |
| **F10** | LOW | open | Uncommitted P1 artifacts risk rebase loss (CLAIMS, SUMMARY, HANDOFF, state.json, 30 new test dirs). |

---

## F1 — Joint-peak checkpoint selection biases the config ranking

### Problem

Reported `joint_f1` in `state.json` is the **harmonic mean** `2·cat·next/(cat+next)` computed at the epoch that maximized `joint_score` during training. Category peaks late (epochs 17–45), next peaks early (epochs 10–20). The joint-peak checkpoint is a compromise between the two — for most configs it is **well past next's peak and well before category's peak**.

The harmonic mean weights the smaller value (next F1) more heavily. So any config whose joint-peak happens to land closer to next's peak scores higher — not because it learned better representations, but because its training dynamics (LR schedule, grad magnitudes) happen to synchronize the two task peaks.

### Evidence

Per-task-best F1 (`diagnostic_task_best`) is already logged in `full_summary.json`. Recomputing `joint_f1` as HM(cat@best, next@best) vs HM(cat@joint-peak, next@joint-peak):

**AL P1c — joint@J (reported) vs joint@T (per-task-best):**

| cell | joint@J | joint@T | Δ |
|------|---------|---------|----|
| cgc21 × bayesagg_mtl | 0.4034 | 0.4215 | +0.0181 |
| cgc22 × equal_weight | 0.4031 | **0.4229** | +0.0198 |
| cgc22 × excess_mtl | 0.4043 | 0.4215 | +0.0173 |
| cgc22 × nash_mtl | 0.4034 | 0.4217 | +0.0183 |
| mmoe4 × gradnorm | **0.4082** | 0.4220 | +0.0138 |

At per-task-best, all five AL configs land within **0.0014 joint**. `equal_weight` tops the list at 0.4229. The gap between `mmoe4 × gradnorm` (the reported "winner") and `equal_weight` flips from +0.0051 to **−0.0009**.

**AZ P1c — same table:**

| cell | joint@J | joint@T | Δ |
|------|---------|---------|----|
| cgc21 × dwa | 0.4352 | **0.4416** | +0.0064 |
| cgc21 × gradnorm | 0.4369 | 0.4406 | +0.0037 |
| cgc21 × pcgrad | 0.4361 | 0.4392 | +0.0031 |
| cgc21 × uncertainty_weighting | **0.4374** | 0.4400 | +0.0026 |
| mmoe4 × nash_mtl | 0.4277 | 0.4367 | +0.0090 |

AZ winner flips from `cgc21 × uw` → `cgc21 × dwa` under per-task-best. Top-4 range tightens from 0.0022 to 0.0024.

### Implications

1. **`mmoe4 × gradnorm` is not a real AL champion.** Under per-task-best selection it lands mid-pack; `cgc22 × equal_weight` is tied or slightly better.
2. **C02 is measurably refuted under per-task-best.** AL: grad − eq = −0.0009. AZ: grad − uw = +0.0006. Average: ~0.
3. **The existing reported rankings in `state.json` are valid only under the specific assumption** that you will deploy the joint-peak checkpoint. Any comparison to single-task baselines (P2 C06) must use a checkpoint policy that does not artificially penalize next.

### Resolution options

- **Option A (recommended):** keep joint-peak as deployment checkpoint but **report both `joint@J` and `joint@T` in every P2+ analysis**. Treat rankings as meaningful only if consistent across both. Add a column to future phase-coordinator summaries.
- **Option B:** change `_joint_score` in `scripts/study/archive_result.py` from harmonic to arithmetic mean. Pro: less next-POI penalty. Con: de-emphasizes the task where MTL gains are most likely to show; invalidates existing P1 state.json joint_f1 values.
- **Option C:** adopt per-task-best as the scientific-comparison metric; keep joint-peak only for "deployment champion" selection. Document both in the paper.
- **Option D (cheapest immediate):** update the coordinator synthesis to always compute joint@T alongside joint@J. No code change, just analysis discipline.

### Action (completed 2026-04-17)

- [x] Added `joint_f1_taskbest`, `cat_f1_taskbest`, `next_f1_taskbest` to
  `_extract_observed` in `scripts/study/archive_result.py`. New runs record them
  automatically.
- [x] Wrote `scripts/study/_backfill_joint_taskbest.py` and ran it:
  181 existing tests in state.json now carry the new fields (re-parsed from
  each test's `full_summary.json`).
- [x] P1 SUMMARY.md and CLAIMS C02/C32 updated with joint@T comparisons.
- [ ] Discipline: all future coordinator synthesis must tabulate joint@J
  and joint@T side-by-side. Consider adding a `--metric {joint_f1,joint_f1_taskbest}`
  flag to `scripts/study/analyze_test.py` for automated verdicts under each.

---

## F2 — No multi-seed replication

### Problem

Every one of 180 P1 runs used `seed=42`. The phase doc (`P1_arch_x_optimizer.md §Seed`) explicitly called for multi-seed (seeds 123, 2024) on the top-3 at P1c confirmation. That step was skipped during execution.

Consequence: we have no estimate of inter-seed variance. The only variance we can report is **fold variance within a single seed**, which is ~0.011 on joint_f1 at 5f × 50ep on AL. Inter-config differences in the P1c top-5 are 0.0005–0.0051 — **all inside fold noise, let alone inter-seed noise**.

### Evidence

C02 signature on AL confirm:
- grad-surgery (mmoe4 × gradnorm) joint = 0.4082 ± 0.0081 (fold-std)
- equal_weight (cgc22 × equal_weight) joint = 0.4031 ± 0.0106 (fold-std)
- Gap: +0.0051. Pooled fold-noise: 0.0133. **Z ≈ 0.39.**

### Implications

- The "mmoe4 × gradnorm is AL champion" is at best a seed-42 observation. Under seed-123 or seed-2024 a different config may win the ordering.
- Cross-state winner disagreement (AL → mmoe4×gradnorm; AZ → cgc21×uw) could be entirely seed artifact.
- Any C06 (MTL vs single-task) comparison built on a single-seed champion will inherit the noise.

### Resolution

- [ ] Run `mmoe4 × gradnorm` + `cgc22 × equal_weight` at seeds 123 and 2024 on AL, 5f × 50ep. 4 runs × ~35 min = ~2.5 h. **Decisive test for both F1 and F2.**
- [ ] Same for AZ's `cgc21 × uw` + `cgc21 × dwa`. Another ~2.5 h.
- [ ] Total new compute to lock P1 honestly: ~5 h sequential, ~2.5 h on two machines.

---

## F3 — Fclass shortcut (C29) not tested on fusion

### Problem

C29 was confirmed on HGI-only: category F1 drops from 0.786 → 0.144 (random-chance 1/7) when fclass labels are shuffled across POIs. This means category F1 on OSM-Gowalla is primarily measuring fclass-identity preservation in the embedding.

**Fusion concatenates HGI (fclass-derived) + Sphere2Vec (pure location).** We have not tested whether Sphere2Vec breaks the fclass shortcut or merely appends spatial signal on top. If the shortcut survives on fusion, **all P1 category F1 comparisons are largely comparing fclass-preservation fidelity**, not learned-representation quality.

### Evidence

- `C29` evidence: Alabama HGI-only category macro F1 0.786 → 0.144 under fclass shuffle.
- `C30` closed the code-leakage audit on HGI; fusion not re-audited.
- Fusion does inherit POI2Vec (fclass-indexed) on its HGI half. Sphere2Vec half is pure coordinates.

### Implications

1. If shortcut dominates on fusion: the paper's "fusion improves representation quality" claim (C01, C11) cannot rely on category F1 and must lean entirely on **next F1** (which C29 showed is largely shortcut-free on HGI).
2. Joint F1 inherits the shortcut in proportion to category's weight. Under harmonic mean this is moderate.
3. Reviewers familiar with OSM will ask this question — and if unanswered, it's a rejection-worthy gap.

### Resolution

- [x] **Cheap proxy completed (2026-04-17):** linear probe on AL fusion category input. See `docs/studies/fusion/results/P1/linear_probe_fusion_AL.json`. Sphere2Vec-half probe F1 = 0.111 (below chance); HGI-half probe F1 = 0.688 (88% of MTL ceiling); full-fusion probe 0.682 (no lift from Sphere2Vec). **Shortcut fully carried by HGI half; Sphere2Vec does not mitigate.** C31 status updated from `pending` → `partial`.
- [ ] **Primary test pending:** extend `experiments/hgi_leakage_ablation.py` to generate a fusion input from a fclass-shuffled HGI embedding, then train MTL. Expected: MTL category F1 ≈ 0.15 (chance), matching the HGI-only arm-C result. ~30 min of plumbing + 1 training run (~10 min at 1f×10ep, or 35 min at 5f×50ep if we want apples-to-apples with the P1c cells).
- [ ] Update CLAIMS C31 status from `partial` → `confirmed` once primary test lands.
- [ ] Paper: add N05 or amend N03 to explicitly cover fusion.

---

## F4 — `base` arch never run at 5f × 50ep

### Problem

`C05` (expert-gating > FiLM base) is listed `confirmed`. Evidence comes entirely from screen (1f × 10ep) and promote (2f × 15ep). No `base` cell is in either state's P1c top-5, so the claim has no 5f × 50ep data under the same protocol as C02.

### Evidence

- AL screen: base mean 0.3519; expert archs 0.3822–0.3964. Consistent.
- AZ screen: base mean 0.4141; expert archs 0.4190–0.4313. Consistent but smaller.
- No base arch at promote or confirm on either state.

### Implications

Reviewer could argue C05 is protocol-conditional: "You showed expert-gating wins at 10 epochs; did it win at 50?". We believe yes (direction very clear at screen), but have no direct measurement.

### Resolution (completed 2026-04-17 — **paradigm-shift finding**)

- [x] Ran `P1_AL_confirm_base_equal_weight_seed42` at 5f × 50ep.
  Result: joint@J = 0.4070, joint@T = 0.4217.
- **Did NOT lock C05 — it falsified the strong form.** At confirm protocol,
  `base × eq` ties with expert-gating cells at joint@T (all 6 configs within
  0.0014) and **beats 4 of 5 expert cells at joint@J**. Matched-optimizer
  comparison (`base × eq` vs `cgc22 × eq`): +0.0040 joint@J (base wins) but
  −0.0011 joint@T (cgc22 wins); flips inside noise.
- **C05 downgraded** from `confirmed` to `partial`. Revised framing:
  "expert-gating accelerates convergence at short training budgets but does
  not raise the 5f × 50ep ceiling."
- [ ] Follow-up for completeness (not blocking): `base × nash_mtl` or
  `base × gradnorm` at 5f × 50ep on AL to cross-check the tie under a
  gradient-based optimizer. Also multi-seed of `base × eq` (F2 analogue).
- [ ] AZ `base × equal_weight` at confirm — low priority; AZ screen already
  showed tighter expert−base gap (+0.005–0.017 vs AL's +0.030–0.045) so a
  tie at confirm is even more expected on AZ.

---

## F5 — NextHead peaks at ~epoch 10–20 regardless of schedule length

### Problem

From fold-info files across all AL P1c cells:

- Category best epoch: 17–45 (uses the full schedule)
- **Next best epoch: 10–22** (peaks in the first 20–40% of training)

The OneCycleLR schedule covers all 50 epochs, but NextHead finishes learning in the first third. The remaining epochs train the next task past its peak, biasing the joint-peak checkpoint toward later-category/weaker-next tradeoffs.

### Evidence

AL P1c per-fold best-epoch:

| cell | cat_best | next_best |
|------|----------|-----------|
| cgc21 × bayesagg_mtl | [41,35,20,28,23] | [10,13,16,14,22] |
| cgc22 × equal_weight | [19,40,31,17,41] | [12,13,11,12,13] |
| cgc22 × excess_mtl | [27,26,25,17,43] | [12,13,12,12,11] |
| cgc22 × nash_mtl | [28,43,31,35,38] | [16,10,19,19,13] |
| mmoe4 × gradnorm | [19,18,45,42,40] | [11,13,10,13,14] |

### Implications

- The cross-stage pattern "next F1 drops from promote (2f × 15ep) → confirm (5f × 50ep)" is **partially a schedule artifact**, not true overfitting. At per-task-best selection, confirm next F1 ≈ promote next F1 (both around 0.283).
- MTL vs single-task (C06) comparisons need matched early-stopping or matched best-epoch selection, otherwise MTL appears artificially worse on next.

### Resolution

- [ ] **Short-term:** report per-task-best in addition to joint-peak (see F1 resolution).
- [ ] **Medium-term (for P4 or separate):** test (a) early-stopping next head at val-F1 plateau while continuing to train cat, (b) separate LR schedule for next head, (c) cycle-based schedule synchronized with next-peak (e.g., 15-epoch training for next, continued for category).
- [ ] **Paper claim:** document the peak-epoch gap as a mechanistic observation; supports C07 (asymmetric MTL benefit).

---

## F6 — `bayesagg_mtl × cgc22` AZ screen collapsed; analyzer didn't flag

### Problem

On AZ screen, `bayesagg_mtl` landed far below peers on some archs:

| arch | joint (AZ screen, bayesagg_mtl) |
|------|----------------------------------|
| base | 0.330 |
| cgc22 | **0.271** |
| cgc21 | 0.405 |
| mmoe4 | 0.401 |
| dsk42 | 0.345 |

`cgc22 × bayesagg_mtl` is 13 p.p. below `cgc21 × bayesagg_mtl` for the same optimizer. The auto-analyzer passed all of them (verdict = `matches_hypothesis`) because the expected band is `[0.1, 0.65]`.

### Implications

- Could be a legitimate optimizer×arch interaction (bayesagg_mtl unstable on cgc22's expert-routing geometry). Worth noting in C15 / robustness claims.
- Could be a transient numerical instability that would not reproduce at another seed (links to F2).
- Either way, **the auto-analyzer should have surfaced it.** The verdict system is currently providing false assurance.

### Resolution

- [ ] Narrow `expected.joint_range` for P2/P3/future phases to something like `[screen_mean − 3σ, screen_mean + 3σ]` per arch-bucket.
- [ ] Optionally, re-run `cgc22 × bayesagg_mtl` on AZ at seed 42 as a reproducibility test.
- [ ] If reproducible, add a one-line note under C15 (hparam robustness) about `bayesagg_mtl × cgc22` instability.

---

## F7 — `expected.joint_range` is a no-op

Essentially the same problem as F6, generalized: every P1 test got the same very-wide `[0.1, 0.65]` range, so 180/180 tests verdict `matches_hypothesis` automatically. The analyzer gave us **zero** useful signal during the drain.

### Resolution

- [ ] For P2/P3/P4, compute per-cell expected ranges from P1c data (same arch+optim, ±3·fold_std) or from screen baselines.
- [ ] Consider using `/study analyze --tolerance <val>` with a tighter default.

---

## F8 — Claim-evidence bookkeeping drift

Across the in-session summaries, "best static" vs "best equal_weight only" vs "best loss-dynamic" got conflated. Example: CLAIMS C02 P1a evidence line read "+0.006 on AZ" — correct against best-equal_weight only, but ambiguous against best-static.

### Resolution

- [ ] Going forward, every C02-style evidence line should spell out: `(comparison bucket) − (baseline bucket) = Δ at [stage, state]`.
- [ ] Audit existing C02, C05 evidence lines; already done in `CLAIMS_AND_HYPOTHESES.md` as of 2026-04-17.

---

## F9 — C02 was initially labeled `partial` on a null effect

Corrected in this session. Status is now `partially_refuted` to reflect: AL +0.0051 gap at Z=0.39; AZ −0.0005; per-task-best flips AL to −0.0009. **Net: no effect.**

---

## F10 — Uncommitted P1 artifacts

30 new test archive dirs + CLAIMS + HANDOFF + SUMMARY + state.json all sit in working tree uncommitted. Single rebase or branch switch loses provenance.

### Resolution

- [ ] Commit now: `git add docs/studies/fusion/ && git commit -m "study(P1): complete + critical review"`.
- [ ] Do not squash — the multiple analysis passes (coordinator, critical review) are legitimate separate history.
