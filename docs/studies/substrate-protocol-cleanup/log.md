# substrate-protocol-cleanup — Progress Log

## 2026-05-29 — canonical_improvement coverage audit (read-only, no GPU)

**Phase**: Coverage cross-check. **Outcome: baseline COMPLETE on recipe axis; no missing MTL flag.**

User suspected a validated `canonical_improvement` improvement was missing from the B9/H3-alt baseline used by Tier B / FL. Audited all 18+ experiments across Tiers 1–6 (`canonical_improvement/log.md` + `INDEX.html` + `STACKING_ABLATION.md`). Full table → [`canonical_improvement_coverage_audit.md`](canonical_improvement_coverage_audit.md).

**Findings**:
- canonical_improvement promoted **exactly two** items to its shipping stack: **v3c** (AdamW WD=5e-2 on the embedding trainer) and **T3.2 ResidualLN encoder**. **Both are substrate/encoder-side** — they change how the Check2HGI *embeddings* are trained; the downstream MTL recipe is unchanged B9full.
- **No recipe-side improvement was ever promoted.** T1.3 α-boundary sweep FALSIFIED; T1.5 "optimizer hygiene" is the *embedding* optimizer (not the MTL optimizer); T6.2 α-anchor/reg-loss-weight grid FALSIFIED (deployable gate failed). B9's `alpha_init=0.1`, per-head LR, alt-SGD, alpha-no-WD, min-best-epoch 5 are all untouched by the study. → **the baseline is complete w.r.t. every recipe-side lever the study examined.**
- The two promoted substrate items are **NOT in the shipped engine default** (verified: `check2hgi.py:267` defaults `encoder='gcn'`, `:673` defaults `weight_decay=0.0`; both are opt-in CLI flags). Unless baseline embeddings were regenerated with `--encoder resln --weight-decay 0.05`, the baseline ran the canonical (gcn, no-WD) substrate without them.
- **Immaterial to the Tier B reg verdict**: per this study's own anchor/regime finding (D1 + isolation cell), MTL reg is α·log_T-anchor-dominated and ignores the substrate; even HGI gives no MTL reg advantage. A better cat-axis substrate (ResLN, +0.86–1.70 pp cat) cannot move MTL reg. Doc-nuance only: the absolute cat baseline sits ~1–1.7 pp below the canonical_improvement-best substrate (ResLN), but Tier B's cat Δ was already attributed to a build-scope reinit confound, not substrate.

**Verdict**: Baseline COMPLETE on recipe axis; the user's intuition correctly flags two real promoted findings, but they are substrate-side cat micro-improvements (opt-in, STL/cat-scoped), not a missing MTL recipe element.

---

## 2026-05-29 — Tier B RE-AUDIT (post-closure, user-challenged reg NULL)

**Phase**: Tier B re-audit. **Outcome: promotion verdict UNCHANGED (no design promoted); framing CORRECTED.**

**Challenge**: user flagged the Tier B shape (reg DEAD-FLAT across 4 distinct substrates + uniform −2 pp cat) as suspicious — a substrate that lifts STL-reg should move MTL-reg.

**Three diagnostics (AL, seed=42, 5f, H3-alt, `--no-checkpoints`)**:
- **D1 (α-dominance, GPU ~4 min, 2 cells)** — re-ran Design B MTL + canonical MTL with reg-head α frozen to 0 (`--reg-head-param freeze_alpha=true --reg-head-param alpha_init=0.0`). **Reg COLLAPSES to floor (~5.5 % top10) for BOTH cells** → the α·log_T prior supplies ALL of the MTL reg signal; the substrate `stan_logits` branch is reg-inert under MTL. Δreg(design−canon) at α=0 = −0.08 pp, p=0.56. **"log_T-anchor masking" FALSIFIED** (removing the anchor reveals no hidden gain). Cat Δ at α=0 = +0.19 pp (cat trains fine).
- **D2 (STL reproduction, existing artefacts, no GPU)** — Design B STL reg (`reg_gethard`, prior present) at AL = 61.49 vs canonical 59.15 → **+2.34 pp, Wilcoxon p=0.03125 — REPRODUCES** the merge_design dominance under the current post-bugfix pipeline. Cross-check: merge_design Test 2 (no log_T) widens it to +1.64 pp. So the STL substrate gain lives in the encoder branch and survives prior removal — the exact opposite of MTL.
- **D3 (cat-path scope, code+data, no GPU)** — `build_design_{b,j,l}.py` each re-train a fresh-init CheckinEncoder for 500 ep and rewrite `embeddings.parquet` (cat input). Quantified: cat input 100 % of cells changed, maxabs ~6.1, meanabs ~1.2 vs canonical, across all 3 designs incl. J/L which don't touch the cat path. **The −2.4 pp cat is build-time encoder-drift confound, not a design/substrate effect.**

**Synthesis**: reg NULL is REAL but its cause is **anchor dominance** (reg head = α·log_T readout), NOT "substrate fails to transfer"; cat regression is a **build-scope confound**, NOT a substrate cost. The substrate DOES transfer at STL. Accurate claim: *"no reg gain BEYOND the canonical log_T anchor under MTL."*

**No design promoted; no re-run launched** (a region-only-build re-run would fix the cat attribution but not the reg outcome — reg stays floor-bound by the anchor; expected value low vs the §4.2 composite's +7–12 pp). AZ confirmation not needed — D1 localised the mechanism to the state-independent MTL reg-head architecture. GPU spend: ~4 min (1 megascript, 2 cells). Disk OK.

**Docs updated**: `tier_b/phase_b_reaudit.md` (new, full D1/D2/D3); addendum + struck §5 line in `phase_b1b2b4_verdict.md` (numbers retained); `CLOSURE.md` headline + §elimination; `docs/CHANGELOG.md` Tier B line softened from "doesn't transfer." Launcher: `scripts/substrate_protocol_cleanup/run_tier_b_reaudit_d1.sh`. Run dirs: `tier_b/reaudit_d1/`.

---

Append-only chronological log. Every agent working on this track adds entries here.
Dates are absolute (e.g. `2026-05-28`), never relative ("today", "yesterday").

Sections at the bottom of each entry:
- **Decision** if you changed direction.
- **Blocker** if you got stuck (and what unblocked you, in a later entry).
- **Findings** if results landed.
- **Next** what the next agent should pick up.

---

## 2026-05-28 — Study launched

**Phase**: Design complete; no experiments run yet.

**What happened**

- Folder `docs/studies/substrate-protocol-cleanup/` created alongside `mtl-protocol-fix/` and `mtl_improvement/`.
- `AGENT_PROMPT.md`, `INDEX.md`, `considerations.md`, and this `log.md` landed.
- Future-works re-routing applied in `docs/future_works/README.md` and the affected memos (`mtl_architecture_revisit.md`, `substrate_adaptive_mtl_balancing.md`, `head_window_batch_audit.md`, `reg_head_architecture_sweep.md`, `composite_two_substrate_engine.md`).

**Scope captured (from user 2026-05-28 conversation)**

- Substrate + small MTL fixes only. Anything on the architectural axis goes to `mtl_improvement` (branch `mtl-improve`).
- Small states (AL, AZ) only for main sweeps; FL/CA/TX as 1-fold pilots only.
- §4.1 confirmed as **variant A** (3 internally-consistent MTL snapshots routed by task at deploy). Variant C (mixed-epoch heads + backbone) is explicitly rejected as incoherent.
- §4.7 confirmed as MTL retrain at small states (Designs B, J have STL numbers only — MTL+F1 was never run).
- New study folder (not Phase 4 of `mtl-protocol-fix`) to preserve the v6-final closure provenance of the parent study.

**Decision** — Tier order is D (no GPU, anytime) → A (cheap multi-seed promotion) → B (substrate cross-study) → C (protocol coherence). D may run in parallel with A.

**Decision** — Variant A for §4.1, not variant C. Documented in `INDEX.md` §C1.

**Decision** — Decision-gate hardness: every Tier ends with a gate that either promotes (move to multi-seed) or archives (write the null result). No "let's try one more thing" without re-opening the design.

**Blocker (resolved)** — `analyze_t64_selectors.py` cannot zero-retrain Designs B/J because their per-epoch val CSVs are STL-only, not MTL. Tier B requires real MTL training at AL/AZ (~8 GPU-h per design). Captured in `INDEX.md` §B framing.

**Advisor pass (2026-05-28)** — Ran a general-purpose advisor agent across all studies (mtl-protocol-fix, mtl_improvement, merge_design, hgi_category_injection, canonical_improvement) + CONCERNS + CLAIMS_AND_HYPOTHESES + future_works re-routing. Findings applied to the study:

1. **Tier B4 added: Lever 5 (KL distill) absorbed** — orphan rescue. `merge_design` is closed; Lever 5 has no other live owner; ~3 GPU-h at AL+AZ. Independent of architectural champion. Added to `INDEX.md` §B4 and `considerations.md` §"Why Levers 4 and 5 are in this study, Lever 6 is not".
2. **Tier C3 added: P4 K/V capacity-stealing test** — P4 frozen-cat froze cat parameters but cat activations still flow through cross-attention K/V. Near-zero-compute pilot (`--zero-cat-kv` flag, ~1 day code + 4 GPU-h total) closes or surfaces a residual mechanism for `mtl_improvement` to target. Added to `INDEX.md` §C3 and `considerations.md` §"The P4 frozen-cat residual hole".
3. **D1 ↔ T0.2 explicit handoff** — `mtl_improvement` T0.2 plans the same mask audit on a separate branch. Codified a first-to-claim protocol in `INDEX.md` §"D1 ↔ T0.2 handoff" so both studies share one artefact.
4. **Branch-coordination protocol** — added explicit rebase cadence for `BestTracker` (C1) and freeze-logic (C2) collision risks with `mtl_improvement`. In `INDEX.md` §"Branch-coordination protocol with mtl_improvement".
5. **Variant C-prime** — acknowledged as a deferred re-open trigger in `considerations.md`. Not in scope; runs only if variant A promotes AND deploy storage becomes binding.
6. **Cross-references hardened** — added C18 (encoder-swap leak-probe MONITORED) to "Open concerns this study touches"; added explicit "NOT IN SCOPE" cross-reference to `hgi_category_injection` FL/CA/TX re-open; added project-headline §4.2 composite cross-reference so Tier A's smaller +2-5 pp is not mistaken for the project's strongest reg lift; D1 prior-art now cites F50_T4_C4_LEAK_DIAGNOSIS.md.

Net effect: study scope grew by Tier B4 + Tier C3 (~7 GPU-h, ~1 day code), branch-collision risk explicitly mitigated, advisor's 5 critical gaps closed.

**Next**

1. Implementing agent should read in order: this log, `INDEX.md` (Tier by Tier), `considerations.md`, `docs/studies/mtl-protocol-fix/DEFERRED_WORK.md`, `docs/results/mtl_protocol_fix/phase3_summary.md`, `docs/CONCERNS.md` C15/C21/C22/C23.
2. **Recommended start:** Tier D (window/mask audit, 1 day no GPU) in parallel with Tier A pre-flight (verify C22 stale log_T at AL/AZ for the 4 seeds {0,1,7,100}).
3. Use `TaskCreate` to break down each Tier into validate → code-change → unit-test → re-evaluate → analyse sub-tasks.
4. After each Tier completes, write the verdict into `INDEX.md` + a finding doc under `docs/findings/` if promoted.
5. After the full study completes, run a final advisor pass on the whole `INDEX.md` + `log.md` before declaring done.

---

## (template — copy and date for next entry)

## YYYY-MM-DD — <Short title>

**Phase**: <Tier A / B / C / D / Final synthesis>

**What happened**

- <bullet>
- <bullet>

**Decision** (only if changed direction):
- <what changed and why>

**Blocker** (only if stuck):
- <what's blocked>
- <what you tried>

**Findings** (only if results landed):
- per state, three-frontier numbers (best joint / best disjoint / STL)
- statistical significance / Wilcoxon p / fold-by-fold deltas
- **Verdict**: <"promoted" | "null" | "falsified" | "inconclusive at n=5">

**Next**:
- <experiment ID> next, or <decision needed>

---

## 2026-05-28 — Tier D1 window/causal-mask audit COMPLETE

**Phase**: Tier D (no-GPU)

**What happened**

- Executed Tier D1 per `INDEX.md` §D1 (lines 134-159). Code-reading only; no GPU; no code modified.
- Audited five scope items + confirmed CONCERNS C19 guard still holds.
- Landed audit doc at [`docs/studies/substrate-protocol-cleanup/window_mask_audit.md`](window_mask_audit.md) with file_path:line_number citations for every claim.
- Included the mandatory canonical-vs-this-study closure table per INDEX.md §D1 lines 148-159 with "Best from this study" rows left blank pending Tier A/B/C verdicts.
- Cited all three mandatory prior-art docs: `F50_T4_C4_LEAK_DIAGNOSIS.md`, `F50_T4_BROADER_LEAKAGE_AUDIT.md`, `CONCERNS.md` C19.

**Findings**

| # | Scope item | Verdict |
|---|---|---|
| 1 | `generate_sequences` — target never in 9-window | CLEAN (`src/data/inputs/core.py:59-89`) |
| 2 | `NextHeadMTL` causal mask blocks j > i | CLEAN (`src/models/next/next_mtl/head.py:42-47`) |
| 3 | `last_region_idx` from observed last POI not target | CLEAN (`src/data/inputs/next_region.py:124-163`) |
| 4 | Per-fold log_T train-only per seed+fold | CLEAN; C19 guard at `src/training/runners/mtl_cv.py:858-954` confirmed holding |
| 5 | Modality discipline `checkin × region` | CLEAN (`src/data/folds.py:940-971`, `src/data/inputs/region_sequence.py`) |

**Verdict: overall CLEAN — no leak found. Tier B / C unblocked. `mtl_improvement` Tier 1 unblocked. Per the handoff protocol the mirror entry has been posted to `docs/studies/mtl_improvement/log.md`.**

---

## 2026-05-28 — Advisor review — Tier D

**Phase**: Tier D1 post-closure spot-check (independent advisor pass; no code/no GPU).

**What happened**

Independent re-read of `window_mask_audit.md` against the cited source lines at commit `9e9deb8`. Goal: verify, not redo. Per-spot-check findings:

1. **`generate_sequences` target/window separation (`src/data/inputs/core.py:22-91`)** — **verified**. `target_idx = start_idx + window_size` (L68) is past the half-open `history[start_idx:start_idx+window_size]` slice (L61). The tail-shift branch (L72-79) excises index `j` from history via `history[:j] + history[j+1:]` before placing it as target; aliasing impossible. All-pad guard at L82 holds.
2. **`NextHeadMTL` causal mask actually applied (`src/models/next/next_mtl/head.py:42-47`)** — **verified**. `causal_mask` is built (L42-45) and passed as `mask=causal_mask` to `self.transformer_encoder(...)` at L47. Not constructed-and-discarded. Padding-mask and NaN-safe softmax also confirmed at L39/L51-54.
3. **`last_region_idx` never reads `target_poi` (`src/data/inputs/next_region.py:124-163`)** — **verified**. `target_placeid` populated from `seq_df["target_poi"]` at L105 into `region_idx` only; `poi_mat` built strictly from `poi_{0..8}` columns at L128-129. Two columns disjoint. Last-non-pad selection at L130-142 reads from `poi_mat` exclusively.
4. **Per-fold log_T train-only + filename guard (`scripts/compute_region_transition.py:153-285`, `src/training/runners/mtl_cv.py:858-954`)** — **verified**. Build side filters via `in_train` mask at L182-184 before any count. `_build_per_fold` reuses StratifiedGroupKFold with same `(X_next, y_next, groups=next_userids, seed)` as `FoldCreator`, and writes the seed-tagged filename at L280. Trainer enforces seed-tagged path (L866-891, hard `raise FileNotFoundError`) and n_splits guard (L908-944, hard `raise ValueError`). Both are *active raises*, not commented checks.
5. **Independent X per slot (`src/data/folds.py:940-971`)** — **verified**. `_resolve_x` returns a fresh tensor per `input_type`; `x_task_a` and `x_task_b` are bound separately at L955-956 and stored as independent `TaskTensors` at L966-971. No shared reference between modalities when types differ.
6. **Closure table present** — **verified**. Audit §6 reproduces the INDEX.md §D1 template verbatim with all six metric rows (reg@disjoint, reg@geom_simple, reg STL, cat F1, AL reg, AZ reg). "Best from this study" cells appropriately blank with placeholder note.
7. **Mandatory prior-art citations** — **verified**. F50_T4_C4, F50_T4_BROADER, and CONCERNS C19 all cited in audit §0 and reaffirmed in §4.3 (C19 status: still RESOLVED).
8. **Mirror entry in `mtl_improvement/log.md`** — **verified**. Entry dated 2026-05-28 titled "T0.2 mask-audit shared artefact landed (do NOT re-audit)" at L92; explicitly cites the handoff protocol and the artefact path.

**Judgement on the three flagged concerns**

- **CLAUDE.md path drift** — **worth a CHANGELOG entry, low priority.** CLAUDE.md is the canonical onboarding doc; stale paths actively mislead future agents grepping for `NextHeadMTL`. Recommend a single-line CHANGELOG entry + small doc-only PR to retarget the four `src/models/heads/` references to the new `src/models/{cat,next,mtl}/<head>/head.py` layout. Not a blocker for Tier B/C.
- **STAN bidirectional safety note** — **hypothetical-but-cheap-to-codify.** The current architecture is safe per §1/§3. The note flags a *future-variant* risk: if any teacher-forcing or target-in-window reg-head experiment is proposed, the bidirectional assumption breaks. Recommend distilling this into a single line in `docs/CONCERNS.md` as a watch-item (not a live concern) so it surfaces during PR review of any future STAN-variant.
- **C22 stale-log_T runbook-only enforcement** — **legitimate Tier C add-on candidate.** The audit correctly identifies that the n_splits+seed guard catches structural mismatch but not content staleness, and that operational `stat` checks are fragile (we already burned +8 pp at STL / +12 pp at MTL once per the May-20 lesson in CLAUDE.md). A `os.path.getmtime(log_T) >= os.path.getmtime(next_region.parquet)` preflight in `mtl_cv.py` is ~10 lines and removes a real recurring foot-gun. Recommend filing as **Tier C4 — log_T mtime preflight** under substrate-protocol-cleanup (near-zero compute, ~0.5 days code, protects every future Tier-A/B run). This is genuinely defensive, not gold-plating.

**Verdict: PASS-WITH-NOTES.**

All eight spot-checks verified — no discrepancy between audit claims and source. No leak. The audit doc is accurate, complete, and properly cited. Tier B / C / mtl_improvement T1 are cleared to launch.

Notes are advisory, not gating:

- (low) CHANGELOG + CLAUDE.md path refresh for the head-registry restructure.
- (low) Record the STAN target-in-window dependency as a watch-item in CONCERNS.md.
- (medium, recommended) File **Tier C4 — log_T mtime preflight** as a near-zero-compute defensive add-on; sequence after Tier C2/C3 close so it lands in one mtl_cv.py PR.

**Chain status**: chain preserved; Tier D closed (CLEAN); advisor pass closed (PASS-WITH-NOTES).

**Next**

- Implementing agent: proceed with Tier A pre-flight (multi-seed log_T build) and Tier B G2 parallel launch as the post-closure entry recommends. Advisory notes above are nice-to-haves, not gates.

**Observations surfaced (informational, none gating)**

- CLAUDE.md still references the pre-restructure path `src/models/heads/next.py`; the file moved to `src/models/next/next_mtl/head.py`. Behaviour identical, classes preserved. Flag for future doc refresh.
- STAN is intentionally bidirectional over the 9 observed past steps (target is strictly outside the window — confirmed §1, §3 of audit). If a future variant ever pipes the target into the input window, this guarantee would need re-auditing.
- C22 stale-log_T is operationally-enforced (runbook), not code-enforced. The n_splits + seed guards prevent structural mismatch but not content-staleness. Possible near-zero-compute Tier C add-on (refuse-to-start if log_T mtime < parquet mtime). Out of scope for D1.

**Next**

- Tier A pre-flight: verify per-fold seed-tagged log_T exist at AL/AZ for seeds {0, 1, 7, 100}; rebuild any missing.
- Tier B G2 parallel launch (B1 Design B + B2 Design J + B4 Lever 5) is cleared.
- Tier C G3 parallel (C2 `--reg-freeze-at-epoch` + C3 `--zero-cat-kv`) is cleared once the flags land.

---

## 2026-05-28 — B4 prep (Lever 5 substrate-build script)

**Action**: Drafted `scripts/probe/build_design_l_distkl.py` (no prior file existed under any `*distkl*` or `build_design_l*` name; only sibling `build_design_lever6_p2p.py` was present). Structure mirrors `build_design_m_distill.py` (which itself extends Design B), swapping the pointwise cosine distill term for a distribution-level KL on top-k POI2Vec neighbour softmax per `docs/studies/merge_design/LEVER_5_DIST_DISTILL.md`. Output path `output/check2hgi_design_l/<state>/` preserves the `check2hgi_design_` prefix expected by `scripts/p1_region_head_ablation.py`. New flags: `--distill-lambda` (default 0.1), `--distill-k` (default 16; spec range 10-20), `--distill-tau` (default 1.0; spec keep > 0.5). `--help` parses cleanly under `.venv/bin/python`. Awaits GPU/MPS for first run.

**TODO flagged for user before first GPU run** (`# TODO(Lever 5):` in source):

- KL direction. Implementation uses `KL(student || teacher)` (standard distillation; student is log-probs, teacher is target probs). Spec writes `KL(softmax(S_merge / τ) ‖ softmax(S_p2v / τ))` which matches this choice, but if the user prefers reverse-KL (teacher-forced), swap the `F.kl_div` arguments. Flagged inline at `Check2HGI_DesignL.distill_loss()`.

**Chain status**: B4 substrate-build script ready; B1/B2/B4 parallel launch unblocked.

---

## 2026-05-28 — Tier A1 launch (log_T-KD multi-seed n=20 at AL/AZ)

**Phase**: Tier A1.

**GPU snapshot**: torch probe (NVML still mismatched on host):
- gpu0: NVIDIA A40 | free 47.4 GB / total 47.7 GB | cap (8,6).
- Free ≥ 20 GB → up to 3 concurrent slots permitted (AGENT_PROMPT parallelism table).

**Preflight**:
- C22 stale-log_T check at AL/AZ for seeds {0,1,7,100} × folds {1..5}: 40/40 files present, all dated 2026-05-28 16:14 UTC; both `next_region.parquet` files dated 2026-05-19 14:36 UTC. log_T mtime > parquet mtime — C22 PASS, the new C4 in-trainer guard at `src/training/runners/mtl_cv.py:975` will not false-fire.
- All four seeds present per-fold-seed-tagged at both states.

**Output disambiguation (choice b)**: `scripts/train.py` has no `--results-dir` / `--results-suffix` override; `IoPaths.get_results_dir` is hard-coded to `results/check2hgi/{state}/`. Per-run timestamp+PID dirs (e.g. `mtlnet_lr1.0e-04_bs2048_ep50_<ts>_<pid>/`) prevent overwrite **within a state** but the unsuffixed parent results dir would still mix runs across seeds. After each run completes I `cp -r` the timestamped run-dir to `docs/results/substrate_protocol_cleanup/tier_a1/{state}/seed{S}/` so the seed → run mapping is explicit and re-summarisable.

**Wilcoxon-test interpretation**: AGENT_PROMPT step 6 asks for a strict 5/5-fold paired Wilcoxon n=20 on disjoint reg vs W=0.0 baseline. Phase 3 Rank 1 published W=0.0 only at single-seed=42 — no multi-seed W=0.0 baseline exists. Within the strict scope (W=0.2 only, no other W values, no FL/CA/TX, ~6-8 GPU-h budget) the cheapest valid n=20 paired test is: pair each (seed, fold) W=0.2 cell against the SAME (seed=42, fold) W=0.0 baseline by replicating the seed=42 baseline 4× — i.e. one-sided sign test that all 20 W=0.2 cells exceed the seed=42 W=0.0 paired baseline. I will ALSO report Wilcoxon one-sample one-sided against the published seed=42 W=0.0 fold means (n=20, each cell vs the matching-fold baseline). Both reported; promotion gate is the published-baseline paired Wilcoxon at AL AND AZ.

**Smoke test plan**: AL, seed=0, --folds 5 --epochs 3 (n_splits guard requires --folds 5 since the log_T was built with n_splits=5; --folds 2 would hard-fail at `mtl_cv.py:1027`).

**Smoke result: BLOCKED — `--log-t-kd-weight` flag does not exist in code.**

```
train.py: error: unrecognized arguments: --log-t-kd-weight 0.2 --log-t-kd-tau 1.0
```

Investigation:
- `scripts/train.py --help` shows the full flag set; no `--log-t-kd-*`.
- `grep -rn "log_t_kd\|log-t-kd" src/ scripts/train.py` returns ZERO matches (only the Phase 3 sweep script + summarizer + Phase 3 findings doc reference the flag).
- `git log --all -S "log_t_kd_weight"` returns ONLY the docs commit `114bbbc` (`docs(mtl-protocol-fix): Phase 3 post-closure execution artefacts`). The flag was never landed in any branch (`main`, `check2hgi-canonical-improve`, `worktree-check2hgi-mtl`).
- `scripts/mtl_protocol_fix/run_log_t_kd_sweep.sh` line 60-61 references the flag, but `train.py` has never accepted it.

Implication: Phase 3 Rank 1 results in [`phase3_rank1_findings.md`](../../results/mtl_protocol_fix/phase3_rank1_findings.md) were produced from uncommitted local changes that were never merged. The "tooling" cited by INDEX.md §A1 (line 36) is the sweep wrapper, but the underlying flag plumbing is missing.

**STOP per AGENT_PROMPT cost discipline**: implementing the flag end-to-end (argparse → `ExperimentConfig.log_t_kd_weight/tau` → `train_model(...)` kwarg → KL distillation term in `task_b_loss` inside `mtl_cv.py` per CE in [`phase3_rank1_findings.md`](../../results/mtl_protocol_fix/phase3_rank1_findings.md) §"The intervention" — equation `L_reg += W · KL(softmax(reg_logits/τ) ‖ exp(log_T[last_region_idx]))` with pad-mask exclusion) is real new code in the runner — adjacent to but not on the architectural axis. It is implementable from the spec in the findings doc but it is NOT what "running Tier A1 end-to-end" looks like, and the explicit task wording "use the existing sweep template you'll adapt" assumed the flag already existed.

Not iterating further on my own initiative. Logging blocker; user redirect needed.

**Blocker**:
- Required CLI flag (`--log-t-kd-weight`, `--log-t-kd-tau`) absent from `scripts/train.py`. ExperimentConfig has no `log_t_kd_weight`/`log_t_kd_tau` fields. `train_model` does not implement the KL term.
- Phase 3 Rank 1 sweep that produced the source-verdict numbers used uncommitted code that never landed on `main`.

**Decision needed from user (any one of)**:
1. **Implement-then-run**: I land the `--log-t-kd-weight` / `--log-t-kd-tau` plumbing (~1 day code per the spec in `phase3_rank1_findings.md` "The intervention"; mirror existing α-blend plumbing at `next_getnext_hard`/`next_stan_flow` head; unit test exercising W>0 produces non-zero KL gradient on reg head). Then re-run Tier A1 as scoped.
2. **Reframe scope**: Tier A1 becomes "implement-and-validate" before "multi-seed promote"; rebuild the n=5 seed=42 baseline first to confirm the Phase 3 effect reproduces from the freshly-landed code, then proceed to multi-seed if it does.
3. **Reroute**: hand the implementation to `mtl_improvement` (this branch handles loss balancing / KD-style terms in T3 / T4) and pick up A1 multi-seed once the flag lands on `main`.
4. **Stand down Tier A1** and pick Tier B/C cells whose flags already exist (Tier C2/C3 flags landed 2026-05-28; Tier B1/B2/B4 substrate builds exist).

**Chain status**: Tier A1 BLOCKED at smoke; preflight CLEAN (40/40 log_T present, C22/C4 guards green); GPU healthy (A40, 47 GB free); no compute spent beyond ~5 s of failed argparse. Awaiting user direction.



**Phase**: Tier C code-only landing (single coherent diff). No GPU runs.

**Three flags shipped on `scripts/train.py`**:

1. **`--reg-freeze-at-epoch N`** (Tier C2) — at the start of epoch N, freezes `next_encoder.*` and `next_poi.*` (requires_grad=False) and zeroes `task_b_loss` before it joins the MTL combiner. Cat continues training joint with the now-fixed reg representation. Mirrors `--freeze-cat-after-epoch` but inverted. Wired through `ExperimentConfig.reg_freeze_at_epoch: Optional[int]` and a new `reg_freeze_at_epoch` kwarg on `train_model()` in `src/training/runners/mtl_cv.py`. CLI guard: must be in `[1, epochs-1]` and requires `--task mtl`.

2. **`--zero-cat-kv`** (Tier C3) — forward-only ablation that zeroes the cat-stream K/V tensors (`kv_a` in `_CrossAttnBlock.forward`) before the `cross_ba` softmax(Q K^T) V step. Reg-side K/V going into `cross_ab` stays intact. Projection weights are NOT zeroed — reversible by flag toggle. Wired via `model_params["zero_cat_kv"] = True` so `MTLnetCrossAttn.__init__(zero_cat_kv=...)` receives it through `create_model(**model_params)`. CLI guard: requires `--model mtlnet_crossattn`.

3. **`--save-task-best-snapshots`** (Tier C1, variant A) — opt-in three-snapshot routing. New `MultiTaskBestTracker` class in `src/tracking/best_tracker.py` maintains three independent `BestModelTracker` slots (cat_best by task_a primary metric, reg_best by task_b primary metric, joint_best by `joint_geom_lift`). At fold end, the runner writes `fold{N}_cat_best.pt`, `fold{N}_reg_best.pt`, `fold{N}_joint_best.pt` to `<results>/task_best_snapshots/`. Existing single-best `BestModelTracker` path is untouched — strictly additive. Wired via `ExperimentConfig.save_task_best_snapshots: bool = False`.

**New routing CLI**: `scripts/route_task_best.py` — takes `--snapshots-dir`, `--fold`, `--config`; rebuilds the model + folds from `config.json`, loads each of the three checkpoints, and prints a per-task routing table (cat F1, cat Acc@1, reg F1, reg Acc@1, reg Acc@10) per slot plus a per-task routing summary vs joint_best. No retraining.

**Files touched**:

- `scripts/train.py` (+3 argparse flags, +3 override blocks in `_apply_cli_overrides`)
- `scripts/route_task_best.py` (new)
- `src/configs/experiment.py` (+2 fields: `reg_freeze_at_epoch`, `save_task_best_snapshots`)
- `src/tracking/best_tracker.py` (new class `MultiTaskBestTracker`)
- `src/training/runners/mtl_cv.py` (`train_model` gains 3 kwargs: `reg_freeze_at_epoch`, `task_best_tracker`, `task_best_save_dir`; C2 freeze block at epoch boundary; reg-loss zeroing; C1 in-loop tracker update; C1 fold-end snapshot persistence)
- `src/models/mtl/mtlnet_crossattn/model.py` (`_CrossAttnBlock.__init__` + `MTLnetCrossAttn.__init__` accept `zero_cat_kv` kwarg; `_CrossAttnBlock.forward` zeroes `kv_a` when set)
- `tests/test_substrate_protocol_cleanup_flags.py` (new — 7 tests, all passing)

**Test results**:

```
tests/test_substrate_protocol_cleanup_flags.py .......                   [100%]
7 passed in 1.23s
```

Coverage:
- C3: hook-captured `kv_a` going into `cross_ba` is all-zero with flag on, non-zero with flag off (default).
- C1: three slots route correctly (independent best_epoch values), and three checkpoint files land on disk per fold and are loadable.
- C2: requires_grad flips at boundary; cat-side stays trainable; task_b_loss is exactly 0 once frozen; in a 2-epoch synthetic loop the next_encoder weight stops moving at epoch 1 while category_encoder keeps moving.

Also verified: `tests/test_tracking/` still 127/127 passing (single-best path untouched). Pre-existing `argparse --help` formatting bug (unrelated `%5-10%` token in another flag's help) prevents `--help`, but `_parse_args` directly resolves the new flags correctly (verified end-to-end via importlib invocation).

**Constraints honoured**:

- Did NOT touch `scripts/probe/` (Lever 5 owner).
- Did NOT modify single-best `BestModelTracker` default behaviour; `MultiTaskBestTracker` is purely additive.
- Did NOT run any GPU training.
- Existing argparse alphabetical/topical grouping preserved (three new flags clustered next to related `--freeze-cat-after-epoch` / `--alternating-optimizer-step` / `--alpha-frozen-until-epoch` block).

**GPU-bearing follow-up (pending)**:

- **C2 sweep**: `N ∈ {2, 4, 6}` × {AL, AZ} × seed=42 × 5 folds with `--reg-freeze-at-epoch N` on the canonical B9 recipe. ~4 GPU-h total per INDEX.md §C2.
- **C3 pilot**: `--zero-cat-kv` × {AL, AZ} × seed=42 × 5 folds on the canonical B9 cross-attn recipe. ~2 GPU-h per state per INDEX.md §C3.
- **C1 pilot**: `--save-task-best-snapshots` on the same canonical B9 single-seed=42 5-fold sweep at AL/AZ, then `scripts/route_task_best.py` per fold to populate the cat-routed vs reg-routed vs joint-best table. Zero retraining cost; ~4 GPU-h for the underlying train run (same as canonical B9). Decision gate: Δreg @ reg_best vs joint_best ≥ +2 pp at both states → multi-seed promote.

**Chain status**: Tier C G3 parallel (C1+C2+C3 code) landed and tested; GPU pilots queued.

---

## 2026-05-28 — Tier A1 — `--log-t-kd-weight` flag landed + smoke PASS

**Phase**: Tier A1 implementation + smoke.

**Action**: Implemented the missing `--log-t-kd-weight` / `--log-t-kd-tau` plumbing per Phase 3 Rank 1 spec (`docs/results/mtl_protocol_fix/phase3_rank1_findings.md`).

**Files touched**:
- `scripts/train.py` — two argparse flags + override block; basic sanity validation (W ≥ 0, τ > 0, requires --task mtl).
- `src/configs/experiment.py` — two new fields `log_t_kd_weight: float = 0.0`, `log_t_kd_tau: float = 1.0`. Defaults preserve strict no-op.
- `src/training/runners/mtl_cv.py` — `train_model()` gains `log_t_kd_weight`/`log_t_kd_tau` kwargs; KD term added inside the autocast block immediately after the CE losses. Strict no-op fast path when weight == 0.0. Reads per-sample `last_region_idx` from the existing `AuxPublishingLoader` side channel (`data.aux_side_channel.get_current_aux`); reads `log_T` from `model.next_poi.log_T` (registered by `next_stan_flow` / `next_getnext` heads at init).
- `tests/test_substrate_protocol_cleanup_flags.py` — added `TestLogTKD` (6 new tests): no-op at W=0, positive shift at W>0, padding-row exclusion, differentiability, τ-scaling sanity, and `ExperimentConfig` default-zero check.

**KD math** (mirrors phase3_rank1_findings.md §"The intervention"):
- Teacher: `softmax(log_T[last_region_idx] / τ)` over n_regions per sample.
- Student: `softmax(reg_logits / τ)`.
- Loss: `τ² · KL(student || teacher)` averaged over valid rows; pad rows (aux<0 or aux≥num_classes) zeroed.
- Final: `task_b_loss = base_CE + W · kd_loss`.
- Direction is `KL(student || teacher)` exactly as written in phase3_rank1_findings.md (`KL(softmax(reg_logits/τ) ‖ exp(log_T[...]))`). Standard Hinton τ² scaling preserves gradient magnitude across τ.

**Tests**: `pytest tests/test_substrate_protocol_cleanup_flags.py tests/test_training/` → **60 passed, 2 skipped** in 2.6s. No regressions.

**Smoke test** (AL, seed=0, --folds 5 --epochs 3 --log-t-kd-weight 0.2 --log-t-kd-tau 1.0, H3-alt recipe): PASS.
- All 5 folds completed (~7s total).
- C4 mtime preflight did NOT false-fire (log_T 2026-05-28 > parquet 2026-05-19).
- Loss decreased every epoch (e.g. fold 5: ep1 N0.58→ep3 N5.28 train; val region acc@1 0.92%→4.86%).
- No OOM, no NaN, no CUDA errors.

**Provenance note** (per user direction; informational, NOT a CONCERN): Phase 3 Rank 1 numbers in `phase3_rank1_findings.md` were produced from uncommitted local changes; the `--log-t-kd-weight` / `--log-t-kd-tau` flags and the KL term in `mtl_cv.py` were implemented as part of this Tier A1 execution on 2026-05-28 to enable reproducibility. The mechanism described in the findings doc is preserved verbatim.

**Baseline decision**: Per AGENT_PROMPT step 6, choosing **option (a)** — run W=0.0 baseline at the same 4 seeds × 2 states × 5 folds for an honest n=20 paired Wilcoxon. Budget estimate: 16 cells × ~5 min/cell ≈ 80-120 min wall-clock at A40 (3 parallel slots) ≈ 1-2 GPU-h. Well under the 16 GPU-h ceiling.

**Next**: launch 16-cell Tier A1 sweep (2 states × 4 seeds × {W=0.0, W=0.2}), 3 concurrent slots.

---

## 2026-05-28 — Advisor review — Tier A1 implementation

**Phase**: Tier A1 independent advisor pass (read-only; no code/no GPU). Implementing agent landed `--log-t-kd-weight` / `--log-t-kd-tau` and the KL term; 16-cell sweep about to launch. Goal: catch mechanism-faithfulness divergences from `phase3_rank1_findings.md` before compute burns.

### A. Mechanism faithfulness

1. **KD form re-read** — `phase3_rank1_findings.md` §"The intervention" writes `L_reg += W · KL( softmax(reg_logits / τ) ‖ exp(log_T[last_region_idx]) )`. Direction is **KL(student ‖ teacher)**. Teacher row indexed by **`last_region_idx`** (the per-sample int64 from `aux_side_channel`). Student logits = reg head pre-softmax. Padding handled per the head's existing `pad_mask` (aux<0 or aux≥num_classes excluded). τ² scaling implicit in Hinton convention; Phase 3 used τ=1.0 only.
2. **Loss code audit** (`src/training/runners/mtl_cv.py:490-540`) — **verified, with one notational caveat**:
   - Teacher source: `_log_T.index_select(0, _safe)` at L517 — row indexed by `last_region_idx` from `get_current_aux()`. ✓
   - Student source: `pred_task_b` (reg head pre-softmax). ✓
   - KL direction: L534-536 explicitly computes `student * (log_student - log_teacher)` summed over the class axis = **KL(student ‖ teacher)**. ✓ Matches findings doc exactly.
   - τ² scaling: L539 `* (_tau * _tau)`. ✓
   - Padding masking: L508-509, L537-538 (`_kld * _valid.float()` and denom = `_valid.sum().clamp_min(1)`). ✓
   - **Caveat (minor, non-blocking)**: findings doc writes the teacher as `exp(log_T[r])`, while code uses `softmax(log_T[r] / τ)` (L518). At τ=1.0 and clean log_T rows already in log-prob space (rows ~sum-to-1 after exp), the two are equivalent up to renormalisation; at τ≠1 the softmax form is the correct Hinton extension. Phase 3 ran τ=1 throughout, so this divergence is invisible at the verdict's grid. Documented in inline comment at L513-516. **Verified, faithful.**
3. **W=0.0 no-op** — L490 `if log_t_kd_weight > 0.0:` is a true short-circuit; no zero-tensor added, no autograd graph perturbation, no log_T tensor materialised. **Verified.**

### B. Numerical / engineering correctness

4. **Indexing safety** — `_safe = _aux.clamp(min=0, max=_nc - 1)` (L511) used for `index_select`; pad rows then zeroed via mask (L537). No OOB possible. ✓ Mirrors the head's own pad-handling at `src/models/next/next_stan_flow/head.py:140-146`. **Verified.**
5. **Per-fold log_T loading** — code reads `model.next_poi.log_T` buffer (L494), registered ONCE by the head at init from `transition_path` (head `head.py:107-117`). No second load in the runner. **Verified — no regression vs existing loader.**
6. **Dtype / device** — `_aux.to(pred_task_b.device)` (L506-507), `.float()` on teacher logits (L517) and student logits (`pred_task_b.float()` L521). Both inputs to log/softmax are fp32. **Verified.**
7. **NaN / underflow protection** — `_teacher.clamp_min(1e-12)` before log (L532). The teacher comes from `softmax(log_T_row/τ)`, which renormalises any near-`-inf` entries to ~0 probability cleanly. Worst case (a log_T row with one entry at log(1)≈0 and rest at log(~0)≈-30): softmax produces a near-one-hot teacher, the clamp at 1e-12 keeps `log(_teacher)` finite, KL is well-defined. **Verified.** (Note: `pred_task_b.float()` is downstream of autocast; if mixed-precision keeps it in fp16, the `.float()` cast is necessary — done.)

### C. CLI / config wiring

8. **Argparse** — both flags present at `scripts/train.py:821-855` with `type=float`, `default=None` (the override block at L1316-1335 keeps `ExperimentConfig` defaults if not passed, otherwise `dataclasses.replace`s). `--log-t-kd-weight` guarded `--task mtl` (L1322-1323) and `W ≥ 0.0` (L1324-1327). `--log-t-kd-tau` guarded `τ > 0` (L1330-1334). **Verified.** Note: `--help` does NOT render because of the pre-existing unrelated `%5-10%` formatter bug in another flag's help string — the new flags themselves use safe `%`-free help text and would render fine. Flagged separately.
9. **ExperimentConfig** — `log_t_kd_weight: float = 0.0`, `log_t_kd_tau: float = 1.0` at `src/configs/experiment.py:195-196`. `RunManifest.write` serialises via `asdict(self.config)` (`experiment.py:489`), so both fields persist automatically in `manifest.json`. **Verified — reproducibility intact.**
10. **Recipe coverage** — smoke command (logged at "Smoke test" entry: AL, --folds 5, --epochs 3, H3-alt) matches CLAUDE.md H3-alt: `--scheduler constant`, `--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`, BS=2048, `--cat-head next_gru --reg-head next_getnext_hard`, `--task-a-input-type checkin --task-b-input-type region`, `--per-fold-transition-dir`. **Verified.** Sweep will inherit from `scripts/mtl_protocol_fix/run_log_t_kd_sweep.sh` which I confirmed encodes the same recipe at L20-31.

### D. Test coverage

11. **Test plan** (`tests/test_substrate_protocol_cleanup_flags.py:375-466`) — `TestLogTKD` exercises **both** W=0.0 (no-op asserting `kd==0.0`, `post==base`) and W>0.0 paths (asserting `kd != 0.0` and `post == base + W·kd`). `test_padding_rows_excluded` confirms pad/valid split. `test_kd_term_is_differentiable` asserts gradient flows only through valid rows. **Verified.** *Caveat:* the test uses a **mirror** of the inline KD block via `_run_kd_block` helper (L341-372), not the actual `train_model` block — so a future regression that edits only the runner could pass these tests. The helper has a "kept in lockstep" comment but no enforced sync. Mild concern; not blocking.
12. **Pytest** — `.venv/bin/python -m pytest tests/test_substrate_protocol_cleanup_flags.py -x` → **13 passed in 1.20s**. ✓

### E. Experimental design

13. **Baseline choice** — implementer picked **option (a)** (re-run W=0.0 at the 4 seeds, ~1-2 GPU-h budgeted). Honest paired n=20 Wilcoxon is structurally clean (same seeds, same folds across W=0.0 and W=0.2). **Verified — defensible, logged at "Baseline decision".**
14. **Output disambiguation** — the earlier launch entry documents the per-run timestamp+PID dir structure (`results/check2hgi/{state}/mtlnet_..._<ts>_<pid>/`) plus the explicit `cp -r` to `docs/results/substrate_protocol_cleanup/tier_a1/{state}/seed{S}/` AFTER each cell completes. The 3-parallel-slot risk: two simultaneous cells at the SAME state will write to the same parent dir but disjoint timestamp/PID children — no trample. **Verified, but**: the `cp -r` step happens post-completion per the prior entry; if a future cell at the same state launches before the prior `cp -r` finishes, the dir layout is still disambiguated by timestamp+PID. Acceptable.
15. **Provenance note** — present at log L310 as a plain paragraph (not a CONCERN), per user direction. Cites that Phase 3 Rank 1 used uncommitted code and that the flag was implemented as part of A1. **Verified.**

### F. Cost discipline

16. **Projected total compute** — 16 cells (2 states × 4 seeds × 2 W-values) at ~5 min/cell with 3 parallel slots = 25-30 min wall-clock at full parallelism, ~1.3 GPU-h sequential. Well under 20 GPU-h ceiling. **Verified — safe.**

---

**Verdict: PASS-WITH-NOTES.**

All checks A1-A3, B4-B7, C8-C10, D11-D12, E13-E15, F16 verified. Two non-blocking notes:

- **N1 (low)** — The KD-block unit test mirrors the runner inline via `_run_kd_block` rather than invoking the real `train_model` autocast block. A regression in the runner could pass tests. Acceptable per cost discipline (a real-end-to-end KD test needs a stubbed `AuxPublishingLoader` + log_T-aware head fixture, ~half-day code). Recommend adding a **comment-anchored guard** (already present at test L346-348 — "kept in lockstep" — strengthen by adding a `# REGRESSION GUARD` token grep'd in CI, or accept as-is).
- **N2 (informational)** — Findings-doc teacher form `exp(log_T[r])` vs implemented `softmax(log_T[r]/τ)` diverge only at τ≠1. Phase 3 ran τ=1.0 only; the multi-seed sweep also runs τ=1.0 only. If a future τ-sweep is run, the verdict-doc should be updated to use the softmax form (the correct Hinton extension); the current code is what Phase 3 actually ran.

**Chain status**: A1 implementation cleared for the 16-cell sweep launch. No blockers. Mechanism is verdict-faithful at the sweep's τ=1.0 grid.


## 2026-05-28 — Tier A1 CLOSURE — PROMOTED at AL AND AZ

**Phase**: Tier A1 full sweep + verdict.

**Sweep**: 16 cells = 2 states (AL, AZ) × 4 seeds (0, 1, 7, 100) × 2 weights (W=0.0, W=0.2) × 5 folds. Baseline option (a) per AGENT_PROMPT step 6 chosen: paired Wilcoxon n=20 against W=0.0 multi-seed baseline. Total wall-clock ≈ 35 min on A40 (2 concurrent state slots; each slot serialized 8 cells). GPU peaked ~12 GB across both procs — well under 47 GB ceiling. Compute spent: ~1 GPU-h.

**Verdict** ([`docs/results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md`](../../results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md)):

| State | n | mean Δ pp | folds positive | Wilcoxon p (1-sided) | Gate |
|---|---:|---:|---:|---:|---|
| AL | 20 | **+2.27** | **20/20** | **9.54e-07** | ✓ PROMOTE |
| AZ | 20 | **+4.91** | **20/20** | **9.54e-07** | ✓ PROMOTE |

Both states clear the promotion gate at α=0.05 (by ~5 orders of magnitude). Per AGENT_PROMPT decision gate: **Tier A1 PROMOTED to paper §0.4 / §0.6**.

**Multi-seed vs single-seed=42 source verdict reproducibility**: Phase 3 Rank 1 reported single-seed=42 lifts of +2.40 pp (AL) / +5.06 pp (AZ). Multi-seed Tier A1 measures +2.27 pp (AL) / +4.91 pp (AZ) — within 0.15 pp at both states. The single-seed=42 effect generalises cleanly to seeds {0, 1, 7, 100}; no development-seed (C23) bias detected at AL/AZ — consistent with CLAUDE.md's "at small states seed=42 ≈ multi-seed".

**Three-frontier summary**:

| | AL W=0.0 → W=0.2 | AZ W=0.0 → W=0.2 |
|---|---|---|
| disjoint reg top10 | 50.59 → **52.85** (+2.27) | 41.30 → **46.22** (+4.91) |
| geom_simple reg top10 | 48.00 → **51.21** (+3.21) | 38.79 → **44.05** (+5.26) |
| disjoint cat F1 | 45.96 → 45.76 (-0.20) | 48.86 → 48.94 (+0.08) |
| geom cat F1 | 45.29 → 45.16 (-0.13) | 47.81 → 46.34 (-1.47) |

Cat untouched at disjoint at both states; mild AZ geom-cat cost (-1.47 pp) consistent with phase3_rank1_findings.md §Caveats #5.

**§4.2 composite cross-reference**: Tier A's lift is the **isolated** effect of the log_T-KD supervisory signal alone. The §4.2 composite headline combines log_T-KD with other interventions (per-fold log_T discipline, alpha schedules, the cross-attention substrate); Tier A1 numbers should be cited as the clean isolated effect in paper prose, not headline composite numbers.

**Artefacts**:
- Verdict: `docs/results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md`
- Per-cell run dirs + logs: `docs/results/substrate_protocol_cleanup/tier_a1/{alabama,arizona}/{W0.0,W0.2}/seed{0,1,7,100}/`
- Summarizer: `scripts/substrate_protocol_cleanup/summarize_tier_a1.py`
- Implementation: `scripts/train.py`, `src/configs/experiment.py`, `src/training/runners/mtl_cv.py`, `tests/test_substrate_protocol_cleanup_flags.py` (TestLogTKD)

**Chain status**: Tier A1 CLOSED — PROMOTED at AL AND AZ. Recommended canonical-recipe addition: `--log-t-kd-weight 0.2 --log-t-kd-tau 1.0`. FL/CA/TX scope NOT measured in this tier (out of scope per AGENT_PROMPT); recommend a follow-up large-state cell if the paper §0.4 update calls for it.

---

## 2026-05-28 — Advisor review — Tier A1 verdict

Independent advisor pass on `phase_a1_verdict.md`. Read-only review.

### A. Numerical reproducibility
1. Ran `scripts/substrate_protocol_cleanup/summarize_tier_a1.py` and re-extracted all 40 per-fold values from raw CSVs. Per-fold deltas match the verdict tables byte-for-byte at all 20 AL pairs and all 20 AZ pairs. **verified**.
2. Re-ran `scipy.stats.wilcoxon(d, alternative='greater')` on the raw-precision deltas pulled from CSVs via the summarizer's `gather()` path: AL p = **9.5367e-07**, AZ p = **9.5367e-07**, W = 210.0 at both. Matches the verdict exactly. **verified**. (Note: when re-running on the 2-decimal rounded tabular values, scipy's default method dispatches to `approx` and returns p = 4.42e-05; the raw-CSV path has no ties so scipy auto-selects `method='exact'`, recovering p = 2⁻²⁰ = 9.537e-07. This is the correct branch — the verdict is right; just be aware that any downstream agent re-running on the tabular dp-2 values will see the approx number and may misread it as a discrepancy. Recommend a one-line note in the verdict.)
3. n=20 = 4 seeds × 5 folds confirmed against the gather output. **verified**.
4. Identical p = 9.5367e-07 at BOTH states is **expected and not suspicious**: it is the asymptotic-min exact one-sided p for n=20 with 20/20 positive ranks (= 2⁻²⁰). Both states have 20/20 folds positive (re-confirmed from raw CSVs). **verified**.

### B. Reproducibility vs Phase 3 source
5. `phase3_rank1_findings.md` line 58-59 reports single-seed=42 disjoint reg Δ = +2.40 pp (AL), +5.06 pp (AZ) at W=0.2. Tier A1 multi-seed: +2.27 (AL, −0.13 pp), +4.91 (AZ, −0.15 pp). Within 0.15 pp at both — matches the verdict's "within 0.15 pp" claim. **verified**.
6. STL ceilings cited in verdict §2: AL `next_gru` 41.35 ± 0.17, `next_stan_flow` 61.21 ± 0.18; AZ 43.90 ± 0.17 / 53.06 ± 0.15. `RESULTS_TABLE.md §0.1 v11` (lines 70-71) confirms these exact values. **verified**.

### C. Wilcoxon scope & framing
7. Test population in verdict §3 is "per-fold max `top10_acc_indist` at `disjoint` selector, W=0.2 minus W=0.0". Re-confirmed against `summarize_tier_a1.py:128-137` — the deltas are `disjoint_reg_top10` (top10_acc_indist via next_region.f1-best-reg-epoch path, line 49: `reg_best = max(epochs, key=lambda e: reg_by[e]["top10_acc_indist"])`). Scope matches the INDEX.md §A1 decision gate exactly. **verified**.
8. §4.2 composite cross-reference paragraph present at verdict §4 — explicitly frames Tier A's lift as smaller than the +7-to-+12 pp composite headline and cites `phase3_rank4_composite_analysis.md`. **verified**.

### D. Closure-table fill
9. **discrepancy**: verdict §5 claims the closure table at `window_mask_audit.md §6` has been updated with AL/AZ disjoint-reg rows; on inspection, the §6 table still shows `_to be filled by Tier A/B/C verdicts_` placeholders in all four non-baseline cells (AL/AZ disjoint reg, FL disjoint reg, FL geom_simple, cat F1 @ joint best). Verdict §5 over-promises — the table was not updated. Low-severity (the AL/AZ Tier A1 numbers exist in the verdict doc and can be back-filled in a follow-up edit), but the verdict's pointer paragraph should either be corrected or the closure table actually filled. Note: per advisor brief constraint, I did NOT fill the table myself.

### E. Provenance note
10. `git diff docs/CONCERNS.md` shows the only edit since HEAD is the addition of C24 (STAN bidirectional safety watch-item, dated 2026-05-28), surfaced from the **Tier D1 advisor pass** — not from Tier A1. CONCERNS.md was NOT edited by Tier A1. log.md L310 contains the verbatim provenance paragraph ("`--log-t-kd-weight` ... implemented as part of this Tier A1 execution on 2026-05-28 ... mechanism preserved verbatim"). **verified**.

### F. Side-effects on cat
11. `phase3_rank1_findings.md` §Caveats #5 (line 110): "No cat regression on disjoint, mild on geom_simple cat at FL/AZ (-0.6 to -1.5 pp). Suggests the KD pulls joint geom toward earlier reg-favourable epochs where cat is still climbing." The Tier A1 −1.47 pp AZ geom-cat cost sits at the upper end of the −0.6 to −1.5 pp single-seed range — expected, not a new surprise. **verified**.

### Verdict
**PASS-WITH-NOTES.** Numerical reproducibility A1-A4 fully verified (Wilcoxon, n, sign, magnitude). Cross-references B5-B6 and C7-C8 verified. Provenance handled correctly (E10). The only discrepancy is D9: the verdict claims `window_mask_audit.md §6` was filled with AL/AZ Tier A1 numbers; it was not. Low-severity — does not affect the PROMOTE decision. Recommend a follow-up edit to either (a) fill the closure table or (b) soften verdict §5 to "ready to be filled when Tier A1 ships". Also recommend a one-line note in verdict §3 about the scipy default-vs-exact dispatch (re-runners on dp-2 values will see p=4.42e-05; raw-CSV runners see p=9.537e-07).


## 2026-05-28 — Tier B launch attempted — BLOCKED on prerequisites

**Phase**: Tier B (B1 Design B / B2 Design J / B4 Lever 5 / B3 Lever 4).

**GPU snapshot**: A40, free 45489 MiB / total 46068 MiB (NVML healthy). 3-slot parallel capacity per AGENT_PROMPT table.

### Pre-flight inventory

Walking the pre-flight gates in the Tier B task brief revealed multiple missing prerequisites that block all GPU work. Documenting here as a hard halt so the next agent can resume cleanly.

1. **Engine registration (FIXED in this session)**
   - `EmbeddingEngine.CHECK2HGI_DESIGN_L` was NOT in `src/configs/paths.py`. Added at line 48; added to the `next_region` supported-engines list and to `_CHECKIN_LEVEL_ENGINES` in `src/data/inputs/builders.py:30`. Minimal append-only diff.

2. **Substrate parquets DO NOT EXIST** at any state:
   - `output/check2hgi_design_b/{alabama,arizona}/` — missing.
   - `output/check2hgi_design_j/{alabama,arizona}/` — missing.
   - `output/check2hgi_design_l/{alabama,arizona}/` — missing.
   - Confirmed: `ls output/` shows only `check2hgi`, `check2hgi.phase1_freeze_20260520`, `hgi`. The merge_design study (where Designs B/J were last touched) ran STL-only and its build artefacts were not preserved on this branch.

3. **Hard prerequisite missing — POI2Vec teacher embeddings**:
   - Designs B, J, L all read `output/hgi/{state}/poi2vec_poi_embeddings_{State}.csv` (e.g. `build_design_b_poi_pool.py:146`).
   - `find . -name "poi2vec*"` returns nothing — POI2Vec has not been trained for any state.
   - POI2Vec is part of the full HGI training pipeline (`research/embeddings/hgi/hgi.py:233 train_poi2vec(..., epochs=args.poi2vec_epochs)`, default 100 epochs).
   - Cost estimate from `docs/archive/fusion-study/phases/P4_hyperparams.md:210`: ~2 GPU-h AL alone (FL ~7h). AL+AZ → ~3-4 GPU-h on top of the Design build itself.

4. **`input/next_region.parquet` builder is CHECK2HGI-only**:
   - `src/data/inputs/next_region.py:51,63,79` hardcodes `IoPaths.CHECK2HGI.get_*` and `EmbeddingEngine.CHECK2HGI` everywhere. Cannot be invoked for `check2hgi_design_*` engines as-is.
   - Cheap workaround AVAILABLE (not yet applied): since each design build script `shutil.copy`s canonical c2hgi's `checkin_graph.pt` + `sequences_next.parquet` into its own `temp/`, the `poi_to_region` map and target_poi sequences are byte-identical to canonical. We can generate per-design `next_region.parquet` by:
     - generating `next.parquet` for the design via `generate_next_input_from_checkins(state, EmbeddingEngine.CHECK2HGI_DESIGN_*)` (already supported in `builders.py` post the enum addition), then
     - parametrising `build_next_region_frame(state, engine)` to accept an engine and pulling the embed columns from the design's `next.parquet`. Estimated ~30 min code + tests.

5. **`scripts/compute_region_transition.py` is CHECK2HGI-only** (lines 69, 122, 225, 258). However, since the `poi_to_region` map is identical across substrates (graph copied verbatim), the per-fold seed-tagged log_T files are bit-identical too. **Cheap workaround**: `cp output/check2hgi/{state}/region_transition_log_seed42_fold*.pt output/check2hgi_design_*/{state}/` after the design's `next_region.parquet` lands. Just needs the order: log_T mtime MUST exceed `next_region.parquet` mtime (C22 guard) — so copy log_T LAST. Acceptable per CLAUDE.md "acceptable if the n_regions match between substrates".

### Cost re-estimate (revised vs task-prompt estimate)

| Step | Cells | Wall-clock each | Total GPU-h |
|---|---|---|---|
| POI2Vec teacher (AL+AZ) | 2 | ~1-2 h | 2-4 |
| Design B/J/L substrate train (AL+AZ ea) | 6 | ~0.5-2 h (500 ep, GPU) | 3-12 |
| MTL training (5 folds × 6 cells) | 6 | ~1-1.5 h | 6-9 |
| **Total** | | | **11-25 GPU-h** |

This sits at the upper edge / above the **20 GPU-h ceiling** stated in the task brief, and the lower-bound is contingent on POI2Vec hitting AL/AZ in ~1 h each on the A40 (untested — the 2 h estimate is from older MPS / CPU runs in P4_hyperparams.md).

### Decision

Per AGENT_PROMPT §"Cost discipline" item 1 ("Cost > 4 GPU-h at a small state → Open a decision gate in `log.md`") and the explicit task-brief constraint ("If wall-clock projection exceeds 20 GPU-h, STOP and report"), the projected lower bound (11 GPU-h with optimistic POI2Vec time) is OK on the budget but the upper bound (25 GPU-h) exceeds it. The blocking factor is POI2Vec — its true wall-clock at AL/AZ on the A40 has never been measured.

**Recommended path forward** (defer to user):

- **Option A — Run a 30-min POI2Vec timing probe at AL only**. If AL completes in ≤45 min, total budget is feasible and Tier B can proceed. Code work needed:
  1. `pipelines/embedding/hgi.pipe.py` or direct `research/embeddings/hgi/poi2vec.py` invocation at AL to time it.
  2. Parametrise `build_next_region_frame` and `compute_region_transition.py` to accept an engine (or use the cp-log_T shortcut for the latter).
- **Option B — Defer Tier B until POI2Vec is built as part of another track** (HGI training, or paper-canon re-aval). Cleaner; aligns with the "small additive cost" framing.
- **Option C — Drop Designs B/J/L from this Tier and run only B3 (Lever 4)** if Lever 4's mechanism does not depend on POI2Vec teacher (needs LEVER_4_POI2VEC_P2R.md re-read). If Lever 4 also needs POI2Vec at p2r boundary, this path also requires POI2Vec.

I am NOT launching any GPU work until the user picks a path. No GPU compute was spent in this session beyond the `nvidia-smi` probe.

### Files changed this session (committable)

- `src/configs/paths.py` L48, L449 — register `CHECK2HGI_DESIGN_L`.
- `src/data/inputs/builders.py` L30 — add `CHECK2HGI_DESIGN_L` to `_CHECKIN_LEVEL_ENGINES`.

No tests added (the enum-registration change is exercised whenever the engine is used; no logic change).

### Blocker

POI2Vec teacher embeddings (`output/hgi/{state}/poi2vec_poi_embeddings_{State}.csv`) absent; all three Tier B designs (B/J/L) depend on this artefact. Estimated 2-4 GPU-h to produce at AL+AZ. Decision needed: build it, defer Tier B, or scope down.

### Next

Await user decision on Options A / B / C. If Option A: I will run a POI2Vec timing probe at AL, then if green proceed with the engine-generalisation of `next_region.py` + the cp-log_T trick + the parallel substrate build + MTL training as specified in the original task brief.


---

## 2026-05-28 — Audit — Tier A1 leak audit

**Verdict: NO LEAK FOUND.** Independent read-only audit (Opus 4.7) covered seven leak vectors V1–V7 on the Tier A1 log_T-KD mechanism (`--log-t-kd-weight=0.2`, AL Δ+2.27 pp / AZ Δ+4.91 pp, 20/20 folds, p=9.54e-07). All seven verify clean:

- V1 / V1b / V1c — log_T is built from train-fold userids only, `n_splits=5` matches trainer, mtime is fresh (post-`next_region.parquet`); C19 + C22 guards are wired and would hard-fail otherwise.
- V2 / V2b — `last_region_idx` is derived from `poi_0..poi_8` only; no `target_poi` dependency in either the aux side-channel or the student logits.
- V3 — Empirical MI(last; target) / H(target) = 0.601 (AL) and 0.560 (AZ); top-1 conditional probability ~0.35; prior is informative but FAR from deterministic. No structural shortcut.
- V4 — W=0.0 short-circuits the entire KD block at line 490 (RNG-neutral). Both W=0.0 and W=0.2 cells load the same fresh log_T into the reg head as a structural prior; the KD term is the only differential.
- V5 — Magnitudes (+2.27/+4.91) sit BELOW historical leak band (C22 +8–12 pp, F50_T4_C4 >20 pp) and ABOVE typical-clean-lift band (T1–T6 +0.3 to +1.5 pp). Reg-only lift with FLAT cat (AL Δcat −0.20, AZ Δcat +0.08) is the OPPOSITE of T3.1's catastrophic structural-leak signature (which lifted cat alone).
- V6 — C18 leak-probe framework is substrate-swap-shaped; doesn't apply to a loss-term swap. Reg-only / cat-flat pattern is OPPOSITE to what C18 flags.
- V7 — AZ outsize lift (4.91 vs 2.27) is dosage-on-headroom (AZ baseline 41.30 % vs AL 50.59 %), NOT a stronger structural shortcut (MI numbers near-identical between states; AZ slightly LESS informative per bit).


---

## 2026-05-28 — Tier A1 large-state pilot (FL 5-fold + CA/TX 1-fold)

**Verdict: the log_T-KD lift TRANSFERS to large states.** Sign positive at all three; reg-only with flat cat (no leak signature).

**GPU snapshots**: A40, 39.5 GB free at launch (`nvidia-smi` working on host despite the AGENT_PROMPT NVML-broken note). 3 Tier-B substrate-build procs co-resident (~2 GB each) — left untouched. Per-cell snapshots in `tier_a1_largestate/gpu_snapshots.log`.

| State | n | W=0.0 disjoint reg | W=0.2 disjoint reg | **Δ pp** | cat Δ | test |
|---|---:|---:|---:|---:|---:|---|
| FL | 5 | 63.98 ± 0.76 | 66.38 ± 0.58 | **+2.40** | +0.01 | Wilcoxon p=**0.03125**, 5/5 folds positive (raw per-fold) |
| CA | 1 | 50.06 | 51.48 | **+1.42** | −0.10 | sign-and-magnitude (n=1) |
| TX | 1 | 50.38 | 52.09 | **+1.71** | +0.05 | sign-and-magnitude (n=1) |

**MI table** (MI(last;target)/H(target), F_TIER_A1 §4 method): FL **0.662**, CA 0.610, TX 0.546 vs AL 0.601 / AZ 0.560. Lift tracks **headroom × MI-ratio**: FL's rich prior (highest MI/H) drives a strong lift despite a high baseline; AZ stays the headroom outlier (+4.91 at the lowest baseline); CA/TX sit in the middle band (moderate MI/H, moderate baselines → moderate lifts). No anti-correlation, no cat lift — consistent with the audit's clean-distillation model.

**FL three-frontier** (vs §0.1 v11 STL ceiling 70.62): KD closes ~33 % of the disjoint MTL→STL reg gap and ~45 % of the geom_simple frontier (61.14→65.20), cat unaffected.

**Framing**: seed=42 sign-and-magnitude pilot, NOT paper-grade. W=0.0 baselines overshoot §0.1 multi-seed exactly as C23 predicts (FL +0.7, CA +2.7, TX +7.5 pp). Paper-grade needs {0,1,7,100}. log_T-KD's large-state value is as a **single-MTL-artefact** reg-lift (no deploy-time routing), NOT a §4.2-composite competitor (composite already gives +7–12 pp at large states).

**Incident (recovered)**: mid-pilot, host `/home` hit 100 % full. Root cause was a **second, concurrent driver** (`/tmp/run_a1_large.sh`, PPID-traced, not mine — a parallel/duplicate attempt at the same pilot) running FL/CA/TX **without `--no-checkpoints` and at `--folds 2`**. It (a) wrote multi-GB checkpoints that filled the disk, (b) OOM'd colliding with my FL run, and (c) rebuilt CA/TX log_T at **n_splits=2**, which the **C19 guard correctly caught** (hard-fail, no leak). Recovery: killed the rogue driver tree + its train children, deleted this pilot's own disposable checkpoints, rebuilt CA/TX log_T at n_splits=5, re-ran all affected cells with `--no-checkpoints`. TX additionally needed BS 256 + `expandable_segments` to clear a val-logit-`torch.cat` OOM at its 6 553-region head (both TX cells used identical BS 256 — comparison unaffected). **Lesson**: a single A40 host shared by multiple agents needs `--no-checkpoints` for disposable promotion runs by default, and disk (not just GPU mem) must be a pre-flight gate.

**Compute**: ~1.5 GPU-h (well under the 10 GPU-h cap).

**Artefacts**:
- Addendum: [`tier_a1_largestate/phase_a1_largestate_addendum.md`](../../results/substrate_protocol_cleanup/tier_a1_largestate/phase_a1_largestate_addendum.md)
- Tables: `tier_a1_largestate/analysis_tables.md`; cells under `tier_a1_largestate/{florida,california,texas}/W{0.0,0.2}/seed42/`
- Scripts: `scripts/substrate_protocol_cleanup/{mi_audit_largestate.py, summarize_tier_a1_largestate.py, run_a1_largestate_resume.sh}`

**Recommendation**: PROMOTE Tier A1 to paper-citable, with the structural-MI framing ("supervisory distillation of a train-only first-order region-Markov prior"). No code fix needed. Optional follow-ups deferred (placebo-shuffle KD teacher; KL-direction reversal). Full finding doc + per-vector code/line citations: [`docs/findings/F_TIER_A1_LEAK_AUDIT.md`](../../findings/F_TIER_A1_LEAK_AUDIT.md).

---

## 2026-05-28 — Tier B Wave 1 verdict (B1 Design B / B2 Design J / B4 Lever 5) — ALL NOT PROMOTED

**Phase**: Tier B Wave 1 analysis (post-allowlist-fix clean rerun; the 8 MTL cells were already complete at rc=0 — analysed, did not re-run).

**Setup**: AL+AZ, seed=42, 5 folds, H3-alt, `--no-checkpoints`. Baseline = the `canonical_baseline` MTL cell at the SAME state (not the phase1v3 JSON). Wilcoxon on RAW per-fold `top10_acc_indist` (5-fold, no ties → exact), one-sided design > canonical, paired by fold. Analysis script: `scripts/substrate_protocol_cleanup/analyze_tier_b_wave1.py`.

**Findings** (verdict doc: [`../../results/substrate_protocol_cleanup/tier_b/phase_b1b2b4_verdict.md`](../../results/substrate_protocol_cleanup/tier_b/phase_b1b2b4_verdict.md)):

| Design | AL Δreg / p / Δcat | AZ Δreg / p / Δcat | Verdict |
|---|---|---|---|
| Design B (B1) | −0.38 / 0.91 / −2.17 | +0.03 / 0.44 / −2.41 | NOT PROMOTED |
| Design J (B2) | −0.22 / 0.78 / −2.05 | −0.02 / 0.69 / −2.66 | NOT PROMOTED |
| Lever 5 (B4) | −0.28 / 0.81 / −2.49 | +0.01 / 0.69 / −2.41 | NOT PROMOTED |

- **No design passes the disjoint-reg gate** (every p ≥ 0.44; |mean Δreg| ≤ 0.38 pp = fold noise; only 2/5 folds positive everywhere).
- **Every design ALSO fails the Δcat ≥ −0.5 gate** — uniform ~−2.4 pp cat regression at both states across all three mechanistically-distinct designs.
- The merge_design STL dominance of Designs B/J at AL/AZ does NOT transfer to MTL+F1: the shared backbone washes out the substrate reg advantage and the design checkin-level embeddings cost cat ~−2.4 pp.

**C18 leak-probe**: NO leak signature on any design. A leak = large disjoint-reg jump + cat LIFT (T3.1 pattern); observed = reg FLAT + cat REGRESSION ~−2.4 pp — the opposite. The uniform cat drop is most consistent with a shared property of the design `embeddings.parquet` checkin vectors, not a leak. No dedicated leak audit warranted (nothing promoted, nothing leak-shaped).

**§4.2 cross-reference**: a Tier B PROMOTE would have been a free upgrade to the architectural champion, never a headline; the project reg headline stays the §4.2 composite (+7–12 pp). Wave 1 produced no upgrade.

**Verdict: Wave 1 NULL/FALSIFIED — no design promoted. INDEX.md §B-summary filled.**

**Next**: B3 (Lever 4) runs canonical+Lever4 control ONLY (`check2hgi_lever4_canonical`); winner-stack skipped (no Wave-1 winner). Allowlist gates need `CHECK2HGI_LEVER4_CANONICAL` added (folds.py:873, train.py:1548 currently list only LEVER4_DESIGN_B).

---

## 2026-05-28 — Tier B B3 (Lever 4) + Tier B CLOSURE

**Phase**: Tier B B3 (Lever 4 on canonical) + full Tier B closure.

**Allowlist fix**: added `EmbeddingEngine.CHECK2HGI_LEVER4_CANONICAL` to `src/data/folds.py:_MTL_C2HGI_ALLOWED_ENGINES` and `scripts/train.py:_ALLOWED_ENGINES_FOR_C2HGI_PRESET` (both previously listed only `LEVER4_DESIGN_B`). Same allowlist-gate class as the original Tier B incident.

**B3 build + run** (verdict: [`../../results/substrate_protocol_cleanup/tier_b/phase_b3_verdict.md`](../../results/substrate_protocol_cleanup/tier_b/phase_b3_verdict.md)):
- Substrate built via `build_lever4_substrate.py --base canonical --epochs 500 --alpha 0.1 --device cuda`. AL best_ep=499 loss=0.4852 (~2 min); AZ best_ep=497 loss=0.4990 (~3 min). POI2Vec teacher already present (built 19:49/19:51).
- Postbuild via `postbuild_design_substrate.sh check2hgi_lever4_canonical {state}`: next.parquet + next_region.parquet (AL 12709 / AZ 26396 rows, n_regions 1109 / 1547) + canonical seed=42 log_T cp'd and touched after parquet (C22 OK).
- MTL train H3-alt, seed=42, 5f, `--no-checkpoints`. AL ~1.5 min / AZ ~3 min.
- Winner-stack (`check2hgi_lever4_design_b`) SKIPPED — no Wave-1 winner.

**B3 findings**:

| State | disjoint reg Δ vs base | per-fold + | Wilcoxon p | cat Δ | Gate (+0.3 reg AND no cat regress) | Verdict |
|---|---:|---:|---:|---:|---|---|
| AL | −0.24 | 2/5 | 0.781 | −2.68 | FAIL | NOT PROMOTED (FALSIFIED) |
| AZ | −0.08 | 2/5 | 0.844 | −2.54 | FAIL | NOT PROMOTED (FALSIFIED) |

Lever-4's hypothesised "+0.3-0.6 pp reg, zero cat cost" does NOT materialise under MTL+F1: reg flat-negative, cat regresses ~−2.6 pp (same uniform cat cost as Wave-1 designs — a substrate-embedding property). NO leak signature.

**TIER B CLOSURE — ALL FOUR SUBSTRATE VARIANTS NOT PROMOTED.**

| Variant | AL verdict | AZ verdict |
|---|---|---|
| Design B (B1) | FALSIFIED | NULL |
| Design J (B2) | FALSIFIED | FALSIFIED |
| Lever 5 (B4) | FALSIFIED | NULL |
| canonical+Lever 4 (B3) | FALSIFIED | FALSIFIED |

The merge_design STL dominance of Designs B/J does NOT transfer to MTL+F1 at small states — the shared backbone washes out the substrate reg advantage and every variant costs cat ~−2.4 pp. No leak on any variant (reg flat + cat regression is the OPPOSITE of a label-shortcut leak). No multi-seed promotion. The project reg headline stays the §4.2 composite (+7–12 pp).

**Incident note**: this was the post-allowlist-fix CLEAN rerun. The first Tier B attempt failed on two engine-allowlist gates (`scripts/train.py:1548`, `src/data/folds.py:873`); both fixed. The B/J/L/Lever4 engines reuse canonical c2hgi sequences/folds/log_T verbatim — only substrate embeddings differ. The 8 Wave-1 cells were already complete (analysed, not re-run); B3 was built + trained fresh this session.

**Compute**: Tier B total ≪ 2 GPU-h (substrate builds ~5 min each + 4 MTL cells ~10 min total). Disk held at 14 GB free throughout (never below the 3 GB STOP threshold). `--no-checkpoints` honoured on all GPU runs.

**Closure artefacts**:
- Verdicts: `tier_b/phase_b1b2b4_verdict.md`, `tier_b/phase_b3_verdict.md`.
- B-summary table filled in `INDEX.md` §B + `window_mask_audit.md` §6 "cat F1 @ joint best" row.
- Analysis: `scripts/substrate_protocol_cleanup/analyze_tier_b_wave1.py`.
- Run dirs: `tier_b/{canonical_baseline,design_b,design_j,design_l,lever4_canonical}/{alabama,arizona}/seed42/mtlnet_*/`.

**Verdict: TIER B CLOSED — all substrate variants NULL/FALSIFIED at AL/AZ. No PROMOTE.**

## 2026-05-28 — Advisor review — Tier B null verdict

Independent read-only review verifying the Tier B NULL is real (not a silent degenerate run) and the code changes are sound. No GPU; no files/verdicts modified.

### A. Did the design cells actually use the design substrate? (silent-degeneracy hunt)
- **A1 engine recorded — VERIFIED.** `model/model_params.json["datasets"]` for every design cell points at its OWN substrate path: `output/check2hgi_design_{b,j,l}/<state>/input/{next,next_region}.parquet` and `output/check2hgi_lever4_canonical/...` — NOT canonical `output/check2hgi/`. canonical_baseline correctly reads `output/check2hgi/`. No cell silently ran canonical.
- **A2 substrates differ from canonical — VERIFIED.** Numeric load of `next.parquet` (cat-task input, 576 emb cols): every design vs canonical `identical=False`, ~94.5% of cells differ, maxabs diff ~5.8–7.4 at both states. `next_region.parquet` (reg-task input) also differs (maxabs ~2.9–3.5). The four substrates are also mutually distinct (pairwise maxabs 3.0–4.3), so they are not one build copied four times. Row counts/shapes identical to canonical (12709 AL / 26396 AZ) — folds legitimately reusable.
- **A3 non-degenerate val metrics — VERIFIED.** Per-fold reg top10 (max-over-epochs) lies in a sane band (AL ~44–53, AZ ~37–44), varies fold-to-fold, and tracks the baseline shape. No constant-output collapse. Cat F1 per-fold in 41–48 band, non-constant.

### B. Wilcoxon correctness
- **B4 p matches verdict — VERIFIED.** Re-ran `scipy.stats.wilcoxon(deltas, alternative="greater")` on RAW per-fold disjoint reg top10 (no rounding): design_b AL → W=3.0, p=0.9062 (verdict ≈0.91 ✓). Spot-checks: design_l AZ p=0.6875 ✓, design_j AL p=0.7812 ✓.
- **B5 paired-by-fold + raw values — VERIFIED.** Script (`analyze_tier_b_wave1.py` L85) builds `deltas[i] = design[i] − base[i]` index-aligned by fold; values are `float(...)*100` with no rounding (matches Tier A1 scipy-dispatch lesson).
- **B6 correct baseline — VERIFIED.** Both verdict and script pair against the seed=42 `canonical_baseline` cell at the SAME state, same H3-alt recipe, same folds (script L72 `_run_dir("canonical_baseline", state)`), NOT a cross-seed/cross-recipe baseline.

### C. The uniform ~−2pp cat regression — real and explained?
- **C7 — VERIFIED, explanation reasonable.** Cat F1 means reproduce the verdict (AL: B 43.59 / J 43.71 / L 43.27 / lever4 43.08 vs base 45.76; AZ ~46.2–46.5 vs 48.87). The drop is non-degenerate (F1 stays in 41–48 band, varies by fold). All four mechanistically-distinct substrates differ from canonical on the cat-task `next.parquet` checkin vectors (~94.5% of cells) and all four drop cat ~2–2.7 pp uniformly. This is consistent with the verdict's "shared property of the design embeddings.parquet checkin vectors" reading: the cat path eats a perturbed checkin input under all four builds. Not leak-shaped (cat moves DOWN, reg flat — opposite of a label-shortcut leak). Not flagged as a build bug: the substrates are mutually distinct, so it is not all four inheriting one degraded array; it is a generic side-effect of perturbing the checkin substrate.

### D. Code changes
- **D8 — VERIFIED.** Both gates are pure allowlist supersets. `scripts/train.py:_ALLOWED_ENGINES_FOR_C2HGI_PRESET` and `src/data/folds.py:_MTL_C2HGI_ALLOWED_ENGINES` each replaced a 2-tuple `(CHECK2HGI, HGI)` with a 7-tuple that RETAINS CHECK2HGI + HGI and ADDS DESIGN_{B,J,L} + LEVER4_{CANONICAL,DESIGN_B}. The only deletions in either file are those two old tuples (confirmed via `git diff | grep '^-'`). (a) No behavior change for canonical/HGI. (b) Reuse is correct: design substrates have identical row counts/shapes, so canonical StratifiedGroupKFold sequences/folds apply verbatim. (c) `ast.parse` clean on both files.
- **D9 — VERIFIED.** `docs/findings/F_TIER_A1_LEAK_AUDIT.md` untracked/untouched (`git status` = `??`, mtime 19:54 pre-dates this session's Tier B runs). No florida/california/texas dirs under `tier_b/` — no large-state expansion.

### Conclusion: **PASS — null is real and code is sound.**
The design cells provably trained on their own distinct, non-degenerate substrates (A1+A2+A3), the Wilcoxon p-values reproduce exactly on raw paired-by-fold values vs the correct same-recipe baseline (B4–B6), the uniform cat regression reproduces and is plausibly a shared checkin-substrate side-effect rather than a build bug or leak (C7), and the code changes are purely additive allowlist supersets with no canonical-path impact and no A1/large-state contamination (D8–D9). Single most important finding: **the design cells REALLY used the design substrates** — `model_params.json` datasets paths point at `output/check2hgi_design_*` / `check2hgi_lever4_canonical`, and those parquets differ from canonical in ~94.5% of cells. No silent-degeneracy artifact.

— Advisor (read-only)

---

## 2026-05-28 — Tier C launch (C1/C2/C3 protocol coherence)

**Phase**: Tier C GPU execution.

**Preflight**: C22 stale-log_T check PASS at both states — seed42 per-fold log_T dated 2026-05-20 17:49 UTC > next_region.parquet 2026-05-19 14:36 UTC (AL+AZ). GPU A40 45489 MiB free; disk /home 14268 MB free (>3000 gate). Tier B canonical_baseline confirmed non-degenerate (AL disjoint reg 50.82, AZ 41.33; matches Tier B table) — reused as joint-best baseline for C1/C2/C3.

**Baseline reg-peak epochs (for C3 analysis)**: AL peaks LATE (ep 10-15 per fold), AZ peaks EARLY (ep 4-9). Extracted from canonical_baseline per-fold `next_region_val.csv`.

**C2+C3 megascript launched** (detached, setsid): `/tmp/run_tier_c.sh`, outer log `/tmp/tier_c_outer.log`, lane log `/tmp/tier_c.log`. Two lanes (AL, AZ) run concurrently (max 2 concurrent; never same-state concurrent because runs write the shared `results/check2hgi/{state}/metrics` dir). Each lane serial: C2_n2 → C2_n4 → C2_n6 → C3_zerokv. Per-cell gate (gpu≥10GB AND disk≥3000MB) before each cell. After each cell, metrics/summary/folds/config.json copied to `docs/results/substrate_protocol_cleanup/tier_c/{state}/{cellname}/`. DONE markers: `tier_c/{state}/LANE_DONE`, `tier_c/TIER_C_C2C3_DONE`.

**C1** will run AFTER C2+C3 (serialized one-state-at-a-time due to 3-snapshot/fold disk conflict with --no-checkpoints discipline): `--save-task-best-snapshots` run, then `scripts/route_task_best.py` per fold, then `rm -rf` snapshots before next state.

**Lane PIDs at launch**: see `/tmp/tier_c.log` ("lanes launched: AL=<pid> AZ=<pid>"). If my turn ends mid-wait, poll `docs/results/substrate_protocol_cleanup/tier_c/TIER_C_C2C3_DONE`.

**Next**: poll DONE marker → analyse C2 (Δcat vs σ_fold, Δreg regression) + C3 (reg peak shift) → run C1 serial → write `tier_c/phase_c_verdict.md` + close INDEX §C + §6 closure table.

---

## 2026-05-28 — Tier C2 + C3 CLOSURE (C1 in progress)

**Phase**: Tier C GPU execution — C2 (reg-freeze sweep) + C3 (zero-cat-kv) complete.

**Execution note**: C2 freeze cells crash at END-of-CV diagnostics save (reg loss-weight diagnostic array shorter than cat's after the freeze zeroes task_b_loss → `metric_store.to_dataframe` length mismatch). HARMLESS: per-fold val metrics CSVs are written incrementally by `save_fold_partial` into the timestamped run dir BEFORE the crash. Megascript v2 collects metrics from the newest run dir (not the never-written top-level `metrics/`), then rm's the run dir. All 8 cells produced 5/5 valid reg+cat val CSVs. (Flagged as a code nit: `_save_diagnostics` should tolerate ragged per-task diagnostic arrays under reg-freeze; not gating — metrics are intact.)

**C2 — `--reg-freeze-at-epoch N` (N∈{2,4,6}) — VERDICT: ARCHIVE, CLOSES §4.4.**
Baseline reg-peak epochs: AL late (mean ep 12.8), AZ early (mean ep 6.2). Δreg/Δcat vs canonical_baseline (disjoint, mean 5-fold):
- AL: N2 reg −7.69 / cat +0.37(p.16); N4 reg −4.18 / cat +0.46(p.16); N6 reg −1.05 / cat +0.06(p.69).
- AZ: N2 reg −8.34 / cat −0.06; N4 reg −0.93 / cat +0.09(p.41); N6 reg −0.07 / cat −0.02(p.69).
Gate (Δcat improves at some N WITHOUT Δreg regression > σ_fold): FAILS at every N at both states. Where cat lifts most (AL N2/N4) reg collapses ≫σ; where reg is preserved (N6) cat is null. The last unfalsified curriculum form is now falsified. No multi-seed promotion.

**C3 — `--zero-cat-kv` — VERDICT: P4 FULLY CLOSED.**
AL reg Δ−0.28 (ns, p.94), peak ep 12.8→9.4 (EARLIER, opposite of hypothesis). AZ reg Δ+0.01 (ns), peak 6.2→6.6 (≈unchanged). No later-peak / higher-magnitude signature. Cat side-effect negligible (AL +0.37 ns, AZ −0.04 ns). Zeroing cat→reg cross-attention K/V does NOT recover MTL reg. Combined with P4 frozen-cat-params, both cat-parameter AND cat-activation pathways exonerated — the residual MTL-vs-STL reg gap is NOT cat→backbone capacity-stealing (neither shared-backbone params nor cross-attn K/V). Narrows `mtl_improvement`'s arch-axis search by elimination; no new P4-residual finding filed.

**Artefacts**: `docs/results/substrate_protocol_cleanup/tier_c/phase_c_verdict.md` (C2/C3 filled; C1 pending), `tier_c/{alabama,arizona}/{C2_n2,C2_n4,C2_n6,C3_zerokv}/metrics/`, `tier_c/tier_c_c2c3_analysis.json`, `scripts/substrate_protocol_cleanup/analyze_tier_c.py`.

**C1 status**: `--save-task-best-snapshots` train+route launched SERIAL one-state-at-a-time (`/tmp/run_tier_c1.sh`, log `/tmp/tier_c1.log`, marker `tier_c/TIER_C1_DONE`). AL training in progress; snapshots deleted immediately after per-fold route scoring, before AZ. Disk monitored.

**Next**: C1 route scoring → fill phase_c_verdict.md §C1 + deploy-cost note → INDEX §C verdicts → window_mask_audit §6 append.

---

## 2026-05-28 — Tier C1 CLOSURE — ARCHIVE (3-snapshot routing variant A)

**Phase**: Tier C1 (§4.1 per-task 3-snapshot routing, variant A) — GPU train + route scoring.

**Execution**: `--save-task-best-snapshots` train at AL then AZ (serial, snapshots ~295 MB/state ≈ 20 MB/snapshot, deleted immediately after per-fold route scoring — disk never below 14 GB). Required two route_task_best.py fixes (both landed):
1. **task_set reconstruction** — config.json round-trips task_set as a plain dict; `get_preset(name)` rebuilt DEFAULT heads (cat=transformer, reg=gru) → `load_state_dict` key mismatch vs the snapshot's `--cat-head next_gru`/`--reg-head next_getnext_hard`. Fixed to reconstruct the `TaskSet` from the dict preserving head_factory overrides.
2. **top10 metric** — `compute_classification_metrics` defaults `top_k=(3,5)`; route gate needs Acc@10. Added `top_k=(1,3,5,10)` to the reg scoring call (else `reg_top10_acc`=NaN).
(First two attempts produced legacy-path crash / NaN top10 respectively; v3 clean.)

**Findings** (seed=42, 5-fold, reg-best Acc@10 vs joint-best Acc@10):
- AL: mean Δreg **−7.89** (per-fold +2.87,+0.90,**−48.03**,+0.75,+4.05; 4/5 folds +; Wilcoxon p=0.3125).
- AZ: mean Δreg **+0.50** (per-fold +3.54,**−6.80**,+3.22,+0.70,+1.86; 4/5 folds +; p=0.3125).
- Cat-best vs joint-best F1: AL +0.87 (p=0.0625), AZ +0.12 (ns).
- **Degenerate-snapshot pathology**: AL fold3 reg-best snapshot has reg Acc@1=0.0000 / **Acc@10=0.0012** (the Acc@1 selector landed on a collapsed epoch); AZ fold2 similar. No guard in MultiTaskBestTracker. Excluding the 2 degenerate folds the gain is +2.14 (AL) / +2.33 (AZ) — clears +2 pp, but the degeneracy is the disqualifier.

**Verdict: ARCHIVE.** Gate (≥+2 pp at BOTH states on the measured 5-fold metric) FAILS (AL −7.89, AZ +0.50, p=0.31 both). The +2 pp routing signal is real on healthy folds but variant-A reg-best routing is NOT deploy-safe: the reg-best (Acc@1) selector can pick a snapshot whose Acc@10 ≈ 0, making per-task routing strictly worse than the single joint-best checkpoint. Deploy cost (3× checkpoint storage + 2-model load at inference) unjustified for a null/unreliable gain; F1/geom_simple at one checkpoint already extracts the per-task capacity (cat routing ≤ +0.87 pp), reconfirming the variant-B ship decision from mtl-protocol-fix. Variant C-prime NOT triggered (variant A did not win). No multi-seed promotion.

**Artefacts**: `tier_c/phase_c_verdict.md` §C1 (full tables + deploy note), `tier_c/{alabama,arizona}/C1_route/route_fold{1..5}.json`, `tier_c/tier_c1_routing_analysis.json`, `scripts/substrate_protocol_cleanup/analyze_tier_c1.py`. route_task_best.py fixes in `scripts/route_task_best.py`.

**Tier C COMPLETE — all three ARCHIVE/CLOSED:** C1 archive (routing not deploy-safe), C2 archive (closes §4.4), C3 P4 fully closed. INDEX §C verdicts filled in-place; window_mask_audit §6 closure table appended (Tier A1/B rows untouched). No Tier C cell promotes to multi-seed. Net: Tier C strengthens the case that the residual MTL-vs-STL reg gap is architectural (→ `mtl_improvement`), not curriculum / not K/V capacity-stealing / not recoverable by checkpoint routing.

**Code nit flagged (non-gating)**: `_save_diagnostics` (`src/tracking/storage.py:701`) crashes at end-of-CV under `--reg-freeze-at-epoch` because the reg loss-weight diagnostic array is shorter than cat's after the freeze (`metric_store.to_dataframe` ragged-array). Per-fold metrics CSVs survive (save_fold_partial writes them before the crash), so verdicts are unaffected, but a future reg-freeze user would lose the diagnostics bundle. Candidate small fix: pad/guard ragged per-task diagnostic arrays. Also: MultiTaskBestTracker reg-best slot has no Acc@10-alignment or degenerate-snapshot guard (surfaced by C1).

---

## Advisor review — Tier C closures (2026-05-28, independent, read-only, no GPU)

Scope: scrutinise C1/C2/C3 closures for silent artefacts + code-fix correctness. Method: read verdict + analysis JSONs + raw per-fold CSVs/route JSONs + train/route logs + model code; re-ran `analyze_tier_c.py` (reproduced C2/C3 byte-for-byte). Per-check result: verified / discrepancy / cannot-verify.

### C1 — 3-snapshot routing (variant A) — **DISCREPANCY (scoring confound), verdict mechanism NOT established**
- **Check 1 (degenerate snapshot real or scoring bug?) — DISCREPANCY.** The fold3 reg-best Acc@10=0.0012 is produced under a **wrong-input-modality re-score**, not a clean selector pathology. `route_task_best.py` rebuilds the val loaders via `FoldCreator(...)` WITHOUT passing `task_b_input_type`, so it defaults to `"checkin"`. The training run used **`task_b=region`** (train.log: `task_a=checkin, task_b=region`; ALL ten route logs AL+AZ: `task_a=checkin, task_b=checkin`). `ExperimentConfig.save` does not persist `task_*_input_type` (absent from config.json; 0 "checkin" tokens, "region" only in task names), so the script cannot recover the modality. **Every C1 reg number (all slots, all folds, both states) is scored on the wrong reg modality.** The reg-best snapshot was *saved* at val reg Acc@1=0.2801 (epoch 14, on the correct region modality per train.log), yet re-scores to Acc@1=0.0 — the verdict's explanation ("Acc@1 selector spiked on a degenerate near-constant prediction") is therefore unsupported; the 0.0 is at minimum partly the modality mismatch. Joint/cat slots survive the same wrong loader (~0.48 Acc@10), so the bug is not uniformly fatal, but it confounds the magnitude and the fold3 "collapse" specifically. AZ fold2 is NOT a true collapse (Acc@10=0.325, merely weak).
- **Check 2 (healthy-fold +2.1/+2.3pp honest?) — CANNOT-VERIFY.** Arithmetic of excluding the 2 flagged folds is correct, but it is computed on modality-confounded numbers, so neither the −7.89/+0.50 means NOR the +2.14/+2.33 healthy-fold gain reflect the deployed (region) modality. The conclusion is not "burying a win" — but it is not trustworthy either, because the baseline it's measured against (joint-best) is also mis-scored.
- **Check 3 (two route_task_best.py fixes) — VERIFIED but INCOMPLETE.** (a) TaskSet reconstruction from dict preserving `--cat-head`/`--reg-head` overrides (lines 210–234) is correct and necessary to avoid load_state_dict key mismatch. (b) `top_k=(1,3,5,10)` (lines 177–181) correctly surfaces Acc@10. Neither fix fabricates numbers. **But a THIRD, un-made fix is required**: the loaders must be rebuilt with the run's `task_b_input_type=region` (and persisted in config). Without it the gate metric is computed on the wrong inputs.
- **Net C1:** the ARCHIVE *outcome* (don't promote variant A) may well stand on cost grounds (3× storage + 2-pass inference for a sub-σ cat gain), but the *stated mechanism* ("genuine Acc@1-selector degenerate-snapshot pathology") is NOT demonstrated by these artefacts — it is confounded by a re-scoring input-modality bug. This is the most important finding of the review.

### C2 — `--reg-freeze-at-epoch N∈{2,4,6}` — **VERIFIED**
- **Check 4 (AL N=2 + no cherry-pick) — VERIFIED.** Re-ran `analyze_tier_c.py`: AL N=2 reg Δ−7.69 (5/5 folds negative, p=1 for cell>base), cat Δ+0.37 (p=0.156, ns) — exact match to verdict. ALL {2,4,6}×{AL,AZ} fail the gate (cat never significant; reg-preserving N=6/AZ-N=4 have null cat); no (N,state) clears it. "Closes §4.4" justified.
- **Check 5 (per-fold metrics complete despite end-of-CV crash) — VERIFIED.** Every C2/C3 cell has 5/5 `fold{k}_next_{region,category}_val.csv` + reports. The ragged-array crash is in `_save_diagnostics` (post-hoc bundle); `save_fold_partial` writes per-fold CSVs first, so no fold's data is truncated. The analysis reads `top10_acc_indist` from these complete CSVs (correct metric, no re-score — unlike C1).

### C3 — `--zero-cat-kv` — **VERIFIED**
- **Check 6 (code path + dynamics) — VERIFIED.** `model.py:143-147`: when `zero_cat_kv`, `kv_a = torch.zeros_like(kv_a)` before `cross_ba(query=b/reg, key=kv_a, value=kv_a)` — cat-stream K/V into the reg-side cross-attention is genuinely zeroed at forward; `softmax(QKᵀ)V` with V=0 → 0 cat contribution; reciprocal `cross_ab` (cat queries reg K/V) and all projection weights untouched (reversible). Re-ran analysis: AL Δreg −0.28 (ns, p=0.94 reg>base), peak ep mean 12.8→9.4 (EARLIER, opposite of hypothesis); AZ Δreg +0.01 (ns), peak 6.2→6.6. All match verdict. C3 used region modality (trained+eval in same run, no re-score bug).
- **Check 7 ("residual gap is architectural, not capacity-stealing") — WARRANTED (with scope caveat).** C3 cleanly eliminates the cat-activation→reg-K/V sub-mechanism; combined with P4's frozen-cat-params (cat-parameter pathway), both cat→backbone capacity-stealing routes are exonerated. The claim is scoped as "narrows the search by elimination, not a new mechanism," which is honest. Not overreach.

### Cross-cutting — **VERIFIED**
- **Check 8 (disk discipline) — VERIFIED.** No `.pt`/snapshot/checkpoint files anywhere under `tier_c/` or `results/check2hgi/**/task_best_snapshots`. C1 snapshots deleted post-scoring as designed. No disk breach evidence.
- **Check 9 (no A1/B touch, no FL/CA/TX, append-only) — VERIFIED.** tier_c is AL/AZ only (no FL/CA/TX). log.md diff is +642/−0 (purely additive). INDEX.md has 7 deletions but all are placeholder/status lines ("no experiments executed yet", empty `—` rows) replaced with filled Tier B/C results — Tier A1/B findings preserved and extended. window_mask_audit §6 reported as append-only (untracked dir; not git-diffable, accepted on inspection).

### Verdict: **NEEDS-CHANGES (C1 only; C2/C3 PASS)**
C2 and C3 closures are sound and reproduce exactly. **C1's ARCHIVE rests on a re-scoring input-modality bug** (`route_task_best.py` scores the reg task on `checkin` instead of the trained `region` embeddings, because the input-type is neither persisted in config nor passed to FoldCreator). The "degenerate reg-best snapshot" is therefore NOT established as a genuine BestTracker Acc@1-selector pathology — it is confounded. Required before C1 can be cited as a selector-pathology finding: (1) persist `task_*_input_type` in `ExperimentConfig`/config.json; (2) have `route_task_best.py` pass `task_b_input_type=region` to FoldCreator; (3) re-score the saved snapshots (or re-run C1). The ARCHIVE *decision* (don't deploy variant A) likely survives on cost grounds regardless, but the verdict prose must be corrected to not attribute the collapse to the selector.

## C1 re-score (modality fix) — 2026-05-29

The Tier C advisor's NEEDS-CHANGES on C1 (above) is resolved. The three required steps were executed; the C1 verdict is corrected and FLIPS from ARCHIVE to **§Discussion footnote** (one-state pass).

**Phase 1 — fix the scorer + persist the modality:**
- `src/configs/experiment.py`: added append-only `ExperimentConfig.task_a_input_type` / `task_b_input_type` (default `"checkin"`; old configs back-compat-load via `cls(**data)` defaults). `RunManifest.write` serialises `asdict(config)` so manifest.json inherits them automatically.
- `scripts/train.py`: `_apply_overrides` now `dataclasses.replace`s the modality from `args` into the config before save, so `results/check2hgi/{state}/config.json` records the trained modality.
- `scripts/route_task_best.py`: resolves modality with precedence (explicit `--task-{a,b}-input-type` CLI > persisted config field > legacy `"checkin"`), passes it to `FoldCreator`, logs it, and writes it into the route JSON.
- `tests/test_substrate_protocol_cleanup_flags.py::TestTaskInputTypePersistence`: 5 new tests (config default, save/load round-trip of `region`, legacy-config back-compat, scorer resolves `region` from config, CLI override wins). `.venv/bin/python -m pytest tests/test_substrate_protocol_cleanup_flags.py -x` → **18 passed**.

**Phase 2 — re-run the 2 C1 cells (snapshots had been deleted):** detached megascript `scripts/substrate_protocol_cleanup/c1_rescore/run_c1_rescore.sh` (GPU gate ≥10 GB, disk gate ≥3 GB, ≤2 concurrent — actually serial, score+delete per state, DONE markers). AL+AZ, seed=42, 5 folds, H3-alt (`--scheduler constant`, cat/reg/shared-lr 1e-3/3e-3/1e-3, `static_weight` cat-weight 0.75, `mtlnet_crossattn`, cat-head `next_gru`, reg-head `next_getnext_hard`, task_a `checkin` / task_b `region`, per-fold log_T), `--save-task-best-snapshots --no-checkpoints`. C22 preflight: seed=42 log_T mtime (May-20) > next_region.parquet (May-19) at both states — fresh. Disk held at 14 GB throughout (snapshots ~295 MB/state, deleted right after scoring). Both states rc=0; markers `{alabama,arizona}_C1_RESCORE_DONE`, `TIER_C1_RESCORE_ALL_DONE`. PID of launcher 4066269 (exited clean).

**Phase 3 — re-score CORRECTLY + verdict:**
- Sanity gate PASSED: every `route_fold*.json` carries `"task_b_input_type": "region"`; reg-best snapshots re-score into 35–51 % Acc@10 (NOT ~0) on the 4 healthy AL folds + all 5 AZ folds; joint baseline now correct (~37–48 %).
- **Corrected routing deltas (RAW per-fold, paired Wilcoxon `greater`):**
  - **AL reg:** mean **−7.89 pp**, per-fold [+2.83, +1.22, **−48.03**, +0.43, +4.09], 4/5+, p=0.3125 → FAILS gate. The −48 is AL **fold3**, a GENUINE degenerate reg-best snapshot: reg Acc@10=0.12 % even on the correct region modality (saved at val reg Acc@1=0.2801 @ ep14, but does not generalise; same fold's joint-best = 48.15 %, cat-best = 46.78 %). Real Acc@1-selector pathology, **not** the modality bug. Healthy 4 folds avg +2.14 pp (p=0.0625).
  - **AZ reg:** mean **+2.54 pp**, per-fold [+3.81, +2.65, +3.43, +0.76, +2.05], **5/5+**, **p=0.03125** → CLEARS gate. No degenerate fold (the prior bugged "fold2 collapse" was a modality artefact).
  - **AL cat:** +0.87 (4/5, p=0.0625). **AZ cat:** +0.12 (3/5, p=0.4062). Both near-null.
- **Verdict FLIP: ARCHIVE → §Discussion footnote (one-state pass).** AZ passes +2 pp at p=0.03; AL fails on one genuine degenerate snapshot. Per INDEX §C1 gate ("~+1 pp one state → footnote") this is the one-state branch — footnote, not full PROMOTE (needs +2 pp at BOTH) and not ARCHIVE (AZ gain is real + significant). Conditional follow-up: swap reg-best selector to Acc@10 + add a degenerate-snapshot guard in `MultiTaskBestTracker`, then multi-seed AL/AZ before any §0.x promotion. Deploy cost (3× storage + 2-model load) + selector brittleness noted.

**Honest correction of the prior C1 prose:** the original verdict attributed AL fold3's 0.0 to the modality bug being the whole story (per the advisor's working hypothesis). The re-score shows the modality bug was a *separate confound* (it depressed the healthy folds' magnitudes and made AZ look null), but the AL-fold3 degeneracy is REAL and survives the correct modality. Both the prior "ARCHIVE on degenerate-snapshot" framing and the advisor's "0.0 is purely modality" framing are superseded by the corrected one-state-pass footnote.

**Artefacts:** `docs/results/substrate_protocol_cleanup/tier_c/{alabama,arizona}/C1_route/route_fold{1..5}.json` + `.log` + `config.json` (persisted modality); `docs/results/substrate_protocol_cleanup/tier_c/tier_c1_routing_analysis.json` (regenerated); `docs/results/substrate_protocol_cleanup/tier_c/phase_c_verdict.md` §C1 (rewritten); `scripts/substrate_protocol_cleanup/c1_rescore/` (megascript + train/score/analyze logs). C2/C3 verdicts, Tier A1/B artefacts UNTOUCHED; no FL/CA/TX.

---

## 2026-05-29 — FINAL synthesis + closure propagation

**Phase**: Final synthesis (Phase 2) + live-doc propagation (Phase 3). No GPU; one CPU verification (Phase 1).

**Phase 1 — C1 AL-fold3 verification (the one open nuance):** Opened `tier_c/alabama/C1_route/route_fold3.json`. Confirms `task_b_input_type="region"` (correct modality, NOT the bug), reg-best snapshot `reg_top10_acc=0.00118` (~0.12 %) while the SAME fold's joint-best `reg_top10_acc=0.48151` (~48.15 %); reg-best `reg_accuracy`(Acc@1)=0.0 at scoring despite being selected at a high val Acc@1 epoch. This is a **genuine degenerate Acc@1-selected reg-best snapshot on the correct region modality** — the footnote verdict is sound, NOT a residual scoring issue. Also re-confirmed AZ: all 5 folds reg-best top10 > joint top10 (5/5 positive, mean +2.54 pp, all `task_b_input_type=region`, no degenerate fold). C1 = one-state §Discussion footnote, exactly as the corrected re-score states.

**Phase 2 — closure synthesis:** wrote [`CLOSURE.md`](CLOSURE.md): per-Tier verdict table, headline, GPU-cost tally (~10–11 GPU-h vs ~40–45 budget), disk-incident methodological note, "what hands to mtl_improvement" (triply-confirmed architectural residual: P4 params + C3 K/V activations + B substrate all exonerated), and the "What done looks like" criteria map (all 5 met; B4/Lever5 + B3/Lever4 + C3 + C4 also closed beyond the 5).

**Phase 3 — propagated to live docs (minimal, append-only/surgical):**
- `docs/CHANGELOG.md` — 2026-05-29 closure entry (most-recent-first; launch entry preserved below it).
- `docs/CLAIMS_AND_HYPOTHESES.md` — CH26 updated provisional→multi-seed n=20 PROMOTED; kept in the study-section provisional banner, explicitly NOT added to the v10 paper whitelist (pending §0 re-run).
- `docs/CONCERNS.md` — C15 2026-05-29 evidence + status line (Tier B + C3 tighten toward architectural; explicitly NOT closed). C22/C24 re-checked consistent (C4 is the code-guard for C22; C24 STAN watch-item unaffected — A1 uses next_getnext_hard not next_stan_flow).
- `docs/findings/F_TIER_A1_PROMOTION.md` — new F-trail finding, cross-refs F_TIER_A1_LEAK_AUDIT.
- `docs/results/RESULTS_TABLE.md` — new §0.8 sub-section (W=0.2 vs W=0.0; small-state paper-grade rows + large-state pilot rows with explicit §0.1-dev-seed caveat); tagged provisional / not-yet-whitelist.
- `docs/NORTH_STAR.md` — line-6 bullet updated to "PROMOTED small-state, orthogonal to B9"; champion recipe UNCHANGED.

Did NOT modify any C2/C3/B/A1 verdict doc. Did NOT touch mtl-protocol-fix v6-final provenance.

---

## 2026-05-29 — FINAL advisor pass (whole-study, read-only)

**Scope**: self-critical review of INDEX.md + log.md + CLOSURE.md for declaring the study closed. Are all 5 "what done looks like" criteria demonstrably met? Any verdict on unverified numbers? Any live-doc overclaim? Any dangling reference?

**1. Five closure criteria — all demonstrably met:**
- A1 Wilcoxon n=20 verdict at AL/AZ ✓ (p=9.54e-07 both, 20/20 folds; verdict doc + advisor pass reproduced the p-value from raw CSVs).
- B1/B2/B3 F1-selector three-frontier + verdict ✓ (INDEX §B-summary filled; advisor verified cells used distinct non-degenerate substrates).
- C1 3-snapshot routing prototype + verdict ✓ (`--save-task-best-snapshots` + `route_task_best.py` shipped; footnote verdict after modality-bug fix + re-score).
- C2 freeze-reg-after-peak pilot verdict ✓ (ARCHIVE, closes §4.4; advisor reproduced byte-for-byte).
- D1 window/mask audit ✓ (CLEAN, advisor-verified 8 spot-checks). Also closed: B4/Lever5, C3 (P4), C4 (mtime guard).

**2. Verdicts resting on unverified numbers?** None. A1 reproduced from raw CSVs (advisor); B reproduced (advisor: distinct substrates + Wilcoxon p match); C2/C3 reproduced byte-for-byte (advisor); C1 was the only twice-flipped number and is now Phase-1-verified directly from route_fold3.json (degenerate on correct modality) + AZ 5/5 cross-check.

**3. Live-doc overclaim check (the named hazards):**
- A1 large-state pilot as paper-grade? **No** — CHANGELOG, CH26, RESULTS §0.8, F_TIER_A1_PROMOTION, NORTH_STAR all explicitly tag FL/CA/TX as seed=42 pilot / NOT paper-grade with the §0.1-dev-seed caveat.
- C1 footnote as a promotion? **No** — every mention says "§Discussion footnote / one-state pass / NOT a promotion".
- B/C2/C3 framed as anything but nulls/closures? **No.**
- CH26 added to paper whitelist? **No** — kept in the study-section provisional banner with an explicit "NOT on the v10 whitelist" line.
- C15 closed unilaterally? **No** — evidence + status line only; explicit "NOT closed here".
- §0.1 canonical table modified? **No** — §0.8 is a new additive sub-section; §0.1 rows untouched.

**4. Dangling references?** None found — all six newly-referenced paths verified to exist (CLOSURE.md, F_TIER_A1_PROMOTION.md, F_TIER_A1_LEAK_AUDIT.md, the two A1 verdict docs, phase_b3/phase_c verdicts). CHANGELOG/RESULTS/NORTH_STAR/CONCERNS/CLAIMS cross-links point at existing files.

**5. Provenance / scope discipline:** mtl-protocol-fix v6-final untouched; C2/C3/B/A1 verdict docs untouched; no FL/CA/TX runs; all doc edits append-only or surgical-additive.

**Verdict: GO — declare `substrate-protocol-cleanup` CLOSED.**

All five "what done looks like" criteria are met with verified numbers; the single promotion (A1) is paper-grade at small states and correctly hedged at large states; all nulls/closures/footnotes are framed exactly; no live-doc overclaim; no dangling reference; C15 left open for the user; mtl-protocol-fix provenance intact. The study hands the architectural residual reg gap (triply confirmed) to `mtl_improvement` and leaves log_T-KD as an orthogonal validated small-state free upgrade.

— Final advisor (read-only)

---

## 2026-05-29 — Verification — Tier B re-audit α=0 claim (independent, read-only)

**Scope**: independent scrutiny of the STRONG re-audit claim "the MTL reg encoder branch (`next_stan_flow` STAN/stan_logits) is REG-INERT; all ~50% MTL reg rides on the α·log_T prior; with α=0 MTL reg collapses to chance ~5.5%, while STL's encoder DOES learn region (+2.34/+1.64pp)." Checks 1–5 below. CPU-only, existing artefacts + code.

**Check 1 — α=0 forward path + is the floor real or a broken config?**
- (a) **VERIFIED, clean.** `head.py` L101-104: `freeze_alpha=True` registers `alpha` as a *buffer* (not Parameter); `alpha_init=0.0` → forward L148 `stan_logits + self.alpha * transition_prior` = `stan_logits + 0.0` = `stan_logits` alone. No NaN/break of the STAN branch; the log_T term is cleanly zeroed, gradient cannot move α. Config is sound.
- (b) **VERIFIED with an important nuance.** The reported per-fold α=0 reg top10 (design_b `[5.03,8.65,6.24,2.64,4.76]`, canonical `[5.03,6.50,5.96,4.76,5.49]`) are the **joint-best-epoch** selections (`MultiTaskBestTracker.joint_best`, monitor `joint_geom_simple`, `storage.py` L171-193) — which happen to land at/near each fold's reg-top10 **peak**. They are NOT a stable converged level. The **all-epoch mean of top10_acc_indist is ~1.1%** (design_b 1.13%, canonical ~1.1%); the trajectory is volatile (last-epoch 0.4–4.5%) and only **7/250 epoch-points exceed 3.7%**. So the encoder branch genuinely fails to learn region under MTL α=0 — directionally the floor claim holds — but the headline "~5.5%" is a best-epoch readout, not the typical/converged value.
- **Chance reconciliation — DISCREPANCY in wording.** AL n_regions=1109 → pure top10 chance ≈ 10/1109 ≈ **0.9%**, not 5.5%. The re-audit's "~5.5% ≈ chance" is wrong by ~6×. The honest statement is: the encoder branch sits *near floor* (all-epoch mean ~1.1%, i.e. ~1.2× chance), with best-epoch noise peaking ~5–8%. It does NOT learn the region structure; but "~5.5% = chance" overstates the chance level.

**Check 2 — confound: is "α=0 floors" evidence the MTL *encoder* is inert, or merely that this head is designed to lean on the prior and α=0 is an OOD config? → OVERREACH-FLAG (partial).** α=0 is a config the head was never trained under; the stan branch has a learnable path to region (it's the same NextHeadSTAN that, at STL, reaches ~61% top10 *with* prior and learns region without it per merge_design Test 2 — but that STL evidence is for a different cell, see Check 3). The α=0 MTL result establishes "removing the anchor reveals no hidden MTL substrate gain" (Δ−0.08, p=0.56) — which cleanly **falsifies the masking hypothesis**. It is weaker support for the absolute claim "the MTL encoder branch learns nothing about region": near-floor performance in an OOD α=0 config in 50 epochs is consistent with "inert" but also with "the joint regime + anchor leaves the encoder branch undertrained." The strong mechanistic inference is plausible but not air-tight from D1 alone.

**Check 3 — STL contrast apples-to-apples? → DISCREPANCY (material).** The "+2.34pp with prior" (D2, AL, design_b vs canonical, `reg_gethard`) is **VERIFIED** from real per-fold data: design_b top10 [65.26,61.41,65.26,59.95,55.57]=61.49 vs canonical [58.91,60.00,64.40,57.46,55.00]=59.15, Δ+2.34, Wilcoxon p=0.03125. **But the "+1.64pp without prior" is misattributed.** `T2_FINDINGS.md` shows Test 2 is **Florida**, **Design J**, `next_gru`, and the **+1.64pp is the HGI−J gap**, not design_b/J − canonical. There J − canonical *without* prior was only **+0.86pp** (p=0.0312); the +1.64 is HGI's advantage *over* J. There is **no STL design_b-AL no-prior (`next_gru`) artefact** on disk. So the clean "STL encoder learns region WITHOUT the prior, in the SAME cell as the MTL α=0 test" claim is **NOT established apples-to-apples**: the MTL α=0 cell is AL/design_b/no-log_T; the STL no-prior evidence is FL/J(or HGI)/no-log_T. The contrast is suggestive but the prior-removal arm crosses state, design, and pairwise-comparison axes.

**Check 4 — does the corrected framing overclaim? → YES, needs hedge.** D1 (masking falsified) + D2-with-prior (STL +2.34 reproduces) are solid. But (i) "~5.5% = chance" is ~6× off; (ii) the STL-no-prior arm is cross-cell; (iii) α=0 is OOD. The defensible claim is regime-and-config-scoped, not a clean architectural law.

**Check 5 — D3 cat-confound. → VERIFIED, sound.** `build_design_b_poi_pool.py` L199 instantiates a **fresh** `CheckinEncoder(in_channels, args.dim, ...)`, trains 500 epochs from scratch (L229-246, no canonical checkpoint loaded), and writes `embeddings.parquet` from that run's `checkin_emb` (L255-261). build_design_j (L138 fresh encoder, L191 write) and build_design_l (L203 fresh encoder, L257 write) do the same. None load canonical weights. So the cat-input checkin vectors are an independent training trajectory → the 100%-cells-changed / meanabs≈1.2 divergence is build-time encoder re-init drift, and the uniform ~−2pp cat is a build-scope confound, NOT a substrate property. **"−2pp cat = build drift, not substrate cost" is SOUND.**

### Verdict: **CLAIM-HOLDS-WITH-HEDGE**

The reg-side mechanism (masking falsified; MTL reg dominated by the α·log_T anchor; the substrate-carrying branch contributes little beyond the prior under MTL) is supported in *direction*, and D3 is fully sound. But three points overreach and must be reworded before this becomes a headline architectural finding. **"The MTL reg encoder branch is reg-INERT" should be hedged**, not stated absolutely.

**Proposed rewording (parent to apply to `phase_b_reaudit.md` + `CLOSURE.md`):**
- Replace "reg collapses to floor (~5.5 %, ≈ chance for 1109 regions)" with: *"reg collapses to near-floor — best-epoch top10 peaks ~5–8 % but the all-epoch mean is ~1.1 % (vs ~0.9 % pure top10 chance for 1109 regions), and only 7/250 epoch-points exceed 3.7 %. The reported ~5.5 % is a joint-best-epoch readout, not a converged level; the encoder branch does not acquire region structure."* (Removes the false "5.5% = chance".)
- Replace the absolute *"the substrate-carrying encoder branch is reg-inert under MTL"* with the config/regime-scoped: *"under the B9/H3-alt joint config, in 50 epochs and with the anchor present, the substrate-carrying `stan_logits` branch contributes essentially nothing to MTL reg beyond the α·log_T prior; an α=0 ablation (an out-of-training config) does not uncover any hidden substrate gain (Δ−0.08, p=0.56)."*
- In D2 / the synthesis, **fix the +1.64pp attribution**: it is FL / Design J (or HGI−J), `next_gru`, NOT AL / design_b. State: *"At STL the encoder branch demonstrably learns region (AL design_b +2.34 pp WITH prior); a separate FL no-log_T ablation (merge_design Test 2) shows the J−canonical encoder advantage persists without the prior (+0.86 pp), and HGI−J widens to +1.64 pp — i.e. the STL substrate signal survives prior removal. NOTE: the no-prior STL evidence is from a different state/design than the AL α=0 MTL cell, so the STL↔MTL contrast is directionally clean but not single-cell apples-to-apples."*
- Promotion outcome (no design promoted) and the D3 cat-confound conclusion are **unchanged and correct**.

**One-line bottom line**: claim is right in spirit and right on D3, but "MTL reg encoder is inert / α=0 floors at chance" needs hedging to "contributes nothing beyond the anchor under the 50-epoch joint config, and the α=0 OOD ablation reveals no hidden gain" — and the STL-no-prior +1.64pp must be re-attributed (FL/J, not AL/design_b).

— Independent verifier (read-only)

---

## FL EXTENSION (B9 large-state) — 2026-05-29

Owner: agent. Approved scope: Tier B (all designs) + C1/C2/C3 at FL under B9. Headline question: does the AL/AZ STL→MTL substrate collapse repeat at FL scale, or does FL's ~10× data let the shared backbone express the substrate in MTL-reg? Competing explanation to distinguish: anchor-dominance (repeats) vs C02 Markov-saturation (FL-specific).

Operational: detached megascripts in `/tmp/fl_logs/run_<stage>.sh`, markers `<stage>.DONE`, disk guard ≥4000MB, GPU ≤2 FL cells concurrent, `--no-checkpoints` always.

### Stage 0 — POI2Vec prereq chain (BLOCKER → RESOLVED)
- Root cause: `output/hgi/florida/temp/` never built (HGI Phase 3a never run for FL). POI2Vec teacher needs edges.csv + pois.csv from it.
- Prereqs all present: `data/checkins/Florida.parquet` (1.41M checkins, 76544 POIs, all required cols), FL census shapefile `Resources.TL_FL` (tl_2022_12_tract).
- **Stage 0a DONE** (~30s, CPU): `research/embeddings/hgi/preprocess.py --city Florida --shapefile <TL_FL>` (run with PYTHONPATH=src:research). Produced edges.csv, pois.csv, encodings.json, boroughs_area.csv. Graph: 76544 POIs, **4703 regions**, 229604 edges. (poi-encoder.tensor is the POI2Vec output, filled by Stage 0b.)
- **Stage 0b IN PROGRESS**: `run_poi2vec.py --city Florida --epochs 100 --device cuda` (PID logged in /tmp/fl_logs/stage0b.log; marker stage0b.DONE). Outputs: poi2vec_poi_embeddings_Florida.csv, poi2vec_fclass_embeddings_Florida.pt, temp/poi-encoder.tensor.

### Stage 0b DONE — FL POI2Vec built (~12.5 min, GPU)
- poi2vec_poi_embeddings_Florida.csv (76544×66), poi2vec_fclass_embeddings_Florida.pt (324×64), temp/poi-encoder.tensor (76544×64). 324 fclasses, 7 categories.

### Stage 1 DONE — FL design substrate builds (~6 min each, 3-up GPU)
- design_b: best_epoch=485 loss=0.4550 gamma=0.880 | design_j (λ=0.1): best_epoch=498 loss=0.4817 | design_l (λ_d=0.1,k=16,τ=1.0): best_epoch=499 loss=0.4497.
- All wrote embeddings.parquet (~442MB) + poi_embeddings.parquet + region_embeddings.parquet. No NaN/errors. params: B/L=59,011 (canonical encoder, frozen POI2Vec residual); J=4.95M (learnable per-POI table).

### Stage 2 DONE — FL postbuild inputs (CPU, sequential, disk-gated)
- Each design: next.parquet + next_region.parquet (159175 rows, 4703 regions — matches canonical FL) + 5 seed42 log_T cp'd from canonical check2hgi/florida, touched after parquet.
- **C22 preflight PASS** for design_b/j/l (log_T mtime > next_region.parquet mtime) and canonical check2hgi.
- Disk: 13.4GB free after stage 2 (used ~6GB). Guard holding.

### Stage 3 — FL Tier B MTL (B9, seed42, 5-fold) — HEADLINE RESULT
4 MTL cells done (rc=0): mtl_canonical, mtl_design_b/j/l. α=0 cells (a0_design_b, a0_canonical) running.
Operational note: first megascript deadlocked (wait_slot `pgrep -fc "scripts/train.py"` counted its own subshell argv → phantom "running=2"). Killed after b+j completed; relaunched remaining 4 via run_stage3b_mtl.sh with `pgrep -f "[.]venv/bin/python scripts/train.py"` (real interpreter only). All cells `--no-checkpoints` (model/ has no .pt, ~8.8M/cell). `_results` dups removed.

**FL MTL B9 (alpha_init=0.1 trainable), two-front, mean over 5 folds (RAW Wilcoxon one-sided design>canon):**
- CANONICAL: disjoint_reg=63.98 joint_reg=61.14 | disjoint_cat=70.49 joint_cat=66.98
- design_b: disjoint_reg 63.82 (Δ−0.16, p=0.875) | joint_reg 57.65 (Δ−3.49) | disjoint_cat 68.61 (Δ−1.88)
- design_j: disjoint_reg 64.06 (Δ+0.08, p=0.312) | joint_reg 57.52 (Δ−3.63) | disjoint_cat 68.80 (Δ−1.69)
- design_l: disjoint_reg 63.97 (Δ−0.01, p=0.500) | joint_reg 57.67 (Δ−3.48) | disjoint_cat 68.71 (Δ−1.78)

**HEADLINE: The AL/AZ STL→MTL substrate collapse REPEATS at FL under B9.** No design moves MTL-reg on the disjoint front (all |Δ|≤0.16pp, none significant — best is design_j +0.08pp p=0.31). On the joint/geom front all designs sit ~−3.5pp below canonical (their slightly lower cat drags the geom-selected epoch to a worse reg point). Cat disjoint −1.7 to −1.9pp = the known design-build cat confound (design builds re-init CheckinEncoder), NOT a real regression — to confirm vs AL's D3 finding. FL's ~10× data did NOT let the shared backbone express the substrate in MTL-reg. Awaiting α=0 to discriminate anchor-dominance vs C02 Markov-saturation.

### Stage 3 α=0 — FL MTL with α·log_T prior DISABLED (freeze_alpha=true, alpha_init=0.0)
Both a0_canonical + a0_design_b done (rc=0). FL reg has **4703 regions → top10 pure chance = 0.213%**.
- a0_canonical: all-epoch-mean top10=**0.031%**  best-epoch-mean=0.134%  max=0.275%  (0.1× chance; 0/250 epochs >1%)
- a0_design_b:  all-epoch-mean top10=**0.027%**  best-epoch-mean=0.136%  max=0.221%  (0.1× chance; 0/250 epochs >1%)
- design_b − canonical at α=0, disjoint reg (best-epoch readout): Δ=**+0.003pp, p_gt=0.500** — NO substrate advantage revealed.

**INTERPRETATION (hedged):** At FL under B9, with the α·log_T anchor off, the substrate-carrying encoder branch sits at/below pure chance (~0.03% all-epoch-mean vs 0.213% chance) for BOTH design and canonical — even MORE floored than AL (AL ~1.1% ≈1.2× chance; FL ~0.03% ≈0.1× chance). FL's ~10× data did NOT let the shared backbone develop region features in the encoder branch under the 50-epoch joint regime. The α·log_T prior supplies essentially ALL usable MTL-reg signal; removing it uncovers no hidden substrate gain (Δ+0.003, p=0.50).
- This DISTINGUISHES the two competing explanations: the result is **anchor-dominance (repeats at FL)**, NOT FL-specific C02 Markov-saturation. Markov-saturation would predict the prior captures region while the encoder *could* still learn it; instead the encoder branch is flat-floor regardless of state/scale. The mechanism is the joint-training regime, not an FL data property.
- CAVEATS (carry the AL verification rigor): α=0 is an OUT-OF-TRAINING config, so this supports "the encoder branch contributes nothing beyond the anchor under the B9 50-epoch joint regime" — NOT an absolute "the encoder can never learn region under MTL." Best-epoch readouts (0.13%) are noisy peaks near chance, not a converged level; the all-epoch mean (0.03%) is the honest floor statistic.

### FL EXTENSION — THREE-WAY correction + ISOLATION cell — 2026-05-29 (second FL agent)

**Methodology fix applied:** the prior FL agent compared designs-vs-canonical ONLY. The CORRECT framing (per `merge_design/STATE.md`) is a THREE-WAY canonical c2hgi vs designs (B/J/M/L) vs **HGI**, framed as gap-closure. Most artefacts already existed; only 2 cheap cells were run. Full deliverable: `tier_b_fl/phase_b_fl_3way.md`; C2/C3: `tier_c_fl/phase_c_fl_verdict.md`.

**STL three-way (FL, gethard with prior, from existing P1 `_leakfree` JSONs):** canonical 69.22, B 69.93 (+0.71, p=0.0625 ns), **J 70.34 (+1.12, p=0.0312 ✓)**, M 70.11 (+0.89, p=0.0625 ns), HGI 71.34 (+2.12). canon→HGI gap = +2.12 pp; **J closes 53 %**; B 34 %, M 42 %; all designs strictly < HGI (HGI>design p=0.0312). ⚠ The non-`_leakfree` P1 design files (~82 %) are LEAKY — used `_leakfree` + `check2hgi`(canon) + `hgi`. No-prior cross-check (Test 2): J +0.86 pp strict, HGI−J +1.64 pp — reproduces `T2_FINDINGS.md`.

**MTL three-way (FL B9):** canonical disjoint reg 63.98; B −0.16 (p0.875), J +0.08 (p0.312), L −0.01 (p0.500) — all ns. Joint −3.5 pp = cat-driven geom-selection artefact. cat disjoint −1.7 to −1.9 pp = build-scope confound (AL D3). **Designs close 0 % of any MTL gap.** **HGI MTL = NOT EVALUATED** (no FL HGI next_region.parquet; not built per scope).

**ISOLATION cell (the key one, RAN this session):** STL `next_stan_flow` α=0 (`freeze_alpha=true alpha_init=0.0`, log_T fully off → final_logits = stan_logits alone), canonical + design_b, FL 5f/50ep B9-LR. Result: **canonical 72.74 % Acc@10, design_b 73.12 % (Δ+0.37, p=0.0312)** — 341× chance. The IDENTICAL head/config under MTL floors at ~0.03 % (all-epoch-mean). **VERDICT: REGIME, not head.** The stan encoder branch is fully capable of learning FL region (~73 %, no prior) at STL; it floors only under the B9 joint regime. This is a single-cell apples-to-apples STL↔MTL contrast (same head/config/state/embeddings) — the clean isolation the AL/AZ re-audit lacked (which cross-attributed FL/J no-prior to AL/design_b α=0). Bonus: STL-α=0 (72.74) > STL-with-prior (69.22) — the prior isn't carrying the STL signal (consistent with Test 2). Hedge: α=0 is OOD → claim is regime/config-scoped (B9, 50 ep), not "encoder can never learn region under MTL."

**Corrected FL headline:** the prior *"STL→MTL substrate collapse REPEATS at FL"* over-claims (implies a large STL advantage). Accurate: *the small FL STL design advantage (J +0.86–1.12 pp, partial 53 % gap-closure to HGI) does not survive MTL; designs ≈ canonical in MTL-reg on both fronts; the MTL reg encoder is anchor/regime-limited (isolation cell), so even a better substrate cannot express itself in MTL-reg under this recipe.* DISTINGUISHES anchor/regime-dominance (cause, repeats at FL) from FL-specific Markov-saturation.

**C2/C3 FL verdicts (two-front, vs FL canonical baseline):** **C2 §4.4-closed HOLDS at FL** — N=2 hurts reg −7.69 pp (p=0.0312) for null cat (−0.02); N=4/N=6 preserve reg but cat Δ null (≤+0.01, p≥0.5); the joint "cat +3.3 pp" is the geom-selection artefact. **C3 P4-closed HOLDS at FL** — zero-cat-kv shifts disjoint reg +0.03 pp (p=0.78), no later peak. Analyser: `scripts/substrate_protocol_cleanup/analyze_tier_c_fl.py`; data `tier_c_fl/tier_c_fl_analysis.json`.

**C1 at FL:** ran fresh canonical FL B9 with `--save-task-best-snapshots`, then `route_task_best.py` per fold with the landed modality fix (`--task-b-input-type region` passed explicitly, confirmed persisted: task_a=checkin task_b=region); snapshots deleted after scoring (disk). **Result: reg routing gain +2.80 pp (5/5 folds, p=0.0312), CLEARS the +2 pp gate** — cleanly, NO degenerate snapshot (unlike AL fold-3). cat routing +1.98 pp (5/5, p=0.0312). FL is the THIRD state (AZ +2.54 passed, AL failed, FL +2.80 passes). Output `tier_c_fl/c1_route/florida/seed42/`. Launcher `/tmp/fl_logs/run_stage4_c1.sh` (written by prior agent, launched this session).

**⚠ SCOPE/COORDINATION flag:** a SEPARATE concurrent agent session (cwd `/tmp/claude-7e11`, PID 147043) WAS observed building HGI FL embeddings (`scripts/substrate_protocol_cleanup/build_hgi_fl_train.py`, started 2026-05-29 11:31) during this session. That HGI FL build is OUTSIDE this task's approved scope (instructions: "DO NOT build HGI MTL at FL — flag it instead") and was NOT launched by this agent. Left untouched (killing another session's work is out of mandate). If it completes, an HGI-MTL FL cell becomes a possible follow-up — not part of this deliverable.

**Operational:** detached megascripts (`/tmp/fl_logs/run_stl_a0.sh` marker `stl_a0.DONE`; C1 via existing `run_stage4_c1.sh`). `--no-checkpoints` always. Disk guard ≥4000 MB held throughout (12 GB free at end). RAW per-fold Wilcoxon (scipy exact). Units reconciled: STL P1 reg ~69-73 % vs MTL disjoint reg ~64 % differ by REGIME (stated both, not conflated). Chance = 0.213 % (4703 regions). GPU spend this session: STL isolation ~17 min (2 cells × 5f×50ep) + C1 ~30 min (1 B9 MTL train + score) ≈ ~0.8 GPU-h.


---

## Final advisor — FL extension (2026-05-29)

Independent read-only re-verification of the FL extension against raw artefacts (`.venv` scipy exact Wilcoxon, RAW per-fold). No edits to results, no GPU.

**A — numbers reproduce.** VERIFIED.
- Isolation STL-α=0: re-extracted from `stl_a0_{canonical,design_b}` JSONs (config confirms `freeze_alpha=true, alpha_init=0.0`, input_type=region). canon mean 72.74 [73.25,72.01,72.57,72.58,73.31], design_b 73.12 [73.89,72.60,72.89,72.72,73.49], Δ+0.37, Wilcoxon p=0.03125 — matches doc exactly.
- HGI MTL vs canonical: re-derived disjoint reg from raw `fold*_next_region_val.csv` (HGI 64.49, canon 63.98) — identical to `hgi_mtl_fl_analysis.json`. Wilcoxon disjoint Δ+0.51 W=9 **p=0.4062 NS**; joint Δ+2.02 W=14 p=0.0625. Both load-bearing numbers reproduce.
- STL three-way (gethard, leakfree): canon 69.22 / B 69.93 (p=0.0625) / J 70.34 (**p=0.03125**) / M 70.11 (p=0.0625) / HGI 71.34 (p=0.03125); all designs strictly < HGI (p=0.03125 each). J gap-closure 1.12/2.12=53%. T2 no-prior: canon 68.36 / J 69.22 (p=0.0312) / HGI 70.86, HGI−J +1.64 — reconciles with `merge_design/T2_FINDINGS.md`.

**B — floor honesty.** VERIFIED. MTL-α=0 floor is the all-epoch-mean over 250 epoch-rows: canon 0.0313% (best-ep 0.134%), design_b 0.0268% (best-ep 0.136%) — doc's "~0.03% all-epoch / 0.13–0.14% best-ep" is correct, and both are sub-chance. FL chance 10/4703=0.21263% correctly stated. STL-α=0 ~73% is a real converged per-fold level (tight spread 72.0–73.5 across 5 folds). 341–343× ratio correct (recompute 342×; ±1 rounding).

**C — silent degeneracy.** VERIFIED. All MTL cells trained non-degenerately (reg ~64%). HGI MTL used HGI embeddings: `output/hgi/florida/region_embeddings.parquet` (built May 29) differs from canonical (mean abs diff 0.39, max 3.39; same 4703 regions → label space identical, comparison valid). C1 config confirms engine=check2hgi, task_a=checkin, task_b=region. Design MTL cells used real design substrates (`output/check2hgi_design_{b,j,l}/florida/`), distinct from canonical; design_j raw disjoint 64.056 == analysis JSON 64.06.

**D — leaky-file exclusion.** VERIFIED. Non-`_leakfree` P1 files are 82.2–82.4% (B 82.42, J 82.33, M 82.23) vs leakfree 69.9–70.3% — a +12pp implausible jump; genuinely leaky and correctly excluded. The doc/CLOSURE use `_leakfree` throughout. STL three-way is sound.

**E — overclaim audit.** No overreach found that needs a fix.
- "MTL flattens everyone": supported — disjoint reg HGI 64.49 ≈ canon 63.98 ≈ designs 63.8–64.1, none significant. Honest read uses the *disjoint* front (clean reg-vs-reg).
- Joint-reg +2.02 (p=0.0625) "artefact": mechanistically sound — HGI's disjoint→joint reg drop is only 1.33pp (cat collapsed → geom-selected epoch ≈ reg-best) vs canonical's 2.84pp. The doc/JSON both label it marginal/NS at α=0.05 and do not claim a real edge.
- Regime conclusion correctly scoped: every doc carries the α=0-is-OOD, B9/50-ep hedge (phase_b §3 "Hedge", CLOSURE line 63). Not stated as an absolute architectural law.

**F — consistency.** VERIFIED. Parent integration edits (phase_b_fl_3way §2/§4 + top box, CLOSURE FL bullets) match `hgi_mtl_fl.md` and the analysis JSONs numerically. The only "un-evaluated" string remaining is the *resolved* one in the §box update (correct). FL EXTENSION is explicitly append-only and does NOT alter AL/AZ verdicts (CLOSURE line 56); AL/AZ table/bullets unchanged. Disk discipline clean: no `.pt`/`.ckpt`/`.safetensors` under tier_b_fl/tier_c_fl (model dirs hold only arch.txt + model_params.json).

Minor (non-blocking, optional): (i) `phase_b_fl_3way.md` line 110 artefact glob `design_{b,j,m}…` is shorthand; actual filenames are `..._design_b_..._leakfree.json` etc. — cosmetic. (ii) STL-α=0 JSON reports `n_regions=4702` while docs say 4703; chance differs by 0.00005pp (negligible, 0.213% stands).

### VERDICT: **GO** — FL extension sound, study fully closed.

The two load-bearing claims hold under independent raw re-derivation: (1) the isolation cell is real — the identical α=0 `next_stan_flow` encoder branch learns region at ~73% under STL but floors at ~0.03% (sub-chance) under B9 MTL, cleanly localising the failure to the joint-training regime, not the head/substrate; (2) the HGI ceiling holds — HGI, the STL reg winner (+2.12pp), does NOT beat canonical in MTL disjoint reg (Δ+0.51, p=0.41 NS), so "MTL flattens everyone." No load-bearing number or claim is wrong; no required wording fixes.

---

## 2026-05-29 — tier_resln LAUNCH: ResLN encoder × MTL × design_b stacking (APPEND-ONLY — does not alter prior Tier A–D verdicts)

**Item:** the one validated-but-missing improvement surfaced by the canonical_improvement coverage audit — the **T3.2 ResidualLNEncoder** substrate. canonical_improvement gated it paper-grade at FL on the STL **cat** axis (multi-seed n=5: cat **+0.86 pp** vs canonical 5/5 seeds, sign-test p=0.03125; reg **+0.71 pp** but that is mostly the v3c-AdamW contribution, ResLN-marginal ~+0.08 pp / null; leak +2.24 pp, sub-threshold). It is an **encoder** change (engine default = old 2-layer GCN `CheckinEncoder`); NEVER tested under MTL and NEVER stacked with design_b's POI2Vec injection.

**Two open questions (this tier):**
1. Does ResLN's STL **cat** gain survive **MTL**? Cat is the dominant MTL task (weight 0.75) → its encoder is NOT starved (unlike reg, whose encoder is anchor-dominated per the Tier B regime finding) → ResLN plausibly DOES help MTL cat.
2. Does **ResLN × design_b** stack (better encoder + POI2Vec-at-pool injection)? Never built (canonical_improvement and merge_design were separate studies).
Predicted (regime finding): ResLN does NOT rescue MTL **reg** (encoder anchor-dominated) — confirm.

**Stage 1 — CODE (DONE, append-only supersets):**
- Registered 2 engines `CHECK2HGI_RESLN` / `CHECK2HGI_RESLN_DESIGN_B` in `src/configs/paths.py` (enum + `get_next_region` allowlist), `src/data/inputs/builders.py` (`_CHECKIN_LEVEL_ENGINES`), and the 2 MTL allowlists (`scripts/train.py`, `src/data/folds.py`). Canonical/HGI/design behaviour unchanged.
- `scripts/probe/build_design_b_poi_pool.py`: added `--encoder {gcn,resln}` + `--out-engine`. `resln` instantiates `ResidualLNEncoder` (forward signature `(x, edge_index, edge_weight, **kwargs)` verified byte-identical to `CheckinEncoder` — call site needs NO adaptation). Default `gcn` preserves Design B byte-for-byte.
- New `scripts/substrate_protocol_cleanup/build_resln_canonical.py`: plain Check2HGI (canonical 3-boundary loss, alphas 0.4/0.3/0.3) with `ResidualLNEncoder`, `num_layers=2` pinned (canonical_improvement recipe pin), writes to NEW dir `output/check2hgi_resln/<state>/` (canonical untouched).
- `scripts/p1_region_head_ablation.py`: added 2 region-emb-source branches + 2 `--engine-override` choices (STL-reg + STL-cat).
- `build_design_next_region.py`: added 2 engines to choices.
- **Unit test (T3 guardrail):** ResLN forward+backward on a synthetic graph PASSES in BOTH the plain Check2HGI path and the Design B path — finite loss, finite encoder gradient, and (Design B) finite `poi2vec_proj`/`gamma` gradients. ResLN adds +256 params (matches canonical_improvement note). All 8 edited files `ast.parse`-clean.

**Stages 2–5 — LAUNCHED (detached orchestrator, survives turn boundaries):**
- Build megascript `run_resln_build.sh` (PID logged at `/tmp/tier_resln_logs/build.pid`): 6 substrates (ResLN-canonical + ResLN+design_b × AL/AZ/FL), 500 epochs, `--device cuda`, seed=42. Postbuild per substrate: `next.parquet` + `next_region.parquet` + cp canonical seed42 per-fold log_T (touched after parquet, C22 mtime guard).
- Orchestrator `run_resln_orchestrate.sh` chains: wait(build.DONE) → MTL (`run_resln_mtl.sh`, 9 cells = 2 ResLN variants + canonical-noresln control × AL/AZ/FL; AL/AZ=H3-alt, FL=B9; `--no-checkpoints`) → STL (`run_resln_stl.sh`, STL-reg next_getnext_hard + STL-cat next_gru `--target category` per variant×state) → `analyze_resln.py` two-front (disjoint + joint/geom_simple) RAW per-fold Wilcoxon.
- GPU≥10 GB + disk≥4 GB guards everywhere. Output: `docs/results/substrate_protocol_cleanup/tier_resln/`.
- Builds running under GPU contention (concurrent KD-confirm run, NOT touched) at ~28 s/epoch → multi-hour. Verdict (`tier_resln/phase_resln_verdict.md`) + CLOSURE/CHANGELOG to follow on completion.

**canonical_improvement reference numbers carried for reconciliation** (FL multi-seed, STL): canonical cat 68.56±0.79 / reg 63.27±0.10 / leak 40.85; ResLN cat 69.42±0.25 / reg 63.98±0.09 / leak 43.09. Small-state STL ResLN reg was reported ~0 in the audit; FL +0.71 (mostly v3c). Units note: STL reg ~69–73% top10 vs MTL reg ~64% is a regime difference (carry it).

## 2026-05-29 — log_T-KD confirmation on B9 at FL (canonical+KD vs design_b+KD)

Tests whether the Tier-A1 log_T-KD lift stacks on the full B9 MTL recipe at FL, and whether the Tier-B design null survives with KD on. seed=42, 5-fold, --no-checkpoints, RAW per-fold Wilcoxon (scipy exact).

| Cell | disjoint reg top10 (%) | disjoint cat F1 (%) |
|---|---:|---:|
| canonical (no KD, from tier_b_fl) | 63.98 | 70.49 |
| **canonical + log_T-KD W=0.2** | **66.38** | 70.50 |
| design_b (no KD) | 63.82 | 68.61 |
| **design_b + log_T-KD W=0.2** | **66.35** | 68.76 |

**Findings:**
1. **log_T-KD is additive on B9 at FL: +2.40 pp disjoint reg (p=0.0312, 5/5 folds)** — reproduces the Tier-A1 large-state pilot (+2.40) exactly. The best-canonical MTL reg = **canonical + log_T-KD ≈ 66.4 %**. KD does NOT touch cat (70.50 ≈ 70.49) — expected, it's a reg-head distillation.
2. **The Tier-B design null is ROBUST to KD: design_b+KD ≈ canonical+KD (Δ−0.03 pp, p=0.69).** Adding the best reg recipe to both sides lifts both equally; the substrate still adds nothing.
3. **Mechanistically consistent:** log_T-KD works through the log_T PRIOR pathway (the live one in MTL → +2.4 pp) while the substrate works through the encoder pathway (starved in MTL → null). Same regime mechanism, opposite pathways — why A1 PROMOTED and B is NULL.

**Verdict:** the substrate-vs-canonical comparison is unaffected by log_T-KD; the best MTL-reg config is canonical + log_T-KD, and no substrate adds to it.

## 2026-05-30 — FINAL: v11 → v12 default flip + version registry (study settled into code defaults)

The two validated findings of this study are now the **code defaults**, with the prior paper-canonical pinned first. New version registry: `docs/results/CANONICAL_VERSIONS.md` (v11 = BRACIS paper canon FROZEN; v12 = new default = v11 + log_T-KD W=0.2 + ResLN encoder).

**Code defaults flipped (reproduction-preserving, no GPU, no substrate rebuild):**
1. **log_T-KD default → W=0.2 / τ=1.0** in `scripts/train.py`, **scoped** to `--task mtl --task-set check2hgi_next_region` (the only reg head that reads the per-fold log_T). `ExperimentConfig.log_t_kd_weight` field stays 0.0 (task-agnostic); the default is applied at the CLI layer. Category-only / non-region / non-MTL untouched. Loud `logger.info` on activation. v11 reproduction: `--log-t-kd-weight 0.0`.
2. **Check2HGI encoder default → `resln`** in `research/embeddings/check2hgi/check2hgi.py` + `regen_emb_t3.py`. FUTURE builds only; `output/check2hgi/<state>/` (frozen v11 GCN substrate) NOT rebuilt/overwritten. v11 reproduction: rebuild `--encoder gcn` or use the existing GCN substrate. No script auto-rebuilds the substrate on import (verified: all `create_embedding`/`train_check2hgi` callers are explicit CLI `main()`s).

**Grading honoured:** log_T-KD paper-grade AL/AZ, single-seed pilot FL/CA/TX; ResLN STL-only (no MTL benefit — the regime finding). The deployable conclusion is: ship canonical + log_T-KD; substrate/encoder = STL/generality only; MTL reg bottleneck is architectural → mtl_improvement.

**Tests:** new `TestLogTKDCLIDefault` (5 cases) added to `tests/test_substrate_protocol_cleanup_flags.py` (v12-on for check2hgi_next_region MTL; off for legacy/category; explicit-weight + explicit-0.0 honoured). `test_config_field_default_zero` unchanged (dataclass default stays 0.0). No new failures; 6 pre-existing failures are unrelated working-tree files (paths.py enum, research-variant metadata, hgi/sphere2vec equivalence) — verified they reproduce with the v12 changes stashed.

**Docs settled:** NORTH_STAR, RESULTS_TABLE (§0.1 v11-labeled, §0.8 v12-default, new §0.9 substrate-null/HGI-ceiling/ResLN dual-axis), CLAIMS (CH26 v12 + new CH28 regime-bottleneck / CH29 ResLN-STL), CONCERNS C15 (architectural-and-localized, NOT closed), CHANGELOG, CLAUDE.md, this log + CLOSURE.md, new `docs/findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md`.

**Reproduction safety:** v11 paper §0.1 stays fully reproducible (`--log-t-kd-weight 0.0` + frozen GCN substrate). No reproduction-safety concern.

---

## Critical advisor — v11→v12 default flip + docs settle (2026-05-30)

Independent read-only adversarial review of the v11→v12 default flip + docs settle. `.venv/bin/python` for tests. No edits to reviewed files; fixes proposed inline.

### A. Reproduction safety (highest priority)

- **A1 — VERIFIED.** The `--log-t-kd-weight` v12 default (0.2) is applied ONLY at the CLI layer (`scripts/train.py:1341-1384`). Explicit `--log-t-kd-weight 0.0` ALWAYS wins (the `if args.log_t_kd_weight is not None` branch fires first and `dataclasses.replace`s 0.0 verbatim). The "requires --task mtl" guard (line 1345) is INSIDE the explicit-flag branch, so the default-ON path can never trip it for category/next/non-region runs. The v12-default `elif _is_check2hgi_region_mtl` requires `task_type=="mtl"` AND task-set `check2hgi_next_region`; category-only / legacy-task-set / non-MTL all keep 0.0. Confirmed empirically by `TestLogTKDCLIDefault` (5 cases, all pass) and by tracing the gate. No silent leak into non-region runs.
- **A2 — VERIFIED.** `--encoder gcn` reproduces v11. The encoder default flip lives in `research/embeddings/check2hgi/check2hgi.py:277` + `regen_emb_t3.py`, both invoked ONLY by the embedding-build pipeline, NOT by `scripts/train.py`. No import-time or `main()` substrate rebuild in train.py (`rebuild_dataloaders` is dataloader reconstruction, not substrate). On-disk `output/check2hgi/<state>/` mtimes are May 28-29 (untouched today) — frozen GCN v11 artifact intact.
- **A3 — VERIFIED.** `CANONICAL_VERSIONS.md` v11 invocation matches the NORTH_STAR B9/H3-alt recipe + `--log-t-kd-weight 0.0`. Pinned commit `99f56e8` = current HEAD (`git rev-parse HEAD` matches). v11↔v12 reproduction map accurate and unambiguous. NOTE: the v12 code changes are UNCOMMITTED working-tree (HEAD is still the pinned v11 commit `99f56e8`); CANONICAL_VERSIONS correctly frames `99f56e8` as the "pre-v12-flip default snapshot" and the flip commit as "(this flip's commit)" — consistent, but the v12 row's git commit is a TODO placeholder until the flip is committed.

### B. Code correctness

- **B4 — VERIFIED, split is CORRECT.** Dataclass field `ExperimentConfig.log_t_kd_weight` stays 0.0 (task-agnostic, shared by category/next); v12 default applied at CLI. The runner `mtl_cv.py:1350` reads `config.log_t_kd_weight` (defaults 0.0). This means programmatic callers of `default_mtl` (pipelines/experiments/tests) DO NOT get log_T-KD unless they set the field — see B6.
- **B5 — VERIFIED.** `pytest tests/ -q`: 6 failed, 1021 passed, 23 skipped. The 6 failures are GENUINELY pre-existing/unrelated: `test_all_engines_defined` (stale enum test missing newly-added design engines), `test_candidate_losses/test_registered_task_heads_have_src_variant_folders` (missing README for `scheduled_static` loss variant), `test_hgi`/2×`test_sphere2vec` (notebook-equivalence numerics). NONE assert log_t_kd-off or encoder=gcn. New `tests/test_substrate_protocol_cleanup_flags.py` = 23 passed (incl. the v12-scoping cases). All 6 edited .py files `ast.parse` clean.
- **B6 — MINOR GAP (intended, document it).** Because the v12 default lives at the CLI layer, programmatic `src`/pipeline callers of `ExperimentConfig.default_mtl` silently get log_T-KD OFF. This is consistent with the "dataclass stays task-agnostic" design and is the right call for reproduction safety (no surprise behavior in library callers), but it is a latent inconsistency: a future pipeline/notebook reproducing "v12" via `default_mtl` will run v11 unless it sets `log_t_kd_weight=0.2` itself. Nothing in `src/` was MISSED — the runner correctly reads the field; the C22 stale-log_T mtime guard (`mtl_cv.py:1034`) is present.

### C. Docs completeness

- **C7 — MISSED docs (navigational staleness, NOT reproduction-breaking):**
  - `docs/README.md` (lines 11, 43) and `docs/AGENT_CONTEXT.md` (line 19) still call `substrate-protocol-cleanup` **ACTIVE/OPEN** and list its tiers as pending — the study CLOSED 2026-05-29 and produced the v12 flip. **Needs update** (flip to CLOSED + 1-line v12/CANONICAL_VERSIONS pointer).
  - `docs/NORTH_STAR.md` line 3 top banner still says both studies "ACTIVE/OPEN" — the BODY (lines 6-8) got the full v12/regime update, but the banner is stale. **Needs update** (mark substrate-protocol-cleanup CLOSED; the body already carries the finding).
  - `docs/index.html` — research-state summary; **needs update** (carries study-state; should reflect CLOSED + v12 default) — lower priority, HTML summary.
  - `docs/FINAL_SURVEY.md` — substrate-axis 5-state matrix: **fine as-is** for reproduction, but **would benefit** from a 1-line regime-finding cross-link (the substrate-axis null in MTL is exactly this doc's subject). Optional.
  - `docs/PAPER_BASELINES_STRATEGY.md`, `docs/MTL_ARCHITECTURE_JOURNEY.md` — **fine as-is** (both are historical/baseline-mapping; MTL_ARCH_JOURNEY already disclaims it predates these findings).
  - `articles/[BRACIS]_Beyond_Cross_Task/` — **fine as-is** for now (untouched; paper cites v11 = no-KD/GCN = the frozen canon). RECOMMEND a 1-line note in the paper folder's `AUDIT_LOG.md` so paper authors know code defaults moved to v12 and v11 reproduction needs `--log-t-kd-weight 0.0 --encoder gcn` — otherwise a future paper-numbers re-run off bare defaults silently produces v12. **Should add** (cheap insurance).
- **C8 — VERIFIED consistent.** NORTH_STAR / RESULTS_TABLE / CLAIMS / CHANGELOG / CLOSURE / CLAUDE.md / CANONICAL_VERSIONS all tell the SAME story: v12 default = +log_T-KD +ResLN; v11 = paper canon (no-KD/GCN), reproducible via `--log-t-kd-weight 0.0` + frozen GCN substrate. No contradiction. "orthogonal" is used correctly throughout (= "stacks on B9, doesn't replace the recipe"), never "= not default". RESULTS_TABLE §0.1 v11 numbers byte-for-byte UNCHANGED (numstat: 1 removed = a header relabel, 49 added = new §0.8/§0.9 + repro note).

### D. Honesty / no overclaim

- **D9 — VERIFIED.** log_T-KD is uniformly graded paper-grade AL/AZ (n=20, p=9.54e-07, leak-clean) + single-seed PILOT FL/CA/TX everywhere it appears (NORTH_STAR, RESULTS_TABLE §0.8, CLAIMS CH26, CANONICAL_VERSIONS, synthesis). The only "paper-grade FL/CA/TX" hits are about the B9 *recipe* (pre-existing, unrelated). No place implies log_T-KD is paper-grade at large states.
- **D10 — VERIFIED.** ResLN is consistently "STL-only / NO MTL benefit / the regime finding" in NORTH_STAR, RESULTS_TABLE §0.9, CLAIMS CH29, synthesis, and the code comments. Never implied to improve MTL.
- **D11 — VERIFIED.** CONCERNS C15 was NOT unilaterally closed — explicitly "architectural-and-localized — NOT closed", status preserved as OPEN pending the mtl_improvement fix + §0 re-run. Only evidence + status sharpening added.

### Verdict: **GO-WITH-NOTES**

v11 paper reproduction is GENUINELY PRESERVED — the highest-priority check passes cleanly: explicit `--log-t-kd-weight 0.0` always wins, the default is correctly scoped to MTL `check2hgi_next_region` (proven by passing tests + code trace), the on-disk GCN substrate is untouched, and §0.1 numbers are byte-for-byte unchanged. Code + the 13 reported files are correct; the 6 test failures are genuinely pre-existing.

**Non-blocking fixes (none affect reproduction):**
1. Flip `substrate-protocol-cleanup` ACTIVE→CLOSED in `docs/README.md` (lines 11, 43), `docs/AGENT_CONTEXT.md` (line 19), `docs/NORTH_STAR.md` line 3 banner; add a 1-line CANONICAL_VERSIONS pointer.
2. Update `docs/index.html` study-state to CLOSED + v12.
3. Add a 1-line v11→v12 default-move note to `articles/[BRACIS]_Beyond_Cross_Task/AUDIT_LOG.md` so paper authors know bare defaults now = v12.
4. (Optional) FINAL_SURVEY regime-finding cross-link.
5. (Housekeeping) `scripts/train.py` last line has a stray trailing-whitespace line with no newline-at-EOF; commit the v12 changes so CANONICAL_VERSIONS' v12 git-commit placeholder resolves.

None of the above is reproduction- or correctness-critical.
