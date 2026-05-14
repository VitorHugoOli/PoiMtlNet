# Check2HGI Study — Compiled Overview (2026-04-28)

**Working scratch file** built from a sweep of the live tracker docs +
review docs + archived original phase plans. Purpose: a single self-contained
picture of where the study stands and what remains, **without losing
narrative-critical detail** (per the user's request).

Sources read:

- `README.md`, `AGENT_CONTEXT.md`
- `PAPER_PREP_TRACKER.md`, `FOLLOWUPS_TRACKER.md`, `PHASE2_TRACKER.md`
- `OBJECTIVES_STATUS_TABLE.md`, `CLAIMS_AND_HYPOTHESES.md`, `CONCERNS.md`
- `PAPER_DRAFT.md`, `PAPER_STRUCTURE.md`
- `review/2026-04-28_post_phase1_overview_and_gap_audit.md`
- `review/2026-04-28_morning_briefing.md`
- `review/2026-04-27_study_overview_and_forgotten_items.md`
- `archive/phases_original/{P0,P5,P7,…}.md`

---

## 0. One-paragraph TL;DR

Three paper-grade tracks are closed. **Phase 1 substrate validation** (AL+AZ)
confirmed Check2HGI > HGI on cat F1 head-invariantly and made MTL B3 a
substrate-specific result (CH16, CH18, CH19). **F49 architecture attribution**
(AL+AZ+FL n=5) refuted the legacy "+14.2 pp transfer" claim by ≥9σ on FL alone
and added a Tier-A methodological note that loss-side `task_weight=0` is unsound
under cross-attention MTL (CH20). **F37 FL closing** (2026-04-28) ran the FL
matched-head STL ceiling and **flipped the FL reg story scale-conditional**:
matched-head STL `next_getnext_hard` beats MTL H3-alt at FL by −8.78 pp paired
Wilcoxon p=0.0312 (5/5 folds negative); architectural Δ FL = −16.16 pp p=0.0312.
The CH18 "MTL exceeds STL on reg" claim survives at AL only; CH21 (top-line) is
reframed per-state. **The paper title and 130-word abstract are committed**;
remaining headline-blocker is **P3** (CA + TX upstream pipelines + 5f H3-alt) on
Colab; statistical sharpening (P4, P5) is done; paper prose drafting has only
section skeletons in `paper/{methods,results,limitations,appendix_methodology}.md`.

Per-state mechanism pattern (the load-bearing per-state sentence):
**AL** = architecture-dominant joint MTL lift on both heads → **AZ** = classical
(architecture costs reg, partial recovery) → **FL** = substrate-only joint lift,
architecture costs reg heavily, matched-head STL ceiling above MTL on reg.

---

## 1. Champion config (NORTH_STAR — unchanged by F49 + F37)

`F48-H3-alt`:

```
arch    : mtlnet_crossattn
mtl_loss: static_weight(category_weight=0.75)
task_a head : next_gru                 (post-F27, was next_mtl)
task_b head : next_getnext_hard        (STAN + α·log_T[last_region_idx])
task_a input: check-in emb (9-window)
task_b input: region emb (9-window)
hparams : d=256, 8h, batch=2048 (1024 on FL), 50 epochs, seed 42
LR sched: constant (no OneCycleLR)
LR group: cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3   ← the H3-alt recipe
```

CLI delta vs predecessor B3: `--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`.

Predecessor **B3** (50ep + OneCycleLR) is preserved as comparand to demonstrate
the per-head LR contribution. Paired Wilcoxon H3-alt vs B3 (P5, done
2026-04-28): AL reg +15.36 pp p=0.0312; AZ reg +10.22 pp p=0.0312; AL cat
−0.88 pp p=0.0312 (small but detectable cat regression).

---

## 2. Headline numbers (5-fold × 50ep, H3-alt, seed 42)

| State | cat F1 | reg Acc@10 | vs matched-head STL `next_getnext_hard` | Pattern |
|-------|-------:|----------:|---:|---|
| AL    | 42.22 ± 1.00 | **74.62 ± 3.11** | **+6.25 pp** EXCEEDS p=0.0312 | architecture-dominant ✓ |
| AZ    | 45.11 ± 0.32 | 63.45 ± 2.49 | −3.29 pp (closes 75% of B3 gap, n.s.) | classical |
| FL    | **67.92 ± 0.72** | 71.96 ± 0.68 | **−8.78 pp** STL EXCEEDS p=0.0312 (5/5 folds) | architecture costs reg |

Cat-side MTL > STL holds at all three states (post-F37 P1):
- AL +3.64 pp (vs STL `next_gru` 38.58)
- AZ +3.03 pp (vs 42.08)
- FL **+0.94 pp** (vs F37 P1 STL `next_gru` 66.98 ± 0.61)

F37 P2 (FL STL `next_getnext_hard`, M4 Pro 29 min on 2026-04-28): **82.44 ± 0.38**.
This is the matched-head STL ceiling at FL — above MTL H3-alt by −8.78 pp.

**Architectural-Δ pattern (F49 frozen-cat λ=0 vs STL F21c, all n=5 paired):**
{AL **+6.48 pp** ~2.7σ, AZ **−6.02 pp** ~3.7σ, FL **−16.16 pp** p=0.0312 (5/5 folds neg)}.
The architectural cost grows with region cardinality (1.1K → 1.5K → 4.7K).

---

## 3. The 4 paper-grade claims locked + 1 top-line

| ID | Claim | Tier | Evidence |
|----|-------|:-:|---|
| **CH16** | Check2HGI > HGI on next-cat F1, head-invariant at AL+AZ | A | 8/8 head-state probes p=0.0312; Δ +11.58…+15.50 pp |
| **CH18** | MTL B3 is substrate-specific (Tier A — scale-conditional after F37) | A | HGI substitution at AL+AZ breaks reg by 30 pp; FL flips at scale (STL > MTL by 8.78 pp) |
| **CH19** | Per-visit context = ~72% of cat substrate gap (mechanism) | A | POI-pooled Check2HGI counterfactual at AL |
| **CH20** | Cat-supervision transfer ≤ |0.75| pp on AL+AZ+FL n=5 (Layer 1) + loss-side `task_weight=0` ablation unsound under cross-attn (Layer 2 methodological) | A | F49 3-way decomp + 4 regression tests + F37 Layer 3 closing |
| **CH21** | TOP-LINE — MTL win is interactional architecture × substrate, **scale-conditional**: AL = architecture-dominant lift; FL = substrate-only (architecture costs reg) | A | Synthesis of CH18+CH19+CH20 + F37 |

Other still-active claims:
- **CH17** — Check2HGI > POI-RGNN (external) — confirmed FL +28-32 pp; CA/TX gated on Phase 2.
- **CH15** — REFRAMED head-coupled (was "HGI > C2HGI on reg under STAN"). Under matched MTL reg head `next_getnext_hard`, C2HGI ≥ HGI everywhere.

---

## 4. PAPER_PREP_TRACKER — what remains

### Headline-blocking before submission

| ID | Item | Status | Owner | Cost |
|----|------|:------:|:-----:|:----:|
| ~~P1~~ | F37 STL `next_gru` cat 5f per state | **DONE 2026-04-28** | M4 Pro | 24.5 min |
| ~~P2~~ | F37 STL `next_getnext_hard` reg 5f on FL | **DONE 2026-04-28** | M4 Pro | 29 min |
| **P3** | **CA + TX upstream pipelines + 5f H3-alt** | **NOT STARTED — CRITICAL PATH** | Colab T4 | F34/F35 ~12h + F24/F25 ~25h |

After F37 the **importance of P3 increased**: tests whether FL's architectural-cost-at-scale pattern replicates at CA+TX (similar size) or is FL-idiosyncratic. CH21 paper framing depends on this — single-state outlier (FL) vs scale-curve (FL+CA+TX).

### Sharpens claims (small wins)

| ID | Item | Status |
|----|------|:------:|
| ~~P4~~ | Paired Wilcoxon p-values on F49 cells per fold | **DONE 2026-04-28** (`results/paired_tests/F49_decomposition_wilcoxon.json`) |
| ~~P5~~ | Paired Wilcoxon H3-alt vs B3 (AL+AZ) | **DONE 2026-04-28** (`results/paired_tests/H3alt_vs_B3_wilcoxon.json`) |
| **P6** | Seed sweep H3-alt on AL+AZ {0,7,100} | NOT STARTED — ~3h MPS — only "nice-to-have" |
| **P7** | `MTL_ARCHITECTURE_JOURNEY.md` post-F49 narrative pass | partial; expand if reviewers ask |

### Camera-ready / post-submission

| ID | Item | Notes |
|----|------|-------|
| **P8** | `POSITIONING_VS_HMT_GRN.md` rewrite | Currently has deprecation note; legacy +14.2 pp transfer + 25 pp arch overhead numbers are refuted |
| **P9** | Block-internal `ffn_a` ablation | F49 plan flags `_CrossAttnBlock.ffn_a/ln_a*` are NOT frozen in either F49 variant (autograd-detach variant likely needed) |
| **P10** | α instrumentation per epoch per fold per F49 variant | Tightens architecture-vs-optimizer attribution |
| **P11** | FL frozen-cat reg-path instability investigation | Per-fold reg-best epochs {2,14,9,4,2} → α-growth doesn't engage when cat is random at FL scale |
| **P12** | Extend `experiments/check2hgi_up` to H3-alt | User-decision: extend / disclaim / defer |

### Pending doc rewrites (active)

- **`research/POSITIONING_VS_HMT_GRN.md`** — full rewrite for camera-ready (P8).
- **NEW `research/F49_METHODOLOGICAL_NOTE.md`** — Layer-2 standalone short note (~3pp) on cross-attn loss-side ablation unsoundness.
- **NEW `research/CH21_JOINT_CLAIM.md`** — top-line synthesis doc with figure suggestions + drafting hooks.

### Drafting (not costed in tracker, ~10–15h estimated)

Already exist as **skeleton-only** v0 drafts in `paper/`:

```
paper/methods.md                  Methods §3
paper/results.md                  Results §4 (Tables 1+3 filled with FL F37 numbers)
paper/limitations.md              Limitations §6 — §6.1 rewritten "scale-conditional"
paper/appendix_methodology.md     Appendix A — F49 Layer 2
```

Title and 130-word abstract are **committed** in `PAPER_DRAFT.md` for BRACIS 2026:
> *Beyond Cross-Task Transfer: Per-Head Learning Rates and Check-In-Level Embeddings for Multi-Task POI Prediction*

Open decisions on the abstract (`PAPER_DRAFT.md §2.3`): D1–D5 (e.g., "up to 33 pp" wording vs alternates; "multiple US-state Gowalla splits" depends on Phase-2 status at submission time).

### Submission-readiness checklist (current)

Substrate-side (Phase 1):
- [x] CH16 head-invariant — AL+AZ
- [x] CH15 reframed — AL TOST non-inf, AZ +2.34 pp p=0.0312
- [x] CH17 external baselines — POI-RGNN audit + faithful baseline ports landed
- [x] CH18 substrate-specific MTL — landed AL+AZ Tier A
- [x] CH19 mechanism — ~72% per-visit at AL
- [ ] **F36 FL Phase-2 substrate grid** — FL data on disk, queued in `PHASE2_TRACKER.md`
- [ ] **P2-CA-grid** (gated on P2-CA-up CA upstream)
- [ ] **P2-TX-grid** (gated on P2-TX-up TX upstream)

Architecture-side (F49):
- [x] H3-alt closes F21c reg gap on AL+AZ+FL — landed (Tier A)
- [x] CH20 Layer 1+2 — landed AL+AZ+FL n=5
- [x] CH20 Layer 3 (FL absolute architectural Δ) — **closed by F37 2026-04-28** (−16.16 pp p=0.0312)

Joint claim:
- [x] CH21 — top-line synthesised
- [ ] **Headline paper section** drafting CH21 (no longer "MTL transfers signal")

Stat strengthening:
- [x] Paired Wilcoxon (P4 + P5) — DONE
- [ ] CH19 cross-state replication — optional Phase-2

Paper drafting:
- [ ] Methods drafted
- [ ] Results 3-state × 4-cell F49 decomposition + 8-cell substrate-Δ + MTL counterfactual tables
- [ ] Methodological appendix Layer 2 + matched-head STL policy revision
- [ ] Limitations: AL is dev state; FL frozen-cat instability; CA/TX gated; CH15 head-coupled
- [ ] Reviewer rebuttal ammo

---

## 5. FOLLOWUPS_TRACKER — open per-experiment items

### Ready-now (P1/P2 paper-blocking)

| # | Item | Status |
|---|------|:------:|
| **F33** | FL 5f×50ep B3+`next_gru` (decisive F27 FL resolution — Path A vs Path B) | NOT STARTED — Colab T4 ~6h |
| **F34** | CA upstream pipeline + CA 1f×50ep B3+gru | NOT STARTED — Colab T4 ~6-12h |
| **F35** | TX upstream pipeline + TX 1f×50ep B3+gru | NOT STARTED — Colab T4 ~6-12h |
| **F24** | CA 5f headline (gated on F34) | gated |
| **F25** | TX 5f headline (gated on F35) | gated |
| **F4** | FL MTL-B3 5-fold clean re-run | **SUPERSEDED 2026-04-28** by H3-alt 5f FL — only needed if reviewer wants a B3 5f comparand at FL specifically |
| **F37** | STL `next_gru` cat 5f + STL `next_getnext_hard` reg 5f on FL | **DONE 2026-04-28** (ran in 53 min on M4 Pro, far below 4050 estimate) |
| **F3 / F26** | AZ HGI STL cat | **CLOSED 2026-04-27** by Phase-1 (matched-head `next_gru` evidence) |
| **F9** | FL HGI STL cat 5f | merged into PHASE2_TRACKER §F36b |

### Ambiguous statuses (per 2026-04-28 review)

- **F31** "B3 post-F27 AL 5f validation" — listed as `pending` in tracker but F27/F32 done 2026-04-24; unclear if executed. Numbers are referenced in OBJECTIVES_STATUS_TABLE so likely effectively done; the tracker row needs updating.
- **F39** appears twice with conflicting status. 1f cat_weight screen DONE; 5f sweep status unclear.

### Deferred (follow-up paper / post-champion-freeze)

| # | Item |
|---|------|
| F8 | Multi-seed n=3 on champion configs — DEFERRED 2026-04-23 by user |
| F12 | Per-fold transition matrix (leakage-safe GETNext) |
| F13 | GETNext with true flow-map Φ |
| F14 | PIF-style user-specific frequency prior |
| F15 | TGSTAN + STA-Hyper full reproductions |
| F16 | Encoder enrichment (P6: temporal/spatial/graph features) |
| F36 | CH15 under graph-prior head (HGI STL `next_getnext_hard` AL) — only if reviewer asks |
| F21a | DROPPED 2026-04-24 by user (FL STL STAN not needed) |
| F21b | Archived — STL GETNext (soft) 5f per state |
| F27-followup | Cat-head ablation (next_stan, ensemble, etc.) |

### Gated

- **F5** FL MTL-GETNext-soft 5-fold — only if paper explicitly needs soft vs B3 at FL 5f.

### Phase 2 (live in PHASE2_TRACKER)

- **F36** — FL substrate grid (Legs I + II + III): probe + cat STL × 2 + reg STL × 2 + MTL counterfactual. **Data on disk**. Colab T4 ~3h or M4 Pro ~30h.
- **P2-CA-up / P2-CA-grid** — CA upstream pipeline + CA Phase-2 grid. Gated on upstream.
- **P2-TX-up / P2-TX-grid** — TX upstream pipeline + TX Phase-2 grid. Gated on upstream.

---

## 6. Original-plan gap audit (`archive/phases_original/`)

Per the 2026-04-27 + 2026-04-28 reviews, the P0–P7 archive has both deliberate retirements and **silent drops**. The archive README is the only acknowledgment of the shift.

### High priority — narrative-changing forgotten items

| # | Item | Severity | Recommended action |
|---|------|:--------:|---|
| 1 | **`next_poi → next_category` task pivot** (P0–P2 originally targeted `{next_poi, next_region}`) | **HIGH** | Memo drafted at `scope/task_pivot_memo.md`; still need to surface in PAPER_STRUCTURE.md or NORTH_STAR.md |
| 2 | **CH15 ID reused** — original = transductive-leakage audit (P0.7); current = head-coupled finding | **HIGH — audit-trail risk** | Decision proposal at `scope/ch15_rename_proposal.md` (3 options: A=CH15→CH15b, B=CH22 fresh ID, C=redefinition note). User decision pending. |
| 3 | **CH11 / F8 seed variance** (entire seed-σ requirement) | **HIGH** | Currently AL/AZ/FL all run at seed=42 only. Reviewers will ask. P6 in PAPER_PREP_TRACKER mentions {0,7,100} but **not on paper-blocking list**. ~3h MPS. |

### Medium priority — sharpens claims

| # | Item | Recommended action |
|---|------|---|
| 4 | **P0.6 CH14 fclass-shuffle ablation** — only code inspection done; unconditional shuffle never executed | ~30 min run on AL OR formally retire in CONCERNS.md (proposal at `scope/ch14_ch10_p02_decisions.md`) |
| 5 | **P0.7 / original CH15 transductive-leakage audit** — never executed | Either run (~20 min Check2HGI retrain + 10 min eval on fold 0) OR declare formally as Tier-E limitation (no current C-entry) |
| 6 | **P5 head-architecture full sweep (CH09)** — only F27 binary swap on cat side; no formal 5-head MTL sweep | Mention in Methods that we did binary swap on AZ as pragmatic substitute |
| 7 | **P5 optimiser sweep on FL (CH10)** — only AL has PCGrad-vs-static attribution; no FL multi-optimizer 5f comparison | Either small AL sweep OR retire formally (proposal at `scope/ch14_ch10_p02_decisions.md`) |
| 8 | **C10 — locating HGI-based next-category external reference (CH17)** | User owes specific HGI-next-category article reference (open in CONCERNS.md, not PAPER_PREP_TRACKER) |
| 9 | **NS-2 Hybrid (cross-attn cat + dselectk reg)** | Listed as future-work in archived P4. Probably permanently deferred; not formally retired. |
| 10 | **P3 dual-stream concat** never tested; superseded by per-task input modality before execution | Add memo: "P3 dual-stream design retired in favor of per-task input modality before execution" |
| 11 | **P1.5 POI2HGI vs Check2HGI** — entire phase abandoned; substrate comparator switched to HGI without memo | Add a sentence in PAPER_STRUCTURE explaining the comparator choice |

### Low priority

| # | Item | Recommended action |
|---|------|---|
| 12 | **CH01/CH02 original literal "MTL > STL on next_poi"** — silently replaced by CH18/CH20 | "Deprecated claims" footnote in CLAIMS_AND_HYPOTHESES |
| 13 | **P0.2 label round-trip spot check** (20-sample placeid verification) — no mention | Either run (~10 min) or note as routine pre-flight done |
| 14 | **P0.8 final preflight gate** (8 boolean conditions) — no documented verdict | Likely done implicitly; note for completeness |

### Properly retired (no action needed)

- F8 multi-seed n=3 → explicit P6 deferral
- Multi-seed n=5 → explicit P7 §11
- NS-2 hybrid cross-attn → explicit future work (still ambiguous status though)
- POI-granularity next-POI → explicit out-of-scope
- CA/TX C2 head-agnostic probes → explicit PHASE2_TRACKER §6

---

## 7. Open concerns (CONCERNS.md, current)

| ID | Severity | Concern | Status |
|----|:-:|---|:--:|
| **C10** | High | POI-RGNN + HGI-next-category external refs — user owes specific HGI-next-category article reference | open |
| **C13** | Med | AL 10K vs FL/CA/TX 127K extrapolation risk | open (informally addressed by per-state architectural-Δ pattern) |
| **C14** | Med | F27 cat-head scale-dependence flag — F33 is decisive test (Path A vs B) | open until F33 |
| **C15** | High | MTL coupling vs matched-head STL on reg | **RE-OPENED 2026-04-28 with FL caveat — scale-conditional** (was resolved 2026-04-26) |
| **C01** | Med | n=2 states thin for final paper | monitored (CA/TX in P3) |
| **C02** | Med | FL Markov saturation (65.05) narrows MTL margin | monitored (Markov caveat approach (a)) |

Resolved (kept for audit): C03 (Δm metric), C04 (`next_mtl→GRU` head swap framing), C06 (GRU champion vs TCN), C07 (substrate-tied at region; framing pivot), C08 (CH04 retired), C11 (fold-leakage fixed 2026-04-17), **C12 fully closed 2026-04-28** (F37 closes Layer 3), C16 (CH15 reframed head-coupled), C17 (`next_single` demoted to head-sensitivity row).

---

## 8. Operational state (most recent)

- **Branch:** `worktree-check2hgi-mtl`
- **Most-recent commit:** `76f2443` *study(check2hgi): F37 FL closing — CH18/CH21 reframed scale-conditional + paper drafts*
- **Untracked (uncommitted) at session start:** `data/`, `docs/studies/check2hgi/PAPER_DRAFT.md`, `.gitignore.local`. Modified: `docs/BRACIS_GUIDE.md`, `docs/studies/check2hgi/PAPER_STRUCTURE.md`.
- **In-flight at last session end:** none (post-F37 doc updates landed).
- **Known operational gotchas (G1–G8):** see SESSION_HANDOFF_2026-04-22 + 2026-04-27. Notable:
  - **C09** SSD reliability — long FL runs need `/tmp`-resident I/O after observed SIGBUS (Thunderbolt SSD).
  - FL needs `--batch-size 1024` (2048 silently OOM-killed in fold 2 ep 23).
  - MPS env: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` + `PYTORCH_ENABLE_MPS_FALLBACK=1`.
  - **macOS `find /Volumes` triggers TCC re-prompts** that can silently revoke SSD access; scope future scans inside the worktree.

### Drafts queued from the overnight session (already moved into the repo)

- `paper/methods.md`, `paper/results.md`, `paper/limitations.md`, `paper/appendix_methodology.md` — section v0 skeletons.
- `scope/task_pivot_memo.md`, `scope/ch15_rename_proposal.md`, `scope/ch14_ch10_p02_decisions.md` — 3 decision memos awaiting user resolution.
- `launch_plans/f33_f36_colab.md`, `launch_plans/ca_tx_upstream.md` — Colab handoffs for the next compute push.
- `results/paired_tests/F49_decomposition_wilcoxon.json` (P4) + `H3alt_vs_B3_wilcoxon.json` (P5).
- `scripts/analysis/p4_p5_wilcoxon_offline.py` (no-scipy port).

---

## 9. Critical path to submission

```
Compute (~60h, mostly Colab T4)
├── F33 (Colab ~6h)        decides cat-head Path A vs B; gates F34/F35 logic
├── F36 FL Phase-2 grid    Colab ~3h; substrate replication FL (data on disk)
├── F34/F35 CA/TX upstream Colab ~24h
└── F24/F25 CA/TX 5f H3-alt Colab ~25h

Drafting (~15h)
├── Methods (paper/methods.md → full prose)
├── Results (paper/results.md → tables T1–T4 finalised)
├── Limitations (paper/limitations.md → AL=dev, FL frozen-cat instability, CA/TX gated)
├── Appendix Methodology — Layer 2 (paper/appendix_methodology.md, expand)
├── CH21_JOINT_CLAIM.md (figure suggestions + drafting hooks)
├── F49_METHODOLOGICAL_NOTE.md (~3pp Layer 2 standalone)
└── POSITIONING_VS_HMT_GRN.md rewrite (camera-ready, P8)

Cheap stat-strengthening (~2h, mostly DONE)
├── ✅ P4 paired Wilcoxon F49 cells (done)
├── ✅ P5 paired Wilcoxon H3-alt vs B3 (done)
├── P6 seed sweep H3-alt {0,7,100} on AL+AZ (~3h MPS) — NOT STARTED, paper-relevant
└── P7 narrative pass MTL_ARCHITECTURE_JOURNEY.md (~30 min)
```

### Proposed next moves (ranked, mirroring the 2026-04-28 review)

1. **Resolve the 3 user-decision memos** in `scope/`: task-pivot memo placement, CH15 rename option (A/B/C), CH14/CH10/P0.2 run-or-retire.
2. **Verify the F31 status** in FOLLOWUPS_TRACKER — is the AL 5f validation actually run, or just inferred from F27 numbers? Update tracker accordingly.
3. **Decide on the seed-sweep escalation** (P6) — at minimum it needs to be paper-blocking-marked or formally retired before reviewers ask.
4. **Run the F33 + F36 Colab pair** (~9h) — closes Path A/B for the cat head and lands FL Phase-2 substrate replication. Plan in `launch_plans/f33_f36_colab.md`.
5. **Launch CA + TX upstream pipelines** (P3) — plan in `launch_plans/ca_tx_upstream.md`. Long pole on the critical path.
6. **Start drafting Methods + Results** prose — claims are locked, no point waiting on CA/TX.
7. **Address forgotten items 4+5** (CH14 shuffle + CH15-original transductive) — either run them (cheap) or formally retire in CONCERNS.md.
8. **Schedule POSITIONING_VS_HMT_GRN.md rewrite** for camera-ready (deprecation note already prevents quoting wrong numbers).

---

## 10. Narrative-changing observations (don't lose these)

1. **`next_poi → next_category` task pivot is the biggest undocumented foundational decision.** Anyone reading the current paper-facing docs cannot reconstruct *why* without diving into `archive/`. The memo at `scope/task_pivot_memo.md` is drafted but not yet wired into PAPER_STRUCTURE.md / NORTH_STAR.md.

2. **CH15 ID reuse is a genuine audit-trail problem.** Two different scientific claims share one ID across the doc history. Decision memo at `scope/ch15_rename_proposal.md` proposes A=CH15→CH15b (default), B=CH22 fresh ID, C=redefinition note.

3. **F4 status (FL MTL-B3 5f clean re-run) was unclear** in the tracker; verified 2026-04-28 as **GENUINELY NOT EXECUTED as B3, but SUBSUMED by H3-alt 5f FL**. 14 bs=2048 B3-style FL runs are 1-fold only; only the bs=1024 H3-alt run is true 5-fold. F4 is now marked superseded.

4. **C15 was "resolved 2026-04-26" but RE-OPENED 2026-04-28** with FL caveat. The FL flip (STL > MTL by 8.78 pp p=0.0312) is the most narrative-changing finding of the post-merge sweep. CH18 retitled "scale-conditional"; CH21 reframed per-state.

5. **The architectural cost grows monotonically with region cardinality** (1.1K → 1.5K → 4.7K regions; arch-Δ +6.5 / −6.0 / −16.2 pp). This is **not a minor outlier**; it's the paper's most quantitatively striking per-state pattern.

6. **F49 Layer 2 (loss-side ablation unsound under cross-attn MTL)** is novel and paper-grade — applies retroactively to MulT/InvPT/HMT-GRN literature. Currently lives only in F49 docs, not yet in standalone `F49_METHODOLOGICAL_NOTE.md` or paper appendix outline (planned).

7. **Cat-side MTL > STL holds at all three states** (+0.94 to +3.64 pp). This is the *substrate-driven* generalisable contribution. The architectural lift is **AL-only** at our 3 states. The paper cannot collapse "MTL > STL" into one sentence — it must be split into substrate-driven cat (generalises) vs architecture-driven reg (AL-only).

8. **F37 P1+P2 ran on M4 Pro in 53 min total** (24.5 + 29 min) — far below the original 4050 GPU estimate. The 4050 critical-path concern in earlier reviews is now resolved.

9. **Phase-2 FL grid data is already on disk** — F36 is the cheapest remaining headline-blocker (~3h Colab). Worth doing before CA/TX simply because the disk artefacts already exist.

10. **The P5 statistical strengthening is COMPLETE** (P4 + P5 both landed 2026-04-28); only seed sweep (P6) remains and that's on the user's "nice-to-have" list, not paper-blocking. This is a recent change worth flagging — older review docs still reference these as outstanding.

---

## 11. File map for navigation

```
docs/studies/check2hgi/
├── README.md                              entry point + 3-track headline
├── AGENT_CONTEXT.md                       long-form briefing (post-H3-alt + F49 era)
├── NORTH_STAR.md                          champion config (F48-H3-alt, unchanged by F49)
├── PAPER_DRAFT.md                         title + abstract + section targets (NEW 2026-04-28)
├── PAPER_PREP_TRACKER.md                  paper-deliverable tracker (P1-P12)
├── PAPER_STRUCTURE.md                     paper scope, baselines, STL-matching policy
├── PHASE2_TRACKER.md                      Phase-2 substrate replication queue
├── OBJECTIVES_STATUS_TABLE.md             v5 scorecard
├── FOLLOWUPS_TRACKER.md                   per-experiment work queue (F-series)
├── CLAIMS_AND_HYPOTHESES.md               CH16-CH21 Tier A; CH15 reframed; etc.
├── CONCERNS.md                            C01-C17, C12 fully closed 2026-04-28
├── MTL_ARCHITECTURE_JOURNEY.md            end-to-end derivation
├── SESSION_HANDOFF_2026-04-{22..27}.md    chronology + operational gotchas
├── archive/
│   ├── phases_original/                   ⚠️ original P0-P7 plans (silently retired)
│   ├── pre_b3_framing/                    pre-2026-04-23 plans
│   ├── research_pre_b3/                   pre-B3 research
│   ├── research_pre_b5/                   pre-B5 research
│   └── 2026-04-20_status_reports/         pre-B5 snapshots
├── baselines/                             POI-RGNN, MHA+PE, STAN, REHDM faithful ports
│   ├── next_category/{poi_rgnn,mha_pe,comparison}.md + results/<state>.json
│   └── next_region/{stan,rehdm,comparison}.md + results/<state>.json
├── issues/                                bug audits (MTL_PARAM_PARTITION_BUG, etc.)
├── launch_plans/                          NEW 2026-04-28
│   ├── f33_f36_colab.md
│   └── ca_tx_upstream.md
├── paper/                                 NEW 2026-04-28 — section drafts v0
│   ├── methods.md
│   ├── results.md
│   ├── limitations.md
│   └── appendix_methodology.md
├── research/                              paper-substantive notes
│   ├── SUBSTRATE_COMPARISON_PLAN.md / FINDINGS.md  Phase 1
│   ├── F49_LAMBDA0_DECOMPOSITION_GAP.md / RESULTS.md  F49
│   ├── F37_FL_RESULTS.md                            F37 closing 2026-04-28
│   ├── F48_H3_PER_HEAD_LR_FINDINGS.md               H3-alt champion
│   ├── F21C_FINDINGS.md / F27_CATHEAD_FINDINGS.md   matched-head STL + cat-head
│   ├── B3_*, B5_*, GETNEXT_*, STAN_*, ATTRIBUTION_*, NASH_MTL_*
│   ├── POSITIONING_VS_HMT_GRN.md                    ⚠️ deprecated (rewrite for camera-ready)
│   └── ...
├── results/                               JSONs + tables
│   ├── RESULTS_TABLE.md                   per-state × per-method canonical
│   ├── BASELINES_AND_BEST_MTL.md          legacy paper-comparison (kept for audit)
│   ├── B3_baselines/, B5/, F2_fl_diagnostic/, F27_*/, F41_preencoder/
│   ├── P0/, P1/, P1_5b/, P1_5b_post_f27/, P2/, P5_bugfix/, P8_sota/
│   ├── phase1_perfold/, probe/, paired_tests/, baselines/
│   └── SCALE_CURVE.md
├── review/                                dated critical reviews
│   ├── 2026-04-22T1040_critical_review.md
│   ├── 2026-04-23_critical_review.md
│   ├── 2026-04-27_study_overview_and_forgotten_items.md
│   ├── 2026-04-28_post_phase1_overview_and_gap_audit.md
│   └── 2026-04-28_morning_briefing.md
└── scope/                                 NEW 2026-04-28 — decision memos
    ├── task_pivot_memo.md
    ├── ch15_rename_proposal.md
    └── ch14_ch10_p02_decisions.md
```

---

## 12. What I deliberately did NOT do

- Did not modify any tracker, claim, or concern doc.
- Did not run any commands against the python venv or training infrastructure.
- Did not commit anything.
- Did not call advisor (this is a compilation task that doesn't yet require external review; reading-and-synthesising is appropriate without it).
