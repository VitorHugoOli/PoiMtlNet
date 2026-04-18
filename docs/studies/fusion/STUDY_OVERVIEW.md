# Fusion Study — Overview & Status

**Last updated:** 2026-04-18 (P2 C06 minimal test **done** — partial verdict)
**Current phase:** P2 (3 tests archived: 1 MTL reused from P1, 2 single-task)
**Branch:** `main`

This is the consolidated picture of what the fusion study is, what's been decided, what's been measured, and what's left. For the authoritative day-to-day state, read `state.json`. For claim-level detail, read `CLAIMS_AND_HYPOTHESES.md`. For methodology flaws discovered after P1 closed, read `issues/P1_METHODOLOGY_FLAWS.md`.

---

## 1. What the paper is supposed to say

Working thesis (subject to revision based on P2 C06 outcome):

> On multi-source POI fusion (HGI + Sphere2Vec, 128-dim), a modern MTL
> configuration (expert-gating architectures, matched effective batch)
> outperforms single-task baselines on joint F1, contradicting CBIC 2025's
> claim that "MTL doesn't help." Fusion is the first-order lever; optimizer
> choice is null at matched batch; expert-gating helps at short training
> but ties with FiLM base at 5-fold × 50-epoch confirmation.

Three Tier-A claims have moved substantially during P1 and the critical
review. The paper narrative is weaker on *MTL optimizer machinery* than
originally planned, and stronger on *embedding-source fusion* as the driver.

---

## 2. Phase overview

```
P0 [completed 2026-04-15]  — embeddings, integrity, leakage audit
P1 [completed 2026-04-18]  — 189 tests, arch × optim × matched-batch grid
P2 [RUNNING]               — heads + MTL vs single-task (C06 #1 paper claim)
P3 [planned]               — cross-engine (HGI/DGI/fusion) on champion
P4 [planned]               — hparam robustness + head co-adaptation
P5 [planned]               — mechanism figures (gradient cosines, etc.)
P6 [planned]               — Caruana-era MTL claims rechecked
```

Compute budget used: ~8 hours GPU over 2 days.
Compute budget remaining (estimated): 12–20 hours for P2 alone; ~30–50 hours for P3–P6.

---

## 3. What P0 established

- AL, AZ, FL embeddings regenerated after the 2026-04-13 label-bug fix.
- AZ and FL DGI parquets are **pre-bugfix** (missing `placeid`); folds fell
  back to StratifiedKFold. **Blocks P3 DGI on AZ/FL** until regenerated.
- FL/fusion half-L2 ratio = 40.99× (AL is ~15×). Scale imbalance between HGI
  and Sphere2Vec halves is state-specific. **Confounds C01 on FL**.
- CBIC sanity run: AL/DGI/nash_mtl/5f×50ep → cat F1 0.461 (target 0.46–0.48 ✓),
  next F1 0.243 (target 0.26–0.28, ~1.7 p.p. below — N04 provisional).
- **HGI leakage audit (C29 confirmed):** On HGI-only, category F1 drops
  0.786 → 0.144 (random chance) when fclass is shuffled across POIs. Category
  F1 on OSM-Gowalla is primarily fclass-identity preservation, not learned
  spatial structure. **Next F1 is the representation-quality metric.** (N03)
- **No classical label leakage in HGI training** (C30 confirmed, grep-verified
  across `research/embeddings/hgi/`).

---

## 4. What P1 established — three stages × two states × multi-seed

**Stages:** screen (1f × 10ep, 75 cells/state) → promote (2f × 15ep, 10/state)
→ confirm (5f × 50ep, 5/state) → multi-seed F2 (8 cells at seeds 123+2024).

**189 P1 tests archived. Zero crashes. Zero surprises triggered. 100% config compliance.**

### 4a. Headline results

**AL champion (3 seeds):**
- `mmoe4 × gradnorm`: joint@J = **0.4080 ± 0.0008**, joint@T = **0.4232 ± 0.0022**
- `cgc22 × equal_weight`: joint@J = 0.4115 ± 0.0087, joint@T = 0.4237 ± 0.0026
- Tied at joint@T (Δ=−0.0005). mmoe4×gradnorm is 10× more *reliable* (lower seed variance).

**AZ champion (3 seeds):**
- `cgc21 × uncertainty_weighting`: joint@T = 0.4394 ± 0.0033
- `cgc21 × dwa`: joint@T = 0.4412 ± 0.0040
- Tied (Δ=+0.0018). Use uw for continuity.

### 4b. Claim verdicts from P1

| Claim | Verdict | Evidence |
|-------|---------|----------|
| C01 fusion > HGI | `pending` | Test is in P3. |
| **C02** grad-surgery > eq on fusion | **`refuted`** | Multi-seed null; all Welch \|t\| < 0.7. No effect under either checkpoint policy on either state. |
| C03 eq on single-source | `pending` | Test is in P3. |
| C04 arch ranking ↔ embedding | `pending` (early hint) | Cross-state arch preference within fusion: AL favors mmoe4/cgc22, AZ favors cgc21. Could be seed artifact. |
| **C05** expert-gating > FiLM base | **`partial`** (downgraded from `confirmed`) | Screen supports (+0.03–0.045 AL); F4 confirm run shows `base × eq` ties with expert cells (spread 0.0014 at joint@T). Benefit is short-training-only. |
| **C18** reproducibility (seed std < 0.01) | **`confirmed`** (early-resolved) | All F2 candidates pass on joint@T; cgc22×eq borderline on joint@J (0.0087). |
| **C31** fclass shortcut on fusion | **`partial`** (new) | Linear probe: HGI-half F1=0.688, Sphere2Vec F1=0.111, full fusion 0.682 (no lift). Shortcut fully inherited; not mitigated. Full arm-C retrain deferred to journal. |
| **C32** joint-peak checkpoint bias | **`confirmed`** (new) | Per-task-best selection inverts AL ranking and changes AZ winner; ranking depends on checkpoint policy. Mandatory to report both going forward. |

### 4c. Narrative implications

- **Paper is not about MTL optimizer machinery.** C02 refuted kills the
  "gradient-surgery is the secret sauce" angle. N02 already anticipated this.
- **Paper is not primarily about architecture.** C05 partial means expert-
  gating is a convergence accelerator, not a ceiling-raiser at the confirm
  budget.
- **Paper has to lead with embedding fusion.** C01 in P3 is now doing the
  heavy lifting. If fusion clearly beats HGI on next F1 (the shortcut-free
  metric), the story works. If not, major reframing.
- **Paper must caveat category F1 on fusion.** C31/N03 apply. Category F1
  measures fclass-preservation fidelity, not representation quality. Next F1
  is the honest metric.
- **Paper must use per-task-best checkpoint selection** or report both
  policies (C32). Under joint-peak, single-task-next would be artificially
  favored vs MTL — C06 would be wrongly refuted.

---

## 5. P2 (C06 multi-seed confirmed, C12 running, 2026-04-18)

**Goal:** answer C06 (the #1 paper claim): does MTL beat single-task on fusion?

**Multi-seed AL result (3 seeds × 5 folds = n=15 paired folds):**

| Metric | MTL (`mmoe4×gradnorm`) | ST (base encoder) | Mean Δ | paired-t (df=14) | Wilcoxon W+/120 | p (two-sided) | folds MTL > ST |
|--------|-----------------------|-------------------|--------|------------------|-----------------|---------------|----------------|
| cat @taskbest | 0.8303 | 0.8089 | **+0.0213 (+2.1 p.p.)** | **+5.01** | **119** | **0.0008** | **14/15** |
| next @taskbest | 0.2840 | 0.2712 | +0.0128 (+1.3 p.p.) | +3.56 | 110 | 0.0045 | 11/15 |
| **joint@T (HM)** | **0.4231** | **0.4062** | **+0.0169 (+1.7 p.p.)** | **+4.28** | **110** | **0.0045** | **11/15** |

**C06 verdict: `confirmed` on AL.** Statistically significant at multi-seed on all three metrics (cat p=0.0008, next p=0.0045, joint p=0.0045). Cat delta +2.1 p.p. clears the phase-doc's 2 p.p. practical-significance bar; joint@T +1.7 p.p. is just under it but highly significant. **Refutes CBIC 2025's "MTL = single-task" claim.**

**AZ cross-state (1 seed × 5 folds):**
- Cat: single-task 0.7508 > MTL 0.7380 → Δ = **−1.3 p.p. (state-asymmetric regression)**
- Next: MTL 0.3103 > single-task 0.3005 → Δ = +1.0 p.p.
- Joint@T: MTL 0.4394 > ST joint@T HM(0.7508, 0.3005) = 0.4306 → Δ = +0.9 p.p.
- **AZ partial — MTL still wins joint but regresses on category.** Needs multi-seed replication.

**C28 (no negative transfer) verdict: `confirmed` on AL, `partial` on AZ.** 14/15 AL cat folds, 11/15 AL next folds show MTL ≥ ST. AZ cat regression is a red flag for the strong form of C28.

**C32 (checkpoint-bias) reinforced.** Joint@J (MTL) vs joint@T (ST) gives +0.002 (n.s.) while joint@T vs joint@T gives +0.017 (p<0.005). The two policies disagree even at n=15. MUST report both.

**Paper narrative (post multi-seed, 2026-04-18):**
- **C06 clear answer: MTL beats single-task on AL fusion**, by ~1.7 p.p. joint, ~2.1 p.p. category, ~1.3 p.p. next. All significant at p < 0.005.
- **Directly refutes CBIC 2025's finding** under the new configuration.
- **Magnitude is modest.** Fusion remains the first-order lever; MTL adds second-order benefit.
- **Cross-state asymmetry on AZ** is a honest-to-report caveat: AZ cat regresses. Needs AZ multi-seed.

**C12 CBIC-contrast running:** 3 MTL runs (base × nash_mtl on DGI, HGI, Fusion AL 5f × 50ep) currently in drain. Will quantify whether CBIC's config fails equally on all embeddings.

**Remaining P2 work (deferred, lower priority now that main story is resolved):**
- AZ multi-seed C06 (4 runs, ~20 min).
- Head grid (C08, C10) — 9 × 10 combinations, ~8-10 h. Now low priority; main MTL story answered.
- Co-adaptation probe (C09) — optional, 10-20 runs.

---

## 6. Open issues

### In `state.json` (study-level)

| ID | Kind | Blocks | Status |
|----|------|--------|--------|
| `az_fl_dgi_stale` | data integrity | P3 DGI on AZ/FL | open |
| `fl_fusion_scale` | data integrity | P3 fusion on FL | open (watch) |
| `ISS-003` (F1) | methodology | resolved via joint@T backfill | mitigated |
| `ISS-004` (F2) | methodology | resolved via multi-seed | resolved |
| `ISS-005` (F3) | methodology | proxy-resolved; full arm-C deferred | partial |

### Not in state.json but real

- **F5 NextHead schedule mismatch.** Next peaks at epochs 10–22, category at 17–45 on AL confirm. Under OneCycleLR over 50 epochs, the next head trains past its peak. Documented in issues; proposed P4 experiment: separate LR schedule per task or early-stopping next head.
- **F6 bayesagg_mtl × cgc22 AZ collapse.** Screen cell joint = 0.271 vs peers ~0.40. Analyzer didn't flag (band too wide). Low priority, but real.
- **F7 expected-range calibration.** P2/P3 should use per-arch ±3σ instead of the uniform [0.10, 0.65] inherited from P1 grid.
- **Full arm-C retrain on fusion** (F3 primary) — deferred to journal extension. Probe evidence is qualitatively decisive.

---

## 7. Claim catalog snapshot (32 claims + 4 negations)

### Confirmed (6)
**C06** MTL vs single-task (multi-seed p<0.005 AL) · **C18** reproducibility · **C28** no negative transfer (AL; partial AZ) · **C29** fclass shortcut (HGI) · **C30** no classical label leakage · **C32** checkpoint-selection bias

### Refuted / partially refuted (1)
**C02** gradient-surgery (refuted at multi-seed; N02 anticipated)

### Partial (3)
**C05** expert-gating (screen yes, confirm tied) · **C07** asymmetric MTL benefit (partial data from C06) · **C31** fclass shortcut on fusion (probe-confirmed, retrain deferred)

### Pending (20)
| Phase | Claims pending |
|-------|---------------|
| P2 | C08, C09, C10 |
| P3 | C01, C03, C04 (partial data), C11, **C12** (running), C13, C14 |
| P4 | C15, C16, C17 (robustness) |
| P5 | C19, C20 (mechanism) |
| P6 | C22, C23, C24, C25, C26 (deferred), C27 |

### Negations (4)
N01 (no universal SOTA) · **N02** (no gradient-surgery requirement) · **N03** (category F1 not a representation metric on OSM-Gowalla) · N04 (protocol-delta provisional)

---

## 8. Paper-readiness assessment

### Can publish today with P0+P1 only?
**No.** C06 is the #1 paper claim and has zero data. Without C06, the paper has no answer to the question "does MTL help?" — which is the whole premise.

### Can publish after P2 C06?
**Depends on the verdict.**
- **If C06 confirmed:** paper is about "MTL + fusion works when configured right (against CBIC 2025's claim)." Needs P3 for external validity (C01 + C11).
- **If C06 refuted:** paper reframes to "fusion > single-source regardless of MTL." Still publishable, different narrative. Needs C28 no-negative-transfer + next F1 improvements from fusion to carry the story.
- **If MTL < single-task:** most interesting outcome — publishable as a honest re-run of CBIC's finding under modern config. Would refute the paper's original thesis but open a richer discussion.

### Reviewer-critical tests still outstanding
- **C28** (no negative transfer) — Wilcoxon paired-fold test, free from P2 logs. MANDATORY.
- **C01** (fusion > HGI) — test of the fusion premise. BLOCKING on P3.
- **C11** (embedding quality is dominant factor) — supports the "focus on embedding" angle. BLOCKING on P3.
- **C14** (Time2Vec helps next-POI) — needed if fusion story hinges on next F1. BLOCKING on P3.

### Reviewer-critical caveats the paper MUST contain
- **N03** category F1 ≠ representation quality on OSM-Gowalla. Reframe metric discussion.
- **N04** user-isolated split explains next-F1 delta vs CBIC / HAVANA.
- **C32** joint-peak checkpoint bias acknowledged; results reported under both policies.
- **C31** fclass shortcut survives on fusion (preliminary; proxy). Defer the full quantification honestly.

---

## 9. Worth-answering questions (prioritized)

### Blocking for paper submission

1. **C06** — MTL vs single-task on AL. Running right now; result ~1 hr away.
2. **C28** — no negative transfer. Free from P2 logs once they land.
3. **C01** — fusion > HGI. Requires P3 champion cross-engine runs.
4. **C14** — Time2Vec specifically helps next-POI. One ablation run.

### Strong-to-have before submission

5. **C12** — CBIC config (base + NashMTL) fails on all engines. 3 runs.
6. **C11** — embedding range dominates. 3 embeddings × champion config.
7. **C27** — MTL backbone transfers better than single-task. Supplementary figure material.

### Post-submission / journal

8. Full arm-C retrain on fusion (C31 `partial` → `confirmed`).
9. **C26** sample-efficiency (Caruana's flagship claim). 60 runs. Defer unless P6 finishes early.
10. Multi-seed extension of C06 once the minimal is in.

---

## 10. Session-to-date commits (current session)

```
ee481d8 study(fusion): archive F4 base cell + 8 F2 multi-seed run dirs
cb4e3d4 study(fusion): F2 multi-seed finalization — C02 refuted, C18 confirmed
42b4035 study(fusion): F4 result — base × equal_weight is competitive at confirm
0161c7a study(fusion): F1 mitigated + F3 proxy — joint@T backfill & fclass probe
bbb5f2d study(fusion): P1 critical review — C32 checkpoint bias + methodology flaw tracker
562bf71 study(fusion): complete P1 — 180-test arch × optim grid on AL + AZ
```

Working tree clean as of the last commit. P2 C06 minimal is running; its artifacts will be committed when it lands.

---

## 11. How to resume

```bash
# 1. See current study state
.venv/bin/python scripts/study/study.py status

# 2. Check on the running P2 C06 drain
tail /tmp/drain_p2_c06.log

# 3. After C06 returns, run the finalization:
#    - Compute MTL joint@T vs single-task joint@T
#    - Paired t-test across 5 folds
#    - Update C06 status in CLAIMS
#    - Consider multi-seed extension (2 more runs × 2 cells)

# 4. Decide P2 scope expansion (head grid) based on C06 verdict.
```

For emergency "what does everything mean" questions, read this file top-down.
For "what to do next" questions, read `HANDOFF.md`.
For "which claim does what" questions, read `CLAIMS_AND_HYPOTHESES.md`.
For "why is X the way it is" questions, read `issues/P1_METHODOLOGY_FLAWS.md`.
