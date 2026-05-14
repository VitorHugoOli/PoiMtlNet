# C05 — Phase-3 Null-Result Fallback (decision tree, 2026-04-29)

**Concern:** P3 (Phase-3 = MTL CH18 grids on CA + TX) may fail to replicate the AL/AZ/FL pattern. The current state in `PHASE2_TRACKER.md`:

| state | STL probes | MTL CH18 |
|---|---|---|
| AL | 🟢 done (Phase-1) | 🟢 done (Phase-1, +16.75 cat / +29.65 reg) |
| AZ | 🟢 done (Phase-1) | 🟢 done (Phase-1, +17.11 cat / +31.72 reg) |
| FL | 🟢 done (T1, 2026-04-28) | 🟢 done — but reg-side gap was inflated by C4 leakage; clean B9 = 63.47 reg, +3.34 pp vs H3-alt clean |
| **CA** | 🟡 STL closed 2026-04-29 | 🔴 **BLOCKED on memory** (Lightning A100 / Colab T4 RAM same as the FL OOM blocker) |
| **TX** | 🔴 STL pending | 🔴 **BLOCKED on memory** |

So C05 splits into TWO related questions:

1. **Infrastructure**: how do we unblock the memory-constrained MTL CH18 ×2 runs?
2. **Scientific**: if we get the runs but they give null/negative, how does the paper handle that?

This doc handles (2) — the scientific contingency. (1) is tracked separately in `PHASE2_TRACKER.md`.

---

## 1 · What "null result" means at P3

A P3 null result is one of three patterns:

| pattern | Δreg vs STL ceiling | Δcat vs STL ceiling | what it means |
|---|---:|---:|---|
| **A. Clean miss** | < +0 pp (regression) | < +0 pp | MTL hurts both tasks; substrate-specific MTL claim refuted at CA/TX |
| **B. Pareto trade** | < +0 pp on reg | > +0 pp on cat | substrate-specific MTL claim **partial** — works for cat-only states |
| **C. Wash** | within ±σ on reg | within ±σ on cat | unexpected: AL/AZ/FL all show clear lift; CA/TX behaves like baseline |

The most likely null is **A** (FL already showed reg < STL ceiling under leak-free conditions — see `F50_T4_FINAL_SYNTHESIS.md`). The remaining hypothesis to confirm is whether MTL can still close the cat-side gap at CA/TX.

---

## 2 · Decision tree

### Branch 1 — Pattern B (Pareto trade, cat lift only, reg null)

**Action:** keep the paper as-is, but reframe CH18 from "MTL beats STL on both tasks at all states" to "MTL beats STL on cat at all states; reg lift is scale-conditional (FL/CA/TX above STL-reg ceiling)."

- Update `OBJECTIVES_STATUS_TABLE.md` Obj 2b row to mark CA/TX reg as 🟡 (cat-only).
- The temporal-dynamics narrative (F50 T3 §6 + D5 + F63) still explains WHY: at high-cardinality reg, MTL's reg encoder saturates earlier than STL's, so STL has more late-window α growth.
- Add a "scale-conditional reg lift" sentence to the abstract.

### Branch 2 — Pattern A (clean miss on both)

**Action:** retract CH18 as a positive claim; reframe to a methodological + temporal-dynamics paper.

The paper currently leans on three claims:
- **CH16/CH18 substrate** — Check2HGI > HGI for cat (AL+AZ Phase-1, FL Phase-2 T1). **Independent of P3** — survives.
- **CH18 MTL > STL** — currently 🟢 AL+AZ, 🟢 FL (cat) / 🟡 FL (reg, scale-conditional). **At risk if CA+TX flip both reg AND cat negative.**
- **Mechanism narrative** — F50 T3 + F50 D5 (encoder saturation timing) + F63 (α trajectory) + B9 paired Wilcoxon p=0.0312. **Independent of P3.**

Under Pattern A:
- Retract Obj 2b reg lift across the board → demote to "cat lift; reg shows scale-conditional behaviour driven by α saturation."
- Lean harder on the **mechanism story** (F50 T4 ≈ 50% of the paper). The phenomenon "MTL reg saturates at ep 5 while STL grows through ep 16-20" is a substrate-agnostic observation; CA/TX null findings would *strengthen* that story by showing the saturation generalizes.
- Re-frame: "We document MTL's reg-side training-dynamics ceiling and propose interventions (B9, P4) that close the gap at FL but do not generalize to higher-cardinality regions (CA/TX). Future work: scaling laws for the MTL reg saturation point."

### Branch 3 — Pattern C (wash on CA+TX)

**Action:** investigate before deciding. This pattern is anomalous given AL/AZ/FL all showed clear lift; a wash on CA/TX with similar settings is likely a data-quality or pipeline issue, not a real null.

Diagnostic steps:
1. Verify per-fold log_T was used (re-check the CH18 commands).
2. Run STL-only on CA/TX with same recipe to recompute STL ceiling (currently CA is 🟡 partial).
3. Sanity-check that the embedding is loading correctly (region cardinality should match CA: 8050+ regions, TX: 9870+ regions).
4. If the wash persists: P3 results don't generalize → degrade to Branch 2 narrative.

---

## 3 · Unblocking infrastructure (cross-ref to PHASE2_TRACKER.md C1)

The MTL CH18 memory blocker is currently the gating issue. Three mitigations from PHASE2_TRACKER:

| mitigation | cost | scientific impact |
|---|---|---|
| **C1 footnote**: declare CA/TX MTL deferred due to memory constraints | 0 min | Per `PHASE2_TRACKER.md §272`: "CH18 confirmed at smaller-scale states; large-state MTL deferred". Honest, conservative. **Recommended default if Branch 1 or 2 above is also expected.** |
| Drop reg head to lower cardinality (subsample regions, take top-K) | ~1h dev | Scientifically muddied — changes the prediction task |
| Find lower-RAM trainer (gradient checkpointing, paged optimizer, distributed) | days | High effort; only worth it if reviewers explicitly demand CA/TX |

**Recommendation:** unless reviewers ask for CA/TX MTL specifically, accept the C1 footnote and lean on AL+AZ+FL evidence.

---

## 4 · Decision matrix at submission time

| evidence available at submission | recommended paper framing |
|---|---|
| AL+AZ+FL only (current state) | "CH18 substrate-specific MTL confirmed at AL/AZ/FL; CA/TX MTL deferred (memory)." Branch 1 framing on reg lift. |
| + CA/TX Pattern A (clean miss) | Branch 2 — pivot to methodological + mechanism paper. Heavy F50 T4 emphasis. |
| + CA/TX Pattern B (cat lift, reg null) | Branch 1 with explicit scale-conditional caveat. |
| + CA/TX Pattern C (wash) | Stop, diagnose, then re-decide. |

---

## 5 · Status at this commit

- Concern **C05** in `CONCERNS.md` is currently `deferred — decision triggered only if P3 falls short.`
- This doc pre-writes the fallback so the decision is fast when P3 lands.
- No code or run changes from this doc; it's a planning artifact.

**To do when P3 lands:** read this doc, identify the pattern, execute the corresponding branch.

---

## 6 · Cross-references

- `CONCERNS.md` C05 entry
- `PHASE2_TRACKER.md` — CA/TX MTL CH18 BLOCKED tracker + C1 mitigation
- `OBJECTIVES_STATUS_TABLE.md` — Obj 2b row (currently 🟡 at FL, 🔴 at CA/TX)
- `F50_T4_FINAL_SYNTHESIS.md` §3.4 — cross-state portability summary (AL/AZ/GA already directional, not paper-grade)
- `F50_D5_ENCODER_TRAJECTORY.md` — substrate-agnostic mechanism that survives all three branches above
