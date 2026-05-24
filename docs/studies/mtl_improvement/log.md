# MTL Improvement Track — Progress Log

Append-only chronological log. Every agent working on this track adds entries here.
Dates are absolute (e.g. `2026-05-16`), never relative ("today", "yesterday").

Sections at the bottom of each entry:
- **Decision** if you changed direction.
- **Blocker** if you got stuck (and what unblocked you, in a later entry).
- **Chain status**: which tier the chain is in; whether the chain is preserved or broken.
- **Next** what the next agent should pick up.

---

## 2026-05-16 — Track designed, awaiting execution

**Phase**: Design complete; no experiments run yet.

**What happened**

- Folder `docs/studies/mtl_improvement/` created. INDEX.html + this log + AGENT_PROMPT.md landed.
- Design session: read every preliminary file in `docs/studies/mtl-exploration/` (INDEX.html, considerations.md, EXPERIMENT_NO_ENCODERS.md, EXPERIMENT_HGI_SUBSTRATE.md, LEAK_BLAST_RADIUS_AUDIT.md, README.md). Re-read `canonical_improvement/` as structural template (INDEX.html, log.md head, AGENT_PROMPT.md, considerations.md).
- Three-dimensional audit (conceptual / technical-feasibility / metrics+baseline-robustness) of the 11 considerations in `mtl-exploration/considerations.md`. Verdict table in INDEX.html §Audit. All 10 numbered considerations accepted; 2 were right-sized (C2 from "experiment" to "pre-flight gate"; C4 with row-pairing-constraint flag).
- Ran 7 parallel breadth sub-agents (heads / backbone / loss / optimization / data-sampling / input-modality / instrumentation). Total: ~288 candidate directions across all angles. Strongest pulls summarized into the experiment slate in INDEX.html §Breadth.
- Critical correction from sub-agent 6: I had the encoder-asymmetry hypothesis inverted relative to AL Step-3 evidence in `EXPERIMENT_NO_ENCODERS.md`. Cat needs the thick MLP at AL scale (−2.57 pp paper-grade loss if simplified); reg fine with Linear+LN. Carried into design.
- Built dependency map. Designed 8-tier chain (T0 hygiene + diagnostics → T1 STL ceilings → T2a cheap backbones → T2b heavy backbones → T3 loss → T4 batch → T5 LR/optim → T6 α formula → T7 final head re-ablation → T8 multi-seed ship). Within each tier, experiments are parallel; tier-to-tier is sequential with explicit decision gates.
- User alignment captured via `AskUserQuestion`:
  - Tier 2 scope: aggressive — 8 archs faithful (4 cheap T2a + 4 heavy T2b).
  - C3 audit: full leak-free 3-way decomposition at 5 states × 1 seed + cross-arch check on T2 winner.
  - Folder/branch: new folder `docs/studies/mtl_improvement/` + new branch `mtl-improve`.
  - HGI substrate re-check: yes — one cheap HGI run per T2 winner.
- Pre-write advisor pass (model-level `advisor()`). Five substantive items addressed before writing:
  1. **Tier 2 LR confound mitigation**: per-arch light LR mini-sweep (constant 1e-3 / B9 per-head / arch-suggested-default; 5f × 30ep × seed=42 at AL+AZ) before judging the winner. Each arch wins or loses under its own best LR regime.
  2. **Per-tier decision gates**: every tier card now has explicit win-condition, no-winner fallback, and early-stop-if-headline-closes language.
  3. **C2 right-sized** to pre-flight gate (not a tier-defining experiment).
  4. **C4 row-pairing constraint flagged** in the audit row.
  5. **T0.7 same-machine sanity check** clarified: NOT v11-redundant; it is the canonical_improvement-style pre-flight pattern. Document the rationale in the experiment card so the implementing agent doesn't skip it.
  - Smaller flags also addressed: per-tier compute estimates, scale-conditional-vs-universal recipe commitment, Tier 2a/2b phased split with stop gate.

**Decisions**

1. **Branch `mtl-improve`** is the dedicated worktree. Do NOT contaminate `canonical_improvement` or `check2hgi-up` work.
2. **Scale-conditional framing preserved**: the ship recipe in T8 may be B9-prime at FL/CA/TX and H3-alt-prime at AL/AZ. Hunting a single universal recipe is permitted as a stretch outcome but is NOT a win condition.
3. **HGI substrate** is locked out as the substrate (per `EXPERIMENT_HGI_SUBSTRATE.md`); each T2 winner gets one cheap HGI sanity probe (single seed, AL+AZ, 5f × 25ep). If MTL+HGI is non-null under any new arch, escalate to user.
4. **Tier 2 phased split**: T2a (4 cheap archs ≤60 LOC each) gates T2b (4 heavy archs 80–250 LOC each). T2b becomes optional if a T2a winner already clears paper-grade significance.
5. **Per-arch light LR mini-sweep is mandatory** for every backbone candidate in T2a + T2b. Cost: 2–3× single-LR-comparison budget, but the alternative is silent confound.
6. **F49 audit scope**: 5 states × seed=42 × leak-free + one cross-arch validation on T2 winner. ~30 GPU-h H100.

**Chain status**: Tier 0 not yet launched.

**Next**

1. **Implementing agent** must read in order: this log, `AGENT_PROMPT.md`, `INDEX.html` (top-down, including §Execution guidelines), then NORTH_STAR.md + RESULTS_TABLE.md §0 + `mtl-exploration/EXPERIMENT_*.md` for grounding.
2. Create the dedicated worktree on branch `mtl-improve` before launching anything.
3. Start with Tier 0 in parallel:
   - **T0.2 must complete BEFORE T1** (mask/pad audit gates Tier 1 launch).
   - **T0.7 must complete BEFORE T1** (same-machine re-baseline; pin `B9_v11_repro`).
   - T0.1 / T0.3 / T0.4 / T0.5 / T0.6 can run in parallel.
4. Use `TaskCreate` to break down each Tier-0 experiment into unit-test → validate → launch → import → analyze sub-tasks.
5. After Tier 0 completes, call advisor with the Tier-0 results before launching Tier 1.

---

## 2026-05-16 — Post-write advisor pass — design tightened

**Phase**: Design hardening; still no experiments run.

**What happened**

Spawned a mandatory final-advisor sub-agent (per the user brief) to stress-test the design before handoff. Advisor surfaced five substantive items, all applied as text edits to INDEX.html:

1. **T2 LR mini-sweep was structurally biased toward the incumbent.** Original sweep had 3 regimes (constant 1e-3 / B9 per-head / arch-suggested-default). Advisor noted: 33% of the sweep is the incumbent recipe; new arch-specific params (cross-stitch per-channel α, AdaShare Gumbel gates, TaskExpert prompts) get no dedicated LR group; 30 epochs is short for slow-α-growth archs; AL+AZ-only sweep excludes states where heavy archs may differentially win. **Fix:** widened to 5 regimes (added R4 per-arch-group LR for new param groups, R5 B9+warmup-5%), extended to 40 epochs, added FL mini-sweep cell for T2b.2 (AdaShare) and T2b.3 (TaskExpert).
2. **Decision gates were loose** — many tier cards had `<div class="gate">Standard.</div>`. Advisor warned: "Pareto-dominates" without a magnitude floor invites cherry-picking. **Fix:** added explicit minimum-effect floor to all gates — `≥ 1 pp on targeted axis (n=5 single-seed) OR ≥ 0.5 pp (multi-seed n=15+) AND other axis non-inferior at TOST δ=2 pp`.
3. **HGI sanity probe was cargo-culted at single-seed n=5** — cannot detect a substrate × arch interaction at fold-σ scale. **Fix:** bumped to 2 seeds {42, 0} × AL+AZ × 5f × 30ep with explicit ≥ 2 pp escalation threshold; below threshold = informational only, no escalation.
4. **T4 was over-scoped.** Advisor: T4.2 (cat oversampling) is approximately subsumed by existing weighted-CE path; T4.4 (geo hard negatives) is too speculative for the cost. **Fix:** marked T4.2 and T4.4 as DEFERRED with explicit deferral-rationale blocks; kept T4.1 and T4.3 only. Tier-4 budget revised 80 → 40 GPU-h.
5. **Two chain-break risks were missing.** **Fix:**
   - Added "T1 STL_v2 winner ≠ Table A head" row: chain compares to BOTH (Table A for v11-paper continuity, STL_v2 for upper bound) in every Tier-2–7 results block.
   - Added "T0.5 instrumentation overhead pushes fold wall > 10%" row: T0.5 acceptance now includes a perf gate (instrumented run must complete within 10% of uninstrumented baseline; if over, demote per-step gradient cosine to per-N-step sampling).

Compute budget revisions per advisor: T2b 300 → 450 GPU-h; T8 200 → 300 GPU-h. Total envelope is now **~1700 GPU-h** (full chain incl. T2b) or **~1250 GPU-h** (if T2a wins the gate and T2b is skipped). Added top-level total-compute summary to TL;DR.

**Items NOT changed** (advisor flagged but design judgment preferred current spec):
- C3 (F49 audit) kept at the full L-cost scope as user-aligned; advisor agreed.
- T2a/T2b structure preserved; only the mini-sweep was widened.
- No tier was added or removed beyond the T4 deferrals.

**Chain status**: design hardened; still pre-launch.

**Next**: same as previous entry. Implementing agent should treat the updated INDEX.html as the source of truth; the changes above are reflected inline (T2a/T2b mini-sweep callout, T4.2/T4.4 deferred blocks, decision-gate floors throughout, chain-break-risk additions, T0.5 perf gate, compute revisions).

---

## How to add an entry to this log

Use this template for every working session:

```markdown
## YYYY-MM-DD — Short title

**Phase**: Tier X.Y in flight / completed / paused.

**What happened**
- Bullet point.
- Bullet point.

**Findings** (if any)
- Numeric, per-state where applicable.

**Decision** (if any)
- What changed and why.

**Blocker** (if any)
- What's blocking; what input is needed to unblock.

**Chain status**: T?-? in flight / chain preserved / chain broken (reason).

**Next**
- What the next agent should pick up. Be specific (experiment ID, state, seed, expected wall).
```

**Rules**:
- Append at the bottom; never edit historic entries.
- Date is the YYYY-MM-DD of the work session (UTC if cross-zone).
- If you finish a tier, flag it explicitly (`Tier X COMPLETE — pinned recipe: ...`).
- If you break the chain (run an out-of-order experiment), document `**Chain status**: broken because <reason>` AND the re-execution plan to restore it.
- If you fork the design (add a new experiment not in INDEX.html), add it to the HTML in the same session AND document here.
