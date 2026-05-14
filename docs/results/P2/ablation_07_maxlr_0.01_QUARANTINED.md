# max_lr=0.01 result — QUARANTINED (power/idle issue)

**Flagged:** 2026-04-19 by user. The max_lr=0.01 step-7 run took 635 min wall time (vs ~20 min expected). Root cause was the Mac discharging / going to idle mid-training, not MTL instability.

**Status:** the numbers from this run are UNRELIABLE:
- reg Acc@10 = 12.33 ± 11.79 (wide std)
- training wall time = 635 min
- Δm = −43.37%

These values were influenced by the training loop pausing/resuming across idle cycles; neither the LR finding nor the "region collapse at 0.01" conclusion can be trusted on this data alone.

**Implication:**
- **max_lr=0.003 remains the valid winner** (ran cleanly in ~20 min).
- **max_lr=0.001 baseline** was also valid.
- The claim "max_lr=0.01 destabilises region" needs a clean rerun or must be withdrawn.

**Action:**
- Phase 2 (5 AL reruns at max_lr=0.003) continues as planned — not affected.
- Consider a clean rerun of max_lr=0.01 after Phase 2 if the paper needs the full sweep data.
- Paper should cite only max_lr=0.001 and max_lr=0.003 for the fair-LR comparison; max_lr=0.01 is noted as "not cleanly measured".

**File kept for transparency:** `ablation_07_maxlr_0.01_al_5f50ep.json` remains in the repo with this note pointing to its unreliability.
