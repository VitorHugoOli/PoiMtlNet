# GradNorm Deadlock / Collapse with Expert-Gating Archs — Audit

**Severity:** MEDIUM — excludes 3 combinations (gradnorm × {MMoE, DSelectK, PLE}) from the P2 grid by prediction, and 1 confirmed hang (gradnorm × CGC).

**Detected:** 2026-04-17 ~00:00 during P2 screen. `mtlnet_cgc + gradnorm` hung in `STAT=U` (uninterruptible disk/syscall wait) for 25 min with no epoch output; killed. Separately, `mtlnet + gradnorm` (base FiLM) completed but collapsed to 0.5% region Acc@10 (near random for a 1109-class problem).

**Status:** WORKAROUND (skip gradnorm × expert-gating combos). **Root cause not diagnosed.**

---

## TL;DR

GradNorm at 15-epoch budget on this task behaves in two failure modes:

1. **Collapse on base FiLM (`mtlnet + gradnorm`)**: training completes but region Acc@10 ends at 0.5% (below random-tier 0.9%). Likely gradnorm's learnable loss weights failed to stabilise in 15 epochs and pushed the shared backbone into a region-blind state.
2. **Deadlock on CGC (`mtlnet_cgc + gradnorm`)**: process hangs indefinitely, never emits first epoch's logs. Likely an incompatibility between gradnorm's `create_graph=True` backward (for auxiliary-loss gradient flow into `loss_scale`) and CGC's expert-routing gradient topology.

The gradnorm fix in `src/losses/gradnorm/loss.py` (commit `97f7fdb` on main) was already merged into our branch. Both failure modes occurred with the fixed version.

---

## Details

### Failure 1: Collapse on FiLM

Config: `--model mtlnet --mtl-loss gradnorm --folds 1 --epochs 15 --task-a-input-type checkin --task-b-input-type region`.

Result (from `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep15_2026041?_<ts>/summary/full_summary.json`):

| Metric | Value | Comment |
|--------|-------|---------|
| Cat F1 | **25.77%** | Far below grid median (~35%) |
| Reg Acc@1 | **0.03%** | Random-tier (~0.9% expected) |
| Reg Acc@10 | **0.51%** | Below random for 1109-class |
| Reg MRR | 0.97% | Below random |

Interpretation: the category head survived partially, but the region head collapsed to essentially random. GradNorm's auxiliary loss pushes the loss-weight `w` such that `(w_i * L_i).backward()` produces balanced gradient norms across tasks. If the region task's gradients are initially much larger (1109-class CE loss magnitude), gradnorm's first updates may rapidly shrink `w_region`, starving the region head of signal. At 15 epochs, there may be no recovery; with more epochs (e.g., 50), gradnorm's alpha schedule might re-grow `w_region`.

### Failure 2: Deadlock on CGC

Config: `--model mtlnet_cgc --mtl-loss gradnorm --folds 1 --epochs 15 --task-a-input-type checkin --task-b-input-type region`.

Symptoms:
- Process reached "Creating 2-fold CV" log line, then silent for 25 min.
- `ps -o pid,state,pcpu` showed `STAT=U` (uninterruptible sleep — waiting on kernel/disk or blocked IPC), `pcpu=6.5%`, `TIME` accumulated 8 min over 25 min elapsed → lots of wall-clock sitting on syscalls, little actual compute.
- No epoch output, no backward-pass start log.
- Killed via `kill -9 $pid`; shell loop moved to next config.

Most plausible root cause: gradnorm uses `create_graph=True` on the backward pass to allow the auxiliary gradient-norm loss to differentiate through the per-task gradient magnitudes into the learnable `loss_scale` parameter. CGC's expert-routing adds several non-standard autograd operations (gated expert combination, per-task routing heads). The combination may produce a backward graph with circular or unusually deep dependencies — triggering a silent deadlock in PyTorch's autograd or MPS kernel dispatch.

### Verification: gradnorm × MMoE also deadlocks (skip decision confirmed)

**Run 2026-04-17 ~05:56, bg `bksjb8gg9`** — `mtlnet_mmoe + gradnorm` at 1f × 20ep:

- Got past model construction and started training.
- Log shows `Epoch 1/20: 1%| | 1/80 [00:01<02:14, 1.70s/batch]` then silent.
- Process alive 17 min later, zero additional batch output. Killed.

The failure mode is **slightly different from CGC but ends the same way**: MMoE + gradnorm reaches the training loop and completes 1 batch before hanging; CGC + gradnorm hangs during setup. Both end in indefinite stalls with no recoverable progress.

This confirms the skip decision for MMoE. DSelectK + gradnorm and PLE + gradnorm were not independently verified (killed the shell loop after MMoE hung to avoid compounding stalled compute), but given two of four expert-gating archs deadlock identically, the pattern is strong evidence that gradnorm is broadly incompatible with expert-routing backbones in this codebase.

**Diagnostic hypothesis refined:** gradnorm's `create_graph=True` on the per-task backward pass creates a second-order autograd graph that interacts pathologically with gating-arch operations that involve non-standard index/gather operations in the expert routing. Not triaged into a minimal reproducer; reported here as a compatibility note.

**Side issue discovered:** the planned `timeout 300` / `gtimeout 300` fallback in the verification shell was ineffective because **neither `timeout` nor `gtimeout` is in `$PATH` on this macOS** (coreutils not installed). The shell had to be killed manually. If we retry any deadlock-prone config in future, we should either install `coreutils` (`brew install coreutils` gives `gtimeout`) or implement a Python-level timeout wrapper.

---

## Workaround

`scripts/` using the P2 grid loop should **not** pair `gradnorm` with expert-gating archs (`mtlnet_cgc`, `mtlnet_mmoe`, `mtlnet_dselectk`, `mtlnet_ple`) until root-caused. GradNorm × `mtlnet` (base FiLM) is allowed in principle but with the observed collapse at 15 ep should be retried at 50 ep to test whether it's a training-duration issue.

---

## Why this isn't a blocker for the paper

GradNorm is one of five optimizers in our grid. Nash-MTL, PCGrad, CAGrad, and equal-weight all work across all archs and cover the plausible champion space. If gradnorm never worked, we'd still have a defensible paper.

If gradnorm on FiLM at 50 ep works and beats the current champion, we'd want it back in the grid — but that's a nice-to-have, not a must-have.

---

## Action items

1. [ ] Verify with bounded timeout: `mtlnet_mmoe + gradnorm` at 20 ep, 5-min timeout. If completes, similar for DSelectK and PLE. Est. 15 min total.
2. [ ] Retest `mtlnet + gradnorm` at 50 ep (matched compute with STL); see if the collapse persists or was a 15-ep artefact.
3. [ ] If gradnorm+FiLM stabilises at 50 ep but gradnorm+gating still deadlocks: file a narrow bug report to the MTL optimizer maintainers with a minimum-reproducer. Otherwise document as a known-incompatibility in the paper's appendix.
