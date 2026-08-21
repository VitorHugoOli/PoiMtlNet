# The consolidated answer and the mechanism — the settled content

> The dissertation's answer to its question, and WHY the joint model wins, with every claim's
> evidence chain. Rendered in: Ch.6 §6.2 (`drafts/6_conclusion.tex`).
> Experiment record: `audit/capacity_baseline_experiment.md` (design + licensing contract +
> results). Number ledger: `drafts/6_citations.md`.

## The answer (conditional — the condition IS the finding)

- Place-level embedding + naive hard sharing → **no** (Ch.3, for that configuration).
- Check-in-level representation + sharing topology built for it → **yes** (Ch.5): category
  outperforms everywhere; region outperforms at 4/6, TOST non-inferior (±2 pp) at AL/AZ.
- What the answer depends on: the representation, together with the sharing topology built
  on it (two factors, always).

## The mechanism chain (three links, each with its evidence and license)

1. **Not task-teaching**: freeze control (MobiWac §6, IN-PAPER, citable from Ch.5) — region
   pathway frozen, category gain survives, at the three datasets where the control ran
   (AL/AZ/FL). "A stronger shared trunk, not the region task teaching the category one."
2. **Not parameter count**: capacity-matched dedicated baseline (POST-SUBMISSION frame
   analysis — never a Ch.5 result; the prose must say when it was run).
   - Alabama, FINAL (n=20, 3 recipes): wide dedicated (h=672, ~4.2M params = joint budget)
     best arm 56.16 ±1.88 vs narrow dedicated optimum 56.82 ±0.03 vs joint 64.54.
   - California, PARTIAL at draft time (n=15, first arm): 68.35 ±0.53 vs ceiling 70.60 vs
     joint 77.05 — same direction. REPLACE with final verdict when job 4cff4b00 completes.
   - Param audit reproduces the paper quote: AL 4,197,621 vs 1,061,476 combined (3.95×);
     CA 5,151,189 vs 2,015,044 (2.56×).
3. **What remains — the shared trunk**: cross-attention stack trained on both tasks' signals
   builds a representation the dedicated model cannot reach at any width tried; width without
   the second task's signal has no new information to spend parameters on.

## The gradient picture (N3 beat — the FULL scope travels with the number, verbatim)

Cosine similarity between the two tasks' gradients averaged **+0.001**, over **four seeds on
three of the six datasets**, measured **during development on an earlier preparation of the
data**, **directional conflict only**, **a finding for this pair of tasks, not a general rule**.
Reading: "sharing stopped hurting" — NEVER "the tasks teach each other". Explains why gradient
balancers had little to correct in this configuration.

## Banned vocabulary in this section

"knowledge gate" (author shorthand — translates to the paper's sharing-by-exchange wording);
any parameter-count credit for the win (it is disclosed as COST); "lower cost"; upgrades of
AL/AZ region results.