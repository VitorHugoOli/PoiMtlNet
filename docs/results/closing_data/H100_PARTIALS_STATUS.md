# H100 — in-flight partials (session-death-resilient capture)

Auto-committed partials from the H100 while runs are in flight (session may die; this
branch captures whatever has landed). Final consolidation follows once runs complete.

## In flight (as of branch creation)
- **CTLE-SC @ FL** (sequential re-run after the parallel attempt OOM'd 2 folds + had a cat-rundir race):
  per-fold reg reports land in `docs/results/P1/region_head_florida_checkin_1f_50ep_blcmp_check2hgi_ctle_f*.json`;
  aggregate `baseline_compare/florida_ctle.json` written at the end. Fold-0 provisional: cat 27.98 / reg 73.00.
- **CA/TX ReHDM faithful (1 seed each, parallel)** — ETLs built (CA 6149 regions/377k train traj; TX 4900/514k).
  ~8–12 h/seed (CPU-collaborator-mining-bound); 1-seed provisional (vs published 5-seed AL/AZ/FL 66.06/54.65/65.68).
  Results in `results/baselines/rehdm/{state}/` (gitignored rundirs) → summary committed when each completes.

Full context: `H100_FL_BASELINES_FINDINGS.md` (merged in PR #45).
