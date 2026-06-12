# Tier 4 (loss / optimization) — audit + final verdict (2026-06-08)

**Verdict: CLOSED — fairness-checked convergent NEGATIVE.** No loss-scale normalization, no
balancer (the full `src/losses` registry), and no alternative static weight Pareto-beats champion
G's `static_weight cw=0.75`. The loss/optimization axis is exhausted. Six convergent lines of
evidence (below). Champion G unchanged.

> ⚠ **Evidence-strength precision (2026-06-12 re-audit; supersedes earlier "per-method-tuned +
> arch-wired" phrasing).** What actually ran: the full screen at **registry DEFAULTS, seed 0,
> AL+FL**; only **GradNorm** genuinely retuned (lr=0.05, α=1.5); the "retuned" Nash arm used
> `max_norm=2.2`, which **is** the registry default — a config-identity re-run, not a new tuning
> point; DWA/FairGrad diagnosed misconfigured-at-defaults and **not** retuned; the three
> gradient-surgery methods (CAGrad/PCGrad/Aligned-MTL) **never validly tested individually** under
> G — as wired they collapse to ≈`equal_weight`, which *was* screened and lost. The negative is
> sound (equal-weight collapse + tuned-static fairness sweep + RLW litmus + cos≈0 mechanism + the
> k=2 literature), but it is a **convergent-evidence negative, not an exhaustive per-method tuning
> study** — word it that way in the paper.

## How we got here (the suspicious result that triggered a deep audit)
The first T4.1 screen showed *every* advanced balancer clustered at the `equal_weight` point and none
beat static_weight. That clustering was suspicious — possibly a bug. A meticulous audit (2 sub-agents +
direct probes) was launched. It found the clustering was **partly a real artifact** (so the first screen
was *partially invalid*) — but the underlying conclusion survives for a deeper, now-proven reason.

## 1. The audit findings (code + live probe)
- **Gradient-surgery family is mis-wired under G's dual-tower (cagrad / pcgrad / aligned_mtl).** G's reg
  signal lives in the *private* STAN tower (`reg_specific_parameters()`, OUTSIDE `shared_parameters()`).
  These methods apply their surgical/combined gradient **only to shared params**; the private reg tower
  (>80% of the reg pathway) trains at **unit weight always**. Live probe: cagrad logs weights `[1,0]`
  yet its private-tower gradient is identical to `equal_weight`'s. → these 3 cells **don't count** as
  balancer tests; they reduce to ≈equal-weighting by construction.
- **Misconfigured at defaults (don't count as fair tests):** gradnorm (lr=1e-3 too small + L1-renorm to
  sum=2 structurally centers it on equal → weight range 0.016, can't reach G's 0.25/0.75); dwa (per-batch
  loss-ratio history → pinned ≈1.0); fairgrad (step_size too small → pinned 0.5).
- **Latent preflight bug (FIXED):** `scripts/train.py` `_BACKWARD_ONLY_LOSSES` omitted cagrad+aligned_mtl
  (would `TypeError` under grad_accum>1; safe here only because `default_mtl` pins grad_accum=1). Fixed.
- **Correctly wired + adapting (valid tests):** nash_mtl, uncertainty_weighting, uw_so, db_mtl,
  bayesagg_mtl, excess_mtl, stch, go4align, famo, scheduled_static, equal_weight, static_weight,
  random_weight. These genuinely reweight the full graph incl. the private tower.
  ⚠ *Exception (2026-06-12):* bayesagg_mtl (AL cat **37.75**) and excess_mtl (AL cat **45.97**)
  crater far below G (52.75) with no diagnosis — treat those two cells as
  **misconfigured-at-defaults, undiagnosed**, not as valid evidence about the methods themselves.
  (Doesn't change the verdict: a method needing a per-state rescue tune to merely reach the
  baseline is not a promotion candidate.)

## 2. The decisive mechanistic finding — task gradients are ORTHOGONAL
`cos(∇L_cat, ∇L_reg)` on the shared trunk ≈ **0** over all 50 epochs — pooled mean **+0.0008** over the
**16 champion-G runs (4 states × 4 seeds), n=3,797 epoch-fold points** (FL +0.0007 / AL +0.0032 / AZ
−0.0005 / GE −0.0004; ~50% negative — zero-mean noise). Figure:
`docs/studies/mtl_improvement/figs/grad_cosine_tasks.png` (widened 2026-06-12 from 2 runs → 16 — H1).
**There is no gradient conflict for any
balancer to resolve.** Gradient-surgery / dynamic-weighting methods only help under *strongly negative*
cosine (high interference); at cos≈0 they cannot help — even correctly wired. This is *why* fixing the
dual-tower wiring is moot for the verdict.

## 3. Literature (corroborates: this is the EXPECTED k=2 result, not a bug)
Tuned scalarization matching/beating advanced MTO at **k=2 with a tuned baseline + standard
regularization** is the central, replicated finding of: Kurin et al. NeurIPS'22 ("In Defense of the
Unitary Scalarization" — balancers are "partly regularization"); Xin et al. NeurIPS'22 ("Do current MTO
methods even help?" — at k=2 none beat a scalarization sweep; **LR/HP variance is 6–7× the method
effect**); Royer et al. NeurIPS'23 (uniform scalarization on-par; value is *search efficiency*); 2025
"Uniform Loss vs Specialized Optimization" (SMTOs only win under high task interference / negative
cosine). The only condition that would make us suspect a bug — an *under-tuned* static baseline — we
ruled out with the cw-sweep below.

## 4. Corrected re-runs (bug-aware retune + Xin'22 fairness sweep), FL+AL seed0
| arm | FL reg | FL cat | FL ΔvsG | AL reg | AL cat | verdict |
|---|---|---|---|---|---|---|
| **G static_weight 0.75** | 72.95 | 73.12 | — | 62.64 | 52.75 | champion |
| gradnorm @ lr=0.05, α=1.5 | 73.07 | 71.83 | +0.12 reg / **−1.29 cat** | 62.52 | 53.40 | trades cat (no) |
| nash_mtl @ max_norm=2.2 | 73.04 | 71.61 | +0.09 reg / **−1.51 cat** | 62.77 | 54.25 | trades cat; AL cat +1.50 **reverses** at FL → state-dependent (no) |
| static cw=0.50 | 72.98 | 72.19 | +0.03 / −0.93 | 62.77 | 53.08 | on the trade curve |
| static cw=0.60 | 72.94 | 72.44 | −0.01 / −0.68 | 62.47 | 52.48 | on the trade curve |
| static cw=0.66 | 72.90 | 72.41 | −0.05 / −0.71 | 62.32 | 52.96 | on the trade curve |
| static cw=0.80 | 72.82 | 73.28 | −0.13 / +0.16 | 62.79 | 52.95 | on the trade curve |

- **The static cw-sweep is a monotone reg↔cat trade; no cw Pareto-beats 0.75** → G's weight is on the
  Pareto front (Xin'22 fairness condition satisfied — the baseline is tuned as hard as the challengers).
- **Retuned gradnorm/nash still just trade ~1.3–1.5 pp cat for ~0.1 pp reg** at FL — exactly the cos≈0
  prediction (no free lunch). nash's AL cat gain does not generalize (FL cat −1.51).

## 5. T4.0a loss-scale normalization — FALSIFIED
Dividing each CE by log(num_classes) before the weight. At G's default cw it **craters FL reg (35.47)**;
with a reg-favorable cw-grid it never recovers (FL: cw0.10 70.25 / cw0.20 68.52 / cw0.35 63.51 — all
≫ below G 72.95). The log-normalization **starves the high-cardinality reg head**: the large reg CE
reflects a genuinely harder task that needs the gradient, not an unfair dominance. G's tuned static
weight already encodes the correct scale balance (consistent with the literature: post-tuning,
explicit normalization is redundant/harmful). AL: a pure weight-reparam, no gain.

## 6. T4.0b RLW litmus (done earlier)
Random per-step weighting ≈ G on reg (±0.33), trades on cat → the inter-task weight is not a sensitive
lever (Lin TMLR'22). First of the six convergent signals.

## Verdict
**Tier 4 CLOSED — the loss/optimization axis yields no Pareto gain over G's `static_weight cw=0.75`.**
Six convergent lines: (1) RLW litmus, (2) full registry screen (defaults), (3) targeted retune
re-run (GradNorm; Nash = default-config identity — see the precision banner), (4) static cw-sweep
(fairness), (5) scale-norm falsified, (6) gradient cosine ≈ 0 (mechanism) + literature. The negative
is paper-grade ("no balancer beats tuned scalarization at k=2", Kurin'22/Xin'22) as a
convergent-evidence negative (see banner for exact wording). **Mechanistic payoff:** the orthogonal task gradients unify the whole study — they
explain why balancers can't help AND why forcing more sharing failed in Tier 2 (it would induce the
conflict that isn't there) AND why the dual-tower wins (protect reg, let cat harvest the shared
representation). No multi-seed promotion warranted.

## Figures
- `figs/grad_cosine_tasks.png` — task-gradient orthogonality (the mechanism).
- `figs/t4_balancer_scatter_FL.png` — every balancer clusters near equal-weighting; none Pareto-beats G.
- `figs/t4_loss_weight_trajectories_FL.png` — engagement audit (static fixed; nash/UW adapt; gradnorm/dwa
  frozen; cagrad collapses).

## Caveats / scope
Single-seed (seed0) screen + corrected re-run, FL+AL (the cosine≈0 + literature + monotone cw-trade make
multi-seed unnecessary — there is no candidate to promote). Gradient-surgery wiring under the dual-tower
is a documented limitation (moot here by cos≈0); if a future *conflicting* task pair is studied, the
surgery methods' task-specific-param handling must be fixed first (see the audit). Data:
`T4_full_screen.json`, **`T4_corrected_rerun.json`** (the §4 corrected re-run + §5 scale-norm cw-grid,
aggregated 2026-06-12 under HANDOFF_AUDIT H2 — previously markdown-only), `T40_rlw_litmus.md`; manifests
`t4_full_manifest.tsv`, `t40a_wgrid_manifest.tsv`, `t4_corrected_manifest.tsv`.
