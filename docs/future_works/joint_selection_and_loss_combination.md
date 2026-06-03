# Future Work — Joint Selection-Metric Aggregation & MTL Loss Combination

**Status:** research memo (2026-06-03). **No code change is proposed by Part 2**; Part 1's
core change (geom_simple selector) already landed as the default — this memo records the
literature backing and the *remaining* optional upgrades. Two literature-grounded critical
analyses, commissioned to validate that (a) we aggregate the two task metrics correctly for
checkpoint selection, and (b) we combine the two task losses correctly in MTL training.

**Context:** joint cross-attention Check2HGI MTL (NORTH_STAR B9 / `static_weight
--category-weight 0.75`). Task A = next_category (7-class, imbalanced, macro-F1). Task B =
next_region (~1k–9k-class, sparse, CE + α·log_T prior, Acc@10). The binding empirical fact
is our **regime finding**: under joint cross-attention training the hard high-cardinality
region task is sacrificed for the easy category task. Related memos:
[`substrate_adaptive_mtl_balancing.md`](substrate_adaptive_mtl_balancing.md),
[`mtl_architecture_revisit.md`](mtl_architecture_revisit.md),
[`reg_head_architecture_sweep.md`](reg_head_architecture_sweep.md). See also
[`../CONCERNS.md` §C21](../CONCERNS.md) (the selector fix) and the 2026-06-03 CHANGELOG entry.

---

## Part 1 — Aggregating the two metrics for checkpoint SELECTION

**Current (as of 2026-06-03, the code default):**
`joint_geom_simple = sqrt(cat_macroF1 · reg_Acc@10)` — geometric mean of each head's reported
headline metric (cat `f1`, reg `top10_acc_indist`), no normalization.

### Verdict: the geometric mean is sound and literature-backed — with two caveats.
GM has exactly the properties we want for a *collapse-averse* single-checkpoint selector:
- **Zero-annihilation / collapse penalty.** For fixed arithmetic mean, GM is maximized only
  when the two metrics are equal, and → 0 if either collapses. This is the explicit rationale
  of the Derringer–Suich desirability index (1980) and the G-mean in imbalanced classification
  (Kubat & Matwin 1997). We want a checkpoint good at *both* tasks → correct bias.
- **Scale-rank invariance.** `log GM = ½(log x + log y)` → GM ranks on *proportional* gains and
  is invariant to per-metric multiplicative rescaling of the ranking. This is precisely what
  fixed our old scale-incoherent arithmetic mean (region macro-F1 ≈0.1 swamped by cat-F1 ≈0.7).
- **Pedigree as a heterogeneous-metric combiner.** Fowlkes–Mallows index (= GM of P/R), the
  Geometric Loss Strategy in MTL (Chennupati et al., MultiNet++ 2019), and the desirability
  literature all use GM to combine heterogeneous quality signals.

**Caveat 1 — commensurability (the real weak point).** macro-F1 and Acc@10 are *different
metric families*. GM is invariant to *scaling* but does **not level-normalize**: it treats raw
0.60 Acc@10 and 0.67 F1 as comparable "fractions of perfect." That is defensible here **only
because** the two raw values happen to sit at similar levels — luck, not principle. If Task B
ever switches to top-1 (≈0.25–0.48) the aggregator silently mis-weights.

**Caveat 2 — noise near the floor.** `∂ log GM/∂x = 1/(2x)` → GM over-weights the smaller/noisier
region metric near its low end. Good for collapse-aversion, but it makes the raw argmax epoch
jumpy. Mitigated by the existing `--min-best-epoch` floor + smoothing (below).

### Optional upgrades (not yet implemented — future work)
1. **Restrict the GM argmax to the Pareto-non-dominated epoch set** and select on a *smoothed*
   (e.g. 3-epoch moving-average) trajectory, keeping the min-epoch floor. ~10 lines, no
   retraining. Addresses validation-argmax instability (Cawley & Talbot 2010; "Don't Stop Me
   Now" arXiv:2602.22107; stability-aware checkpoint selection arXiv:2605.18852).
2. **Feed GM lift-over-baseline instead of raw metrics:** `GM(lift_cat, lift_reg)`,
   `lift = (m − chance)/(1 − chance)` — the desirability-transform answer to the commensurability
   challenge, and our own earlier intermediate idea done right (use Acc@10 lift, **not** the
   noisy top-1 majority lift). Adopt if a reviewer challenges mixing F1 with Acc@10. Use a
   *random-stratified macro-F1* baseline for cat (not majority accuracy) and a top-10
   random/popularity baseline for reg.
3. **Hygiene (always):** report per-task `(macro-F1, Acc@10)` as primary; the scalar is a
   *selection rule only*. Selecting and reporting on the same validation data is mildly
   optimistic (Cawley & Talbot 2010); confirm on held-out test where possible.

**Do NOT** revert to arithmetic mean (scale-incoherent — the C21 bug) or escalate to
Tchebycheff/hypervolume (over-engineered for two bounded metrics + one checkpoint).

---

## Part 2 — Combining the two task LOSSES in MTL (no code change now)

**Current:** static linear scalarization `L = (1−w)·L_reg + w·L_cat`, `w = 0.75`.

### Verdict: `w=0.75` is a legitimate baseline — but it doesn't address our actual failure mode.
The literature has converged (somewhat against the 2018–21 grain) that a *tuned* linear
scalarization is a genuinely strong baseline: Kurin et al. (NeurIPS 2022) — unitary
scalarization matches/beats specialized MTL optimizers; Xin et al. / Google (NeurIPS 2022) —
gradient methods give no gain over tuned scalarization at equal compute; Lin et al. (TMLR 2022)
— even random weighting is competitive. So `w=0.75` is not an embarrassment.

**But two weaknesses are specific to us:**
1. It is a single, possibly Pareto-incomplete point (Hu et al. NeurIPS 2023) that likely drifts
   across our five states (region cardinality ~1k→9k; we already see B9-vs-H3-alt recipe drift).
2. It over-weights the **easy** task (category, 0.75), **entrenching** the region sacrifice we
   observe — `w` is a blunt compensator for a deeper scale mismatch (next section).

### The likely *real* lever: loss-scale normalization (cheap, high-value)
Two distinct problems are conflated under "weighting"; the dominant one for us is **scale**, not
the inter-task weight:

- **Inter-task scale mismatch.** Random-init 7-class CE ≈ ln 7 ≈ 1.95; ~9000-class CE ≈ ln 9000
  ≈ 9.1 — a **~4.7× built-in magnitude gap before any weighting**. The region loss dominates the
  gradient by construction, and `w=0.75` on category is partly just undoing that. Cheapest
  principled fix: **normalize each CE by `log(num_classes)`** (or a running-EMA baseline) *before*
  applying `w` — the principle behind DB-MTL's log-transform (Lin 2023) and Uncertainty Weighting
  (Kendall 2018). **Highest expected value, near-free to test.**
- **Intra-task class imbalance (orthogonal — fix inside the heads, not via `w`).** For the
  category head (macro-F1 metric): **logit adjustment** (Menon et al. ICLR 2021, provably
  consistent for balanced error) or class-balanced loss (Cui et al. 2019). For region: leave the
  α·log_T prior largely intact — don't double-count frequency information; tune any imbalance
  correction jointly with α.

### Negative transfer / "easy task wins"
Our regime finding is textbook task-dominance (Vandenhende et al. survey, TPAMI 2021). The
**balanced-loss-decrease** family targets exactly "a task is under-optimized": **FAMO** (Liu
NeurIPS 2023) is the standout — it matches CAGrad/Nash-level balance at **O(1)** cost (no
per-task gradient storage), which is the only thing feasible with our ~9k-class region head;
PCGrad/CAGrad/NashMTL are O(k)-gradient and a poor fit at that cardinality. **Warning:**
Uncertainty Weighting (down-weights hard/noisy tasks) and MGDA (collapses to smallest-gradient
task) can *worsen* region neglect — do not deploy blind.

**Architectural flag (heavier, beyond loss-combine scope).** Since substrate gains wash out under
the *shared cross-attention* backbone, the negative transfer may be partly representational; a
task-routed / partially-shared region pathway is the deeper fix — see
[`mtl_architecture_revisit.md`](mtl_architecture_revisit.md) and
[`part2_mtl_dual_substrate_routing.md`](part2_mtl_dual_substrate_routing.md).

### Ranked future-work order (what to TEST; we beat *tuned, scale-normalized* scalarization first)
- **Tier 0 (cheap diagnostics):** (1) divide each CE by `log(num_classes)` (or EMA baseline) then
  re-tune `w` — highest EV; (2) **RLW litmus test** (Lin 2022) — if random weighting ≈ `w=0.75`,
  the inter-task weight is *not* the bottleneck → spend effort on scale + imbalance.
- **Tier 1 (cheap principled):** (3) Uncertainty Weighting (Kendall 2018, monitor σ so it doesn't
  shrink region); (4) **FAMO** (Liu 2023, O(1), directly targets under-optimized tasks); (5)
  DB-MTL (Lin 2023, log-scale + grad balance in one recipe).
- **Tier 2 (heavier, only if 0–1 stall):** (6) CAGrad; (7) Aligned-MTL (stability + keeps explicit
  target `w`).
- **Orthogonal (head-level):** (8) logit adjustment on the category head (consistent for macro-F1).
- **Skip / deprioritize:** NashMTL (cvxpy/ECOS instability we've already hit; overkill at k=2),
  MGDA (collapses to smallest-gradient task → risks worse region neglect), Pareto-MTL (analysis only).

Most of our `src/losses/` registry (NashMTL, GradNorm, PCGrad, CAGrad, uw_so, db_mtl, go4align,
excess_mtl, stch, scheduled_static) already exists — a study can A/B these against
scale-normalized tuned scalarization with no new infra. This overlaps with and should be merged
into the plan in [`substrate_adaptive_mtl_balancing.md`](substrate_adaptive_mtl_balancing.md).

---

## References (both parts)
**Aggregation:** Derringer & Suich (1980, desirability/GM); Kubat & Matwin (ICML 1997, G-mean);
Fowlkes & Mallows (1983); Chennupati et al. (CVPRW 2019, GLS); Lin et al. (arXiv:2402.19078,
Smooth Tchebycheff); Cawley & Talbot (JMLR 2010, selection bias); "Don't Stop Me Now"
(arXiv:2602.22107); robust checkpoint selection (arXiv:2605.18852); Royer et al.
(arXiv:2310.08910, scalarization reaches the Pareto front).
**Loss combination:** Kurin et al. (NeurIPS 2022, 2201.04122); Xin et al. (NeurIPS 2022,
2209.11379); Hu et al. (NeurIPS 2023, 2308.13985); Lin et al. RLW (TMLR 2022, 2111.10603);
Kendall et al. (CVPR 2018, 1705.07115); Chen et al. GradNorm (ICML 2018, 1711.02257); Liu et al.
DWA (CVPR 2019); Yu et al. PCGrad (NeurIPS 2020, 2001.06782); Liu et al. CAGrad (NeurIPS 2021,
2110.14048); Navon et al. NashMTL (ICML 2022, 2202.01017); Liu et al. FAMO (NeurIPS 2023,
2306.03792); Lin et al. DB-MTL (2023, 2308.12029); Senushkin et al. Aligned-MTL (CVPR 2023,
2305.19000); Sener & Koltun MGDA (NeurIPS 2018, 1810.04650); Lin et al. Pareto-MTL (NeurIPS 2019,
1912.12854); Vandenhende et al. survey (TPAMI 2021, 2004.13379); Menon et al. logit adjustment
(ICLR 2021, 2007.07314); Cui et al. class-balanced loss (CVPR 2019); Lin et al. focal (ICCV 2017,
1708.02002).
