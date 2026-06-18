# MEMO — the cat↑/reg↓ "tradeoff" is roughly half confound, half config; the confound-free champion is reg-parity + a free cat lift

> **Status:** study-finding memo (NOT paper prose — §0 of `PAPER_DRAFT.md` is not being edited here).
> **From:** `mtl_frontier` audit (2026-06-17) + `mtl_improvement` closure (2026-06-12, §C25). **Adversarially
> fact-checked 2026-06-17** — an earlier draft *overstated* the attribution; this version is corrected.
> **For:** the BRACIS paper team — input for a recommended **revision of contribution C2**.
> **One sentence:** the paper's current C2 ("classic MTL tradeoff: small cat lift, **−7…−17 pp reg cost**")
> is **NOT a pure representational tradeoff** — on a matched config, flipping the class-weighting confound
> (C25) recovers ~**half** the reg gap, and the **confound-free champion (G)** — unweighted CE + v14 substrate
> + dual-tower — reaches **reg-parity at the STL ceiling (−0.09…−0.31 pp) with a +2.6…+4.1 pp cat lift**.

---

## 1. Why this memo exists

A user asked: *"why does cat improve while reg degrades — it seems we are losing something."* A 4-lens,
adversarially-verified audit (`docs/studies/mtl_frontier/FINDINGS.md §CAT↑/REG↓`) shows **we are not losing
recoverable reg** — and the intuition most likely tracks the paper's **current C2**, which reports the
**pre-C25 (class-weighted)** reg numbers. This memo gives the corrected picture, reconciles it with C2, and
recommends a reframing. It does not touch locked prose.

## 2. The two numbers — and the honest decomposition of the gap

| | reg vs STL ceiling | cat vs STL ceiling | config |
|---|---|---|---|
| **Paper C2 (current draft, v11 canon)** | **−7.34 (FL) / −9.50 (CA) / −16.59 (TX)** | +1.40 / +1.68 / +1.89 | GCN substrate, B9 recipe, **class-WEIGHTED** MTL reg CE |
| **Confound-free champion G** | **−0.09 AL / −0.12 AZ / −0.09 GE / −0.31 FL — *matches*** | **+2.56 / +4.08 / +3.93 / +3.20** | v14 substrate, dual-tower, **unweighted** CE |

**The gap is roughly half confound, half config — not a pure tradeoff, but also not "just the confound."**
The decisive evidence is the one clean controlled A/B in the repo
(`scripts/mtl_improvement/c25_fl_b9_{weighted,continuity}.sh` — byte-identical except the class-weight
flags, holding B9 recipe + GCN substrate + seeds {0,1,7,100} fixed):

- **WEIGHTED** MTL reg = **63.91**;  **UNWEIGHTED** MTL reg = **67.06** → flipping *only* class-weighting
  recovers **+3.15 pp reg** (+3.52 cat). Against the FL/GCN STL ceiling (70.62), Δ_reg moves
  **−6.71 → −3.56 — the gap is HALVED, not closed** (the study's own log: *"reg gap ~halved"*,
  `mtl_improvement/log.md:1676`).
- The **remaining ~−3.5 pp closes only with the champion-G stack** (v14 substrate + dual-tower + prior-OFF +
  `aux` fusion): the unweighted cross-attn baseline still trails ~−1.8 pp at FL (`PAPER_UPDATE.md:103-105`),
  and the full champion G reaches **−0.31** at FL on the v14 STL ceiling (73.27).

So: **C25 (class-weighting) ≈ half the paper's reg cost; the v14-substrate + dual-tower change ≈ the other
half.** The honest claim is **"the −7…−17 pp reg cost is not irreducible/representational — ~half is a
class-weighting confound and the rest closes under the confound-free champion configuration, which reaches
matched-metric reg-parity."** Neither "irreducible architectural cost" (the old framing, falsified) nor
"it's just the confound" (an overstatement) is correct.

> **Retracted from the earlier draft (do not use):** the claim that class-weighting alone depresses MTL reg
> "~10–14 pp, FL −14, exactly matching the state-scaling." The clean isolated class-weighting effect at FL is
> **+3.15 pp**, not 14; the "10–14 pp" figure conflated class-weighting with a buggy-default plateau and an
> fp16-no-GradScaler harness artifact (`log.md:689`). The clean per-state class-weighting deltas were **only
> measured at FL** (B9-canon A/B); CA/TX were never isolated (`CONCERNS §C25:594`).

C25 status: DISCOVERED + CLOSED 2026-06-05; `CONCERNS §C25` records *"only paper-doc restatement remains."*
The full §0 re-baseline on the confound-free champion is the scaffolded `closing_data` study.

## 3. The mechanism (what the mtl_frontier audit adds) — the β-gated regime split

Why reg is immovable and cat is movable (every number reproduced from rundir diagnostics; **4/4 lenses
adversarially verified — no bug, no reg-shortchange**):

- **reg is already at its single-task ceiling** (champion G reg = STL reg within −0.09…−0.31 pp) — *maxed*,
  not shortchanged, so no recoverable headroom to "lose."
- **The reg head's only shared-trunk channel** is `feat = priv + β·aux_proj(shared)`, and the trained **β is
  gated by state scale**: at FL **β→≈0 in every arm** (nothing on the shared trunk for cat gains to rob); at
  small states β is live (+0.078 GE … +0.12 AL) and levers move **both heads together** (R1 AL cat +0.20 /
  reg +0.21). cat harvests the shared cross-attn trunk (easy 7-class); reg at scale is a saturated, insulated
  private STAN tower. cos(∇cat,∇reg) ≈ 0.
- The **apparent cat↑/reg↓ at the margins** (R5/R10) is **selective perception + a diagnostic-best
  epoch-decoupling artifact**: across 41 lever-deltas the largest quadrant is cat+/**reg+**, the cat–reg
  correlation is *positive*, and at the single deployable checkpoint the one "reg-down" lever (R5) collapses
  to reg-**flat** (cat & reg peak ~26 epochs apart).

This is the post-2022 literature-expected regime (Kurin/Xin NeurIPS'22; Mueller TMLR'25): parity on the hard
task + a free lift on the easy one; reg cannot be exceeded (no headroom, no transfer channel at scale).

## 4. Recommended C2 reframing (for the paper team to evaluate)

**From** (current): *"Classic MTL tradeoff — joint training adds a small cat lift and pays a sign-consistent
−7…−17 pp reg cost."*

**To** (corrected): *"On the confound-free configuration (unweighted CE, the `train.py --task mtl` default),
multi-task joint training is a **Pareto gain on the easy task at parity on the hard one**: it beats the
single-task **category** ceiling by **+2.6…+4.1 pp** while **matching** the single-task **region** ceiling
(within 0.3 pp) at four states. The large region 'cost' reported in pre-2026-06 measurements was **not
representational**: about half was a class-weighting confound (the MTL region head trained on class-balanced
CE against an unweighted metric and STL ceiling), and the remainder closes under the confound-free champion
(v14 substrate + dual-tower). Mechanistically, the region head reaches its single-task ceiling through a
private tower that, at scale, is insulated from the shared trunk (learned β→0), so joint training moves the
shared-trunk-harvesting category task without disturbing region."*

Stronger and more honest than "textbook tradeoff."

## 5. Caveats the paper team MUST weigh (strengthened after the fact-check)

1. **The parity is a confound-fix + champion-G-config result, NOT a flag flip.** Unweighting alone, on the
   paper's v11/GCN/B9 config, recovers only ~half the FL reg gap (−6.71 → −3.56). **Claiming parity by
   unweighting the existing v11 config would be wrong by ~3.5 pp at FL.** Parity requires re-baselining §0 on
   the confound-free champion (the `closing_data` plan). Report *those* numbers; retire the v11 reg-cost.
2. **State coverage.** Parity is established at **AL/AZ/GE/FL** (4×4 seeds). The paper headline is FL/CA/TX;
   **CA/TX have no v14 substrate yet** — FL parity is solid, CA/TX parity is *expected* (same mechanism) but
   **unmeasured.** State this honestly.
3. **The cat lift is partly the C25 cat-fix and partly head-config, not a pure MTL effect.** The cat
   ceiling-beating lift is *produced by* cat-unweighting (AL: weighted cat 48.5 < STL 50.0; unweighted 53.5
   = +3.5), and the G-vs-STL cat comparison spans head-config differences (STL ceiling: 2-layer GRU,
   dropout 0.3, logit-adjust τ=0.5; G: 4-layer, dropout 0.1, plain CE — "architecture-dominated",
   `PAPER_UPDATE.md:27-28`). So the +2.6…+4.1 is a real, A/B-validated *deployable* gain, but it is **not a
   pure "MTL beats STL" effect** — disclose the recipe/head-config asymmetry.
4. **Parity, not a reg win.** Claim reg-**parity** (within noise), never "MTL beats STL on reg."

## 6. Sources (all reproduced during the audit + fact-check)

- `docs/studies/mtl_frontier/FINDINGS.md §CAT↑/REG↓` + §AUDIT (the β-gated mechanism; 18-agent, 4-lens verified).
- `scripts/mtl_improvement/c25_fl_b9_{weighted,continuity}.sh` + `mtl_improvement/log.md:1676` — the
  controlled class-weighting A/B (+3.15 reg, gap "~halved").
- `docs/studies/archive/mtl_improvement/PAPER_UPDATE.md:27-28,103-105` — the −1.8 residual after
  unweighting + the cat head-config asymmetry.
- `docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md` + `docs/results/mtl_improvement/R0_matched_metric_bar.json`
  (the verified reg-parity + cat-lift table; champion G = canon v16).
- `docs/CONCERNS.md §C25` (the confound; "only paper-doc restatement remains"; per-state A/B only at FL).
- `docs/results/RESULTS_TABLE.md §0.1` (the v11 paper-canon reg-cost being superseded).
- `docs/studies/closing_data/` (the scaffolded full §0 re-run on the confound-free champion).
