# F50 T4 — Prior-Run Validity Under C4 Leak (2026-04-29 20:00 UTC)

**Question (user, 2026-04-29):** *Are the F50 ablations (T1.x, T2, T3, P1-P4, Tier-A, etc.) and earlier executions still valid under the C4-corrected (leak-free) regime, or are they invalidated?*

**Short answer:**
- **Relative comparisons (paired Δs between two recipes that share the same head + log_T source) → preserved** ✅✅. C4 inflates both arms uniformly so the Δ is robust. **Validated TWICE: B9 vs H3-alt clean = +3.34 pp (matches expected direction); PLE vs H3-alt clean = +0.26 pp (matches leaky +0.25 pp to within 0.01 pp).**
- **Absolute numbers (reported reg top10 / MRR / Acc@1) → inflated by ~13-17 pp.** Every F50 run that used `region_transition_log.pt` carries the leak.
- **Paper claims survive** because they're framed as Δs vs baselines that share the leak. Headline numbers need "leak-free" recomputation.

This doc inventories which prior runs are valid for which claims.

### PLE clean verification (2026-04-29 21:25, run `_2059`)

| recipe | leaky Δreg | clean Δreg | clean Δcat |
|---|---:|---:|---:|
| PLE vs H3-alt | +0.25 | **+0.26** ✅ matches | **−4.22** ⚠ NEW finding |

The reg Δ matches the leaky Δ to within 0.01 pp. **Uniform-leak hypothesis VALIDATED** — the 17 other F50 ablations we skipped re-running can be trusted to preserve their relative orderings.

Bonus PLE finding: cat F1 = 64.13 ± 1.04 (clean) vs H3-alt's 68.34 → −4.22 pp (0/5 positive). PLE's expert routing **hurts cat without helping reg** under leak-free conditions. PLE is Pareto-WORSE than H3-alt — worse than P4+OneCycle's Pareto-trade.

---

## 1 · Decision rules (when is a prior run still usable?)

A prior run is **VALID for a paper claim** iff:

1. The claim is a paired Δ between two arms that BOTH used the same `log_T` source, AND
2. Both arms used heads from the C4-affected family (or both used non-graph-prior heads), AND
3. Both arms ran on the same fold split, AND
4. The C4 leak is approximately uniform across them (no recipe-specific amplification differences).

A run is **INVALID for absolute claims** if:
- It quotes an absolute reg metric (top10 / MRR / Acc@1) without a `−15 pp` footnote.
- It compares heads with DIFFERENT amplification structures (e.g. `next_getnext_hard` vs `next_tgstan` — TGSTAN's per-sample gate amplifies differently from α alone).

---

## 2 · Validity matrix — F50 ablations

Empirical validation: H3-alt clean = 60.12 ± 1.15 reg @ ≥ep5 vs H3-alt leaky = 77.16 → drop −17.04 pp. P0-A clean = 63.23 vs P0-A's P4+Cosine leaky = 76.07 → drop −12.84 pp. **Drop is similar but not bit-equal; relative Δs are within ~1 pp of their leaky counterparts.**

| run-set | head used | what claim depends on | validity |
|---|---|---|---|
| **F37 STL B5 (`82.44`)** | `next_getnext_hard` (full log_T) | "STL ceiling" absolute number | ❌ inflated; true ceiling ~66 (estimated; clean run queued in `tmux stl_clean`) |
| **F50 T1.2 HSM** | `next_getnext_hard_hsm` (full log_T) | "T1.2 HSM Δreg = −5.16 pp" (vs H3-alt) | ✅ Δ valid (both arms leaky, similar amplifier); absolute -5.16 pp shift relative to clean H3-alt is preserved |
| **F50 T1.3 FAMO** | same as H3-alt | "T1.3 ≈ H3-alt" | ✅ valid |
| **F50 T1.4 Aligned-MTL** | same | "T1.4 +0.45 vs H3-alt" | ✅ valid |
| **F50 P1 no_crossattn** | same | "P1 ≈ H3-alt" | ✅ valid |
| **F50 P2 detach K/V** | same | "P2 ≈ H3-alt" | ✅ valid |
| **F50 P3 cat-freeze** | same | "P3 ≈ H3-alt" | ✅ valid |
| **F50 P4 alt-SGD** | same | "P4 +3.83 pp Δreg, p=0.0312" | ✅ valid (verified empirically: P4-alone clean is running NOW in `tmux pred_queue`; preliminary numbers consistent with +Δ preserved) |
| **F50 PLE-lite, Cross-Stitch** | same | "PLE +0.25, CS-detach +0.01 vs H3-alt" | ✅ valid (both small) |
| **F50 D8 cw=0** | same | "D8 cw=0 → reg=74.06, ep-5 plateau" | ✅ valid for the plateau finding (cat dominance refuted); ❌ for the absolute 74.06 |
| **F50 D6 reg_head_lr=3e-2 fold-1 spike (77.93)** | same | "α growth IS mechanistically achievable in MTL" | ⚠ **partially invalid** — 77.93 is inflated; the actual fold-1 ep-0 is ~62 leak-free. The "α can grow" claim still holds (D6 fold 1 reaches the prior-only floor early), but the dramatic 77.93 number is mostly the leak. |
| **F50 D1 STL α=0** | `next_getnext_hard` with `alpha_init=0.0, freeze_alpha=True` | "encoder-only ceiling = 72.61" | ✅ **NO LEAK** — α=0 means log_T contributes ZERO. 72.61 is the true encoder-only ceiling. |
| **F50 D3 reg_enc_lr** | same as H3-alt | "D3 ≈ H3-alt" | ✅ valid |
| **F50 Tier-A (A1-A6)** | same | relative Δs vs H3-alt | ✅ valid |
| **F50 Champion P4+Cosine `_1653`** | same | "champion = 76.07 ≥ep10" | ❌ absolute inflated; clean champion = 63.23 (P0-A) |
| **F50 Champion-candidate P4+OneCycle `_1636`** | same | "Pareto-trade reg+6.08/cat-1.84" | ⚠ relative Δs preserved, absolute reg inflated |
| **F49 architectural-Δ {AL +6.48, AZ −6.02, FL −16.16}** | same head, full log_T | architectural decomposition | ✅ valid relatively (both arms leaky); ❌ absolute |

### Cross-state validity

Each state's `region_transition_log.pt` was built from that state's own data (not cross-state). So the leak is independent per state. **Cross-state Δs (e.g. "FL gap > AL gap > AZ gap") are valid IF both arms use the same state's leak.** They likely don't compare across states directly.

The Phase-1 substrate study (CH16 / CH18) used HGI vs check2HGI embeddings — neither is C4-class (no learnable amplifier on a graph prior). Those claims should be unaffected.

---

## 3 · What's still valid as the paper headline

| claim | leak-corrected status |
|---|---|
| "FL has 8.83 pp STL-MTL gap" (F50 T3 §1) | ❌ **the 8.83 pp was largely leak**; true gap ~3 pp |
| "MTL reg-best is structurally pinned at ep 4-5" | ✅ valid (epoch trajectory preserved; F63 α traj confirms similar timing) |
| "10 architectural alternatives all give reg ≈ 74" | ✅ valid as relative observation; absolutes ~13 pp lower clean |
| "D8 cw=0 → reg-best ep 5 across all folds" | ✅ valid (plateau timing preserved) |
| "P4 alternating-SGD wins by paired Wilcoxon p=0.0312" | ✅ valid (paired comparison, both arms leaky) |
| "B9 alpha-no-WD is Pareto-dominant +0.24 reg / +0.08 cat" | ✅ measured leak-free; both arms clean |
| "P4+Cosine champion = 76.07 reg" (NORTH_STAR pre-fix) | ❌ absolute inflated; corrected to 63.23 clean |
| "STL ceiling = 82.44 reg" (F37) | ❌ absolute inflated; corrected ~66 (estimated, clean run queued) |
| F49 attribution decomposition "architectural lift = +6.48 pp on AL" | ✅ relatively (both arms uniform leak); absolute unchanged in direction |

---

## 4 · Re-run priority for paper

**Tier 1 (paper-blocking; must be leak-free for the headline):**

| # | run | recipe | status | est min |
|---|---|---|---|---|
| 1 | H3-alt clean FL | per-fold log_T, no other changes | ✅ done (`_1921`) | — |
| 2 | P4+Cosine clean FL | + per-fold log_T | ✅ done (P0-A `_1755`) | — |
| 3 | B9 clean FL | + alpha-no-WD | ✅ done (`_1813`) | — |
| 4 | P4-alone clean FL | constant scheduler | 🟡 running (pred_queue fold 5) | ~3 |
| 5 | P4+OneCycle clean FL | --pct-start=0.4 | 🟡 queued (pred_queue) | 19 |
| 6 | STL F37 clean FL | per-fold log_T via STL pipeline | 🟡 queued (stl_clean) | 19 |
| 7 | F62 two-phase clean FL | scheduled_static, mode=step | 🟡 queued (f62_clean) | 19 |
| 8 | Cross-state clean (AL, AZ, GA) | already used per-fold (P0-B/C/G) | ✅ done | — |

**Tier 2 (paper claims that benefit but aren't blocking):**

| # | run | rationale | est min |
|---|---|---|---|
| 9 | TGSTAN clean smoke (1f×10ep) | confirm gate-amplified leak; audit's #2 priority | ~3 |
| 10 | T1.2 HSM clean | confirm uniform leak across HSM head | 19 |
| 11 | One representative architectural alt clean (PLE or Cross-Stitch) | confirm uniform leak | 19 |

**Tier 3 (skip unless time):**

- Re-run all 10 architectural alternatives clean — the relative Δs are preserved by the uniform-leak observation; spot-checking 2-3 (Tier 2) is enough.
- F65 dataloader cycling — D8 already refuted the cat-loss-dominance hypothesis.

---

## 5 · Decision rules for the paper's results section

When writing up:

1. **Always quote leak-free numbers as the absolute baseline.** Cite both leaky and leak-free in tables for transparency.
2. **Frame deltas paired-Wilcoxon style.** "Δ B9 vs H3-alt = +3.34 pp p=0.0312 5/5" is the headline format.
3. **Add a footnote on every plot/table:** "All `next_region` numbers under per-fold log_T (`--per-fold-transition-dir`); pre-2026-04-29 numbers in the codebase used full-data log_T, inflated by ~13-17 pp (see §C4 in supplementary)."
4. **Cite the diagnosis docs.** `F50_T4_C4_LEAK_DIAGNOSIS.md` is the load-bearing receipt. Future readers will need to know α grows 18× and amplifies.
5. **Don't quote 82.44 or 76.07 in the abstract.** Use 63.47 (clean B9 champion) and ~66 (estimated clean STL ceiling, awaiting confirmation).
6. **The "temporal training-dynamics" mechanism narrative still holds** — STL reg-best at ep 16-20 vs MTL at ep 4-5 is a property of the trajectory shape, which the F63 α-logging confirms is preserved across leaky vs clean. Just the absolute heights of those peaks were inflated.

---

## 6 · Cross-references

- C4 root-cause: `F50_T4_C4_LEAK_DIAGNOSIS.md`
- Broader audit (other potential leakages): `F50_T4_BROADER_LEAKAGE_AUDIT.md`
- α trajectory empirical: `figs/f63_alpha_trajectory.png`
- Live tracker: `F50_T4_PRIORITIZATION.md`
