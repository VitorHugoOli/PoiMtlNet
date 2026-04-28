# F21c — STL GETNext-hard (matched-head STL baseline) findings

**Date:** 2026-04-24. **Tracker:** `FOLLOWUPS_TRACKER.md §F21c`. **Sources:** `results/B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json`.

> **Phase-1 update (2026-04-27):** F21c gave Check2HGI side at AL+AZ. Phase 1 added the **HGI side** at AL+AZ under the same matched head, allowing a direct substrate comparison. Findings (5f × 50ep, seed 42):
>
> | State | C2HGI Acc@10 | HGI Acc@10 | Δ | Wilcoxon p_greater |
> |---|---:|---:|---:|---:|
> | AL | 68.37 ± 2.66 | 67.52 ± 2.80 | +0.85 | 0.0625 marginal · TOST δ=2 pp ✅ non-inf |
> | AZ | 66.74 ± 2.11 | 64.40 ± 2.42 | **+2.34** | **0.0312** ✅ |
>
> This **flips the existing CH15 verdict** ("HGI > C2HGI on reg under STAN at all 3 states"). Under the matched MTL reg head — the same `next_getnext_hard` head F21c uses — C2HGI ≥ HGI: AL tied within σ (TOST non-inferior at δ=2 pp), AZ significantly C2HGI. CH15 was head-coupled to STAN's POI-stable preference, not pure substrate quality.
>
> Phase-1 also lands MTL+HGI counterfactual (CH18 in `CLAIMS_AND_HYPOTHESES.md`): MTL B3 with HGI substrate produces Acc@10_indist of 29.95 (AL) / 22.10 (AZ) — **worse than STL+HGI gethard alone by ~37 pp at AL**. The MTL configuration breaks when paired with HGI; F21c's "STL > MTL on reg" finding under Check2HGI substrate becomes "STL > MTL > broken" when extended to the HGI substrate.
>
> See `baselines/check2hgi_v_hgi/phase1_verdict.md` and `SESSION_HANDOFF_2026-04-27.md`.

## Headline

**The hard graph prior alone, as a single-task STL model, beats MTL-B3 on next-region by 12–14 pp Acc@10 on AL and AZ.** The MTL coupling we've been treating as the paper's contribution provides no region-side lift — it dilutes.

## Numbers

| Metric | State | STL STAN | STL GRU | **STL GETNext-hard (F21c)** ⭐ | MTL-B3 | Δ (F21c − STL STAN) | Δ (F21c − MTL-B3) |
|---|:-:|---:|---:|---:|---:|---:|---:|
| Acc@10 | AL | 59.20 ± 3.62 | 56.94 ± 4.01 | **68.37 ± 2.66** | 56.33 ± 8.16 | **+9.17** | **+12.04** |
| MRR | AL | 36.10 ± 1.96 | 34.57 ± 2.34 | **41.17 ± 2.28** | 28.55 ± 5.33 | **+5.07** | **+12.62** |
| F1 (macro-reg) | AL | 24.64 ± 1.38 | — | 11.91 ± 0.86 | 9.43 ± 0.71 | −12.73 | +2.48 |
| Acc@10 | AZ | 52.24 ± 2.38 | 48.88 ± 2.48 | **66.74 ± 2.11** | 52.76 ± 3.92 | **+14.50** | **+13.98** |
| MRR | AZ | 33.70 ± 2.36 | 32.13 ± 2.21 | **41.15 ± 2.13** | 26.40 ± 2.45 | **+7.45** | **+14.75** |
| F1 (macro-reg) | AZ | 24.48 ± 2.29 | 23.63 ± 2.04 | 12.28 ± 0.91 | **9.17** | −12.20 | +3.11 |

## Interpretation

### What F21c proves
The MTL coupling (shared backbone + cross-attn + joint cat/reg training) does **not** lift region quality beyond what the GETNext-hard head achieves standalone. On the contrary: at AL scale MTL loses 12 pp Acc@10, at AZ scale MTL loses 14 pp Acc@10. The STL with the same head class wins cleanly and outside σ.

### What F21c does NOT refute
- **STL GETNext-hard produces only next-region.** MTL-B3 produces both tasks in one forward pass. For deployment scenarios requiring joint prediction in one model, MTL-B3 is still the joint-output option.
- **Cat F1 under MTL-B3 is competitive.** At AZ the Wilcoxon p=0.0312 vs STL cat still holds — the MTL framework *does* add a small cat-F1 lift.
- **The macro-F1 on region is lower for GETNext-hard** (11–12 %) than STL STAN (24 %). This is because the prior's top-k behaviour is hot-start — many correct predictions on high-prior rare regions drive Acc@10 up but the sample-weighted macro-F1 lags. This is a secondary metric issue worth reporting.

### What it means for the paper

The F21c result forces a reframing. Three honest options:

**Option A — Multi-task motivation (deployment-focused).** "MTL-B3 is the best *single-model* joint predictor for {next_category, next_region}. A practitioner choosing to deploy two STL models (one GETNext-hard for region + one Check2HGI cat-MLP for category) achieves better region quality but doubles inference cost and training complexity. We report both."

**Option B — Matched-head ablation as the paper's main finding (honest & interesting).** "We investigated whether MTL coupling lifts performance beyond a matched-head STL baseline. On two mid-scale states, the single-task model with the graph-prior head outperforms the MTL configuration on region by 10+ pp Acc@10. The cat-side MTL lift (+1.65 pp F1, p=0.0312) is real but small, and is potentially obtainable with a similarly matched STL cat head (not measured here). We recommend future MTL-POI work carry matched-head STL baselines; our MTL variants improved over unmatched STL baselines but not over matched ones."

**Option C — Shift the paper to a graph-prior focus.** "We show that a faithful graph-transition prior (GETNext-hard), single-task trained, is the new SOTA on Check2HGI region prediction at AL/AZ scale. The MTL framing is reported as a joint-deployment option but not as the primary method." This reframes around the **graph prior**, not the MTL coupling.

### Recommendation

Adopt **Option B or a hybrid B+A**. The paper's contribution becomes:

1. **Methodological honesty**: matched-head STL baselines change the MTL-coupling claim. We document this as a finding.
2. **Substrate claim (CH16)**: Check2HGI > HGI on cat remains unchanged and paper-worthy.
3. **Mechanism claim**: the PCGrad × hard-prior × FL gradient-starvation finding from F2 is independently paper-worthy.
4. **Applied claim**: deployment simplicity — one MTL model for both tasks at a measurable but bounded region cost.

## Caveats

- **FL not yet tested.** F21c on FL is pending (~5–6 h). FL reg is Markov-saturated (Markov-1-region 65.05; STL GRU 68.33). STL GETNext-hard on FL may not show the same magnitude of lift. If FL STL GETNext-hard lands ≤ 68.33 pp, the "STL > MTL" conclusion is state-scale-dependent.
- **Cat F1 comparison is still the standalone MTL win.** The F21c finding is entirely on the region head; the cat comparison stands.
- **Matched-head STL cat not measured.** If STL with `next_mtl` (matching MTL's task_a head) also outperforms the MTL cat F1, the entire MTL value proposition collapses. F27 (cat-head ablation) does not directly test this, but indirectly — if MTL-B3 with different cat heads produces varying cat F1, a matched-head comparison on cat becomes informative.
- **AL/AZ are ablation states**, not headline. The paper's headline is FL/CA/TX. If FL/CA/TX show a different pattern (less pronounced STL>MTL gap due to scale), Option B's framing is less severe.

## Next steps

1. **F27 cat-head ablation on AZ** — does choice of task_a head change the MTL-B3 story on cat? Testing this sharpens Option B above. **Launching now.**
2. **F21c FL 5-fold** — ~5–6 h. Needs to run to settle the state-scale dependence.
3. **Matched-head STL cat** — add as F28: run STL `next_mtl` on AL/AZ/FL as a matched-head cat baseline. Would isolate whether MTL's cat lift (+1.65 pp AZ) survives matched-head comparison. ~30 min per state.

## Files

- `results/B3_baselines/stl_getnext_hard_al_5f50ep.json`
- `results/B3_baselines/stl_getnext_hard_az_5f50ep.json`
- `results/P1/region_head_{alabama,arizona}_region_5f_50ep_stl_gethard.json` (source)
- `scripts/run_f21c_stl_getnext_hard.sh` (launcher, AL + AZ)
- `scripts/p1_region_head_ablation.py` (extended to support `next_getnext_hard` — commits 2026-04-24)
