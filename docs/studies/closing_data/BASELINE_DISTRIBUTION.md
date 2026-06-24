# Baseline execution — machine distribution (2026-06-24)

> Executable per-machine plan for the **baseline phase**, derived from the LOCKED decisions in
> [`../../../articles/[mobiwac]/BASELINE_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/BASELINE_HANDOFF.md)
> (the *why* + roles + drop list). Machines pull THIS doc. Protocol: seed 0 × 5 folds (n=5), gated stride-1
> overlap, leak-free per-fold train-only priors, user-disjoint folds, fp32 for large-state CUDA. Supersedes the
> prior CTLE-SC-at-CA/TX handoff (removed): CTLE is **FL-only** now.
>
> **Machine state:** A40 frees in ~1 h · H100 finishing TX MTL but has spare capacity NOW · M4 Pro free.

## A · DROP — do NOT run (re-steered away)
CTLE-SC at AL/AZ/CA/TX as a ladder (CTLE is **FL-only**, role-2) · ReHDM CA/TX faithful (footnote infeasible) ·
STAN-faithful at scale (strawman; `stl_hgi` is the point) · MHA+PE (redundant w/ POI-RGNN) · SC-region (quarantined) ·
full SC cross-state program · LLM zero-shot · any new from-scratch region SOTA.

## B · DONE — zero compute, just TABULATE (orchestrator/doc; not a machine job)
POI-RGNN faithful (canonical FL 34.49/CA 31.78/TX 33.03 + AL/AZ/GA) · Markov-9-cat floor · Markov-1 floor ·
ReHDM faithful AL/AZ/FL (66.06/54.65/65.68) · STAN `stl_hgi` all states (AL 62.88…TX 62.70) · HMT-GRN
AL/AZ/FL/CA/Istanbul (`docs/baselines/`). → slim Tables A/B (`BASELINE_HANDOFF §6`).

## C · MISSING — distributed (each machine reads its OWN self-contained handoff)
> [`BASELINE_H100.md`](BASELINE_H100.md) · [`BASELINE_A40.md`](BASELINE_A40.md) · [`BASELINE_M4.md`](BASELINE_M4.md).
> The summary below is the cross-machine map; the per-machine files carry the full env/commands/traps/validation.

### H100 (CUDA) — the FL ROLE-2 novelty block + CSLSL FL  ·  **paper-critical, keep on ONE device**
All FL, seed 0 × 5f, dk_ovl, leak-clean per-fold → one device so every Δ is clean. Start now (spare capacity);
the FL CTLE-SC step waits on the M4 diagnosis (§D).
1. **Check2HGI-SC @ FL** (the comparand) — `comparand_check2hgi_sc.py --state florida` (next_gru cat + next_stan_flow reg, checkin-modality).
2. **CTLE-SC @ FL** — `build_ctle_substrate.py --state florida --fold {0..4} --stride 1` (per-fold, leak-clean) → matched head. *(Gated on §D diagnosis.)*
3. **CTLE-E2E @ FL** — `ctle_e2e.py --state florida --seed 0 --folds 5` (CTLE's true fine-tuned strength; the headline CTLE number).
4. **feature-concat @ FL** — HGI ⊕ raw per-visit features (category one-hot + hour/day sin/cos) → `next_gru`. Needs a thin builder (extend `scripts/probe/build_design_a_concat.py`, which currently concats c2hgi⊕HGI — swap the 2nd stream for raw features). No new embedding training.
5. **CSLSL cascade @ FL** — `b4_cascade.py --state florida --seed 0 --folds 5` (champion-G FL is already on H100 → clean same-device cascade-vs-parallel Δ).

### A40 (CUDA, stable) — ROLE-3 CSLSL on the small/mid states  ·  when its current run ends (~1 h)
- **CSLSL cascade @ AL, AZ** — `b4_cascade.py --state {alabama,arizona} --seed 0 --folds 5`. CA/TX only if cheap.
- **Clean-Δ note:** the cascade comparand is champion-G (ran on H100). For a same-device Δ, re-run champion-G AL/AZ on the A40 alongside (cheap), OR accept the documented cross-GPU ±0.05 pp caveat (acceptable for this internal cascade-vs-parallel ablation, where the signal ≫ 0.05 pp).
- If H100 is saturated, the A40 can take **CSLSL @ FL** instead (it's the stable card for the long run).

### M4 Pro (MPS) — diagnosis + safety-net  ·  free now
1. **DIAGNOSE the CTLE frozen-below-floor** (role-2 step 1, **CRITICAL PATH — gates H100 step C.2**): the recorded frozen CTLE-SC cat (AL 17.8, *below* the bigram floor) is suspicious. Confirm the frozen CTLE embedding is actually feeding the head (not a pipeline/leak-fix artifact) — `build_ctle_substrate.py` + `mac_baseline_compare.py` on AL (where the 17.8 was seen; small, MPS-fast). Report: real CTLE weakness vs pipeline bug. **Post the verdict before H100 runs FL CTLE-SC.**
2. **TX HMT-GRN** — finish the in-flight safety-net row (already a Mac chain, "building dk_ovl→HMT"). Completes the HMT-GRN external-MTL row at all 6 states.
3. *(optional, if A40 busy)* **CSLSL @ AL/AZ** on MPS — HMT-GRN validated MPS==CPU within 0.06 pp, so MPS from-scratch training is trustworthy here; label `[M4/MPS]` and cross-check one fold vs CPU.

## D · Sequencing / clean-Δ rules
- **CTLE FL gate:** M4 diagnosis (C.M4.1) → if real CTLE weakness, H100 runs FL CTLE-SC (C.H100.2); if pipeline bug, fix first. CTLE-E2E (C.H100.3) and the comparand (C.H100.1) and feature-concat (C.H100.4) do NOT wait — run them now.
- **Same-device Δ:** keep each comparison's two sides on one device-class (FL role-2 all on H100; each CSLSL paired with its champion-G comparand on the same card). Cross-GPU is tolerable only for the internal CSLSL ablation (signal ≫ ±0.05 pp), with a footnote.
- **Honesty:** n=5 provisional everywhere; never print a 5f TX mean (TX MTL 2/5); never cite VOID fp16/bf16 JSONs; CTLE presented as "even in its best (E2E) form CTLE is well below ours" — never "we crushed CTLE."

## E · Priority (matches BASELINE_HANDOFF §7)
1. Tabulate DONE (B) — now, zero compute. 2. FL role-2 block (H100) + CTLE diagnosis (M4) — decisive for novelty.
3. CSLSL (A40 small states + H100 FL) — highest new-run story value; HMT-GRN already covers the fallback.
4. Post-deadline: {1,7,100}→n=20; CA/TX faithful region SOTA if compute frees.
