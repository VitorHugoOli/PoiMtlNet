# CSLSL / cascade (B4) — the "published MTL alternative" (cascade-vs-parallel)

> **Scope caveat (read first):** this baseline is an **internal cascade-vs-parallel ablation on the v14 `design_k`
> substrate at set-a windowing** — it is NOT on the dk_ovl stride-1 board base. **Its numbers are only comparable to
> the matched champion-G run on that SAME v14 set-a base** (provided here), never to the board `RESULTS_BOARD §1`
> dk_ovl champion. Kept out of `next_region/comparison.md` for exactly this reason.

## Source
- **Papers:** CSLSL (when→what→where) — Wang et al., *EPJ Data Science* 2024; CatDM (cat-pref → POI-pref) — Yu et al., *WWW* 2020.
- **Impl:** `scripts/baselines/b4_cascade.py` — a **pattern port**, not a faithful re-implementation (deferred). It reuses our champion heads as the two cascade stages so the ONLY varying factor vs champion-G is the coupling topology.

## Why this is a baseline (not our model)
CSLSL/CatDM are the dominant **published multi-task pattern**: a *directed* cascade (predict category, then predict location **conditioned on** the predicted category — no reverse path). Our champion-G couples the two tasks **symmetrically** (bidirectional cross-attention). B4 isolates that single factor — **cascade vs parallel** — on the frozen substrate + matched heads, answering "does our symmetric coupling beat the published directed cascade?"

## What's faithful, what's adapted (deviation ledger, from the b4 docstring)
- **D1 Heads:** reuse OUR champion heads (`next_gru` cat, `next_stan_flow_dualtower` reg) as the cascade stages — not the papers' RNN/LSTM decoders — so coupling topology is the only varying factor.
- **D2 No next-POI head:** stage-2 target is next-**REGION** (our board target), so the cascade is cat→region.
- **D3 Labels:** 7 Gowalla root categories + TIGER-tract regions (not the source vocabularies).
- **D4 Substrate:** runs over the frozen Check2HGI v14 substrate + matched heads, not raw-trajectory encoders.
- **D5 Coupling:** additive **zero-init** posterior injection (`cond_proj` zero → untrained cascade ≡ champion-G); the iMTL/GETNext input-feature form.

**Pinned config** (the only diff from champion-G): `cond_coupling=posterior, cond_signal=softmax, cond_inject=add, cond_detach=True` + `disable_cross_attn=True` (sever the symmetric channel so the directed cat→region edge is the only coupling).

## Leak-safety — AUDITED CLEAN (advisor-reviewed, 2026-06-24)
The cascade edge injects the model's **own forward-pass category prediction** `cat_cond = softmax(out_cat)` (from input embeddings), read **detached** (`cond_detach=True`, `head.py:453`) — never a label/target. `cond_proj` zero-init. This is the standard CSLSL "downstream reads upstream *prediction*", not label access. The comparand is **fair** (both models see both streams; only topology differs). Inherited guards verified at runtime per cell: leak-preflight, train-only per-fold log_T, user-disjoint `StratifiedGroupKFold`. Full audit: `../results/closing_data/CSLSL_CASCADE_RESULTS.md`.

## Variants we run
- `cascade_sc` — directed cat→region cascade on the frozen v14 substrate + matched heads. Comparand = **champion-G (parallel) on the SAME v14 set-a base** (the b4 command minus the 5 cascade pins, cross-attn ON).

## Results — v14 `design_k` engine, set-a windowing, MPS, seed 0 × 5 folds (n=5 provisional)

| State | cascade cat / reg@10 | champion-G (parallel) cat / reg@10 | **Δ (parallel − cascade)** | folds | device |
|---|---|---|---|---|---|
| Alabama | 45.93 / 63.42 | 50.97 / 63.98 | cat **+5.04**, reg **+0.56** | 5 ✅ | M4/MPS |
| Arizona | 53.21 / 54.48 | 54.83 / 54.43 | cat **+1.62**, reg **−0.05** (tie) | 5 ✅ | M4/MPS |
| Florida | 71.08 / 72.81 *(4-fold partial)* | — (not obtained) | **N/A — do not cite** | 4/5 ⚠ | M4/MPS — **MPS-OOM** |

(cat = macro-F1; reg = top10_acc_indist; both at the joint geom_simple checkpoint. MPS validated == CPU within 0.63 pp at AL.)

**Read:** the **parallel symmetric cross-attention coupling (champion-G) beats or matches the directed cat→region cascade** wherever it completed — cat **+5.04 (AL) / +1.62 (AZ)**, reg **+0.56 (AL) / −0.05 (AZ, tie)**. Expected cascade signature: severing cross-attn costs the **category** head most (it loses the reg→cat help), while reg still gets the cat→reg edge. Substantiates that our parallel cross-task coupling ≥ the published cascade pattern under matched substrate + heads. Honest framing: "parallel beats the cascade on category at both states, beats/ties on region" — never "crushes."

## Status / TODO
- ✅ **AL, AZ** done on the M4 with matched comparand + MPS==CPU cross-check. **n=5 provisional (seed 0).**
- ⚠ **FL** attempted on the M4 → **MPS-OOM** (24 GB insufficient for 159k rows / 4,703 regions under desktop load); 4/5 folds, no comparand. **Run on CUDA (A40/H100)** — its documented lane (`../studies/closing_data/HANDOFF_A40.md`).
- **CA / TX**: A40/post-deadline lane.
- Multi-seed {1,7,100}: post-deadline.

## Provenance
- Driver: `scripts/baselines/b4_cascade.py`; comparand = same train.py invocation minus the 5 cascade pins.
- Per-state JSONs: `../results/closing_data/baseline_compare/{alabama,arizona,florida}_cslsl_cascade.json`. Narrative + engineering knowledge: `../results/closing_data/CSLSL_CASCADE_RESULTS.md`.
