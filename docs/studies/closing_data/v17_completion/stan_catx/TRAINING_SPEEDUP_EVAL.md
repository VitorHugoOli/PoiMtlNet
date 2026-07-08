# Training-speedup eval — faithful STAN (+ carry-over practices) — 2026-07-06

> 3-advisor static analysis (model/algo · pytorch-systems · perf-history guardrail) run while the CA/TX STAN job
> trained. Goal: **quality-neutral** speedups (bit-exact eager or within-fold-noise). **Outcome: the trainer is already
> near-optimal; the two "obvious" PyTorch wins are traps; the only transformative lever is a high-effort rewrite.**
> **Decision (user 2026-07-06): document + carry the learnings; do NOT retrofit the running STAN job** (near-optimal
> coverage cell, changes no verdict, already beaten at every state).

## The bottleneck (grounded)
STAN's step (~427 ms at bs2048/S=50, TX R=6553 / CA R=8501) is **HBM-bandwidth-bound on the `[B,S,R]` region tensors**
(2.7-3.5 GB each; the matching layer materializes ~4-5 of them per forward; **backward = 82 % of the step**). The
matching layer's O(B·n·R·D) distance-gated ranking is **mathematically irreducible**. So precision/kernel/fusion tweaks
help the *bandwidth*; FLOP or data-pipeline tweaks do **not** (the `content` matmul is ~6 ms of the 427).

## Already applied + validated (do NOT re-propose)
`F.embedding` for all gathers (the 85× backward fix — the generic `indexing_backward` was 97.7 % of backward) ·
`dd_poi` + per-POI-bias precompute · distinct-POI interp (opt-D, **bit-identical**) · val-once (opt-C, bit-identical) ·
`torch.compile(dynamic=True)` default · opt-in bf16 (A/B quality-neutral, big states) · all fold tensors GPU-resident,
**no DataLoader** · streaming ETL (this session, `06c24757`). Commits `1b83c1c1 / abcd7a06 / 507a5f22 / 1eeb43fd`.

## ⚠ Traps — tried/rejected, DO NOT apply (all 3 advisors converged)
- **`--compile-mode max-autotune`** — **SEGFAULTS** during Triton autotune (opt-E, core dump). The flag exists but crashes.
- **Fused/foreach AdamW** — rejected in the main pipeline: ~0 gain, **unknown-sign FP-reorder noise**, high repro-risk.
- **TF32** — **moot under the bf16 recipe** (matmuls already autocast below TF32) AND breaks the deliberate "true-fp32"
  board protocol (`allow_tf32=False`, `set_float32_matmul_precision("highest")`). No free win.
- **`--compile-mode reduce-overhead`** (CUDA graphs) — forces `distinct_poi=False` → interps the **full `[n_pois,R]`** table
  (TX ~5 GB) every step; the launch saving is dwarfed. No measured speedup.
- **Larger batch (4096/8192)** — no wall-time gain (matching saturates at bs2048) **and −0.6…−0.8 pp quality**.
- **SDPA/flash in `_SelfAttn`** — the `[B,50,50]` attn is <5 % of the step; additive learned bias + all-pad rows don't map.
- **`num_workers>0`** — moot (no DataLoader); breaks RNG byte-identity elsewhere.

## Actionable quality-neutral wins (ranked) — for FUTURE STAN runs, if ever needed
| # | win | speedup | quality | effort |
|---|---|---|---|---|
| 1 | **Skip pad positions** in the matching layer (at S=50 most positions are zeroed pad but fully computed) → gather real tokens, `index_add_` back | **2-6×** (matching layer) | bit-exact if ascending-n order kept | HIGH (needs golden-logit parity + AL 5f A/B) |
| 2 | **Warm/shared inductor cache** across the 5 folds (`TORCHINDUCTOR_CACHE_DIR` + `torch._dynamo.config.cache_size_limit`) — each fold recompiles cold today | up to ~21 % (compile) | **zero numerical effect** (infra) | trivial |
| 3 | **Fuse mask+collapse** into one multiply-reduce (drop the `gated` materialization + `masked_fill` pass + separate `collapse` GEMM) | ~1.1× | bit-exact (≤1 ULP) | low |
| 4 | Remove the per-val-batch `.item()` sync (accumulate `top10_correct` on-GPU, sync once/epoch) | tiny | bit-exact | trivial |
| — | *bf16 bias-path cast* — the doc's "bf16 halves `[B,S,R]` traffic" is **FALSE**: `F.embedding`/interp stay fp32 under autocast, so `content(bf16)·bias_match(fp32)` promotes back to fp32. Casting `bias_match` to bf16 halves the dominant traffic | ~1.3× | **numeric → needs A/B on FL** | low |

Key lines: matching fwd `model.py:195-215`; bias path `:186-193,:67-88,:294-301`; amp/compile `train.py:131-150`; val `.item()` `:201`.

## Biggest wall-time lever overall (needs the H100, not this A40)
`--only-fold` **fold fan-out on the H100** — all 5 bf16 folds fit in 80 GB with genuine overlap → est. CA ~50-75 min /
TX ~40-60 min vs **6-11 h** sequential on the A40. Zero quality risk (independent folds). Mechanism already wired.

## Carry-over practices → ReHDM (the 75-120 h/state next job — where speed actually matters)
The STAN matching-layer wins are model-specific and **don't transfer**, but the *practices* do — bake these into the
ReHDM setup from the start:
1. **Warm/persistent `TORCHINDUCTOR_CACHE_DIR`** if ReHDM uses `torch.compile` (skip per-fold/per-seed cold compiles).
2. **bf16 opt-in only** — never bf16-default; the A40 bf16-grad-NaNs at large C (CLAUDE.md). Keep an fp32 fallback.
3. **Avoid the traps**: no `max-autotune` (segfault), no fused-AdamW, no TF32-vs-fp32-protocol, no `num_workers` for
   byte-identity, no larger-batch-for-speed (it moved STAN quality −0.8 pp).
4. **GPU-resident data + no DataLoader** where the dataset fits (STAN's biggest single win at small scale).
5. **Profile first** (`profile_forward.py`-style component timing) to find ReHDM's true hotspot before optimizing.
6. **H100 fold fan-out** is the real throughput lever if a card is free — prefer it over single-GPU kernel work.
