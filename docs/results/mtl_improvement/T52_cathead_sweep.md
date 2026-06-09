# T5.2 — cat-head encoder sweep UNDER champion G (2026-06-09)

**Re-run properly under champion G** (the earlier T5.2 "DONE" rested on the Tier-S STL head search +
the B-A4 cat-*loss* family; a clean cat-*encoder* swap under MTL-G across the registry had never been
run). Motivated by the cat-transfer ablation finding that the cat gain is **architecture-driven** — so a
better head on the cross-attn trunk could plausibly add. Swap `--cat-head` only; everything else = G.
10 cat-capable encoders × {AL, FL}, seed0, matched-metric. Driver
`scripts/mtl_improvement/t52_cathead_sweep.sh`; data `T52_cathead_sweep.json`.

## Result (Δcat vs G's `next_gru`; band gate = win at BOTH states)

| cat-head | AL cat | ΔAL | FL cat | ΔFL | verdict |
|---|---|---|---|---|---|
| **next_gru (G)** | **52.75** | +0.00 | **73.12** | +0.00 | **CHAMPION (only head strong at both)** |
| next_conv_attn | 31.25 | **−21.50** | 74.18 | **+1.06** | FL-only (craters AL) |
| next_temporal_cnn | 29.15 | **−23.60** | 73.71 | +0.59 | FL-only (craters AL) |
| next_lstm | 51.47 | −1.28 | 73.46 | +0.34 | FL-only (loses AL) |
| next_hybrid | 50.20 | −2.55 | 73.34 | +0.22 | ≈/worse |
| next_tcn_residual | 36.48 | −16.27 | 73.21 | +0.09 | ≈/worse |
| next_transformer_relpos | 37.18 | −15.57 | 72.25 | −0.87 | worse |
| next_mamba | 43.40 | −9.36 | 71.93 | −1.19 | worse |
| next_single | 41.01 | −11.75 | 70.08 | −3.04 | worse |
| next_transformer_optimized | 37.04 | −15.71 | 63.58 | −9.54 | worse |

reg-full is **flat** across all cat heads (AL range 0.31, FL range 1.16) — the dual-tower isolates reg
from the cat head, as designed.

## Verdict — `next_gru` is the multi-state cat champion under G (now on real under-G evidence)

**No cat head wins at both states.** Every head that beats next_gru at FL — `next_conv_attn` (+1.06),
`next_temporal_cnn` (+0.59), `next_lstm` (+0.34), `next_hybrid` (+0.22) — **craters at AL** (−1.3 to
−23.6 pp). `next_gru` is the unique head that is strong at both scales. This is the **same FL-only trap**
as G′ (CONCERNS §C26) and next_single (O2): a large-state win that reverses at small data fails the
multi-state band gate. **T5.2 CLOSED under G: `next_gru` confirmed.**

**Bonus (scale-conditional, future-work — not a champion change):** at FL specifically, a
convolutional-attention cat head (`next_conv_attn`) beats next_gru by **+1.06 pp** at flat reg — a real
large-state cat lever, in the same family as the other scale-conditional findings (overlap, design_k).
Like those, it is **not adopted** (craters small states; the study ships one multi-state recipe). Logged
as a CA/TX-relevant option should the paper ever go scale-conditional on the cat head.

**Why this matters / what it changes:** nothing in the champion — but it converts T5.2 from a
closed-by-inheritance card into a **closed-by-direct-under-G-evidence** card, and it strengthens the
paper's cat story: the recurrent `next_gru` head is the right multi-state choice on top of the cross-attn
trunk; the more expressive encoders (transformers, mamba, conv) over-fit / mismatch at small data exactly
as the orthogonality + small-data theme predicts.

## Scope
seed0, FL+AL, matched-metric. reg side (T5.1): the coded private-tower types (stan/gru/lstm/tcn) were
swept under G in B-A1 (STAN load-bearing, others −1.8…−3.4 pp); other reg architectures as the private
tower would need new code (low-EV — STAN-family also won the STL search). T5.3 (HSM) falsified
(`cat_transfer_and_T53.md §b`). → **Tier 5 fully closed on under-G evidence.**
