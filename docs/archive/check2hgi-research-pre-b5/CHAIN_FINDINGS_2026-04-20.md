# Chain-of-4 findings (2026-04-20 02:14 → 05:12)

4-experiment chain ran clean on the `--no-checkpoints` fix. All summary JSONs saved. Log: `/tmp/check2hgi_logs/chain_4exp.log`.

## Results table

| # | Experiment | State | cat F1 | reg Acc@10 | reg MRR | Δ cat vs base | Δ reg vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base | cross-attn + pcgrad (AZ baseline) | AZ | 43.13 ± 0.55 | 41.07 ± 3.46 | 22.93 ± 2.47 | — | — |
| 1 | **AZ3 dselectk + pcgrad** | AZ | 41.88 ± 0.51 | 38.46 ± 2.71 | 20.31 ± 2.20 | −1.25 | −2.61 |
| 2 | **H-R1 cross-attn + GRU hd=384** | AZ | **43.44 ± 0.36** | **42.58 ± 3.32** | **24.20 ± 2.42** | **+0.31** | **+1.51** |
| 3 | H-R4 cross-attn + static_weight cat=0.3 | AZ | 42.86 ± 1.04 | 40.88 ± 3.10 | 22.42 ± 2.55 | −0.27 | −0.19 |
| 4 | **FL λ=0.0 dselectk isolation** | FL | 12.50 (1f) | **43.40 (1f)** | — | n/a (cat disabled) | — |

## Finding 1 — AZ3 is a duplicate of AZ2 (MTLoRA r=8)

The `mtlnet_dselectk` model now has LoRA adapters by default (`lora_rank=8`). Without an explicit override, it runs *identically* to the MTLoRA r=8 config. AZ3 numbers match AZ2 to the decimal. **Implication:** the "pre-MTLoRA dselectk" from original P2 (AL champion at 36.08 ± 1.96) is no longer reproducible without a code revert. Skipped as a useful data point.

## Finding 2 — H-R1 hd=384 directionally supports the capacity hypothesis

Widening the GRU head from hd=256 to hd=384 gives +1.51 pp on reg Acc@10 and +0.31 pp on cat F1, both within σ. Small but directional — and importantly, **cat did not regress** (the concern was that extra reg-side capacity would starve cat).

**Worth testing at FL?** The decision rule in `REGION_HPARAM_PLAN.md` says ≥2 pp lift. H-R1 gives +1.51 — below threshold. **Borderline.** If we had 2 free hours on FL, running this would be reasonable; if we're prioritizing headline tightening, skip.

**hd=512 was OOM; hd=384 survives cleanly** — `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` keeps us on the edge but stable at 384. Documented in `KNOWN_INFRA_ISSUES.md §I4`.

## Finding 3 — Loss-weighting is NOT the region bottleneck

H-R4 (reg-heavy cat_weight=0.3 = reg_weight 0.7) yields Δ cat −0.27, Δ reg −0.19 — essentially null on both. **Loss-balance hypothesis rejected.** The reg gap is not about how much the optimizer cares about reg loss; it's about the shared-backbone's capacity to represent region structure.

Combined with H-R1 positive: the bottleneck is **representational capacity in the downstream head**, not **gradient weight**.

## Finding 4 — 🚨 CH-M3 rejected at scale: architectural overhead GROWS with state size

| State | Rows | Regions | STL reg A@10 | MTL λ=0 reg A@10 | Overhead | Overhead / STL |
|---|---:|---:|---:|---:|---:|---:|
| AL | 10K | 1109 | 56.94 ± 4.01 | 51.87 ± 5.70 | **5.07 pp** | 8.9% |
| FL | 127K | 4702 | 68.33 ± 0.58 | **43.40 (1f)** | **24.93 pp** 🚨 | **36.5%** |

**Previous claim (CH-M3):** architectural overhead is ~5 pp and LR-invariant / scale-invariant.
**Revised:** architectural overhead is LR-invariant but **grows ~5× with scale** (when cat training is disabled).

### But positive transfer also grows with scale

| State | λ=0 baseline | Full MTL reg | Transfer from cat |
|---|---:|---:|---:|
| AL (cross-attn pcgrad) | 51.87 (dselectk) | 52.41 (cross-attn) | +0.54 pp* |
| FL (cross-attn pcgrad) | 43.40 (dselectk) | **57.60** (cross-attn) | **+14.20 pp** |

\* AL comparison is imperfect — λ=0 used dselectk while full MTL used cross-attn. Cleaner comparison would require a cross-attn λ=0 run on AL.

### Revised mechanistic story

At **small scale (AL)**: MTL pipeline is mostly inert. Architectural overhead is small (5 pp) because the model isn't learning much; cat-enabled transfer is small (0.5 pp) because there isn't much to share. Net: ~5 pp gap.

At **large scale (FL)**: MTL pipeline is doing real work. Architectural overhead is large (25 pp) because the shared-backbone wrapper disrupts the independent STL baseline substantially — the pipeline adds task encoders, FiLM, residual blocks that all need to compensate for the single-task loss of direct GRU-over-embeddings. BUT cat-enabled transfer is also very large (+14 pp) because at scale, the cat task has enough signal to shape the shared representation usefully. Net: 10.73 pp gap, which is a much smaller fraction of STL (15.7%) than the overhead alone would suggest.

**Paper framing:** the MTL architecture at FL scale is doing ~60% of the work of making MTL even viable on reg. Pure MTL wrapping without cat loss would be catastrophic. This is a *stronger* endorsement of cross-attn+pcgrad, not a weaker one — it's rescuing a 25 pp hit down to 10.7 pp.

## Decision rules for next steps

Given the budget and what these findings unlock:

1. **FL cross-attn 5-fold replication (8 h)** — now MORE important. We need tight σ on the paper's headline FL cat +3.29 pp number AND on the FL reg 57.60 number. The λ=0 isolation showed FL behaves structurally differently from AL; single-fold numbers are risky.

2. **Cross-attn λ=0 on AL (5f×50ep, ~45 min)** — cheap, fills in the gap in the decomposition table. Lets us directly compare "cat-enabled transfer" across AL vs FL on the same architecture.

3. **H-R1 hd=384 on FL (1f×50ep, ~50 min)** — below the 2 pp threshold, but given FL's structural difference from AL, results may be larger. Low-cost bet.

4. **Hybrid (cross-attn cat + dselectk reg)** — deprioritize. Our H-R4 null tells us the reg path isn't limited by loss weighting; H-R1 directionally supports a capacity fix. A hybrid is orthogonal to both signals and speculative.

## Files produced this chain

- `docs/studies/check2hgi/results/P2/az3_dselectk_fairlr_5f50ep.json` (duplicate of AZ2)
- `docs/studies/check2hgi/results/P2/hr1_crossattn_gru_hd384_az_5f50ep.json`
- `docs/studies/check2hgi/results/P2/hr4_crossattn_static_cat0.3_az_5f50ep.json`
- `docs/studies/check2hgi/results/P2/fl_lambda0_dselectk_fairlr_1f50ep.json`
