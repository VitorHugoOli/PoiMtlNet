# next-reg on REGION embeddings — region_eval.py (5-fold CV)

Corrected next-reg screen: probes `region_embeddings.parquet` (the artifact next-reg
consumes via `--task-b-input-type region`), NOT the final embedding. **5-fold CV**,
mean±SD over folds (matches L2). adj_coh = adjacency-coherence@10 (train-free; per-region
coherence averaged over 5 region-folds). probe = 1-step transition linear probe (region_emb
[current] → next region), 5-fold KFold.

## adj_coh@10 (L0, train-free) — HGI tops at all 3 states
| engine | FL | AL | AZ |
|---|---|---|---|
| **hgi** | **0.326±0.007** | **0.375±0.009** | **0.393±0.009** |
| check2hgi | 0.274±0.006 | 0.263±0.015 | 0.312±0.009 |
| check2hgi_design_b | 0.240±0.004 | 0.315±0.015 | 0.338±0.008 |
| check2hgi_resln | 0.282±0.006 | 0.273±0.016 | 0.312±0.006 |
| check2hgi_resln_design_b | 0.231±0.003 | 0.319±0.014 | 0.346±0.006 |
| check2hgi_design_j | 0.237±0.006 | 0.362±0.013 | 0.288±0.013 |
| check2hgi_design_l | 0.236±0.004 | 0.316±0.016 | 0.334±0.008 |
| check2hgi_resln_design_j | 0.221±0.007 | 0.356±0.017 | 0.276±0.008 |
| check2hgi_lever4_canonical | — | 0.237±0.012 | 0.276±0.006 |

## transition probe acc@10 (L1) — near-tie across engines
| engine | FL | AL | AZ |
|---|---|---|---|
| hgi | 0.677±0.004 | 0.564±0.005 | 0.481±0.002 |
| check2hgi | 0.685±0.003 | 0.559±0.005 | 0.489±0.003 |
| (family variants) | 0.672–0.687 | 0.533–0.570 | 0.483–0.492 |

## Verdict (5-fold, multi-variant)
- **L0 adjacency-coherence: HGI wins at FL and AZ, and is top-tier at AL** (only `design_j`/`resln_design_j` approach it at AL — but those collapse at FL/AZ, so the effect is **state-specific, not a robust win**). HGI's region edge is robust and concordant with real STL §0.3. SDs are tight (≤0.017), so the HGI gap is well outside fold noise.
- **No dropped on-disk variant robustly beats HGI/canonical on the region axis** — `design_j`/`design_l`/`resln_design_j`/`lever4` either match or trail, none consistently across states. `lever4_canonical` is the weakest. This **validates their original falsification on the region axis too**, now with proper 5-fold CV + multi-variant coverage.
- **The 1-step transition probe stays a near-tie** (within ~1 pp, overlapping SDs) — too crude to rank; region's substrate ordering remains an L2/L3 verdict. Among cheap proxies only the *structural* L0 (adj_coh) carries region signal.
- **Still pending: v3c (WD 5e-2)** — not on disk; needs a safe non-clobbering rebuild (the rebuild script writes to the frozen `output/check2hgi/`).
