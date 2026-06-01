# next-reg on REGION embeddings — region_eval.py

Corrected next-reg screen: probes `region_embeddings.parquet` (the artifact the
task actually consumes via `--task-b-input-type region`), NOT the final embedding.

- **L0 adj_coh@10** = fraction of each region's cosine-kNN that are graph-adjacent (train-free; uses check2hgi `region_adjacency`).
- **L1 probe** = 1-step transition linear probe: region_emb[current] → next region. acc@k. Mean±SD over seeds {0,1,7,100}.
- HGI region embeddings regenerated 2026-05-31 at canonical config (lr 0.006 / warmup 40 / 2000 ep); region partition verified aligned with check2hgi (counts match AL 1109, AZ 1547, FL 4703; adj_coh ≫ random confirms positional alignment).

## Florida (4703 regions)
| engine | adj_coh@10 | acc@1 | acc@5 | acc@10 |
|---|---|---|---|---|
| **hgi** | **0.326** | 0.482 | 0.629 | 0.676 |
| check2hgi | 0.274 | 0.479 | 0.634 | 0.685 |
| check2hgi_design_b | 0.240 | 0.483 | 0.633 | 0.681 |
| check2hgi_resln | 0.282 | 0.480 | 0.635 | 0.687 |
| check2hgi_resln_design_b | 0.231 | 0.482 | 0.633 | 0.680 |

## Alabama (1109 regions)
| engine | adj_coh@10 | acc@1 | acc@5 | acc@10 |
|---|---|---|---|---|
| **hgi** | **0.375** | 0.297 | 0.497 | 0.573 |
| check2hgi | 0.263 | 0.281 | 0.477 | 0.561 |
| check2hgi_design_b | 0.315 | 0.295 | 0.488 | 0.565 |
| check2hgi_resln | 0.273 | 0.282 | 0.478 | 0.561 |
| check2hgi_resln_design_b | 0.319 | 0.296 | 0.491 | 0.569 |

## Arizona (1547 regions)
| engine | adj_coh@10 | acc@1 | acc@5 | acc@10 |
|---|---|---|---|---|
| **hgi** | **0.393** | 0.279 | 0.418 | 0.480 |
| check2hgi | 0.312 | 0.278 | 0.422 | 0.489 |
| check2hgi_design_b | 0.338 | 0.281 | 0.427 | 0.489 |
| check2hgi_resln | 0.312 | 0.281 | 0.422 | 0.490 |
| check2hgi_resln_design_b | 0.346 | 0.283 | 0.426 | 0.490 |

## Verdict
- **L0 adjacency-coherence reproduces HGI's region win at ALL THREE states** (HGI tops adj_coh: FL 0.326, AL 0.375, AZ 0.393 — clearly above the check2hgi family). This is **concordant with the real STL §0.3**, where HGI wins next-reg Acc@10 everywhere (AL 61.9>59.2, AZ 53.4>50.2, FL 71.3>69.2). ⇒ The cheap *structural* proxy on the *correct artifact* (region embeddings) correctly predicts the substrate ranking for region.
- **The 1-step transition probe does NOT track it** (mixed/near-tie: FL/AZ check2hgi marginally ahead on acc@10, AL HGI ahead). It is too crude — self-transition rate 0.30–0.50, no 9-window, no log_T prior — to resolve the ~2–3 pp gap. ⇒ Among cheap proxies, only adjacency-coherence carries region signal; the full ranking is an L2/L3 verdict.
- Concordance call: **adj_coh is a valid SCREEN for region embeddings** (3/3 states agree with L2/L3 direction); the transition probe is not. HGI's edge is *spatial-structural* (hierarchical region graph), exactly what adjacency-coherence measures.
