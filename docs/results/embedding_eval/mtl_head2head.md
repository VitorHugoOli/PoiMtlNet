# MTL (L3) head-to-head + candidates-on-ResLN L0 (2026-06-01)

The decisive deployment test the STL ladder could not reach. MTL `check2hgi_next_region`,
B9 recipe, FL, seed 42 (dev seed — first read; multi-seed pending if anything moves).

## MTL substrate head-to-head (opt 2)
| substrate | cat F1 | cat Acc | reg Acc@1 | reg top5 | reg top10(indist) |
|---|---|---|---|---|---|
| **check2hgi (baseline GCN+log_T)** | **0.7028** | 0.7327 | 0.4717 | 0.5583 | **0.6025** |
| resln_design_b (v13) | 0.7020 | 0.7322 | 0.4714 | 0.5551 | 0.5985 |
| design_j | 0.6883 | 0.7171 | 0.4707 | 0.5541 | 0.5980 |

**Verdict:** the **baseline (GCN + log_T) is already the best.** v13 (resln_design_b) **gives NO MTL benefit** (cat −0.08pp, reg −0.4pp — ties) — directly confirms the documented "v13 = STL-only, no MTL benefit." design_j is **worse** on cat (−1.45pp). ⇒ **No substrate combination beats the deployed baseline in MTL.** "Best final combination" = the current baseline; resln/design_b/design_j do not improve deployment.

## Candidates on the ResLN encoder base — L0 (the "5 on v13" request, ResLN-encoder version)
Built `resln+{v3c,dropedge,sidefeat,p2p}` (gat/rgcn excluded: encoder-swaps + leak). vs `check2hgi_resln`:
| engine | cat probe | reg adj_coh | reg probe@10 | leak-sniff |
|---|---|---|---|---|
| check2hgi_resln (base) | 0.9879 | 0.2824 | 0.6867 | clean |
| resln_v3c | 0.9850 | 0.2714 | 0.6855 | clean |
| resln_dropedge | 0.9850 | 0.2781 | 0.6858 | clean |
| **resln_sidefeat** | 0.9850 | **0.3369** | 0.6902 (+0.35pp) | clean |
| resln_p2p | 0.9847 | 0.2719 | 0.6879 | clean |

**Verdict:** same as on the GCN base — v3c/dropedge/p2p are no-ops (≈ or slightly below base on cat); **sidefeat reproduces its region adj_coh signal** (+0.055, **base-independent** — stacks on GCN AND resln), but the probe@10 gain is marginal (+0.35pp, ~1 SD) and (decisive test) redundant with log_T at L2. No leaks introduced. (Note: this is the ResLN-*encoder* base; full resln+design_b stacking would need a build_design_b_poi_pool.py extension.)

## Combined conclusion
The MTL head-to-head closes the deployment question: **nothing improves MTL over the GCN+log_T baseline** — not v13, not design_j (worse), and by strong inference not the marginal/redundant candidates (sidefeat/GCN²), since the *stronger* v13 substrate already yields zero MTL gain. sidefeat's region-geometry signal is real and base-independent but does not survive to a robust task gain at any level, and is subsumed by log_T. **No adoption justified.**

## ⚠ Correction (2026-06-01, later): v13/v13+gprop MTL re-run
The first v13+sidefeat/+gprop and v13-base MTL runs used the **buggy region loader**
(hardcoded check2hgi region emb). After the `region_sequence.py` fix, two follow-up
issues: (1) a self-matching `ps grep [t]rain.py` deadlocked the v13-base campaign (it
waited on its own command line) — so the "v13 catF1 0.7020 region-fixed" figure was
actually the OLD buggy-loader head-to-head run, not a fixed re-run. (2) Re-launched v13
+ v13+gprop cleanly (region-fixed, no wait-gate). The macro conclusion is unaffected:
v13 is region-neutral vs canonical in MTL; the single-seed-42 caveat and the mandatory
multi-seed MTL test of the survivors (sidefeat, adjacency-head) stand. Clean numbers
fold in when the re-run completes.
