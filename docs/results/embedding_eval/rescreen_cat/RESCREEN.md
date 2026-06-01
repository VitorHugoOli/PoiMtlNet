# Re-screen of dropped candidates — L0/L1 (5-fold, isolated placeid-folds)

All variants rebuilt via `rescreen_build.sh` (OUTPUT_DIR-scratch; frozen
`output/check2hgi/` md5-verified untouched). Compared to **`check2hgi_gcn_ctrl`**
(fresh GCN wd=0, same-protocol control). Frozen `check2hgi` shown for reference —
it sits ~+0.5pp above the fresh control (epoch/code/seed offset), which is exactly
why a same-protocol control was needed.

## next-cat — L1 probe acc (mean±SD, 5 folds)
| engine | alabama | arizona |
|---|---|---|
| check2hgi (frozen ref) | 0.9853±0.0037 | 0.9833±0.0019 |
| **check2hgi_gcn_ctrl** | 0.9804±0.0033 | 0.9802±0.0019 |
| check2hgi_v3c_wd05 | 0.9800±0.0034 | 0.9802±0.0019 |
| check2hgi_t24_dropedge | 0.9797±0.0035 | 0.9798±0.0016 |
| check2hgi_t43_sidefeat | 0.9803±0.0029 | 0.9800±0.0020 |
| check2hgi_t61_p2p | 0.9800±0.0041 | 0.9801±0.0021 |
| check2hgi_gat | 0.9296±0.0033 | 0.9459±0.0036 |
| check2hgi_rgcn | 0.9579±0.0031 | 0.9828±0.0014 |

## next-reg (region embeddings) — adj_coh@10 (L0) / probe acc@10 (L1)
| engine | AL adj_coh | AL probe@10 | AZ adj_coh | AZ probe@10 |
|---|---|---|---|---|
| **check2hgi_gcn_ctrl** | 0.2172±0.014 | 0.5472±0.009 | 0.2635±0.006 | 0.4859±0.005 |
| check2hgi_v3c_wd05 | 0.2063±0.015 | 0.5476±0.013 | 0.2641±0.005 | 0.4856±0.004 |
| check2hgi_t24_dropedge | 0.2156±0.014 | 0.5473±0.008 | 0.2498±0.007 | 0.4795±0.004 |
| **check2hgi_t43_sidefeat** | **0.2909±0.011** | **0.5635±0.003** | **0.3202±0.007** | **0.4946±0.003** |
| check2hgi_t61_p2p | 0.2235±0.014 | 0.5454±0.013 | 0.2356±0.007 | 0.4843±0.005 |
| check2hgi_gat | 0.2479±0.011 | 0.5582±0.007 | 0.2634±0.005 | 0.4866±0.004 |
| **check2hgi_rgcn** | **0.3090±0.013** | **0.5770±0.008** | **0.3318±0.007** | **0.4929±0.005** |

## Verdict (L0/L1 screen, AL+AZ vs control)
- **v3c (WD 5e-2): FALSIFIED on the region axis.** No gain over the clean control on cat OR reg at either state (reg adj_coh slightly below ctrl). The "region benefit" was an artifact of comparing to a differently-built baseline. ✗
- **T4.3 POI side-features: consistent small REGION gain** (adj_coh +0.05–0.07, probe@10 +0.9–1.6pp at both states; cat neutral). The original eval falsified it on a single-state *category* cell — but on the *region* axis it shows a real, two-state-consistent signal. → L2 confirmation warranted.
- **T3.3 R-GCN: biggest REGION gain** (adj_coh +0.07–0.09, probe@10 +0.7–3pp), without the catastrophic cat-leak in this static probe (cat 0.96–0.98). The relation-typed graph plausibly encodes region structure — but needs a leak-check (relation edges could leak adjacency) + L2. → L2 + leak audit.
- DropEdge, T6.1 p2p: no gain (≈ctrl). GATv2: cat hurt (no leak here, just worse). ✗
- **Methodology win:** the ladder on the *correct artifact* (region embeddings) with a *same-protocol control* found a region-axis signal (sidefeat, rgcn) that a single-MTL-metric eval missed — and killed v3c cleanly. These are L0/L1 proxies; **L2/L3 is the ranking authority** (see nextreg_stl.md protocol).
