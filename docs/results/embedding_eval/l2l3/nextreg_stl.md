# L2 — STL next-reg (region sequence task), same-protocol run

`scripts/p1_region_head_ablation.py --heads next_stan_flow --input-type region
--region-emb-source <engine> --folds 5 --epochs 50 --seed 42` at Florida. This is
the REAL region sequence task (region-embedding input + α·log_T prior), 5-fold CV.

| engine (region-emb source) | Acc@1 | Acc@10 | MRR |
|---|---|---|---|
| **hgi** | **0.4740±0.0053** | **0.7362±0.0043** | **0.5640±0.0047** |
| check2hgi (canonical) | 0.4687±0.0059 | 0.7274±0.0054 | 0.5575±0.0047 |
| check2hgi_resln | 0.4687±0.0048 | 0.7275±0.0054 | 0.5578±0.0044 |

## Verdict (answers "resln looked good on next-reg — confirm at L2")
- **resln ties canonical on L2 next-reg** (Acc@10 0.7275 vs 0.7274 — identical within ±0.005). resln's marginally higher L0 adjacency-coherence (0.282 vs 0.274) **does NOT translate** to a region sequence-task win. ⇒ **resln's value is the CATEGORY axis** (where it consistently leads L0/L1/L2 across FL/AL/AZ), **not region** — on region it is neutral vs canonical.
- **HGI wins next-reg** (+0.9 pp Acc@10, consistent across folds), concordant with the L0 adj_coh ranking (HGI top) and with RESULTS_TABLE §0.3/§0.5. The proxy correctly resolves the BIG gap (HGI vs family) but the small *within-family* adj_coh differences are below L2 resolution — matches the earlier "within-family region ordering is below proxy resolution" call.
- **Pipeline validation:** these same-protocol numbers (check2hgi 72.74, hgi 73.62 Acc@10) match §0.5 (stl_check2hgi 72.62, stl_hgi 73.58) — the L2 region run reproduces canonical numbers.

**Takeaway for a future MTL improvement:** resln is the category-axis substrate (best family member on cat, neutral on region); for region, the lever is HGI-style spatial/region-graph structure, not resln. A dual-axis MTL gain would need to *combine* check-in-category strength (resln/check2hgi) with region-graph structure (HGI) — the design_b/j attempts to inject this did not robustly help either axis (see region_eval/summary.md).
