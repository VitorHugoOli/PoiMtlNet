# v18 — PROVENANCE

> Generated 2026-08-09T06:30:14.493296+00:00. Every rundir behind every number, with the recipe and the commit it was produced from.

## Recipe (identical across all cells except where a probe says otherwise)

```
engine   check2hgi_v18 (forward-only graph + 4 elapsed-time node cols, in_channels 15)
repr     seed 42, 500 epochs, resln, dim 64, 2 layers -- one per state, seed-independent
-- APPROVED RECIPE 2026-08-09 (FINAL_SETTINGS.md). The pre-2026-08-09 cells used
-- class-weighted CE, cw 0.75 and bs2048 at AL/IST; those are SUPERSEDED and live in
-- docs/results/closing_data/v18_superseded_oldrecipe/. Per-cell truth: the `recipe`
-- field on each sidecar, reproduced in the table below.
joint    train.py --task mtl --canon none --task-set check2hgi_next_region
         bs 8192, static_weight category-weight 0.50, logit-adjust-tau 0.5 (CAT HEAD
         ONLY -- mtl_cv.py:500 keeps the region criterion on plain CE), onecycle
         max-lr 3e-3, cat-lr 1e-3 (AL/AZ/IST) / 2e-3 (FL/CA/TX), reg-lr 3e-3,
         shared-lr 1e-3, MTL_ONECYCLE_PER_HEAD_LR=1, cat-head next_gru,
         reg-head next_stan_flow_dualtower, geom_simple, fp32, --compile --tf32
cat      train.py --task next --model next_gru --embedding-dim 64, bs 8192 ALL states,
         max-lr 0.0025 (AL) / 0.0005 (AZ, IST) / 0.005 (FL, CA, TX),
         logit-adjust-tau 0.5 -- REPLACES class weighting (next_cv.py:123-141)
reg      p1_region_head_ablation.py --heads next_stan_flow --input-type region
         --region-emb-source check2hgi_design_k_resln_mae_l0_1 --max-lr 0.003
         logit-adjust-tau 0 (OFF -- it is Bayes-consistent for BALANCED error, so it
         significantly HURTS Acc@10: AL -1.841 p=0.0002, IST -2.749 p<0.0001, while
         macro-F1 rises. Region reports Acc@10.) UNCHANGED by the 2026-08-09 recipe
         change, which is why those 10 sidecars were kept, not regenerated.
```

| state | seed | family | rundir | pid | commit |
|---|---:|---|---|---|---|
| alabama | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/next_lr1.0e-04_bs8192_ep50_20260809_061500_710881` | 710881 | `e351d4b0` |
| alabama | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_alabama_region_5f_50ep_v18_alabama_reg_s0.json` | — | `e351d4b0` |
| alabama | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_alabama_region_5f_50ep_v18_alabama_reg_s1.json` | — | `e351d4b0` |
| arizona | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_arizona_region_5f_50ep_v18_arizona_reg_s0.json` | — | `e351d4b0` |
| arizona | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_arizona_region_5f_50ep_v18_arizona_reg_s1.json` | — | `e351d4b0` |
| california | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_california_region_5f_50ep_v18_california_reg_s0.json` | — | `e351d4b0` |
| florida | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_florida_region_5f_50ep_v18_florida_reg_s0.json` | — | `e351d4b0` |
| florida | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_florida_region_5f_50ep_v18_florida_reg_s1.json` | — | `e351d4b0` |
| istanbul | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/next_lr1.0e-04_bs8192_ep50_20260809_061502_710882` | 710882 | `e351d4b0` |
| istanbul | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_istanbul_region_5f_50ep_v18_istanbul_reg_s0.json` | — | `e351d4b0` |
| istanbul | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_istanbul_region_5f_50ep_v18_istanbul_reg_s1.json` | — | `e351d4b0` |
| texas | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_texas_region_5f_50ep_v18_texas_reg_s0.json` | — | `e351d4b0` |
