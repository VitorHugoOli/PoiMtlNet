# v18 — PROVENANCE

> Generated 2026-08-06T23:04:46.730403+00:00. Every rundir behind every number, with the recipe and the commit it was produced from.

## Recipe (identical across all cells except where a probe says otherwise)

```
engine   check2hgi_v18 (forward-only graph + 4 elapsed-time node cols, in_channels 15)
repr     seed 42, 500 epochs, resln, dim 64, 2 layers -- one per state, seed-independent
joint    train.py --task mtl --canon none --task-set check2hgi_next_region
         bs 8192, static_weight cw 0.75, onecycle max-lr 3e-3, cat-lr 1e-3, reg-lr 3e-3,
         shared-lr 1e-3, MTL_ONECYCLE_PER_HEAD_LR=1, cat-head next_gru,
         reg-head next_stan_flow_dualtower, geom_simple, fp32, --compile --tf32
cat      train.py --task next --model next_gru --embedding-dim 64, per-state bs/lr
reg      p1_region_head_ablation.py --heads next_stan_flow --input-type region
         --region-emb-source check2hgi_design_k_resln_mae_l0_1 --max-lr 0.003
```

| state | seed | family | rundir | pid | commit |
|---|---:|---|---|---|---|
| alabama | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260806_152940_617690` | 617690 | `9240da4f` |
| alabama | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/next_lr1.0e-04_bs2048_ep50_20260806_152616_616433` | 616433 | `9240da4f` |
| alabama | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_alabama_region_5f_50ep_v18_alabama_reg_s0.json` | — | `9240da4f` |
| arizona | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260806_161634_625368` | 625368 | `9240da4f` |
| arizona | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/next_lr1.0e-04_bs8192_ep50_20260806_161027_623811` | 623811 | `9240da4f` |
| arizona | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_arizona_region_5f_50ep_v18_arizona_reg_s0.json` | — | `9240da4f` |
| florida | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260806_173530_649170` | 649170 | `9240da4f` |
| florida | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/next_lr1.0e-04_bs8192_ep50_20260806_163558_627960` | 627960 | `9240da4f` |
| florida | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_florida_region_5f_50ep_v18_florida_reg_s0.json` | — | `9240da4f` |
| istanbul | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260806_154211_620510` | 620510 | `9240da4f` |
| istanbul | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/next_lr1.0e-04_bs2048_ep50_20260806_152619_616431` | 616431 | `9240da4f` |
| istanbul | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_istanbul_region_5f_50ep_v18_istanbul_reg_s0.json` | — | `9240da4f` |
