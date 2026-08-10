# v18 — PROVENANCE

> Generated 2026-08-10T21:48:38.673504+00:00. Every rundir behind every number, with the recipe and the commit it was produced from.

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
| alabama | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260809_062021_712015` | 712015 | `dafdc74d` |
| alabama | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/next_lr1.0e-04_bs8192_ep50_20260809_061500_710881` | 710881 | `dafdc74d` |
| alabama | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_alabama_region_5f_50ep_v18_alabama_reg_s0.json` | — | `dafdc74d` |
| alabama | 1 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260809_234634_882153` | 882153 | `dafdc74d` |
| alabama | 1 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/next_lr1.0e-04_bs8192_ep50_20260809_234110_881088` | 881088 | `dafdc74d` |
| alabama | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_alabama_region_5f_50ep_v18_alabama_reg_s1.json` | — | `dafdc74d` |
| alabama | 7 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260810_043258_136` | 136 | `dafdc74d` |
| alabama | 7 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/next_lr1.0e-04_bs8192_ep50_20260810_043251_134` | 134 | `dafdc74d` |
| alabama | 7 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_alabama_region_5f_50ep_v18_alabama_reg_s7.json` | — | `dafdc74d` |
| alabama | 100 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260810_061258_103` | 103 | `dafdc74d` |
| alabama | 100 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/alabama/next_lr1.0e-04_bs8192_ep50_20260810_051925_72` | 72 | `dafdc74d` |
| alabama | 100 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_alabama_region_5f_50ep_v18_alabama_reg_s100.json` | — | `dafdc74d` |
| arizona | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260809_070421_718456` | 718456 | `dafdc74d` |
| arizona | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/next_lr1.0e-04_bs8192_ep50_20260809_065925_717493` | 717493 | `dafdc74d` |
| arizona | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_arizona_region_5f_50ep_v18_arizona_reg_s0.json` | — | `dafdc74d` |
| arizona | 1 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260810_003044_887833` | 887833 | `dafdc74d` |
| arizona | 1 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/next_lr1.0e-04_bs8192_ep50_20260810_002547_886880` | 886880 | `dafdc74d` |
| arizona | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_arizona_region_5f_50ep_v18_arizona_reg_s1.json` | — | `dafdc74d` |
| arizona | 7 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260810_150831_98` | 98 | `dafdc74d` |
| arizona | 7 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/next_lr1.0e-04_bs8192_ep50_20260810_150829_99` | 99 | `dafdc74d` |
| arizona | 7 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_arizona_region_5f_50ep_v18_arizona_reg_s7.json` | — | `dafdc74d` |
| arizona | 100 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260810_162415_95` | 95 | `dafdc74d` |
| arizona | 100 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/arizona/next_lr1.0e-04_bs8192_ep50_20260810_162414_92` | 92 | `dafdc74d` |
| arizona | 100 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_arizona_region_5f_50ep_v18_arizona_reg_s100.json` | — | `dafdc74d` |
| california | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/california/mtlnet_lr1.0e-04_bs8192_ep50_20260809_185016_843391` | 843391 | `dafdc74d` |
| california | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/california/next_lr1.0e-04_bs8192_ep50_20260809_174021_821539` | 821539 | `dafdc74d` |
| california | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_california_region_5f_50ep_v18_california_reg_s0.json` | — | `dafdc74d` |
| california | 1 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/california/mtlnet_lr1.0e-04_bs8192_ep50_20260810_152217_1060899` | 1060899 | `dafdc74d` |
| california | 1 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/california/next_lr1.0e-04_bs8192_ep50_20260810_124703_1019205` | 1019205 | `dafdc74d` |
| california | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_california_region_5f_50ep_v18_california_reg_s1.json` | — | `dafdc74d` |
| california | 7 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/california/mtlnet_lr1.0e-04_bs8192_ep50_20260810_135832_72` | 72 | `dafdc74d` |
| california | 100 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/california/mtlnet_lr1.0e-04_bs8192_ep50_20260810_162607_74` | 74 | `dafdc74d` |
| florida | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260809_075343_731514` | 731514 | `dafdc74d` |
| florida | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/next_lr1.0e-04_bs8192_ep50_20260809_072320_720702` | 720702 | `dafdc74d` |
| florida | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_florida_region_5f_50ep_v18_florida_reg_s0.json` | — | `dafdc74d` |
| florida | 1 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260810_012014_900489` | 900489 | `dafdc74d` |
| florida | 1 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/next_lr1.0e-04_bs8192_ep50_20260810_004946_890326` | 890326 | `dafdc74d` |
| florida | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_florida_region_5f_50ep_v18_florida_reg_s1.json` | — | `dafdc74d` |
| florida | 7 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260810_070923_60` | 60 | `dafdc74d` |
| florida | 7 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/next_lr1.0e-04_bs8192_ep50_20260810_132852_113` | 113 | `dafdc74d` |
| florida | 7 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_florida_region_5f_50ep_v18_florida_reg_s7.json` | — | `dafdc74d` |
| florida | 100 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260810_070901_60` | 60 | `dafdc74d` |
| florida | 100 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/florida/next_lr1.0e-04_bs8192_ep50_20260810_133145_488` | 488 | `dafdc74d` |
| florida | 100 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_florida_region_5f_50ep_v18_florida_reg_s100.json` | — | `dafdc74d` |
| istanbul | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260809_062824_713692` | 713692 | `dafdc74d` |
| istanbul | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/next_lr1.0e-04_bs8192_ep50_20260809_061502_710882` | 710882 | `dafdc74d` |
| istanbul | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_istanbul_region_5f_50ep_v18_istanbul_reg_s0.json` | — | `dafdc74d` |
| istanbul | 1 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260809_235441_883226` | 883226 | `dafdc74d` |
| istanbul | 1 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/next_lr1.0e-04_bs8192_ep50_20260809_234113_881087` | 881087 | `dafdc74d` |
| istanbul | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_istanbul_region_5f_50ep_v18_istanbul_reg_s1.json` | — | `dafdc74d` |
| istanbul | 7 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260810_162444_90` | 90 | `dafdc74d` |
| istanbul | 7 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/next_lr1.0e-04_bs8192_ep50_20260810_061136_73` | 73 | `dafdc74d` |
| istanbul | 7 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_istanbul_region_5f_50ep_v18_istanbul_reg_s7.json` | — | `dafdc74d` |
| istanbul | 100 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260810_162501_100` | 100 | `dafdc74d` |
| istanbul | 100 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/istanbul/next_lr1.0e-04_bs8192_ep50_20260810_162457_97` | 97 | `dafdc74d` |
| istanbul | 100 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_istanbul_region_5f_50ep_v18_istanbul_reg_s100.json` | — | `dafdc74d` |
| texas | 0 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/texas/mtlnet_lr1.0e-04_bs8192_ep50_20260809_112624_771463` | 771463 | `dafdc74d` |
| texas | 0 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/texas/next_lr1.0e-04_bs8192_ep50_20260809_095453_746546` | 746546 | `dafdc74d` |
| texas | 0 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_texas_region_5f_50ep_v18_texas_reg_s0.json` | — | `dafdc74d` |
| texas | 1 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/texas/mtlnet_lr1.0e-04_bs8192_ep50_20260810_063214_968091` | 968091 | `dafdc74d` |
| texas | 1 | cat | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/texas/next_lr1.0e-04_bs8192_ep50_20260810_032240_916753` | 916753 | `dafdc74d` |
| texas | 1 | reg | `/home/vitor.oliveira/PoiMtlNet/docs/results/P1/region_head_texas_region_5f_50ep_v18_texas_reg_s1.json` | — | `dafdc74d` |
| texas | 7 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/texas/mtlnet_lr1.0e-04_bs8192_ep50_20260810_070235_71` | 71 | `dafdc74d` |
| texas | 100 | joint | `/home/vitor.oliveira/PoiMtlNet/results/check2hgi_v18/texas/mtlnet_lr1.0e-04_bs8192_ep50_20260810_135751_72` | 72 | `dafdc74d` |
