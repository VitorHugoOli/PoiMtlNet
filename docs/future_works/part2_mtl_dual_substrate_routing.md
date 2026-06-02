# Future Work — Part 2: MTL dual-substrate routing on the v14 base

**Date drafted:** 2026-06-02
**Source:** `docs/studies/embedding_eval/FINAL_SYNTHESIS.md` (Part-1 closed). Sibling memo:
[`composite_two_substrate_engine.md`](composite_two_substrate_engine.md) (same idea, earlier framing).

## Why this exists

Part-1 (embedding_eval) closed the **substrate** axis:
- **v14 = `check2hgi_design_k_resln_mae_l0_1`** is the dual-axis champion: next-cat 67.36 (≈
  frozen-canon, ≫ HGI) + next-reg 0.7024 (leak-free multi-seed FL; closes ~78 % of the
  canonical→HGI gap; −0.36pp residual to HGI).
- **But the substrate gain does NOT survive MTL** (2-fold seed42 pilot: v14 ≈ canonical in MTL,
  cat −0.21pp / reg-Acc +0.03pp) — reproduces the documented "v13 = STL-only, no MTL benefit"
  regime finding. So **more substrate work will not move MTL deployment.**

The residual next-reg gap to HGI (−0.36pp FL STL; larger at small states) and the MTL regime
limit are **head/fusion problems**, not substrate problems. Part-2 attacks them.

## The lever: dual-substrate task-routing

The one log_T-orthogonal, unfalsified lever (per the embedding_eval advisor + the regime
finding): in MTL, **route HGI's region tower to the reg head while the cat head uses the v14
substrate.** Distinct from Design-A late-fusion concat (−18pp AL, dead): this routes each task to
its strongest substrate's region/checkin embedding via SEPARATE towers, not a concatenated input.

```
cat head  (next_category) ← v14 (check2hgi_design_k_resln_mae) check-in embedding   [v14 wins cat]
reg head  (next_region)   ← HGI region embedding (or v14's Delaunay region emb)      [HGI wins reg]
shared MTL backbone + cross-attn as in B9
```

Open question Part-2 answers: does routing the HGI region tower into the reg head (a) recover
HGI's STL reg lead inside MTL, and (b) without hurting cat — OR does the MTL regime wash it out as
it washed out the substrate gains?

## Pilot hook (implemented 2026-06-02)

`src/data/folds.py` `_resolve_x`: env-var **`REGION_EMB_ENGINE`** overrides which engine's
`region_embeddings.parquet` the reg (task_b region) slot consumes, while `--engine` still drives
the cat slot. Unset → canonical (reg uses same engine as cat). Pilot:

```bash
REGION_EMB_ENGINE=hgi python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state florida --engine check2hgi_design_k_resln_mae_l0_1 --seed 42 \
    --folds 2 --epochs 50 ... (B9 recipe) ... \
    --per-fold-transition-dir output/check2hgi/florida --log-t-kd-weight 0.0
# cat ← v14 ; reg region emb ← HGI. Compare to v14-only and canonical MTL pilots.
```
(For a production Part-2 study, promote the env-var to a proper `--region-emb-engine` CLI flag
threaded through `ExperimentConfig`.)

## Sequencing / cost

1. **Routing pilot (2-fold seed42 FL)** — running now; signal on whether routing beats v14-only MTL.
2. If positive → **full 5-fold multi-seed {0,1,7,100} + AL/AZ + CA/TX(1f) on GPU** (MTL B9 is
   ~20h/run on CPU — GPU mandatory; RunPod/Lightning per `docs/infra/`).
3. Alternatives if routing washes out: reg-head architecture sweep
   ([`reg_head_architecture_sweep.md`](reg_head_architecture_sweep.md)), substrate-adaptive MTL
   balancing ([`substrate_adaptive_mtl_balancing.md`](substrate_adaptive_mtl_balancing.md)).

## Acceptance criterion

Routing promoted only if MTL reg (Acc@10, leak-free seeded log_T, multi-seed) beats the v14-only
and canonical MTL baselines by > 1 fold-σ at FL, with no cat regression > 0.5pp. Same dev-seed /
leak-free / provenance discipline as Part-1.
