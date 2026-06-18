# Check2HGI Study — Issues & Flaws Log

Deep-dive audits of flaws, bugs, and methodological issues discovered during the study. Complements `CONCERNS.md` (lightweight index of open concerns) with full investigations.

Convention (mirrors `docs/studies/archive/fusion/issues/`):
- `<ISSUE_NAME>_AUDIT.md` — investigation + fix + verification
- `<ISSUE_NAME>_EXPLAINED.md` — optional deeper explanation for reviewers

## Index

| Issue | Severity | Detected | Status | Fix |
|-------|----------|----------|--------|-----|
| [MTL_PARAM_PARTITION_BUG](MTL_PARAM_PARTITION_BUG.md) | HIGH | 2026-04-22 (pre-P5 model/optim review) | CODE FIXED 2026-04-22 — **SUPERSEDED-BY `CONCERNS.md` C27** (off-board, not a freeze blocker) | Extended `task_specific_parameters` iterators on `MTLnet` (`adashare_logits`) and `MTLnetDSelectK` (`lora_A/B_*`, `skip_alpha_*`). Parameterised invariant test across 7 MTL variants (`tests/test_regression/test_mtl_param_partition.py`). 6 contaminated runs + optimizer probe launcher at `scripts/rerun_partition_bugfix.sh`. **C27 (2026-06-08) carries this forward to the dual-tower regime** — gradient-surgery balancers reweight only `shared_parameters()`, so champion-G's private reg tower trains at unit weight; the same `test_mtl_param_partition.py` guard now covers the dualtower family + PCGrad (per C28). Moot for G (cos(cat,reg)≈0 → no conflict to resolve). |
| [CROSSATTN_PARTIAL_FORWARD_CRASH](CROSSATTN_PARTIAL_FORWARD_CRASH.md) | MEDIUM | 2026-04-22 (pre-P5 model/optim review) | FIXED 2026-04-22 — **SUPERSEDED-BY the closing_data C1 reg-metric caveat + the dual-tower partial-forward test** (off-board, not a freeze blocker) | `MTLnetCrossAttn` now overrides both partial-forward methods with a deterministic zero-opposite-stream approximation (V=0 → cross-attention contribution from unused stream is zero; no pad-mask needed, which avoids softmax-over-all-masked NaN). Caveat documented; not bit-exact with `forward((cat, real_b))[idx]`. Regression test at `tests/test_regression/test_mtlnet_crossattn_partial_forward.py`. **Forward to:** the champion-G **dual-tower** has its own `next_forward` zero-A override (`src/models/mtl/mtlnet_crossattn_dualtower/model.py`); the closing_data **C1 (3-snapshot routing)** panel reports reg Acc@10 under this partial `next_forward` (zero-cat-stream), which is **not board-comparable** to the joint-forward `top10_acc_indist` — see `closing_data/FREEZE_READINESS.md §🟠 MAJOR · C1 absolute reg metric` + `C1_VERDICT.md`. The crash itself is fixed; the live concern is the metric caveat, not a bug. |
| [MODEL_DESIGN_REVIEW_2026-04-22](MODEL_DESIGN_REVIEW_2026-04-22.md) | LOW-MEDIUM | 2026-04-22 (pre-P5 model/optim review) | §3 + §4 FIXED 2026-04-22; **§1 SUPERSEDED-BY `CONCERNS.md` C28** (off-board, not a freeze blocker); others OPEN | §3 DSelectKLiteLayer docstring rewritten to acknowledge dense convex combination (not sparse top-k). §4 NextHeadSTAN default flipped to `bias_init="alibi"`. Items §2, §5, §6, §7, §9, §10 deferred to post-partition-rerun. **§1 (GETNext probe / α·log_T prior has no direct supervision — "is the prior doing material work?") is resolved by C28:** the prior + log_T-KD were structurally DEAD on champion-G's dual-tower (head missing from `_HEADS_REQUIRING_AUX_MTL`, FIXED 2026-06-12), and the now-real KD-on-G test (HANDOFF_AUDIT X2) is **NULL** (FL reg +0.05 / AL −0.13 ≪ 0.3 pp) → the prior contributes nothing on G; G pins prior-OFF + KD 0.0. |
| [FOLD_LEAKAGE_AUDIT](FOLD_LEAKAGE_AUDIT.md) | HIGH | 2026-04-17 (P2 critical review) | FIXED + VERIFIED | `src/data/folds.py::_create_single_task_folds` uses `StratifiedGroupKFold` for NEXT task |
| [REGION_HEAD_MISMATCH](REGION_HEAD_MISMATCH.md) | HIGH | 2026-04-16 (P1) | FIXED — +1 pp lift only — **SUPERSEDED-BY `CONCERNS.md` C04** (off-board, not a freeze blocker) | GRU preset + pad-mask re-zero in MTL forward (commit `b92fc62`). Reveals deeper BACKBONE_DILUTION issue. **C04 carries the framing forward:** the Transformer→GRU region-head swap is a like-for-like head-registry substitution within the MTLnet framework (head-task capacity mismatch, not an MTL-framework property); paper Methods framing committed. |
| [BACKBONE_DILUTION](BACKBONE_DILUTION.md) | HIGH | 2026-04-17 (P2-validate) | CHARACTERISED — **SUPERSEDED-BY `CONCERNS.md` C25/C28** (off-board, not a freeze blocker) | Shared backbone dilutes strong standalone heads (−8 pp GRU-on-region) while lifting weak ones (+40 pp Transformer-on-region). CH01 fails on AL. FL test pending. **C25 (CLOSED 2026-06-05) DISSOLVES the dilution story:** the MTL→STL reg gap was substantially the class-weighted-CE-vs-unweighted-ceiling confound — champion G (v16, dual-tower, unweighted CE) MATCHES the STL reg ceiling (Δ −0.09…−0.31) and BEATS the STL cat ceiling (+2.6…+4.1). **C28 (2026-06-12)** is the dual-tower code-correctness follow-up (aux-gate dead-codepath + the `test_mtl_param_partition.py` dualtower guard). |
| [GRADNORM_EXPERT_GATING](GRADNORM_EXPERT_GATING.md) | MEDIUM | 2026-04-17 (P2 screen) | WORKAROUND | Skip gradnorm × expert-gating combos until root-caused |

## Returning-agent status note (at the freeze)

These issues are **off-board / fixed — none is a live `closing_data` freeze blocker.** The SUPERSEDED-BY
stamps above map each to the `CONCERNS.md` entry (or the closing_data caveat) that carries it forward
under champion G / the v14 substrate, so a stale "OPEN"/"CHARACTERISED" label does not mislead. Verified
against `docs/CONCERNS.md` (C04/C25/C27/C28 all exist and match) on the 2026-06 freeze pass.
(`GRADNORM_EXPERT_GATING` and the deferred `MODEL_DESIGN_REVIEW` items §2/§5/§6/§7/§9/§10 are not on the
champion-G path and carry no freeze stamp.)

**`scripts/evaluate.py` on `mtlnet_crossattn` checkpoints (CROSSATTN_PARTIAL_FORWARD_CRASH) is a
pre-M2/M4 audit prereq, NOT a freeze gate.** The crash was FIXED 2026-04-22: `evaluate.py` routes MTL
through `m.cat_forward(x)` / `m.next_forward(x)`, both of which are overridden partial-forward methods —
`MTLnetCrossAttn` and, for champion G, the **dual-tower** `next_forward` zero-A override
(`src/models/mtl/mtlnet_crossattn_dualtower/model.py`). So evaluate.py is **not** on the board's eval
path as a crasher. The only residual is the **metric caveat** (C1's partial-`next_forward` reg Acc@10 is
not board-comparable to the joint-forward `top10_acc_indist`); confirm the dual-tower partial-forward
override + the metric caveat before the M2 (C1 routing panel) / M4 (A2/A4) audits, but it does not gate
the freeze.

## Related logs

- `../CONCERNS.md` — lightweight concerns index (C01–C28)
- `../CLAIMS_AND_HYPOTHESES.md` — every claim with its evidence + status
- `../HANDOFF.md` — session-level state
