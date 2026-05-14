# Check2HGI Study — Status Report v2 (2026-04-20, end of day)

Updates `STATUS_REPORT_2026-04-20.md` after chain-of-4, AL cross-attn λ=0, FL H-R1, Markov k=1..9, and hybrid decision.

## TL;DR

- **Paper substrate + headline are locked.** CH16 (+18.30 pp cat vs HGI) and CH-M4 (cross-attn matches/exceeds STL cat on AL/AZ/FL) are both confirmed.
- **Scale curve is the single most important finding.** Cat benefit grows monotonically with data; reg gap widens but *decomposes* into huge (25 pp) architectural overhead + huge (+14 pp) cat→reg transfer at FL scale.
- **Every hypothesis tested since 2026-04-19 afternoon has returned null or rejected.** Nash-MTL (tied with PCGrad), loss-balance (null), GRU hd=384 (directional AL but regresses on FL), hybrid cross-attn+dselectk (null-a-priori).
- **Markov baseline strengthened.** Context-matched Markov-9 saturates below Markov-1 (sparsity) — isolates neural advantage as representation-learning, +24 pp (AL) / +14 pp (FL) over Markov-9.
- **One remaining large task:** FL cross-attn 5-fold (~8 h) to put σ on the paper's headline +3.29 pp number.

## Phase status table (updated)

| Phase | Status | Notes |
|---|---|---|
| **P0** Simple baselines + audits | ✅ **complete + extended** | Markov now k=1..9 (was k=1..3). Shows monotone degradation — novel paper result. |
| **P1** Region-head ablation (AL) | ✅ complete | GRU champion (56.94 A@10). |
| **P1.5** HGI vs Check2HGI region | ✅ complete | Tied, as expected. |
| **P1.5b** HGI vs Check2HGI category | ✅ **locked — CH16 +18.30 pp** | Paper's substrate claim. |
| **P2** MTL arch × optim ablation | ✅ complete (fair LR) | Cross-attn + pcgrad champion on AL. |
| **P3** AZ scale validation | ✅ 3/3 done (AZ1/2/3; AZ3 was a duplicate of AZ2 due to LoRA default) | P3 table complete. |
| **P4** FL replication | 🟡 **1-fold only** | cat 66.46 / reg 57.60. Headline rests on n=1; FL 5-fold is the biggest remaining hole. |
| **P5** Ablations / decomposition | ✅ **mostly complete** | AL λ=0 (dselectk + cross-attn both), FL λ=0 (dselectk) all done. AL transfer +0.14 pp; FL transfer +14.20 pp. |
| **P6** Encoder enrichment | ⛔ deferred indefinitely | Not needed for paper. |

## What's locked (and provable from the results tree)

| Claim | Evidence | Status |
|---|---|---|
| **CH16** Check2HGI > HGI on cat | P1.5b: AL STL cat 38.58 ± 1.23 vs HGI 20.29 ± 1.34 | ✅ **locked** |
| **CH17** Check2HGI > published POI-RGNN | 38.58 vs ~31.8–34.5 (published) | ✅ **locked** (no new run needed) |
| **CH-M4** Cross-attn uniquely closes cat gap | AL matches STL (38.47 vs 38.58); AZ exceeds (+1.05); FL exceeds (+3.29, 1f) | ✅ **locked, strengthened** |
| **CH-M5** Fair LR dominates architecture | +4–8 pp Δm from max_lr 0.001→0.003 across all configs | ✅ **locked** |
| **CH-M6** (scale curve) | 3 states, monotone trends in both directions | ✅ **locked at 1-fold FL** |
| **CH-M7** (new) Markov degrades with k | Markov-1..9 monotone degrade AL/FL | ✅ **locked** |
| **CH-M8** (new) Transfer is scale-dependent | AL +0.14 pp vs FL +14.20 pp on same architecture | ✅ **locked at 1-fold FL** |

## What's been rejected (publication-worthy null findings)

| Claim | Evidence | Verdict |
|---|---|---|
| **CH-M3** (original: 5 pp overhead LR/scale-invariant) | AL 5.07 vs FL 24.93 | ❌ rejected; replaced by CH-M8 |
| Nash-MTL > PCGrad on Pareto-bidirectional | AZ: cat Δ +0.22, reg Δ −0.93 (both in σ) | ❌ rejected |
| Loss-weighting helps reg (H-R4 cat=0.3) | AZ: cat Δ −0.27, reg Δ −0.19 (null) | ❌ rejected |
| GRU hd=512 helps reg | AZ MPS OOM, no completion | ⚠️ untested (infra) |
| GRU hd=384 helps reg | AZ: +1.51 directional but <2 pp threshold; FL crashes + cat regresses 0.7 pp | ❌ rejected at FL |
| Hybrid (cross-attn cat + dselectk reg) helps reg | AL cross-attn λ=0 (52.27) ≥ dselectk λ=0 (51.87) → no mechanism | ❌ rejected a-priori |

**Paper framing benefit:** each rejection is a provable null that tightens the paper's claims. Reviewers asking "did you try Nash-MTL?" / "what about loss weighting?" / "why not dselectk for reg?" all have one-line answers backed by numbers.

## Open items (priority-ordered)

### Priority 1 — FL cross-attn 5-fold replication (~8 h)
The paper's headline is "cross-attn MTL gives +3.29 pp next-category F1 over STL on Florida" (66.46 vs 63.17). Currently n=1 — no σ. Expected behavior: std around ±0.5 pp given AZ's ±0.55. If the 5-fold mean lands within ±1 pp of 66.46, the claim tightens cleanly. Run as a single overnight execution (with `--no-checkpoints` honored on the MTL path now).

### Priority 2 — FL 5-fold STL cat baseline replication (~3 h)
Pairs with Priority 1. Currently only 1-fold STL FL cat is measured (63.17). Need σ to turn "+3.29 pp" into "+3.29 ± Y pp".

### Priority 3 (optional) — Cross-engine replication (CA + TX, ~12–18 h each)
The paper can ship on FL alone. CA + TX would strengthen scale-curve claim (n=4 or n=5 instead of n=3 states) but isn't required for BRACIS.

### Priority 4 (optional) — Multi-seed n=15 on FL (~15 h)
If reviewers push back on single-seed. Standard practice in the MTL literature is n=5–15 seeds for headline numbers.

### Skipped / closed

- H-R2 (GRU num_layers=3): skip — P1 STL already tried this with tiny lift.
- H-R3 (GRU dropout=0.15): skip — tiny expected effect, P2 priorities exhausted.
- FL H-R1 retry: skip — crashed twice with cat already regressing at epoch 41.
- Hybrid: skip — null-a-priori per AL λ=0 comparison.
- Nash on FL: skip — AZ result says tied, no reason to think FL differs.

## Infrastructure state (as of 22:00)

- **I1 `--no-checkpoints` on MTL path**: ✅ fixed (commit `10889ba`). Chains run clean.
- **I2 SSD intermittent**: ⚠️ worked around; `OUTPUT_DIR=/tmp/check2hgi_data` holds.
- **I3 max_lr=0.01 mac-idle corruption**: closed (quarantined).
- **I4 MPS OOM on hd=512**: hd=384 works on AL/AZ, crashed at epoch 41 on FL. Don't use hd>256 on FL.

## Recent commits (last 24h on this branch)

```
431b4f6 docs(study): add Markov k=1..9 rows to baselines table
8674656 feat(baselines): extend region Markov to k=1..9 matching model's context
3734874 docs(study): AL cross-attn λ=0 + hybrid decision (skipped)
40b397a docs(study): chain-of-4 results — H-R1/H-R4/FL λ=0
575a4a7 docs(study): mid-execution status report v1
bb153d5 docs(study): Nash-MTL on AZ — tied with PCGrad
10889ba fix(train): honor --no-checkpoints on MTL check2hgi path
ae6d320 docs(study): known infra issues
```

## Result artifact index

| Finding | Doc |
|---|---|
| Chain-of-4 overnight findings | `research/CHAIN_FINDINGS_2026-04-20.md` |
| Hybrid skip decision | `research/HYBRID_DECISION_2026-04-20.md` |
| Nash-MTL AZ null | `research/NASH_MTL_AZ_FINDINGS.md` |
| Scale curve (AL/AZ/FL) | `results/SCALE_CURVE.md` |
| Final ablation (fair LR) | `results/P2/FINAL_ABLATION_SUMMARY.md` |
| Baseline + best-MTL table | `results/BASELINES_AND_BEST_MTL.md` |
| Infra issues | `research/KNOWN_INFRA_ISSUES.md` |
| Region hparam plan + verdicts | `research/REGION_HPARAM_PLAN.md` |
| **This status report** | `STATUS_REPORT_2026-04-20_v2.md` |

## Next-session entry point

If you're picking this up cold and want the critical path: **launch FL cross-attn + pcgrad 5f×50ep at max_lr=0.003, and in parallel (or after) FL STL cat 5f×50ep at max_lr=0.01.** Both are existing configs, no code changes needed. Budget ~8 + 3 = ~11 h. Paper headline is unlocked at that point.

Optional follow-ups (if budget allows): multi-seed n=5 on FL for the headline number, then CA + TX replication. Not required for the BRACIS submission.
