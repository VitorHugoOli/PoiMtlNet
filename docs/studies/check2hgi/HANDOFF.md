# Check2HGI Study — Session Handoff (2026-04-16)

## Status at a glance

| Phase | Status |
|---|---|
| **P0** | ✅ complete — integrity, simple baselines, CH14 audit |
| **P1** | ❌ next — region-head validation + head ablation |
| P2 | blocked on P1 + MTL arch TaskSet port |
| P3 | blocked on P2 |
| P4 | blocked on P3 |
| P5 | gated on P4 |

## Task pair (CORRECTED 2026-04-16)

**{next_category (7 classes), next_region (~1K classes)}** — NOT next_poi (11K classes). The scope confusion from earlier sessions is resolved. Original plan (commit 727c426) was correct all along.

## Key numbers

| Metric | AL | FL |
|---|---|---|
| Check2HGI single-task next-cat macro-F1 | **38.67%** (5f×50ep) | TBD |
| POI-RGNN published (FL/CA/TX) | 31.8–34.5% | 34.5% |
| Simple-baseline floor: majority next-cat | 34.2% | 24.7% |
| Simple-baseline floor: Markov next-cat | 31.7% | 37.2% |
| Simple-baseline floor: Markov next-region Acc@10 | 21.3% | 45.9% |

## Pre-requisites for P1

1. Region head needs testing — all 5 heads available in the registry.
2. Just run `scripts/train.py --state alabama --engine check2hgi --task next_region` with different `--model-param` for each head variant (pending: `--task next_region` single-task path needs wiring if not already there).

## Pre-requisites for P2

1. **Parameterise CGC/MMoE/DSelectK/PLE** with TaskSet — ~150 LOC × 4 variants. Same pattern as the base MTLnet parameterisation from P1-b.
2. Research alternative MTL approaches that suit {next_cat, next_region} better than fusion-era designs.

## Commits on this branch

```
d173a85 fix(check2hgi): scope correction — task pair is {next_category, next_region}
44975c8 docs(check2hgi): P0.6 + P0.7 audit docs
98c91a7 feat(check2hgi): P0.3g OOD-restricted Acc@K
aebc20c feat(check2hgi): P0.5 simple baselines
e38e063 feat(check2hgi): P0.3 code deltas (next_poi — now removed)
e6ecb1b refactor(check2hgi): rescope standalone + review findings
... (earlier commits: MTLnet/runner parameterisation, embedding generation, study setup)
```

## Open decisions

1. **Multi-seed for P3:** 3 seeds × 5 folds = n=15 paired samples. Cost: ~3× more compute on P3 only.
2. **Class-weighted CE for FL:** `--use-class-weights` is wired; enable for all FL runs.
3. **P5 gate threshold:** CH03 needs ≥ 2pp next-category F1 lift on FL to trigger cross-attention work.
