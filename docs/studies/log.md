# docs/studies/ — Cross-study outcomes log

One line per study per closure or major direction shift. **Outcomes only.** Process narrative lives in each study's own `log.md`. Append-only.

| Date | Study | Outcome |
|---|---|---|
| 2026-05-04 | `hgi_category_injection` | **CLOSED** — AZ falsified; FL/CA/TX re-open standby. |
| 2026-05-06 | `merge_design` | **ACTIVE-CLOSING** — Designs A-M and Levers 1-6 saturated/falsified; Lever 5 orphaned (rescued by `substrate-protocol-cleanup` Tier B4 on 2026-05-28). |
| 2026-05-19 | `canonical_improvement` | **CLOSED** — Tier 1-6 substrate axis exhausted at ±0.8 pp; no further single-substrate ceiling. |
| 2026-05-24 | `mtl-protocol-fix` | **CLOSED v6 final** — F1 selector recovers ~95% substrate capacity at deploy (+5.6 pp FL); P4 identifies residual gap as architectural (NOT cat-vs-reg, NOT long-tail, NOT substrate). |
| 2026-05-24 | `mtl-protocol-fix` Phase 3 | **3 follow-ups** — log_T-KD §4.5 PROMOTED (+2-5 pp Wilcoxon-strict); class-balanced sampler §4.6 FALSIFIED (−18 to −30 pp); composite STL c2hgi+HGI §4.2 ESTABLISHED as current project headline (+7 to +12 pp). |
| 2026-05-16 | `mtl_improvement` | **LAUNCHED** — T0-T8 chain on branch `mtl-improve` (backbones, loss, batch, LR, α, heads, multi-seed champion); execution pending. |
| 2026-05-28 | `substrate-protocol-cleanup` | **LAUNCHED** — Tier A-D (log_T-KD multi-seed, Designs B/J/Lever 4/Lever 5 MTL+F1, 3-snapshot routing, freeze-reg-after-peak, K/V capacity-stealing pilot, window/mask audit); small-state only; ~40-45 GPU-h budget. |

## How to append

One row per closure or major direction shift. Format:

```
| YYYY-MM-DD | <study-name> | <Outcome in one sentence, lead with verb (CLOSED / PROMOTED / FALSIFIED / LAUNCHED / RE-OPENED)>. |
```

**Outcomes only.** Save the why/how for the study's own `log.md`. If you can't summarise the outcome in one sentence, the outcome isn't crisp enough to be logged yet.
