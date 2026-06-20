# Windowing & input-build correctness audit — 2026-06-20

> `study/pre-freeze-a40`. Full-flow audit of the next/next-region input build (multi-agent workflow
> `wvr1k2c9d`, 4 independent audits + adversarial verify, on 367+ synthetic windows + real alabama
> 12 709 rows). Triggered by the CA box-RAM OOM — the question was "is the RAM blowup a windowing bug,
> and does the windowing logic have other flaws?" **Answer: the windowing MATH is sound; the RAM blowup
> is purely the `<U32` upcast (see `OOM_MEMORY_FIX.md`); the real findings are alignment guards + a
> stride-1 label-distribution skew.**

## Headline verdict
**Windowing math is correct on every axis** (verified, not asserted):
- **Target = chronologically-next check-in** — `core.py` `target_idx = start_idx + window_size` (NOT `start+window-1`). No off-by-one.
- **No within-history leak** — the OOB/tail branch removes the demoted target from its own history before emission. The 22 % of alabama rows where the target also appears in history are *genuine user revisits* (valid next-POI samples), not a leak.
- **No cross-user leak** — every window is built per-user (`groupby('userid')`); no window spans two users.
- **Region label correct** — `region_idx` = the **target's** region; `last_region_idx` = region of the last non-pad history POI (0/367 chronology violations). Canonical (reverse-argmax) and probe (`valid[-1]`) produce identical labels on all 12 709 alabama rows.
- **Windowing is LINEAR (O(n))**, not O(n²): stride-1 ≈ (n−1) windows/user; non-overlap = n/9. The ~8.5× row multiplier of stride-1 fully explains the RAM blowup — it is memory *inefficiency* (the `<U32` upcast), **not** a correctness/complexity bug.

## ⚠ M1 — stride-1 tail-window label skew (the ONE finding that touches board NUMBERS — DECISION NEEDED)
At **stride=1 (the adopted overlap board)**, every user emits a run of **tail windows** (the sliding window
walks off the end of the user's history) in addition to the normal full windows. The OOB/tail branch
demotes the last POI to the target, so for a user with `n` check-ins there are up to ~8 tail windows whose
**target is the user's LAST POI** — i.e. the last POI is over-represented as a prediction target, and those
tail windows carry mostly-padding histories (down to a single real token).

- **It is leak-free and chronologically valid** (the target always post-dates its history). This is NOT a leak.
- **But it skews the label distribution** toward "predict the end-of-history POI" and injects trivial
  near-all-padding samples — a *data-quality* / representativeness concern for the overlap board, not a bug.
- **Why it matters now:** the P3 board adopts stride-1 overlap. At stride-9 (the frozen default) the effect is
  negligible (few windows/user); at stride-1 it is ~8 tail windows/user.

**Options for the P3 board build (USER/board decision — NOT changed in code):**
1. **Gate the tail branch for stride-1** (`emit_tail=False` when stride==1) — drops the ~8 OOB tail
   windows/user, removing the skew *and* shaving ~8 rows/user off the RAM multiplier. Changes board numbers.
2. **Keep tail windows, document the skew** — they add short-context training signal; accept the
   end-of-history weighting as a property of the overlap recipe.

This is deliberately left for the freeze decision because it changes board numbers. Flag to the user before launching the P3 board.

## Fixed in this branch (byte-identical / robustness — safe to ship)
| id | file | fix |
|---|---|---|
| **C1** | `src/data/folds.py` | next↔next_region alignment was guarded by **row count only**. Now asserts **userid content-equality row-for-row** at load (a stale region file with matching row count but different per-row users would otherwise silently mis-pair every row). Verified no false-positive on alabama. |
| **M2** | `src/data/inputs/builders.py` (×2) | `sort_values(['userid','datetime'])` used unstable quicksort → non-deterministic "next" under same-timestamp ties. Now `kind='mergesort'`. Byte-identical where no ties (alabama: 0). |
| **M3** | `scripts/mtl_improvement/build_overlap_probe_engine.py` | target_poi→idx map had no OOV guard (NaN→garbage negative int64 → wrong/cryptic region). Now fails loud (mirrors the last_region path). |
| **M6** | `src/data/inputs/next_region.py` | corrected the misleading "shares identical feature tensors" docstring — the copied X cols are **vestigial** at MTL train time (each slot picks its own X); kept only for schema/row-alignment. |

## Deferred / documented (not changed)
- **M5** — `build_overlap_probe_engine.py` keeps an extra `next_category` column vs the canonical `next_region.py` drop (schema divergence; harmless today, align if the probe artifact is ever consumed where schema parity is assumed).
- **M7** — non-overlap (stride-9) default emits a fixed-phase subset (~n/9) of each user's transition pairs. This is the documented "non-overlapping" semantics, not a bug; relevant when reasoning about what the frozen §0.1 inputs cover.
- **C2** — `core.py` maps a missing target category (`None`) to class 7 (via the CATEGORIES_MAP inversion), which can inflate class 7 for genuinely-missing categories. Collapses into C1 on real data (no NaN labels reach the loader); revisit per-state if a non-alabama state has missing categories.

## Provenance
Workflow `wvr1k2c9d` (windowing correctness) + `w6kfedte3` (RAM fix) — both 2026-06-20, both converge on
"windowing is sound; the blowup is the `<U32` upcast." RAM fix shipped as `eb45c744`; correctness fixes as
`ea7a57d3`.
