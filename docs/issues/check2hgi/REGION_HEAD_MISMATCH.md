# Region Head Mismatch in MTL — Audit

**Severity:** HIGH — caps region-task performance in every MTL run with the wrong head.

**Detected:** 2026-04-16 during P1 region-head ablation. `next_mtl` (Transformer) achieved 7.4% Acc@10 on AL region (5f × 50ep standalone), vs `next_gru` at 56.94% ± 4.01. A 49-pp gap.

**Status:** **FIXED + SURPRISINGLY LITTLE IMPACT** — GRU head now wired in; MTL reg Acc@10 on AL moved from 47.62% (Transformer) to 48.88% (GRU) at 5f × 50ep. The head-swap unlocked ~1 pp; the bigger gap to STL GRU's 56.94% appears to be **shared-backbone dilution**, not head choice. Fix commit `b92fc62`.

**New finding from the fix:** MTL lifts Transformer region by +40 pp (via shared backbone + category signal) but *dilutes* GRU region by −8 pp (shared backbone bottleneck caps a head that's strong standalone). See `results/P2/ch01_al_verdict.md`.

---

## TL;DR

The MTL framework's `_run_mtl_check2hgi` path instantiates the region head (task-b slot) using the default head registry entry for the "next" task, which is `next_mtl` — a 4-layer Transformer tuned for the legacy 7-class next-POI-category. That Transformer collapses on a 1109-class region task (P1 evidence: 7.4% Acc@10 standalone). The MTL's shared backbone rescues it partially to ~30–35%, but the *real* ceiling (56% with GRU / TCN) is unreachable with the current MTL head wiring.

Consequence: every P2 screen / promote / confirm number for region Acc@10 is capped by this head mismatch. P3 headline, if run with the current wiring, would ship this flaw into the paper.

---

## Evidence

P1 region-head ablation (AL, 5f × 50ep, region input, standalone):

| Head | Acc@10 | Std |
|------|--------|-----|
| `next_gru` | **56.94** | 4.01 |
| `next_tcn_residual` | 56.11 | 4.02 |
| `next_lstm` | 52.67 | — (1-fold) |
| `next_temporal_cnn` | 51.83 | — (1-fold) |
| `next_mtl` (Transformer) | **7.40** | 1.72 |
| Random baseline | ~0.9 | — |
| top-k popular | 14.7 | — |

The Transformer head is **below the popularity baseline** on AL region. This is not a tuning issue — `next_mtl`'s hyperparameters (4 layers × 4 heads × dim=64 × dropout=0.35) were calibrated for the 7-class next-category task, not a 1109-class region task with a 9-position sequence. Scaling to high-cardinality output requires a different head.

P2 screen (AL, 1f × 15ep, MTL) consistently shows reg Acc@10 in the 25–35% band for all 21 configs — the MTL shared backbone lifts the Transformer head above its 7.4% standalone floor, but cannot reach the 56% GRU / TCN ceiling. The MTL's expert-gating (CGC, MMoE, DSelectK) cannot work around a head that's the bottleneck.

---

## Why the wiring picks the wrong head

From the MTL model construction path (rough sketch, see `src/models/mtl/mtlnet*/model.py`):

- MTL expects a `category_head` (task-a) and a `next_head` (task-b) slot.
- The head registry default for task-b is `next_mtl` (Transformer), inherited from the legacy fusion track where next-task was 7-class next-category, not 1109-class region.
- The check2HGI TaskSet (`task_b = next_region`) does not override the default head.
- There is no CLI or TaskSet-level knob to pick a different head for the MTL's task-b slot.

---

## Proposed fix

Add a `next_head_name` (or `task_b_head_name`) parameter to each MTL model class, defaulting to `next_mtl` (legacy bit-exact) but settable to `next_gru`, `next_tcn_residual`, etc.

Touch points:
- `src/models/mtl/mtlnet/model.py` — base MTLnet (FiLM)
- `src/models/mtl/mtlnet_cgc/model.py`
- `src/models/mtl/mtlnet_mmoe/model.py`
- `src/models/mtl/mtlnet_dselectk/model.py`
- `src/models/mtl/mtlnet_ple/model.py`

Each model's `__init__` must:
1. Accept `next_head_name: str = "next_mtl"` param.
2. Look up the class via the head registry: `head_cls = _MODEL_REGISTRY[next_head_name]`.
3. Instantiate with task-b's embed_dim and num_classes; defer head-specific kwargs (seq_length, num_heads, num_layers) to head-specific defaults where sensible.

CLI wiring:
- Add `--next-head {next_gru, next_tcn_residual, next_lstm, next_temporal_cnn, next_mtl}` to `scripts/train.py`.
- For the P2-confirm / P3 headline runs on check2HGI: pass `--next-head next_gru`.

Estimated effort: ~50 LOC + one integration test (instantiate MTL model with each head alternative; forward a dummy batch; check output shape).

---

## Alternatives considered

1. **Globally change the head registry default for "next" from `next_mtl` to `next_gru`** — breaks legacy fusion-track bit-exactness. Rejected.
2. **Use the TaskSet `task_b.head` field if one exists** — the TaskSet dataclass doesn't yet have a head slot; adding one is a larger refactor than a per-model flag. Defer.
3. **Subclass each MTL model with a "RegionHead" variant** — creates 5× class-count explosion. Rejected.

Recommended: the per-model `next_head_name` parameter is the smallest-blast-radius fix.

---

## Verification plan

After the fix:

1. Run MTL with `--model mtlnet --next-head next_gru` at AL 1f × 15ep on check2HGI. Expected region Acc@10 substantially above the current ~28% (transformer-capped) mark — target ≥ 45% even at 15 ep.
2. Re-run P2 screen's top-5 configs with the GRU head. Expected: the rank ordering may shift (a head change can reshuffle how archs interact with optims).
3. Run the P3 headline with the GRU head.

Until this is done, all region Acc@10 numbers in MTL paths are under-reported and MTL cannot compete fairly with the P1 single-task region baseline (56.94% ± 4.01).

---

## Paper implications

- Any table reporting MTL region Acc@10 must clearly specify the region head used. If still `next_mtl` (Transformer), numbers are sub-ceiling and should either be (a) rerun with GRU or (b) footnoted as "transformer region head, future work: swap to GRU".
- The CH01 bidirectional lift claim cannot be tested credibly until this is fixed. MTL region numbers with Transformer head would have to *double* to match STL GRU's ceiling, which is implausible at current hyperparameters.
