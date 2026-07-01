# Future work — Overlapping windows (a validated, head-independent ceiling cap)

> ⚠ **STATUS SUPERSEDED — overlap WAS ADOPTED (2026-06-19/21).** The `closing_data` pre-freeze gates validated overlap (PR #28 ADOPT verdict: training-length confound refuted, leak clean — `docs/studies/log.md` 2026-06-19) and gated stride-1 overlap was **adopted unconditionally** (log.md 2026-06-21). The champion/board engine is now `check2hgi_dk_ovl` (stride-1, tail-gated, MIN_SEQ=10); every `RESULTS_BOARD §1` number is overlap-based. The "keeps the non-overlapping canon" decision below described the pre-closing_data state — only the frozen v11 §0.1 paper canon remains non-overlapping. Kept for the finding + validation record.

**Status:** validated end-to-end (isolated harness + real STL + real MTL) at AL, 2026-06-03.
**Decision:** documented as a key finding + future-work; the study **keeps the non-overlapping canon**
for internal consistency (frozen (c)/(d), MTL board, per-fold log_T, and the v11 paper canon are all
non-overlapping). NOT adopted — adoption would require a multi-state substrate + log_T + ceiling +
board rebuild and would change every number in the paper.

## The finding
`src/data/inputs/core.py::generate_sequences` defaults `stride = window_size` (=9) → **non-overlapping**
windows. A user with N check-ins yields ~N/9 training sequences instead of ~N. On Alabama: 113,846
check-ins → **only 12,709 sequences** (the "~12,709 samples" everywhere in the study is *windows*, not
check-ins). Stride=1 (overlapping) → ~108,073, a **7.5–8.4× increase** in (history→next) supervision.
Non-overlapping windowing is **not** standard for next-POI/next-cat (STAN/GETNext/DeepMove/Flashback use
stride 1). The cap is **head-independent** — which is exactly why none of the Tier-S head swaps could
surface it.

**Leak-safe.** The CV split is `StratifiedGroupKFold(groups=userid)`; all of a user's windows land in one
fold → overlapping windows cannot cross the train/val boundary. Overlap → correlated val rows within a
held-out user (wider σ, a variance effect, NOT upward bias). Overlapping windows are genuinely different
prediction pairs (shifted history + different target), not duplicates.

## Validated results (AL, real pipeline)
| metric | non-overlap | overlap (stride=1) | lift |
|---|---|---|---|
| STL cat (next_gru, logit-adjust τ=0.5) | 49.97 | 59.74 | **+9.77** |
| STL reg v14 (next_stan ≡ α=0) | 62.88 | 68.01 | **+5.13** |
| STL reg HGI (next_stan, HGI region emb) | 63.58 | 68.47 | **+4.89** |
| MTL cat joint / disjoint | 46.30 / 46.52 | 55.21 / 55.90 | **+8.92 / +9.39** |
| MTL reg joint / disjoint | 54.54 / 53.47 | 55.05 / 54.46 | **+0.50 / +1.00** |

Scale-dependent at STL (cat): AL (small, 12.7k) +9.77; **FL (large, 159k) +1.30** — at FL the model is
near data-saturation. Harness-isolated control reproduced the frozen ceilings within ~1pp, and the real
pipeline confirmed the lift, so the effect is robust to the harness.

## Why this STRENGTHENS the regime finding (the important part)
- **Cat = rising tide.** Overlap lifts cat ~equally in STL (+9.8) and MTL (+8.9) — the cat MTL pathway
  exploits the extra data.
- **Reg = the STL→MTL gap WIDENS.** Overlap lifts STL reg (+5.1) but the MTL reg barely moves (+0.5 joint
  / +1.0 disjoint). The STL→MTL reg gap goes **8.34 → 12.96** (joint) with overlap. The shared-backbone
  reg pathway — the architectural bottleneck — **cannot absorb the extra data** while STL reg fully does.
  More data makes the bottleneck *more* visible, which is direct evidence FOR the dual-tower motivation.

So the windowing experiment is not a threat to the study's thesis — it sharpens it (for reg) and lifts cat
across the board.

## How to run / reproduce (isolated, frozen substrate untouched)
- Stride is threaded (backward-compatible) through `generate_sequences` /
  `convert_user_checkins_to_sequences` / `generate_next_input_from_checkins`. Region-seq builders are
  engine-aware (`seq_engine`) in `region_sequence.py`, `p1_region_head_ablation.py`, and the MTL fold
  creator (`src/data/folds.py`).
- Isolated probe engine `check2hgi_dk_ovl` = v14 embeddings re-windowed stride=1; embeddings/region
  symlinked from v14. Build: `scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1`.
- STL harness (no disk writes): `scripts/mtl_improvement/overlap_probe.py <state> <cat|reg> 9,1`.
- Real STL: `train.py --task next --engine check2hgi_dk_ovl --model next_gru --logit-adjust-tau 0.5`;
  `p1 --heads next_stan --target region --engine-override check2hgi_dk_ovl --region-emb-source <v14|hgi>`.
- Real MTL: `train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_dk_ovl ... --log-t-kd-weight 0.0`.

## If adopted later (the rebuild checklist)
1. Default `stride=1` (or a compute-compromise 2–3) in the real input pipeline; rebuild `next.parquet` /
   `sequences_next.parquet` / `next_region.parquet` at all states. (Also consider lowering
   `MIN_SEQUENCE_LENGTH=5` → 2; it currently drops 58% of AL users / 3.65% of check-ins.)
2. Rebuild the per-fold log_T from the overlapping next_region (windowing-dependent).
3. Re-freeze (c)/(d) ceilings; re-run the MTL board at all states; re-pin the paper canon (§0.1).
4. Expect: large STL/MTL cat lifts + STL reg lifts at small states (AL/AZ/GE), small at large (FL/CA/TX);
   the MTL reg gap widens (report it — it is a headline, not a regression).

## Open / caveats
- Validated at **AL only, single seed**; the MTL test used a **KD-off, prior-free-reg** recipe (control
  uses the same recipe → the windowing delta is valid; absolutes are not board-comparable). Confirm the
  cat-rising-tide / reg-gap-widens pattern at AZ/GE/FL + multi-seed before any paper-level claim.
- FL STL cat lift was only +1.3 (data-saturation) — the headroom is concentrated at small/middle states.
