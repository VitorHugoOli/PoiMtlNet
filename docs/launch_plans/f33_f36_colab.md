# N6 — F33 + F36 Colab launch plan

**Date:** 2026-04-28
**Goal:** Stage a clean Colab handoff for the next two paper-blocking experiments. After SSD is restored + user reviews, can be launched immediately.

---

## F33 — FL 5f × 50ep B3+`next_gru` decisive test

### Why
- C14 (cat-head scale-dependence flag) is the open concern. Path A (universal `next_gru`) is currently in NORTH_STAR; F32 1-fold FL flipped sign (−0.93 pp). F33 is the n=5 decisive test.
- If F33 cat F1 5f-mean falls within σ-envelope of pre-F27 FL points (0.6623 - 0.6706) → Path A confirmed.
- If F33 cat F1 5f-mean below envelope → Path B (scale-dependent: `next_gru` for AL/AZ, `next_mtl` for FL/CA/TX) → headline table needs footnote.

### Config (B3 predecessor recipe with cat-head swapped to `next_gru`)
```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --folds 5 --epochs 50 --batch-size 1024 \
    --max-lr 3e-3 --scheduler one_cycle \
    --state florida \
    --output-dir results/check2hgi/florida/f33_b3_gru_5f50ep
```
Note: batch-size **1024** (not 2048) on FL — F32 confirmed 2048 OOMs at fold 2 ep 23 even on Colab T4 due to FL's 4702 regions.

### Cost
~6h Colab T4. Use the detached-subprocess pattern documented in `docs/COLAB_GUIDE.md` (mandatory for runs > 5 min — cell timeouts SIGINT foreground commands).

### Decision rule (from FOLLOWUPS_TRACKER.md)
- **Pass:** cat F1 5f-mean within σ-envelope of {0.6623, 0.6706} (F32 sigma + 0.005 buffer) → **Path A locked**, no headline table changes.
- **Fail:** cat F1 5f-mean below envelope → **Path B** → headline table footnote: "FL cat head uses `next_mtl` (scale-dependent), AL/AZ use `next_gru`."

### Acceptance bookkeeping
- Update `FOLLOWUPS_TRACKER.md §F33` row to "done DATE" with the path verdict.
- Update `CONCERNS.md §C14` to "resolved DATE" with chosen path.
- If Path B: update `NORTH_STAR.md §H3-alt recipe` to note FL exception.

---

## F36 — FL Phase-2 substrate grid

### Why
- Phase-1 closed at AL+AZ with 8/8 head-invariant probes. Phase-2 replicates the substrate-comparison grid at FL.
- 4 cells: substrate-only linear probe, cat STL × 2 (Check2HGI + HGI), reg STL × 2, MTL counterfactual (HGI substrate).

### Config — 4 cells
1. **F36a** Substrate-only linear probe (head-free)
   ```bash
   python scripts/study/run_substrate_probe.py \
       --state florida --substrate check2hgi --epochs 50 --folds 5
   python scripts/study/run_substrate_probe.py \
       --state florida --substrate hgi --epochs 50 --folds 5
   ```

2. **F36b** Cat STL `next_gru` × 2 substrates
   ```bash
   python scripts/run_stl_next_gru_cat.sh florida check2hgi 5 50
   python scripts/run_stl_next_gru_cat.sh florida hgi 5 50
   ```

3. **F36c** Reg STL `next_getnext_hard` × 2 substrates
   ```bash
   python scripts/run_stl_next_getnext_hard_reg.sh florida check2hgi 5 50
   python scripts/run_stl_next_getnext_hard_reg.sh florida hgi 5 50
   ```

4. **F36d** MTL B3 counterfactual (HGI substrate, all else equal)
   ```bash
   python scripts/train.py --task mtl --task-set check2hgi_next_region \
       --substrate hgi \
       --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
       --cat-head next_gru --reg-head next_getnext_hard \
       --folds 5 --epochs 50 --batch-size 1024 \
       --scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
       --state florida
   ```

### Cost
- F36a: ~30 min Colab T4 (linear probe is cheap)
- F36b: ~1h × 2 = 2h
- F36c: ~1h × 2 = 2h (reg head is heavier but FL has many regions)
- F36d: ~1h
- **Total: ~5–6 h Colab T4** — fits in one Colab session.

### Acceptance criteria (from PHASE2_TRACKER.md §6)
- CH16 cross-state: cat F1 paired Wilcoxon p<0.05 between C2HGI and HGI on FL (Path A target: same direction as AL+AZ).
- CH18 substrate-specific: F36d cat F1 + reg Acc@10 must show ≥10 pp degradation vs MTL+C2HGI (current FL H3-alt: cat 67.92, reg 71.96).
- CH15 reframing: matched-head TOST non-inferiority on reg (margin δ=2 pp).

---

## Pre-launch checklist

Before kicking off either experiment:

- [ ] Verify SSD permission restored (current blocker — see `MORNING_BRIEFING.md`).
- [ ] Verify FL data on disk: `ls /Volumes/Vitor's\ SSD/.../output/check2hgi/florida/` shows embeddings + region transitions + inputs.
- [ ] Verify 4050 GPU also free if planning to parallelise F37 (P1+P2) on the same overnight.
- [ ] Check `docs/COLAB_GUIDE.md §detached-subprocess pattern` (mandatory for >5min runs).
- [ ] Have `notebooks/colab_check2hgi_mtl.ipynb` open in Colab; clone fresh repo / git pull.
- [ ] Confirm `STATE = "florida"` in the notebook config cell.

## Post-completion checklist

- [ ] Tarball results: `tar -czf results_f33_f36_$(date +%F).tar.gz results/check2hgi/florida/f33_b3_gru_5f50ep results/check2hgi/florida/f36_*`.
- [ ] Copy back to SSD: `scp` or rsync — boot vol → SSD.
- [ ] Update FOLLOWUPS_TRACKER.md §F33, §F36 with done DATE + per-fold means.
- [ ] Append cells to `results/RESULTS_TABLE.md` Phase-2 section.
- [ ] Run paired Wilcoxon stats via `scripts/study/paired_test.py` (or extend `scripts/analysis/p4_p5_paired_wilcoxon.py`).

---

## Risk notes

1. **FL OOM at batch=2048** (C09 / SSD reliability). Use bs=1024 throughout.
2. **Colab T4 ~12h max session** — F33 (6h) + F36 (5-6h) might exceed one session. Suggest splitting across two days OR using two Colab tabs in parallel.
3. **Detached subprocess required** — foreground `!{cmd}` will SIGINT if MCP/cell timeout fires.
4. **F37 (P1+P2) on 4050 in parallel** — independent machine, no conflict.
