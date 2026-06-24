# HANDOFF — A40 (CUDA) · **CTLE-SC + Check2HGI-SC comparand for FL · CA · TX · Istanbul**

> The Mac (M4 Pro, 24 GB MPS) board run (2026-06-24, `study/board-m2pro`, see
> [`docs/results/closing_data/MACS_BOARD_RESULTS.md`](../../results/closing_data/MACS_BOARD_RESULTS.md))
> completed AL + AZ CTLE-SC and HMT-GRN for all 6 states, but **could not run CTLE-SC for the large states
> (FL/CA/TX) or Istanbul** — they exceed 24 GB unified memory. This handoff is exactly those missing cells.
> **1 seed (0) × 5 folds**, fp32, device-internal. Read §4 (problems) BEFORE running — they cost the Mac run hours.

## 0 · MISSING PIECES (what the A40 must produce)
| Cell | State | Why it's here, not on the Mac |
|---|---|---|
| **CTLE-SC** (cat next_gru + reg next_stan_flow, checkin-modality) | FL, CA, TX | 1.27M–3.5M overlapping rows × 2 heads × 50 ep × 5 folds; the input build + per-fold staging OOM'd 24 GB (FL CTLE-SC watchdog-killed at 14 % free *solo*). |
| **Check2HGI-SC comparand** (matched STL heads on `check2hgi_dk_ovl`) | FL, CA, TX | `compute_region_transition` loads the full N×576 embedding matrix (CA = **12.6 GB**) → `MemoryError` on 24 GB. This is the *exact* step that failed on the Mac. |
| **CTLE-SC + comparand** | Istanbul | needs a 5-fold CTLE substrate built (MLM pretrain) + matched heads on the Phase-V substrate. |
| *(optional, recommended)* **HMT-GRN re-run on CUDA** | all 6 | the Mac HMT numbers are M4/MPS-labeled and **device-confounded** vs the CUDA MTL champion (HMT learns embeddings from scratch → float trajectory diverges ~5 pp; AL reg M4 57.05 vs recorded 62.37). For the official "we-beat-SOTA-on-region" row, run HMT on the same CUDA basis as the champion. |

Already done (do NOT redo): **AL, AZ** CTLE-SC + comparand + HMT (committed, validated — comparand reproduced recorded CUDA within noise: AL cat 55.59≈55.72, reg 61.86≈61.94). HMT-GRN AL/AZ/CA/FL/TX/Istanbul exist on the Mac as a labeled set.

## 1 · CODE — you MUST have the Mac-run fixes (or the cells break the way they broke before)
The A40 lane (`study/board-a40`) does **NOT** contain the fixes below. **Check out / merge `study/board-m2pro` @ `711aeb59`** (pushed to origin). Seven fixes are load-bearing here:
1. **`f421f1fb` p1 reg-head collate** — without it the comparand reg head crashes immediately (`stack [B,9,64] vs [B]`): `_dataloader` must pass `collate_fn=_batched_collate` (the `__getitems__` perf fix added 2026-06-24 needs it). **This blocks every reg run.**
2. **`b195b89f` min_seq=10** — `mac_baseline_compare.build_inputs` must pin `min_sequence_length=10` so CTLE-SC and the comparand share the identical row set + fold partition (else the W3 Δ is invalid).
3. **C-1/C-2/C-3 reg fixes** (already in main, verify present): comparand reg is **checkin-modality** (`run_reg` → `--input-type checkin`), and the per-fold log_T is **built per-engine** (`compute_region_transition --engine <e> --per-fold`), NOT a copy of the canonical stride-9 prior.
4. **`comparand_check2hgi_sc.py`** (new tool) — produces the Check2HGI-SC reference.
5. **`b9073396` fsq_tree** — only needed if you REBUILD Istanbul ETL (you shouldn't; it's staged — see §2).
6. **b3 OOM fixes** (`a27b6e1e`,`68d304f4`,`cf292c3d`,`250f003a`) — only needed if you run HMT on CUDA; harmless otherwise. (Side benefit: they fix a latent OOM in the eval for any wide-region state.)

Also pull **main** for the merged **`perf(mtl)` 4.5× wide-head fix (#33)** — without it TX/CA wide-head (6553/8501 regions) training is ~70× slower.

## 2 · DATA CHECK — verify on the A40 BEFORE running (the box reclaimed its disk; durable copy = the SSD)
The A40 box's local disk was reclaimed ("all regennable"). Transfer from the SSD (`/Volumes/Vitor's SSD/ingred/output`) — or rebuild from v14. Per state **{florida, california, texas}** confirm:

| Artifact | Path | Check |
|---|---|---|
| v14 substrate | `output/check2hgi_design_k_resln_mae_l0_1/<s>/{embeddings,region_embeddings,poi_embeddings}.parquet` | present (FL 3.3G, CA 969M, TX 1.2G). **Gate** — `build_overlap_probe_engine` needs it. |
| CTLE per-fold cells | `output/board_baselines/ctle/<s>/s0_f{0..4}/embeddings.parquet` | **5 folds**; each dir has `LEAK_MARKER.txt` reading `TRAIN-ONLY per fold` (verify — leak-safety gate). FL 8.3G / CA 18G / TX 23G. |
| check2hgi graph maps | `output/check2hgi/<s>/` (graph maps + region/poi embeddings) | present (FL/CA/TX 4–4.7G). Needed for input build + region symlinks. |

**Istanbul** (staged by the Mac run, reuse — do NOT rebuild ETL): `output/check2hgi/istanbul/` (graph `temp/checkin_graph.pt` = 29945 POI / 520 region, Phase-V substrate `embeddings.parquet`, `input/next.parquet` = 58,297 rows, `temp/sequences_next.parquet`, `input/next_region_labels.parquet`, 25 per-fold priors) + `data/massive_steps_istanbul/` (parquets + `category_map.csv` 580/580). **Verify** `pickle.load(checkin_graph.pt)['num_pois']==29945, num_regions==520` (the Phase-V alignment guard).

Disk: the dk_ovl builds are LARGE — CA `next.parquet` 9.5 GB + `next_region.parquet` 7.6 GB ≈ 17 GB; TX ≈ 22 GB. Keep ≥30 GB free per large state. (`next_region.parquet` wastefully carries the 576 embedding cols — it's `next_df.copy()`+region cols; harmless, just disk.)

## 3 · COMMANDS (per state; seed 0 × 5 folds)
```bash
cd <A40 repo>; export PYTHONPATH=src    # MTL_RAM_HEADROOM_GB=2 default; raise if the box is tight
S=florida   # then california, then texas

# (A) CTLE-SC — self-contained (builds its own check2hgi_ctle inputs at min_seq=10,
#     stages per-fold leak-clean ctle emb, builds per-engine per-fold log_T):
python scripts/closing_data/mac_baseline_compare.py --state $S --baseline ctle \
    --cells-root output --folds 5 --heads cat reg
#  -> docs/results/closing_data/baseline_compare/${S}_ctle.json

# (B) Check2HGI-SC comparand — needs the dk_ovl base + per-fold log_T FIRST:
python scripts/mtl_improvement/build_overlap_probe_engine.py $S 1
python scripts/compute_region_transition.py --state $S --engine check2hgi_dk_ovl \
    --per-fold --seed 0 --n-splits 5
python scripts/closing_data/comparand_check2hgi_sc.py --state $S --folds 5 --heads cat reg
#  -> docs/results/closing_data/baseline_compare/${S}_check2hgi_sc.json
```
**Istanbul CTLE-SC** (base = the Phase-V substrate, `--engine check2hgi`, set-a windowing — **NOT** dk_ovl overlap, per HANDOFF_BOARD_MACS §4):
```bash
# build the per-fold leak-clean CTLE substrate for istanbul (engine = check2hgi base):
for f in 0 1 2 3 4; do python scripts/baselines/build_ctle_substrate.py --state istanbul --fold $f --stride 1; done
# then matched heads (cat next_gru + reg next_stan_flow checkin-modality) on the ctle engine,
# comparand = the Istanbul champion-G (cat +8.06 / reg −0.58 over Markov; STL reg ceiling 70.4, Markov floor 52.5).
# (Adapt mac_baseline_compare/comparand to engine check2hgi + the ctle-istanbul cells; report gap-to-ceiling, not Acc@k.)
```
**(optional) HMT-GRN on CUDA** (device-consistent vs champion): `python scripts/baselines/b3_hmt_grn.py --state <s> --engine check2hgi_dk_ovl --seed 0 --folds 5 --epochs 50` (US) / `--engine check2hgi` (Istanbul).

## 4 · PROBLEMS I HIT ON THE MAC — pre-empt them here
1. **reg-head collate crash** — guaranteed if the A40 code predates `f421f1fb`. Symptom: `RuntimeError: stack expects each tensor to be equal size [B,9,64] vs [B]` at the first reg batch. Fix = pull `study/board-m2pro` (§1). **Smoke first**: `p1_region_head_ablation.py --engine check2hgi_dk_ovl --state alabama --only-fold 0 --heads next_stan_flow --input-type checkin --epochs 2` should train, not crash.
2. **`compute_region_transition` MemoryError** — it calls `load_next_data` which materialises N×576 float32 (CA 12.6 GB). On the A40 ensure host RAM ≥ ~16 GB free for that step (it's CPU-resident, guarded by `_guard_cpu_resident_ram` + `MTL_RAM_HEADROOM_GB`). This was the step that killed the Mac comparand; on the A40 it should just work — but it's a **host-RAM** need, not GPU.
3. **min_seq desync (silent invalidation)** — if CTLE-SC builds at min_seq=5 but the comparand base at min_seq=10, the two are scored on different rows/folds and the Δ is meaningless. The committed code pins both to 10; verify `len(check2hgi_ctle/<s>/input/next.parquet) == len(check2hgi_dk_ovl/<s>/input/next.parquet)` before trusting a cell.
4. **C-1/C-2/C-3 reg validity** — the comparand reg MUST be checkin-modality. **Sanity gate**: comparand reg should be checkin-scale and *substrate-sensitive* — i.e. `CTLE-SC reg ≠ Check2HGI-SC reg` (on the Mac AL: 62.23 vs 61.86). If they're bit-identical across baselines, or land at ~53–70 % region-modality scale, C-1/C-2 regressed — STOP.
5. **wide-head speed** — TX/CA reg head is 6553/8501-wide; without the `perf(mtl)` #33 fix, training is ~70× slower. Pull main.
6. **CA kNN OOM-reboot** (original board §3) — that was the *MPS embedding-quality kNN eval* on CA's 169k POIs, NOT CTLE-SC training. Irrelevant to these cells, but don't co-schedule a CA embedding-eval on a memory-tight box.
7. **build_next_region_for is slow** — a Python loop over N rows (2.7–3.5 M for CA/TX) → minutes, not seconds. Not a hang.
8. **fold-split / leak-safety** — all paths are train-only per fold (vocab, prior, OOD set, StratifiedGroupKFold(groups=userid, y=next_category, seed 0)). The b3 `build_fold_split` assumes `load_b3_data` ran first (0-NaN). Val users are disjoint from train (asserted). Don't "optimise" these away.

## 5 · VALIDATION (expected shapes — sanity-check against these)
- **CTLE-SC cat** ≈ 17–20 macro-F1 (CTLE frozen-SC is weak — *below* the random floor; AL 17.77, AZ 19.30). **Check2HGI-SC cat** ≈ 55–57 → **Δ cat ≈ +35–37 pp** (the W3 headline). If CTLE cat ≈ Check2HGI cat, the substrate isn't being swapped — bug.
- **reg (checkin-modality)** ≈ low-50s to low-60s for AL/AZ-scale; lower for wide-region CA/TX (more classes). CTLE-SC reg ≈ Check2HGI-SC reg (region is a near-tie — that's the honest finding).
- **rowcounts** (dk_ovl, stride-1 overlap): FL 1,274,418 · CA 2,925,466 · TX ~3.5 M · Istanbul 58,297.
- Comparand cat should match the existing dk_ovl cat ceilings where they exist (`docs/results/closing_data/h100/{california,florida}_s0_stl_cat_ceiling.json`; TX cat ceiling may be missing — produce it). Note the existing **TX reg ceiling is region-modality** (`a40/tx_stl_reg_ceiling_s0.json`, `input_type=region`) → **void for the matched Δ; recompute checkin-modality**.

## 6 · OUTPUTS + commit
`docs/results/closing_data/baseline_compare/<state>_{ctle,check2hgi_sc}.json` (per-fold + aggregate; NOT gitignored). Append rows to `MACS_BOARD_RESULTS.md` §CTLE-SC table. Incremental commit per (state) cell. HMT JSONs land in gitignored `results/baseline_b3_hmt_grn_style/<state>/` → record numbers in the doc.
