# HANDOFF — REDUCED board · **Macs** (M2 Pro 32 GB + M4 Pro 24 GB, MPS) · branch `study/board-m2pro`

> Deadline-grade 1-seed board (`RUN_MATRIX_REDUCE.md`). The Macs run **ALL baselines, device-internal** — both
> the baseline embedding AND our STL head on it AND our STL head on Check2HGI, on the SAME Mac, so every
> baseline-vs-ours Δ is clean (MPS = fp32, no fp16 confound). This offloads all baseline work from the CUDA
> cards. **1 seed (0) × 5 folds.** Continue on the existing `study/board-m2pro` / PR #30.

## 0 · SCOPE — the baseline comparison ladder (Table 3 rows + §7 checklist), seed 0 × 5f
Substrate-column baselines (baseline-emb → our heads), built on the **gated stride-1 overlap** base (`--stride 1`):
- **one-hot64** (B2c) — zero-training random projection (the floor)
- **skip-gram** (B2b) — SGNS over check-in POI sequences
- **POI2Vec** (faithful, Feng 2017) — finishing on the A40; pull its cells here
- **CTLE** (Lin 2021) — the closest competitor, **most important**; `--stride` fix already landed (PR #30 `d7064fbf`)
- **STAN** — SOTA region model on our representation (E2E; SSD has `baselines/stan`)
- **Markov-1** region transition baseline (the floor for the region claim)

## 1 · Reuse first — the SSD already has much of this
SSD `/Volumes/Vitor's SSD/ingred/output`: `b2b_skipgram_s{0,1,7,100}_f{0-4}` (seed 0 DONE), `baseline_b2c_onehot64`,
`board_baselines/{b2b,ctle,poi2vec}` (CTLE = AL only), `baselines/{stan,rehdm,poi_rgnn,mha_pe}`. **Build only the
missing (baseline,state) cells at seed 0**; do not rebuild what exists. CTLE needs the other 4 states.

## 2 · Build ORDER by cost (cheap → expensive); split across the two Macs
1. **one-hot64** — trivial, windowing-independent. `build_b2c_onehot64_substrate.py <state> 0 1` (the `1` = stride).
2. **skip-gram** — light SGNS (seed 0 mostly on SSD; fill gaps). `build_b2b_skipgram_substrate.py <state> --seed 0 --fold f --stride 1`.
3. **POI2Vec** — heavier (geotree + hierarchical softmax); finish on A40, comparison here. `--stride 1`.
4. **CTLE** — heaviest (bidirectional-Transformer MLM pretrain per fold). `build_ctle_substrate.py --state <s> --fold f --stride 1`.
   Run on the M4 Pro / MPS (`--device` auto-MPS landed, PR #30 `a7adad20`); stagger to not oversubscribe RAM.
- **Split:** M2 Pro 32 GB takes the heavier (CTLE/POI2Vec); M4 Pro 24 GB takes one-hot/skip-gram + Markov-1.
- **`--stride 1` is mandatory** on every builder (row-aligns to the board base). min_seq 5≡10 at stride-1 (no-op).

## 3 · The comparison (device-internal) — run the head on each Mac
For each (baseline, state): train our STL **cat** head (`next_gru`) and STL **reg** head (`next_stan_flow`) on the
baseline embedding AND on the Check2HGI (v14/dk_ovl) embedding, SAME Mac, SAME fold split, seed 0 × 5f → report
the baseline-vs-Check2HGI Δ (macro-F1 / Acc@10). MPS runs fp32 → no precision confound, no CUDA needed. Markov-1:
the transition-matrix Acc@10 floor (no embedding) — report MTL-vs-Markov and ceiling-vs-Markov rows (answers R1).

## 4 · DATA CONSOLIDATION (the SSD → A40 → Drive task)
The SSD is now on the M2 Pro. Stabilize the data so it is portable:
1. **rsync** the SSD baseline embeddings + v14 to the A40 (so the CUDA cards can regen any baseline cell):
   `rsync -av --progress "/Volumes/Vitor's SSD/ingred/output/{b2b_skipgram_*,baseline_b2c_onehot64,board_baselines,baselines,check2hgi_design_k_resln_mae_l0_1}" <a40>:~/ingred/output/`
2. **Manifest per folder** — mirror the v14 `metadata.json` pattern
   (`output/check2hgi_design_k_resln_mae_l0_1/metadata.json` on Drive): a `metadata.json` listing
   {engine, state, seed, fold, rows, stride, min_seq, leak-marker, sha}. Use `scripts/closing_data/m2pro_manifest.py` as the base.
3. **Drive** — push the **trained baseline embeddings** (NOT regennable cheaply) + manifests. Do **NOT** push
   `check2hgi_dk_ovl` (see the side-note in HANDOFF_BOARD.md §6: it is deterministically regennable from v14+v11,
   both already on Drive).

## 5 · PROCESS / STOP
- Branch `study/board-m2pro`; incremental commits per (baseline,state) cell + result JSON.
- First confirm AL overlap STL head FITS MPS memory (fold-1 fit check) before fanning out the large states.
- **STOP for the user:** MPS OOM on a large state (CA/TX baselines may need a CUDA card — flag it); any leak-marker
  assertion failure. Do NOT run the MTL champion (CUDA cards) or merge.
