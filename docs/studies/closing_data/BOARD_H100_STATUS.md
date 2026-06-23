# Board lane — H100 (CUDA) · branch `study/board-h100`

> This box is an **H100 80GB** running the **A100-lane role** (`HANDOFF_BOARD_A100.md`). Both are CUDA →
> the recipe is identical. Per `OOM_MEMORY_FIX.md`, the H100 is no longer *required* for memory (the A40 fits
> the whole board); it is a **speed** choice here. Work only on `study/board-h100`; never commit to `main`,
> never merge another lane.

## Environment (pinned)
- **torch 2.11.0+cu128** (installed 2026-06-22 to match the study pin; `EXECUTION_PLAN §10`). pyg stack
  rebuilt to pt211: `torch_scatter 2.1.2+pt211cu128`, `torch_sparse 0.6.18+pt211cu128`,
  `torch_cluster 1.6.3+pt211cu128`; `torchvision 0.26.0+cu128`. `torch_geometric 2.7.0`. All project modules
  + `Node2Vec` import; CUDA ops verified on the H100. Conda env `cloudspace` (NOT `.venv` — adapt the handoff
  commands accordingly).

## Key decision (user, 2026-06-22) — OWN-STATES on a rebuilt substrate
The frozen v14 substrate (`check2hgi_design_k_resln_mae_l0_1`) was **absent** on this box. It was **rebuilt
here** from the on-disk prerequisites (base check2hgi graph + HGI Delaunay edges + POI2Vec teacher, all
present) via `scripts/probe/build_design_k_delaunay.py --seed 42 --encoder resln --anchor-lambda 0.1
--mae-poi-lambda 0.3 --out-suffix resln_mae_l0_1` (FL + CA).

**The rebuild is functionally faithful but NOT byte-identical** to `V14_HASH_MANIFEST.json` (see
`V14_REBUILD_H100_PROVENANCE.json`). Root cause: the builder sets only `torch.manual_seed`/`np.random.seed`
— no `use_deterministic_algorithms`/cudnn-deterministic — so 500-epoch GCN training is not bit-reproducible
across GPU archs (H100 ≠ the GPU that built the frozen FL/CA artifacts). `EXECUTION_PLAN §11 #2` already
conceded "strict one-environment identity is not provable from the manifest."

**Consequence (decided): this H100 OWNS whole states end-to-end on its own rebuilt substrate** — substrate +
STL ceilings + champion-G MTL + baselines all run HERE. Per the governing device-class rule, every per-state
Δ (MTL-vs-STL, baseline-vs-ours) is device-internal and clean. We do **NOT** run the cross-device A/B against
the A40's FL reference (`cat 79.0083 / reg 75.5000`) as a byte-equivalence gate — the substrate differs by
design, so that Δ would conflate substrate + arch. The absolute cross-state table therefore carries a
substrate+device-class footnote for the H100-owned states.

## Substrate provenance (rebuilt here)
`V14_REBUILD_H100_PROVENANCE.json` records the rebuilt FL+CA hashes (all `matches_frozen: false`, by design).

## ⭐ UPDATE 2026-06-22 — SWITCHED TO THE FROZEN v14 SUBSTRATE (supersedes own-states)
The user provided the frozen v14 + hgi via Drive. Downloaded with `scripts/phase3_download_drive.py`
(direct-requests downloader — sidesteps gdown's confirm-interstitial/rate-limit failures) and **verified:
all 6 states' frozen `{embeddings,poi,region}.parquet` match `V14_HASH_MANIFEST.json` exactly**.

**Why we switched away from the rebuilt substrate:** the rebuilt embeddings are distributionally equivalent
(mean/std/row-norm match) and downstream-equivalent (FL STL reg ceiling 76.77 vs A40 76.71, +0.05pp), BUT
per-POI **cosine(frozen, rebuilt) ≈ 0.02** — an arbitrary rotation (independent unsupervised GNN runs share
geometry, not basis). So a rebuilt-substrate MTL run is *not the same experiment* as the A40's. Using the
**frozen** substrate makes every H100 cell byte-identical-input to the A40 → a clean arch-only A/B, and
satisfies the freeze-plan's "ONE frozen substrate for all lanes." **This reverts the own-states decision.**

- Frozen v14 installed for **all 6 states** → `output/check2hgi_design_k_resln_mae_l0_1/<state>/` (verified).
- HGI copied for all states → `output/hgi/<state>/` (georgia partial — see below).
- Rebuilt embeddings archived at `output/_h100_rebuilt_v14_archive/{florida,california}/`.
- **FL ceilings computed on the rebuilt substrate (cat 75.46 / reg 76.77) are SUPERSEDED** — must be
  recomputed on the frozen overlap engine (will then be directly A40-comparable).

### Overlap engines on the FROZEN substrate (rebuild status)
- **AL, AZ, GA**: overlap engine (gated min_seq10/stride1) + seed-0 log_T built & fresh on this box. ✓ READY.
- **FL, CA, TX**: stale rebuilt-overlap cleared; **rebuild on the 108GB GPU box** via
  `scripts/closing_data/h100_prep_state.sh <state>` (the `next_region` build does a full in-memory `.copy()`
  of the multi-GB overlap frame → OOMs a 14GB VM). FL/CA log_T present but stale → the prep script refreshes.
- **Georgia**: UNBLOCKED 2026-06-22 — user fixed GA in Drive; re-downloaded canonical `check2hgi/georgia`
  (graph maps in `temp/checkin_graph.pt`, n_regions=2283) + GA hgi, both VERIFIED vs their manifests
  (`GE_BASE_CHECK2HGI_HASH_MANIFEST.json`, `HGI_HASH_MANIFEST.json`). GA overlap + log_T now staged.

### Integrity — all downloaded artifacts verified vs manifest
- V14/design_k (6 states) ✓ · HGI (6 states) ✓ · GA canonical check2hgi (6 files) ✓.

## Scope (this lane)
- **FL** — full cell set on the rebuilt-substrate gated-overlap engine `check2hgi_dk_ovl`: STL cat ceiling,
  STL reg ceiling (`next_stan_flow`), champion-G MTL. Matched scorer (FULL `top10_acc`, fp32, both sides).
  Report device-internal Δcat (+) and Δreg vs δ_reg = 2 pp.
- **CA** — early large-state matched B-A2 reg pair (STL `next_stan_flow` ceiling vs champion-G MTL reg), the
  most at-risk for the δ_reg margin (8501 regions). Report Δreg vs δ_reg = 2 pp.

## Recipe pins (every CUDA cell)
`--compile --tf32` + `MTL_COMPILE_DYNAMIC=1` + per-box `TORCHINDUCTOR_CACHE_DIR` · `MTL_STRICT=1` ·
`MTL_CHUNK_VAL_METRIC=1` · auto-fit dataset (NEVER `MTL_DATASET_GPU=1` for CA) · `--canon none` + explicit
recipe pins on the overlap engine (canon wrong-substrate guard hard-fails otherwise) · per-fold seed-0 log_T
**copied + touched** into the v14 dir the trainer reads · freshness preflight before every
`--per-fold-transition-dir` run.
