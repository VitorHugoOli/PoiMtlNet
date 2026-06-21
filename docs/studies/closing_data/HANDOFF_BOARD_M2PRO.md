# HANDOFF — board launch · **M2 Pro** (Apple Silicon / MPS) · branch `study/board-m2pro`

> One-machine handoff. You are the **M2 Pro** lane. The shared governing rule, launch sequence, and the
> per-machine branch+PR process live in the **index**: [`HANDOFF_BOARD.md`](HANDOFF_BOARD.md) — read it once, then
> work only from this file. Recipe/guards ground-truth:
> [`../pre_freeze_gates/DEFAULTS_AND_GUARDS.md`](../pre_freeze_gates/DEFAULTS_AND_GUARDS.md) +
> [`../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md`](../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md);
> sequence + device rule: [`../EXECUTION_PLAN.md §12–§13`](../EXECUTION_PLAN.md); stats:
> [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md); baseline design + the SC-substrate-column rung:
> [`RUN_MATRIX.md §2.5, §2`](RUN_MATRIX.md).

---

## 0 · SCOPE BOUNDARY (what THIS machine does — and what it must NOT touch)

**The M2 Pro BUILDS the LIGHT substrate-column baseline EMBEDDINGS on the gated-overlap base.** These are
**device-tolerant INPUTS** that the CUDA board consumes — small fp differences between MPS-built and CUDA-built
embeddings are absorbed because the **matched-head COMPARISON runs on CUDA**, head-level, on one device-class.

For each board state × seed `{0,1,7,100}` × 5 folds, **train-only per fold**, on the **gated-overlap windowing**
(so they row-align with the frozen base): build the four light SC baselines (RUN_MATRIX §2.5 run-type 3):
- **CTLE** (Lin 2021) — `build_ctle_substrate.py`
- **POI2Vec faithful** (Feng 2017, AAAI'17) — `build_poi2vec_substrate.py`
- **skip-gram B2b** (word2vec / SGNS over check-in seqs) — `build_b2b_skipgram_substrate.py`
- **one-hot64 B2c** (zero-training random-projection floor) — `build_b2c_onehot64_substrate.py`

MPS **fp32, no compile** (these are builds, not the comparison).

**Do NOT do (these belong to other machines):**
- **Do NOT run the matched-head baseline COMPARISON cells on MPS.** The actual "does our Check2HGI embedding beat
  theirs" Δ runs **on CUDA** with the state's STL/MTL (device-class rule). MPS only PRODUCES the embedding inputs.
- **Do NOT run the A100-equivalence A/B** (A40 + A100), the **TX** reg cell (A40), or the **CA** reg cell (A100).
- **Do NOT** split a state's baselines onto MPS while its STL/MTL is on CUDA — that confounds the paired Δ. The
  embeddings you build are consumed by the CUDA comparison; you are not running the comparison.
- **Do NOT** overwrite the frozen v14 / frozen check2hgi substrate. The faithful-POI2Vec and skip-gram builders
  have a hard guard that REFUSES to clobber the frozen `check2hgi` dirs unless writing into a scratch
  `OUTPUT_DIR`; respect it (use the namespaced engine dirs / scratch `OUTPUT_DIR`).
- **Do NOT** trigger the P2 freeze or launch the full P3 board.
- **Do NOT** commit to `main`, do NOT merge any branch. Work only on `study/board-m2pro`.

> **Optional path — "own whole small states AL/AZ"** (only if 32 GB + wall-time allow): the M2 Pro MAY own AL
> entirely (its MTL + STL + baselines all on MPS) so AL's Δ's are MPS-internal (footnote the device class on
> absolutes). ⚠ **First confirm AL overlap-MTL fits MPS memory** (a fold-1 fit check) **and** that the MPS-fp32
> wall-time is acceptable. If you take this path, AL's STL/MTL/baselines ALL stay on MPS — do NOT mix AL's
> baselines (MPS) with AL's STL/MTL (CUDA). AZ (24 GB) is the M4 Pro's option, not this box's. **Default plan is
> embeddings-only; treat AL-ownership as opt-in and gate it on the memory/wall-time check.**

---

## 1 · PARALLELISM on the M2 Pro (embedding builds are embarrassingly parallel)

The substrate-column builds are **embarrassingly parallel per `baseline × state × seed × fold`** — each build is
an independent process writing to its own namespaced engine dir, so you can run several concurrently up to the
M2 Pro's core/RAM budget (each MPS/CPU build is mostly single-process; saturate cores by fanning out across
baselines and folds, not by threading one build).

**Build ORDER by cost (cheapest → heaviest)** — start the cheap ones to fill cores while the heavy ones run:
1. **one-hot64 (B2c)** — TRIVIAL, zero-training. A seeded random projection of the POI id; windowing-INDEPENDENT
   and fold-INDEPENDENT (one table per state reused for every seed/fold). Build it first, it's nearly free.
2. **skip-gram (B2b)** — light SGNS over check-in POI sequences, train-portion-only per fold (CPU, fast).
3. **POI2Vec (faithful)** — heavier (recursive geotree + hierarchical softmax + CBOW + user term), per fold.
4. **CTLE** — HEAVIEST (bidirectional Transformer MLM+MH pretrain per fold). Run these in the background and let
   the lighter builds finish around them.

- **CONCURRENT:** different `(baseline, state, seed, fold)` builds at once (e.g. all B2c tables while a CTLE
  pretrain runs); the seeded fold split is reproduced bit-identically inside each builder, so they don't contend.
- **SEQUENTIAL-ish:** keep heavy CTLE/POI2Vec pretrains from oversubscribing RAM — stagger them.

---

## 2 · THE PINS (checklist)

- [ ] **Train-only per fold (LEAK-SAFETY — HARD).** Every builder reproduces the champion fold split
      bit-identically (`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` over
      `load_next_data(state, CHECK2HGI)`, grouped by userid) and pretrains ONLY on that fold's TRAIN users; each
      asserts val users are disjoint from train users before training. A single emitted dir is leak-clean for
      **exactly ONE (state, seed, fold)**. Do not shortcut to "all data" (`--all-data` is a LEAKY smoke flag,
      never scored).
- [ ] **Gated-overlap windowing — pass `--stride 1`** so the emitted `next`/`sequences`/`next_region` match the
      frozen gated-overlap base (stride-1, MIN_SEQ=10) the CUDA board reads. The builders default to the
      canonical **stride-9 non-overlap** (the smoke path); the **board paper-grade build is `--stride 1`**
      (B2b/POI2Vec expose `--stride`; B2c takes stride as positional arg 3; CTLE inherits the windowing from
      `generate_next_input_from_checkins`). B2c is windowing-independent at the embedding level but its
      `next/next_region` are rebuilt at the passed stride.
- [ ] **Do NOT clobber the frozen substrate.** POI2Vec + skip-gram refuse to write into the frozen `check2hgi`
      dirs unless `OUTPUT_DIR` is a scratch dir / the engine dir is namespaced (`*_s{seed}_f{fold}`). Honor it.
- [ ] **MPS fp32, no compile.** These are builds; `--device cpu` or MPS as the builder defaults. No `--compile`,
      no `--tf32` (those are CUDA board knobs only).
- [ ] **Region label space is SHARED / substrate-independent.** The `region_embeddings.parquet`, `poi_to_region`
      map, and per-fold seeded `log_T` are the canonical check2hgi geographic artifacts — the builders symlink
      `region_embeddings` from check2hgi and reuse `--per-fold-transition-dir output/check2hgi/<state>` at the
      (CUDA) train step. You build only the **substrate column** (the per-POI / per-visit embedding); you do not
      build new region embeddings or log_T.
- [ ] **Row-alignment.** Every emitted `embeddings.parquet` keeps the EXACT metadata columns + row order of the
      frozen check2hgi substrate (`userid, placeid, category, datetime`) and replaces only the 64 numeric
      columns; `generate_next_input_from_checkins` + the per-baseline `build_next_region_for` then row-align by
      construction. (The builders assert this.)
- [ ] **Seeds = {0, 1, 7, 100}; folds = 5** for the scored board. (B2c's *substrate* seed is the fixed projection
      seed `1234`, separate from the fold seed — leave B2c's projection seed at the default; it's the same table
      for all fold seeds.)
- [ ] **Device-class rule (why this is safe on MPS):** the embeddings are device-tolerant INPUTS; the
      head-level COMPARISON that produces the paired Δ runs on the uniform CUDA board. Never run that comparison
      here.

---

## 3 · EXACT COMMANDS (from the builder CLIs)

> Each builder runs with `PYTHONPATH=src` and the repo `.venv`. Emit into the board `OUTPUT_DIR` (or a scratch
> `OUTPUT_DIR` for the clobber-guarded ones); the embeddings are device-tolerant. **Pass `--stride 1`** for the
> paper-grade gated-overlap build. Below: one `(state, seed, fold)` example each — loop states × `{0,1,7,100}` ×
> `{0..4}` (B2c is per-state only).

### 3a · one-hot64 (B2c) — build FIRST (trivial, zero-training, per-state)
```bash
# Usage:  build_b2c_onehot64_substrate.py <state> [seed] [stride]
#   seed   = substrate projection seed (LEAVE AT DEFAULT 1234 — same table for all fold seeds)
#   stride = 1 for the P3 overlapping-window build
PYTHONPATH=src OUTPUT_DIR=output \
  .venv/bin/python scripts/baselines/build_b2c_onehot64_substrate.py alabama 1234 1
# Emits output/baseline_b2c_onehot64/<state>/{embeddings,input/next,input/next_region,...}.parquet
# region_embeddings + poi_embeddings symlinked from check2hgi; log_T reused via --per-fold-transition-dir
#   output/check2hgi/<state> at the CUDA train step.
```

### 3b · skip-gram (B2b) — per (state, seed, fold), train-only, stride-1
```bash
PYTHONPATH=src .venv/bin/python scripts/baselines/build_b2b_skipgram_substrate.py \
    --state alabama --seed 0 --fold 0 --n-splits 5 --epochs 5 --dim 64 --stride 1 --device cpu
# Emits a FOLD-NAMESPACED engine dir (baseline_b2b_skipgram_s{seed}_f{fold}); run train.py --folds 1 on it (CUDA).
# Use --read-output-dir output / --engine-value to overlay into a scratch OUTPUT_DIR without touching frozen check2hgi.
```

### 3c · POI2Vec faithful (Feng 2017) — per (state, seed, fold), train-only, stride-1, into a scratch OUTPUT_DIR
```bash
OUTPUT_DIR=/tmp/bl_poi2vec PYTHONPATH=src .venv/bin/python \
  scripts/baselines/build_poi2vec_substrate.py alabama --seed 0 --fold 0 --n-splits 5 \
    --epochs 30 --embed-dim 64 --user-dim 64 --theta 0.05 --route-count 4 --context-window 9 \
    --loss-form mixture --stride 1 --device cpu
# Hard guard refuses to clobber the frozen check2hgi substrate (writes only into the scratch OUTPUT_DIR).
# Leave --embed-dim 64 (matched-to-board; the paper's native POI2Vec is 200-d — DIM=64 is the documented
# matched-protocol deviation). DO NOT pass --all-data for scored runs (it's the LEAKY smoke flag).
```

### 3d · CTLE (Lin 2021) — HEAVIEST, per (state, seed), builds all 5 folds, train-only, stride inherited
```bash
# Full per-fold build for one (state, seed) — builds all 5 leak-clean folds:
PYTHONPATH=src .venv/bin/python scripts/baselines/build_ctle_substrate.py \
    --state alabama --seed 0 --pretrain-epochs 10 --batch-size 256 --max-len 64 --lr 1e-3
# Emits output/check2hgi_ctle/<state>/ per built fold; CTLE_FOLD.txt records the (seed,fold) it's leak-clean for.
# A single dir is leak-clean for ONE fold → the scored driver loops seeds×folds (see the script's
# "P3 SCORED-RUN DRIVER" docstring). The contextual embedding for every row (train+val) comes from the
# train-pretrained FROZEN encoder run in inference mode → no transductive leak (CTLE is inductive).
# NOTE: confirm CTLE inherits stride-1 windowing via generate_next_input_from_checkins on the board base
# (CTLE has no --stride flag; it follows the engine's windowing).
```

> ⚠ **The matched-head COMPARISON (`train.py --engine <baseline-engine>` cat=`next_gru` reg=`next_stan_flow_dualtower`,
> + the matched rescore) runs on the CUDA board, NOT here.** Hand the built engine dirs to the orchestrator /
> commit them so the CUDA lanes (or the by-state owner) consume them. The builder docstrings print the exact
> `train.py --folds 1` smoke command for plumbing verification only.

---

## 4 · PROCESS

1. **Branch `study/board-m2pro`** (off `main`; never commit to `main`, never merge another lane).
2. **Open a DRAFT PR early** and push as you go (same pattern as PR #26–#29).
3. **Commit INCREMENTALLY** — per built artifact / per baseline×state batch: a small commit carrying the build
   provenance (the engine dir manifest / a one-line finding, e.g.
   `M2Pro: B2c one-hot64 AL stride-1 built (per-state); B2b AL s0 f0..f4 built`). For the large embedding
   parquets, follow the repo's data-tracking convention (they live under gitignored `output/` — commit the
   build manifests / provenance + the leak-clean fold markers, not the multi-GB parquets, unless instructed).
4. When a baseline × state batch completes → **flag the PR for audit**; the orchestrator audits, gives further
   instructions, and reconciles/merges. You do not merge.
5. End commit messages with the required `Co-Authored-By:` trailer; end the PR body with the required
   `🤖 Generated with [Claude Code]` line. Only on `study/board-m2pro`.

---

## 5 · STOP conditions (M2 Pro-specific)

- **Any leak-safety assertion fails** (val users not disjoint from train users; a clobber-guard trips; row-count
  / row-alignment assert in `build_next_region_for`) → **STOP**; do not emit a leaky substrate. Re-derive the
  fold split for the exact `(state, seed, fold)`.
- **A target POI is OOV vs the check2hgi `placeid_to_idx` vocabulary** (the `build_next_region_for` raises) →
  the engine's sequences are out of sync with the graph maps → **STOP** and rebuild the engine's sequences
  against the same graph (do not coerce the NaN).
- **MPS OOM / unacceptable wall-time on the optional AL-ownership path** → fall back to embeddings-only (do NOT
  run AL's STL/MTL on MPS). Confirm the fold-1 fit before committing to AL ownership.
- **Tempted to run a matched-head COMPARISON on MPS** → STOP. That is a device-class violation; the comparison
  is CUDA-only.
