# Check2HGI substrate / version map — operational reference

> Generated 2026-06-22 by the `substrate-overview` multi-agent workflow (11 agents, claims
> adversarially verified against code + docs) and confirmed against the on-disk artifacts on the
> A40 box. Purpose: a single, correct picture of the check2hgi version landscape so nobody deletes
> a load-bearing artifact under disk pressure. Ground-truth definitions live in
> [`../../results/CANONICAL_VERSIONS.md`](../../results/CANONICAL_VERSIONS.md); this doc adds the
> **scientific-vs-operational** distinction and the **per-state `next_region` status** that caused
> confusion.

## 1 · Version map

| Version | Engine dir | What it is | Role | Reported by the NEW paper (netcore)? |
|---|---|---|---|---|
| **v11** | `output/check2hgi/<state>/` (~19 G) | GCN encoder, B9/H3-alt recipe, log_T-KD OFF | **OLD BRACIS canon — FROZEN; now the operational label/fold/log_T backbone** | **NO** — cited only as superseded |
| **v12** | same on-disk artifact as v11 | v11 + log_T-KD ON + ResLN encoder *for future builds*; on-disk artifact unchanged (still GCN) | code-default flag set, not a distinct substrate | NO |
| **v13** | `output/check2hgi_resln_design_b/` | ResLN + Design-B POI2Vec-at-pool | opt-in STL base, **superseded by v14** (deleted to reclaim disk 2026-06-22, regennable) | NO |
| **v14** | `output/check2hgi_design_k_resln_mae_l0_1/` (~12 G) | ResLN encoder + Delaunay POI–POI GCN reg lever (design_k, λ=0.1) + masked-POI MAE cat lever (λ=0.3) | **CURRENT substrate** (decided 2026-06-12) | **YES — the single reported Check2HGI substrate** |
| **dk_ovl** | `output/check2hgi_dk_ovl/` (~30 G) | **v14 embeddings (symlinked) re-windowed at gated stride-1 overlap, MIN_SEQ=10** | **the engine the P3 board TRAINS on** (`--engine check2hgi_dk_ovl`) | **YES, operationally** — scientifically it *is* v14 at the embedding level |

## 2 · `dk_ovl` = v14 + gated-overlap window (confirmed)

`build_overlap_probe_engine.py <state> 1` builds `check2hgi_dk_ovl/<state>/` by:
- **symlinking** the three windowing-independent v14 artifacts — `embeddings.parquet`,
  `region_embeddings.parquet`, `poi_embeddings.parquet` → `check2hgi_design_k_resln_mae_l0_1/<state>/…`
  (verified: the symlinks resolve to the v14 dir);
- **regenerating as REAL files** the windowed pipeline: `input/next.parquet`,
  `temp/sequences_next.parquet`, `input/next_region.parquet` at `stride=1, emit_tail=false`
  (the M1 gate auto-disables tail-emit at stride==1; `next_build_provenance.json` confirms).

So `dk_ovl` is v14 — only the windowing differs. Its embeddings are free to drop (symlinks); its
windowed parquets are **regennable iff v11 (sequences/graph/fold groups) AND v14 (embeddings) both
survive**.

## 3 · The `next_region` confusion — v14-direct vs dk_ovl-overlap

**Question that exposed it:** "if v14 has no `next_region`, how did the TX MTL run?"
**Answer:** the board trains on **`dk_ovl`**, not v14-direct. The TX MTL rundir is
`results/check2hgi_dk_ovl/texas/…` → it read `check2hgi_dk_ovl/texas/input/next_region.parquet`
(9.4 G, present). v14's *own* non-overlap `next_region` is irrelevant to the overlap board.

Per-state `next_region` status (verified on disk 2026-06-22):

| state | v14-direct (non-overlap) | dk_ovl (overlap — **board uses this**) |
|---|---|---|
| TX | ❌ missing | ✅ 9.4 G — **the TX MTL ran on this** |
| CA | ❌ missing | ❌ missing (reclaimed 2026-06-22; A100 lane; regennable) |
| FL | ✅ 503 M | ✅ 3.3 G |
| AL | ✅ 36 M | ✅ 285 M (**built at MIN_SEQ=5, off the board's 10 standard**) |

**Takeaway:** v14-direct `next_region` missing at TX/CA is **not** a board blocker — the overlap board
uses `dk_ovl`. The only real gaps are: CA `dk_ovl` next_region (regennable) and AL `dk_ovl` MIN_SEQ=5
(rebuild for a consistent set).

## 4 · `output/check2hgi` (v11): scientifically retired, operationally indispensable

The crucial nuance. v11's **embedding vectors are never consumed** by any netcore-reported run — but
the **v11 dir** is the hardwired source (`IoPaths.CHECK2HGI`, code-level, not just data presence) for:

1. **Live baseline fan-out** — `m2pro_baseline_fanout.py`'s `stage_scratch()` copies
   `output/check2hgi/<state>/{temp/sequences_next.parquet, input/next.parquet, embeddings.parquet}`
   and symlinks `region_embeddings.parquet` + `temp/checkin_graph.pt` for **every** baseline cell.
   b2b/poi2vec read it for **metadata columns only** (`userid, placeid, category, datetime`) — they
   train their own 64-d embeddings from scratch (no v11-vector contamination) but cannot stage without it.
2. **Region label space** (`poi_to_region` via `checkin_graph.pt`) — `region_sequence.py`,
   `build_design_next_region.py` emit *any* substrate's `next_region` from the CHECK2HGI graph maps.
3. **Canonical fold groups** — the StratifiedGroupKFold split read off `check2hgi/<state>/input/next.parquet` userids.
4. **Per-fold `log_T` producer** — `compute_region_transition.py --per-fold` reads CHECK2HGI graph/sequences/folds and **writes to the CHECK2HGI dir** (hardwired; KNOWN GAP per `p3_board.sh`); the board then copies log_T into the v14/dk_ovl dir.

The trainer log_T *consumer* is engine-flexible (reads from `--per-fold-transition-dir`), so a *re-train*
of an already-staged cell no longer needs v11 — but any *new* state/seed log_T or label/fold build does.

## 5 · Disk-deletion verdict

| Target | Size | Verdict | Why |
|---|---|---|---|
| **v14** `check2hgi_design_k_resln_mae_l0_1` | 12 G | **NEVER** | reported substrate + symlink target of all dk_ovl files; deleting dangles every dk_ovl symlink |
| **v11** `output/check2hgi` | 19 G | **do NOT delete for disk** | scientifically retired BUT the only source for region labels / fold groups / log_T producer, and the live fan-out stages from it. Highest blast-radius-to-size ratio → keep it last |
| **dk_ovl** `check2hgi_dk_ovl` | 30 G | **SAFE-AFTER-RUN, not now** | the P3 board trains on it; largest *eventually*-reclaimable target; regennable only while v11 + v14 survive |

**Safely deletable (already done, validated):** v13 (superseded), `docs/results/canonical_improvement`
T6 sweeps (git-ignored, closed study), stale `checkpoint_epoch_*.pt`, CA `dk_ovl` next/next_region
(A100 lane, regennable). For more space the answer is **`/dados` access** or **batched completion** —
never the v11/v14/dk_ovl trio while the board is live.

## 6 · Scientific honesty caveat (from the workflow verdicts)

v14 is an **STL-only / dual-axis** substrate win — it gives **zero MTL benefit** (v14 ≈ canonical in
MTL, 5-fold × 4-seed leak-free). The netcore parity story rests on the C25 class-weighting-confound
dissolution + champion G, **not** on v14 lifting MTL. (Recorded so the framing isn't overclaimed.)
