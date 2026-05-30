# D1 — Window / Causal-Mask Audit (no-GPU code-reading)

**Tier:** D1 (substrate-protocol-cleanup) — shared artefact for `mtl_improvement` T0.2 per the [INDEX.md handoff protocol](INDEX.md#d1--t02-mask-audit-handoff-mandatory-before-launching-d1).
**Date:** 2026-05-28
**Auditor:** code-reading agent (no GPU)
**Headline verdict:** **CLEAN** on all 5 scope items. No leak found. Tier B / C / T1 may proceed.

---

## 0. Scope and methodology

This audit answers the five questions raised in [INDEX.md §D1 (lines 134–159)](INDEX.md). Every claim below is anchored to `file_path:line_number` citations against the working tree at commit `9e9deb8` (branch `main`). No code was modified.

**Mandatory prior-art consulted** (per INDEX.md §D1):

- [`docs/findings/F50_T4_C4_LEAK_DIAGNOSIS.md`](../../findings/F50_T4_C4_LEAK_DIAGNOSIS.md) — the last leak-class fix produced for this code path (C4: val users contributing to log_T via the legacy single-prior build).
- [`docs/findings/F50_T4_BROADER_LEAKAGE_AUDIT.md`](../../findings/F50_T4_BROADER_LEAKAGE_AUDIT.md) — the broader leak inventory that produced the per-fold seed-tagged log_T migration.
- [`docs/CONCERNS.md` C19](../../CONCERNS.md) — n_splits guard, RESOLVED 2026-05-15. This audit confirms the guard still holds (§4 below).

**Note on a stale path reference:** `CLAUDE.md` lists the next-MTL head at `src/models/heads/next.py`. That path no longer exists; the file moved to `src/models/next/next_mtl/head.py` during the head-registry restructure. The class name `NextHeadMTL` is preserved. All `src/models/heads/` references in CLAUDE.md should be read as the corresponding `src/models/{cat,next,mtl}/<head>/head.py` after the restructure. Not a leak; a documentation lag (flag for future CLAUDE.md refresh, low priority).

---

## 1. `generate_sequences` — target check-in never appears in input window

**File:** [`src/data/inputs/core.py:22-91`](../../../src/data/inputs/core.py)

**Verdict: CLEAN.**

The sliding-window construction is non-overlapping by default (`stride = window_size`, line 56). For each iteration:

- Lines 61: `history = places_visited[start_idx : start_idx + window_size]` — picks positions `[start_idx, start_idx+window_size)`.
- Line 68: `target_idx = start_idx + window_size` — strictly past-the-end of the history slice. Python half-open slicing guarantees `target_idx` cannot equal any index inside `history`.
- Line 70: `target_poi = places_visited[target_idx]` — fetches the position immediately after the window.
- Line 85: `seq = history + [target_poi]` — concatenation, no aliasing.

**Edge case — tail-shift path (lines 72-79):** when the window straddles end-of-history, the function shifts the last non-pad entry of `history` into the target slot and pads the vacated position. Walk-through:

1. Line 73-77: find the rightmost non-pad index `j`, set `target_poi = history[j]`, and rewrite `history` as `history[:j] + history[j+1:] + [pad_value]`. The shifted entry is removed from `history` and inserted as the target — it cannot appear in both because slicing excludes index `j` from the new history.
2. Line 78-79: if no non-pad entry exists, `target_poi = pad_value` and line 82 skips the row.

The all-padding guard at line 82 (`all(x == pad_value for x in history) or target_poi == pad_value`) eliminates degenerate rows.

**Position-based variant ([`convert_user_checkins_to_sequences`, lines 337-429](../../../src/data/inputs/core.py)):** consumes `generate_sequences(..., return_start_indices=True)`. The target embedding is fetched at `target_idx = history_start_idx + window_size` (line 415), again strictly past the window slice indices `[history_start_idx, history_start_idx + window_size)`. The target embedding is NOT placed into `seq_embeddings` — only its category label is read (line 417). Clean.

**Region-sequence builder ([`build_region_sequence_tensor`, src/data/inputs/region_sequence.py:55-104](../../../src/data/inputs/region_sequence.py)):** iterates `poi_{0..8}` columns only (line 85: `for i in range(slide_window)`, slide_window=9). The target POI is read separately into `region_idx` in `build_next_region_frame` from the `target_poi` column of `sequences_next.parquet`. The two code paths never alias the target into the window tensor.

---

## 2. `NextHeadMTL` causal mask — position i cannot attend to j > i

**File:** [`src/models/next/next_mtl/head.py:35-58`](../../../src/models/next/next_mtl/head.py)

**Verdict: CLEAN.**

Lines 42-45:

```python
causal_mask = torch.triu(
    torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
    diagonal=1,
)
```

`torch.triu(..., diagonal=1)` produces a strict upper-triangular boolean matrix: entry `[i, j]` is `True` iff `j > i`. PyTorch's `nn.TransformerEncoderLayer(..., batch_first=True)` interprets a boolean `mask` argument as "True = position is masked / cannot be attended to". Line 47 passes this directly to `self.transformer_encoder(x, mask=causal_mask, ...)`. Therefore position `i` cannot attend to any position `j > i`. The diagonal (`j == i`) is unmasked (self-attention allowed).

**Padding-mask consistency** (lines 39, 47, 51): `padding_mask = (x.abs().sum(dim=-1) == 0)` is applied as `src_key_padding_mask` AND as a fill for the attention-pooling logits (`attn_logits.masked_fill(padding_mask, float("-inf"))`). The same mask is used at both the encoder layer and the pooling step — no per-step inconsistency.

**NaN safety** (lines 53-54): softmax over a fully-masked row would produce NaN; `torch.nan_to_num` rescues that. The pooled output (line 56) is well-defined for all-pad rows.

**Note on non-`NextHeadMTL` heads in the canonical B9 recipe** ([CLAUDE.md §B9 invocation](../../../CLAUDE.md)): the canonical reg head is `next_stan_flow` (alias `next_getnext_hard`), and the canonical cat head is `next_gru`. Although the audit scope item explicitly names `NextHeadMTL`, we verified the canonical heads as well for completeness:

- **`next_gru`** ([`src/models/next/next_gru/head.py:35-53`](../../../src/models/next/next_gru/head.py)) — `nn.GRU` is inherently causal (left-to-right recurrence); position `i` cannot see `j > i` by construction. Last-valid-timestep extraction at line 50-52 reads from the GRU output, not from any future position. Clean.
- **`next_stan_flow` / `next_stan`** ([`src/models/next/next_stan/head.py:13-16`](../../../src/models/next/next_stan/head.py)) — explicitly bidirectional ("STAN has no causal mask — all 9 inputs are past observations of a target that comes later"). The 9-step window is fully observed history; the target POI is NOT in the window per §1 above; therefore bidirectional self-attention among the 9 past steps does not constitute a target leak. The trajectory-flow prior (line 16 of [`next_stan_flow/head.py`](../../../src/models/next/next_stan_flow/head.py)) reads `last_region_idx` from the auxiliary side-channel — derived from the observed last POI, see §3. Clean.

---

## 3. `last_region_idx` — derived from observed last POI, not target

**File:** [`src/data/inputs/next_region.py:124-163`](../../../src/data/inputs/next_region.py)

**Verdict: CLEAN.**

The derivation walks the observed window columns `poi_{0..8}` (lines 128-129):

```python
poi_cols = [f"poi_{i}" for i in range(9)]
poi_mat = seq_df[poi_cols].astype(np.int64).to_numpy()  # [N, 9]
```

`target_poi` is loaded SEPARATELY into `region_idx` (lines 105, 116) and is NEVER read into the `poi_mat` array used for last-region computation. The two label columns (`region_idx` = target region, `last_region_idx` = observed last region) are populated from disjoint source columns of `sequences_next.parquet`.

**Last-non-pad selection** (lines 130-142):

- Line 130: `valid = poi_mat >= 0` — pad sentinel is `-1` (consistent with `core.py:16 PADDING_VALUE = -1`).
- Lines 132-136: `last_pos = where(any-valid, last-valid-index, -1)` — rightmost non-pad column index per row.
- Lines 138-142: `last_poi = poi_mat[arange(N), last_pos]` (or `-1` for all-pad rows). Reads strictly from the observed window, not from `target_poi`.

**Pad-row sentinel** (lines 150-158): all-pad rows get `last_region_idx = -1`. The `next_getnext_hard` / `next_stan_flow` head documents this contract ([`src/models/next/next_stan_flow/head.py:18-20`](../../../src/models/next/next_stan_flow/head.py)) and zero-suppresses the graph prior for those rows.

**Row alignment** ([`build_next_region_frame`, lines 82-98](../../../src/data/inputs/next_region.py)): the function hard-asserts row-by-row userid agreement between `next.parquet` and `sequences_next.parquet`. If the two files drift out of alignment (e.g., re-running the check2HGI pipeline partially), the assertion at line 94 raises before any leak can be silently introduced.

---

## 4. Per-fold log_T builder — train-only transitions per seed+fold

**File:** [`scripts/compute_region_transition.py`](../../../scripts/compute_region_transition.py)

**Verdict: CLEAN, with active n_splits + seed guards on the load side.**

### 4.1 Build side — train-userid-only filter

[`build_transition_matrix_from_userids`, lines 153-204](../../../scripts/compute_region_transition.py):

- Lines 182-184: `train_set = set(int(u) for u in train_userids); in_train = seq_df["userid"].astype(np.int64).isin(train_set).to_numpy(); sub = seq_df.loc[in_train]` — restricts the transition-count source to rows whose userid is in the fold's train set. Val rows are excluded BEFORE any `np.add.at` count update.
- Lines 191-197: `_log_probs_from_rows` is then called on `sub` (the filtered DataFrame), so only train-fold transitions enter the prior.

The split that drives `train_userids` is reproduced bit-equally to the trainer's split in [`_build_per_fold`, lines 246-285](../../../scripts/compute_region_transition.py):

- Line 255: `from sklearn.model_selection import StratifiedGroupKFold`
- Line 258: `X_next, y_next, next_userids, _ = load_next_data(state, EmbeddingEngine.CHECK2HGI)` — same data loader the trainer uses.
- Lines 261-265: `sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed); for fold_idx, (train_idx, _val_idx) in enumerate(sgkf.split(X_next, y_next, groups=next_userids))` — same algorithm, same y, same groups, same seed as the trainer's `FoldCreator._create_mtl_folds_with_isolation` (verified against [`src/data/folds.py:975-983`](../../../src/data/folds.py)).
- Lines 266-272: per-fold train userids → per-fold log_T → saved to `region_transition_log_seed{seed}_fold{fold_idx+1}.pt`.

### 4.2 Save side — seed + n_splits stashed in payload

[`save`, lines 207-243](../../../scripts/compute_region_transition.py): payload includes `seed` and `n_splits` keys (lines 234-237), set from the build call. Files are seed-tagged in the filename (line 280: `region_transition_log_seed{seed}_fold{fold_idx + 1}.pt`).

### 4.3 Load side — guards (CONCERNS C19 confirmation)

[`src/training/runners/mtl_cv.py:858-954`](../../../src/training/runners/mtl_cv.py):

- **Seed guard (lines 865-891):** the trainer constructs the expected path as `region_transition_log_seed{seed}_fold{i_fold + 1}.pt` using `seed = int(getattr(config, "seed", 42))`. If the file is missing, a legacy unseeded fallback is detected and the trainer hard-fails (line 876: `raise FileNotFoundError`) with explicit remediation instructions.
- **n_splits guard (lines 908-944):** the trainer loads the payload, reads `pf_n_splits`, and:
  - If `pf_n_splits is None` (legacy file) and `trainer_n_splits != 5`: hard-fail (line 915-927).
  - If `pf_n_splits is None` and `trainer_n_splits == 5`: warning, accept (line 928-933).
  - If `int(pf_n_splits) != trainer_n_splits`: hard-fail (line 934-944).

**CONCERNS C19 status:** the guard described in [`docs/CONCERNS.md:360-381`](../../CONCERNS.md) ("F51 `--folds 1` × 5-fold log_T leak bug") is implemented at the exact line range cited above. The condition that re-opens C19 (a new code path invoking `scripts/train.py --folds <5` without rebuilding log_T at the same `n_splits`) is now caught BEFORE training starts. **C19 remains RESOLVED.**

### 4.4 What this guard does NOT cover (and that's OK)

The n_splits/seed guards prevent **structural** leak (wrong fold partition or wrong seed); they do NOT detect **stale-content** leak (a log_T built before the input parquet was regenerated). The latter is covered by the C22 stale-log_T preflight gate documented in [`CLAUDE.md` §"STALE log_T preflight"](../../../CLAUDE.md), enforced operationally via `stat -c '%y %n'` and the runbook in [`INDEX.md` §Workflow](INDEX.md). Not a code-side leak; flagged here for completeness.

---

## 5. Modality discipline — `task_a_input_type=checkin` × `task_b_input_type=region`

**Files:** [`src/data/folds.py:540-573, 920-961`](../../../src/data/folds.py), [`src/data/inputs/region_sequence.py:55-104`](../../../src/data/inputs/region_sequence.py)

**Verdict: CLEAN.**

### 5.1 Per-slot independent X tensors

[`src/data/folds.py:940-971`](../../../src/data/folds.py):

```python
def _resolve_x(input_type: str) -> torch.Tensor:
    if input_type == "checkin":
        return x_checkin
    ...
    if input_type == "region":
        return build_region_sequence_tensor(state)
    if input_type == "concat":
        return build_concat_sequence_tensor(state, x_checkin)
    raise ValueError(f"Unknown input_type: {input_type}")

x_task_a = _resolve_x(self.task_a_input_type)
x_task_b = _resolve_x(self.task_b_input_type)
```

Each task slot independently picks its X tensor. `task_a=checkin, task_b=region` (the canonical B9 setting) produces two distinct tensors (`x_checkin` of shape `[N, 9, 64]` and `x_region` of shape `[N, 9, region_emb_dim]`).

### 5.2 Shared row index, NOT shared X

Lines 966-971: `TaskTensors(TaskType.NEXT, x_task_b, ...)` and `TaskTensors(TaskType.CATEGORY, x_task_a, ...)` carry their own X tensors. The fold-split row indices (lines 972-973, 990-994) are shared across tasks (StratifiedGroupKFold over the same `y_cat` and `groups=userids`) so the train/val partition is consistent, but the feature tensor for each task slot is the modality-correct one.

### 5.3 Padding-mask discipline is the same for both modalities

Both `build_region_sequence_tensor` and the check-in tensor populate **zero vectors at padded positions**:

- [`region_sequence.py:83`](../../../src/data/inputs/region_sequence.py): `out = np.zeros((n, slide_window, emb_dim), dtype=np.float32)` — initialised to zero.
- [`region_sequence.py:88-102`](../../../src/data/inputs/region_sequence.py): only `valid` (placeid `!= -1`) positions are filled with `region_emb[region_idx]`; pad positions remain zero.

Downstream heads detect padding via `padding_mask = (x.abs().sum(dim=-1) == 0)` ([`next_mtl/head.py:39`](../../../src/models/next/next_mtl/head.py), [`next_gru/head.py:36`](../../../src/models/next/next_gru/head.py), [`next_stan/head.py:237`](../../../src/models/next/next_stan/head.py)). The convention works identically for both check-in and region embeddings because both pad-positions are exact-zero — no modality-specific mask code path.

### 5.4 No cross-modality leak via cross-attention K/V

The canonical MTL model is `mtlnet_crossattn` (CLAUDE.md). The cross-attention K/V flow goes encoder-output → backbone; each task's encoder consumes its OWN modality (cat ← x_task_a = checkin; reg ← x_task_b = region). The K/V projections cannot smuggle target-side information across because (i) the input tensors are constructed from the observed 9-window only, never from `target_poi` (§1, §3), and (ii) the encoders are run in parallel on the same row index with no future-step lookahead.

**Side note on C3 (P4 K/V capacity-stealing pilot, [INDEX.md §C3](INDEX.md)):** that test is about whether the cat encoder's K/V contribution affects reg-side LEARNING DYNAMICS (a capacity-stealing question), not whether it leaks the reg target. The two concerns are orthogonal; this audit clears the leakage axis, leaving C3 free to investigate the capacity-stealing axis on substrate-protocol-cleanup's schedule.

---

## 6. Closure table (canonical-vs-this-study comparison)

Per [INDEX.md §D1 lines 148-159](INDEX.md). "Best from this study" columns are left blank with the prescribed placeholder note; they are filled by Tier A / B / C verdicts as each lands.

| Metric (FL multi-seed, disjoint reg unless noted) | Canonical c2hgi MTL (Phase 2 fresh-log_T baseline) | Best from this study | Δ pp | Source |
|---|---:|---:|---:|---|
| reg Acc@10 @ disjoint | 63.91 ± 0.16 | **66.38 ± 0.58** (log_T-KD W0.2, seed=42 pilot n=5) | **+2.40** (vs W0.0 63.98; Wilcoxon p=0.03125, 5/5 folds) | Tier A1 large-state pilot — `tier_a1_largestate/phase_a1_largestate_addendum.md` |
| reg Acc@10 @ geom_simple | 61.54 ± 4.54 | **65.20 ± 0.74** (log_T-KD W0.2, seed=42 pilot n=5) | **+4.06** (vs W0.0 61.14) | Tier A1 large-state pilot — `tier_a1_largestate/phase_a1_largestate_addendum.md` |
| reg Acc@10 @ STL ceiling | 70.92 ± 0.10 | (unchanged baseline) | — | §0.1 v11 |
| cat F1 @ joint best (AL/AZ geom_simple, canonical) | AL 45.18 / AZ 47.30 (canonical_baseline geom cat F1, seed=42 5f) | **no Tier B substrate improved it** — best Tier B design REGRESSES cat (all B/J/L/Lever4 ~−2.0 to −2.7 pp at disjoint) | **AL +0.0 / AZ +0.0** (Tier B null; cat best stays canonical) | Tier B Wave 1 + B3 — `tier_b/phase_b1b2b4_verdict.md`, `tier_b/phase_b3_verdict.md` |
| AL reg Acc@10 @ disjoint | 50.59 ± 3.53 (W0.0 n=20) | **52.85 ± 3.48** (W0.2 n=20) | **+2.27** (p=9.537e-07, 20/20 folds) | Tier A1 — `tier_a1/phase_a1_verdict.md` |
| AZ reg Acc@10 @ disjoint | 41.30 ± 2.60 (W0.0 n=20) | **46.22 ± 2.75** (W0.2 n=20) | **+4.91** (p=9.537e-07, 20/20 folds) | Tier A1 — `tier_a1/phase_a1_verdict.md` |

**Note:** D1 itself is a no-GPU audit and produces no new MTL training numbers; this table is here per the INDEX.md template so a future reader sees the comparison surface at the doc level. The first rows of "Best from this study" will be populated by Tier A (log_T-KD multi-seed) and Tier B (Designs B / J, Lever 4, Lever 5) as they close.

**Tier B closure (2026-05-28):** the "cat F1 @ joint best" row is now filled from Tier B Wave 1 + B3 at AL/AZ (small states; Tier B did not run FL multi-seed). **No Tier B substrate variant improved cat or reg under MTL+F1** — all of Design B, Design J, Lever 5, and Lever-4-on-canonical are NULL/FALSIFIED on disjoint reg (every Wilcoxon p ≥ 0.44) and each regresses cat ~−2.0 to −2.7 pp at both states. The AL/AZ disjoint-reg rows above remain Tier-A1 (log_T-KD); Tier B added no improvement to them. Per-design detail: `tier_b/phase_b1b2b4_verdict.md` (B/J/Lever5) + `tier_b/phase_b3_verdict.md` (Lever 4).

**FL-row caveat (2026-05-28):** the two FL log_T-KD rows above are from the **seed=42 sign-and-magnitude pilot** (Tier A1 large-state), NOT multi-seed. The W=0.0 baseline (63.98) overshoots the §0.1 v11 multi-seed MTL number (63.27) by the C23 development-seed bias; the W=0.2 lift (+2.40 disjoint / +4.06 geom_simple) is the pilot Δ, not a paper-grade claim. Paper-grade FL requires multi-seed {0,1,7,100}. The "Canonical c2hgi MTL" column (63.91 ± 0.16) is the Phase-2 fresh-log_T baseline and is left unchanged.

**Tier C closure (2026-05-28; APPEND — Tier A1/B rows above untouched):** Tier C added **no improvement to any frontier above** — all three protocol-coherence experiments are null/archive at AL/AZ vs the Tier B `canonical_baseline` (seed=42, 5-fold). Verdict doc: `tier_c/phase_c_verdict.md`.

| Tier C experiment | AL outcome | AZ outcome | Verdict |
|---|---|---|---|
| C1 3-snapshot routing (variant A) | 5-fold Δreg Acc@10 = −7.89 (degenerate reg-best snapshot fold3 Acc@10=0.001; +2.14 on 4 healthy folds); Δcat +0.87 (p.06) | 5-fold Δreg = +0.50 (degenerate fold2; +2.33 on 4 healthy); Δcat +0.12 (ns) | ARCHIVE — fails +2 pp gate (Wilcoxon p=0.31 both); reg-best Acc@1 selector not deploy-safe |
| C2 reg-freeze-at-epoch N∈{2,4,6} | best case (N6) reg −1.05 pp / cat +0.06 pp (ns) | best case (N6) reg −0.07 pp / cat −0.02 pp (ns) | ARCHIVE — closes §4.4 |
| C3 zero-cat-kv | reg Δ−0.28 (ns); peak ep 12.8→9.4 (earlier) | reg Δ+0.01 (ns); peak 6.2→6.6 | P4 FULLY CLOSED |

No Tier C cell touches the canonical-vs-this-study frontier numbers; the Tier A1 log_T-KD lift remains the only positive in this study. C2/C3 strengthen the case that the residual MTL-vs-STL reg gap is architectural (handed to `mtl_improvement`) — not curriculum, not cat→backbone K/V capacity-stealing.

---

## 7. Concerns / observations surfaced

1. **CLAUDE.md path drift (low):** the canonical-recipe block references `src/models/heads/next.py` and similar `src/models/heads/...` paths that no longer exist after the head-registry restructure. The classes (`NextHeadMTL`, `CategoryHeadMTL`) and behaviour are intact at the new locations under `src/models/{next,category,mtl}/<head>/head.py`. Flag for a future CLAUDE.md doc refresh; not a leak.
2. **STAN bidirectionality is intentional (informational):** `next_stan` / `next_stan_flow` (B9 reg head) is bidirectional — no causal mask among the 9 past steps. This is correct because the target POI is strictly outside the 9-window per §1 / §3. If a future variant ever pipes the target into the input window (e.g., a teacher-forcing reg-head experiment), this guarantee would need to be re-audited; document this dependency if such a variant is proposed.
3. **C22 stale-log_T is operationally-enforced, not code-enforced (informational):** the n_splits + seed guards (§4.3) prevent structural mismatch but cannot detect a log_T whose content is stale relative to a regenerated input parquet. The runbook gate is documented in CLAUDE.md and INDEX.md; an automated freshness check (compare `stat` mtimes inside `mtl_cv.py` and refuse to start if log_T is older than `next_region.parquet`) would be a low-cost belt-and-braces addition, but is out of scope for D1 (no code change rule). If a future Tier adds this, recommend doing it under `substrate-protocol-cleanup` Tier C as a near-zero-compute protocol improvement, not as a finding.
4. **No leak found requiring code change.** Tier B / C / T1 (mtl_improvement) may proceed without remediation.

---

## 8. Verdict summary

| # | Scope item | Verdict | Primary citation |
|---|---|---|---|
| 1 | `generate_sequences` — target never in input window | **CLEAN** | `src/data/inputs/core.py:59-89` |
| 2 | `NextHeadMTL` causal mask blocks j > i | **CLEAN** | `src/models/next/next_mtl/head.py:42-47` |
| 3 | `last_region_idx` derived from observed last POI | **CLEAN** | `src/data/inputs/next_region.py:124-163` |
| 4 | Per-fold log_T builder — train-only per seed+fold | **CLEAN** | `scripts/compute_region_transition.py:153-285`, guards at `src/training/runners/mtl_cv.py:858-954` |
| 5 | Modality discipline (task_a=checkin, task_b=region) | **CLEAN** | `src/data/folds.py:940-971`, `src/data/inputs/region_sequence.py:55-104` |

**Overall verdict: CLEAN — no leak found; Tier B/C and mtl_improvement Tier 1 are cleared to proceed. CONCERNS C19 guard confirmed still holding.**

---

## 9. Handoff

- `mtl_improvement` T0.2 (mask-audit dependency) should cite this document at `docs/studies/substrate-protocol-cleanup/window_mask_audit.md` and NOT re-audit, per the handoff protocol in [INDEX.md §"D1 ↔ T0.2 handoff"](INDEX.md).
- Tier A (log_T-KD multi-seed) and Tier B (Designs B / J / Lever 4 / Lever 5) are unblocked.
- Tier C (3-snapshot routing, freeze-reg-after-peak, K/V capacity-stealing) is unblocked.
- A short mirror entry has been appended to [`docs/studies/mtl_improvement/log.md`](../mtl_improvement/log.md) under 2026-05-28 pointing here.
