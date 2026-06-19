# Baselines — implementation + adversarial audit synthesis (pre-freeze)

> A40, `study/pre-freeze-a40`, 2026-06-19. 16-agent workflow (`w6c7su7g7`): 7 INCLUDE external
> baselines from `docs/research/` implemented in isolated worktrees, then each independently
> audited (leak / faithfulness / integration) by a separate adversarial agent. The workflow's
> own synthesize step died on a transient 529; this doc is the hand-written synthesis + the
> merge/fix actions taken. **Scope:** implement + audit + land in the branch + apply the cheap
> unambiguous fixes. The scored **n=20 runs are P3** (post-freeze, after the stride-1 rebuild) —
> NOT run here (correct).

## Verdict table

| # | Baseline | Class | Leak-safe | Faithful | Integrated | Verdict | Fix status |
|---|---|---|---|---|---|---|---|
| **B3** | HMT-GRN-style (SIGIR'22) — the SOLE external MTL baseline | B (E2E) | ✓ | ✓ | ✓ | **READY** | 2 hardening applied |
| B2c | one-hot-POI 64-d (zero-train floor) | A (SC) | ✓ | ✓ | ✓ | NEEDS_FIX | **fixed now** (smoke .venv) |
| B4 | Cascade (CSLSL/CatDM pattern), pinned-SC | C | ✓ | ✗ | ✓ | NEEDS_FIX | **fixed now** (engine→v14) |
| B5 | Flashback (IJCAI'20) | B (E2E) | ✓ | ✗ | ✓ | NEEDS_FIX | **fixed now** (λ_s→0.3) |
| B1 | CTLE (AAAI'21) | A (SC) | ✗¹ | ✓ | ✓ | NEEDS_FIX | recipe fix + P3 infra |
| B2b | Skip-gram / SGNS (NeurIPS'13) | A (SC) | ✓ | ✓ | ✗ | NEEDS_FIX | P3 infra (`--only-fold`) |
| B2a | POI2Vec (AAAI'17) | A (SC) | ✓ | ✗² | ✓ | NEEDS_FIX | **USER DECISION** |

¹ B1 leak is in the *documented run recipe* (`--folds 1`), not the builder — see below; trivially closed by `--folds 5`.
² B2a's *core mechanism* diverges from the paper (see B2a section) — a paper-baseline-identity call.

## Fixes applied in this session (cheap, unambiguous, internal to `scripts/baselines/`)

- **B2c** — `smoke_b2c_onehot64.py`: hard-coded `repo/.venv/bin/python` (absent in the worktree) →
  `sys.executable` fallback so the smoke's determinism / POI-constant / leak / row-align asserts
  actually execute. Substrate builder + the 3 enum edits unchanged.
- **B4** — `b4_cascade.py`: `DEFAULT_ENGINE = "check2hgi"` → `"check2hgi_design_k_resln_mae_l0_1"`
  (v14 = the champion-G/v16 substrate per RUN_MATRIX L15 + canon v16 bundle). This makes the
  cascade coupling the **only** varying factor vs G (it was silently 2-factor: coupling **and**
  GCN-vs-v14 substrate). `_logT_dir()` derives `--per-fold-transition-dir` from `args.engine`, so
  log_T now auto-points at the v14 dir.
- **B5** — `flashback_e2e.py`: `lambda_s` default `100.0` → `0.3` (CLI **and** model `__init__`).
  At km distances `exp(-ds_km·100)` underflowed for any gap >0.1 km (~85% of past positions pinned
  to the 1e-10 floor → aggregation collapsed onto the anchor = vanilla RNN, the paper's "flashback"
  re-weighting numerically disabled). 0.3 restores `exp(-ds_km·0.3)~O(0.1–1)` for the observed
  2–9 km gaps (effective attended positions ~3.6). The 6h-uniform-dt / 24h-cosine zeroing note (D2b)
  carries to P3 (use true per-step datetimes for the paper run).
- **B3** — two hardening items (verdict was already READY): (1) a hard NaN-misalignment assert in
  `load_b3_data` (build_fold_split derives indices on the NaN-dropped `load_next_data` arrays while
  B3's feature arrays are un-filtered — dormant today, 0 NaN across all 6 states, now fails loud if
  that ever changes); (2) corrected the region-prior docstring — it is **analogous**, not
  "bit-identical", to `compute_region_transition._log_probs_from_rows` (it conditions on
  `last_region_idx` = last non-pad POI, so it retains the ~1471 short-sequence rows the champion's
  `poi_8`/valid_mask prior skips; both train-only/leak-safe).

## Deferred to P3 (post-freeze) — structural, NOT pre-freeze blockers

These don't gate the freeze: scored runs are P3. They DO gate paper-grade n=20.

- **B1 + B2b per-fold scoring (`--only-fold k`).** Both pretrain a substrate **per (state,seed,fold)**.
  To score fold-k leak-cleanly the trainer must run *exactly* fold-k of the canonical 5-split with the
  fold-k substrate + the n_splits=5 seeded log_T. `train.py` has no `--fold k` selector: `--folds N`
  runs folds 0..N-1 at `n_splits=max(2,N)`. So:
  - **B1 leak (the ✗):** the documented `--folds 1` builds an `n_splits=2` partition ≠ the builder's
    `n_splits=5` fold-0 → 81.8% of the n2 val users were in the CTLE n5 pretrain corpus → transductive
    leak. **Closed for fold-0** by running `--folds 5` (matches the builder) and scoring fold-0 only;
    **fully closed at n=20** only by the rotation driver below.
  - **Fix (P3):** add `--only-fold k` to `train.py` (additive, default None = no behavior change;
    runs solely fold k of `n_splits=5`), plus a per-(seed,fold) build→score rotation driver for B1,
    B2a, B2b. Harden each builder with an `n_splits` marker the trainer asserts before scoring.
- **B2b / B2a frozen-substrate overwrite guard.** With `--engine-value check2hgi` and no scratch
  `OUTPUT_DIR`, the builder's `dst_dir` resolves onto the **frozen** substrate and would overwrite it.
  Add `assert dst_dir.resolve() != frozen.resolve()` before any write. (Operationally avoided so far
  by always pointing `OUTPUT_DIR` at scratch; make it structural for P3.)

## ✅ RESOLVED (2026-06-19) — POI2Vec faithfulness + architecture

> Decisions taken (user-confirmed) + a 9-agent audit workflow (`wf_37d016d2`, all 4 streams adversarially
> verified) + a build workflow (`wf_0df62a24`). Outcomes:
>
> - **B2a is NOT POI2Vec → relabeled.** The unfaithful impl was renamed `GeoPOI2Vec → GeoTreeSkipGram`
>   (`scripts/baselines/geotree_skipgram_lib/`, `build_geotree_skipgram_substrate.py`), all false AAAI'17
>   claims scrubbed, **kept as a separate honest baseline** (a geo-tree-regularized skip-gram). Commit `27aafa7c`.
> - **Faithful AAAI'17 POI2Vec is being built** as the new B2a (`scripts/baselines/poi2vec_lib/`): CBOW +
>   fixed midpoint-grid tree + **overlap-area φ** + user latent, **64-d** (matched to the board; the paper's
>   200-d would confound the substrate axis — intentional matched-protocol deviation), loss = paper
>   `pr_user·pr_path` (stable-NLL fallback documented), φ unit-tested on a 2×2 grid, leak-safe per-fold,
>   `dst!=frozen` overwrite guard. Build + adversarial review in flight.
> - **HGI teacher left as-is** (`research/embeddings/hgi/poi2vec.py`): it's an fclass-Node2Vec teacher
>   mislabeled "POI2Vec", load-bearing for frozen HGI/v14 **build-time only**. Per user choice, **not
>   renamed** (would churn ~25 consumer scripts reading its CSV by hardcoded path for zero board benefit);
>   added a clarifying docstring note only.
> - **Architecture (SC baselines via `train.py --engine`) CONFIRMED.** Own-pipes would reintroduce
>   head/recipe/selector/metric drift the SC axis exists to prevent. The per-fold-substrate scoring gap is
>   closed by an additive, **default-inert `--only-fold k`** flag (lands in the consolidated [ENUM-MERGE]
>   commit AFTER the FL byte-identity scans; spec: keep `train.py:1862`, restructure 1863-1865 + wrap
>   2034-2036). B3/B5 stay own E2E drivers; B4 stays the train.py-wrapper.
> - **Lane 2(d) leak re-audit:** CLOSED CLEAN (see `pre_freeze_gates/STRIDE1_LEAK_REAUDIT.md`).

### (historical) the faithfulness finding that drove the above

The audit found the original B2a's **core mechanism** departs from POI2Vec (Feng et al., AAAI'17), not just
the documented task-adaptation deviations:
1. **Objective inverted** — paper is CBOW (sum a *context window* into one vector, route the *target*
   POI's tree path against it); impl is true skip-gram (single center → single context, one pair).
2. **Wrong φ (the defining mechanism)** — paper's geographical influence φ = normalized **overlap
   area** of the POI's θ-buffered box with each leaf rectangle; impl uses an ad-hoc "within
   boundary_frac of the split line → both children at half φ" heuristic.
3. **Wrong tree** — paper builds a fixed recursive rectangular **midpoint** grid to a θ cell size;
   impl builds a data-dependent **median kd-tree**.
4. **User latent dropped** from the objective (paper includes `pr_user`); the "user is the skip-gram
   center" claim in the impl notes is false (centers are POIs).

It is leak-safe, integrated, and runs — but it is **not POI2Vec**. Two options:

- **(A) Make it faithful** (recommended if B2a must be the AAAI'17 POI2Vec row): port CBOW + the
  midpoint rectangular tree + overlap-area φ + the user term. ~half a day of work; new substrate.
- **(B) Rename it honest**: call it a *geographically-tree-regularized skip-gram*, drop the AAAI'17
  citation as the implemented model, and let **B2b (faithful SGNS)** carry the "POI-seq embedding"
  baseline slot. Zero new compute.

This changes a paper baseline's identity → flagged for the user, not auto-resolved.

## Merge status

- All 7 baselines' code landed in `scripts/baselines/` on `study/pre-freeze-a40` (worktrees were
  ephemeral). Cheap fixes above applied in-place; all four edited files `py_compile`-clean.
- **Shared-file `[ENUM-MERGE]` edits (B2c + B1 only)** — append-only EmbeddingEngine members +
  allow-list entries in `paths.py` / `folds.py` / `train.py` (+ `builders.py` for B1). Inert for the
  champion (value-keyed routing). **Held until the FL byte-identity scan completes** so that scan's
  provenance stays on champion-only code, then applied as one consolidated commit.
- B2a/B2b/B3/B4/B5 need no shared-file edits (B2a/B2b use the zero-enum escape hatch; B3/B4/B5 don't
  register an engine).
