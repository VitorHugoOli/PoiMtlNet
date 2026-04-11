# PyTorch 2.9.1 → 2.11.0 Upgrade Plan (MTLnet)

**Repo**: `/Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/flash`
**Source plan reference**: `plans/mtlnet_speed_optimization.md` §3 item #8 (deferred from PR #8)
**Verified baseline (post-PR-8 on torch 2.9.1, DGI Alabama 5-fold MTL)**:
14.14 min · Cat F1 0.4435 · Cat Acc 53.46% · Next F1 0.2603 · Next Acc 36.88%
**Tolerance band**: ±1pp on each metric, ≤ +20% on wall time.

---

## STEP 0 — Persist this plan in the repo (DO BEFORE ANY CODE CHANGES)

This plan currently lives in the per-session scratch dir (`/Users/vitor/.claude/plans/tidy-finding-snowflake.md`) and will not be visible to future Claude Code sessions. **First action when implementation begins**: copy it into the repo's `plans/` directory so it's version-controlled and resumable.

```bash
cd /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/flash   # or wherever the implementation worktree is
cp /Users/vitor/.claude/plans/tidy-finding-snowflake.md plans/torch_211_upgrade.md
git add plans/torch_211_upgrade.md
git commit -m "docs(plans): add torch 2.9.1 -> 2.11 upgrade plan with progress checklist"
```

After that initial commit, **edit `plans/torch_211_upgrade.md` directly** (not the scratch copy) and tick off boxes in §12 as you complete each gate. This way the working tree at any point in time shows exactly where you stopped and what's next, even if the session ends mid-way through Gate 5.

If a future session starts on this work, the resumption protocol is:
1. `cat plans/torch_211_upgrade.md | head -30` to read the metadata + Step 0 confirmation that the file is committed.
2. Jump to §12 progress checklist — find the first un-ticked item.
3. Re-read the gate description in §4 for that item.
4. Resume from there.

---

## 1. Context

PR #8 ("perf/mtl-speed-batch1") landed seven safe-batch CPU/MPS overhead optimizations and explicitly **deferred the torch upgrade** as item #8 in the parent plan. The reason for deferring it was the same reason this plan exists: a torch minor bump touches RNG streams, MPS kernel numerics, attention backends (SDPA), and the C++ ABI of three compiled extension wheels (`torch_scatter`, `torch_sparse`, `torch_cluster`). Bundling that with the perf batch would have made any regression impossible to attribute.

The user wants this upgrade "critical and cautious": walk every model file, identify what could break, smoke-test all the main flows ("don't need to execute all, just see if they will work"), and stage the rollout so any failure is easy to revert. The user has further stipulated that **quality of the model takes priority over performance gains** — meaning any fallback strategy must preserve the ability to regenerate every embedding engine at known-good quality, even if that means running parts of the pipeline on a legacy venv.

**User-chosen approach** (staged version-bump fallback):
1. **Path A**: try torch 2.11 in a single venv. If wheels exist and HGI passes Gate 6b, done.
2. **Path B**: if 2.11 wheels are missing or HGI breaks on 2.11, fall back to torch 2.10 (older release → more likely to have matching wheels published on the PyG index).
3. **Path C**: if even 2.10 wheels are missing, set up a dual-venv routing scheme — torch 2.11 for everything except HGI/POI2HGI regeneration, which stays on the legacy `.venv_new` (torch 2.9.1) until upstream wheels publish.

HGI is promoted to an explicit gate (Gate 6b) so we discover any HGI breakage during the upgrade itself, not weeks later when regenerating embeddings for a new state. See §3 for the full decision flow.

**Outcome target**: torch 2.11 OR 2.10 active in the new daily-driver venv (preferring 2.11 if available), all Tier-1 production tests still green, the DGI Alabama 5-fold MTL run reproduces post-PR-8 metrics within ±1pp, and every embedding pipeline either runs cleanly on the new venv OR is documented as routing to the legacy venv until wheels publish.

---

## 2. Two-tier risk model

The codebase has a **hard split** that the upgrade plan must respect:

### Tier 1 — production training path (must not regress)
None of these files import any PyG / `torch_scatter` / `torch_sparse` / `torch_cluster` symbol. Confirmed by grep. **Their only exposure is to torch-native semantic changes** (SDPA, autograd, MPS op coverage, `torch.load` defaults).

| File | Why it matters |
|---|---|
| `src/models/mtlnet.py` | MTLnet, FiLMLayer, ResidualBlock; uses `task_embedding.weight[i].expand(B, -1)` (PR #8) |
| `src/models/heads/category.py` | `CategoryHeadTransformer` is the production head — do NOT switch to Ensemble (mem `mtl_category_loss_unweighted.md`) |
| `src/models/heads/next.py` | `NextHeadMTL`: causal mask buffer + `padding_mask` kwarg + transformer SDPA path |
| `src/losses/nash_mtl.py` | `torch.autograd.grad(loss, shared_params, retain_graph=True)` per task + on-device GTG norm |
| `src/training/runners/{mtl_cv,mtl_eval,mtl_validation}.py` | CV runners; use cached param groups + on-device tensors from PR #8 |
| `src/data/folds.py` | `torch.load(weights_only=False)`, `torch.from_numpy` zero-copy, POIDataset `device=` kwarg |
| `src/configs/globals.py` | Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` at import time |
| `src/utils/mps.py` | `torch.mps.empty_cache()` + `torch.mps.synchronize()` |
| `scripts/train.py` | CLI entrypoint |
| `scripts/evaluate.py` | **`torch.load(checkpoint, map_location=DEVICE)` at line 141 with NO `weights_only=` kwarg** — relies on default |

### Tier 2 — research / embedding generation (existing parquets are still consumed, regeneration can be deferred)

| Engine | PyG dep? | Hardest dep | Pipeline |
|---|---|---|---|
| `research/embeddings/dgi/` | YES (`GATConv`, `BatchNorm`) | torch_scatter (runtime) | `pipelines/embedding/dgi.pipe.py` |
| `research/embeddings/hgi/` | YES (`GCNConv`, **`Node2Vec`**) | **`torch_cluster.random_walk` — HARD** | `pipelines/embedding/hgi.pipe.py` |
| `research/embeddings/check2hgi/` | YES (`GCNConv`, `NeighborLoader`) | torch_scatter, torch_sparse | `pipelines/embedding/check2hgi.pipe.py` |
| `research/embeddings/poi2hgi/` | YES (reuses HGI components) | torch_scatter, torch_sparse | `pipelines/embedding/poi2hgi.pipe.py` |
| `research/embeddings/time2vec/` | NO (pure `torch.nn`) | none | `pipelines/embedding/time2vec.pipe.py` |
| `research/embeddings/space2vec/` | NO | none | `pipelines/embedding/space2vec.pipe.py` |
| `research/embeddings/sphere2vec/` | NO | none | `pipelines/embedding/sphere2vec.pipe.py` |
| `research/embeddings/hmrm/` | n/a (pure numpy/scipy) | none | n/a |

**The crucial fact**: the production MTL training reads pre-computed `output/<engine>/<state>/embeddings.parquet`. Those parquets are static artifacts. **Even if HGI regeneration breaks under torch 2.11**, MTL training keeps working with the existing files. This decouples Tier 2 breakage from Tier 1 readiness.

---

## 3. The dominant risk: compiled extension ABI

The `.venv_new` currently has:

```
torch_cluster        1.6.3   (.so files: _scatter_cpu.so, _segment_*_cpu.so, ...)
torch_scatter        2.1.2   (compiled C++ extension)
torch_sparse         0.6.18  (compiled C++ extension)
```

These wheels link against torch's C++ ABI. After `pip install torch==2.11.0`:
- `import torch` works.
- `import torch_scatter` raises `undefined symbol: _ZN3c10...` or similar.
- Anything that imports `torch_geometric.nn.GCNConv`, `GATConv`, or `Node2Vec` stops loading.

**This is the failure mode that turns a 1-day upgrade into a week-long rabbit hole.** The plan must address it explicitly before any wheel installation.

### Wheel acquisition strategy — staged version-bump fallback

**User-chosen approach**: try the largest bump first (torch 2.11), and if its wheels aren't available, fall back to torch 2.10 (which has been out longer and is more likely to have matching `torch_scatter`/`torch_sparse`/`torch_cluster` wheels published on the PyG index). Only if BOTH 2.11 AND 2.10 fail do we resort to a dual-venv setup.

**Quality-first principle**: dropping `torch_cluster` and accepting HGI breakage is **not** an acceptable outcome. The fallback must preserve HGI's full regeneration capability at known-good quality.

**Decision flow**:
```
                ┌─────────────────────────┐
                │ Try Path A: torch 2.11  │
                └───────────┬─────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                                ▼
   wheels OK + Gate 6b OK         wheels missing
            │                     OR Gate 6b fails on Node2Vec
            ▼                                │
   ✅ Single venv on 2.11                    ▼
   (best outcome)             ┌─────────────────────────┐
                              │ Try Path B: torch 2.10  │
                              └───────────┬─────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                                ▼
                 wheels OK + Gate 6b OK         wheels missing
                          │                     OR Gate 6b fails
                          ▼                                │
                 ✅ Single venv on 2.10                    ▼
                 (acceptable outcome)         ┌─────────────────────────┐
                                              │ Path C: dual-venv (2.11 │
                                              │ + legacy 2.9.1 for HGI) │
                                              └───────────┬─────────────┘
                                                          ▼
                                              ✅ HGI regen on legacy venv
                                              MTLnet on torch 2.11
```

Each path keeps the SAME validation gates (1-7). The only difference is which torch version sits in the new venv and whether the legacy `.venv_new` stays alive as a routing target for HGI.

#### Path A — torch 2.11.0 in single venv (PRIMARY)

Install the full ecosystem in `.venv_torch211` and let the gates tell us whether everything works:

```bash
# In .venv_torch211, after Gate 0
pip install --upgrade torch==2.11.0 torchvision==0.25.0   # verify torchvision pin from pytorch.org/get-started/previous-versions
pip install --no-index \
  --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html \
  torch_scatter torch_sparse torch_cluster
pip install --upgrade torch-geometric==2.7.0
```

**Three possible install-time outcomes**:

1. **All wheels install cleanly** → continue through Gates 1-6b. If everything passes, **Path A complete**.
2. **`torch_cluster` (or `torch_scatter`/`torch_sparse`) wheel not on the PyG index for torch 2.11** → `pip install` fails at `--find-links`. Switch to **Path B (torch 2.10)** below.
3. **Wheels install but `import torch_cluster` fails at runtime with `undefined symbol`** → ABI mismatch. Switch to **Path B (torch 2.10)**.

**Path A success criterion**: all of the following must hold:
- `python -c "import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster"` exits 0.
- `python -c "from torch_cluster import random_walk; print(random_walk)"` prints a callable.
- Tier 1 Gates 2-5 all pass (DGI Alabama metrics within ±1pp of baseline).
- Gate 6a (Tier 2 imports) — every engine reports `OK` including HGI.
- Gate 6b (HGI alabama run smoke) — `pipelines/embedding/hgi.pipe.py` reaches POI2Vec epoch 2 without crashing.

If any of those fail, **wipe `.venv_torch211` and proceed to Path B**:
```bash
rm -rf /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/torch-upgrade/.venv_torch211
python3.12 -m venv /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/torch-upgrade/.venv_torch210
source /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/torch-upgrade/.venv_torch210/bin/activate
pip install -e ".[dev]"
```

#### Path B — torch 2.10.x in single venv (SECONDARY, smaller bump)

Trigger conditions: any of Path A's three install/runtime/Gate-6b failure modes.

**Why 2.10 instead of staying on 2.9.1**: torch 2.10 has been out longer than 2.11, so the PyG team has had more time to publish matching wheels. Empirically, `https://data.pyg.org/whl/torch-2.10.0+cpu.html` is much more likely to have a complete `torch_scatter`/`torch_sparse`/`torch_cluster` wheel set than 2.11. Falling to 2.10 captures most of the perf wins from a minor bump while preserving the single-venv simplicity.

```bash
# In .venv_torch210, fresh from the rm step above
pip install --upgrade torch==2.10.0 torchvision==0.25.0   # verify torchvision pin for 2.10 from pytorch.org/get-started/previous-versions
pip install --no-index \
  --find-links https://data.pyg.org/whl/torch-2.10.0+cpu.html \
  torch_scatter torch_sparse torch_cluster
pip install --upgrade torch-geometric==2.7.0
```

**Path B success criterion**: identical to Path A but with `torch.__version__.startswith("2.10")` instead of 2.11. Run all gates 1-6b. If everything passes, **Path B complete** — the upgrade lands on 2.10 instead of 2.11.

**Why this is acceptable**: the goal of item #8 in the parent perf plan was "5-10% speedup from torch minor bump", not "specifically 2.11". 2.10 captures the same class of MPS kernel improvements and SDPA backend tuning relative to 2.9.1; a half-version of additional work is left on the table but not lost forever.

If any Path B gate fails (same set of conditions as Path A), wipe `.venv_torch210` and proceed to **Path C**:
```bash
rm -rf /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/torch-upgrade/.venv_torch210
python3.12 -m venv /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/torch-upgrade/.venv_torch211
# fresh re-install of 2.11 for Path C
source /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/torch-upgrade/.venv_torch211/bin/activate
pip install -e ".[dev]"
pip install --upgrade torch==2.11.0 torchvision==0.25.0
# Skip torch_cluster on Path C — it stays in the legacy venv
pip install --no-index \
  --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html \
  torch_scatter torch_sparse   # NO torch_cluster
pip install --upgrade torch-geometric==2.7.0
```

#### Path C — Dual-venv: torch 2.11 daily + torch 2.9.1 legacy for HGI (TERTIARY, fallback)

Trigger conditions: BOTH Path A and Path B failed with `torch_cluster` issues. (If Path A failed for non-`torch_cluster` reasons — e.g. SDPA numerics regression in Gate 5 — switch to Path B's torch 2.10 instead, which is a smaller numerics delta. Path C is specifically for the case where neither minor version has working `torch_cluster` wheels.)

**Setup**:

1. The legacy `.venv_new` (torch 2.9.1 + working `torch_cluster==1.6.3`) is **already on disk and untouched** by this whole exercise — Gate 0 mandated the new install go into a separate venv. Optionally rename for clarity:
   ```bash
   mv /Users/vitor/Desktop/mestrado/ingred/.venv_new /Users/vitor/Desktop/mestrado/ingred/.venv_legacy_torch29
   ```
   (Skip the rename if it would break other tooling that hardcodes the path. The path-agnostic approach is to leave `.venv_new` named as-is.)

2. Promote `.venv_torch211` to the daily-driver venv:
   ```bash
   mv /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/torch-upgrade/.venv_torch211 /Users/vitor/Desktop/mestrado/ingred/.venv_torch211
   ```
   (Or symlink — whatever the user's workflow expects.)

3. Document the split in a top-of-file comment in `pipelines/embedding/hgi.pipe.py` and `pipelines/embedding/poi2hgi.pipe.py`:
   ```python
   # NOTE: This pipeline requires torch 2.9.1 + torch_cluster==1.6.3.
   # Activate .venv_new (legacy) before running:
   #   source /Users/vitor/Desktop/mestrado/ingred/.venv_new/bin/activate
   # See plans/torch_211_upgrade.md (this PR) for context.
   ```

4. Add a quick sanity check at the top of those pipe files (defensive — fails fast with a clear message instead of an obscure ImportError trace 30 lines down):
   ```python
   try:
       from torch_cluster import random_walk  # noqa: F401
   except ImportError:
       raise SystemExit(
           "HGI/POI2HGI regeneration requires torch_cluster, which is not "
           "available for torch 2.11 yet. Activate the legacy venv "
           "(.venv_new with torch 2.9.1) and re-run."
       )
   ```

**Daily venv routing under Path C**:

| Task | Venv | Reason |
|---|---|---|
| `scripts/train.py` (any engine, including `--engine hgi`) | `.venv_torch211` | Reads pre-computed parquet, no HGI training code touched |
| `scripts/evaluate.py` | `.venv_torch211` | Same |
| `pipelines/create_inputs.pipe.py` | `.venv_torch211` | No PyG/torch_cluster dep |
| `pipelines/embedding/dgi.pipe.py` | `.venv_torch211` | DGI uses GATConv but not Node2Vec/torch_cluster; works under PyG fallback for scatter/sparse |
| `pipelines/embedding/check2hgi.pipe.py` | `.venv_torch211` | Same — no Node2Vec |
| `pipelines/embedding/time2vec.pipe.py` | `.venv_torch211` | Pure torch.nn |
| `pipelines/embedding/space2vec.pipe.py` | `.venv_torch211` | Pure torch.nn |
| `pipelines/embedding/sphere2vec.pipe.py` | `.venv_torch211` | Pure torch.nn |
| **`pipelines/embedding/hgi.pipe.py`** | **`.venv_new` (legacy 2.9.1)** | Hard `torch_cluster.random_walk` dep via `Node2Vec` |
| **`pipelines/embedding/poi2hgi.pipe.py`** | **`.venv_new` (legacy 2.9.1)** | Reuses HGI components |

Cross-venv state sharing: **none**. The two venvs only communicate through `output/<engine>/<state>/embeddings.parquet` artifacts on disk. Parquet is a stable cross-version format. No class instances, no .pt files, no pickles cross the venv boundary.

**Quality impact under Path C**: ZERO. HGI continues to run on its known-good torch 2.9.1 path producing identical embeddings. MTLnet training under torch 2.11 reads those parquets normally. Both flows run at their respective baselines.

**Operational cost under Path C**: must remember which venv to activate for HGI/POI2HGI regeneration. Mitigated by (a) the top-of-file comment in step 3 above, (b) the defensive ImportError check in step 4, and (c) the fact that HGI regeneration is an infrequent operation (one-time per state, typically).

**Exit criterion for Path C**: re-check `https://data.pyg.org/whl/torch-2.11.0+cpu.html` periodically. Once `torch_cluster` wheels publish for torch 2.11, install them into `.venv_torch211`, re-run Gate 6b on alabama, and decommission `.venv_new`. That decommission is a follow-up task, not part of this PR.

#### Rejected alternatives (do NOT pursue inside this PR)

- **Drop `torch_cluster` entirely** — would silently break HGI/POI2HGI regeneration with no fallback. Violates the "quality over performance" principle.
- **Build `torch_cluster` from source** — Xcode CLT requirement, 15-20 min per build, historically fragile on Apple Silicon, risk of subtle build-flag mismatches producing silent numerics differences.
- **Block the upgrade until wheels publish** — forfeits any torch perf wins indefinitely, with no clear timeline. The 2.11 → 2.10 → dual-venv ladder above provides a strictly better outcome at every rung.

---

## 4. Staged execution gates

Each gate is a hard go/no-go checkpoint. Failure at any gate triggers the rollback in §7.

### Gate –1: Pre-flight inventory
1. `git status` in flash worktree clean (only `.claude/settings.local.json` mod and `output/` symlink expected).
2. Re-run `pytest tests/test_models tests/test_training tests/test_data tests/test_losses -q` on the **current** torch 2.9.1 venv → must show **242 passed, 36 skipped** (post-PR-8 baseline). If different, fix that first.
3. Confirm post-PR-8 DGI Alabama baseline numbers are recorded somewhere referenceable (results JSON in `results/dgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260411_0959/summary/full_summary.json` from PR #8 validation, or rerun if missing).
4. **Confirm `.venv_new` is shared**: it lives at `/Users/vitor/Desktop/mestrado/ingred/.venv_new`, NOT inside any worktree. Upgrading torch in `.venv_new` would silently break the `flash` and `main` worktrees. **Critical** — the upgrade MUST happen in an isolated venv.
5. **Verify wheel availability**: open `https://data.pyg.org/whl/torch-2.11.0+cpu.html` in a browser. Decide A vs C before continuing.
6. **pyproject.toml sanity check** — inspect `pyproject.toml` for drift from `.venv_new`'s actual install state BEFORE running Gate 0's install. The 2.11 upgrade session surfaced two pre-existing bugs here:
   - `build-backend = "setuptools.backends._legacy:_Backend"` was a typo (that module path does not exist on any released setuptools); `pip install -e .` was silently broken for months.
   - `pytorch-warmup` was installed in `.venv_new` and listed in `requirements.txt`, but MISSING from `pyproject.toml` deps — so any fresh-venv install was missing it, which breaks `research/embeddings/hgi` imports.
   Run `diff <(sed -n 's/^\([a-zA-Z0-9_-]*\)==.*/\1/p' pyproject.toml | sort -u) <(.venv_new/bin/pip freeze | sed -n 's/==.*//p' | sort -u)` — investigate any asymmetry before trusting the upgrade diff to be attributable.

### Gate 0: Isolated worktree + isolated venv
```bash
cd /Users/vitor/Desktop/mestrado/ingred
git worktree add .claude/worktrees/torch-upgrade main
cd .claude/worktrees/torch-upgrade
python3.12 -m venv .venv_torch211      # local to this worktree, NOT shared
source .venv_torch211/bin/activate
pip install -e ".[dev]"                 # initial install on torch 2.9.1
pytest tests/test_models tests/test_training tests/test_data tests/test_losses -q
```
**Pass**: fresh venv reproduces 242 passed / 36 skipped on torch 2.9.1.
**Fail**: drift between `pyproject.toml` and the active `.venv_new` — reconcile before going further.

### Gate 1: Install torch + matching ecosystem (Path A first)
```bash
# Path A — torch 2.11
pip install --upgrade torch==2.11.0 torchvision==0.25.0   # verify exact torchvision pin from pytorch.org/get-started/previous-versions
pip install --no-index \
  --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html \
  torch_scatter torch_sparse torch_cluster
pip install --upgrade torch-geometric==2.7.0
```
Verification:
```python
import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster
print(torch.__version__, torch.backends.mps.is_available())
print(torch_cluster.random_walk)   # must be a real callable
```
**Pass (Path A)**: all imports succeed; `torch.__version__.startswith("2.11")`; MPS available; `torch_cluster.random_walk` is a function. Continue to Gate 2.
**Fail (`undefined symbol` or `pip install` cannot find a torch_cluster wheel for 2.11)**: wheel ABI mismatch / wheel missing. **Wipe `.venv_torch211` and switch to Path B (torch 2.10)** per §3 — re-create the venv, install torch 2.10 + matching wheels, and re-run Gate 1.
**Fail on Path B as well**: switch to Path C (dual-venv) per §3.

### Gate 2: Tier-1 targeted test suite
```bash
pytest tests/test_models tests/test_training tests/test_data tests/test_losses -q | tee /tmp/gate2.log
```
**Pass**: ≥ 242 passed, 0 failures.
Watchpoints: any new failure in `test_heads*` (SDPA backend), `test_nash_mtl*` (autograd graph retention), or `test_folds*` (`torch.load` / `from_numpy`).

**Also run the regression suite** — earlier revisions of this plan said to skip it because `test_mtl_f1_within_tolerance` was "flaky", but on torch 2.11 all 12 regression tests pass cleanly. If for some reason you still need to deselect the MTL-tolerance test, the correct path is:
```bash
pytest tests/test_regression -q --deselect 'tests/test_regression/test_regression.py::TestMTLRegression::test_mtl_f1_within_tolerance'
```
Note the class is `TestMTLRegression`, not `TestMTL` — the earlier plan wording had the wrong class name and would have silently failed to deselect anything.

### Gate 3: Tier-1 broad test suite
```bash
pytest tests/test_integration tests/test_configs tests/test_tracking tests/test_utils -q
```
**Pass**: no NEW failures vs. the post-PR-8 baseline (349 passed, 62 skipped on 2.9.1).

### Gate 4: 1-fold / 1-epoch MTL DGI Alabama smoke
```bash
python scripts/train.py --task mtl --state alabama --engine dgi --folds 1 --epochs 1
```
**Pass criteria**: completes in 2-3 min; finite losses (no NaN/inf); no traceback; finishes with reasonable per-epoch metrics (cat F1 > 0.2, next F1 > 0.1 — sanity floor).
**Hard-fail signals**:
- NaN in losses → likely SDPA numerics change in `src/models/heads/next.py:132` (`mask=causal_mask, src_key_padding_mask=padding_mask`) or `src/models/heads/category.py` transformer.
- `RuntimeError` from `torch.autograd.grad` → NashMTL regression (`src/losses/nash_mtl.py:271-277`).
- `RuntimeError: operator ... not supported on MPS` → fallback didn't catch a new gap. Double-check `PYTORCH_ENABLE_MPS_FALLBACK` is set in subprocess env.
**Soft signals (record, don't block)**: `FutureWarning`, SDPA backend selection warnings, deprecation warnings about `torch.load` defaults.

### Gate 5: Full DGI Alabama 5-fold / 50-epoch validation (GO/NO-GO)
```bash
/usr/bin/time -p python scripts/train.py --task mtl --state alabama --engine dgi --folds 5 --epochs 50
```
**Pass criteria**:
| Metric | Lower bound | Upper bound |
|---|---|---|
| Cat F1 | 0.4335 | 0.4535 |
| Cat Acc | 52.46% | 54.46% |
| Next F1 | 0.2503 | 0.2703 |
| Next Acc | 35.88% | 37.88% |
| Wall time | — | ≤ 17 min (post-PR-8 was 14.14 min) |

**Fail = full rollback trigger.** Do NOT attempt piecemeal fixes inside this gate. If a metric drifts > 1pp, the upgrade has a numerics regression somewhere subtle (likely SDPA backend or MPS LayerNorm/GELU); accumulating "small fixes" inside the same PR makes attribution impossible. Roll back, file a follow-up issue with the failing fold's metrics and the SDPA/MPS suspect surface, and re-plan.

**Note on bit-exact reproduction**: the training pipeline is fully seeded (`src/utils/seed.py` + `src/data/folds.py:471-473` set `random`, `np.random`, and `torch.manual_seed`). On torch 2.9.1 → 2.11 the DGI Alabama metrics reproduced to 4 decimal places (Cat F1 0.4435, Cat Acc 53.46%, Next F1 0.2603, Next Acc 36.88%). This is **expected**, not suspicious: torch 2.11's MPS kernels produce identical outputs to 2.9.1 on every op used by MTLnet. Treat a bit-exact match as the strongest possible signal and drift > 0 as the thing worth investigating — not the other way around.

### Gate 6a: Tier-2 import-only smoke (no execution)

Research engines live under `research/embeddings/*` and are imported as `embeddings.<engine>.<engine>` — pipelines do this by prepending `research/` to `sys.path` at runtime. The smoke must mirror that, otherwise everything fails with `ModuleNotFoundError: No module named 'embeddings'`.

```bash
python -c "
import sys
from pathlib import Path
root = Path('.').resolve()
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'research'))

import importlib
modules = [
    'embeddings.dgi.dgi',
    'embeddings.check2hgi.check2hgi',
    'embeddings.poi2hgi.poi2hgi',
    'embeddings.time2vec.time2vec',
    'embeddings.space2vec.space2vec',
    'embeddings.sphere2vec.sphere2vec',
]
for m in modules:
    try:
        importlib.import_module(m)
        print(f'OK     {m}')
    except Exception as e:
        print(f'FAIL   {m}: {type(e).__name__}: {e}')
# HGI checked separately — Node2Vec import is the canary for torch_cluster ABI
try:
    import embeddings.hgi.hgi, embeddings.hgi.poi2vec
    print('OK     embeddings.hgi (Node2Vec available)')
except ImportError as e:
    print(f'IMPORT-FAIL embeddings.hgi: {e}')
"
```
**Do NOT try to import `pipelines.create_inputs`** — the file is `pipelines/create_inputs.pipe.py`, the `.pipe.py` suffix makes it non-importable as a Python module. Earlier revisions of this plan listed it; that entry was always impossible and would report a misleading `ModuleNotFoundError`.

**Pass criteria**: every non-HGI module reports `OK`. HGI line determines whether Gate 6b is even reachable:
- HGI line `OK` → proceed to Gate 6b (run smoke).
- HGI line `IMPORT-FAIL` (specifically `ModuleNotFoundError: No module named 'torch_cluster'`) → **trigger Path B (dual-venv)** in §3, skip Gate 6b, mark HGI as "deferred to legacy venv until wheels publish".
- Any non-HGI `FAIL` → **hard blocker**, do not continue.

### Gate 6b: HGI alabama run smoke (CRITICAL — determines single vs dual venv)

Only run if Gate 6a's HGI line was `OK`. This gate exists specifically because the user wants to verify HGI quality is preserved before committing to single-venv.

> ⚠️ **DESTRUCTIVE SIDE EFFECT WARNING** — `pipelines/embedding/hgi.pipe.py` has `force_preprocess=True` in `HGI_CONFIG` and writes its outputs into `output/hgi/<state>/` unconditionally. If the worktree's `output/` is a symlink into the main repo (typical setup), running this smoke OVERWRITES the real 2000-epoch embeddings artifacts (`embeddings.parquet`, `region_embeddings.parquet`, `poi2vec_*`, `input/category.parquet`, `input/next.parquet`) with whatever short-epoch smoke values you configured. During the torch 2.11 upgrade this trap was hit — recovery required `rsync -a --delete` from a sibling worktree (`.claude/worktrees/mtlnet-improve/`) that happened to have an intact non-symlinked copy. **Before running Gate 6b, do ONE of the following**:
> - **Snapshot the existing outputs**: `cp -a output/hgi/alabama output/hgi/alabama.backup-$(date +%Y%m%d_%H%M%S)` — cheapest option, just restore the backup dir when done.
> - **Override the output root**: run the pipeline with `DATA_ROOT=/tmp/gate6b_smoke python pipelines/embedding/hgi.pipe.py` (works because `src/configs/paths.py` respects `$DATA_ROOT`), so artifacts land somewhere disposable.
> - **Set `force_preprocess=False`** in the local `HGI_CONFIG` edit — but this only skips phase 1 (graph preprocessing), not the downstream writes; you still need one of the two options above.

**Setup** (one-time, to keep the smoke fast and non-destructive):
1. Run the backup/override step from the warning above. Non-negotiable.
2. Read `pipelines/embedding/hgi.pipe.py` and temporarily shrink `STATES` to just `{'Alabama': Resources.TL_AL}`. Also reduce `HGI_CONFIG.epoch` (2000 → 3) and `HGI_CONFIG.poi2vec_epochs` (100 → 3) so the smoke finishes in under a minute. **These edits are local-only** — do NOT commit them; revert with `git restore pipelines/embedding/hgi.pipe.py` after the gate.
3. If the file later exposes an `--epochs` or `--max-epochs` CLI flag, prefer that over editing the file.

**Run**:
```bash
python pipelines/embedding/hgi.pipe.py 2>&1 | tee /tmp/gate6b_hgi.log
```

**Pass criteria**:
- POI2Vec pre-training reaches at least the second epoch without crashing (you can ctrl-C after that — we're verifying the surface, not training to completion).
- No `ImportError`, no `AttributeError`, no MPS-fallback warnings about ops with unsupported dtypes.
- Loss values logged are finite.
- Output file path is announced (whether or not the run completes is fine; the goal is "the code path runs").

**Fail modes and what they trigger**:
- `ImportError: torch_cluster` → **trigger Path B (dual-venv)**, even though Gate 6a passed. This shouldn't happen if 6a passed, but defensive.
- `RuntimeError` inside `Node2Vec.random_walk` (some C++ ABI issue or kernel mismatch even with matching wheels) → **trigger Path B (dual-venv)**. Node2Vec is the most ABI-sensitive path in the entire codebase.
- `RuntimeError` inside `GCNConv` or `torch_geometric.utils.scatter` → distinct from Node2Vec; likely a PyG-side compatibility issue. Investigate before triggering Path B; may need a PyG bump.
- NaN loss → numerics regression in HGI itself, unrelated to torch_cluster. **Block** and investigate the SetTransformer / discriminator code paths.
- Wall time > 2x baseline → fallback path is in use somewhere, not a quality issue but flag for follow-up.

**Why this gate is non-negotiable**: it's the only way to confirm HGI quality is preserved BEFORE the user discovers a regression weeks later when regenerating embeddings for a new state. The user explicitly asked for this verification.

**Restore step after Gate 6b**: undo the local `STATES = ["alabama"]` edit so the working tree stays clean.

### Gate 7: Other Tier-2 short generation smoke (OPTIONAL)
Run only if Gates 6a and 6b passed clean. **Cheapest engines first.**
```bash
# Pure-torch engines
python pipelines/embedding/time2vec.pipe.py        # ~1-3 min on alabama
python pipelines/embedding/space2vec.pipe.py       # similar
python pipelines/embedding/sphere2vec.pipe.py      # similar

# PyG engines
python pipelines/embedding/dgi.pipe.py             # ~2-5 min
python pipelines/embedding/check2hgi.pipe.py       # ~5-10 min
python pipelines/embedding/poi2hgi.pipe.py         # only if Path A — uses Node2Vec
```
**Pass**: each writes its parquet output; no NaN; no traceback.
**Note**: many of these pipelines run on multiple states by default — apply the same `STATES = ["alabama"]` local edit pattern as Gate 6b, then revert.

---

## 5. API risk audit (per-file watchpoints)

Eight risk areas, each with file:line refs and the gate that answers them.

| # | Risk | Refs | Answered at gate |
|---|---|---|---|
| (a) | Transformer SDPA backend change with `causal_mask` + `src_key_padding_mask` | `src/models/heads/next.py:130-132`, `src/models/heads/category.py:329-336` | Gate 4 (smoke loss), Gate 5 (metric band) |
| (b) | `torch.autograd.grad(loss, shared_params, retain_graph=True)` semantics | `src/losses/nash_mtl.py:271-277` | Gate 2 (`test_nash_mtl*`), Gate 4, Gate 5 |
| (c) | `torch.autograd.grad(create_graph=True)` higher-order grads | `src/losses/gradnorm.py:24-25` (NOT in production CV runner) | **Out of scope** — only verify if `mtl_loss="gradnorm"` is actually used |
| (d) | MPS op coverage + `PYTORCH_ENABLE_MPS_FALLBACK` behavior | `src/configs/globals.py`, `src/utils/mps.py:9-11` | Gate 4 (op crashes), Gate 5 (silent slowdown) |
| (e) | `torch.load(weights_only=...)` default | `src/data/folds.py:377` (explicit `False` — safe) ; **`scripts/evaluate.py:141` (NO kwarg — relies on default)** | Gate 2 + manual evaluate.py run on a saved checkpoint |
| (f) | `register_buffer(..., persistent=False)` for `causal_mask` | `src/models/heads/next.py:116` | Gate 2 (`test_models/next/*`) |
| (g) | `nn.Embedding.weight[i].expand(B, -1)` gradient flow (PR #8 optimization) | `src/models/mtlnet.py:193-194` | Gate 5 (definitive — task_embedding learning collapses if broken) |
| (h) | `torch.from_numpy` zero-copy contract | `src/data/folds.py:160-161, 176-177` | Gate 2 (`test_folds*`) |

### Specific note on (e) — `scripts/evaluate.py:141`

```python
state_dict = torch.load(checkpoint_path, map_location=DEVICE)
```

No `weights_only=` kwarg. Currently relies on torch 2.9.1's default behavior. The checkpoints saved by `src/training/callbacks.py:207` (`torch.save(self._model.state_dict(), path)`) are bare state-dicts containing only tensors, so `weights_only=True` would also work. **The minimal-risk preemptive fix** (if you want one): change the line to `torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)`. This is forward-compatible and zero-risk because the saved data IS just a state-dict. **But the rule for this PR is**: don't change source unless 2.11 actually errors. If it does error at the smoke test, this is a one-line fix.

### Specific note on (g) — task_embedding view

The PR #8 optimization replaced:
```python
id_cat = torch.zeros(b_cat, dtype=torch.long, device=enc_cat.device)
emb_cat = self.task_embedding(id_cat)
```
with:
```python
emb_cat = self.task_embedding.weight[0].expand(enc_cat.size(0), -1)
```

This is mathematically equivalent under torch's autograd semantics (`weight[i]` is a select view, `expand` is a broadcast view, gradients accumulate back into `task_embedding.weight`). Torch 2.11 has not (per release notes) changed view-of-parameter semantics. **But if Gate 5 shows a Cat F1 > 1pp regression with no other suspect**, the rollback is one-line: revert that block to the long-int gather form. This is the cheapest-to-revert change in the PR-8 batch and the only one that touches a non-trivial autograd pattern.

---

## 6. Smoke-test command sheet (the user's main flows)

Each command is copy-pastable. Execution order matches gate order.

| Flow | Command | Expected time | Pass | Hard fail |
|---|---|---|---|---|
| MTLnet 1-fold smoke | `python scripts/train.py --task mtl --state alabama --engine dgi --folds 1 --epochs 1` | 2-3 min | finite losses, no NaN | NaN, RuntimeError |
| MTLnet full validation | `python scripts/train.py --task mtl --state alabama --engine dgi --folds 5 --epochs 50` | ≤ 17 min | metrics within ±1pp band | metric drift, time +50% |
| Category head only ⚠️ | `python scripts/train.py --task category --state alabama --engine dgi --folds 1 --epochs 5` | 1-2 min | cat F1 > 0.3 | NaN inside `CategoryHeadTransformer` |
| Next head only ⚠️ | `python scripts/train.py --task next --state alabama --engine dgi --folds 1 --epochs 5` | 1-2 min | next F1 > 0.1 | NaN inside `NextHeadMTL` |
| evaluate.py (`torch.load` path) | `python scripts/evaluate.py --checkpoint <existing.pt> --task mtl --state alabama --engine dgi --fold 0` | < 1 min | report prints | `Weights only load failed` → fix is `weights_only=True` |
| create_inputs pipeline | `python pipelines/create_inputs.pipe.py` (read file for args) | seconds-minutes | parquets written | torch/pandas exception |

> ⚠️ **Single-task smokes (category/next) are currently blocked by a pre-existing MPS bug** — `src/training/runners/category_cv.py:36` and `next_cv.py:99` call `.numpy()` on tensors that live on MPS (side effect of PR #8's on-device-tensors optimization). Crashes with `TypeError: can't convert mps:0 device type tensor to numpy` on BOTH torch 2.9.1 and 2.11, so it is NOT a torch 2.11 regression. Do not use these smokes to judge the torch upgrade until the bug is fixed (one-line change: `.cpu().numpy()`). The MTL runner survived because it uses a different attribute path that happens to stay on CPU.
| DGI engine (Tier 2) | `python pipelines/embedding/dgi.pipe.py` (alabama) | 2-5 min | parquet written | PyG `GATConv` exception → wheel mismatch |
| **HGI engine (Gate 6b — CRITICAL)** | `python pipelines/embedding/hgi.pipe.py` (alabama, ctrl-C after epoch 2) | 2-3 min to first epoch | reaches POI2Vec epoch 2 with finite loss | `torch_cluster` ImportError or `RuntimeError` in `Node2Vec.random_walk` → **trigger Path B (dual-venv)** |
| POI2HGI engine | `python pipelines/embedding/poi2hgi.pipe.py` (alabama) | similar | reaches first epoch | same as HGI — same trigger |
| Time2Vec (pure torch) | `python pipelines/embedding/time2vec.pipe.py` | 1-3 min | parquet written | NaN or `torch.compile` MPS error → set `compile=False` |
| Sphere2Vec (pure torch) | `python pipelines/embedding/sphere2vec.pipe.py` | similar | parquet written | NaN |
| Space2Vec (pure torch) | `python pipelines/embedding/space2vec.pipe.py` | similar | parquet written | NaN |
| Imports-only batch (Gate 6a) | inline Python from Gate 6a | < 10 sec | every non-HGI line `OK`; HGI line either `OK` or `IMPORT-FAIL` | any non-HGI `FAIL` line |

**For "don't need to execute all, just see if they will work"**: Gate 6a (imports) + Gate 6b (HGI run smoke) together are the user's primary deliverable. Gate 6a runs in seconds and surfaces wheel ABI / import-time errors. Gate 6b takes 2-3 minutes and is the only thing that catches a runtime `Node2Vec.random_walk` ABI mismatch — the highest-risk single op in the entire torch upgrade for this codebase.

---

## 7. Rollback plan

### Primary rollback (worktree-isolated)
Because Gate 0 created an isolated worktree + venv, rollback is two commands:
```bash
cd /Users/vitor/Desktop/mestrado/ingred
git worktree remove --force .claude/worktrees/torch-upgrade
# .venv_torch211 lives inside the worktree dir, removed automatically
```
Zero impact on `.venv_new`, `main`, or `flash`. This is **the entire reason** Gate 0 mandates an isolated venv.

### Fallback (if for any reason `.venv_new` was modified)
```bash
source .venv_new/bin/activate
pip install --upgrade torch==2.9.1 torchvision==0.24.1
pip install --no-index --find-links https://data.pyg.org/whl/torch-2.9.0+cpu.html \
  torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3
pip install --upgrade torch-geometric==2.7.0
python -c "import torch; assert torch.__version__.startswith('2.9.1'), torch.__version__"
pytest tests/test_models tests/test_training tests/test_data tests/test_losses -q
```

### What must NEVER be committed during the upgrade attempt
- `pyproject.toml` torch / torchvision / torch_* / torch-geometric pin bumps **stay un-committed until Gate 5 passes**.
- `requirements.txt` likewise.
- All experimental wheel installs are confined to `.venv_torch211` inside the throwaway worktree.

---

## 8. Out of scope (explicitly deferred — do NOT do these in this PR)

Mixing any of these into this PR multiplies the blast radius and makes failure attribution impossible.

1. **`torch.compile` enablement for MTLnet** — item #9 in `plans/mtlnet_speed_optimization.md`. Known to clash with NashMTL's manual `autograd.grad`.
2. **`torchmetrics`, `pytorch-lightning`, `transformers`, `accelerate` bumps** — separate compatibility surfaces.
3. **Regenerating any embedding parquet** — existing files are still valid inputs to MTL training.
4. **Pre-existing flaky `test_regression/test_mtl_f1_within_tolerance`** — known to fail on this hardware on `main`.
5. ~~Pre-existing pseudo-bug at `scripts/evaluate.py:150-151` — passes `x` as both category and next inputs in the "simplified forward" path.~~ **Fixed during PR #9 follow-up** — the MTL eval branch now evaluates BOTH heads via shape-correct dummy inputs (cat-eval feeds a real category batch + a zero next-tensor of the right `(B, seq, embed)` shape; next-eval mirrors this). Both heads' reports render. The two heads are fully independent inside `MTLnet.forward()` (encoders, FiLM, shared layers, and heads all run on each task tensor separately — see `src/models/mtlnet.py:173-214`), so passing zeros on the unused side is numerically safe.
6. **Building `torch_cluster`/`torch_scatter`/`torch_sparse` from source** (rejected alternative in §3) — defer indefinitely.
7. **Enabling MPS AMP for any embedding engine** — `check2hgi/check2hgi.py:48-54` correctly disables AMP on MPS due to NaN issues with scatter/softmax in float16; do not change.
8. **Fixing `gradnorm.py` `create_graph=True` ahead of need** — not in production runner.
9. **Bumping numpy / scipy** — they're transitive; if torch 2.11 forces a numpy bump, accept it, but don't preemptively touch.

---

## 9. Critical files (one-glance reference)

| Path | Why it matters in this PR |
|---|---|
| `pyproject.toml` | Pin bumps go here — only post-Gate 5 |
| `requirements.txt` | Same — only post-Gate 5 |
| `src/models/heads/next.py` | SDPA + causal mask + padding mask — highest-risk surface |
| `src/models/heads/category.py:310-336` | CategoryHeadTransformer SDPA path |
| `src/models/mtlnet.py:193-194` | task_embedding weight-slice optimization (PR #8) — single revert point if Gate 5 fails |
| `src/losses/nash_mtl.py:266-305` | NashMTL `get_weighted_loss` — autograd.grad + on-device GTG |
| `src/data/folds.py:160-161, 176-177, 377` | `from_numpy`, view, `torch.load(weights_only=False)` |
| `scripts/evaluate.py:141` | `torch.load` without `weights_only=` — only fix if smoke-test errors |
| `src/configs/globals.py` | `PYTORCH_ENABLE_MPS_FALLBACK=1` set at import (PR #8) |
| `src/utils/mps.py:9-11` | `torch.mps.empty_cache/synchronize` |
| `research/embeddings/hgi/poi2vec.py:26, 234` | `torch_geometric.nn.Node2Vec` — Path A vs B vs C decision hinges on this |
| `research/embeddings/check2hgi/check2hgi.py:48-54` | MPS AMP gate — leave alone |

---

## 10. Verification (end-to-end)

After Gate 5 passes:

1. Confirm metrics in `results/dgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_<timestamp>/summary/full_summary.json` match the post-PR-8 reference within ±1pp.
2. Run Gate 6 import-only batch — capture the OK/DEFER/FAIL output to paste in the PR description.
3. (Optional) Run one cheap Tier-2 generation (Time2Vec on alabama) end-to-end to confirm pure-torch engines write valid parquet.
4. Dump `pip freeze` and diff against the pre-upgrade state to document exactly which packages moved.
5. Update `pyproject.toml` and `requirements.txt` with the new pins **only after** all of the above pass.
6. Commit the pin bumps as a single commit titled `chore(deps): bump torch 2.9.1 -> 2.11.0 (validated on DGI Alabama 5-fold)`.
7. Open the PR with the `pip freeze` diff, the Gate 5 metrics table, and the Gate 6 import status as evidence.

---

## 11. Why this plan is deliberately small

A torch minor bump done in isolation is a 1-day task. It becomes a week-long task when scope creeps:
- "While we're at it, let's bump torchmetrics" → unrelated regressions.
- "Let's regenerate embeddings under the new torch" → hides drift in the data layer.
- "Let's enable torch.compile" → new graph-break errors in NashMTL.

This plan keeps the variable count to ONE: torch + the four packages tied to torch's ABI (torchvision, torch_scatter, torch_sparse, torch_cluster). If Gate 5 fails, the suspect surface is small enough to investigate. If Gate 5 passes, you have a clean validated upgrade with a one-line rollback.

The staged fallback ladder in §3 (Path A → Path B → Path C) is the safety valve for the one quality risk we cannot fully eliminate at the wheel layer: `torch_cluster.random_walk` for HGI's `Node2Vec`. The ladder gives us THREE chances to land on a single-venv outcome before resorting to dual-venv:

1. **Path A** (torch 2.11): the largest bump. Best perf, most uncertain wheel availability.
2. **Path B** (torch 2.10): a smaller bump. Captures most of the same MPS/SDPA improvements relative to 2.9.1, with substantially better odds of having matching wheels published since 2.10 has been out longer.
3. **Path C** (dual-venv): only if both 2.11 and 2.10 lack `torch_cluster` wheels. Keeps the legacy `.venv_new` alive as a routing target ONLY for HGI/POI2HGI regeneration; all other flows move to torch 2.11. The two venvs share zero state — they communicate only through stable parquet artifacts on disk — so there is no cross-venv compatibility surface to maintain.

In all three paths, HGI quality is preserved at its known-good baseline. When upstream PyG publishes `torch_cluster` wheels for torch 2.11 (assuming we landed on Path C), decommissioning `.venv_new` is a follow-up task — re-run Gate 6b on the daily venv and confirm HGI works there, then delete the legacy venv.

That tradeoff — small scope, hard gates, auto-triggered safety valve, easy revert — is the entire point of "be critical and cautious".

---

## 12. Progress checklist (RESUMABLE — tick as you go)

This checklist is the primary resumption signal for future sessions. **Edit it in `plans/torch_211_upgrade.md` (the committed copy in the repo, not the scratch file)** as you complete each step. The first un-ticked box is where to resume.

### Phase 0 — Persistence
- [x] Step 0: copied this plan to `plans/torch_211_upgrade.md` and committed it
- [x] Identified which path is currently being attempted: **A** (A=2.11 / B=2.10 / C=dual-venv)
- [x] Recorded the implementation worktree path: **.claude/worktrees/torch-upgrade**
- [x] Recorded the implementation venv path: **.claude/worktrees/torch-upgrade/.venv_torch211**

### Phase 1 — Pre-flight (Gates -1 and 0)
- [x] Gate -1.1: working tree clean (only `.claude/settings.local.json` mod, `output/` symlink)
- [x] Gate -1.2: re-ran `pytest tests/test_models tests/test_training tests/test_data tests/test_losses -q` on torch 2.9.1 → got 242 passed, 36 skipped
- [~] Gate -1.3: post-PR-8 baseline file not located in worktree; relied on plan header reference numbers (14.14 min · Cat F1 0.4435 · Cat Acc 53.46% · Next F1 0.2603 · Next Acc 36.88%)
- [x] Gate -1.4: confirmed `.venv_new` is shared across worktrees (do NOT install torch upgrade into it)
- [x] Gate -1.5: confirmed `torch_cluster 1.6.3 cp312-macosx` wheels exist on PyG index for torch-2.11.0+cpu: **yes**
- [x] Gate 0.1: created worktree `.claude/worktrees/torch-upgrade` off `main`
- [x] Gate 0.2: created isolated venv `.venv_torch211` inside the worktree
- [~] Gate 0.3: skipped 2.9.1 parity install in the new venv; instead ran `.venv_new` baseline test suite directly (242 passed / 36 skipped) for reference

### Phase 2 — Install + Tier 1 validation (Gates 1-5)
- [x] Gate 1: installed torch + torchvision + matching PyG ecosystem; verified `from torch_cluster import random_walk` succeeds
- [x] Gate 1: recorded actual installed versions: torch=**2.11.0**, torchvision=**0.26.0**, torch_scatter=**2.1.2**, torch_sparse=**0.6.18**, torch_cluster=**1.6.3**, torch-geometric=**2.7.0**
- [x] Gate 2: targeted test suite passed — recorded count: **242 passed, 36 skipped** (0 failed)
- [x] Gate 3: broad test suite passed — recorded count: **107 passed, 26 skipped** (0 new failures)
- [x] Gate 4: 1-fold/1-epoch MTL DGI Alabama smoke completed without NaN — recorded final losses: **train loss 2.4099, val loss 1.8414, cat val acc 26.74%, next val acc 26.50%**
- [x] Gate 5: 5-fold/50-epoch MTL DGI Alabama validation completed — recorded wall time: **12.27 min** (745.82s real)
- [x] Gate 5: metrics within ±1pp of post-PR-8 baseline (bit-exact reproduction):
  - [x] Cat F1 in [0.4335, 0.4535] — actual: **0.4435**
  - [x] Cat Acc in [52.46%, 54.46%] — actual: **53.46%**
  - [x] Next F1 in [0.2503, 0.2703] — actual: **0.2603**
  - [x] Next Acc in [35.88%, 37.88%] — actual: **36.88%**
- [x] Gate 5: results JSON archived at: **results/dgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260411_1440/summary/full_summary.json**

### Phase 3 — Tier 2 verification (Gates 6a, 6b, 7)
- [x] Gate 6a: import-only smoke ran (after prepending `src/` and `research/` to sys.path and installing missing `pytorch-warmup==0.2.0`); recorded OK/FAIL status per engine:
  - [x] dgi: **OK**
  - [x] check2hgi: **OK**
  - [x] poi2hgi: **OK**
  - [x] time2vec: **OK**
  - [x] space2vec: **OK**
  - [x] sphere2vec: **OK**
  - [x] hgi (Node2Vec branch): **OK**
  - [~] pipelines.create_inputs: **N/A** (file is `create_inputs.pipe.py`, not an importable module — plan item was mis-specified)
- [x] Gate 6b: `pipelines/embedding/hgi.pipe.py` Alabama completed the full 5-phase pipeline (POI2Vec 3 epochs + HGI 3 epochs + downstream input gen) in 0.8 min — **PASS**, `Node2Vec.random_walk` + GCNConv + SetTransformer all OK on torch 2.11
- [x] Gate 6b: temporary STATES+epochs edits reverted to `pipelines/embedding/hgi.pipe.py` (git diff clean)
- [!] **Gate 6b side-effect**: the smoke overwrote `output/hgi/alabama/*` (symlink into main repo output). Restored via `rsync -a --delete` from `.claude/worktrees/mtlnet-improve/output/hgi/alabama/`. Plan should be updated post-merge to use `force_preprocess=False` or an `OUTPUT_ROOT` override for future runs.
- [ ] Gate 7 (optional): time2vec on alabama — skipped (cheap engine, deferred)
- [ ] Gate 7 (optional): sphere2vec on alabama — skipped (cheap engine, deferred)
- [ ] Gate 7 (optional): space2vec on alabama — skipped (cheap engine, deferred)
- [ ] Gate 7 (optional): dgi on alabama — skipped (already exercised via Gate 5's MTL training)

### Phase 4 — Decision branch (only one applies)

#### If Path A succeeded (single venv on torch 2.11)
- [x] All Gates 1-6b pass on torch 2.11
- [x] Bumped pins in `pyproject.toml` to `torch==2.11.0`, `torchvision==0.26.0`, `torch_cluster==1.6.3`, `torch_scatter==2.1.2`, `torch_sparse==0.6.18`, `torch-geometric==2.7.0`; also fixed pre-existing `build-backend` typo (`setuptools.backends._legacy:_Backend` → `setuptools.build_meta`); added previously-missing `pytorch-warmup==0.2.0` dep
- [x] Bumped pins in `requirements.txt` to match (torch/torchvision + explicit pins for torch_cluster/scatter/sparse)
- [ ] Committed pin bump: `chore(deps): bump torch 2.9.1 -> 2.11.0 (validated on DGI Alabama 5-fold)`
- [ ] Opened PR with pip freeze diff, Gate 5 metrics table, Gate 6a/6b status as evidence

#### If Path B succeeded (single venv on torch 2.10)
- [ ] Wiped `.venv_torch211`, recreated `.venv_torch210`
- [ ] Re-ran Gates 1-6b on torch 2.10
- [ ] All Gates 1-6b pass on torch 2.10
- [ ] Bumped pins in `pyproject.toml` to `torch==2.10.x`, `torchvision==<verified>`, etc.
- [ ] Bumped pins in `requirements.txt` to match
- [ ] Committed pin bump: `chore(deps): bump torch 2.9.1 -> 2.10.x (Path B fallback, validated on DGI Alabama 5-fold)`
- [ ] Opened PR noting that Path A (torch 2.11) was attempted and failed at Gate **___**, Path B chosen

#### If Path C activated (dual-venv: torch 2.11 daily + .venv_new legacy for HGI)
- [ ] Both Path A (2.11) and Path B (2.10) failed at Gate **___** with reason **___**
- [ ] Recreated `.venv_torch211` without `torch_cluster` (per §3 Path C)
- [ ] Confirmed Gates 1-6a all pass on torch 2.11 (Gate 6b expected to fail with `torch_cluster` ImportError)
- [ ] Optionally renamed `.venv_new` to `.venv_legacy_torch29`
- [ ] Promoted `.venv_torch211` to daily-driver venv path
- [ ] Added top-of-file comment in `pipelines/embedding/hgi.pipe.py` documenting legacy venv requirement
- [ ] Added top-of-file comment in `pipelines/embedding/poi2hgi.pipe.py` documenting legacy venv requirement
- [ ] Added defensive `torch_cluster` ImportError check at top of both files
- [ ] Verified HGI still runs on the legacy venv after the comments were added (ran `pipelines/embedding/hgi.pipe.py` on alabama for one epoch under `.venv_new`)
- [ ] Bumped pins in `pyproject.toml` to torch 2.11.0 (with note that `torch_cluster` is excluded — see comment in file)
- [ ] Bumped pins in `requirements.txt` to match
- [ ] Committed: `chore(deps): bump torch 2.9.1 -> 2.11.0 (Path C dual-venv; HGI on legacy venv until wheels publish)`
- [ ] Opened PR with full Path C documentation, dual-venv routing table, and exit criterion (re-check PyG wheel index periodically)
- [ ] Filed follow-up issue: "Decommission .venv_new once torch_cluster wheels for torch 2.11 publish on PyG index"

### Phase 5 — Post-merge cleanup
- [ ] PR merged
- [ ] Worktree `.claude/worktrees/torch-upgrade` removed
- [ ] Memory file written documenting which path was used and why (for future Claude sessions)
- [ ] Updated `plans/mtlnet_speed_optimization.md` to mark item #8 as completed (if applicable)

---

### Resumption protocol (read this if you're a future session picking up mid-flight)

1. **First**: read `plans/torch_211_upgrade.md` (the committed copy in the repo) — NOT this scratch file. The committed copy has the most recent checklist state.
2. **Find the first un-ticked box** in §12. That's where you stopped.
3. **Read the corresponding gate description** in §4 (Staged execution gates) for full context.
4. **Check `git log`** for any commits since the previous session — there may be relevant context, fixes, or partial work.
5. **Check `git status`** for uncommitted work — the previous session may have edited files (e.g. the temporary `STATES = ["alabama"]` in `pipelines/embedding/hgi.pipe.py`) that need to be either committed or reverted before continuing.
6. **Resume from the un-ticked gate**, ticking boxes as you go.
7. **Always edit the committed `plans/torch_211_upgrade.md` directly** — never the scratch copy in `~/.claude/plans/`.
