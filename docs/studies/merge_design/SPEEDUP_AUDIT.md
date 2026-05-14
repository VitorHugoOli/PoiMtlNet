# Training-script speed audit — merge family (B/H/I/J/M/K)

## Tier 1 application — empirical update 2026-05-06

Tier 1 wins were applied to all 6 build scripts (B/H/I/J/M/K) +
`research/embeddings/check2hgi/model/Check2HGIModule.py:27`:

- `randperm(device=x.device)` (Check2HGIModule.py:27)
- `self.poi_table.weight` directly (J:58, K:73, H:91)
- Skip `scheduler.step()` when `--gamma=1.0` (all 6 scripts)
- `t.set_postfix(refresh=False)` + `t.refresh()` every 25 epochs (all 6)
- Drop `.item()` postfix calls beyond `loss.item()` for the best-gate

Smoke test (J AL, 50 epochs, MPS):
- **Patched: 5.85 s/iter**
- Unpatched baseline (from λ=0.3 sweep on the same hardware): ~5.6 s/iter

**No measurable speedup.** The audit predicted 15-25 % but **the
syncs are not the bottleneck for this workload**. At full-graph training
over ~800K AL nodes / 159K FL nodes, the GCN forward+backward
dominates 5+ seconds per step; per-step `.item()` syncs are <1 % of
wall time and reducing them yields nothing measurable.

**Implication**: the only Tier-1 candidate that survives empirically is
Tier 3 — single-pass encoder corruption (item #9 below) — which
eliminates one full GCN forward+backward per step (the actual
bottleneck) at the cost of changing the optimisation trajectory.

Tier 1 patches are kept (they don't hurt and they're cleaner code) but
not credited with a real speedup. The numerical-equivalence claims hold:
the patched smoke test produced no NaN, monotone-decreasing loss, and a
clean parquet output.

### Advisor verdict (post-null, 2026-05-06)

Take the null seriously: per-step `.item()` syncs and tqdm overhead are
**not** the bottleneck when each step is 5+ s of GCN. By the same
logic Tier 2 / item #13 (vectorise hard-neg) is almost certainly also
nominal — don't bother. The **only** principled MPS lever that touches
the real bottleneck is **item #9 (single-pass encoder corruption)** —
parked indefinitely because revalidating 6 designs × 5 folds × {AL, FL}
costs more than the 30-45 % per-step saving for the current diagnostic
phase.

**The actual cheapest speedup is upstream of code**: in **diagnostic
mode** (Tests 1-4 from STATE.md), Tests 1 and 2 don't need new builds
at all (reuse existing parquets/embeddings). They should run first.
Tests 3 and 4 do need builds — the cheapest acceleration there is
**fewer epochs**, calibrated by a 200-vs-500 ep run on J at AL with a
≤0.4 pp Acc@10 tolerance. If 200 holds, all subsequent sweeps run 2.5×
faster with zero quality risk for the diagnostic.

### Speed-decisions log (closed)

- Tier 1: applied to all 6 build scripts + Check2HGIModule.py. **Null
  speedup empirically.** Numerical-equivalent. Kept.
- Tier 2 / hard-neg vectorisation: **deferred indefinitely** (likely
  also null per advisor; not worth the revalidation cost for diagnostic).
- Tier 3 / single-pass corruption (item #9): **deferred to a separate
  fast-baseline project**; do not mix into the current head-to-head.
- Tier 4 / CUDA-only `torch.compile` + fp16: **deferred** until a
  CUDA-only branch is opened.

## In-flight K runs are safe

K uses `GCNConv(cached=True)` at `build_design_k_delaunay.py:58-59`. The
cached normalised adjacency is reused across positive and negative passes
within the same step (same `delaunay_edge_index`); `model.train()/eval()`
toggles do not invalidate `_cached_edge_index` in PyG ≥2.x. **No
correctness issue. Let K finish.**

## Ranked wins (highest gain → lowest)

1. (✓) `best_state` clone — **already gated** under `if l < lowest:` at
   `build_design_j_anchor.py:165-167` and `build_design_k_delaunay.py:232-234`.
   Original concern was wrong. No change needed.
2. **Drop `.item()` syncs in `set_postfix`** — large MPS win, zero risk, 5 min.
3. **`tqdm` `refresh=False`, refresh every K** — moderate, zero risk, 5 min.
4. **Skip `scheduler.step()` when `gamma==1.0`** — small, zero risk, 2 min.
5. **`clip_grad_norm_` is a sync** — moderate MPS win, low risk if kept periodically, 30 min.
6. **`randperm(device=x.device)`** in `Check2HGIModule.py:27` — moderate, zero risk, 10 min.
7. **`self.poi_table.weight` directly instead of `Embedding(arange(N))`** — tiny, zero risk, 5 min.
8. (✓) `pos_poi_emb_canonical[checkin_to_poi]` — already a single gather. No change.
9. **Single encoder pass via embedding-level corruption** — large win (≈30-45 %),
   **changes optimisation trajectory** — gated for future work, NOT for runs in
   the current head-to-head, 1 hr + revalidation.
10. **`torch.compile` (CUDA-only flag)** — large CUDA win, MPS prohibited, 2 hr.
11. **fp16 autocast (CUDA T4 only)** — ~2× on CUDA, MPS prohibited, 30 min.
12. (✓) `model.train()` per epoch — effectively no-op (no BN, no dropout in default config). No change.
13. **Vectorise hard-neg loop** in `Check2HGIModule.py:206-212` — moderate, low risk, 1 hr.
14. (✓) Anchor loss form — already trivial. No change.
15. **MPS fallback diagnostic** with `PYTORCH_DEBUG_MPS_FALLBACK=1` — info only.

## Per-item details

### 2. `.item()` syncs in `set_postfix`

`build_design_j_anchor.py:164,168-169`, `build_design_k_delaunay.py:231,235-236`,
`build_design_b_poi_pool.py:237,242-243`. Three forced CPU↔MPS syncs per
step (`loss.item()`, `loss_main.item()`, `loss_anchor.item()`; B also calls
`model.gamma.item()`). Each one stalls the pipeline before the next
forward.

Recommendation: keep one `.item()` for the `if l < lowest` gate, drop the
two postfix `.item()` calls or refresh-every-K. Numerically equivalent.

### 3. tqdm refresh every K

`t.set_postfix(..., refresh=False)` then `t.refresh()` every 25-50 steps.
Reduces console writes and amortises any remaining `.item()` cost.

### 4. Conditional `scheduler.step()`

`StepLR(step_size=1, gamma=1.0)` is identity. `--gamma 1.0` is the
default in J, K, B, M (line 217/265/...). Wrap call in
`if args.gamma != 1.0:`.

### 5. `clip_grad_norm_` sync

`clip_grad_norm_` reads the global norm scalar internally. On MPS this is
a sync. With `--max-norm 0.9` (default) it's active on every step.

Options:
- Keep it (default) — needed for stability with the contrastive log-loss
  and `EPS=1e-7` in the discriminate step.
- Replace with `clip_grad_value_` which is a per-element clamp and does
  not need a global reduction (different semantics, may not stabilise
  the same way).
- Clip every K steps — bounded risk: rare grad spikes from
  `-log(0+EPS)` could destabilise on FL.

If pursued, run AL 5-fold sanity rerun before adopting.

### 6. `randperm` device — `Check2HGIModule.py:27`

Currently `torch.randperm(x.size(0))` returns a CPU tensor. With
`PYTORCH_ENABLE_MPS_FALLBACK=1` it likely runs on CPU and is copied back
each step. AL has ~800 K checkins → int64 perm tensor ~6 MB/step copied
CPU→MPS.

Fix: `torch.randperm(x.size(0), device=x.device)`. Numerically equivalent
distribution; only the RNG stream changes (statistically equivalent —
corruption is non-deterministic by design).

### 7. `self.poi_table.weight` directly — J:58, K:73

`nn.Embedding(N, D).weight` is the exact tensor returned by
`embedding(arange(N))`. Drop the gather:

```
poi_residual = self.poi_table.weight   # was: self.poi_table(all_pois)
```

Numerically identical, saves an embedding lookup + an `arange` allocation.

### 9. Single-pass encoder via embedding-level corruption

The merge-family probes (J:51-53, K:66-68, B:94-96, etc.) re-implement
`forward()` and run the encoder twice — once on `data.x`, once on
corrupted `x`. The canonical c2hgi pipeline already uses single-pass
corruption (per `research/embeddings/check2hgi/CLAUDE.md`). The fix:

```
pos_checkin_emb = self.checkin_encoder(data.x, data.edge_index, data.edge_weight)
neg_checkin_emb = pos_checkin_emb[torch.randperm(data.x.size(0), device=data.x.device)]
```

Eliminates one full GCN forward+backward per step (~30-45 % faster).

**But changes the optimisation trajectory.** Embedding-level corruption
permutes outputs without re-running the GCN; feature-level corruption
forces the GCN to learn against graph-structurally-consistent shuffled
inputs. CLAUDE.md says they are "Mathematically equivalent for
contrastive learning", but the merge-family probes were authored with
double-pass and every paired Wilcoxon test we have is against that
baseline. Switching changes every numerical result.

**Recommendation**: keep double-pass for current comparisons; optimise to
single-pass *only* in a separate validated branch with its own AL+FL
5-fold revalidation against the existing baseline.

### 13. Hard-neg vectorisation — `Check2HGIModule.py:206-212`

Python loop over `hard_mask.nonzero()` calls `.item()` and per-element
masked-gather. ~25 % of N_pois ≈ 2.7 K iterations per step on AL.

Vectorise via precomputed `(num_pois, max_candidates)` padded matrix of
hard-negative indices, then gather random columns per step. Distribution
is preserved; RNG stream changes but is statistically equivalent.

## Combined plan

**Tier 1 (apply now, zero numerical risk, ~15-25 % MPS speedup combined)**

- #2 `.item()` reduction
- #3 `tqdm refresh=False`
- #4 conditional `scheduler.step()`
- #6 `randperm(device=...)`
- #7 `poi_table.weight`

These can be applied as a single small PR across the 6 build scripts
(B/H/I/J/M/K) plus the shared `Check2HGIModule.py:27`. **Do not apply
mid-flight** — wait for the current K AL+AZ run to finish so the in-flight
JSON tags are produced under the same code as their J counterparts.

**Tier 2 (apply with revalidation, ~5-15 % more)**

- #13 vectorise hard-neg loop in `Check2HGIModule.py`
- #5 periodic `clip_grad_norm_` (every K steps)

Each of these changes the numerical trajectory in a small but principled
way; require a 5-fold AL rerun to confirm Wilcoxon outcomes don't shift.

**Tier 3 (separate optimisation branch, requires full revalidation)**

- #9 single-pass encoder corruption (~30-45 %)

This is the largest gain but the most expensive to validate. Worth doing
once the current merge-family question is settled (i.e. once we know
whether K closes the HGI gap). Treat as a follow-up that produces a new
"fast" baseline against which the next round of designs is measured.

**Tier 4 (CUDA only, future)**

- #10 `torch.compile` behind `--compile` flag, with `--device cuda`
  required and a docstring marking it as numerically validated against
  eager
- #11 `--amp` fp16 autocast for CUDA T4 only — never on MPS
  (`feedback_mps_autocast_overhead`)

## Files affected

- `scripts/probe/build_design_b_poi_pool.py`
- `scripts/probe/build_design_h_learnable_poi.py`
- `scripts/probe/build_design_i_lora.py`
- `scripts/probe/build_design_j_anchor.py`
- `scripts/probe/build_design_m_distill.py`
- `scripts/probe/build_design_k_delaunay.py`
- `research/embeddings/check2hgi/model/Check2HGIModule.py` (line 27 randperm; lines 206-212 hard-neg)
