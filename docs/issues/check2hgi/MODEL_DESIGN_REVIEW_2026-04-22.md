# Model / Optimizer Design Review — 2026-04-22

**Severity:** LOW–MEDIUM — these are design smells and science-quality limitations, not bugs. None contaminate existing results. Each either (a) caps how strong the paper's claim on a specific head can be, or (b) leaves a known-flaky path in place that will bite future work.

**Detected:** 2026-04-22, pre-P5 critical review of models + optimizers (same review as `MTL_PARAM_PARTITION_BUG.md` and `CROSSATTN_PARTIAL_FORWARD_CRASH.md`; this file holds the non-blocker items).

**Status:** OPEN — prioritise after the partition bug is fixed and the contaminated runs are regenerated. The items below are ordered by expected impact on paper claims.

---

## 1. GETNext probe has no direct supervision

**Where:** `src/models/next/next_getnext/head.py:120, 150-154`.

**What:** The trajectory-flow prior multiplies `softmax(region_probe(last_emb))` by `log_T`. The probe is `nn.Linear(embed_dim, num_classes)` with **no auxiliary CE loss** against the true last-step region label. Its only gradient path is "contribution to the final region-classification loss via `α · (probs @ log_T)`". That is a long, indirect signal — the gradient has to traverse the softmax-over-1000-regions → matmul → sum → final logits before reaching the probe.

**Expected consequence:** the probe likely stays close to its random init; the GETNext lift you observed (B-M6b: 56.49 ± 4.25 Acc@10, +11 pp vs GRU) comes primarily from the **STAN backbone**, not from the transition-flow prior. The ALiBi variant (B-M6d: 57.46 ± 3.66) supports this reading — ALiBi stabilises the backbone, and the marginal lift over vanilla GETNext is plausible without any prior contribution.

**Test:** After the P5 batch is in, ablate the prior by loading the trained GETNext checkpoint with `α = 0` (or `log_T = 0`). If Acc@10 drops by ≤ 1 pp, the prior is not doing material work → retract the GETNext story, rename the head to "next_stan_v2" or similar. If it drops by > 2 pp, the prior *is* contributing; keep GETNext but add direct supervision in the next iteration.

**Fixes, in order of preference:**
1. **Hard index** (cleanest; matches GETNext paper): plumb `last_region_idx` through `next_region.parquet` and replace the soft probe with `log_T[last_region_idx]`. Requires +10 LOC in `src/data/inputs/next_region.py` to include the column, +5 LOC in the head. Makes the prior **deterministic** and aligned with the paper's formulation.
2. **Auxiliary CE**: add a supervision loss `aux = CE(region_probe(last_emb), last_region_idx)` with a small weight (e.g. 0.1), returned alongside the main logits. Still requires the label in the dataset; marginally softer than hard-index because the probe has to learn the last-embedding → region mapping rather than have it given.
3. **Do nothing** + ablate α → if the ablation shows the prior is dead weight, revert to `next_stan` and drop GETNext from the paper.

---

## 2. MoE variants (CGC / MMoE / DSelectK / PLE) have no load-balancing penalty

**Where:** `src/models/mtl/_components.py:131-136, 192-195, 296-302` (entropy is logged but never added to the loss).

**What:** The three mixer classes compute per-task gate entropy and log it as a diagnostic, but the training loss is pure task-loss + MTL-balancing. Without a load-balancing or entropy-bonus term, gates are free to collapse onto one or two experts — the classic MoE failure mode that Switch Transformer / GShard / DSelect-k all patch with an auxiliary term.

**Expected consequence:** you are likely under-using the expert count. CGC / MMoE / PLE headline numbers may improve by 0.5–1.5 pp with a ~0.01-weighted load-balancing term; DSelectK especially, because its design target is genuine sparsity (see item 3).

**Action:** add a `load_balance_coefficient: float = 0.01` config field. In the MTL forward, add `loss += load_balance_coefficient * (1.0 - mean_gate_entropy / log(num_experts))`. Start with 0.01, ablate to zero to check the lift. Not a blocker for the current paper but worth a one-pass sweep on AL post-bugfix.

---

## 3. DSelectK is not actually sparse

**Where:** `src/models/mtl/_components.py:272-294`.

**What:** The implementation does two softmaxes — `softmax(selector_logits / temperature)` over experts for each of `K` selectors, then `softmax(selector_weights)` mixing `K` selectors — and averages them. The output is a **dense** convex combination over all `N` experts. The paper's selling point (sparse top-k routing) is not implemented.

The normalisation step `weights = weights / weights.sum(-1, keepdim=True)` at line 293 is a no-op: a convex combination of simplex vectors is already on the simplex.

**Expected consequence:** "DSelectK" in this codebase behaves like a multi-softmax MMoE with extra plumbing. The P2 screen leaderboard's "DSelectK is dominant" finding does not conflict with this — it just means the extra flexibility of a K-selector gate helps, not that sparsity helps.

**Action (lowest-risk):** rename the class to something honest, e.g. `MTLnetMultiGate`, and drop the "DSelect-k style" language in the docstring. This avoids over-claiming in the paper. **Higher-risk (do later):** implement actual top-k routing via Gumbel-top-k + straight-through, and re-run the A-M2 / B-M2 row.

---

## 4. STAN `pair_bias` has full DOF with no regularisation

**Where:** `src/models/next/next_stan/head.py:82, 95`.

**What:** The pairwise bias is a `[num_heads, seq_length, seq_length] = [4, 9, 9] = 324` parameter matrix per attention block, Gaussian-init at `std=0.02`, no weight-decay, no structural prior. Nothing stops individual entries from growing arbitrarily during training; overfit risk is real, especially on AL (smaller state).

**Empirical alignment:** the AZ ALiBi finding (commit `f1ea416`: "AZ ALiBi confirms scale-dependent σ reduction") showed ALiBi init reduces variance. ALiBi constrains the bias shape to a monotonic recency prior — exactly the kind of constraint that a fully-free `pair_bias` lacks. The σ reduction is consistent with "free bias was overfitting, ALiBi regularises it."

**Action:** make `bias_init="alibi"` the **default** in `NextHeadSTAN` (currently `"gaussian"`). Alternatively, add weight-decay specifically on `pair_bias` (exclude it from the no-decay list of LayerNorms / biases). One-line change in the default kwarg; re-run STAN configs if you want the numbers under the new default. Not a re-run for the paper because the best STAN+GETNext numbers (B-M6d) already use ALiBi.

---

## 5. AdaShare has no temperature annealing and no sparsity pressure

**Where:** `src/models/mtl/mtlnet/model.py:88, 214, 363-379`.

**What:** Gumbel-sigmoid temperature is fixed at `0.5` throughout training; gates init at logit=2 (`sigmoid(2) ≈ 0.88`); no L0 / sharing-cost term in the loss.

**Expected consequence:** even *after* the partition bug is fixed (so gates actually train), gates will tend to stay near "always on" because (a) no pressure to close them, (b) no temperature annealing to push them to near-discrete values. AdaShare will behave close to baseline MTLnet.

**Expected impact on the re-run:** the post-bugfix AdaShare result is likely still neutral, just for a different reason than before. The bug re-run will tell us whether gates get any useful signal; if they do but still converge near "always on", this item kicks in.

**Action (after the partition-bug re-run):** if post-fix AdaShare is still neutral, add (a) temperature annealing from 2.0 → 0.2 over the first 30 epochs, (b) an L0 cost `λ_sparsity * mean(sigmoid(adashare_logits))` with `λ_sparsity ≈ 0.01`. If still neutral, retire AdaShare from this study.

---

## 6. DSelectK + MTLoRA + α-skip are composed without an isolation study

**Where:** `src/models/mtl/mtlnet_dselectk/model.py:140-143`.

**What:** Three parallel paths summed — `shared_cat + lora_cat + α · enc_cat`. Once the partition bug is fixed, all three train. The current paper narrative ("MTLoRA closes the architectural overhead") is not supported by a single-term ablation — we don't know whether the LoRA branch, the α-skip, or both together are doing the work. Even the commit message (`a9309cb`) notes "α gets stuck at 0" as observed behaviour, but that observation was made **under the partition bug** where α never received a gradient at all.

**Action (after the partition-bug re-run):** add two more rows to the ablation table:
- `MTLoRA only` (force `skip_alpha_*` to be non-trainable / zeroed)
- `α-skip only` (force `lora_B_*` to be non-trainable / zeroed)
Both as 5f × 50ep on AL at fair LR. ~1 hour each on the P7 machine. Do this alongside re-runs 1–6 in `MTL_PARAM_PARTITION_BUG.md` so the contamination fix and the isolation story land together.

---

## 7. `shared_parameters` / `task_specific_parameters` use fragile substring matching

**Where:** every MTL variant, e.g. `src/models/mtl/mtlnet/model.py:516-538`.

**What:** The partition is implemented by iterating `named_parameters()` and matching against a hand-maintained list of substrings. `MTL_PARAM_PARTITION_BUG.md` is the consequence — a new `nn.Parameter` or submodule can be added to a class without its name matching any substring, and the partition silently breaks.

**Action:** refactor to an explicit registration pattern. Each variant's `__init__` appends its newly-installed submodule / parameter names to `self._shared_param_names` and `self._task_specific_param_names`. The iterators then look up by **exact name**, not substring. Subclasses inherit their parent's registrations and add their own. This makes item 1 in `MTL_PARAM_PARTITION_BUG.md` impossible by construction.

This is a larger refactor (~50 LOC across 6 classes). Not a blocker for the current paper. Put it on the P5 / post-paper cleanup list.

---

## 8. GRU consumes pad steps unmasked

**Where:** `src/models/next/next_gru/head.py:38`.

**What:** `self.gru(x)` processes the full sequence including zero-pad steps. For right-padded inputs the GRU hidden state keeps updating on zeros *after* the real data, wasting compute and mildly drifting the extracted "last valid output" even though the extraction itself is correct.

**Expected consequence:** small — right-padded GRUs over 9 steps with typically ≤ 2 pad steps per row. No paper claim affected.

**Action (optional):** use `torch.nn.utils.rnn.pack_padded_sequence` if you want to revisit. Not worth the disruption mid-paper.

---

## 9. Grad-cosine diagnostic is sampled at batch 0 only

**Where:** `src/training/runners/mtl_cv.py:350`.

**What:** `_compute_gradient_cosine` runs once per epoch, on batch 0 — a single noisy observation. For the "task conflict" story that several docs reference (e.g. `BACKBONE_DILUTION.md`), an epoch-level mean over 8–16 sampled batches would give a usable signal; a single point does not.

**Action:** rotate the diagnostic batch across the epoch, or sample 8 batches randomly per epoch and average. ~10 LOC. Gives you a much stronger figure for the paper's "gradient-conflict" narrative if you end up telling that story.

---

## 10. CUDA autocast path has no GradScaler (dormant — MPS branch used)

**Where:** `src/training/runners/mtl_cv.py:267-271, 376`.

**What:** `torch.autocast(DEVICE.type, dtype=torch.float16)` is entered on CUDA but no `torch.cuda.amp.GradScaler` is set up; FP16 gradient underflow can silently diverge training. The user's machine runs MPS per `AGENT_CONTEXT.md`, so this branch is dormant today.

**Action (if / when CUDA gets used):** add a `GradScaler`, or switch the CUDA autocast dtype to `torch.bfloat16` (no scaler needed on Ampere+). Document this in `AGENT_CONTEXT.md` so a future collaborator on a CUDA workstation doesn't silently hit this.

---

## Summary

| # | Item | Impact if ignored | Work |
|---|---|---|---|
| 1 | GETNext probe has no supervision | GETNext story may not survive ablation | 1h fix + 1 re-run |
| 2 | MoE no load-balancing | Leaves 0.5–1.5 pp on the table | 20 LOC + sweep |
| 3 | DSelectK not sparse | Over-claim in paper; rename fixes | Rename only |
| 4 | STAN `pair_bias` unregularised | Already mitigated by ALiBi default-to-change | Flip default |
| 5 | AdaShare no annealing / sparsity | Predicts post-bugfix AdaShare still neutral | After re-run |
| 6 | DSelectK+MTLoRA+α-skip not isolated | Can't attribute MTLoRA lift | Post-bugfix re-run |
| 7 | Fragile substring partition | Future bug recurrence | Refactor, post-paper |
| 8 | GRU pad unmasked | Negligible | Optional |
| 9 | Grad-cosine batch-0 only | Weakens gradient-conflict figure | 10 LOC |
| 10 | CUDA AMP no GradScaler | Dormant on MPS | When moving to CUDA |

---

## References

- Sibling issues from the same review:
  - `MTL_PARAM_PARTITION_BUG.md` — blocker, 6 re-runs required
  - `CROSSATTN_PARTIAL_FORWARD_CRASH.md` — blocker on eval path, no re-runs
- Related prior audits:
  - `BACKBONE_DILUTION.md` — items 2, 6, 9 here interact with that characterisation
  - `REGION_HEAD_MISMATCH.md` — item 4 (STAN pair_bias) is the current-generation region-head story
