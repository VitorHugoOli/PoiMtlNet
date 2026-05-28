# Deferred-work priority & impact ranking — mtl-protocol-fix

**Drafted:** 2026-05-21
**Purpose:** Convert [`DEFERRED_WORK.md`](DEFERRED_WORK.md)'s inventory (7 deferred items + 3 cross-study Levers) into an EV-ranked execution plan, grounded in current artefact availability (A40 idle, HGI substrate only at AL/AZ on disk, Designs J/B/H/M/K/L6 substrate parquets NOT on disk, per-fold val CSVs partially lost in post-Phase-2 disk cleanup).
**Reads:** [`DEFERRED_WORK.md`](DEFERRED_WORK.md), [`log.md`](log.md) 2026-05-20 P5/P6 entry, [`docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md`](../../results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md), [`docs/studies/merge_design/STATE.md`](../merge_design/STATE.md) + [`LEVER_6_FINDINGS.md`](../merge_design/LEVER_6_FINDINGS.md).

**Scope (user direction 2026-05-21):**
- **Only items NOT already executed.** Re-running closed/falsified work is wasted compute.
- **States: AL/AZ + FL only.** CA/TX skipped (substrate + MTL closure deems large-state runs not worth the ~5-7 h GPU each).
- **Single-seed=42 first.** Multi-seed only on a promotion candidate.

---

## Headline ranking (EV / cost / risk-adjusted)

| Rank | Item | Cost | Expected lift | Confidence | Tier |
|---:|---|---:|---:|---|---|
| 1 | §4.5 — log_T as supervisory signal (reg-head loss term) — AL/AZ+FL | ~2-4 GPU-h | 0.5-2 pp at FL reg | Medium | **T1** |
| 2 | §4.6 — Class-balanced batch sampler at reg head — AL/AZ+FL | ~2-4 GPU-h | 0.5-2 pp at FL reg | Medium | **T1** |
| ~~3~~ | ~~§4.1 — Per-task best-epoch shipping~~ | — | — | — | **DROPPED 2026-05-21 by user** |
| 4 | §4.2 — HGI reg-head + c2hgi cat-head COMPOSITE (AL/AZ only) | ~2-3 GPU-h (substrates exist) | Reg ceiling = HGI STL (+1.6-3.1 pp over c2hgi STL) AT NO CAT COST | High | **T1** |
| 5 | §4.4-partial — Freeze reg-after-peak, train cat only | ~4-6 GPU-h | 0-1 pp (mostly diagnostic; P4 already says cat is not the bottleneck) | Low | **T2** |
| 6 | §4.7 — Designs J/B re-eval under F1 selector | ~10-15 GPU-h (substrate REBUILD + MTL train) | 0-1 pp (J/M already beat canonical on STL reg at AL/AZ; F1-selector lift on MTL unknown) | Medium | **T2** |
| 7 | Levers 4/5 (merge_design) — POI2Vec p2r + KL distill | ~6-10 GPU-h | 0-0.6 pp (substrate-axis; LEVER_6 already falsified on the same gap) | Low | **T3** |
| 8 | §4.8 — POI decoder with HGI-emb as target (NEW memo first) | ~8-12 GPU-h + new build script | Unknown; speculative re-open of substrate axis | Low | **T3** |

**Notes**
- Lever 6 of merge_design is **already executed and FALSIFIED** (LEVER_6_FINDINGS.md 2026-05-06). Listed in DEFERRED_WORK.md by accident of inheritance; re-running would be wasteful. Excluded from rank.
- §3 Rank 3 sub-Bonferroni candidates (T6.2 / T5.3 / T5.2b) were re-evaluated in Phase 2 P6 and FALSIFIED — closed; not in this ranking.

---

## Reasoning per item

### Rank 1 — §4.5 log_T as supervisory signal

**Mechanism.** Today the reg head's `next_stan_flow` (`next_getnext_hard`) consumes `log_T[last_region_idx]` as a feature blended via `α`. **Never tested:** an explicit *supervisory* term `L_logT = KL(softmax(reg_logits) ‖ log_T[last_region_idx])` summed into the reg loss — i.e. teach the head to *match* the empirical Markov-1 transition row, not just to use it as an additive prior.

**Why it could work.** P5 (stale log_T audit) proved log_T's content is **load-bearing for reg Acc@10** — a stale log_T silently dropped reg ~8 pp at FL. A signal that powerful is being used *only* as a feature, not as supervision. Distillation-on-prior is well-trodden territory in next-token prediction.

**Cost.** 3-5 GPU-h: code addition (`src/losses/log_t_kd.py` + flag), single-seed=42 at FL × 5 folds × 2 settings (`τ ∈ {1.0, 2.0}`) baseline + variants. Per-fold log_T already exists on disk.

**Risk.** Cat regression (reg pulls more of the joint loss). Mitigation: gate behind a `--log-t-kd-weight` flag, sweep at 0.05/0.1/0.2.

**Acceptance.** Wilcoxon p ≤ 0.05 vs shipping on FL n=5 (single-seed first); promote to multi-seed at FL only if signal survives.

**Lands in:** [`reg_head_architecture_sweep.md`](../../future_works/reg_head_architecture_sweep.md) §"log_T as supervisory signal" (new sub-section per DEFERRED_WORK.md §4.5).

---

### Rank 2 — §4.6 Class-balanced batch sampler at reg head

**Mechanism.** Today reg head uses weighted-CE (class weights from `src/data/folds.py`). The FL ~4 700-region long-tail makes weighted-CE noisy at the tail. Class-balanced batch sampler (BalancedBatchSampler from `src/data/folds.py` if scaffolded; else from `torch.utils.data.WeightedRandomSampler`) draws ~uniformly across regions per batch.

**Why it could work.** F2 mechanism (Phase 1 v4): MTL reg val peaks at ep 2-4 then crashes. Class-imbalance noise amplifies the crash (rare-class gradient bursts are noisy). Balanced batching reduces the variance.

**Cost.** 3-5 GPU-h: implement WeightedRandomSampler path in `src/data/folds.py` (probably already there, just unused), single-seed FL × 5 folds, A/B weighted-CE vs balanced-sampler.

**Risk.** Cat throughput drop (balanced sampling forces over-sampling rare regions ⇒ smaller effective epoch). Mitigation: scale total samples to keep epoch length equal.

**Lands in:** [`head_window_batch_audit.md`](../../future_works/head_window_batch_audit.md) §C (already enumerated; just needs execution).

---

### ~~Rank 3 — §4.1 Per-task best-epoch shipping~~ — DROPPED 2026-05-21 by user

> Dropped per user direction 2026-05-21. The original analysis (preserved below for record) flagged this as the highest-deploy-time lift in the inventory, but the 1-2 day implementation cost is out of scope for the current pass. Sub-track remains absorbed into [`mtl_architecture_revisit.md`](../../future_works/mtl_architecture_revisit.md) for the successor study.

#### Original analysis (now DROPPED)

**Mechanism.** Each MTL training already produces per-fold val CSVs (cat & reg). Today shipping picks ONE epoch (joint selector). The variant: **deploy two checkpoints** — cat from cat-best epoch, reg from reg-best epoch, shared backbone from joint-best (or per-task). At inference, route through the head whose task is being scored.

**Why it could work.** P3 disjoint-vs-joint capacity gap is **2.4 pp at FL** (geom_simple 61.54 vs disjoint 63.91); **~12 pp at AL/AZ** (much larger because small-state reg peaks at ep 2-4 and cat peaks at ep 30-40 — no single epoch is good for both).

**Cost.** Implementation (1-2 days): two-checkpoint save logic in `BestTracker` (`src/tracking/best_tracker.py`), inference path that loads cat_head from cat-ckpt and reg_head from reg-ckpt, MLHistory reporting "deploy-disjoint" as a 4th frontier. **Zero retraining** — all existing run dirs already have the per-fold val CSVs that pick the epochs.

**Risk.** "Two-checkpoint deployment" is non-standard; needs a clean inference API and a paper-side defense ("at deploy time, route by task" — natural for an MTL paper).

**Acceptance.** A clean four-frontier table (best joint / best disjoint / **deploy-disjoint** / STL ceiling) on all 5 states.

**Lands in:** [`mtl_architecture_revisit.md`](../../future_works/mtl_architecture_revisit.md) (highest-EV per DEFERRED_WORK.md). May warrant its own brief sub-study under that future_work.

---

### Rank 4 — §4.2 Composite (HGI reg-head + c2hgi cat-head) at AL/AZ

**Mechanism.** Two STL models — one Check2HGI for cat (already trained for §0.1 v11), one HGI for reg (already trained at AL/AZ). At deploy, route requests by task. NOT an MTL training; bypasses the MTL gap entirely.

**Why it could work.** §0.3 RESULTS_TABLE confirms HGI STL reg = +1.6-3.1 pp over c2hgi STL reg. Composite IS the substrate-capacity ceiling for both heads simultaneously. Should match HGI reg at HGI cat-cost = 0 (cat comes from c2hgi).

**Cost.** AL/AZ only (HGI substrate on disk). ~2-3 GPU-h to (a) load both checkpoints, (b) score val sets, (c) report three-frontier table. **FL/CA/TX requires HGI substrate regen** (~3-5 GPU-h per state × 3 states = +10-15 GPU-h) — out of scope here; gate decision on AL/AZ first.

**Risk.** Paper framing: composite at deploy time is a legitimate "deployable" baseline but reviewers may want a single integrated engine. The §4.2 framing is "deployable composite", distinct from merge_design Lever 6's "integrated engine" framing.

**Acceptance.** Deployable composite reg matches HGI STL reg (within σ) and cat matches c2hgi STL cat at AL/AZ.

**Lands in:** NEW MEMO [`composite_two_substrate_engine.md`](../../future_works/composite_two_substrate_engine.md) (per DEFERRED_WORK.md).

---

### Rank 5 — §4.4-partial Freeze reg after peak, train cat only

**Mechanism.** P4 frozen-cat showed cat is NOT the bottleneck. The symmetric test — freeze reg after its peak (ep 2-4), then unfreeze cat to converge — is untested. Equivalent to: train normally to ep 4, snapshot reg head + shared backbone, then continue training with reg loss = 0 and cat_head only.

**Why it MAY work.** Symmetric to P4 but reg-side. If MTL reg peaks at ep 2-4 because of CAT-task gradient pushing the shared backbone toward cat-friendly geometry, freezing reg's path *might* stabilise reg at its peak while cat catches up.

**Cost.** 4-6 GPU-h: training flag (`--freeze-reg-after-epoch 4`), single-seed FL × 5 folds + AL/AZ × 5 folds.

**Risk.** P4 already says the gap is architectural (MTL backbone), not negative transfer. This test could easily falsify too. EV is mostly DIAGNOSTIC.

**Acceptance.** Either: (a) reg @ deploy lifts >2 pp at FL → reopens scheduling axis; (b) flat → architectural-bottleneck framing confirmed from the symmetric side.

**Lands in:** [`substrate_adaptive_mtl_balancing.md`](../../future_works/substrate_adaptive_mtl_balancing.md) §"asymmetric freezing" sub-track.

---

### Rank 6 — §4.7 Designs J/B re-eval under F1 selector

**Mechanism.** Designs J (anchor) and B (POI2Vec @ pool) beat canonical c2hgi on STL reg at AL/AZ (J/H Wilcoxon-strict at FL too). Never run as the MTL substrate. F1 selector fix may interact with them differently than with canonical c2hgi.

**Why it MAY work.** STL J/B at AL/AZ outperforms canonical by ~+2 pp. If MTL reg under F1 selector retains a fraction of this STL substrate Δ, J/B as the MTL substrate could yield +1-2 pp at AL/AZ MTL reg.

**Cost.** Substrate parquets NOT on disk (verified). REBUILD via `scripts/probe/build_design_j_anchor.py` + `build_design_b_poi_pool.py` at all 5 states (~3-5 GPU-h × 2 designs × 5 states = 30-50 GPU-h) THEN MTL train at all 5 states × 2 designs = ~10-15 GPU-h. **Real cost ≈ 40-60 GPU-h.**

**Mitigation.** Run J only (the strongest STL candidate) at AL/AZ first (~10 GPU-h gate). Only escalate to FL/CA/TX + B if AL/AZ shows lift.

**Risk.** Substrate axis was declared exhausted by canonical_improvement Tier-6 closure. J/B beat canonical on STL but NOT HGI. F1 selector fix is *deployable-axis* only — substrate-axis ceiling is unchanged. Expected lift ≤ residual STL Δ to HGI (1-3 pp).

**Lands in:** Cross-study to [`docs/studies/merge_design/`](../merge_design/) next-pass agenda.

---

### Rank 7 — Levers 4/5 (merge_design)

**Mechanism.** Lever 4: add POI2Vec at the p2r boundary (~4h). Lever 5: replace pointwise distill with KL-on-top-k distill in Design M (~3h).

**Why it MAY work.** Both target the residual ~1 pp gap to HGI at AL/AZ on STL reg. Cheaper than Lever 6 (which was falsified).

**Risk.** Lever 6 was *the principled candidate* and was falsified at the same gap. Levers 4/5 attack adjacent surfaces; same EV ceiling, lower confidence.

**Lands in:** [`docs/studies/merge_design/`](../merge_design/) next-pass agenda (already located there).

---

### Rank 8 — §4.8 POI decoder with HGI-emb as target (NEW memo first)

**Mechanism.** Instead of decoding raw POI features (Tier 4 family), decode HGI's POI embeddings (distillation framing). Same hard-rule-compatible scope as T4.

**Why it MAY work.** HGI's POI embeddings encode Delaunay POI-POI structure. Decoder framing transfers that structure without the concat-style leak T4 risked.

**Cost.** NEW BUILD SCRIPT (`scripts/probe/build_design_t4_hgi_decoder.py`), ~6-8 hours code + ~6-10 GPU-h × multi-state.

**Risk.** New mechanism in already-exhausted substrate family. EV is "re-open the substrate axis if HGI's spatial bias can be transferred" — speculative.

**Lands in:** NEW MEMO [`poi_decoder_hgi_distill.md`](../../future_works/poi_decoder_hgi_distill.md) (per DEFERRED_WORK.md).

---

## Execution recommendation

**Tier 1 (execute first, ≤ 1 day total):** Ranks 1, 2 — the two cheap reg-head interventions that target the load-bearing gap directly. Both ~3-5 GPU-h, both add a single flag, no new architecture.

**Tier 1 (cheap diagnostic):** Rank 4 — composite at AL/AZ only (substrates already on disk). One-shot scoring; if HGI substrate must be rebuilt for FL/CA/TX, defer those.

**~~Rank 3 (per-task best-epoch shipping):~~** DROPPED 2026-05-21 by user. The 1-2 day impl cost is out of scope; the sub-track stays absorbed into [`mtl_architecture_revisit.md`](../../future_works/mtl_architecture_revisit.md) for the successor study.

**Tier 2 (after Tier 1 closes):** Ranks 5, 6 — bigger compute, smaller expected lift.

**Tier 3 (memos to draft now, execute later):** Ranks 7, 8 — write memos so they exist, but defer execution until Tier 1/2 close.

## Open coordination question

Each Tier-1 execution lands data in a different `future_works/` memo. Two pragmatic options:

- **Option A (lean):** treat execution as "polishing the closed mtl-protocol-fix study" — run Tier 1 here, write findings into `docs/results/mtl_protocol_fix/` follow-up JSONs, append a final `log.md` entry. Don't reopen the study formally.
- **Option B (proper):** spawn a successor study `docs/studies/mtl-protocol-fix-followup/` (or graduate into `mtl_architecture_revisit/`) and execute under that study's banner.

Recommendation: **Option A for Ranks 1-4** (low-blast-radius polish on top of the closed study), then if any Tier-1 item promotes a winner, that's the trigger to formally launch `mtl_architecture_revisit` as the successor.
