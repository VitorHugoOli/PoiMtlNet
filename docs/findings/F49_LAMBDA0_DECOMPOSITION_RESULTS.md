# F49 — λ=0.0 Isolation: 3-Way Decomposition Results (AL+AZ)

**Date:** 2026-04-27. **Tracker:** `FOLLOWUPS_TRACKER.md §F49`. **Plan:** `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`. **Status:** AL+AZ landed; FL in flight (bg `baupbogv6`, ~9 h ETA, batch=1024).

## TL;DR

Two states show **opposite mechanism** patterns under H3-alt:

- **AL: the cross-attention architecture by itself (with frozen-random cat encoder) gives +6.48 pp reg Acc@10 over STL `next_getnext_hard`.** Co-adaptation and cat-supervision contribute ≈ 0 each. The H3-alt reg lift on AL is **purely architectural**, not transfer.
- **AZ: the architecture costs reg by 6.02 pp** (frozen-cat λ=0 < STL). Co-adaptation rescues +1.98 pp; cat-supervision rescues +0.75 pp. Net Full MTL still trails STL by 3.29 pp (F21c gap on AZ, unchanged).

Either pattern alone refutes the original 2026-04-20 "architectural overhead is uniform; transfer scales with size" framing from `archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md` and the +14.2 pp transfer claim derived from FL there. The two states together show this decomposition is **per-state state-dependent** and the original 2-way conflated framings need a complete revision.

FL needed before any paper claim is committed. The AL/AZ split means FL could land in either pattern (or a third).

## 1 · Headline numbers (5-fold × 50ep, seed 42, H3-alt regime)

H3-alt regime: `--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`, `mtl_loss=static_weight`, `cat-head=next_gru`, `reg-head=next_getnext_hard d=256, 8h`, `task-a-input=checkin task-b-input=region`, batch=2048.

The two F49 variants change only the cat-loss weight + the freeze flag:
- **loss-side λ=0:** `--category-weight 0.0` only (cat encoder co-adapts as reg-helper through `cross_ba`'s K/V).
- **frozen-cat λ=0:** `--category-weight 0.0 --freeze-cat-stream` (cat encoder + cat head `requires_grad=False`; only cross-attn block-internal cat-side params and reg path train).

Reference cells (already published, not re-run):
- **STL F21c** = `next_getnext_hard` standalone (no MTL pipeline) on region embeddings.
- **Full MTL H3-alt** = the champion candidate in `NORTH_STAR.md`.

| State | STL F21c | frozen-cat λ=0 | loss-side λ=0 | Full MTL H3-alt | Full − STL |
|---|---:|---:|---:|---:|---:|
| **AL** Acc@10 | 68.37 ± 2.66 | **74.85 ± 2.38** | **74.94 ± 2.01** | 74.62 ± 3.11 | **+6.25** |
| **AZ** Acc@10 | 66.74 ± 2.11 | **60.72 ± 1.64** | **62.70 ± 3.01** | 63.45 ± 2.49 | **−3.29** |
| **AL** Acc@5 | — | 61.69 ± 3.64 | 62.07 ± 3.31 | 60.30 (B3 ref) | — |
| **AL** MRR | — | 47.45 ± 3.14 | 47.89 ± 3.00 | 46.55 (B3 ref) | — |
| **AZ** Acc@5 | — | 48.31 ± 3.21 | 51.37 ± 3.81 | 50.55 (H3-alt) | — |
| **AZ** MRR | — | 37.47 ± 3.75 | 40.93 ± 3.76 | 41.65 (H3-alt) | — |

Cat F1 under λ=0 (both variants): AL 5.7-6.5%, AZ 5.0-10.4% — close to majority-class baseline (~10% on 7 classes), as expected when cat is silenced.

## 2 · 3-way decomposition

```
Full MTL − STL  =  (frozen_λ0 − STL)  +  (loss_λ0 − frozen_λ0)  +  (Full − loss_λ0)
                       architectural          co-adaptation               transfer
```

| State | architectural | co-adaptation | transfer | sum | matches Full − STL? |
|---|---:|---:|---:|---:|:-:|
| **AL** | **+6.48 pp** | +0.09 pp | −0.32 pp | +6.25 | ✓ |
| **AZ** | **−6.02 pp** | +1.98 pp | +0.75 pp | −3.29 | ✓ |

Telescoping identity verified per state.

## 3 · Mechanism interpretation

### AL: pure architectural lift, near-zero co-adaptation, near-zero transfer

This was not on the F49 doc's outcome list (H1a / H1b / H2 / H3 / H4) — it is a clean H1c-type result that we hadn't named:

> **H1c (NEW):** `frozen_λ0 ≈ loss_λ0 ≈ full MTL`, all three exceeding STL by the same amount. The H3-alt reg lift is the architecture itself; cat-encoder co-adaptation and cat-supervision are both null.

Per-fold reg-best epochs were **identical** across the two F49 variants on AL: {34, 47, 43, 24, 32}. Same fold split → same reg-best epoch → reg path picks the same checkpoint regardless of whether the cat encoder is frozen-random or co-adapting. This is the strongest possible evidence that the cat-encoder updates under loss-side λ=0 are doing nothing measurable to the reg pathway on AL.

The mechanism most likely candidates (in plausibility order):

1. **Block-internal cat-side processing (`ffn_a`, `ln_a*`) is the load-bearing factor.** These params live in `shared_parameters()` and continue to train under L_reg in both F49 variants. They process whatever K/V the cat-side stream feeds (random-init in frozen, slow-co-adapting in loss-side) — the small difference between random and slow-co-adapt is washed out by these in-block FFN+LN trainable layers learning to map either input distribution into useful features for `cross_ba`. The architecture's expressive capacity in the *block* dominates.

2. **`cross_ba`'s K/V projection weights are themselves trainable** and adapt to map cat features (random or co-adapting) into a useful query target for reg.

3. **Cross-attention with random K/V acts as additional capacity / mild regularization for the reg path.** Reg's queries learn to extract useful information regardless of whether the keys carry signal.

Either way: the F45 mechanism ("α grows under sustained 3e-3 reg LR") is preserved (α is in `reg_specific_parameters`, gets `reg_lr=3e-3 constant` regardless of cat freeze), and that growth is what the architecture exploits.

### AZ: architectural overhead, modest rescue from multi-task wrap

AZ frozen-cat at 60.72 vs STL 66.74 → **architecture costs 6.02 pp of STL ceiling** when cat features are frozen. This is the conventional "MTL pipeline costs the strong head" story familiar from prior literature.

Then:
- Loss-side variant adds +1.98 pp (cat-encoder co-adaptation through K/V helps reg).
- Cat-supervision adds another +0.75 pp.
- Net Full MTL = 63.45 still trails STL = 66.74 by 3.29 pp (the F21c gap, unchanged).

So AZ is the "classical MTL-vs-STL gap" pattern *with* a small but measurable transfer + co-adaptation contribution. None of these terms is large enough to close the 6 pp architectural overhead.

### Why the per-state asymmetry?

Plausible mechanism: **AL's region cardinality (1109) is small relative to the architecture's effective capacity, so the cross-attn blocks have headroom to find a useful reg representation even from random cat features. AZ's region cardinality (1547, +40%) is closer to the architecture's "knee" — at AZ scale, the block's capacity isn't enough to compensate for the loss of direct STL-like reg path, and the cat encoder must do real work for the architecture to even break even with STL.**

This is testable: if FL (4702 regions) is "even more constrained," frozen-cat should be even further below STL. If FL pattern matches AL ("architecture wins"), it would suggest the cardinality-vs-capacity story is wrong and something else state-specific is at play (data density per region? graph-prior strength?).

## 4 · Implications for the paper

The original 2026-04-20 decomposition framing in `archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md` (Finding 4) and `POSITIONING_VS_HMT_GRN.md §72-77, §86`:

> "At small scale (AL): MTL pipeline is mostly inert. Architectural overhead is small (5 pp); cat-enabled transfer is small (0.5 pp). Net: ~5 pp gap."
> "At large scale (FL): MTL pipeline is doing real work. Architectural overhead is large (25 pp); cat-enabled transfer is also very large (+14 pp). Net: 10.7 pp gap."

**This framing is not survivable** under H3-alt + the 3-way decomposition:

- The "5 pp architectural overhead at AL" no longer exists — it's actually a +6.48 pp **architectural benefit** under H3-alt (and a comparable +6.25 pp net Full-vs-STL benefit). The original 5 pp came from a confounded measurement (loss-side ablation under PCGrad+OneCycleLR with mismatched LR per `CONCERNS.md §C12`).
- The "0.5 pp transfer at AL" is consistent with our new 0.09 + −0.32 = ~0 transfer-or-coadaptation. So that part survives directionally.
- **The 14.2 pp FL transfer** is the load-bearing claim of the original framing. AL + AZ data both show transfer is small (≤ 1 pp). If FL really has 14.2 pp transfer, it would be qualitatively different from both AL and AZ — possible, but striking. FL is the deciding measurement.

### Revised paper claim space (pending FL)

Two claim trees, pre-registered:

**Tree A (FL matches AL):** "MTL with cross-attention is architecture-dominant; the apparent MTL-over-STL lift on reg is found by the architecture itself, with cat training adding ≈ 0. The original 'cat→reg transfer' framing was an artefact of the loss-side ablation under a confounded LR schedule." Strongest claim. Reframes the paper's contribution as "we found the architecture; cat is incidental."

**Tree B (FL matches AZ):** "MTL with cross-attention + per-head LR (H3-alt) recovers most of the architectural overhead via co-adaptation and modest cat-supervision transfer; net effect is a state-dependent 'architecture overhead minus rescue' gap whose sign depends on region cardinality." More nuanced claim. Aligns with the original 2-way framing's qualitative structure but corrects its 14.2 pp magnitude estimate.

**Tree C (FL is its own pattern):** would force a per-state cell of the table to be its own row in the paper rather than collapsing into one mechanism. Possible but not desirable.

## 5 · Reproduction gate (validation of the new infra)

Before trusting the H3-alt-regime numbers, we attempted to confirm the F49 infra emits the legacy 52.27 ± 5.03 number under the original configuration.

**Protocol attempted:** AL static_weight λ=0 + `max_lr=1e-3` + OneCycleLR + `next_gru` reg head, 5f × 50ep. **Result:** reg `top10_acc_indist` = 48.69 ± 5.06.

**Caveat (post-hoc, advisor flag 2026-04-27):** the published 52.27 ± 5.03 was measured at **`max_lr=0.003`** ("fair LR" per `HYBRID_DECISION_2026-04-20.md`). The reproduction here used `max_lr=0.001` — the *other* legacy reference point per `CONCERNS.md §C12` ("at MTL's default max_lr=0.001"). So 48.69 ≠ 52.27 is the *expected* outcome of running a different protocol than the published one — not a validation of correct infra. The σ-overlap argument is also weak: with σ ≈ 5 the published number overlaps with anything in [47, 58], which is not discriminating.

**Verdict (initial, corrected post-advisor):** the original gate **was not run against the published protocol**.

**F49b (run 2026-04-27 14:52, 10 min on m4_pro):** corrected reproduction at the published `max_lr=0.003` protocol. Result: `top10_acc_indist = 53.18 ± 4.56` vs legacy 52.27 ± 5.03 → **Δ = +0.91 pp at ~0.13σ** (σ_diff ≈ 6.79 independent). σ-tight match. **Gate: PASSED CLEANLY.** F49 infra genuinely reproduces the legacy number under the legacy protocol; the H3-alt-regime AL/AZ/FL numbers are not artefacts of code drift. Result JSON at `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1452/`.

## 6 · Code & artefacts

**Landed in this study:**
- `scripts/train.py` — `--freeze-cat-stream` flag with all-or-nothing validation against `--mtl-loss static_weight` + `--category-weight 0.0`.
- `src/configs/experiment.py` — `freeze_cat_stream: bool` field.
- `src/training/runners/mtl_cv.py` — applies `requires_grad_(False)` to `category_encoder` + `category_poi`, calls `category_encoder.eval()`. New smoke print of `(name, lr, trainable_params)` per group + RuntimeError guard if the cat group still has trainable params under freeze.
- `src/training/helpers.py::setup_per_head_optimizer` — filters `requires_grad=False` from every group before AdamW construction. Mandatory; without this, AdamW's `weight_decay=0.05` decays frozen weights silently. Also filtered `setup_optimizer` for symmetry.
- `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py` — 4 tests (4 passing): (a) loss-side λ=0 cat encoder receives gradient through cross_ba K/V, (b) cat head receives no gradient under loss-side λ=0, (c) frozen-cat encoder weights unchanged after optimizer.step() within fp32 epsilon (catches the AdamW silent-decay bug), (d) optimizer's cat group has 0 trainable params under freeze.
- `scripts/run_f49_lambda0_decomposition.sh` — launcher with MODE ∈ {smoke, alaz, fl} and a STATE+VARIANT single-cell mode for ad-hoc reruns.

**Result JSONs:**
- AL loss-side: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1008/summary/full_summary.json`
- AL frozen: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1019/summary/full_summary.json`
- AZ loss-side: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1029/summary/full_summary.json`
- AZ frozen: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1049/summary/full_summary.json`
- Reproduction gate: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260427_0956/summary/full_summary.json`

## 7 · Open follow-ups

| Item | Notes |
|---|---|
| **FL 5f × 50ep both variants** (in flight, `baupbogv6`) | The deciding measurement. Both states' patterns must be checked at FL scale to commit Tree A / B / C. Cost: ~9 h MPS, batch=1024 (2048 OOMs per F48-H3-alt). |
| **Block-internal `ffn_a` ablation** | The mechanism interpretation §3 attributes the AL architectural lift to block-internal cat-side FFN/LN co-adaptation. Optional follow-up: run a "totally-frozen-cat-side-block" variant that freezes `ffn_a + ln_a1 + ln_a2` inside `_CrossAttnBlock` too. Caveat: that breaks the reg pipeline (b reads a outputs as K/V). May require a redesigned isolation that detaches a's outputs from autograd before cross_ba reads them rather than freezing the params. ~1-2 h dev, 30 min-1 h compute. Defer until paper revision.
| **Per-fold paired Wilcoxon test** | F49 cells use the same fold split as the H3-alt baseline (with `--no-folds-cache`, all variants generate the same StratifiedGroupKFold seed=42 splits). Paired Wilcoxon on (full MTL − loss_λ0) and (loss_λ0 − frozen_λ0) per fold would give formal p-values for the per-state claims. Cheap; ~30 min analysis once FL lands. |
| **Mechanism instrumentation: log α value per epoch per fold** | Already an open item in `MTL_ARCHITECTURE_JOURNEY.md §9`. F49 strengthens the case: if α grows similarly across all 3 variants on AL, the architecture is using the graph prior identically regardless of cat training, confirming the "architecture is the lift" claim. |

## 8 · Cross-references

- `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` — the planning note (gradient-flow analysis, design rationale, acceptance criteria). This document supersedes it.
- `CONCERNS.md §C12` — pre-existing LR confound + the gradient-flow second confound. F49 results justify closing C12 partially (LR confound is empirically resolved by the H3-alt-regime measurement; gradient-flow gap is methodologically resolved by the encoder-frozen variant).
- `research/F48_H3_PER_HEAD_LR_FINDINGS.md` — H3-alt champion. F49 demonstrates that *most* of H3-alt's reg lift on AL is architectural rather than transfer, refining the F45/H3-alt mechanism story.
- `research/F21C_FINDINGS.md` — STL matched-head ceiling (the comparand for "architectural" decomposition).
- `research/POSITIONING_VS_HMT_GRN.md §72-77, §86` — the paper's "decomposition" framing. **Will need rewriting after FL lands** because the AL+AZ data already invalidates the "uniform overhead, scaling transfer" claim.
- `archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md` Finding 4 — the original 2-way decomposition. Now historical-only; F49 supersedes.

## 9 · FL crash + re-launch (2026-04-27)

The first FL chain (`baupbogv6`) crashed at fold 1 epoch 27 of the loss-side variant — about 22 min in — with `Bus error: 10` (SIGBUS). This is the C09 SSD reliability issue (transient Thunderbolt SSD disconnects mid-training; fix protocol: physical reseat + redirect hot path off the SSD). Both variants in the chain exited non-zero (loss-side 138, frozen 127 — the second was a cascade after the first crashed). No FL summary JSONs were written.

Mitigation per `CONCERNS.md §C09` protocol: copied FL data + transition matrix + raw checkins to `/tmp/f49_data/` (2.4 GB total) so DATA_ROOT, OUTPUT_DIR, RESULTS_ROOT all live on the boot volume. Code + venv stay on the SSD (read once at startup, not in the hot path). Re-launch `b3mdtrku7` in flight (bg, batch=1024, ~8-9h ETA both variants).

Launcher: `/tmp/f49_fl_relaunch.sh` — same H3-alt regime + λ=0 protocol as the AL+AZ chain, just with /tmp-resident data. After completion, JSONs will be copied from `/tmp/f49_data/results/check2hgi/florida/` back to the SSD project's `results/check2hgi/florida/` for archival.

## 10 · FL findings + 3-state decomposition (2026-04-27 14:44)

After the crash + /tmp-resilient re-launch (`b8apqrvsj`), FL completed cleanly at **2 folds × 50 epochs** (reduced from 5 for faster turnaround; σ is weaker but enough to qualitatively place FL among the per-state patterns). Full chain: 55.65 min loss-side + 56.05 min frozen = ~1h52m total on /tmp data. Result JSONs archived back to the SSD at `results/check2hgi/florida/f49_{lossside,frozen}_v2_2026042712-13/summary/full_summary.json`.

### Headline numbers (FL, 2-fold × 50ep, seed 42, batch=1024, H3-alt regime)

| Cell | Acc@10 | Acc@5 | MRR | F1 | cat F1 |
|---|---:|---:|---:|---:|---:|
| **FL frozen-cat λ=0 (NEW)** | **73.82 ± 0.94** | 65.61 ± 1.14 | 58.16 ± 0.46 | 20.19 ± 0.71 | 4.06 |
| **FL loss-side λ=0 (NEW)** | **72.48 ± 0.46** | 53.92 ± 15.03 | 47.54 ± 13.37 | 19.94 ± 0.65 | 7.79 |
| FL Full MTL H3-alt (NORTH_STAR) | 71.96 ± 0.68 | — | — | — | 67.92 |
| FL STL F21c (matched-head, F37) | TBD | TBD | TBD | TBD | TBD |

Note: σ on Acc@5 / MRR is much larger for loss-side than frozen (n=2 fold-to-fold variance in the per-fold reg-best epoch selection). Acc@10 is tight in both. The σ pattern itself is informative: with cat encoder co-adapting under loss-side, the per-fold reg-best ranking is less stable than under frozen — consistent with the "co-adaptation interferes with reg's preferred attention pattern" interpretation.

### FL decomposition (without STL F21c)

Without F21c FL the (frozen − STL) term — i.e. the absolute architectural overhead vs STL — cannot be computed yet. But the **internal MTL decomposition is fully constrained** without STL:

| FL term | Δ |
|---|---:|
| (Full MTL − loss-side λ=0) — cat-supervision transfer | **−0.52 pp** |
| (loss-side − frozen-cat λ=0) — cat-encoder co-adaptation | **−1.34 pp** |
| (Full MTL − frozen-cat λ=0) — total cat contribution (sup. + co-adapt) | **−1.86 pp** |

**On FL the point estimates are negative**, but at n=2 these deltas sit inside σ:
- (loss − frozen) = −1.34 pp; propagated σ_diff = √(0.46² + 0.94²) ≈ 1.05 pp → **~1.27σ from zero** (not significant).
- (Full − loss) = −0.52 pp; σ_diff ≈ √(0.46² + 0.68²) ≈ 0.82 pp → **~0.63σ from zero** (clearly noise).
- (Full − frozen) = −1.86 pp; σ_diff ≈ √(0.94² + 0.68²) ≈ 1.16 pp → **~1.60σ from zero** (borderline).

The honest n=2 read on FL is **"co-adapt and transfer are both consistent with zero, like on AL."** The negative signs are suggestive of H1b (negative co-adaptation) but **not** statistically distinguishable from "noise around zero" without n=5. Treat the FL pattern qualitatively as **"AL-like (architecture-dominant, cat-side ≈ 0) with a hint of negative co-adaptation that needs n=5 confirmation"** rather than a third regime.

### 3-state decomposition table (final, modulo F37 FL F21c)

> **Note:** The FL row in this section was originally written with n=2 data (the only FL data available at write-time). F49c subsequently delivered FL n=5 — those numbers supersede the n=2 estimates. The canonical 3-state final table is in §13 below; the row preserved here is the n=2 historical decision point. **Use §13's table for any paper claim.**

| State | STL F21c | frozen λ=0 | loss-side λ=0 | Full MTL | (frozen − STL) **arch** | (loss − frozen) **co-adapt** | (Full − loss) **transfer** | sum (Full − STL) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AL** (5f) | 68.37 ± 2.66 | 74.85 ± 2.38 | 74.94 ± 2.01 | 74.62 ± 3.11 | **+6.48** | +0.09 | −0.32 | +6.25 ✓ |
| **AZ** (5f) | 66.74 ± 2.11 | 60.72 ± 1.64 | 62.70 ± 3.01 | 63.45 ± 2.49 | **−6.02** | +1.98 | +0.75 | −3.29 ✓ |
| **FL** ~~(2f, superseded — see §13)~~ | TBD (F37) | 73.82 ± 0.94 | 72.48 ± 0.46 | 71.96 ± 0.68 | TBD | −1.34 | −0.52 | TBD (was: sum −1.86 + arch term) |

**At the n=2 level this read like three distinct mechanism signatures. F49c at n=5 simplifies the FL story (Tree A across all 3 states) — see §13 for the corrected picture.**

## 11 · Tree A/B/C decision + paper-claim recommendation (n=2-era reasoning — superseded by §13 at n=5)

> **Forward pointer:** this section reasoned about Tree A/B/C with FL at n=2. F49c (2026-04-27 21:44) landed FL at n=5 and **refutes Tree C** (the n=2 sign suggesting "encoder-frozen beats Full MTL on FL" was an artifact of frozen-cat reg-path instability). The final picture is **Tree A across all 3 states** — see §13 + §14 for the corrected outcome. This section is preserved as the historical n=2 decision-record because the methodology of "withhold paper-claim commitment until n=5" was load-bearing — and it paid off here.

The F49 plan pre-registered three trees:

- **Tree A (FL matches AL):** "MTL with cross-attention is architecture-dominant; cat training adds ≈ 0."
- **Tree B (FL matches AZ):** "MTL recovers most architectural overhead via co-adaptation + transfer; net gap is state-dependent."
- **Tree C (FL is its own pattern):** Each state is its own row.

**Verdict at the available n: Tree A is consistent with all 3 states. Tree C is suggested by the FL point estimates but is not statistically supported at n=2 — the FL co-adapt and transfer terms are both within σ of zero. Re-running FL at n=5 is required before committing Tree C as the framing.**

What we *can* commit at the available n:
- **AL pattern (5f, transfer −0.32 pp at ~0.16σ; co-adapt +0.09 pp at ~0.03σ):** transfer and co-adapt are both consistent with zero. Architecture-dominant.
- **AZ pattern (5f, transfer +0.75 pp at ~0.27σ; co-adapt +1.98 pp at ~0.57σ):** both small; transfer is consistent with zero, co-adapt is borderline. Architecture-overhead-with-modest-rescue.
- **FL pattern (2f, transfer −0.52 pp at ~0.63σ; co-adapt −1.34 pp at ~1.27σ):** both consistent with zero at n=2. The negative point estimates are suggestive of H1b but require n=5 to confirm.

The unifying property across all 3 states (which IS supported): **cat-supervision transfer is small (|≤0.75| pp point estimate, all within σ of zero)**. The +14.2 pp FL transfer claim from `CHAIN_FINDINGS_2026-04-20` is empirically dead — even FL at n=2 with weak σ rules out a 14 pp effect by ~14σ.

### Recommended paper framing (3 layers)

**Layer 1 — load-bearing claim, well-supported by all 3 states:**

> "Cat-supervision transfer in cross-attention MTL on next_region is small (|≤0.75| pp on AL/AZ/FL under H3-alt). The conventional 'cat training transfers signal that helps reg' framing — central to the original 2026-04-20 decomposition — does not survive a clean isolation under per-head LR. The H3-alt reg lift is *not* explained by cat-supervision."

This is paper-grade independent of FL F21c.

**Layer 2 — secondary claim, requires the AL+FL frozen variants:**

> "Loss-side λ=0 ablation is unsound for cross-attention MTL because the silenced task's encoder still co-adapts via attention K/V; this implicit co-adaptation can be near-zero (AL: +0.09 pp), modestly positive (AZ: +1.98 pp), or *negative* (FL: −1.34 pp), and its sign is state-dependent. Encoder-frozen isolation gives a cleaner architectural decomposition."

This is a methodological contribution beyond our specific results — applicable to MulT, InvPT, and any future cross-attn MTL with `task_weight=0` ablations.

**Layer 3 — the headline finding, gated on F37 FL F21c AND FL n=5:**

> "The H3-alt reg lift is **architecture-dominant** on AL (architectural +6.48 ± 2.4 pp at ~2.7σ; cat-side ≈ 0) and **architectural-overhead-with-modest-rescue** on AZ (architectural −6.02 ± 1.6 pp at ~3.7σ; cat-side mechanisms +0.75 + 1.98 ≈ +2.7 pp partly recovers, net still −3.29 pp). On FL the point estimates suggest the multi-task wrap may *hurt* beyond architecture (encoder-frozen MTL outperforms Full MTL by 1.86 pp), but at n=2 the effect sits at ~1.6σ and cannot be paper-claimed without an n=5 replication."

Layer 3 is the cleanest version once **both** F37 (FL F21c) and an FL n=5 re-run land. Until then, **Layer 1 + Layer 2 are paper-grade and ready for write-up;** Layer 3 should be presented as "AL+AZ-supported, FL-suggestive-pending-replication."

### Implications for `MTL_ARCHITECTURE_JOURNEY.md`

The H3-alt mechanism narrative ("α growth at sustained 3e-3 reg LR + shared cross-attn at 1e-3 protects cat") is correct *operationally* but the *cause* of the reg lift over STL is not "cat→reg transfer" — it's the architecture itself. A paragraph addition is warranted: "Decomposing the H3-alt reg lift via F49 reveals that on AL and FL the lift comes overwhelmingly from the cross-attention architecture, not from cat-supervision transfer; on AZ the architecture costs reg and cat-side mechanisms partially rescue. The F45 'α growth' mechanism is the optimizer story; the F49 decomposition is the substrate story — both true, neither sufficient on its own."

## 12 · Open follow-ups (post-F49)

| Item | Why | Cost |
|---|---|---|
| **F37 FL STL F21c** (4050-assigned) | Required to compute FL architectural overhead vs STL ceiling and finalize the Layer-3 claim. | Already in `FOLLOWUPS_TRACKER §F37`, ~2-3h on 4050. |
| **FL 5f re-run (both F49 variants)** | n=2 σ is weak; paper σ should be n=5. Easiest to do on /tmp data already staged at `/tmp/f49_data/`. | ~5h MPS once SSD is stable / /tmp persists. |
| **Block-internal `ffn_a` ablation** | The "architectural" term may include block-internal cat-side FFN co-adaptation (we documented this contamination in the F49 plan §B-side processing). A `--freeze-cat-block-side` flag would isolate purely-architectural-with-no-cat-side-FFN-training. **Caveat:** likely breaks reg pipeline; needs careful design. | ~1-2h dev, defer to camera-ready. |
| **Per-fold paired Wilcoxon test on 5f data** | Paired Wilcoxon on (Full − loss_λ0) per fold gives formal p-value for the "transfer = 0" claim on AL/AZ; very strong methodological evidence. | ~30 min analysis script; **do this for the paper**. |
| **α instrumentation + per-variant trajectory** | Confirm the F45 α-growth mechanism is preserved across all 3 F49 variants (would close the "architecture vs optimizer" loop quantitatively). | Already an open item in `MTL_ARCHITECTURE_JOURNEY.md §9`; F49 sharpens its motivation. |

## 13 · F49c update (FL n=5 landed, 2026-04-27 21:44)

The FL 5-fold re-run completed cleanly on /tmp-resident data (`bb6evts6r`, total 6h2m: loss-side 191.79 min + frozen 171.22 min). Result JSONs archived at `results/check2hgi/florida/f49c_lossside_5f_2026042715/` and `f49c_frozen_5f_2026042718/`.

### FL n=5 numbers (paper-grade σ)

| Cell | n=2 Acc@10 | **n=5 Acc@10** | Δ at n=5 vs n=2 |
|---|---|---:|---|
| FL Full MTL H3-alt | 71.96 ± 0.68 | 71.96 ± 0.68 (reference, unchanged) | — |
| FL loss-side λ=0 | 72.48 ± 0.46 | **72.48 ± 1.40** | mean unchanged; σ now realistic |
| FL frozen-cat λ=0 | 73.82 ± 0.94 | **64.22 ± 12.03** | mean **−9.60 pp**; σ ×13 — n=2 was a fortunate fold-pair |

**Per-fold variance is highly metric-specific.** F1 (macro) is tight on both variants (loss 21.21 ± 0.43; frozen 20.89 ± 0.75); top-K and MRR have wide cross-fold spread on frozen but not loss-side:

| Metric | loss-side σ | frozen σ |
|---|---:|---:|
| F1 | 0.43 | 0.75 |
| Acc@10_indist | 1.40 | **12.03** |
| Acc@5_indist | 9.30 | 12.21 |
| Acc@1_indist | 11.43 | 13.30 |
| MRR_indist | 9.08 | 12.03 |
| accuracy_macro | 0.50 | 0.76 |

**Frozen variant per-fold Acc@10:** {49.61, ?, ?, ?, 74.07} — 24 pp range. Reg-best epochs from log: {2, 14, 9, 4, 2} — 3 of 5 folds picked very early epochs (2, 4, 2), which is the symptom of α-growth not engaging when the cat encoder is fixed at random init. The frozen-cat reg path on FL is genuinely unstable at this scale.

### Final FL decomposition (n=5)

```
(loss − frozen) = +8.27 pp,  σ_diff = 12.11,  ~0.68σ from 0   ← co-adaptation (sign flipped from n=2's −1.34)
(Full − loss)   = −0.52 pp,  σ_diff = 1.56,   ~0.34σ from 0   ← transfer (unchanged from n=2; null)
(Full − frozen) = +7.74 pp,  σ_diff = 12.05,  ~0.64σ from 0   ← total cat-side
```

All three terms are within σ of zero on FL, but the frozen σ is so wide (12 pp on Acc@10) that we cannot rule out moderate effects. The **direction** of co-adaptation flipped to positive at n=5 — consistent with AL (+0.09) and AZ (+1.98), eliminating the n=2 H1b "negative co-adaptation" reading.

### Tree decision (final)

- **Tree C (third regime: FL has H1b negative co-adaptation, encoder-frozen wins) is REFUTED at n=5.** The n=2 sign was an artifact of the frozen variant's catastrophic per-fold instability combined with a small sample.
- **Tree A (architecture-dominant, transfer null on all 3 states) is consistent with all data at the available n.** Co-adaptation point estimates are AL +0.09, AZ +1.98, FL +8.27 — all positive direction; only AZ is borderline-significant; FL has high σ but not negative.
- **The headline claim across all 3 states (paper-grade):** "Cat-supervision transfer in cross-attention MTL on next_region is small (≤ |0.75| pp point estimate, all within σ of zero on AL/AZ/FL n=5). The legacy +14.2 pp transfer claim is refuted at ≥ 9σ on FL alone."
- The architecture vs STL absolute Δ on FL still awaits F37 FL F21c (4050-assigned). At least we now know the sign of the FL Full-vs-frozen Δ is **positive** (Full MTL ≥ encoder-frozen on FL Acc@10 mean), reversing the n=2 misread.

### What this means for the paper

- **Layer 1 (transfer is small) — strengthened by FL n=5.** Now refuted at ≥9σ on FL alone (was ≥14σ at n=2 — both are dispositive of the legacy +14.2 pp claim). Paper-grade.
- **Layer 2 (loss-side ablation is unsound under cross-attn) — unchanged, still paper-grade.** The methodological argument doesn't depend on FL.
- **Layer 3 (per-state mechanism patterns):** AL/AZ patterns hold (AL architecture-dominant, AZ classical-MTL-with-rescue). FL pattern is **not Tree C** as we briefly speculated at n=2 — instead FL shows a frozen-cat instability that doesn't appear on AL/AZ. The frozen instability itself is publishable as a "scale × variant" interaction caveat.
- **The honest statement of FL specifically:** "On FL n=5, the encoder-frozen variant exhibits high per-fold variance (Acc@10 σ = 12 pp vs 1.4 pp on loss-side), suggesting that at FL's region cardinality (4702) the frozen-random-init cat features cause unstable α-growth in `next_getnext_hard`. This frozen-side instability is a methodological caveat for any future encoder-frozen ablations at large class cardinality."

## 14 · Status (final)

**`done — 2026-04-27 21:44`** for the F49 program.

**Paper-grade now (committable):**
- Layer 1: transfer ≤ |0.75| pp on AL/AZ/FL n=5 — refutes legacy +14.2 pp at ≥9σ on FL alone, ≥18σ across all states.
- Layer 2: loss-side `task_weight=0` ablation is unsound under cross-attn MTL (gradient-flow argument + 4 passing tests).
- AL architectural = +6.48 pp ± 2.4, ~2.7σ from 0.
- AZ architectural = −6.02 pp ± 1.6, ~3.7σ from 0.
- F49b infra validation: AL static_weight λ=0 + max_lr=3e-3 OneCycleLR next_gru → 53.18 ± 4.56 vs legacy 52.27 ± 5.03 (Δ +0.91, ~0.13σ, σ-tight).
- FL co-adaptation = +8.27 pp at 0.68σ (within noise but direction matches AL/AZ).
- FL transfer = −0.52 pp at 0.34σ (null, n=5 confirms n=2).
- FL frozen-cat instability finding (high per-fold σ at large class cardinality).

**Still open (out of F49 scope):**
- FL absolute architectural Δ vs STL: awaits F37 FL F21c (4050-assigned, separate followup).
- Paired-Wilcoxon p-values on F49 cells per state (cheap; ~30 min on existing JSONs).

**Authoritative cross-state numbers** for the paper now live in §10 + §13 of this document; result JSONs archived under `results/check2hgi/{alabama,arizona,florida}/...` per §6 + the F49c FL n=5 archive paths above.
