# HANDOFF — MTL Improvement track (read this FIRST, then `log.md` + `INDEX.html`)

**As of 2026-06-04 (TIER 2 COMPLETE).** Branch `mtl-improve`, all pushed, working tree clean.
This is a single "you are here" snapshot. Full chronology: `log.md` (the 2026-06-04 Tier-2 entries).
Design + per-tier results: `INDEX.html`. **Tier-2 close-out (read this): `PAPER_UPDATE.md`.**

---

## 0. ⭐ TIER 2 IS COMPLETE — two results (2026-06-04). Read `PAPER_UPDATE.md`.
**(1) Architecture: clean, multi-seed-hardened NEGATIVE.** No single-model MTL architecture closes the
MTL→STL reg gap. The reg-private **dual-tower LOSES** to the matched baseline (FL multi-seed −3.35); a
5-point sharing dose-response (`CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower`) shows **more sharing
helps reg** — refuting the §6.4 "missing private backbone" hypothesis; 3 mechanism cells (cat-weight=0,
prior-OFF+wd0.01) localize the gap to the **joint cross-attn harness itself** (not interference/prior/wd).
**→ the composite (two-model) is the deployable reg answer; the gap is irreducibly architectural.** The
architecture axis is near-exhausted — **T2.3 (MoE) + T2.4 (hybrids) remain as the confirmatory close of
the card set (§0b, user-requested next step; expected negative per §6.3).** CrossStitch = a real-but-small partial (+1pp reg multi-seed, mixed cat,
−5..−10 below ceiling, NOT a closer). Implementation (`next_stan_flow_dualtower`,
`mtlnet_crossattn_dualtower`) + unit gate + all drivers are committed; capstone advisor verified the
code is correct and the decisions sound.

**(2) Recipe WIN: `onecycle` is the new recommended SMALL-STATE recipe.** onecycle (aggressive schedule,
NO alt-opt) dominates H3-alt at AL/AZ (v14 multi-seed +6–9pp reg / +1–2pp cat) and beats B9 on the v11
paper substrate (AL reg +2.98 / cat +7.36; AZ reg +0.76 / cat +4.69). **alt-opt flips sign by scale** →
keep B9 at large states (FL/CA). Adopted in `NORTH_STAR.md`. **§0.1 small-state arch-Δ annotated in
`results/RESULTS_TABLE.md §0.1` — author sign-off needed** (it reshapes a central claim; reg shrink is
modest on v11, the cat-flip is mostly a B9→deployable-recipe fix — read the nuance in `PAPER_UPDATE.md`).

**⭐ 2026-06-04 REDIRECT (user-approved) — NEW Tier 2P (joint-training protocol). This supersedes "ship
the composite."** Independent review found the close-out's "irreducibly architectural" headline is
**contradicted by its own data**: the `private_only` dual-tower arm IS the STL reg topology by
construction, ran under the GOOD (onecycle) recipe, and still lost ~10pp to STL-standalone (AL 52.41 vs
62.88). An identical topology failing only when trained jointly ⟹ the residual is **the joint training
PROTOCOL, not the topology** — which Tier 2 never varied (onecycle, a pure protocol change, already
recovered +5–9pp). **New `Tier 2P` (INDEX `#tier2p`): T2P.0 linchpin → T2P.1 staged / T2P.2 asymmetric-
recipe / T2P.3 distillation, goal = composite-quality reg in ONE model.**

**THE IMMEDIATE NEXT STEP is `T2P.0` (the linchpin, ~few GPU-h, DECISIVE) — see §0c below.** Then:
(a) finish T2.3+T2.4 as the confirmatory close of the *topology* card-set (§0b — no longer the headline,
just completeness); (b) run Tier 2P per the T2P.0 verdict; (c) the §0.1 paper re-statement decision
(author); (d) Tier 3 / close only after Tier 2P; (e) optional onecycle CA/TX.

---

## 0b. T2.3 + T2.4 — confirmatory close of the TOPOLOGY card-set (run after/alongside T2P.0; no longer the headline)

**You are running T2.3 (faithful MoE family) + T2.4 (per-task-input mixers/hybrids)** — the two Tier-2
cards not yet executed (INDEX `#T2-3`, `#T2-4`). Tier 2's verdict so far is a hardened NEGATIVE (§0); the
prior (§6.3) says MMoE/CGC lose ~2.7pp reg and PLE collapses, so **these are expected to be confirmatory
negatives** that complete the architecture card-set for the paper. Promote only on a genuine surprise
(≥1pp on the targeted axis, cat non-inferior TOST δ=2, at ≥2 of {AL,AZ,FL}).

**The comparand (CRITICAL — use the matched baseline, not the landed (a)):** score Δ vs **`base_a` @
onecycle MULTI-SEED**, which already exists as the **`onecyc_val`** rows in
`scripts/mtl_improvement/t21_harden_manifest.tsv` (= `mtlnet_crossattn + next_getnext_hard` @ onecycle,
{0,1,7,100}, AL/AZ/FL). Also position each arm vs the frozen (c)/(d) ceilings (§2) and **fold the new
arms into the sharing dose-response** (`scripts/mtl_improvement/t21_doseresp.py` +
`docs/results/mtl_improvement/T21_dose_response_50ep_seed42.txt`) — MoE/hybrids are new points on the
`CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower` curve.

**Recipe (the now-adopted one):** `onecycle` for AL/AZ (`--scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3
--reg-lr 3e-3 --shared-lr 1e-3`), `mtlnet_crossattn`-style COMMON: `--mtl-loss static_weight
--category-weight 0.75 --cat-head next_gru --reg-head next_getnext_hard --task-a-input-type checkin
--task-b-input-type region --log-t-kd-weight 0.0 --engine check2hgi_design_k_resln_mae_l0_1
--per-fold-transition-dir output/.../<state> --no-checkpoints`, 5f×50ep, seeded per-fold log_T. AL+AZ
first; FL only for a promoted arm.

**T2.3 — faithful MoE.** Models `mtlnet_mmoe`, `mtlnet_cgc` are REGISTERED and (verified 2026-06-04)
their `cat/reg/shared` partition is already **bijective+exhaustive (0 uncovered params)** → they run
under the per-head onecycle recipe with NO code change. **Skip PLE** (robust collapse, §6.3/F50).
**Fidelity caveat:** the on-disk MMoE/CGC are "-lite" per-task-input adaptations (not canonical; DSelect-K
is misnamed — dense convex combo), see `MTL_FLAWS_AND_FIXES.md §3.1`. Pragmatic path given the §6.3
prior + exhausted axis: run the existing -lite MMoE+CGC as the confirmatory multi-seed check + document
the fidelity caveat; only build faithful versions if -lite surprises. DSelect-K only if MMoE/CGC surprise.

**T2.4 — 3 hybrids (must BUILD — not in registry).** (i) MulT-faithful (intra-task self-attn BEFORE the
cross-attn block — extend `mtlnet_crossattn`); (ii) cross-stitch→cross-attn series (compose
`mtlnet_crossstitch` then `mtlnet_crossattn`); (iii) pre-norm + SwiGLU FFN (a `_CrossAttnBlock` variant).
Each = new `@register_model` subclass + the gates below. (The card says "compose with T2.1 winner" — but
T2.1 was negative, so the comparand is `base_a`; reframe T2.4 as "does per-task-input mixing beat plain
cross-attn?".)

**HARD GATES (do not skip — same as T2.1):**
1. **Unit-test gate (hard rule 10)** before any multi-fold launch — adapt `scripts/mtl_improvement/
   t21_unit_gate.py`: forward/backward on a synthetic 100-user batch, loss-finite, param-count within
   ~10% of B9 at D=256, and `shared/cat_specific/reg_specific_parameters()` **bijective+exhaustive**
   (for any NEW T2.4 module, wire it into the right group — experts/mixers → `shared`, reg-private bits
   → `reg_specific`; the base `MTLnet.shared_parameters()` is a NAME-SUBSTRING match, so a new module
   whose name lacks `shared_layers/film/task_embedding` will be SILENTLY dropped from `shared` → fix by
   overriding the partition like `mtlnet_crossattn` does). MMoE/CGC already pass; verify anyway.
2. **Per-arch LR mini-sweep (hard rule 7)** for each T2.4 hybrid (new arch): 5 regimes × AL+AZ × 5f×40ep
   × seed42, then full-protocol at the winner. (Reuse `t21_lr_sweep.sh` pattern.) MoE may start at
   onecycle (existing arch) + mini-sweep only if it surprises.
3. Stay at `shared_layer_size=256`; no fclass-as-feature; log_T-KD OFF; **stale-log_T preflight**
   (`stat` log_T vs next_region.parquet) + `freeze_folds.py --check` before each sweep.

**Reusable assets (this session):** `t21_harden.sh` (copy the `harden2` stage pattern — add a `t23`/`t24`
arms function + STAGE case; it has the **PID-safe rundir capture** + the **process-substitution wait-fix**
+ idempotent manifest skip), `t21_doseresp.py` / `t21_agg.py` (aggregation), `t21_unit_gate.py` (gate
template). **CONC discipline:** small states ~5GB (CONC=4 ok), FL ~14GB (CONC≤2), CA ~31GB (CONC=1 only).
**Rundir-race trap:** never capture rundirs via `ls -dt | head -1` under concurrency — use the `$!`
PID-suffix (the driver already does; see memory `ref-concurrent-rundir-race`).

**Tier-2 close after T2.3+T2.4:** if nothing recovers a meaningful fraction of the composite gap
(expected), the architecture axis is CLOSED with the complete card set (T2.0–T2.4) → the paper's negative
("MTL reg gap irreducibly architectural; ship composite") is final. If a hybrid surprises → promote,
re-judge under MTL + HGI sanity probe (2 seeds × AL+AZ × 5f×30ep), compose with base_a. Then advisor pass
→ update `PAPER_UPDATE.md` + this HANDOFF → surface to user (tier-boundary cadence).

---

## 0c. ⭐ NEXT AGENT STARTS HERE — T2P.0 (the linchpin; decisive, ~few GPU-h)

**Run T2P.0 FIRST** (INDEX `#T2P-0`) — it decides whether Tier 2P's primary lever is staged training
(T2P.1) or asymmetric per-task recipe (T2P.2). It is one clean knob change on an existing cell.

**The experiment.** The `private_only prior-OFF` dose-response cell (AL 52.32) ran at the joint **wd=0.05**;
the frozen (c) STL reg ceiling (62.88) used **wd=0.01**. Re-run that exact cell at **wd=0.01**, everything
else identical to (c):
```
train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_design_k_resln_mae_l0_1 \
  --model mtlnet_crossattn_dualtower --reg-head next_stan_flow_dualtower \
  --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=private_only \
  --reg-head-param alpha_init=0.0 --reg-head-param freeze_alpha=True \
  --category-weight 0.0 --weight-decay 0.01 --scheduler onecycle --max-lr 3e-3 \
  --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 --log-t-kd-weight 0.0 \
  --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state> --no-checkpoints
```
AL+AZ+FL, 5f×50ep seed42, seeded per-fold log_T. **Pre-check (advisor):** confirm in `mtl_cv.py` that
`--category-weight 0.0` zeroes cat's gradient into the shared/cat params (not just the loss scalar) — in
`private_only` mode reg never touches the cross-attn, so with cat-weight 0 the only residual difference
from STL is the joint loop (`max_size_cycle` mixed-batch iteration + the shared optimizer/scheduler step).

**Decision.** Recovers to ≈62 (STL-level) → the collapse was wd/recipe mismatch, the joint loop is NOT the
poison → **T2P.2 (asymmetric per-task recipe) primary.** Still ≈52 → the joint LOOP caps reg even with
identical arch+HP → **T2P.1 (staged / sequential) primary.** Then STOP + surface to the user (tier-boundary
cadence) before launching the chosen lever. Honest-framing reminder: staged reg→freeze→cat gives reg≈STL by
construction — the real question is whether CAT survives the frozen-reg trunk (composite-quality in one
model); the 2-model composite is the null every Tier-2P arm must beat. See INDEX `#tier2p`.

---

## 1. Where we are (pre-Tier-2 context, all still true)
- **Tier 0 + Tier 1 are COMPLETE and FROZEN.** The (c)/(d) STL ceilings are the immutable track yardstick
  (UNTOUCHED by Tier 2; `t14_freeze_sanity.py` GREEN).
- **Tier S (STL head search) is COMPLETE — a reviewer-proof NEGATIVE**: the head is NOT the lever.
- **Tier 2 (T2.1 dual-tower) is now COMPLETE (NEGATIVE)** — see §0 above.
- A major out-of-band finding (overlapping windows) was validated + documented as future-work; **the
  non-overlapping canon is deliberately KEPT** for whole-study consistency.

## 2. The FROZEN ceilings (immutable — T2-T5 Δ are measured against these)
Recipe: **reg** = `next_stan_flow` α=0 (log_T prior OFF); **cat** = `next_gru` logit-adjust τ=0.5.
| | AL | AZ | GE | FL |
|---|---|---|---|---|
| (c) cat macro-F1 | 49.97 | 51.01 | 58.12 | 69.97 |
| (c) reg Acc@10 | 62.88 | 55.11 | 58.45 | 73.31 |
| (d) composite reg (max v14/HGI α0) | 63.58 | 55.11 | 58.76 | 73.62 |
Guard: run `scripts/mtl_improvement/t14_freeze_sanity.py` after ANY ceiling change (asserts cat-ceiling
arch == NextHeadGRU + every ceiling ≥ the MTL it bounds). Currently GREEN. **Caveat:** these are
seed=42 single-seed; §0.1 paper numbers use seeds {0,1,7,100}.

## 3. What got done this session (newest first) + pointers
1. **Overlap-window study** (`future_works/overlapping_windows.md`, `results/.../overlap_window_probe.md`):
   non-overlapping windows cap ceilings ~+5–9.8pp at small-state STL (head-independent). Validated harness
   + real STL + real MTL. **Documented as future-work; canon kept.** See §4.
2. **Pipeline deep-dive audit** (`PIPELINE_AUDIT_2026-06-03.md`): the windowing cap (HIGH) + secondary levers
   (58% short-user drop, cat fp16-autocast-no-scaler, 50ep overfit). Everything else CLEAN.
3. **Critical advisor on heads + per-task tuning + STAN** → STAN-attention for cat FALSIFIED even tuned
   (−7.4pp); next_lstm tuned ties but never wins. Mechanistic: recurrence wins cat, attention wins reg —
   the frozen ceilings already use the right head per task.
4. **Tier S Prong A + B**: all coded heads + new SSM (`next_mamba`) + SimGCL aux (`next_gru_simgcl`) — none
   beats the incumbent. (INDEX S.1/S.2/S.3/S.4.)
5. **CAT CEILING BUG (caught + fixed)**: `train.py --cat-head` is silently ignored on `--task next` (MTL-track
   only) → the cat ceiling had run `next_single` not `next_gru`. Re-pinned with `--model next_gru` (the +8pp
   AL correction in §2). Two advisor passes confirmed the re-pin sound + added the freeze-sanity guard.
6. **T1.4** (the tuning that set the frozen ceilings): new leak-free loss code `src/losses/calibrated.py`
   (logit-adjust/focal/tail-loss; 19 tests). Full per-task tune → reg α=0 + cat logit-adjust τ=0.5 win.

## 4. The overlap finding — what the next agent MUST understand
Non-overlapping windows (stride=9) train on ~8.5× less data than overlap (stride=1). Validated at AL (real
pipeline): **cat rising-tide** (STL +9.77 / MTL +8.92), **reg gap WIDENS** (STL +5.13 / MTL +0.50; the
STL→MTL reg gap 8.34→12.96). The MTL reg shared-backbone bottleneck *can't absorb the extra data* → this
**STRENGTHENS the regime/dual-tower thesis**. HGI behaves like v14 (+4.89), stays tied. **Decision: keep
non-overlap canon; it's future-work.** All probe code is isolated in engine `check2hgi_dk_ovl` — the
canonical substrate is UNTOUCHED. If ever adopted, see the rebuild checklist in the future-work memo.

## 5. Traps / gotchas (don't repeat these)
- **`--cat-head`/`--reg-head` are MTL-track-only.** For STL `--task next`, use `--model <head>`. Always
  verify the arch that actually ran (`results/.../model/arch.txt`).
- **Sanity-check any STL ceiling ≥ the MTL it bounds** (the cat bug was visible as ceiling < MTL). Use
  `t14_freeze_sanity.py`.
- **`--per-fold-transition-dir` must be the seeded log_T** (`region_transition_log_seed{S}_fold{N}.pt`);
  default log_T leaks ~+3pp. Stale-log_T guard: mtime(log_T) > mtime(next_region.parquet).
- **The registry silently drops unknown kwargs** — a head can ignore a param you think you set.
- **Frozen folds + the moving-baseline guard**: Tier-S/T5 winners feed candidates, NEVER re-open (c)/(d).
- Repo pre-stages unrelated `articles/*` — always `git add` with explicit pathspec + check `git show --stat`.

## 6. New reusable assets (this session)
- `src/losses/calibrated.py` (+ test) — leak-free logit-adjust/focal/balanced/CB/LDAM; wired into
  `next_cv.py` via `ExperimentConfig.loss_calibration` + `p1` + `train.py` flags.
- Heads: `next_mamba` (selective-SSM), `next_gru_simgcl` (SimGCL aux + `model.aux_loss` trainer hook in
  `_single_task_train.py`). Both sound but lose/tie — keep as tested assets, not in use.
- Overlap infra: backward-compatible `stride` param (`core.py`/`builders.py`), engine-aware region-seq
  (`region_sequence.py` + `folds.py` + `p1` `seq_engine`), isolated engine `check2hgi_dk_ovl`,
  `build_overlap_probe_engine.py`, `overlap_probe.py`.
- Scripts: `scripts/mtl_improvement/` — `t14_*` (T1.4 sweep/validate/repin/agg/sanity), `tierS_*`
  (screen/confirm/unit), `stan_for_cat.sh`, `overlap_*`.

## 7. THE NEXT STEP (open decision — surface to the user, do NOT autopilot)
Per the AGENT_PROMPT tier-boundary cadence, the user decides. The clear next is **Tier 2 — T2.1 dual-tower**
(INDEX `#tier2`): the reg-private full-STAN backbone vs the frozen (c)/(d), regime×substrate 2×2, frozen-fold
paired, with the mandatory **unit-test gate + per-arch LR mini-sweep** (hard rule 7/10) BEFORE the multi-fold
launch. The overlap finding makes T2 better-motivated (more data widens the reg gap → the bottleneck is real).
Other open/optional items: T4.0 loss-scale/RLW litmus (cheap, ungated); the cheap training levers (cat fp32 vs
fp16-autocast, shorter 25ep schedule — audit MED); overlap-pattern confirm at AZ/GE/FL + multi-seed.

## 7b. Audit close-out (O1–O5) — ✅ ALL CLOSED 2026-06-04 (`AUDIT_TIER1_TIERS_2026-06-03.md §6`)
The 5 audit items are closed + advisor-reviewed (leak audit: NONE). Full write-ups: `TIER01_RESULTS.md
§Audit close-out`. Frozen (c)/(d) UNCHANGED; `t14_freeze_sanity.py` GREEN. Commits `4fba15b` → `b94b29f`
→ `87e3f62`.
- **O1 (α=0 "prior is a drag") — both audit hypotheses FALSIFIED.** A faithful re-run (`o1_alpha_probe.py`;
  reproduces 62.32/70.28/52.87/55.81) shows the learnable α converges **large** (AL +0.45 / AZ +0.79 / GE
  +0.94 / FL +1.09 — larger at higher-coverage states, n=4 suggestive), i.e. the model *leans into* the
  prior, yet prior-ON stays 0.56–3.03pp BELOW α=0. The prior carries real signal (standalone Acc@10 50.86/66.15
  ≈ Markov-1-region floors 47.01/65.05). **Reframed claim: "the fixed additive log_T prior is a net drag on
  the STL-reg ceiling"** — NOT "embeddings subsume transitions," NOT a stuck-α bug; mechanism (train/val gap
  vs additive scale-mismatch vs double-counting) NOT isolated. Strengthens the §2c HGI-prior-artifact corollary.
- **O2 (Tier-S cat crack) — multi-band negative HOLDS.** Multi-seed {0,1,7,100}: next_lstm's single-seed wins
  evaporate → tie at all 4 states. next_single GE +1.54±0.17 (robust) but GE-SPECIFIC (AL −8.11) → fails the
  ≥2-band gate → a **T5.2 candidate** (re-judged under MTL), does NOT re-open (c). NB the per-state GE-cat STL
  ceiling is next_single 59.66 > (c) 58.12; (c) is the scale-robust incumbent, not the per-state max.
- **O3 (FL (c)-cat inversion).** Multi-seed 69.96±0.08 validates seed42 69.97; the −0.30pp inversion vs MTL
  diag-best 70.26 PERSISTS multi-seed (not a seed artifact) but is tiny + explained (oracle epoch + small FL
  cat transfer); (c) validly bounds the *deployable* MTL cat (≫66.73). Not a bug. CAT-side, orthogonal to T2.
- **O4** next_hybrid accounted (AL cat 49.34 < floor; reporting omission) + `*_hsm` deferral noted.
- **O5** paper limitation (vi) (non-overlap windows + AL rebuttal) added to `PAPER_DRAFT.md §7`; dense-rebuild
  deferred to `future_works/overlapping_windows.md`.

## 8. How to resume
1. Read this → `log.md` (the two 2026-06-04 entries + advisor pass) → `INDEX.html` §"The regime finding" + the
   Tier 2 cards → `TIER01_RESULTS.md` (incl. §Audit close-out).
2. `git pull`; confirm `t14_freeze_sanity.py` is GREEN.
3. For Tier 2: follow §9 below.
4. STOP + surface at the tier boundary (advisor pass → summary → user decision).

## 9. TIER 2 onboarding (next agent starts HERE) — added 2026-06-04
**You are starting Tier 2. Tier 0/1/S + the audit close-out are DONE and FROZEN; nothing upstream is open.**
The headline (the regime finding) is confirmed at AL/AZ/GE/FL (`v14_mtl_vs_canonical.md`): v14 ≈ matched
canonical in MTL — the STL substrate gains wash out jointly. **The locus is the joint-training architecture, not
the substrate or the per-task head** (Tier-S proved the head is not the lever; T1.3 proved the upstream encoder
is not the residual). Tier 2 attacks that locus.

**The one experiment: T2.1 — reg-private dual-tower** (INDEX `#tier2`). Build a reg-private full-STAN backbone
(the §6.4 decomposition says ~75% of the MTL→STL reg residual is the *missing private backbone*) so the reg head
stops sharing the cross-attn/shared trunk with cat. Primary arm = gated-fusion (b); + a PCGrad-off arm.

**Hard gates BEFORE any multi-fold launch (do NOT skip — these are why prior arch swaps collapsed):**
1. **Unit-test gate** (hard rule 10): forward/backward shapes on a synthetic 100-user batch, loss-finite,
   param-count within ~5% of B9 at D=256, and `shared/cat_specific/reg_specific_parameters()` partition
   bijective+exhaustive — **the dual-tower's private backbone is a NEW param group; wire it into the partition**
   (a silent omission here = the F49 class of bug).
2. **Per-arch LR mini-sweep** (hard rule 7): 5 regimes × 5f × 40ep × seed42 × AL+AZ, then full-protocol at the
   winner. (The B9_STL_STAN_SWAP collapse = B9 recipe blindly applied to a non-α head — don't repeat it.)
3. Stay at `shared_layer_size=256` (F51 widening falsified). No fclass-as-feature. log_T-KD ON, seeded per-fold
   log_T mandatory.

**Design discipline:** frozen-fold paired (hard rule 2b) — score **Δ vs the frozen (c)/(d)**, not bare absolutes.
Run the regime×substrate 2×2: {v14-fresh, canonical-fresh `gcn_ctrl`} × {B9, dual-tower}. HGI sanity probe per
promoted arch (2 seeds × AL+AZ × 5f × 30ep; escalate if |MTL+HGI − STL+HGI| ≥ 2pp).

**The yardstick you measure against (FROZEN — do NOT recompute or re-pin; see §2):**
| | AL | AZ | GE | FL |
|---|---|---|---|---|
| (c) cat macro-F1 | 49.97 | 51.01 | 58.12 | 69.97 |
| (c) reg Acc@10 | 62.88 | 55.11 | 58.45 | 73.31 |
| (d) composite reg | 63.58 | 55.11 | 58.76 | 73.62 |
| MTL deployable reg (v14, the (a) baseline T2 must beat) | 50.14 | 37.78 | 42.64 | 61.21 |

The composite (d) beats single-model MTL reg by **+12.4 to +17.3pp** — that is the gap the dual-tower must close
*inside one model*. If nothing recovers a meaningful fraction → composite is the deploy fallback (a paper-grade
negative). **Do NOT add AdaShare/Learning-to-Branch** (collapse to branch-depth at 2 tasks, already spanned).

**Carry-overs from the close-out into Tier 2/5:** (i) the O1 reframe — log_T is a KD loss in MTL (helps) vs an
additive bias in STL (hurts); T3.1 will re-sweep log_T-KD on the new stack, so do not assume the prior behaves
the same. (ii) `next_single@GE` is a logged T5.2 cat candidate (state-conditional; re-judge under MTL, do not
auto-pick). (iii) Tier S is an OPEN sandbox running parallel to Tiers 2-4 (must not starve the regime headline).

**Files to read for the build:** `docs/findings/B9_STL_STAN_SWAP_AZ_FL.md §6.4` (gap decomposition + the residual-skip falsification),
`future_works/mtl_architecture_revisit.md`, `src/models/mtl/mtlnet_crossattn/model.py` (current backbone),
`src/models/mtl/mtlnet_crossstitch/model.py` (scaffolded for T2.2), `src/training/runners/mtl_cv.py` +
`src/training/helpers.py` (`setup_per_head_optimizer`). Drivers template: `scripts/_v14_run/` (currently SERIAL —
that's the parallelization headroom; MPS-collocate small states per AGENT_PROMPT §14).
