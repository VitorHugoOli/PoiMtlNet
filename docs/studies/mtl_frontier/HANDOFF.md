# mtl_frontier — HANDOFF (continue R4–R9) ⭐ read this first

**You are the next `mtl_frontier` agent.** Wave 1 (R1/R2/R3) + R10 + 3 user follow-ups + conditional
coupling + **R-CC+** are DONE (2026-06-17, branch `study/mtl-frontier`). **R4–R9 are OPEN** (R-CC+ was
the live thread and is now CLOSED NULL — start at **R4**). This file is your entry point: the state, the
regime you're working in, the reusable code, the R4–R9 specs (updated with what we learned), and the
protocol you MUST follow.

> **R-CC+ closed 2026-06-17 (NULL).** The conditional-coupling family is fully mapped along signal
> {calibrated/argmax/topk} × injection {FiLM/input-side concat} × output-side {learned logit prior}.
> **No variant exceeds the original additive-posterior cc** (FL multi-seed: cc_e2e +0.235, calib +0.237,
> argmax +0.214 — all 4/4 seeds +, reg +0.066…0.070 p<0.05; **input-side concat washes out +0.033**;
> output-side logitp +0.016 null; richer `features` already HURT). The +0.235 sub-threshold cap is the
> **regime** (weak 2.8-bit auxiliary), not the injection knob. Advisor audit 5/5 PASS (G bit-identical,
> leak-free). Code: `cond_signal`/`cond_temp`/`cond_topk`/`cond_inject`/`cond_logit_prior` (G unchanged).
> **One untested future lever (not pursued):** cross-attn cat↔reg coupling (cat penultimate as K/V queried
> by the reg pooled feature) — expected sub-threshold. See `FINDINGS.md §R-CC+`. **Next priority: R4.**

> **Read in this order:** this file → [`FINDINGS.md`](FINDINGS.md) (every result + mechanism) →
> [`STATE.md`](STATE.md) (queue + decisions log) → [`AGENT_PROMPT.md`](AGENT_PROMPT.md) (original scope) →
> [`docs/research/mtl_frontier.md §4`](../../research/mtl_frontier.md) (R-program rationale + citations) →
> [`../archive/mtl_improvement/FINAL_SYNTHESIS.md`](../archive/mtl_improvement/FINAL_SYNTHESIS.md) (the regime).

---

## 1. Where the study is (one paragraph)

8 levers tested, **7 nulls + 1 genuine sub-threshold positive**; **champion G is unchanged, no v17
promotion, nothing flows to `closing_data` G0.2.** The output-prior family (R1 log_C, R3 CrossDistil),
the sharing-topology family (R2 binary AFTB, R10 learned GRM), and input-dependent fusion (aux_gated) are
all null/harmful — they re-gate the **cos≈0 cross-task gradient**, which is empty. The ONE thing that
produced real transfer is **input-side conditional coupling** (feed the cat head's predicted posterior as
an *input feature* to the reg head, iMTL/GETNext): FL cat **+0.235** + reg **+0.070** (4/4 seeds positive,
audit-confirmed deterministic) — but the **weak 7-class (~2.8-bit) auxiliary caps it below the 0.3 gate**,
and a richer 256-dim conditioning HURTS (−0.31). Net: the post-2022 MTL frontier *replicates* champion
G's two wins (dual-tower + log_T-KD) but does not exceed them in this regime — a strong, citable negative,
with conditional coupling as the one place a real (capped) gain lives.

## 2. The regime — what is already settled (do NOT re-litigate)

- **cos(∇cat,∇reg) ≈ 0** on the shared trunk, tested intrinsic (16 runs, n=3,797). ⇒ no first-order
  cross-task transfer through the gradient; re-gating it (R2/R10/aux_gated) is empty. **Anything that
  works must change the reg head's INPUT or OUTPUT supervision, not the trunk gradient.**
- **Data-rich main task + weak 7-class auxiliary** ⇒ gains are small and concentrate at small states;
  conditional coupling's gain is real but capped by the 2.8-bit category.
- **log_T-KD saturates the output-prior family** (R1/R3 add nothing over it).
- **The dual-tower is the architecture optimum** (aux fusion + β→0 + prior-OFF; `gated`/`aux_gated`
  lose; capacity falsified 5 ways in mtl_improvement).
- **Optimizer aisle is CLOSED** (19-arm null + Kurin/Xin/Mueller). Only R9 residual sanity arms remain.
- **Scale axis:** AL (~1.1k regions) is the noisiest cat state; FL (~4.7k) is the tightest and the
  paper-headline state. **A lever that helps only AL and not FL is not a champion lever** (inverse-G′).

## 3. Reusable infrastructure (already built, champion G unchanged — all flags default-off)

| Mechanism | How to enable | Builder / code |
|---|---|---|
| log_C co-location KD (R1) | `--log-c-kd-weight W` | `scripts/compute_region_colocation.py --per-fold --seed S --engine ENG` (builds P(region\|cat) **and** P(cat\|region), per-fold/seed, leak-clean) |
| log_C warm-up + error-correction (R3) | `--log-c-kd-warmup-epochs N --log-c-kd-ec-lambda L` | `mtl_cv.py` KD branch |
| reverse reg→cat KD (R3) | `--cat-kd-weight W` (needs the colocation file) | `log_C_rev` buffer on the reg head |
| directional STEM-AFTB gates (R2) | `--model-param aftb_spec="none,ab+ba"` (per-block, `+`-join {ab,ba}/none) | `_CrossAttnBlock.detach_ab/ba` in `mtlnet_crossattn/model.py` |
| GRM-gated cross-attn read (R10) | `--model-param crossattn_grm=True` (logs `grm_gamma_*`) | `_CrossAttnBlock` + `_masked_mean_seq` |
| aux_gated fusion (input-dep β) | `--reg-head-param fusion_mode=aux_gated` | reg head `_fuse` (logs `aux_gamma`) |
| **conditional coupling (the one that worked)** | `--reg-head-param cond_coupling=posterior` (or `features`) `--reg-head-param cond_dim=7` (256 for features) `--reg-head-param cond_detach=False` | reg head `cond_proj` (zero-init → ≡ G) + `mtlnet_crossattn_dualtower.forward` + `next_gru.forward_features` (logs `cond_norm`) |
| **R-CC+ family axes** (CLOSED null) | `--reg-head-param cond_signal=`{softmax,calibrated,argmax,topk} `cond_temp=T` `cond_topk=k` · `cond_inject=`{add,film,concat_seq,none} · `cond_logit_prior=True` | reg head `next_stan_flow_dualtower` (FiLM/concat_seq/logit injections, all zero-init → ≡ G) + signal transform in `mtlnet_crossattn_dualtower.forward` (logs `cond_norm`) |

**Screening pattern (copy it):** a `*_screen.sh` driver (PID-suffix rundir capture, `--no-checkpoints`,
writes a `*_manifest.tsv`) + a `*_agg.py` (reads the manifest, matched reg = `top10_acc_indist·(1−ood)`
@ indist-best epoch, cat = `diagnostic_task_best.next_category.f1.mean`, gate ≥0.3 either head). Every
prior lever's driver+agg+manifest+results JSON are in `scripts/mtl_frontier/` and
`docs/results/mtl_frontier/` — clone the closest one.

## 4. Champion-G baseline invocation (the comparand for everything)

```bash
python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region \
  --engine check2hgi_design_k_resln_mae_l0_1 --state <st> --seed <S> --epochs 50 --folds 5 --batch-size 2048 \
  --no-reg-class-weights --no-cat-class-weights --mtl-loss static_weight --category-weight 0.75 \
  --cat-head next_gru --reg-head next_stan_flow_dualtower \
  --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
  --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
  --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
  --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --model mtlnet_crossattn_dualtower --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<st> --no-checkpoints
```
(R1/R3 used this **+ `--log-t-kd-weight 0.2`** as the comparand, since they add to log_T-KD. R2/R10/cc
used the pure KD-off form above.) v14 substrate + seeded per-fold log_T built for AL/AZ/GE/FL; **CA/TX
have NO substrate** (a build is `closing_data` scope, deferred).

## 5. PROTOCOL — non-negotiable (the lessons that cost us)

1. **Champion G is DETERMINISTIC** (verified: 2 runs/seed bit-identical). So run-to-run noise is 0 —
   BUT **run matched same-batch baselines anyway** when you can. We reused single baselines across many
   levers; it was harmless only because G is deterministic, and it created a confusing "seed-0 always
   best" pattern (seed-0 is genuinely the weakest seed → levers lift it most). If you ever see "all
   levers best at one seed," check whether that seed's *baseline* is just low.
2. **CODE DRIFT is real:** champion-G absolute numbers differ ~0.1–0.4 cat between the `mtl_improvement`
   (June-6) code and current code. **Never compare mtl_frontier absolute numbers to the old R0 bar** —
   always re-baseline on the current code, in-study.
3. **Multi-seed {0,1,7,100} before ANY claim.** Every single-seed flare (R3-rev +0.45, R10 +0.32, cc AL
   +0.96) washed out multi-seed. Promote gate = **≥0.3 pp either head, 4-seed mean**, paired Wilcoxon
   (reg over n=20 fold-seed pairs, cat over n=4 seeds), report n and p.
4. **Scale-conditional check:** screen AL+FL seed-0 first. **FL (headline) decides** — an AL-only effect
   is not a champion lever. If positive multi-seed at FL → STOP and hand to user (recipe → v17) + write
   the gate row into `closing_data/PLAN.md` G0.2.
5. **C28 mechanism-fires:** assert your lever actually fires (log a diagnostic — γ, cond_norm, a
   "FIRED" line). A dead codepath produces a confident false null (mtl_improvement C28). We logged
   `grm_gamma`, `aux_gamma`, `cond_norm`, and KD "FIRED" lines — keep that discipline.
6. **Leak hygiene:** any train-derived prior/matrix is **train-userids-only, per-fold, per-seed**, built
   like `compute_region_transition.py` / `compute_region_colocation.py`; run the stale-log_T freshness
   preflight (`CLAUDE.md`) before every run; the trainer hard-fails on stale/mismatched-seed/n_splits.
7. **Keep champion G bit-identical** when your flag is off (verified for every lever — the audit agent
   re-checked). Commit per lever; pin `--canon none` + full explicit recipe in every driver.

## 6. R4–R9 queue (specs updated with what we learned)

> Source rationale + citations: `docs/research/mtl_frontier.md §4`. Re-prioritized given the wave-1
> result: **conditional coupling is the only family that produced real transfer — push it FIRST.**

### ✅ R-CC+ (CLOSED 2026-06-17 — NULL, the falsifier fired) — conditional-coupling family fully mapped
All the probes below were run. **The falsifier fired:** no conditioning variant clears FL 0.3 multi-seed
→ the **2.8-bit category is a hard regime cap** (the honest paper framing). Details:
- **Signal** (calibrated-τ / discrete-argmax-GETNext / top-k): all **tie** the plain posterior (FL
  multi-seed calib +0.237, argmax +0.214 vs cc_e2e +0.235) — signal form is not the bottleneck; the
  ~2.8-bit content is. (`cond_proj(softmax)` is already a soft cat-embedding lookup.)
- **Injection** (FiLM, input-side concat-into-sequence): both **worse** than additive; the input-side
  concat was multi-seeded to rule out an identity-init confound and **washed out** (+0.033, 2 seeds neg)
  → additive-late is the family optimum.
- **Output-side logit prior** (CatDM/LBPR `logits_reg += W·P̂(cat)`): **null** (+0.016) — the output
  channel is saturated (same wall as R1/R3).
- **Richer `features`** (256-dim penultimate): already shown to **HURT** (−0.31). Raising raw capacity is
  worse, not better.
- **Conclusion:** champion G unchanged, no v17; the cc direction is closed at sub-threshold. See
  `FINDINGS.md §R-CC+`. The one mechanism NOT yet run (deferred): cross-attn cat↔reg coupling.

### ✅ R4 — Pareto-front profiling — DONE 2026-06-17 (paper-narrative; resolves C21)
Profiled the cat↔reg front on the frozen champion (FL multi-seed). **(1) Loss-weight/mixture axis is a
near-corner** — champion cw=0.75 Pareto-dominant (lowering→0.55 = +0.05pp reg/−0.87pp cat; raising→0.85
dominated): tasks weakly coupled (the falsifier's publishable regime datum). **(2) Deployment-epoch axis
carries the real, STABLE trade** — 12–16 Pareto epochs/run, geom_simple ep18–20 every seed → **C21 is an
epoch-deployment choice, not a weight/arch problem**; publish the epoch-front + geom_simple pick.
**PaLoRA-proper NOT built** (mechanistic: shared-trunk adapter mixture can't move the private-tower reg →
reproduces the near-collapsed weight-front; the dual-tower's reg-privacy defeats a shared-trunk
Pareto-profiler — itself a citable point). See `FINDINGS.md §R4`. **Next: R5.**

### ✅ R5 — Per-instance KD gating — DONE 2026-06-17 (NULL; gate fired but comparand artifact)
Gated log_T-KD by Markov-coverage (covmax=teacher max-prob, coventr=entropy), batch-mean-fixed. FL
multi-seed **clears the ≥0.3 cat gate vs the global-W-KD-ON base** (covmax +0.472, 4/4 seeds) — but that
is a **comparand artifact**: log_T-KD(0.2) is cat-harmful at FL (−0.70 cat), gating only recovers ~2/3 of
it, and vs the true KD-OFF champion R5 is **−0.224 cat / −0.063 reg (worse on both)**, dominated by the
trivial "KD-off at FL". reg ≤ global-W (falsifier met); AL gate fails. **No v17** (advisor-audited).
**Lesson: read the promote-gate against the deployable champion, not a lever's internal control.** Code
`--log-t-kd-gate {none,coverage_max,coverage_entropy}` (G bit-identical off). See `FINDINGS.md §R5`. **Next: R9.**

### R6 — ForkMerge-style weight forking — MED
Periodic forks with different (w_cat,w_reg); select/merge on **validation** error. Conflict-agnostic,
K=2-friendly, optimizer-agnostic. **Falsifier:** merged ≤ champion G.

### R7 — Merge-vs-joint (ZipIt!/SIMO) — MED, expect citable negative
Warm-start shared trunk → fine-tune two STL specialists → partial-depth merge (share early, privatize
late — the merging-side mirror of the dual-tower). **Expectation bounded by tangent-space theory**
(from-scratch experts don't share a basin) → likely Merge < G, but a clean LBSN-space negative.

### R8 — Auxiliary third task: next-visit time — LOW, rising-tide caution
GETNext/Where-and-When pattern (time loss as auxiliary). **MUST run with a paired STL control** — the
rising-tide rule predicts it lifts STL as much as MTL. **Falsifier:** lifts STL and MTL equally.

### R9 — Residual optimizer sanity — TRIVIAL, expected null
(a) **BayesAgg-MTL** — verify the `src/losses/registry.py` impl matches ICML'24 and whether it was in the
19-arm null before re-running (weights by gradient *uncertainty*, the one non-vacuous mechanism at
cos≈0). (b) **Smooth-Tchebycheff** — only if R4's measured front is non-convex. Closes the optimizer
aisle citably.

**Sequencing recommendation (R-CC+ now done):** **R4** (paper-narrative, runs on the frozen champion) →
R5 (cheap, reuses code) → R9 (trivial close-out) → R6/R7/R8 (medium, lower priority). Do NOT re-open the
trunk-gradient / output-prior / sharing-gate / conditional-coupling families — all nulled now.

## 7. Logistics

- **Branch** `study/mtl-frontier` (this work; do NOT commit to `main`). 26 commits; PR to be opened.
- Everything is committed: code (champion G unchanged), drivers, aggregators, manifests, results JSON,
  FINDINGS/STATE/this file, `docs/studies/log.md` rows.
- **A40**, ~unmetered. FL run ≈ 30 min (5f/50ep); AL/AZ/GE faster. CONC=2 is safe on the A40 (~24 GB/run).
- When you close an R-lever: FINDINGS.md §R<n> (mechanism + numbers), STATE.md row + decisions-log entry,
  one `docs/studies/log.md` row. Promote → STOP + user sign-off + `closing_data/PLAN.md` G0.2.
