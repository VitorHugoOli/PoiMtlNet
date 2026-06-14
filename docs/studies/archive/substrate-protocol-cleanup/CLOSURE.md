# substrate-protocol-cleanup — Study CLOSURE

**Closed:** 2026-05-29
**Predecessor:** [`../mtl-protocol-fix/`](../mtl-protocol-fix/) (CLOSED 2026-05-24, v6 final)
**Parallel (independent branch `mtl-improve`):** [`../mtl_improvement/`](../mtl_improvement/)
**Full chronological record:** [`log.md`](log.md) · **Tier specs + gates:** [`INDEX.md`](INDEX.md)

This is the one-page closure of the study. It synthesises five Tiers of cheap, non-architectural cleanup carved out of `mtl-protocol-fix/DEFERRED_WORK.md`. The study did **not** need a champion — its purpose was to land cleanly-closed verdicts for items orthogonal to the architectural revisit happening in parallel in `mtl_improvement`.

---

## Headline

- **Tier A1 (log_T-KD) is the single PROMOTION** — a small-state, paper-grade reg lift (AL +2.27 pp, AZ +4.91 pp, n=20, p=9.54e-07, 20/20 folds), leak-audited clean, orthogonal to the B9 recipe. Large-state pilot TRANSFERS (FL/CA/TX positive) but is a seed=42 pilot, **not paper-grade**.
- **Tier B CLOSED the substrate axis: no design promoted under MTL+F1** — all four variants (Designs B/J, Lever 4, Lever 5) are NULL/FALSIFIED at AL/AZ, on BOTH the disjoint oracle front AND the deployable joint/geom_simple front (`tier_b/phase_b_two_front.md`: all |Δreg|≤1.22 pp, every p≥0.21). **[RE-AUDIT-CORRECTED 2026-05-29, `tier_b/phase_b_reaudit.md`; hedged per verification, see log.md]** The accurate mechanism is **anchor dominance**, not "non-transferring substrate": under the joint config the MTL reg head is dominated by its α·log_T transition anchor — with α frozen to 0 (an out-of-training config), reg sits at near-floor (all-epoch mean ~1.1% vs ~0.9% pure chance for 1109 regions) for BOTH design and canonical, so the substrate-carrying encoder branch contributes ~nothing beyond the prior and cannot move MTL reg. The substrate's STL reg advantage is **real and reproduces** (AL +2.34 pp, Wilcoxon p=0.0312, with the prior; a separate FL no-prior ablation gives J−canonical +0.86 pp). The ~−2.4 pp cat is a **build-scope confound** (every design build re-trains a fresh-init CheckinEncoder, drifting the cat input 100 % vs canonical; at α=0 the cat Δ is +0.19 pp), NOT a substrate cost. Corrected wording: Tier B measured *"no reg gain beyond the canonical log_T anchor under the joint config,"* not *"the substrate fails to transfer."* (Avoid the absolute "reg encoder is inert / α=0 floors at chance" — chance is ~0.9% not 5.5%, the 5.5% was a best-epoch readout, and α=0 is OOD; the claim is regime-scoped.)
- **Tier C closed §4.4 + the P4 residual hole** — C2 falsifies the last unfalsified curriculum form (freeze-reg-after-peak); C3 exonerates cat→reg cross-attention K/V capacity-stealing. Both push the residual MTL-vs-STL reg gap firmly to **architectural** → hands to `mtl_improvement`.
- **Tier C1 is a one-state §Discussion footnote** — 3-snapshot routing clears +2 pp at AZ (Δreg +2.54, p=0.031, 5/5) but fails at AL on one genuine degenerate Acc@1-selected snapshot. Not a promotion.
- **Tier D1 (window/mask audit) is CLEAN** — no leak; shared artefact with `mtl_improvement` T0.2.
- **Tier C4 (added)** — C22 stale-log_T mtime guard landed in `src/training/runners/mtl_cv.py` (defensive, near-zero compute).

---

## Per-Tier verdict table

| Tier | Item | Verdict | Scope / grade | Source |
|---|---|---|---|---|
| **D1** | Window / causal-mask audit | **CLEAN** (no leak) | code-read, no GPU; shared with `mtl_improvement` T0.2 | [`window_mask_audit.md`](window_mask_audit.md) |
| **A1** | log_T-KD §4.5 multi-seed (W=0.2) at AL/AZ | **PROMOTED** | paper-grade, n=20 (seeds {0,1,7,100}), small-states only | [`tier_a1/phase_a1_verdict.md`](../../../results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md) |
| **A1** | log_T-KD large-state pilot (FL 5-fold; CA/TX 1-fold) | **TRANSFERS** | seed=42 pilot, **NOT** paper-grade (sign-and-magnitude only) | [`tier_a1_largestate/phase_a1_largestate_addendum.md`](../../../results/substrate_protocol_cleanup/tier_a1_largestate/phase_a1_largestate_addendum.md) |
| **A1** | log_T-KD leak audit | **NO LEAK** | 7-vector independent audit | [`F_TIER_A1_LEAK_AUDIT.md`](../../../findings/F_TIER_A1_LEAK_AUDIT.md) |
| **B1** | Design B (POI2Vec @ pool boundary) MTL+F1 | **NOT PROMOTED** (FALSIFIED AL / NULL AZ) | seed=42 5-fold AL/AZ | [`tier_b/phase_b1b2b4_verdict.md`](../../../results/substrate_protocol_cleanup/tier_b/phase_b1b2b4_verdict.md) |
| **B2** | Design J (H + anchor λ=0.1) MTL+F1 | **NOT PROMOTED** (FALSIFIED both) | seed=42 5-fold AL/AZ | same |
| **B4** | Lever 5 (KL distill, orphan rescue) MTL+F1 | **NOT PROMOTED** (FALSIFIED AL / NULL AZ) | seed=42 5-fold AL/AZ | same |
| **B3** | Lever 4 (POI2Vec @ p2r, additive) on canonical | **NOT PROMOTED** (FALSIFIED both) | seed=42 5-fold AL/AZ | [`tier_b/phase_b3_verdict.md`](../../../results/substrate_protocol_cleanup/tier_b/phase_b3_verdict.md) |
| **C1** | §4.1 per-task 3-snapshot routing (variant A) | **§DISCUSSION FOOTNOTE** (one-state pass) | seed=42 5-fold; AZ +2.54 p=0.031, AL fails on 1 degenerate fold | [`tier_c/phase_c_verdict.md`](../../../results/substrate_protocol_cleanup/tier_c/phase_c_verdict.md) §C1 |
| **C2** | §4.4 freeze-reg-after-peak (N∈{2,4,6}) | **ARCHIVE — closes §4.4 entirely** | seed=42 5-fold AL/AZ | `phase_c_verdict.md` §C2 |
| **C3** | P4-residual: zero cat→reg K/V | **P4 FULLY CLOSED** (no finding filed) | seed=42 5-fold AL/AZ | `phase_c_verdict.md` §C3 |
| **C4** | C22 stale-log_T mtime preflight guard | **LANDED** (defensive) | code-only, near-zero compute | `src/training/runners/mtl_cv.py` |

---

## "What done looks like" — closure-criteria map (AGENT_PROMPT §"What 'done' looks like")

| Criterion | Status | Evidence |
|---|---|---|
| A1 has a Wilcoxon **n=20** verdict for log_T-KD at AL/AZ | ✓ | AL/AZ both p=9.54e-07, 20/20 folds; PROMOTED |
| B1/B2/B3 have F1-selector MTL three-frontier numbers + verdict | ✓ | INDEX §B-summary table filled; both verdict docs |
| C1 has a 3-snapshot routing prototype + verdict (Δreg vs joint-best) | ✓ | prototype shipped (`--save-task-best-snapshots`, `route_task_best.py`); footnote verdict |
| C2 has a small-state freeze-reg-after-peak pilot verdict | ✓ | ARCHIVE, closes §4.4 |
| D1 has a written audit confirming/fixing window/mask correctness | ✓ | CLEAN, advisor-verified |

**Also closed (beyond the five stated criteria):** B4 (Lever 5) + B3 (Lever 4) absorbed and falsified; C3 (P4 K/V) fully closed; C4 (mtime guard) landed.

---

## FL EXTENSION (2026-05-29, append-only — does NOT alter the AL/AZ verdicts above)

The Tier B/C verdicts were extended to **Florida (B9 large-state recipe)** as a **THREE-WAY** comparison (canonical c2hgi vs designs B/J/M/L vs **HGI**), framed as gap-closure per `merge_design/STATE.md`. Full doc: [`tier_b_fl/phase_b_fl_3way.md`](../../../results/substrate_protocol_cleanup/tier_b_fl/phase_b_fl_3way.md); C2/C3: [`tier_c_fl/phase_c_fl_verdict.md`](../../../results/substrate_protocol_cleanup/tier_c_fl/phase_c_fl_verdict.md).

- **STL three-way (gethard, with prior):** canonical→HGI gap = **+2.12 pp**. Only **J** is Wilcoxon-strict over canonical (+1.12 pp, p=0.0312), closing **53 %** of the gap; B/M positive but not strict (+0.71/+0.89 pp, p=0.0625). All designs remain strictly below HGI (HGI>design p=0.0312). No-prior cross-check (Test 2) reproduces: J +0.86 pp strict, HGI−J +1.64 pp.
- **MTL three-way (B9):** designs give NO reg gain over canonical on either front (disjoint |Δ|≤0.16 pp, none significant; joint −3.5 pp is a cat-driven geom-selection artefact). **Designs close 0 % of any MTL gap.** Cat disjoint −1.7 to −1.9 pp = the **build-scope CheckinEncoder-reinit confound** (consistent with AL D3), not a substrate cost.
- **THE CEILING — HGI MTL+F1 at FL (evaluated 2026-05-29, user-approved follow-up):** HGI does **NOT** beat canonical in MTL disjoint reg (64.49 ± 0.55 vs 63.98, Δ+0.51, **p=0.41 NS**). HGI's STL reg win (+2.12 pp) **vanishes** under B9 joint training (70.9→64.5 ≈ canonical ≈ designs). HGI cat collapses to 34.84 % F1 (−35.6 pp; non-viable as an MTL substrate, as expected). **MTL flattens EVERYONE — even the STL winner.** So the designs are not "failing to carry HGI's advantage"; HGI has no MTL advantage to carry. Build cost ~0.36 GPU-h. Full doc: [`tier_b_fl/hgi_mtl_fl.md`](../../../results/substrate_protocol_cleanup/tier_b_fl/hgi_mtl_fl.md).
- **ISOLATION cell (the clean experiment AL/AZ lacked):** STL `next_stan_flow` α=0 (log_T fully off) LEARNS region at **~73 % Acc@10** (canonical 72.74, design_b 73.12, Δ+0.37 p=0.0312) — while the IDENTICAL head/config under MTL floors at **~0.03 %** (chance 0.213 % for 4703 FL regions). **Verdict: it is the joint-training REGIME, not the head, that kills the MTL reg encoder.** This is now a single-cell apples-to-apples STL↔MTL contrast (same head/config/state/embeddings). Hedge: α=0 is OOD → the claim is regime-and-config-scoped (B9, 50 ep), not an absolute "encoder can never learn region under MTL." This DISTINGUISHES anchor/regime-dominance (the cause, repeats at FL) from any FL-specific Markov-saturation explanation.
- **Corrected FL headline:** the prior FL-extension claim *"STL→MTL substrate collapse REPEATS at FL"* over-claims (it implies a large STL advantage). Accurate: *the small FL STL design advantage (J +0.86–1.12 pp, partial 53 % gap-closure to HGI) does not survive MTL; designs ≈ canonical in MTL-reg on both fronts; the MTL reg encoder is anchor/regime-limited (isolation cell), so even a better substrate cannot express itself in MTL-reg under this recipe.*
- **C2 / C3 at FL:** both HOLD at large-state scale on both fronts. **C2 §4.4-closed** (no N gives a significant cat gain without reg cost; N=2 hurts reg −7.69 pp p=0.0312); **C3 P4-closed** (zero-cat-kv shifts reg +0.03 pp ns).
- **C1 at FL: CLEARS the +2 pp reg routing gate (+2.80 pp, 5/5 folds, p=0.0312)** — cleanly, no degenerate snapshot (unlike AL fold-3). cat routing +1.98 pp (5/5, p=0.0312). FL is the THIRD state (AZ +2.54 passed, AL failed, FL +2.80 passes); strengthens the §Discussion variant-A case, though an Acc@10-aligned reg-best selector + degenerate-snapshot guard is still warranted before promotion (AL's failure). See `tier_c_fl/c1_route/`.
- **Three converging lines → the bottleneck is the joint-training regime, not the substrate:** (a) STL-α=0 73 % vs MTL-α=0 0.03 % (regime kills the encoder); (b) designs null in MTL reg; (c) **even HGI null in MTL reg** (ceiling). No substrate engineering or prior helps MTL reg under B9; the fix belongs to `mtl_improvement` — make the reg encoder learn region under joint training. (The earlier "HGI MTL un-evaluated" caveat is now RESOLVED by the ceiling run above.)

---

## GPU-cost tally

| Tier | Compute |
|---|---|
| D1 | 0 (no GPU) |
| A1 small-state sweep (16 cells) | ~1 GPU-h |
| A1 large-state pilot (FL 5-fold + CA/TX 1-fold) | ~1.5 GPU-h |
| B (4 substrate builds + MTL cells) | ≪ 2 GPU-h |
| C (C2 6 cells + C3 2 cells + C1 train×2 + C1 re-score×2) | ~6 GPU-h |
| **Total** | **~10–11 GPU-h** — well under the ~40–45 GPU-h study budget |

The low spend reflects that A1 re-used existing recipe code (only the missing `--log-t-kd-weight` flag was implemented), Tier B substrates reuse canonical sequences/folds/log_T verbatim, and Tier C cells are short small-state runs.

---

## Methodological note — the disk incident (Tier A1 large-state pilot)

Mid-pilot, the shared A40 host `/home` hit 100 % full. Root cause was a **second, concurrent driver** (`/tmp/run_a1_large.sh`, not this agent's) running FL/CA/TX **without `--no-checkpoints` and at `--folds 2`**. It (a) wrote multi-GB checkpoints that filled the disk, (b) OOM'd colliding with the in-flight FL run, and (c) rebuilt CA/TX log_T at `n_splits=2` — which the **C19 guard correctly hard-failed** (no leak slipped through). Recovery: killed the rogue tree, deleted disposable checkpoints, rebuilt CA/TX log_T at `n_splits=5`, re-ran affected cells with `--no-checkpoints`.

**Lessons (now operating discipline for any shared-host promotion run):**
1. `--no-checkpoints` by default for disposable promotion runs on a shared host.
2. **Disk free-space**, not just GPU memory, must be a pre-flight gate (≥3 GB STOP threshold used thereafter).
3. The C19 n_splits+seed guard is load-bearing: it caught a concurrent-agent leak vector that no human noticed in real time.

---

## What hands to `mtl_improvement` (architectural residual reg gap — now TRIPLY confirmed)

The residual −7 to −17 pp MTL-vs-STL reg gap (RESULTS_TABLE §0.1) is now exonerated of every non-architectural cause this study could test:

1. **P4 cat-parameter pathway** (prior, `mtl-protocol-fix` Phase 2 P4): frozen cat encoder params do not recover MTL reg.
2. **C3 cat-activation pathway** (this study): zeroing cat→reg cross-attention K/V does not recover MTL reg or delay its peak — exonerates the activation channel P4 left open.
3. **B substrate axis** (this study): four mechanistically-distinct substrate variants show no MTL+F1 reg gain (on both fronts). **[RE-AUDIT 2026-05-29, hedged per verification]** The locus is the **reg head**, not the substrate: under the joint config MTL reg is dominated by its α·log_T anchor (α=0 → near-floor), so the substrate-carrying encoder branch contributes ~nothing beyond the prior under MTL while the SAME substrate lifts STL reg (+2.34 pp with prior, reproduced). This still hands the residual gap to the architecture — the encoder branch not learning region under joint training is a shared-backbone limitation — but the precise statement is "MTL reg ignores the substrate because the reg head is anchor-dominated **under this 50-epoch joint regime**," not "the substrate doesn't transfer" and not an absolute "the encoder is inert" (α=0 is an OOD ablation).

Combined with the earlier Phase-3 falsifications (long-tail sampler §4.6; substrate Tier-6), the residual reg gap is, **by elimination, the shared-backbone architecture itself**. `mtl_improvement` T2 (backbone alternatives) owns the fix. This study contributes the elimination evidence, not a mechanism.

**Orthogonal free upgrade available now:** Tier A1's log_T-KD (W=0.2) is a validated small-state reg lift that is independent of the backbone champion and stacks onto whatever `mtl_improvement` lands. It is **not** a competitor to the §4.2 composite headline (+7–12 pp); it is a single-MTL-artefact lift (~+2 to +5 pp small-states / ~+1.4 to +2.4 pp large-state pilot) with no deploy-time routing cost.

---

## Live docs touched on closure (2026-05-29)

- `docs/CHANGELOG.md` — closure entry.
- `docs/CLAIMS_AND_HYPOTHESES.md` — CH26 updated from "provisional, PROMOTED single-seed" to multi-seed n=20 PROMOTED (study-section, not paper whitelist — pending §0 re-run).
- `docs/CONCERNS.md` — C15 evidence + status line (NOT closed); C22/C24 consistency confirmed.
- `docs/findings/F_TIER_A1_PROMOTION.md` — new finding for the A1 promotion (cross-refs `F_TIER_A1_LEAK_AUDIT`).
- `docs/results/RESULTS_TABLE.md` — §0.8 sub-section for the log_T-KD lift (W=0.2 vs W=0.0), large-state pilot tagged with §0.1-dev-seed caveat.
- `docs/NORTH_STAR.md` — note that log_T-KD W=0.2 is a validated small-state reg lift orthogonal to B9 (champion recipe unchanged).

---

## DEFAULT FLIP + VERSIONING (2026-05-30, append-only — settles the study into code defaults)

The two validated findings are now the **code defaults**, with the prior paper-canonical pinned first in a new version registry: [`../../../results/CANONICAL_VERSIONS.md`](../../../results/CANONICAL_VERSIONS.md) (**v11** = BRACIS paper canon, FROZEN, no-KD/GCN; **v12** = new default = v11 + log_T-KD W=0.2 + ResLN encoder).

- **log_T-KD default → W=0.2/τ=1.0** (`scripts/train.py`), scoped to MTL `check2hgi_next_region` only; dataclass field stays 0.0, default applied at the CLI layer; loud log line on activation. v11 reproduction: `--log-t-kd-weight 0.0`.
- **Check2HGI encoder default → `resln`** (`research/embeddings/check2hgi/check2hgi.py` + `regen_emb_t3.py`); FUTURE builds only. The frozen v11 GCN substrate `output/check2hgi/<state>/` was NOT rebuilt/overwritten. v11 reproduction: `--encoder gcn` (or reuse the frozen GCN substrate).
- **Grade honoured:** log_T-KD paper-grade AL/AZ + single-seed pilot FL/CA/TX; ResLN **STL-only, no MTL benefit** (the regime finding).
- **Tests:** new `TestLogTKDCLIDefault` (5 cases); dataclass-default-zero test unchanged; no new failures (6 pre-existing failures are unrelated working-tree files).
- **Reproduction safety:** v11 paper §0.1 stays reproducible; substrate untouched; no GPU runs.
- New synthesis for future readers: [`../../../findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md`](../../../findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md).

### Evidence behind the v12 flip (2026-05-30 follow-up experiments, ~6-8 GPU-h)
The flip is backed by four follow-ups run after the 2026-05-29 closure (full numbers in the synthesis + per-doc sources; not re-tabulated here):
- **log_T-KD B9 confirmation** (`tier_b_fl/`, log.md): canonical+KD vs design_b+KD at FL — KD is **additive on B9** (+2.40 pp reg, p=0.031, 5/5) and the substrate null is **robust to KD** (design_b+KD ≈ canonical+KD, Δ−0.03 p=0.69). → justifies log_T-KD as the v12 reg default.
- **canonical_improvement coverage audit** (`canonical_improvement_coverage_audit.md`): the entire 18-experiment slate promoted exactly **two** items not in baseline — **ResLN encoder (+cat STL)** and WD (absorbed); all recipe-side knobs falsified/absorbed. → ResLN is the one missing improvement; baseline complete on the recipe axis.
- **ResLN matrix** (`tier_resln/phase_resln_verdict.md`): ResLN-canonical + ResLN+design_b across AL/AZ/FL, STL+MTL two-front. STL dual-axis: ResLN+design_b is the best all-around STL engine (equalises HGI reg at AL, keeps/widens cat); **MTL: ResLN's cat win does NOT survive on either axis** (refutes "cat encoder isn't starved") → confirms the regime finding extends to cat.
- **ResLN+design_j** (`tier_resln/phase_resln_verdict.md` §3b): the **AL specialist** — only variant to nominally exceed HGI reg at any state (AL 62.10), but AL-specific (worse than ResLN+design_b at AZ/FL and on cat everywhere); reproduces merge_design's "J is AL-specific". MTL flat as expected.
- **Note:** the GPU-cost tally above (~10-11 GPU-h) covers Tiers A-D only; the 2026-05-30 follow-ups (KD-confirm, ResLN matrix, design_j, FL HGI ceiling) add ~6-8 GPU-h.
- **Navigational docs refreshed (2026-05-30):** README, AGENT_CONTEXT, NORTH_STAR banners (study CLOSED + v12 flip); BRACIS `AUDIT_LOG.md` carries a v11-vs-v12 warning so paper authors don't regen tables with bare (v12) defaults. (`docs/index.html` research-state summary left stale — regenerated artifact, non-reproduction, deferred.)

Full record: [`log.md`](log.md) 2026-05-30 entries.

---

## Constraints honoured

- A1 small-state framed paper-grade (n=20); A1 large-state framed as seed=42 pilot; C1 framed as one-state footnote; B/C2/C3 framed as nulls/closures. No overclaim.
- C2/C3/B/A1 verdict docs NOT modified (verified, only synthesised + propagated).
- C15 NOT closed unilaterally — evidence + status line only.
- `mtl-protocol-fix` v6-final provenance untouched.
- **(2026-05-30) v11 paper-canon reproduction preserved at every step; `output/check2hgi/` not rebuilt; `mtl_improvement` files untouched.**
