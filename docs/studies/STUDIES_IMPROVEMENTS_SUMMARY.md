# Studies Improvements Summary — all `docs/studies/*` tracks

**Generated:** 2026-05-29 (extended with the 2026-05-30 v12 default-flip artefacts that were already on disk).
**Author:** audit pass (read-only synthesis; no numbers invented).

**Scope note.** This document enumerates every study under `docs/studies/*` and synthesises, per study, the concrete improvements proposed, their CONFIRMED / FALSIFIED / OPEN status, the exact outcome numbers vs the baseline each was compared against, the canonical recipe/code implication, and whether each was tested in isolation or combined. Special attention is paid to the active **`substrate-protocol-cleanup`** study (closed 2026-05-29, with a 2026-05-30 v12 default-flip appendix) and the cross-cutting scientific question — *does the Next-Reg STL→HGI gap close, and does MTL keep/improve Next-Reg and Next-Cat?*

> **Two joint tasks, two evaluation fronts.** Tasks: **next-cat** (next-category F1) and **next-reg** (next-region Acc@1/5/10). Fronts: **disjoint** = each head at its own best-val epoch (oracle upper bound, two checkpoints); **joint/geom_simple** = single deployable epoch maximising `sqrt(cat_f1 · reg_top10)`. "STL" = single-task matched-head ceiling; "MTL" = joint B9/H3-alt model.
>
> **Recipe vocabulary.** B9 = `mtlnet_crossattn + static_weight(cat=0.75) + next_gru(cat) + next_stan_flow/next_getnext_hard(reg) + per-head LR + cosine + alt-SGD + α-no-WD + per-fold log_T`, large states FL/CA/TX (`RESULTS_TABLE.md:64`). H3-alt = B9 minus alt-SGD/cosine/α-no-WD, `--scheduler constant`, small states AL/AZ. Reg head `next_getnext_hard`/`next_stan_flow` carries an additive `α·log_T` Markov-1 region prior with `alpha_init=0.1`.

---

## Master table — one row per concrete improvement

Δ values are in percentage points (pp). "Iso/Comb" = tested in isolation vs combined with other improvements. Citations in the per-study sections use `path:line`.

| # | Study | Improvement | Status | Baseline compared against | Task/metric (front) | Δ vs baseline (exact) | Canonical/code implication | Iso/Comb |
|---|---|---|---|---|---|---|---|---|
| 1 | substrate-protocol-cleanup (A1) | **log_T-KD** reg-loss term, W=0.2, τ=1.0 (KL of train-only log_T into reg logits) | **CONFIRMED** (paper-grade small states) | same recipe, W=0.0 (KD off) | next-reg Acc@10 (disjoint) | AL **+2.27**, AZ **+4.91** (n=20, 20/20 folds, Wilcoxon p=9.54e-07); cat AL −0.20 / AZ +0.08 (flat) | **v12 code default** ON for MTL `check2hgi_next_region` (`scripts/train.py`); v11 repro = `--log-t-kd-weight 0.0` | Isolated (single lever, paired W=0.0 vs W=0.2) |
| 2 | substrate-protocol-cleanup (A1 large) | log_T-KD W=0.2 at large states | **CONFIRMED-as-pilot** (seed=42, NOT paper-grade) | same recipe, W=0.0, seed=42 | next-reg Acc@10 (disjoint) | FL **+2.40** (5-fold, p=0.031, 5/5); CA **+1.42** (1-fold); TX **+1.71** (1-fold); cat flat (−0.10..+0.05) | Single-seed pilot; paper-grade needs {0,1,7,100} | Isolated |
| 3 | substrate-protocol-cleanup (A1 +B9 confirm) | log_T-KD additive on B9 + robust to substrate swap | **CONFIRMED** (2026-05-30 follow-up) | canonical+KD vs design_b+KD, FL | next-reg Acc@10 | KD additive on B9 **+2.40** (p=0.031, 5/5); design_b+KD ≈ canonical+KD (Δ−0.03, p=0.69) | Confirms KD as v12 reg default; substrate null robust to KD | Combined (KD × substrate) |
| 4 | substrate-protocol-cleanup (B1) | Design B (POI2Vec @ pool) as MTL substrate | **FALSIFIED** (AL) / **NULL** (AZ) | canonical c2hgi MTL, seed=42, 5f, H3-alt | next-reg (disjoint+joint), next-cat | reg disjoint AL −0.38 (p=0.91) / AZ +0.03 (p=0.44); joint AL +0.12 / AZ −0.49; cat −2.17/−2.41 | No promotion; cat drop = build-scope confound | Isolated substrate swap |
| 5 | substrate-protocol-cleanup (B2) | Design J (H + anchor λ=0.1) as MTL substrate | **FALSIFIED** (both) | same | next-reg, next-cat | reg disjoint AL −0.22 (p=0.78) / AZ −0.02 (p=0.69); joint AL −1.22; cat −2.05/−2.66 | No promotion | Isolated |
| 6 | substrate-protocol-cleanup (B4) | Lever 5 (KL distill / orphan rescue) substrate | **FALSIFIED** (AL) / **NULL** (AZ) | same | next-reg, next-cat | reg disjoint AL −0.28 (p=0.81) / AZ +0.01 (p=0.69); cat −2.49/−2.41 | No promotion | Isolated |
| 7 | substrate-protocol-cleanup (B3) | Lever 4 (POI2Vec @ p2r, additive) on canonical | **FALSIFIED** (both) | canonical c2hgi MTL, seed=42, 5f | next-reg, next-cat | reg disjoint AL −0.24 / AZ −0.08; cat −2.68/−2.54; all p≥0.78 | No promotion; B3b (winner+L4) skipped (no Wave-1 winner) | Combined (L4 on canonical) |
| 8 | substrate-protocol-cleanup (B FL 3-way) | Designs B/J/M as MTL substrate at FL | **FALSIFIED in MTL** (J CONFIRMED in STL only) | canonical c2hgi + HGI, FL seed=42 5f | STL & MTL next-reg | STL: J **+1.12** (p=0.031, 53% of HGI gap), B +0.71/M +0.89 (ns). MTL: all disjoint \|Δreg\|≤0.16 (ns) | STL design win does NOT survive MTL | Isolated; J vs canonical vs HGI |
| 9 | substrate-protocol-cleanup (HGI ceiling) | HGI as MTL substrate at FL (the missing control) | **FALSIFIED** (no MTL advantage) | canonical c2hgi MTL, FL seed=42 5f | next-reg disjoint; next-cat | reg **64.49 vs 63.98, Δ+0.51, p=0.41 NS**; cat collapses to 34.84 (−35.6) | HGI's STL reg win vanishes under MTL; "MTL flattens everyone" | Isolated |
| 10 | substrate-protocol-cleanup (C1) | Per-task 3-snapshot routing (variant A) | **§DISCUSSION FOOTNOTE** (one-state pass) | joint-best single checkpoint, seed=42 5f | next-reg Acc@10 routing | AZ **+2.54** (5/5, p=0.031) ✓; AL −7.89 (p=0.31, 1 degenerate fold; 4 healthy folds +2.14); FL +2.80 (5/5, p=0.031) ✓; cat routing near-null | NOT promoted; needs Acc@10 reg-best selector + degenerate-snapshot guard | Isolated; prototype shipped (`--save-task-best-snapshots`, `route_task_best.py`) |
| 11 | substrate-protocol-cleanup (C2) | Freeze-reg-after-peak curriculum (N∈{2,4,6}) | **FALSIFIED** — closes §4.4 | canonical baseline MTL, seed=42 5f | next-cat F1; next-reg | best cat AL N=2/4 +0.37/+0.46 (p=0.16) at reg −7.69/−4.18; reg-safe N=6 → cat null (≤+0.06) | No N gives cat gain without reg cost > σ_fold; archived | Isolated |
| 12 | substrate-protocol-cleanup (C3) | Zero cat→reg cross-attention K/V (P4 residual test) | **FALSIFIED hypothesis** — P4 fully closed | canonical baseline MTL, seed=42 5f | next-reg dynamics; next-cat | AL reg Δ−0.28 (ns), peak ep 12.8→9.4 (earlier); AZ Δ+0.01 (ns), peak 6.2→6.6; cat AL +0.37/AZ −0.04 (ns) | No finding filed; residual gap NOT K/V capacity-stealing → architectural | Isolated |
| 13 | substrate-protocol-cleanup (C4) | C22 stale-log_T mtime preflight guard | **LANDED** (defensive) | n/a (code-only) | n/a | n/a | `src/training/runners/mtl_cv.py` mtime guard | Code-only |
| 14 | substrate-protocol-cleanup (D1) | Window/causal-mask leak audit | **CLEAN** (no leak) | n/a (code read) | n/a | 5/5 scope items clean; C19 guard confirmed | Shared with `mtl_improvement` T0.2; no code change | Code audit |
| 15 | substrate-protocol-cleanup (ResLN) | ResidualLN encoder (T3.2) × substrate, STL+MTL | **CONFIRMED STL / FALSIFIED MTL** | canonical (no-ResLN), designs, HGI | STL next-cat & next-reg; MTL both | STL cat ResLN+design_b widens lead (AL +2.53, AZ +1.31); STL reg ties HGI at AL (61.99 vs 61.86). MTL: all Δ NS | **v12 encoder default `resln`** for FUTURE builds; STL-only, NO MTL benefit | Combined (ResLN × design_b / design_j) |
| 16 | canonical_improvement (T3.2) | ResidualLN encoder (origin) | **PROMOTED** (paper-grade STL cat) | canonical GCN encoder, 5 seeds | STL next-cat F1 | cat **+0.86 FL / +1.48 AL / +1.70 AZ** (5/5 seeds, p=0.03125); reg ≈0 small / +0.71 FL | opt-in `--encoder resln`; defaulted in v12 | Isolated |
| 17 | canonical_improvement (T1.5 v3c) | AdamW weight-decay 5e-2 on embedding trainer | **PROMOTED-provisional** (absorbed) | plain Adam embedding trainer, n=5 | STL cat/reg (substrate) | AL cat +0.38, reg +0.09 (within σ); FL reg +0.63 standalone, **absorbed by T3.2** | opt-in `--weight-decay 0.05`; statistically equivalent to dropping it in stack | Combined (stack with ResLN) |
| 18 | canonical_improvement (T1–T6, 16 other exps) | α-sweep, hard-neg rate, DropEdge, GATv2, R-GCN, Time2Vec, GraphMAE, POI side-feats, Delaunay edges, native POI-ID, Node2Vec, multi-view, log_T-KD-λ, α/w_r grid, low-rank side-channel, substrate variants | **FALSIFIED / §Discussion-only / SKIPPED** | STL matched-head ceiling, n=5 (Table 2) | STL cat/reg | all \|Δreg\|≤~0.3 pp or cat-negative; T3.1 GATv2 = catastrophic structural leak (cat→99%) | None promoted; **baseline complete on recipe axis** | Various |
| 19 | merge_design (Designs B/H/I/J/M) | POI2Vec @ pool-boundary substrate merge | **CONFIRMED STL (Pareto over canonical)** / does NOT beat HGI | canonical c2hgi + HGI, STL, leak-free | STL next-reg Acc@10; next-cat | AL/AZ ~+2 pp reg over canonical (J/M strict p=0.031); FL H/J +1.1–1.2 pp (p=0.031), still 0.9–1.0 below HGI; cat non-inferior (TOST p<0.01) | STL substrate variants; never adopted into ship recipe | Isolated per design |
| 20 | merge_design (Design A late-fusion) | Concat at head | **FALSIFIED** | canonical | both | cat −9, reg −10 pp at AL/AZ | Dead | Isolated |
| 21 | merge_design (Design D heterograph) | Reverse visit-edges + 2-hop GCN | **FALSIFIED (label leak)** | canonical | next-cat probe | +20 pp last-step linear-probe leak | Disqualified despite dominance | Isolated |
| 22 | merge_design (Design K, Levers 4/5/6) | Delaunay POI-POI GCN; p2r POI2Vec; KL distill; POI↔POI contrastive boundary | **FALSIFIED / saturated** | Design J | next-reg | K = J (AL Δ−0.02, AZ −0.06); L6 ≈ J; residual ~1 pp to HGI not closed | Structural axis closed | Combined (on J) |
| 23 | mtl-protocol-fix (Phase 3 Rank 4 composite) | Deploy STL c2hgi-cat + STL HGI-reg routed by task | **CONFIRMED (project reg headline)** | MTL shipping @ disjoint reg | next-reg Acc@10 (deploy) | **+11.04 AL / +12.04 AZ / +7.43 FL (fresh) / +7.16 CA / +9.64 TX** at zero cat cost | Two-model deploy; the deployable reg ceiling | Combined (two-substrate routing) |
| 24 | mtl-protocol-fix (F1 selector fix) | `joint_geom_simple` deployable selector vs `joint_canonical_b9` | **CONFIRMED** | legacy joint_canonical_b9 selector | next-reg (deployable joint) | +5.6 pp FL multi-seed recovered; C21 RESOLVED | Default deployable selector | Isolated (selector swap) |
| 25 | mtl-protocol-fix (P4 frozen-cat) | Freeze cat encoder params → does MTL reg recover? | **FALSIFIED hypothesis** | MTL baseline | next-reg | reg does NOT recover; gap is architectural not cat-interference | Drives `mtl_improvement` arch axis | Isolated |
| 26 | mtl-protocol-fix (§4.6 balanced sampler) | Class-balanced sampler at reg head | **FALSIFIED** | MTL baseline | next-reg | −18 to −30 pp regression | Dead | Isolated |
| 27 | hgi_category_injection | Inject category into HGI POI2Vec (6 variants) | **FALSIFIED** (AZ, closed) | HGI baseline, AZ seed=42 30ep | next-cat F1, next-reg | best (D_orth) cat +0.36 / reg +0.10, no Wilcoxon at n=5 | Closed; do not re-open without commit | Isolated per variant |
| 28 | mtl_improvement (T0–T8) | Backbones, loss, batch, LR, α, heads, multi-seed champion | **OPEN / DESIGN-ONLY** (no experiment results on file) | TBD | next-cat & next-reg MTL | none yet (T0.2 mask audit closed-by-handoff to D1) | Owns the architectural-residual-gap fix | n/a |
| 29 | mtl-exploration | Pre-study scoping (no-encoders, HGI-substrate, leak-blast-radius) | **SCOPING** (fed into mtl_improvement) | n/a | n/a | cat needs thick MLP at AL (−2.57 pp if simplified); reg ok with Linear+LN | Locked HGI substrate out; informed mtl_improvement design | n/a |

---

## Per-study detail

### 1. `substrate-protocol-cleanup/` — ACTIVE → CLOSED 2026-05-29 (the focus study)

Five Tiers (A–D) of cheap, non-architectural cleanup carved out of `mtl-protocol-fix/DEFERRED_WORK.md`. The study did not need a champion; its purpose was clean verdicts orthogonal to the architectural revisit in `mtl_improvement`. Per-tier verdict table: `docs/studies/substrate-protocol-cleanup/CLOSURE.md:25-39`.

#### Tier A1 — log_T-KD (the single PROMOTION)

- **Mechanism** (`F_TIER_A1_PROMOTION.md:11`): `task_b_loss = CE + W·τ²·KL(softmax(reg_logits/τ) ‖ softmax(log_T[last_region_idx]/τ))`, W=0.2, τ=1.0. Teacher = train-only per-fold first-order region-transition log-prior, indexed by observed last region (`poi_0..8`, never `target_poi`). A *second* pressure on top of the head's additive `α·log_T` prior.
- **Small-state paper-grade (n=20, seeds {0,1,7,100}, AL/AZ, 5f, H3-alt)** — `tier_a1/phase_a1_verdict.md:24-27`:
  - AL disjoint reg: W0.0 **50.59 ± 3.53** → W0.2 **52.85 ± 3.48**, mean Δ **+2.27**, median +2.15, 20/20 folds positive, Wilcoxon one-sided p=**9.537e-07**.
  - AZ disjoint reg: W0.0 **41.30 ± 2.60** → W0.2 **46.22 ± 2.75**, mean Δ **+4.91**, 20/20 folds, p=**9.537e-07**.
  - geom_simple reg: AL 48.00→51.21; AZ 38.79→44.05.
  - cat F1 disjoint: AL 45.96→45.76 (−0.20); AZ 48.86→48.94 (+0.08) — **flat** (`tier_a1/phase_a1_verdict.md:12-20`).
  - **scipy dispatch caveat** (`tier_a1/phase_a1_verdict.md:29`): raw-value exact p=9.537e-07; 2-dp rounded → approx p≈4.42e-05. Always test on raw CSV.
- **Large-state pilot (seed=42, NOT paper-grade)** — `tier_a1_largestate/phase_a1_largestate_addendum.md:13-17`:
  - FL (5-fold): 63.98 ± 0.76 → 66.38 ± 0.58, Δ **+2.40**, p=0.03125, 5/5 folds; per-fold Δ +2.34,+2.63,+2.09,+2.28,+2.65.
  - CA (1-fold): 50.06 → 51.48, Δ +1.42. TX (1-fold): 50.38 → 52.09, Δ +1.71. Cat flat at all three (Δ∈[−0.10,+0.05]).
  - FL gap-closure: closes ~33% of the FL MTL→STL reg disjoint gap (gap 70.62−63.98=6.64; recovers 2.40), ~45% on geom_simple (61.14→65.20) — `tier_a1_largestate/...addendum.md:34`.
- **Leak audit (independent, 7-vector, NO LEAK)** — `F_TIER_A1_LEAK_AUDIT.md`: train-fold-only log_T (V1), `last_region_idx` from `poi_0..8` only (V2), W=0.0 byte-identical fast path (V4), reg-only lift + flat cat (V6) — the opposite of every historical leak signature. MI/H(target) ≈ 0.601 (AL) / 0.560 (AZ), top-1 determinism ~0.34–0.37 (informative but NOT near-deterministic). AZ's larger lift = dosage-on-headroom (AZ baseline 9 pp below AL), not stronger shortcut (`F_TIER_A1_LEAK_AUDIT.md:32,62-66`).
- **B9 confirmation (2026-05-30)**: KD additive on B9 at FL (+2.40 reg, p=0.031, 5/5); design_b+KD ≈ canonical+KD (Δ−0.03, p=0.69) — substrate null robust to KD (`CLOSURE.md:135`).
- **Framing caveat**: NOT a competitor to the §4.2 composite (+7–12 pp); it is the best *single-MTL-artefact* reg lift, no deploy-time routing (`F_TIER_A1_PROMOTION.md:50-51`).
- **Cross-check**: FL STL α=0 canonical JSON `region_head_florida_region_5f_50ep_stl_a0_canonical_florida.json` confirms top10_acc_mean = 0.72743 (72.74%), matching the isolation-cell number.

#### Tier B — substrate axis under MTL+F1 (all NULL/FALSIFIED)

B-summary table `INDEX.md:88-95`; two-front re-analysis `tier_b/phase_b_two_front.md:26-34`. Baseline = canonical c2hgi MTL, seed=42, 5f, H3-alt; canonical reference: AL disjoint reg 50.82 / AZ 41.33.

| Variant | reg disjoint Δ (AL/AZ) | reg joint Δ (AL/AZ) | cat Δ (AL/AZ) | Wilcoxon p (AL/AZ) | Verdict |
|---|---|---|---|---|---|
| Design B (B1) | −0.38 / +0.03 | +0.12 / −0.49 | −2.17 / −2.41 | 0.91 / 0.44 | FALSIFIED AL / NULL AZ |
| Design J (B2) | −0.22 / −0.02 | −1.22 / −0.17 | −2.05 / −2.66 | 0.78 / 0.69 | FALSIFIED both |
| Lever 5 (B4) | −0.28 / +0.01 | −0.72 / +0.63 | −2.49 / −2.41 | 0.81 / 0.69 | FALSIFIED AL / NULL AZ |
| canonical+Lever 4 (B3) | −0.24 / −0.08 | — | −2.68 / −2.54 | 0.78 / 0.84 | FALSIFIED both |

- **Both fronts agree** — no design is null-on-disjoint-but-positive-on-joint; all reg \|Δ\|≤1.22, every p≥0.21 (`tier_b/phase_b_two_front.md:54-58`).
- **Re-audit mechanism** (`tier_b/phase_b_reaudit.md`, `CLOSURE.md:15`, hedged): at α=0 (log_T anchor frozen off, OOD) reg collapses to near-floor (~5% AL top10 = chance for 1109 regions) for BOTH design and canonical (`tier_b/phase_b_two_front.md:50-52`). So MTL reg is dominated by the α·log_T anchor; the substrate-carrying encoder branch contributes ~nothing beyond the prior under MTL. The substrate's STL reg advantage is real and reproduces (AL +2.34 pp with prior, p=0.0312). The cat −2.4 pp is a **build-scope CheckinEncoder-reinit confound** (at α=0 cat Δ is +0.19), NOT a substrate cost.
- **Corrected wording** (hedged per verification): "no reg gain beyond the canonical log_T anchor under the joint config," NOT "the substrate fails to transfer," and NOT an absolute "encoder is inert at chance."

#### Tier B FL three-way (canonical vs designs B/J/M/L vs HGI) — `tier_b_fl/phase_b_fl_3way.md`, `tier_b_fl/hgi_mtl_fl.md`

- **STL three-way (gethard, with prior)** `phase_b_fl_3way.md:18-24`: canonical 69.22; B 69.93 (+0.71, p=0.0625 ns); **J 70.34 (+1.12, p=0.0312 ✓, 53% of HGI gap)**; M 70.11 (+0.89, ns); **HGI 71.34 (+2.12)**. All designs strictly below HGI (HGI>design p=0.0312). No-prior cross-check: J +0.86 (p=0.0312), HGI−J +1.64.
- **MTL three-way (B9, disjoint)** `phase_b_fl_3way.md:46-52`: canonical 63.98; B 63.82 (−0.16, p=0.875); J 64.06 (+0.08, p=0.312); L 63.97 (−0.01); **HGI 64.49 ± 0.55 (+0.51, p=0.41 NS)**. Designs close **0%** of any MTL gap; joint front −3.5 pp is a cat-driven geom-selection artefact.
- **HGI ceiling** (`hgi_mtl_fl.md`, `CLOSURE.md:62`): HGI's STL reg win (+2.12) **vanishes** under MTL (70.9→64.5 ≈ canonical); HGI cat collapses to 34.84% (−35.6). "MTL flattens everyone." ~0.36 GPU-h.
- **Isolation cell** (the clean experiment) `phase_b_fl_3way.md:68-85`: STL `next_stan_flow` α=0 LEARNS FL region at **72.74%** (canonical) / **73.12%** (design_b, Δ+0.37 p=0.0312); the IDENTICAL head/config under MTL floors at **~0.03%** (chance 0.213% for 4703 regions). **Verdict: the joint-training REGIME, not the head/substrate, kills the MTL reg encoder.** Hedge: α=0 is OOD → regime-and-config-scoped (B9, 50 ep).

#### Tier C — protocol coherence — `tier_c/phase_c_verdict.md`, `tier_c_fl/phase_c_fl_verdict.md`

- **C1 (3-snapshot routing, variant A) — §DISCUSSION FOOTNOTE, one-state pass.** Re-scored 2026-05-29 after a reg-modality scoring bug (`route_task_best.py` rebuilt val loaders on `task_b=checkin` while the run trained `region`; `ExperimentConfig` did not persist modality). Fixed (persist `task_*_input_type`; 5 new unit tests). Corrected (`phase_c_verdict.md:105-118`):
  - AZ: reg-best vs joint-best **+2.54** (5/5 folds, Wilcoxon p=0.03125) ✓.
  - AL: **−7.89** (p=0.31) — driven by ONE genuine degenerate fold3 reg-best snapshot (Acc@10=0.12% even on correct modality, saved at val reg Acc@1=0.2801 @ ep14; same fold joint-best = 48%). A real `MultiTaskBestTracker` Acc@1-selector pathology, NOT the modality bug. 4 healthy AL folds avg +2.14.
  - FL (added): reg routing **+2.80** (5/5, p=0.0312) ✓, cat +1.98; no degenerate fold (`CLOSURE.md:66`). FL is the third state (AZ pass, AL fail, FL pass).
  - Cat-best routing near-null (AL +0.87 p=0.06 / AZ +0.12 ns).
  - **Conditional follow-up**: swap reg-best selector to Acc@10 + add degenerate-snapshot guard, then multi-seed before any §0.x promotion. Deploy cost: 3× storage + 2-model load.
- **C2 (freeze-reg-after-peak N∈{2,4,6}) — ARCHIVE, closes §4.4** (`phase_c_verdict.md:24-54`). No N gives cat gain without reg cost > σ_fold. Where cat lifts most (AL N=2/4 +0.37/+0.46, p=0.156, never sig) reg collapses −7.69/−4.18 (σ_fold 3.21); where reg preserved (N=6 AL −1.05 / AZ −0.07) cat null (≤+0.06). FL: C2 holds, N=2 hurts reg −7.69 p=0.0312 (`CLOSURE.md:65`).
- **C3 (zero cat→reg K/V) — P4 FULLY CLOSED** (`phase_c_verdict.md:58-84`). AL reg Δ−0.28 (ns), peak ep 12.8→9.4 (EARLIER, opposite of hypothesis); AZ Δ+0.01 (ns), peak 6.2→6.6. Cat AL +0.37/AZ −0.04 (ns). FL: zero-cat-kv shifts reg +0.03 ns. Exonerates the cat-activation pathway P4 left open → residual gap is architectural.
- **C4 (mtime guard) — LANDED** in `src/training/runners/mtl_cv.py` (`CLOSURE.md:19,38`).

#### Tier D1 — window/mask audit — `window_mask_audit.md` — **CLEAN**

All 5 scope items clean with `file:line` citations: `generate_sequences` target never in window (`core.py:59-89`); `NextHeadMTL` causal mask blocks j>i (`next_mtl/head.py:42-47`); `last_region_idx` from observed last POI (`next_region.py:124-163`); per-fold log_T train-only per seed+fold + C19 n_splits guard still holding (`mtl_cv.py:858-954`); modality discipline `task_a=checkin × task_b=region` clean. Shared artefact with `mtl_improvement` T0.2.

#### ResLN matrix (2026-05-30 follow-up) — `tier_resln/phase_resln_verdict.md`

- **STL next-reg Acc@10** (`phase_resln_verdict.md:13-17`): ResLN+design_b is best non-HGI at all 3 states — AL **61.99** (ties HGI 61.86, beat-HGI p=0.31), AZ 52.98 (HGI 53.37, closes ~80% of canonical's −2.04 gap), FL 70.21 (HGI 71.34, closes ~30%).
- **STL next-cat F1** (`phase_resln_verdict.md:28-33`): check2hgi stays decisively best (48–70% vs HGI ~20–25%); ResLN+design_b widens lead AL +2.53 / AZ +1.31; flat FL.
- **ResLN+design_j** (`phase_resln_verdict.md:39-55`): the AL specialist — AL reg 62.10 (only variant to nominally exceed HGI at any state) but worse than ResLN+design_b at AZ/FL and on cat everywhere. Reproduces merge_design's "J is AL-specific."
- **MTL: none transfers** (`phase_resln_verdict.md:57-64`): all 9 MTL cells Δ NS on both axes (cat AL −0.01/+0.67, AZ −0.44/−0.12, FL −0.34/−0.06; reg \|Δ\|≤0.67). ResLN's STL cat win does NOT survive MTL → "cat encoder isn't starved" hypothesis **refuted**; regime finding extends to cat.

#### canonical_improvement coverage audit — `canonical_improvement_coverage_audit.md`

The 18-experiment canonical_improvement slate promoted exactly **two** items — v3c (AdamW WD=5e-2) and T3.2 ResLN — both **substrate/encoder-side, not recipe-side**, both opt-in (engine still defaults `gcn`+plain-Adam). **Baseline is COMPLETE on the recipe axis** (no promoted recipe-side flag missing; T1.3 α-sweep, T1.5-MTL, T6.2 all falsified or substrate-scoped). ResLN's cat lift is immaterial to the MTL **reg** verdict by the regime finding.

#### v12 default flip (2026-05-30) — `CLOSURE.md:122-142`, `CANONICAL_VERSIONS.md`

- **v11** = BRACIS paper canon, FROZEN, GCN substrate + log_T-KD OFF (`RESULTS_TABLE.md §0.1`, commit `99f56e8`).
- **v12** = new default = v11 + log_T-KD W=0.2/τ=1.0 (CLI-layer default, scoped to MTL `check2hgi_next_region`) + ResLN encoder (future builds only; frozen GCN substrate NOT rebuilt). v11 repro: `--log-t-kd-weight 0.0` (+ `--encoder gcn` / frozen substrate). New tests `TestLogTKDCLIDefault` (5 cases).

---

### 2. `canonical_improvement/` — CLOSED 2026-05-19

18-experiment, 5-tier slate to improve canonical Check2HGI on the **STL** axis (cf. `README.md:9`, INDEX.html). Compared against the STL matched-head ceiling (Table 2, seed=42 n=5). Verdict-and-classification table: `canonical_improvement_coverage_audit.md:32-59`.

- **Promoted (2):** T3.2 ResidualLN encoder (STL cat **+0.86 FL / +1.48 AL / +1.70 AZ**, 5/5 seeds, p=0.03125; reg ≈0 small / +0.71 FL; leak +2.24 IJM-honest), and v3c AdamW WD=5e-2 (FL reg +0.63 standalone, absorbed by T3.2). Both substrate/encoder-side, opt-in.
- **Falsified/skipped (16):** T1.3 α-ratio sweep (\|Δreg\|≤0.30); T2.1 p2r hard-neg rate (reg flat 0.09–0.11); T2.4 DropEdge (doesn't stack); **T3.1 GATv2 (catastrophic structural leak, cat→99%)**; T3.3 R-GCN (K=2 leak +27.85, K=1 collapse); T3.4 Time2Vec (loses cat −0.56); T4.1 GraphMAE; T4.3 POI side-feats (AL cat +0.63 doesn't replicate at FL); T4.4 Delaunay edges (cat −11.30 AL); T5.1 native POI-ID (reg −6.37); T5.2a/b, T5.3 (§Discussion-only, sub-Bonferroni); T6.1 log_T-KD-λ (joint_geom_simple gate failed every cell — note: this is the *earlier* gate failure that Tier A1 later salvaged on the disjoint front); T6.2 α/w_r grid (reg +0.76 at cat −3.55, Pareto); T6.3 low-rank side-channel; T6.4 substrate variants ("+11 pp" was a cross-selector artefact).
- **Verdict:** substrate axis exhausted at ±0.8 pp ceiling (`README.md:9`).

---

### 3. `merge_design/` — ACTIVE-CLOSING (structural axis closed 2026-05-07)

Goal: merge HGI POI semantics into Check2HGI to close the **STL** canonical→HGI next-region gap while preserving next-cat (`STATE.md:8-16`). All numbers leak-free, STL.

- **Settled positive** (`STATE.md:36-49,51-64`): B/H/I/J/M all dominate canonical on both axes at AL/AZ (cat non-inferior TOST p<0.01, reg superior ~+2 pp; J nominally beats HGI on AL by +0.10). At FL only H and J Wilcoxon-strict over canonical (+1.1–1.2 pp, p=0.0312), still 0.9–1.0 pp below HGI. fclass probe 4%→98% (POI-semantic recovery, generality not a next-task gain). Headline table `STATE.md:20-29`: canonical AL cat 40.76 / reg 59.15; HGI AL cat 25.26 / reg 61.86; J AL cat 41.81 / reg 61.95.
- **Settled negative** (`STATE.md:65-82,149-166`): Design A late-fusion (cat −9/reg −10); Design D heterograph (+20 pp label leak); Design K Delaunay GCN = J (Δ−0.02/−0.06); λ-anchor inactive (warm-start ‖E−POI2Vec‖²≈0); Lever 6 ≈ J (falsified 2026-05-07); Test 1 next-POI J closes 77% of gap but does NOT overcome HGI; Test 2 no-log_T gap *widens* to 1.64 pp; Test 2½ seed reroll Δ+0.10; Test 3 POI2Region heads — adding hurts.
- **Open residual** (`STATE.md:84-101`): the ~1 pp AZ/FL gap to HGI is below the study's architectural resolution; not features, not structure → in the training recipe (HGI's hierarchical fclass L2 regulariser + POI↔POI contrastive boundary). **All structural research questions closed.**
- **Canonical implication:** none adopted into the ship recipe; the merge family is an STL Pareto improvement over canonical, not a deployable MTL gain (confirmed inert in MTL by substrate-protocol-cleanup Tier B).

---

### 4. `mtl-protocol-fix/` — CLOSED 2026-05-24 (v6 final), the predecessor study

EV-ranked execution of `DEFERRED_WORK.md` (`PRIORITY_IMPACT.md:18-30`). Headline outcomes:

- **§4.2 composite (Rank 4) — the project reg headline** (`phase3_rank4_composite_analysis.md`): deploy STL HGI reg + MTL c2hgi cat routed by task → vs MTL@disjoint reg: **AL +11.04, AZ +12.04, FL +7.43 (fresh log_T), CA +7.16, TX +9.64** at zero cat cost. Composite reg = STL HGI reg (AL 61.86, AZ 53.37, FL 71.34, CA 57.77, TX 60.47). Two-model deploy footprint. This is the *strongest* reg lift in the project — larger than and distinct from Tier A1's single-model log_T-KD.
- **§4.5 log_T-KD (Rank 1) — PROMOTED single-seed** (+2.40/+5.06/+2.32 at AL/AZ/FL @ W=0.2, p=0.0312 all 9 cells; `RESULTS_TABLE.md:13`) → handed to substrate-protocol-cleanup Tier A1 for multi-seed (confirmed).
- **F1 selector fix:** `joint_geom_simple` recovers +5.6 pp FL multi-seed vs `joint_canonical_b9`; C21 RESOLVED (`RESULTS_TABLE.md:11`).
- **P4 frozen-cat:** residual MTL-vs-STL reg gap is **architectural** (not cat-interference, not long-tail, not substrate) (`RESULTS_TABLE.md:15`).
- **§4.6 balanced sampler (Rank 2) — FALSIFIED** (−18 to −30 pp).

---

### 5. `mtl_improvement/` — ACTIVE (branch `mtl-improve`) — DESIGN-ONLY

T0–T8 chain (backbones, loss, batch, LR, α, heads, multi-seed champion). Owns the **architectural** fix for the residual MTL-vs-STL reg gap that every other study eliminated as non-architectural. **No experiment results on file** (`docs/results/mtl_improvement` does not exist); T0.2 mask audit closed-by-handoff to substrate-protocol-cleanup D1 (`mtl_improvement/log.md` tail). Design notes: cat needs the thick MLP at AL scale (−2.57 pp paper-grade loss if simplified); reg fine with Linear+LN. Full chain ~1250–1700 GPU-h estimate. **This is the live open frontier for next-reg MTL improvement.**

---

### 6. `mtl-exploration/` — pre-study scoping

Fed into `mtl_improvement` design. Files: EXPERIMENT_NO_ENCODERS, EXPERIMENT_HGI_SUBSTRATE (locked HGI out as substrate), LEAK_BLAST_RADIUS_AUDIT, FUTUREWORK_substrate_aware_mtl_balancing. No standalone numeric verdicts to promote.

---

### 7. `hgi_category_injection/` — CLOSED (AZ falsified 2026-05-04)

6 variants on Arizona (seed=42, 30ep) injecting category into HGI's POI2Vec. None lifts next-cat F1 or next-reg Acc@10 above noise. Best (D_orth, orthogonal projection): cat +0.36 / reg +0.10, no Wilcoxon at n=5 (best p=0.0625 floor vs strict gate 0.0312) (`STATUS.md` table). Confirms "HGI POI2Vec ≈ fclass lookup" (dropping fclass→category collapses fclass probe 71%→13%). Kept pending FL/CA/TX re-open decision; do NOT treat as active without explicit re-open commit.

---

### 8. `fusion/` — minimal/archived content

`docs/studies/fusion/` contains only `results/P0/folds/frozen.json` (a frozen-fold artefact). The fusion *study* is archived under `docs/archive/fusion-study/`; the FUSION *engine* remains first-class in code (per CLAUDE.md). No new improvement verdicts here.

---

## Next-Reg STL gap vs HGI benchmark — every relevant number

The scientific goal: close the **Next-Reg STL canonical→HGI gap**, while in MTL keeping/improving Next-Reg and Next-Cat. Below, every number bearing on whether the gap closes.

### A. STL: canonical c2hgi vs HGI (the gap to close)

| State | C2HGI STL reg Acc@10 | HGI STL reg Acc@10 | Gap (HGI−C2HGI) | Source |
|---|---:|---:|---:|---|
| AL | 59.15 / 61.02 | 61.86 | +2.71 / +0.84 | merge_design `STATE.md:23`; ResLN `phase_resln_verdict.md:14` |
| AZ | 50.24 / 51.33 | 53.37 | +3.13 / +2.04 | `STATE.md:23`; `phase_resln_verdict.md:15` |
| FL | 69.22 (gethard) / 69.68 | 71.34 | +2.12 / +1.66 | `phase_b_fl_3way.md:20-24`; `phase_resln_verdict.md:16` |

(Two C2HGI columns reflect two protocols/recipes across docs; both are cited. `RESULTS_TABLE.md §0.3` reports HGI nominally above C2HGI by 1.6–3.1 pp under TOST δ=2pp at AL/AZ/FL, tied at CA/TX — `RESULTS_TABLE.md:134`.)

### B. STL: how much of the gap each improvement closes

| Improvement | State | reg Acc@10 | Δ vs canonical | gap-closure vs HGI | strict? | Source |
|---|---|---:|---:|---:|---|---|
| Design J (with prior) | FL | 70.34 | +1.12 | **53%** | p=0.0312 ✓ | `phase_b_fl_3way.md:22` |
| Design B | FL | 69.93 | +0.71 | 34% | p=0.0625 ns | `phase_b_fl_3way.md:21` |
| Design M | FL | 70.11 | +0.89 | 42% | p=0.0625 ns | `phase_b_fl_3way.md:23` |
| Design J (no-prior) | FL | 69.22 | +0.86 | — (HGI−J +1.64) | p=0.0312 ✓ | `phase_b_fl_3way.md:34` |
| ResLN+design_b | AL | 61.99 | +0.97 | **EQUALISES HGI** (beat p=0.31) | tie | `phase_resln_verdict.md:14` |
| ResLN+design_b | AZ | 52.98 | +1.65 | ~80% (HGI still nominal lead) | — | `phase_resln_verdict.md:15` |
| ResLN+design_b | FL | 70.21 | +0.53 | ~30% | — | `phase_resln_verdict.md:16` |
| ResLN+design_j | AL | 62.10 | +1.08 | **>HGI nominally** (61.86; p=0.50) | tie | `phase_resln_verdict.md:46` |

**STL verdict:** the gap is **partially closed** at FL (best: J, 53% with prior) and **equalised at AL** (ResLN+design_b ties HGI; ResLN+design_j nominally exceeds it). At AZ/FL HGI retains a ~0.4–1.1 pp nominal lead. The residual is real embedding-quality (no-prior cross-check preserves J's edge and *widens* HGI−J to +1.64). **No variant significantly beats HGI on STL reg anywhere.**

### C. MTL: does any substrate (incl. HGI) carry the STL reg advantage? — NO

| Substrate | FL MTL reg disjoint | Δ vs canonical (63.98) | p | Source |
|---|---:|---:|---:|---|
| canonical c2hgi | 63.98 | — | — | `phase_b_fl_3way.md:48` |
| Design B | 63.82 | −0.16 | 0.875 | `phase_b_fl_3way.md:49` |
| Design J | 64.06 | +0.08 | 0.312 | `phase_b_fl_3way.md:50` |
| Lever 5 (L) | 63.97 | −0.01 | 0.500 | `phase_b_fl_3way.md:51` |
| **HGI (the ceiling)** | **64.49 ± 0.55** | **+0.51** | **0.41 NS** | `phase_b_fl_3way.md:52`, `hgi_mtl_fl.md` |

**MTL verdict: the gap does NOT close in MTL because there is NO MTL gap to close.** HGI's STL reg win (+2.12) vanishes under B9 joint training (70.9→64.5 ≈ canonical ≈ designs). The isolation cell proves it is the **regime**: identical α=0 reg head learns FL region at ~73% STL but floors at ~0.03% MTL.

### D. The ONE lever that moves MTL reg — log_T-KD (the prior pathway, not the encoder)

| State | grade | W0.0 disjoint reg | W0.2 disjoint reg | Δ | gap-closure to STL ceiling | Source |
|---|---|---:|---:|---:|---:|---|
| AL | paper (n=20) | 50.59 | 52.85 | **+2.27** | (STL ceiling 61.21) | `phase_a1_verdict.md:12` |
| AZ | paper (n=20) | 41.30 | 46.22 | **+4.91** | (STL ceiling 53.06) | `phase_a1_verdict.md:18` |
| FL | pilot (seed=42) | 63.98 | 66.38 | **+2.40** | ~33% of MTL→STL disjoint gap (45% geom) | `phase_a1_largestate/...addendum.md:15,34` |

### E. The deployable reg ceiling (mtl-protocol-fix composite — strongest of all)

STL HGI reg routed at deploy beats MTL@disjoint by **+7 to +12 pp at every state** (AL +11.04 / AZ +12.04 / FL +7.43 / CA +7.16 / TX +9.64), zero cat cost, two-model deploy (`phase3_rank4_composite_analysis.md`).

### F. The §0.1 v11 architectural gap (the problem statement, paper-canon, n=20)

`RESULTS_TABLE.md:72-76`: MTL B9 reg Acc@10 vs STL `next_stan_flow` ceiling: AL 50.17 vs 61.21 (**−11.04**); AZ 40.78 vs 53.06 (**−12.27**); FL 63.27 vs 70.62 (**−7.34**); CA 47.35 vs 56.85 (**−9.50**); TX 42.84 vs 59.44 (**−16.59**); all p≤2e-06. Cat side MTL ≥ STL at 4/5 states (AZ/CA/TX/FL +1.20/+1.68/+1.89/+1.40; AL −0.78 small-sig). **This −7 to −17 pp residual is the gap `mtl_improvement` owns; every non-architectural cause has been eliminated.**

---

## Open questions / not-yet-combined / not-yet-verified (for the downstream verification agent)

1. **log_T-KD large states are seed=42 PILOTS only.** FL +2.40 (n=5), CA +1.42 / TX +1.71 (n=1). Paper-grade requires {0,1,7,100} per C23. The W=0.0 baselines overshoot §0.1 multi-seed (FL +0.7, CA +2.7, TX +7.5 pp dev-seed bias). **NOT yet paper-grade at FL/CA/TX.** (`phase_a1_largestate/...addendum.md:7`)

2. **log_T-KD is now the v12 code default but NOT yet re-run into §0 paper tables.** CH26 updated to multi-seed-PROMOTED "study-section, not paper whitelist — pending §0 re-run" (`CLOSURE.md:114`). §0.1 stays v11 (KD off). The §0.8 rows isolate the lift but a full §0 v12 re-table is pending.

3. **log_T-KD × architectural champion NOT yet combined.** A1 is framed as a "free upgrade that stacks onto whatever `mtl_improvement` lands," but `mtl_improvement` has produced no champion yet — the stack is untested. Also untested: log_T-KD at AL/AZ multi-seed *combined with* the §4.2 composite deploy.

4. **C1 routing is one-state-pass and selector-brittle.** AZ +2.54 and FL +2.80 pass; AL fails on a genuine degenerate Acc@1-selected snapshot. The recommended fix (Acc@10-aligned reg-best selector + degenerate-snapshot guard in `MultiTaskBestTracker`) is **out of scope / NOT implemented**; multi-seed re-run is required before any §0.x promotion. (`phase_c_verdict.md:120-125`)

5. **Tier B cat regression (−2.4 pp) is a build-scope confound, NOT a substrate property — but the clean region-only build was never run.** A build reusing canonical `embeddings.parquet` byte-identical and swapping only `region_embeddings.parquet` would test "cat flat"; current builds re-init the CheckinEncoder so cannot. (`phase_b_fl_3way.md:104`)

6. **ResLN is v12 encoder default for FUTURE builds only; the frozen v11 GCN substrate was NOT rebuilt.** The substrate-protocol-cleanup Tier B/FL baseline ran on the GCN (no-ResLN) substrate, so its absolute STL cat is ~1–1.7 pp below the canonical_improvement-best substrate. Documentation nuance, not a verdict error, but any new comparison should pin the encoder. (`canonical_improvement_coverage_audit.md:106`)

7. **The α=0 "encoder is reg-inert under MTL" claim is regime-and-config-scoped (B9, 50 ep) and rests on an OOD ablation.** The hedged wording ("no reg gain beyond the canonical log_T anchor under the joint config") is the defensible one. A non-OOD test (e.g. longer epoch budget, or a reg encoder that learns region under joint training) has NOT been run — that is exactly the `mtl_improvement` open frontier. (`CLOSURE.md:15,103`)

8. **HGI MTL ceiling is FL seed=42, 5-fold only.** AL/AZ/CA/TX HGI-MTL not run; the "MTL flattens everyone" claim is FL-anchored (plus the AL/AZ α=0 isolation). (`hgi_mtl_fl.md`)

9. **merge_design residual ~1 pp to HGI at AZ/FL is OPEN** but below the study's resolution; attributed to HGI's hierarchical-fclass-L2 + POI↔POI-contrastive training recipe, never re-trained into the merge family. (`STATE.md:84-101`)

10. **hgi_category_injection is CLOSED at AZ only** (seed=42, 30ep); FL/CA/TX re-open could flip the read (per-visit substrate gap differs at large states). Do NOT re-open without explicit commit. (`STATUS.md` re-open criteria)

11. **`mtl_improvement` (the architectural fix) has NO results yet.** Every other study has eliminated non-architectural causes; the actual next-reg MTL fix is unverified and unbuilt.

12. **Composite §4.2 STL HGI reg numbers are single-seed=42** (except where noted); FL multi-seed STL HGI reg not on file. Substrate on disk at AL/AZ only; FL/CA/TX productionisation needs HGI substrate regen. (`phase3_rank4_composite_analysis.md` caveats)
