# Canonical Versions Registry

> ⛔ **2026-06-05 — UNDOCUMENTED RECIPE DEFAULT flagged (CONCERNS C25): `default_mtl use_class_weights=True`.** Every canonical MTL run (v11–v14) trained the reg head on **class-weighted CE** (`src/configs/experiment.py:364` → `mtl_cv.py:1283-1291`), while the STL reg ceiling (`default_next`) is **unweighted**. Class-balancing optimizes macro accuracy *away from* the reported Acc@10 metric → MTL reg is depressed ~10-14pp (verified: `--no-class-weights` recovers MTL reg to ≥ the STL ceiling). This was NEVER recorded as part of any version recipe. **CODE FIX LANDED 2026-06-05** (commit on `mtl-improve`): per-task class-weighting — `default_mtl` now sets **BOTH heads' CE UNWEIGHTED** (`use_class_weights_{reg,cat}=False`). reg-unweighted matches the Acc@10 metric + the STL ceiling; cat-unweighted was EMPIRICALLY validated (+5.1pp cat macro-F1 at AL — the "balancing helps macro-F1" assumption was tested and FALSE). **Reproduction:** to recover the pre-C25 (both-weighted) behaviour for v11–v14 numbers, pass **`--reg-class-weights --cat-class-weights`** (or `--use-class-weights`). **The absolute §0.1 MTL-reg numbers are UNDER RE-VALIDATION** (AL/GE/FL re-baseline under the fixed recipe in flight); a new pinned version will follow once it lands. Until then, treat absolute MTL-reg figures as provisional. See `CONCERNS.md §C25` + `docs/studies/archive/mtl_improvement/log.md` 2026-06-05.

**Purpose:** a single source of truth for which *recipe + substrate + code-default*
combination each named canonical version (`v11`, `v12`, …) refers to, and the
**exact reproduction map** between them. This file exists because the code
defaults flip over time, but the **BRACIS 2026 paper numbers must remain
reproducible forever**. Whenever a default changes, a new version is pinned here
with the flags needed to recover every prior version.

---

## How versioning works (preamble — keep this current)

1. A "canonical version" is the tuple **(MTL recipe, Check2HGI substrate/encoder,
   code-default state, git commit)** that produces a published or shipping set of
   numbers.
2. **Versions are append-only.** Never edit a frozen version's reproduction
   recipe — it is a historical contract. Add a NEW version row instead.
3. Every version row must state, explicitly:
   - the exact `scripts/train.py` invocation (or the delta vs the previous
     version);
   - which on-disk substrate (`output/check2hgi/<state>/`) it corresponds to;
   - the canonical results pointer (`RESULTS_TABLE.md §…`);
   - the validation grade of any change (paper-grade vs pilot vs STL-only);
   - the **reproduction map**: the flags to recover this version from the
     CURRENT code defaults.
4. When you flip a code default, you MUST: (a) pin the prior version here with
   its reproduction flags, (b) add the new version, (c) update
   `CLAUDE.md` + `NORTH_STAR.md` to point here.
5. **The on-disk substrate is a version artifact.** `output/check2hgi/<state>/`
   is the embeddings the model trains on. A version's numbers are only
   reproducible against the substrate it was built on. Do NOT overwrite a frozen
   version's substrate with a new-encoder rebuild — build the new substrate to a
   distinct path or accept that the old version becomes a build-from-source step.

---

## ⭐ The `--canon` selector (2026-06-07) — versions are now one flag

> 🏆 **2026-06-30 (board champion engine + the new v17 candidate):** the closing_data **board** champion-G runs on
> the **`check2hgi_dk_ovl`** engine (gated stride-1 overlap of v14) — the v16 bundle's per-head LRs are **inert under
> onecycle** (effective uniform 3e-3). **`--canon v17`** (now `DEFAULT_CANON`) = v16 +
> bs8192 + `--onecycle-per-head-lr` and **beats v16 board-wide at n=20** (AL/AZ/FL; see §v17 +
> [`../studies/closing_data/perhead_lr_n20.md`](../studies/closing_data/perhead_lr_n20.md)). CA/TX (running now at n=20) + the flag-OFF
> parity test gate it entering the §1 headline.

**As of 2026-06-07, `scripts/train.py --task mtl` takes `--canon {v11,v12,v15,v16,v17,none}`,
default **`v17`** (champion; = v16 base + bs8192 + per-head cat-lr).** Each version is a *bundle* of CLI flags (`src/configs/canon.py
::CANON_BUNDLES`) injected **before** your own flags, so **explicit flags always override the
bundle** (argparse last-wins). This makes the champion the default while keeping every prior
version reproducible with a single flag — instead of the ~15-flag invocations below.

- **Run the champion (default):** `train.py --task mtl --state <s> --seed <S>` → **v17** (bs8192 + per-head).
- **Traceback to a prior version:** `--canon v16` (bs2048 uniform-LR champion-G) · `--canon v11` (paper canon) · `--canon v12` (log_T-KD, weighted) · `--canon v15` (C25-unweighted) · `--canon none` (bare smoke defaults, no injection).
- **Scope:** MTL only; no-op under `--config` or `--task category|next`.
- **`--per-fold-transition-dir`** auto-derives to `output/<engine>/<state>` under `--canon` (so a
  bare run still uses leak-free seeded log_T); pass it explicitly to override.

> ⚠ **Contract — pin `--canon` in every script/driver.** Because the default bundle (v17) merges
> with whatever flags you pass, a driver that passes *some* recipe flags but omits `--canon` and
> some others will inherit the rest from v16 (e.g. a v12 driver that omits `--log-t-kd-weight`
> would get v16's 0.0). Partial specification was never safe (the old bare default was a smoke
> `mtlnet`); now full specification is one flag. **Scripted runs must pass `--canon vNN`** (or
> `--canon none` for a fully hand-specified recipe). The bundles are **append-only** (never edit a
> frozen one) and **guarded by `tests/test_configs/test_canon.py`** (asserts each `--canon vNN`
> resolves to the documented config field-by-field — a future default flip that breaks v11
> reproduction fails CI). The §vNN blocks below remain the authoritative recipe definitions; the
> bundles encode them.

---

## v11 — BRACIS paper canon (FROZEN 2026-05-30)

**This is the version the BRACIS 2026 submission numbers come from. It is FROZEN.
Do not change its reproduction recipe.**

- **Recipe:** B9 (FL/CA/TX) / H3-alt (AL/AZ) — the NORTH_STAR §Champion recipe.
- **Encoder / substrate:** **GCN** (canonical 2-layer `CheckinEncoder`). NOT ResLN.
- **log_T-KD:** **OFF** (`log_t_kd_weight = 0.0`).
- **Canonical numbers:** [`RESULTS_TABLE.md §0.1`](RESULTS_TABLE.md) (five-state
  architectural-Δ, n=20 seeds {0,1,7,100}) and the §0.4 recipe-selection rows.
  **§0.1 is v11.**
- **On-disk substrate:** `output/check2hgi/<state>/` AS IT EXISTS NOW **IS the v11
  (GCN) artifact.** It MUST NOT be overwritten by a ResLN rebuild. The §0.1 v11
  numbers correspond to this exact substrate.
- **Git commit (default-state snapshot, pre-v12-flip):** `99f56e8`
  (`99f56e80fa8961fd7280799aea24924c3afb30ca`, branch `main`). This is the last
  commit where the code defaults still produced v11 (log_T-KD off, encoder gcn)
  with no flags.
- **Reporting seeds:** §0.1 uses seeds **{0, 1, 7, 100}** (NOT the development
  seed 42 — see CLAUDE.md §development-vs-reporting-seed split).

### v11 exact invocation (B9, FL/CA/TX)

```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state {state} --engine check2hgi --seed {0|1|7|100} \
    --epochs 50 --folds 5 --batch-size 2048 \
    --model mtlnet_crossattn \
    --mtl-loss static_weight --category-weight 0.75 \
    --scheduler cosine --max-lr 3e-3 \
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --task-a-input-type checkin --task-b-input-type region \
    --log-t-kd-weight 0.0 \
    --per-fold-transition-dir output/check2hgi/{state}
```

(`--encoder` is not a `scripts/train.py` flag — the GCN encoder is the substrate
on disk. v11 means the substrate at `output/check2hgi/<state>/` was built with the
default GCN encoder. To rebuild a v11 substrate from source:
`python scripts/canonical_improvement/regen_emb_t3.py --state {State} --encoder gcn`.)

**H3-alt (small-state, AL/AZ):** drop `--alternating-optimizer-step`,
`--alpha-no-weight-decay`, `--min-best-epoch 5`; replace `--scheduler cosine
--max-lr 3e-3` with `--scheduler constant`. Heads + input-modality + `--log-t-kd-weight 0.0`
identical.

> ⚠ **Mechanism note (2026-06-12 code audit, P1-E — documentation only, numbers unchanged).**
> Under `--alternating-optimizer-step` the trainer bypasses `_get_weighted_loss` entirely
> (`src/training/runners/mtl_cv.py:596-612`): even batches step on the raw cat loss, odd batches
> on the raw reg loss — **`--mtl-loss static_weight --category-weight 0.75` never enters the
> objective.** So v11/B9 (FL/CA/TX) actually trained as 50/50 alternating single-task steps; the
> 0.75 weighting was only ever live at H3-alt small states (no alt-opt). Keep both flags in the
> invocation for bit-exact reproduction, but do NOT describe B9's mechanism as "static-weighted
> 0.75", and treat any historical `category-weight` sweep under B9 as a no-op. Full trail:
> `docs/studies/archive/mtl_improvement/CODE_AUDIT_2026-06-12.md`.

### Reproduction map — getting v11 from v12-default code

After the 2026-05-30 default flip (v12), recover v11 by passing the OLD defaults
explicitly:

| Axis | v12 default (new) | v11 flag to pass |
|---|---|---|
| log_T-KD weight | 0.2 (on, scoped to MTL check2hgi_next_region) | `--log-t-kd-weight 0.0` |
| Check2HGI encoder | `resln` (future builds) | rebuild substrate with `--encoder gcn` **OR** use the existing frozen `output/check2hgi/<state>/` GCN substrate (do not rebuild) |
| **MTL joint checkpoint selector** (2026-06-03 default flip) | `geom_simple` = sqrt(cat_F1·reg_Acc@10) | **`--checkpoint-selector joint_f1_mean`** (the v11 `0.5*(cat_f1+reg_f1)`) |

> ⚠ **2026-06-03 — joint-selector default flipped (C21).** The default checkpoint selector is now
> `joint_geom_simple` (correct). The v11 paper-canon JOINT/deployable numbers were produced with the
> broken `0.5*(cat_f1+reg_f1)` selector, so reproducing v11's **joint-selected** numbers requires
> `--checkpoint-selector joint_f1_mean`. **§0.1 itself is UNAFFECTED** — it reports per-task
> diagnostic-best epochs, which are selector-independent. See CONCERNS §C21 + CHANGELOG 2026-06-03.

---

## v12 — NEW DEFAULT (2026-05-30)

**This is the new code default as of 2026-05-30. It layers the two validated
`substrate-protocol-cleanup` findings onto v11.**

- **Recipe:** v11 recipe (B9 / H3-alt) **unchanged** + **log_T-KD W=0.2, τ=1.0**.
- **Encoder / substrate:** **ResLN** (`ResidualLNEncoder`) is now the default for
  FUTURE Check2HGI builds. (The existing on-disk substrate is still the v11 GCN
  artifact — see below.)
- **log_T-KD:** **ON by default**, `--log-t-kd-weight 0.2 --log-t-kd-tau 1.0`,
  **scoped** to `--task mtl --task-set check2hgi_next_region` only. Category-only
  runs, non-region task-sets, and non-MTL tasks are unaffected (weight stays 0.0).

### What changed v11 → v12, and the evidence grade

| Change | Mechanism | Evidence | Grade |
|---|---|---|---|
| **log_T-KD W=0.2 ON** | KL distillation of the per-fold train-only log_T (region Markov-1 prior) into the reg-head logits; lifts MTL reg via the live prior pathway | AL +2.27 / AZ +4.91 pp disjoint reg, n=20 (seeds {0,1,7,100}), Wilcoxon p=9.54e-07, 20/20 folds; leak-audited clean (7-vector audit, NO LEAK). FL +2.40 pp single-seed pilot (p=0.031, 5/5); CA/TX +1.42/+1.71 pp single-fold sign-and-magnitude. | **PAPER-GRADE at AL/AZ; single-seed PILOT at FL/CA/TX** |
| **ResLN encoder default** | best STL cat encoder (residual + LayerNorm GCN-family); promoted by `canonical_improvement` T3.2 | cat F1 +0.86 FL / +1.48 AL / +1.70 AZ STL (5/5 seeds, p=0.03125); reg ≈0 small states / +0.71 FL (mostly v3c). Leak +2.24 IJM-verified honest. | **STL-only / representation-quality.** **NO MTL benefit** — under the cross-attn MTL joint-training regime the substrate axis is washed out (the regime finding). ResLN is the default for substrate/STL/generality, NOT for any MTL improvement. |

### CRITICAL — the on-disk substrate is still v11 (GCN)

The v12 ResLN default affects **FUTURE builds only**. The current
`output/check2hgi/<state>/` directory is the **frozen v11 GCN substrate** and was
**NOT rebuilt** during the v12 flip. Therefore:

- Running the v12-default `scripts/train.py` against the existing
  `output/check2hgi/<state>/` substrate trains on the **GCN (v11) substrate** with
  **log_T-KD ON** — i.e. it is "v11 substrate + log_T-KD", not full v12.
- A full v12 (ResLN) substrate requires an explicit rebuild
  (`python scripts/canonical_improvement/regen_emb_t3.py --state {State}` — now
  defaults to `--encoder resln`) to a path you choose. Per the regime finding,
  this rebuild changes the **STL/cat** picture, NOT the MTL reg verdict.
- **Do NOT overwrite `output/check2hgi/<state>/`** — it is the frozen v11 paper
  substrate and the §0.1 numbers depend on it byte-for-byte.

### v12 exact invocation (B9, FL/CA/TX) — code defaults

With v12 defaults you no longer need to pass the log_T-KD flags (they default ON
for this task-set):

```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state {state} --engine check2hgi --seed {0|1|7|100} \
    --epochs 50 --folds 5 --batch-size 2048 \
    --model mtlnet_crossattn \
    --mtl-loss static_weight --category-weight 0.75 \
    --scheduler cosine --max-lr 3e-3 \
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --task-a-input-type checkin --task-b-input-type region \
    --per-fold-transition-dir output/check2hgi/{state}
    # log_T-KD W=0.2 τ=1.0 now ON by default for check2hgi_next_region MTL
```

The `--mtl-loss`/`--cat-head`/`--reg-head`/`--task-b-input-type` flags are STILL
required — only the log_T-KD default flipped. The remaining "silently wrong"
defaults (loss, heads, input modality) are unchanged from v11.

### v12 reproduction map — recovering v11

To reproduce the v11 paper-canon numbers from v12-default code, pass:

```bash
    --log-t-kd-weight 0.0        # turn KD back off (v11)
    # AND use the existing GCN substrate at output/check2hgi/<state>/ (do not rebuild)
    # (or rebuild a GCN substrate explicitly: regen_emb_t3.py --encoder gcn)
```

---

## v13 — recommended STL / forward-MTL base engine (BLESSED 2026-05-30, opt-in)

**Not a code-default flip.** `--engine` is an explicit, required CLI argument — there is
no silent engine default to change. v13 *blesses* a specific engine as the **recommended
"best base substrate"** for STL/representation work and as the **forward base for future
MTL improvement work** (per user decision 2026-05-30). The canonical `check2hgi` engine
identity is **unchanged** — the BRACIS paper (v11/v12) is unaffected.

- **Engine:** **`check2hgi_resln_design_b`** = **ResLN encoder** (v12 default) **+ Design B**
  (POI2Vec injected at the POI-pool boundary, `poi_emb_for_reg = poi_emb.detach() + γ·Linear(POI2Vec)`,
  cat path detached so cat stays canonical). It is the best all-around STL engine in the
  investigation.
- **Why it's the base:** STL dual-axis champion — **equalises HGI on reg at AL** (62.10 ≈ HGI
  61.86), closes ~80 %/~30 % of the canonical→HGI gap at AZ/FL, **while keeping/widening
  check2hgi's 2–3× cat lead** over HGI. (`tier_resln/phase_resln_verdict.md`.) ResLN+design_j
  is a registered AL-specialist alternative.
- **Grade — STL-only.** **NO MTL benefit.** Per the regime finding (substrate/encoder gains
  wash out under cross-attn MTL — even HGI ≈ canonical in MTL), v13 does **not** improve
  MTL reg or cat *today*. Its purpose is to be the **best representation/generality base**
  so that when `mtl_improvement` fixes the joint-training regime, the substrate is already
  the strongest available. Do **not** cite v13 as an MTL improvement.
- **Build dependency (important):** requires the **POI2Vec teacher**
  (`output/hgi/<state>/poi2vec_poi_embeddings_<State>.csv`) + the Design-B build. Built at
  **all five states (AL / AZ / FL / CA / TX)** as of 2026-05-30 (CA/TX added — see
  `studies/archive/substrate-protocol-cleanup/log.md`). The canonical `check2hgi` engine remains
  the safe default for any state without a v13 substrate.
- **Reproduction safety:** v13 is additive/opt-in. v11 (paper) and v12 (default) are
  untouched; nothing about `output/check2hgi/<state>/` changes.

### Build a v13 substrate (per state)
```bash
# 1. POI2Vec teacher (if absent for the state)
python scripts/substrate_protocol_cleanup/run_poi2vec.py --city {State} --epochs 100 --device cuda
# 2. ResLN + Design-B substrate
python scripts/probe/build_design_b_poi_pool.py --state {state} --encoder resln \
    --out-engine check2hgi_resln_design_b --epochs 500 --device cuda
# 3. postbuild inputs + log_T (next.parquet + next_region.parquet + cp canonical seed log_T)
bash scripts/substrate_protocol_cleanup/postbuild_design_substrate.sh check2hgi_resln_design_b {state}
```

### v13 train invocation
Same recipe as v12 (B9 / H3-alt + log_T-KD default ON) — only the engine + transition dir change:
```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state {state} --engine check2hgi_resln_design_b --seed {seed} \
    ... (B9/H3-alt recipe flags, identical to v12) ... \
    --per-fold-transition-dir output/check2hgi_resln_design_b/{state}
```
For STL evaluation use the `scripts/p1_region_head_ablation.py` path with the same engine.

---

## v14 — dual-axis champion base engine (BLESSED 2026-06-02, opt-in)

**Not a code-default flip; same opt-in posture as v13.** v14 *blesses* a stronger dual-axis
substrate. The canonical `check2hgi` engine identity is **unchanged** — BRACIS paper (v11/v12)
unaffected; `output/check2hgi/<state>/` untouched.

- **Engine:** **`check2hgi_design_k_resln_mae_l0_1`** = **ResLN encoder + HGI Delaunay POI-POI GCN
  on the reg path (design_k, λ=0.1) + T5.2b masked-POI category-aggregate recon (mae λ=0.3)**.
  Three orthogonal axes stacked: **resln+mae → next-cat** (encoder/cat path), **Delaunay →
  next-reg** (detached reg path).
- **Why it supersedes v13 as the base:** v13 (resln+design_b) raised the *fclass* axis (log_T-
  redundant for reg) and only closed ~30 % of the FL reg gap. v14 adds the **spatial axis**
  (Delaunay POI edges — imports HGI's own spatial graph), the ONE axis that translates to L2-reg.
  **Leak-free multi-seed FL: cat 67.36 (≈ frozen-canon, ≫ HGI +33pp) + reg 0.7024 (closes ~69 % of
  the canonical→HGI gap; −0.36pp residual).** Dual-axis safe (the −2.5pp cat vs frozen-v11 is
  fresh-vs-frozen, not a cost — matched-fresh control gcn_ctrl 64.61 ≈ v14 cat-path).
- **Grade — STL-only (same regime limit as v13). CONFIRMED multi-seed 2026-06-03.** The earlier
  2-fold seed42 MTL pilot (cat −0.21pp, reg-Acc +0.03pp) is now superseded by a full **5-fold ×
  4-seed {0,1,7,100}, FL/AL/AZ, leak-free** MTL run vs matched canonical: **v14 ≈ canonical** (FL
  tie both tasks; AL/AZ mixed within noise) — the STL dual-axis gain does **not** survive cross-attn
  MTL (the regime finding). Do **not** cite v14 as an MTL improvement; it is the **strongest forward
  base** for the Part-2 MTL/routing study. Full tables + audit:
  [`v14_mtl_vs_canonical.md`](v14_mtl_vs_canonical.md).
- **Provenance caveat (paper):** v14 IMPORTS HGI's Delaunay POI graph (`output/hgi/<state>/temp/
  edges.csv`) — credit the borrowed spatial structure; v14 learns orthogonal embeddings (cosine to
  HGI ≈ −0.003), not a roundabout HGI clone.
- **Build dependency:** POI2Vec teacher (anchor) + HGI Delaunay edges + HGI POI emb. Built at **FL**
  as of 2026-06-02; **AL/AZ/CA/TX pending** (the Delaunay reg lever is state-dependent — confirmed
  +reg at FL; smaller at small states).
- **Reproduction safety:** additive/opt-in. v11/v12/v13 + all on-disk frozen substrates untouched.

### Build a v14 substrate (per state)
```bash
# design_k + resln + mae (the new build-script DEFAULT: resln+mae ON; pass --encoder gcn
# --mae-poi-lambda 0 to recover plain design_k).
python scripts/probe/build_design_k_delaunay.py --state {state} \
    --out-suffix resln_mae_l0_1 --epochs 500 --device cuda
# postbuild inputs + log_T (next.parquet + next_region.parquet + seeded log_T)
bash scripts/substrate_protocol_cleanup/postbuild_design_substrate.sh check2hgi_design_k_resln_mae_l0_1 {state}
```
For STL eval: `scripts/p1_region_head_ablation.py --region-emb-source check2hgi_design_k_resln_mae_l0_1`
(reg) + `scripts/train.py --task next --engine check2hgi_design_k_resln_mae_l0_1` (cat).
**Reg ranking MUST use `--per-fold-transition-dir` with seeded log_T** (default leaks ~+3pp).

---

## v15 — UNWEIGHTED-LOSS recipe (C25 fix; the re-validated MTL recipe, 2026-06-05)

**What v15 IS:** v11/v12-recipe + substrate, but with the **C25 unweighting fix** — the load-bearing change that re-validated the MTL story. Both MTL heads train on **UNWEIGHTED CrossEntropyLoss** (per-task `use_class_weights_{reg,cat}=False`, now the `default_mtl` default) instead of the silent class-weighted CE that depressed reg ~10-14pp / cat ~3-5pp. Plus the **Acc@10 reg checkpoint monitor** (was Acc@1). The substrate is unchanged (v11 GCN for the paper table; v14 design_k for the forward base).

**Why it exists:** the pre-v15 class-weighted reg CE was an **objective mismatch** vs the reported Acc@10 metric (CONCERNS C25). v15 is the recipe under which the MTL→STL reg gap CLOSES, the substrate gain transfers to MTL, and the composite advantage dissolves — see `studies/archive/mtl_improvement/PAPER_UPDATE.md`.

**Re-validated numbers (multi-seed {0,1,7,100}, unweighted real-joint, onecycle):**
| state | MTL reg (v14) | MTL reg (canon v11-GCN) | STL ceiling | MTL cat (v14) | STL cat ceiling |
|---|---|---|---|---|---|
| AL | 64.52 | 62.60 | 62.88 | 53.38 | 49.97 |
| GE | 57.84 | 56.34 | 58.45 | 61.37 | 58.12 |
| FL | 71.55 (dual-tower 73.06) | 70.74 | 73.31 | 71.89 | 69.97 |

**Reproduction map:**
- **To recover pre-C25 (class-weighted) MTL numbers** (v11–v14 as previously reported): pass **`--reg-class-weights --cat-class-weights`** (or `--use-class-weights`) + (for the old reg monitor) note the Acc@1→Acc@10 monitor change is a code default, not flag-gated (the disjoint `per_metric_best.top10_acc_indist` is unaffected by it).
- **v15 is the new code default** — bare `default_mtl` runs are now v15 (unweighted). 
- **Recipe note:** the re-validation used **onecycle**; an FL-B9 §0.1-continuity run (exact paper-table recipe) is the pending follow-up. The large-state reg champion is the **dual-tower** (`mtlnet_crossattn_dualtower`, closes the FL gap to −0.25).

**Caveat:** the FL numbers above use onecycle; the §0.1 paper table uses B9 at FL — the FL-B9 v15 run pins exact §0.1 continuity (LANDED 2026-06-05: +3.15 reg / +3.52 cat same-harness A/B; see RESULTS_TABLE §0.1 annotation). Frozen (c)/(d) STL ceilings are v15-comparable (always unweighted p1).

---

## v16 — CHAMPION MTL config "G" (Pareto-positive single model, 2026-06-06)

**What v16 IS:** v15 (unweighted) recipe + the **champion ARCHITECTURE "G"** found by `mtl_improvement`: the reg-private **dual-tower** with **`aux` fusion** and the **α·log_T prior OFF**. The first config where a SINGLE MTL model **MATCHES the STL reg ceiling (Pareto-non-inferior) AND substantially beats the STL cat ceiling (+3pp)**, at all 4 available states, 4-seed. ⚠ **REG verb CORRECTED 2026-06-07 (B-A2):** earlier "beats reg ceiling" compared G's *indist* Acc@10 to the (c) ceiling's *full* `top10_acc`; on a matched metric G is ~0.35pp BELOW (FL 72.93 vs 73.31) → "matches", not "beats". Cat beat exact. ⚠ **G′ (cat-private, FL-ONLY — DEMOTED 2026-06-07):** giving the CAT head a private tower too (both-private dual-tower, `mtlnet_crossattn_dualtower_catpriv`) gained cat at FL only (74.77, +1.61); the multi-state confirm (AL/AZ/GE × 4 seeds) **FALSIFIED** it — cat CRATERS at small states (AL 37.66 = −15.25 vs G, AZ −12.45, GE −3.59; reg flat). The cat-private tower **UNDERFITS** small-state cat (NOT overfit — the off-label STAN-flow head is over-regularized: AL train-F1 caps ~0.45 vs the GRU head's 0.98; a rescue screen of lower dropout / softer LR / smaller tower **CLOSED 2026-06-07 with NO rescue** — best AL lever still −14.5pp vs G, and the FL gain survives ONLY at the original `priv_dropout=0.3` (lowering it erases the gain, 74.74→73.17); the STAN flow/attention head is architecturally mismatched for a 7-class target at small data) → G′ is a **CLOSED FL-only experimental dead-end**, **NOT a champion**; **G (cat-SHARED) remains the multi-state champion**. See CHAMPION.md / INDEX `#T2V-5`.

**Config (full rationale + runnable command: [`../studies/archive/mtl_improvement/CHAMPION.md`](../studies/archive/mtl_improvement/CHAMPION.md)):**
`--model mtlnet_crossattn_dualtower --reg-head next_stan_flow_dualtower --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 --cat-head next_gru --mtl-loss static_weight --category-weight 0.75 --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 --log-t-kd-weight 0.0` on the **v14 substrate** (`check2hgi_design_k_resln_mae_l0_1`), unweighted (v15 default).

**Numbers — 4-seed {0,1,7,100}, vs (c) STL ceilings (reg / cat):**
| state | G reg | (c) reg | Δreg | G cat | (c) cat | Δcat |
|---|---|---|---|---|---|---|
| AL | 64.47±0.11 | 62.88 | **+1.59** | 52.91±0.27 | 49.97 | **+2.94** |
| AZ | 55.75±0.21 | 55.11 | **+0.64** | 54.48±0.74 | 51.01 | **+3.47** |
| GE | 59.37±0.04 | 58.45 | **+0.92** | 61.43±0.26 | 58.12 | **+3.31** |
| FL | 73.57±0.06 | 73.31 | **+0.26** | 73.16±0.04 | 69.97 | **+3.19** |

FL also ties the (d) composite reg (73.62) while winning cat → composite strictly dominated.

**Status / scope:** v16 is a **study champion, NOT the paper §0 canon** (paper still v11). It is **opt-in** (explicit `--model`/`--reg-head` flags; the code default model is still canon cross-attn). The mechanism (aux fusion + prior-OFF) is decisive — `gated` fusion or re-enabling the additive prior REGRESSES it (CHAMPION.md §5). Architecture capacity is NOT the lever (falsified 5 ways). CA/TX need a v14 build first. **Reproduce:** the command above + seeded fresh per-fold log_T at `--per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state>`.

**VALIDATED 2026-06-07 (Tier 2V — `studies/archive/mtl_improvement/CRITIQUE_TIER2_C25_2026-06-06.md` §7 + `INDEX.html #tier2v`):** v16 survived a skeptical re-test. The (c)/(d) ceilings were re-run at G's seeds {0,1,7,100} (they were seed-42 only) — stable (σ≤0.7), G still beats both at 4/4 states. Alt-archs re-ranked FAIRLY (standalone, post-C25, per-arch `category-weight`) all lose by 1.6–2.1pp → the "architecture-capacity is not the reg lever" claim is un-confounded. No tail regression; no hypertuning lever beats G (logit-adjust HURTS the MTL cat — plain CE is the MTL cat optimum; private STAN right-sized; FAMO ≈ G). Param-honest: G = base_a +4.9% (one model, not "½ of two"). The v16 champion is **paper-safe**.

**v16 RE-CONFIRMED 2026-06-08 (Tiers 3/4/5 close — no recipe change):** the loss/optimization axis was exhausted with no Pareto gain over v16's `static_weight cw=0.75` (full balancer registry + a new gated `--loss-scale-norm` flag both FALSIFIED; gradient cosine(cat,reg)≈0 = no conflict for balancers to resolve). Reg-input levers (overlap, HGI routing) and the HSM reg head also gave no gain. **v16 recipe is unchanged + further hardened.** Note: `--loss-scale-norm` (new CLI flag, `loss_scale_norm` config field) is **experimental, default OFF, and FALSIFIED** (starves the high-cardinality reg head) — it is NOT part of any canon version; do not enable. See `results/mtl_improvement/T4_audit_and_verdict.md` + `studies/archive/mtl_improvement/WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`.

## v17 — CHAMPION CANDIDATE: per-head cat-lr + bs8192 (PROMOTED opt-in 2026-06-30; AL/AZ/FL; CA/TX PENDING)

**What v17 IS:** v16 + **`--batch-size 8192`** + **`--onecycle-per-head-lr`** (sets `MTL_ONECYCLE_PER_HEAD_LR=1`).
The discovery (`train_perf_multifold`): the v16 per-head LRs (`cat 1e-3 / reg 3e-3 / shared 1e-3`) are **INERT under
onecycle** — a scalar `max_lr` broadcasts 3e-3 to every head, so v16 effectively trains all heads at 3e-3. The fix
passes a per-group `max_lr` list so cat-lr 1e-3 actually applies; combined with the larger batch (gradient-noise
benefit), it beats v16 at every tested state. Mechanism = the v16 cat head was **overdriven at the uniform 3e-3**
(cat-LR overshoot, exposed by the bigger batch at FL); reg-capture refuted by an isolation decomposition.

**Config:** v16 recipe **+ `--batch-size 8192 --onecycle-per-head-lr`** (engine `check2hgi_dk_ovl` on the board).
Invocation: `train.py --task mtl --canon v17 --state <state> --seed <S> --engine check2hgi_dk_ovl --canon none …`
(or `--canon v17` directly for the v14-substrate engine). The `--onecycle-per-head-lr` flag is **required** — a v17
bundle without it would silently reproduce v16-uniform-3e-3.

**Numbers — n=20 {0,1,7,100} (`../studies/closing_data/perhead_lr_n20.md`):**
| state | v17 cat | v16/champion cat | Δ cat | reg |
|---|---|---|---|---|
| AL | **64.54** | 63.55 | **+0.99** (+0.64 from the per-head lever alone) | 69.80 (flat) |
| AZ | **65.84** | 63.57 | **+2.27** (+1.52 per-head) | 59.56 (flat) |
| FL | **79.85** | base 79.68 | **+0.17 cat / +0.20 reg**, ~7% faster | 77.42 |

**Status / scope:** **v17 is now `DEFAULT_CANON`** (bare `train.py` runs it; v16 via `--canon v16`; `MTL_ONECYCLE_PER_HEAD_LR` stays default-OFF, v17 sets `--onecycle-per-head-lr`). **§0.1
(v11) is UNAFFECTED** — v11 is a separate frozen bundle on the GCN substrate using `cosine` (the per-head fix is
onecycle-only); adding v17 changes nothing in v11. **CA/TX are RUNNING now at n=20** (reg C 6.5–8.5k, fp32) — they enter the RESULTS_BOARD §1 headline once matched-n. Open: the flag-OFF eager
byte-identical parity test (`future_works/per_head_lr_onecycle_fix.md`). Full record:
`../studies/train_perf_multifold/{BATCH_SIZE_SWEEP.md,RESULTS_SUMMARY.md,CLOSURE.md}` + `perhead_lr_n20.md`.

---

## Quick reference — version × axis matrix

| Axis | v11 (paper canon) | v12 (new default) | v13 (recommended base, opt-in) |
|---|---|---|---|
| MTL recipe | B9 / H3-alt | B9 / H3-alt (unchanged) | B9 / H3-alt + log_T-KD (= v12) |
| log_T-KD | OFF (0.0) | **ON (0.2, τ=1.0)** — MTL check2hgi_next_region only | ON (= v12) |
| Engine | `check2hgi` | `check2hgi` | **`check2hgi_resln_design_b`** (ResLN + POI2Vec@pool) |
| Encoder (future builds) | GCN | **ResLN** | ResLN |
| Substrate dependency | none | none | **POI2Vec teacher** (built at all five states AL/AZ/FL/CA/TX) |
| MTL benefit | (baseline) | log_T-KD reg only | **none today** (STL/forward base; regime-limited) |
| Status | FROZEN paper canon | code default | **blessed opt-in** (engine identity unchanged) |
| Canonical numbers | `RESULTS_TABLE.md §0.1` | §0.1 + §0.8 log_T-KD lift | `tier_resln/phase_resln_verdict.md` (STL) |
| Recover from current code | n/a | `--log-t-kd-weight 0.0` + GCN substrate | n/a (additive opt-in) |
| Git commit (default snapshot) | `99f56e8` | `4414a82` (v12 flip) | (this commit) |

**v14 (dual-axis champion, opt-in, 2026-06-02):** engine `check2hgi_design_k_resln_mae_l0_1`
(ResLN + Delaunay-POI-GCN/design_k + mae). Recipe = v12 (B9/H3-alt + log_T-KD). Adds the
**spatial axis** v13 lacked → cat 67.36 (≫ HGI) + reg 0.7024 (closes ~69 % of HGI gap).
**STL-only (no MTL benefit, regime-limited); strongest forward base for Part-2.** Engine identity
unchanged → v11/v12/v13 untouched. Build default = resln+mae (opt out: `--encoder gcn --mae-poi-lambda 0`).
Reg ranking requires seeded `--per-fold-transition-dir`. See the v14 section above + `studies/archive/embedding_eval/FINAL_SYNTHESIS.md`.

---

## See also

- [`../NORTH_STAR.md`](../NORTH_STAR.md) — champion recipe + the regime finding.
- [`RESULTS_TABLE.md §0.1`](RESULTS_TABLE.md) (v11 canon) + §0.8 (log_T-KD v12 lift).
- [`../studies/archive/substrate-protocol-cleanup/CLOSURE.md`](../studies/archive/substrate-protocol-cleanup/CLOSURE.md) — the study that validated v12.
- [`../findings/F_TIER_A1_PROMOTION.md`](../findings/F_TIER_A1_PROMOTION.md) + [`../findings/F_TIER_A1_LEAK_AUDIT.md`](../findings/F_TIER_A1_LEAK_AUDIT.md) — log_T-KD promotion + leak audit.
- [`../findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md`](../findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md) — one-stop investigation synthesis.
