# Canonical Versions Registry

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

### Reproduction map — getting v11 from v12-default code

After the 2026-05-30 default flip (v12), recover v11 by passing the OLD defaults
explicitly:

| Axis | v12 default (new) | v11 flag to pass |
|---|---|---|
| log_T-KD weight | 0.2 (on, scoped to MTL check2hgi_next_region) | `--log-t-kd-weight 0.0` |
| Check2HGI encoder | `resln` (future builds) | rebuild substrate with `--encoder gcn` **OR** use the existing frozen `output/check2hgi/<state>/` GCN substrate (do not rebuild) |

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
  `studies/substrate-protocol-cleanup/log.md`). The canonical `check2hgi` engine remains
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

---

## See also

- [`../NORTH_STAR.md`](../NORTH_STAR.md) — champion recipe + the regime finding.
- [`RESULTS_TABLE.md §0.1`](RESULTS_TABLE.md) (v11 canon) + §0.8 (log_T-KD v12 lift).
- [`../studies/substrate-protocol-cleanup/CLOSURE.md`](../studies/substrate-protocol-cleanup/CLOSURE.md) — the study that validated v12.
- [`../findings/F_TIER_A1_PROMOTION.md`](../findings/F_TIER_A1_PROMOTION.md) + [`../findings/F_TIER_A1_LEAK_AUDIT.md`](../findings/F_TIER_A1_LEAK_AUDIT.md) — log_T-KD promotion + leak audit.
- [`../findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md`](../findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md) — one-stop investigation synthesis.
