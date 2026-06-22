# HANDOFF — A40 recreates the canonical CA/TX HGI substrate · 2026-06-22

> Why this exists: the closing_data freeze needs **HGI at all 6 board states** (AL, AZ, FL, CA, TX, GE) as a
> single canonical build — it is the substrate-comparison anchor (RUN_MATRIX §2: T2 ablation + substrate
> probes + STL STAN/STAN-Flow × {v14, HGI}). The A40 had HGI for **AL/AZ/FL/GE only** — **CA/TX were never
> trained there** (only the POI2Vec teacher + Delaunay `temp/` graph exist, built earlier as a v14 dependency).
> This handoff records the CA/TX HGI rebuild **on the A40** (the canonical-build machine) so the whole 6-state
> HGI set is one consistent build, then synced to the Mac/board.

## 0 · Why the A40, not locally (the comparability reason)

HGI is **NOT bit-reproducible across machines** — verified 2026-06-22: the shared states AL/AZ/FL had
**byte-different** `embeddings.parquet` on the Mac vs the A40 (Mac copies were divergent local rebuilds, the
exact `FREEZE_READINESS §2` 🔴 BLOCKER). The canonical AL/AZ/FL/GE HGI all live on the **A40**, so CA/TX must
be built on the **same machine** to keep the 6-state HGI set one canonical build. The Mac's pre-existing
HGI AL/AZ/FL have since been **overridden** with the A40 copies; CA/TX (the Mac had its own divergent CA/TX,
the A40 had none) are being regenerated on the A40 and pulled back.

## 1 · State at handoff time

| Engine | AL | AZ | FL | CA | TX | GE |
|---|---|---|---|---|---|---|
| **v14** `check2hgi_design_k_resln_mae_l0_1` | ✅ canon (manifest) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HGI** | ✅ A40-canon | ✅ | ✅ | ⏳ **rebuilding on A40** | ⏳ **rebuilding on A40** | ✅ |

- **v14:** all 6 states verified byte-identical to `V14_HASH_MANIFEST.json` on both A40 and Mac (18/18 anchors OK).
- **HGI AL/AZ/FL/GE:** A40-canonical, synced to the Mac (overriding the Mac's divergent copies).
- **HGI CA/TX:** launched on the A40 (this handoff).

## 2 · The build (already launched 2026-06-22 ~04:09 UTC)

**Script:** [`scripts/closing_data/build_catx_hgi.sh`](../../../scripts/closing_data/build_catx_hgi.sh) — a
parametrized mirror of `scripts/mtl_improvement/build_ge_hgi_train.sh` (the recipe that produced the
**canonical GE HGI**, so CA/TX get the identical recipe).

**Canonical recipe (frozen HGI config — `research/embeddings/hgi/CLAUDE.md` + `hgi.pipe.py`):**
`dim=64, alpha=0.5, attention_head=4, lr=0.006, gamma=1.0, max_norm=0.9, epoch=2000, warmup_period=40,
poi2vec_epochs=100, cross_region_weight=0.7, device=cpu`. HGI trains on **CPU** (the inner loop is ~176× slower
on MPS/GPU; CPU is correct, not a shortcut) → the A40 GPU stays free.

**Prereqs (present on the A40, verified):** `output/hgi/{california,texas}/poi2vec_poi_embeddings_*.csv` (teacher)
+ `temp/{edges.csv, boroughs_area.csv, pois.csv, encodings.json, poi-encoder.tensor}`. `gowalla.pt` was absent
→ Phase 4 (`create_hgi_graph_pickle`, `force_preprocess=False` reuses `temp/`) builds it, then Phase 5 trains.

**Launch (per state, detached, parallel — 16 threads each on the 32-core A40):**
```bash
cd /home/vitor.oliveira/PoiMtlNet
setsid bash scripts/closing_data/build_catx_hgi.sh California > /tmp/catx_hgi/california.log 2>&1 < /dev/null &
setsid bash scripts/closing_data/build_catx_hgi.sh Texas      > /tmp/catx_hgi/texas.log      2>&1 < /dev/null &
```
**ETA:** ~17–20 min/state at ~1.9 it/s (CA: 169,145 POIs / 8,501 regions). Output per state:
`output/hgi/<state>/{embeddings.parquet, region_embeddings.parquet}`.

## 3 · Monitor / verify (on the A40)

```bash
tail -f /tmp/catx_hgi/california.log    # look for: DONE_California_HGI_TRAIN
tail -f /tmp/catx_hgi/texas.log         # look for: DONE_Texas_HGI_TRAIN
pgrep -af build_catx_hgi                # 0 procs + DONE marker = finished
```
The script self-verifies at the end (`region_embeddings` shape, `nan=False`, `std`, and
`embeddings.parquet exists=True`). If a run dies before the DONE marker, re-launch that state (Phase 4 then
skips because `gowalla.pt` now exists).

## 4 · Pull back to the Mac (automated; manual fallback below)

A background watcher on the Mac polls for both DONE markers, then rsyncs the two anchor files per state and
hash-compares Mac↔A40. **Manual equivalent:**
```bash
H=vitor.oliveira@nespedgpu.caf.ufv.br; R=/home/vitor.oliveira/PoiMtlNet
for s in california texas; do for f in embeddings.parquet region_embeddings.parquet; do
  rsync -a -e ssh "$H:$R/output/hgi/$s/$f" output/hgi/$s/ ; done ; done
# verify identical:
ssh $H "cd $R && sha256sum output/hgi/{california,texas}/{embeddings,region_embeddings}.parquet"
shasum -a 256 output/hgi/{california,texas}/{embeddings,region_embeddings}.parquet
```

## 5 · Follow-up — give HGI the same hash-manifest treatment as v14 (recommended)

v14 has `V14_HASH_MANIFEST.json`; **HGI has none**, which is exactly how the Mac's AL/AZ/FL drifted unnoticed.
Once CA/TX land, generate an `HGI_HASH_MANIFEST.json` (mirror the v14 one: per-state `embeddings.parquet` +
`region_embeddings.parquet` bytes + sha256, built on the A40) and make every board machine verify against it
before use. The A40 then becomes the single canonical HGI source for all 6 states, just like v14.
