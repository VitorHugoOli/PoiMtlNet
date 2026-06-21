#!/usr/bin/env python3
"""M2 Pro lane — substrate-column baseline embedding fan-out driver (P3 board).

Builds the LIGHT baseline embeddings (B2b skip-gram, POI2Vec faithful, CTLE) on the
GATED stride-1 overlap base, TRAIN-ONLY per fold, for states x seeds {0,1,7,100} x
5 folds. (B2c one-hot64 is per-state and built separately — it is fold-independent.)

Per cell:
  1. STAGE frozen check2hgi reads into a per-cell scratch OUTPUT_DIR (POI2Vec + CTLE
     use the non-namespaced CHECK2HGI probe-engine and must not clobber the frozen
     substrate; B2b writes to its own namespaced dir and needs no staging).
  2. BUILD with --stride 1 and MTL_RAM_HEADROOM_GB=2 (the 16 GB default guard is too
     aggressive on a 24 GB box; the fold-split read is <=0.1 GB).
  3. The builder asserts row-alignment (next==next_region==seq) + val-user disjointness.
  4. KEEP embeddings.parquet + provenance + leak/fold markers into a consolidated
     handoff dir; DELETE the regennable next/next_region/sequences (CUDA regenerates
     them deterministically — see M2PRO_BUILD_LOG.md §4). DISK is the binding constraint.

Disk-floor stop: aborts before a cell if free space < --disk-floor-gb.
Resumable: skips a cell whose handoff embeddings.parquet already exists.

Handoff layout (the orchestrator consolidates / registers engines for the CUDA score):
  output/board_baselines/<baseline>/<state>/s<seed>_f<fold>/
      embeddings.parquet  next_build_provenance.json  LEAK_MARKER.txt

This driver does NOT run the matched-head comparison (CUDA-only, device-class rule).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/closing_data/m2pro_baseline_fanout.py \
      --baselines b2b poi2vec ctle --states alabama arizona \
      --seeds 0 1 7 100 --folds 0 1 2 3 4 [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUTPUT = REPO / "output"
HANDOFF = OUTPUT / "board_baselines"
PY = str(REPO / ".venv" / "bin" / "python")

# Builders that use the non-namespaced CHECK2HGI probe-engine -> need staged scratch.
STAGED = {"poi2vec", "ctle"}


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def stage_scratch(state: str, scratch: Path) -> None:
    """Replicate the smoke-script staging: copy the frozen reads the builder OVERWRITES
    (embeddings/next/sequences), symlink the read-only ones (region/graph)."""
    real = OUTPUT / "check2hgi" / state
    dst = scratch / "check2hgi" / state
    (dst / "temp").mkdir(parents=True, exist_ok=True)
    (dst / "input").mkdir(parents=True, exist_ok=True)
    shutil.copy2(real / "temp" / "sequences_next.parquet", dst / "temp" / "sequences_next.parquet")
    shutil.copy2(real / "input" / "next.parquet", dst / "input" / "next.parquet")
    shutil.copy2(real / "embeddings.parquet", dst / "embeddings.parquet")
    for name, sub in [("region_embeddings.parquet", ""), ("checkin_graph.pt", "temp")]:
        link = dst / sub / name if sub else dst / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to((real / sub / name if sub else real / name).resolve())


def builder_cmd(baseline: str, state: str, seed: int, fold: int, scratch: Path | None):
    """Return (argv, env, native_engine_dir) for one cell. native_engine_dir is where
    the builder writes embeddings.parquet (scratch for staged builders)."""
    env = dict(os.environ, PYTHONPATH="src", MTL_RAM_HEADROOM_GB="2")
    base = [PY, ]
    if baseline == "b2b":
        argv = [PY, "scripts/baselines/build_b2b_skipgram_substrate.py",
                "--state", state, "--seed", str(seed), "--fold", str(fold),
                "--n-splits", "5", "--epochs", "5", "--dim", "64",
                "--stride", "1", "--device", "cpu"]
        native = OUTPUT / f"b2b_skipgram_s{seed}_f{fold}" / state
    elif baseline == "poi2vec":
        env["OUTPUT_DIR"] = str(scratch)
        argv = [PY, "scripts/baselines/build_poi2vec_substrate.py", state,
                "--seed", str(seed), "--fold", str(fold), "--n-splits", "5",
                "--epochs", "30", "--embed-dim", "64", "--user-dim", "64",
                "--theta", "0.05", "--route-count", "4", "--context-window", "9",
                "--loss-form", "mixture", "--stride", "1", "--device", "cpu"]
        native = scratch / "check2hgi" / state
    elif baseline == "ctle":
        env["OUTPUT_DIR"] = str(scratch)
        argv = [PY, "scripts/baselines/build_ctle_substrate.py",
                "--state", state, "--seed", str(seed), "--fold", str(fold),
                "--pretrain-epochs", "10", "--batch-size", "256", "--max-len", "64",
                "--lr", "1e-3", "--stride", "1"]
        native = scratch / "check2hgi_ctle" / state
    else:
        raise ValueError(baseline)
    return argv, env, native


def keep_and_trim(baseline, state, seed, fold, native: Path):
    """Copy embeddings + markers to the handoff dir; delete regennable next/next_region."""
    hd = HANDOFF / baseline / state / f"s{seed}_f{fold}"
    hd.mkdir(parents=True, exist_ok=True)
    emb = native / "embeddings.parquet"
    assert emb.exists(), f"missing embeddings.parquet for {baseline}/{state} s{seed} f{fold}: {emb}"
    shutil.copy2(emb, hd / "embeddings.parquet")
    for marker in ["input/next_build_provenance.json", "CTLE_FOLD.txt"]:
        src = native / marker
        if src.exists():
            shutil.copy2(src, hd / Path(marker).name)
    (hd / "LEAK_MARKER.txt").write_text(
        f"baseline={baseline} state={state} seed={seed} fold={fold} stride=1 gated\n"
        f"TRAIN-ONLY per fold; row-align (next==next_region==seq) + val-user disjoint "
        f"asserted by the builder. next/next_region dropped (regen on CUDA, --stride 1).\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["b2b", "poi2vec", "ctle"])
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 7, 100])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--disk-floor-gb", type=float, default=18.0,
                    help="abort before a cell if free space < this (CA/TX spike ~9 GB)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = [(b, st, sd, fd)
             for b in args.baselines for st in args.states
             for sd in args.seeds for fd in args.folds]
    print(f"=== M2 Pro baseline fan-out: {len(cells)} cells "
          f"(baselines={args.baselines} states={args.states} "
          f"seeds={args.seeds} folds={args.folds}) ===", flush=True)
    print(f"    disk free={free_gb(OUTPUT):.1f} GB  floor={args.disk_floor_gb} GB", flush=True)

    done = skipped = failed = 0
    for i, (b, st, sd, fd) in enumerate(cells, 1):
        tag = f"[{i}/{len(cells)}] {b}/{st} s{sd} f{fd}"
        hd_emb = HANDOFF / b / st / f"s{sd}_f{fd}" / "embeddings.parquet"
        if hd_emb.exists():
            print(f"{tag}  SKIP (handoff exists)", flush=True)
            skipped += 1
            continue
        fg = free_gb(OUTPUT)
        if fg < args.disk_floor_gb:
            print(f"{tag}  STOP: disk {fg:.1f} GB < floor {args.disk_floor_gb} GB", flush=True)
            break
        if args.dry_run:
            print(f"{tag}  (dry-run)  free={fg:.1f} GB", flush=True)
            continue

        scratch = (OUTPUT / "_scratch" / f"{b}_{st}_s{sd}_f{fd}") if b in STAGED else None
        t0 = time.time()
        try:
            if scratch is not None:
                if scratch.exists():
                    shutil.rmtree(scratch)
                scratch.mkdir(parents=True)
                stage_scratch(st, scratch)
            argv, env, native = builder_cmd(b, st, sd, fd, scratch)
            log = OUTPUT / "_scratch" / f"log_{b}_{st}_s{sd}_f{fd}.txt"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as lf:
                r = subprocess.run(argv, env=env, cwd=str(REPO), stdout=lf,
                                   stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"{tag}  FAIL (exit {r.returncode}) — see {log}", flush=True)
                failed += 1
                continue
            keep_and_trim(b, st, sd, fd, native)
        finally:
            if scratch is not None and scratch.exists():
                shutil.rmtree(scratch)
        done += 1
        print(f"{tag}  OK  {time.time()-t0:.0f}s  free={free_gb(OUTPUT):.1f} GB", flush=True)

    print(f"=== fan-out end: done={done} skipped={skipped} failed={failed} "
          f"free={free_gb(OUTPUT):.1f} GB ===", flush=True)


if __name__ == "__main__":
    main()
