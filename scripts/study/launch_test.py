"""Launch the next planned test from state.json.

A test entry is expected to carry:
  - test_id, phase, claim_ids
  - config: {state, engine, task?, model_name?, mtl_loss?, seed, epochs, folds, ...}
  - (optional) command: pre-built CLI; if present, used verbatim
  - (optional) expected: {joint_range, cat_f1_range, ...}

This script does NOT enumerate the full phase grid — that lives in phase docs.
It just picks one entry marked `planned` and runs it.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _state import (
    REPO_ROOT,
    find_next_planned,
    mutate_state,
    read_state,
    utcnow,
)


def _build_command(config: dict[str, Any]) -> list[str]:
    cmd: list[str] = [sys.executable, "scripts/train.py"]
    arg_map = {
        "task": "--task",
        "state": "--state",
        "engine": "--engine",
        "model_name": "--model",
        "candidate": "--candidate",
        "mtl_loss": "--mtl-loss",
        "seed": "--seed",
        "epochs": "--epochs",
        "folds": "--folds",
        "embedding_dim": "--embedding-dim",
        "gradient_accumulation_steps": "--gradient-accumulation-steps",
        "next_target": "--next-target",
        "category_weight": "--category-weight",
    }
    for key, flag in arg_map.items():
        if key in config and config[key] is not None:
            cmd.extend([flag, str(config[key])])

    for k, v in (config.get("model_params") or {}).items():
        cmd.extend(["--model-param", f"{k}={v}"])
    for k, v in (config.get("mtl_loss_params") or {}).items():
        cmd.extend(["--mtl-loss-param", f"{k}={v}"])

    if config.get("mtl_loss") in {"cagrad", "aligned_mtl", "pcgrad"}:
        if "--gradient-accumulation-steps" not in cmd:
            cmd.extend(["--gradient-accumulation-steps", "1"])
    return cmd


def _current_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _preflight(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    engine = config.get("engine")
    state = config.get("state")
    if not engine or not state:
        errors.append("config missing engine/state")
        return errors
    cat = REPO_ROOT / "output" / engine / state / "input" / "category.parquet"
    nxt = REPO_ROOT / "output" / engine / state / "input" / "next.parquet"
    task = config.get("task", "mtl")
    if task in {"mtl", "category"} and not cat.exists():
        errors.append(f"missing {cat}")
    if task in {"mtl", "next"} and not nxt.exists():
        errors.append(f"missing {nxt}")
    return errors


def _detect_run_dir(config: dict[str, Any], before: set[Path]) -> Path | None:
    """Glob the results dir for a newly-created training run matching config."""
    engine = config.get("engine")
    state = config.get("state")
    if not (engine and state):
        return None
    parent = REPO_ROOT / "results" / engine / state
    if not parent.exists():
        return None
    task = config.get("task", "mtl")
    # scripts/train.py writes run dirs like `mtlnet_lr...bs..._ep..._<ts>`.
    candidates = [
        p for p in parent.iterdir()
        if p.is_dir() and p not in before and (p / "summary" / "full_summary.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def launch(
    phase: str | None,
    test_id: str | None,
    dry_run: bool,
) -> int:
    # Resolve target test + build command WITHOUT mutating state yet.
    state = read_state()
    if test_id:
        phase = phase or state.get("current_phase")
        phase_tests = state.get("phases", {}).get(phase, {}).get("tests", {})
        entry = phase_tests.get(test_id)
        if entry is None:
            print(f"[launch] no test {test_id} in phase {phase}", file=sys.stderr)
            return 2
    else:
        found = find_next_planned(state, phase)
        if found is None:
            print(f"[launch] no planned tests in phase {phase or state.get('current_phase')}")
            return 1
        phase, test_id, entry = found

    config = entry.get("config", {})

    errors = _preflight(config)
    if errors:
        print(f"[launch] preflight failed for {test_id}:")
        for e in errors:
            print(f"  FAIL  {e}")
        return 2

    if entry.get("command"):
        cmd = shlex.split(entry["command"])
    else:
        cmd = _build_command(config)

    print(f"[launch] {test_id}")
    print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")

    if dry_run:
        print("[launch] dry-run — state unchanged, not executing")
        return 0

    # Snapshot existing run dirs BEFORE training so we can identify the new one.
    engine = config.get("engine", "")
    state_name = config.get("state", "")
    parent_dir = REPO_ROOT / "results" / engine / state_name
    before_run_dirs = set(p for p in parent_dir.iterdir() if p.is_dir()) if parent_dir.exists() else set()

    with mutate_state() as st:
        st["phases"][phase]["tests"][test_id] = {
            **entry,
            "command": " ".join(shlex.quote(c) for c in cmd),
            "git_commit": _current_commit(),
            "started_at": utcnow(),
            "status": "running",
        }

    env = os.environ.copy()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    except KeyboardInterrupt:
        print("[launch] interrupted", file=sys.stderr)
        with mutate_state() as st:
            st["phases"][phase]["tests"][test_id].update(
                {"status": "failed", "finished_at": utcnow(), "notes": "interrupted"}
            )
        return 130

    elapsed = time.time() - t0
    run_dir = _detect_run_dir(config, before_run_dirs)
    with mutate_state() as st:
        entry_out = st["phases"][phase]["tests"][test_id]
        entry_out["finished_at"] = utcnow()
        entry_out["wall_clock_seconds"] = round(elapsed, 1)
        if run_dir is not None:
            entry_out["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
        if proc.returncode == 0:
            entry_out["status"] = "completed"
            where = entry_out.get("run_dir", "(run_dir not detected)")
            print(f"[launch] completed in {elapsed:.0f}s.")
            print(f"  run_dir: {where}")
            print(f"  next:    /study import --run-dir {where} --phase {phase} --test-id {test_id}")
        else:
            entry_out["status"] = "failed"
            entry_out["notes"] = f"exit_code={proc.returncode}"
            print(f"[launch] FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the next planned study test.")
    parser.add_argument("--phase", default=None, help="Phase to launch from (default: current_phase).")
    parser.add_argument("--test-id", default=None, help="Explicit test_id (overrides next-planned picker).")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing.")
    args = parser.parse_args()
    return launch(args.phase, args.test_id, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
