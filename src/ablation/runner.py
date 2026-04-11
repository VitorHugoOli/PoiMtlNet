"""Run staged MTL ablations from the candidate matrix.

The runner launches canonical ``scripts/train.py`` subprocesses, giving every
candidate its own RESULTS_ROOT so storage folder names cannot collide.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parents[2]
_src = str(_root / "src")
if _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ablation.candidates import MTLCandidate, get_candidate, iter_candidates


@dataclass
class AblationResult:
    candidate: str
    stage: str
    epochs: int
    folds: int
    seed: int | None
    status: str
    returncode: int
    run_dir: str
    log_file: str
    duration_seconds: float
    command: str
    joint_score: float | None = None
    next_f1: float | None = None
    category_f1: float | None = None
    error: str = ""


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _candidate_argv(
    candidate: MTLCandidate,
    state: str,
    engine: str,
    epochs: int,
    folds: int,
    seed: int | None,
) -> list[str]:
    argv = [
        sys.executable,
        "scripts/train.py",
        "--task",
        "mtl",
        "--state",
        state,
        "--engine",
        engine,
        "--epochs",
        str(epochs),
        "--folds",
        str(folds),
        "--model",
        candidate.model_name,
        "--mtl-loss",
        candidate.mtl_loss,
    ]
    if seed is not None:
        argv.extend(["--seed", str(seed)])
    for key, value in candidate.model_params.items():
        argv.extend(["--model-param", f"{key}={_format_value(value)}"])
    for key, value in candidate.mtl_loss_params.items():
        if candidate.mtl_loss == "static_weight" and key == "category_weight":
            argv.extend(["--category-weight", str(value)])
        else:
            argv.extend(["--mtl-loss-param", f"{key}={_format_value(value)}"])
    return argv


def _latest_run_dir(candidate_root: Path, engine: str, state: str, epochs: int) -> Path | None:
    state_root = candidate_root / engine / state
    if not state_root.exists():
        return None
    runs = [
        path
        for path in state_root.iterdir()
        if path.is_dir() and f"_ep{epochs}_" in path.name
    ]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def _parse_metrics(run_dir: Path | None) -> tuple[float | None, float | None, float | None]:
    if run_dir is None:
        return None, None, None
    summary_path = run_dir / "summary" / "full_summary.json"
    if not summary_path.exists():
        return None, None, None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    next_f1 = data.get("next", {}).get("f1", {}).get("mean")
    category_f1 = data.get("category", {}).get("f1", {}).get("mean")
    joint_score = data.get("model", {}).get("joint_score", {}).get("mean")
    if joint_score is None and next_f1 is not None and category_f1 is not None:
        joint_score = 0.5 * (float(next_f1) + float(category_f1))
    return joint_score, next_f1, category_f1


def _write_summary(path: Path, rows: list[AblationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _run_candidate(
    candidate: MTLCandidate,
    state: str,
    engine: str,
    epochs: int,
    folds: int,
    seed: int | None,
    label_root: Path,
    data_root: Path,
    output_dir: Path,
) -> AblationResult:
    candidate_root = label_root / candidate.name
    log_dir = label_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{seed}" if seed is not None else ""
    log_file = log_dir / f"{candidate.name}{seed_tag}.log"

    env = os.environ.copy()
    env["DATA_ROOT"] = str(data_root)
    env["OUTPUT_DIR"] = str(output_dir)
    env["RESULTS_ROOT"] = str(candidate_root)
    env["PYTHONPATH"] = "src:."

    argv = _candidate_argv(candidate, state, engine, epochs, folds, seed)
    command = " ".join(argv)
    started = time.time()
    print(f"[ablation] running {candidate.name} ({folds} fold(s), {epochs} epochs)")
    with log_file.open("w", encoding="utf-8") as log:
        log.write("$ " + command + "\n\n")
        proc = subprocess.run(
            argv,
            cwd=_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    duration = time.time() - started
    run_dir = _latest_run_dir(candidate_root, engine, state, epochs)
    joint_score, next_f1, category_f1 = _parse_metrics(run_dir)
    status = "ok" if proc.returncode == 0 else "failed"
    error = "" if proc.returncode == 0 else f"see {log_file}"
    print(
        "[ablation] finished "
        f"{candidate.name}: status={status} joint={joint_score} "
        f"next={next_f1} category={category_f1}"
    )
    return AblationResult(
        candidate=candidate.name,
        stage=candidate.stage,
        epochs=epochs,
        folds=folds,
        seed=seed,
        status=status,
        returncode=proc.returncode,
        run_dir=str(run_dir) if run_dir else "",
        log_file=str(log_file),
        duration_seconds=duration,
        command=command,
        joint_score=joint_score,
        next_f1=next_f1,
        category_f1=category_f1,
        error=error,
    )


def _top_candidates(rows: list[AblationResult], top_k: int) -> list[str]:
    successful = [
        row for row in rows
        if row.status == "ok" and row.joint_score is not None
    ]
    successful.sort(key=lambda row: row.joint_score or float("-inf"), reverse=True)
    return [row.candidate for row in successful[:top_k]]


def _write_manifest(
    path: Path,
    *,
    stage: str,
    state: str,
    engine: str,
    epochs: int,
    folds: int,
    seed: int | None,
    candidates: tuple[MTLCandidate, ...],
    promoted_from: str | None = None,
) -> None:
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "state": state,
        "engine": engine,
        "epochs": epochs,
        "folds": folds,
        "seed": seed,
        "promoted_from": promoted_from,
        "candidates": [
            {
                "name": candidate.name,
                "stage": candidate.stage,
                "model_name": candidate.model_name,
                "model_params": dict(candidate.model_params),
                "mtl_loss": candidate.mtl_loss,
                "mtl_loss_params": dict(candidate.mtl_loss_params),
                "rationale": candidate.rationale,
            }
            for candidate in candidates
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run staged MTL ablation candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--state", default="alabama")
    parser.add_argument("--engine", default="dgi")
    parser.add_argument("--stage", choices=("phase1", "phase2", "all"), default="phase1")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--promote-top", type=int, default=0)
    parser.add_argument("--promote-epochs", type=int, default=15)
    parser.add_argument("--promote-folds", type=int, default=2)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_root / "results" / "ablations",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/Users/vitor/Desktop/mestrado/ingred/data")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "/Users/vitor/Desktop/mestrado/ingred/output")),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    candidates = (
        tuple(get_candidate(name) for name in args.candidate)
        if args.candidate
        else iter_candidates(args.stage)
    )

    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    label = f"{args.stage}_{args.folds}fold_{args.epochs}ep{seed_suffix}"
    label_root = args.results_root / label
    label_root.mkdir(parents=True, exist_ok=True)
    manifest_path = label_root / "manifest.json"
    _write_manifest(
        manifest_path,
        stage=args.stage,
        state=args.state,
        engine=args.engine,
        epochs=args.epochs,
        folds=args.folds,
        seed=args.seed,
        candidates=candidates,
    )
    print(f"[ablation] wrote manifest: {manifest_path}")

    rows = [
        _run_candidate(
            candidate,
            state=args.state,
            engine=args.engine,
            epochs=args.epochs,
            folds=args.folds,
            seed=args.seed,
            label_root=label_root,
            data_root=args.data_root,
            output_dir=args.output_dir,
        )
        for candidate in candidates
    ]
    summary_path = label_root / "summary.csv"
    _write_summary(summary_path, rows)
    print(f"[ablation] wrote summary: {summary_path}")

    if args.promote_top > 0:
        promoted_names = _top_candidates(rows, args.promote_top)
        promoted_candidates = tuple(get_candidate(name) for name in promoted_names)
        promoted_label = (
            f"{args.stage}_promoted_{args.promote_folds}fold_"
            f"{args.promote_epochs}ep{seed_suffix}"
        )
        promoted_root = args.results_root / promoted_label
        promoted_root.mkdir(parents=True, exist_ok=True)
        promoted_manifest = promoted_root / "manifest.json"
        _write_manifest(
            promoted_manifest,
            stage=f"{args.stage}_promoted",
            state=args.state,
            engine=args.engine,
            epochs=args.promote_epochs,
            folds=args.promote_folds,
            seed=args.seed,
            candidates=promoted_candidates,
            promoted_from=str(summary_path),
        )
        print(f"[ablation] wrote promoted manifest: {promoted_manifest}")
        print(f"[ablation] promoting: {', '.join(promoted_names)}")
        promoted_rows = [
            _run_candidate(
                candidate,
                state=args.state,
                engine=args.engine,
                epochs=args.promote_epochs,
                folds=args.promote_folds,
                seed=args.seed,
                label_root=promoted_root,
                data_root=args.data_root,
                output_dir=args.output_dir,
            )
            for candidate in promoted_candidates
        ]
        promoted_summary = promoted_root / "summary.csv"
        _write_summary(promoted_summary, promoted_rows)
        print(f"[ablation] wrote promoted summary: {promoted_summary}")


if __name__ == "__main__":
    main()
