"""Shared helpers for a study's state.json.

The active study directory is resolved from the ``STUDY_DIR`` environment
variable (absolute or repo-relative path).  If the variable is not set, it
defaults to ``docs/studies/fusion`` so existing workflows keep working on the
fusion branch without any change.

Atomic read/modify/write + small query helpers. No training logic here —
each script composes these primitives.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

_study_dir_env = os.environ.get("STUDY_DIR", "")
if _study_dir_env:
    STUDIES_DIR = Path(_study_dir_env)
    if not STUDIES_DIR.is_absolute():
        STUDIES_DIR = REPO_ROOT / STUDIES_DIR
else:
    STUDIES_DIR = REPO_ROOT / "docs" / "studies" / "fusion"

STATE_PATH = STUDIES_DIR / "state.json"
RESULTS_DIR = STUDIES_DIR / "results"

STUDY_VERSION = "1.0"
PHASES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_state() -> dict[str, Any]:
    now = utcnow()
    return {
        "study_version": STUDY_VERSION,
        "study_started": now,
        "last_update": now,
        "current_phase": "P0",
        "open_issues": [],
        "phases": {
            phase: {
                "status": "running" if phase == "P0" else "planned",
                "started": now if phase == "P0" else None,
                "finished": None,
                "tests": {},
            }
            for phase in PHASES
        },
    }


def read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"{STATE_PATH} does not exist. Run `python scripts/study/launch_test.py --init` "
            "or `/study init` first."
        )
    with STATE_PATH.open("r") as fh:
        return json.load(fh)


def write_state(state: dict[str, Any]) -> None:
    state["last_update"] = utcnow()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".state.", suffix=".json", dir=str(STATE_PATH.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(state, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_path, STATE_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@contextmanager
def mutate_state() -> Iterator[dict[str, Any]]:
    """Read, yield for mutation, then write atomically."""
    state = read_state()
    yield state
    write_state(state)


def ensure_initialized(force: bool = False) -> Path:
    if STATE_PATH.exists() and not force:
        return STATE_PATH
    write_state(empty_state())
    return STATE_PATH


def get_test(state: dict[str, Any], phase: str, test_id: str) -> dict[str, Any] | None:
    return state.get("phases", {}).get(phase, {}).get("tests", {}).get(test_id)


def set_test(state: dict[str, Any], phase: str, test_id: str, entry: dict[str, Any]) -> None:
    state["phases"].setdefault(phase, {"status": "planned", "tests": {}})
    state["phases"][phase].setdefault("tests", {})
    state["phases"][phase]["tests"][test_id] = entry


def iter_tests(state: dict[str, Any], phase: str | None = None) -> Iterator[tuple[str, str, dict[str, Any]]]:
    phases = [phase] if phase else PHASES
    for ph in phases:
        tests = state.get("phases", {}).get(ph, {}).get("tests", {})
        for tid, entry in tests.items():
            yield ph, tid, entry


def find_next_planned(state: dict[str, Any], phase: str | None = None) -> tuple[str, str, dict[str, Any]] | None:
    target = phase or state.get("current_phase")
    for ph, tid, entry in iter_tests(state, target):
        if entry.get("status") == "planned":
            return ph, tid, entry
    return None


def open_issue(
    state: dict[str, Any],
    *,
    test_id: str,
    issue_type: str,
    description: str,
) -> str:
    issues = state.setdefault("open_issues", [])
    issue_id = f"ISS-{len(issues) + 1:03d}"
    issues.append(
        {
            "issue_id": issue_id,
            "test_id": test_id,
            "type": issue_type,
            "raised_at": utcnow(),
            "description": description,
            "status": "open",
            "resolution": None,
        }
    )
    return issue_id


def summarize_phase(state: dict[str, Any], phase: str) -> dict[str, Any]:
    tests = state.get("phases", {}).get(phase, {}).get("tests", {})
    counts: dict[str, int] = {}
    for entry in tests.values():
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "phase": phase,
        "status": state.get("phases", {}).get(phase, {}).get("status"),
        "total_tests": len(tests),
        "by_status": counts,
    }
