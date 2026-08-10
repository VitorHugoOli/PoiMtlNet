#!/usr/bin/env python3
"""Submit a v18_2 lane to Modal, then ALWAYS harvest, archive and tear down.

WHY
  Three failure modes cost real money and data in this project, all of them avoidable:
    1. a finished or crashed job leaves its sandbox warm, billing until the 15-min idle window
       expires. Six strays were found live at once on 2026-08-10.
    2. results are harvested into a job-scoped hpc/ directory and then forgotten; nothing merges
       them into the repo, so a rerun silently redoes the work.
    3. a failure leaves no local trace of WHY, because the cell log stays on the volume and
       reading it costs another job.

  Two functions, and BOTH must be called. `submit()` dispatches and returns at once (the result
  arrives asynchronously, so it cannot block). `finish()` archives the harvested files and then
  tears the sandbox down from a `finally`, so the teardown runs whether archiving succeeded,
  failed or raised. A submit whose finish() is never reached leaves the container billing until
  Modal's 15-minute idle window closes.

USAGE (from the repl tool, where host.compute lives)
    from run_modal_cell import submit, finish
    d = submit(host, state="florida", seed=7, gpu="A100-40GB", parallel=True)
    # park on wait_for_notification for d["job_id"], then ALWAYS:
    out = finish(host, payload, state="florida", seed=7, gpu=d["gpu"])

RETURNS a dict with state/seed, job id, exit code, wall, the archived directory, and the parsed
metric values -- enough to decide the next step without re-entering the kernel.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import time

V14 = "check2hgi_design_k_resln_mae_l0_1"
ENGINE = "check2hgi_v18"
IMAGE = "im-XdbJMoxM9gCDEtaTZctMV0"


def lane_command(state: str, seed: int, parallel: bool = True, cells: str = "cat,reg,joint") -> str:
    """The in-container script. Preflights every input, runs the lane, harvests incrementally."""
    par = "PARALLEL=1 " if parallel else ""
    return f"""
set -uo pipefail
J="$(pwd)"; mkdir -p "$J/out"
cd /data/repo || {{ echo NO_REPO; exit 3; }}
[ -s /data/scripts/v18_2_scripts.tgz ] || {{ echo "ABORT: no scripts bundle"; exit 8; }}
mkdir -p docs/studies/closing_data/v18_2
tar xzf /data/scripts/v18_2_scripts.tgz -C docs/studies/closing_data/v18_2 2>/dev/null || exit 8
S=docs/studies/closing_data/v18_2/scripts
ST={state}; SEED={seed}
python "$S/preflight.py" --state $ST --cells {cells} || {{ echo PREFLIGHT_ABORT; exit 9; }}
for f in docs/results/closing_data/v18_2/${{ST}}_s${{SEED}}_*.json; do [ -s "$f" ] || rm -f "$f"; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T0=$SECONDS
{par}HARVEST=1 HARVEST_OUT="$J/out" LIVE_DIR=/data/live/${{ST}}_s${{SEED}} \
  LIVE_VOLUME=poimtl-v18-data INDUCTOR_ROOT=/data/inductor \
  REPO=/data/repo PY=/usr/local/bin/python \
  bash "$S/run_lane.sh" $ST $SEED {ENGINE} {V14} docs/results/closing_data/v18_2 2>&1 | tail -30
echo "LANE_WALL=$((SECONDS-T0))s"
cp docs/results/closing_data/v18_2/${{ST}}_s${{SEED}}_*.json "$J/out/" 2>/dev/null
# The cell logs are the ONLY way to diagnose a failure; ship their tails home unconditionally.
mkdir -p "$J/out/logs"
for f in docs/results/closing_data/v18_2/logs/${{ST}}_s${{SEED}}*; do
  [ -f "$f" ] && tail -c 200000 "$f" > "$J/out/logs/$(basename "$f")" 2>/dev/null
done
echo CELL_DONE
"""


def archive(payload: dict, dest_root: str = "modal_runs") -> pathlib.Path:
    """Copy every harvested file into a durable, self-describing local directory."""
    jid = payload["job_id"]
    st = payload.get("_state", "unknown")
    sd = payload.get("_seed", "x")
    dest = pathlib.Path(dest_root) / f"{st}_s{sd}_{jid[:8]}"
    dest.mkdir(parents=True, exist_ok=True)
    kept = []
    for f in payload.get("output_files", []):
        src = pathlib.Path(f)
        if not src.exists():
            continue
        parts = src.parts
        if "rundirs" in parts:
            # PRESERVAR a arvore: out/rundirs/<caminho literal do rundir>/<nome> e exatamente
            # o que push_to_host espelha no host. Achatar aqui apagava o caminho de destino e
            # o push devolvia "no rundir scores in this archive" sem erro nenhum.
            rel = str(pathlib.Path(*parts[parts.index("rundirs"):]))
        elif src.parent.name == "out":
            rel = src.name
        else:
            rel = f"{src.parent.name}__{src.name}"
        out_p = dest / rel
        out_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_p)
        kept.append(rel)
    meta = {
        "job_id": jid,
        "state": st,
        "seed": sd,
        "provider": payload.get("provider"),
        "gpu": payload.get("_gpu"),
        "parallel_cells": payload.get("_parallel"),
        "job_state": payload.get("state"),
        "exit_code": payload.get("exit_code"),
        "wall_seconds": payload.get("job_wall_s"),
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": kept,
        "notes": payload.get("notes", []),
    }
    (dest / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return dest


def values_from(dest: pathlib.Path) -> dict:
    """Parse whatever metrics landed, so the caller sees numbers rather than filenames."""
    out = {}
    # archive() flattens nested paths as "<parent>__<name>", so match by SUFFIX, not exact name.
    # Executing this on a real payload is what caught the exact-name version returning only reg.
    for p in dest.glob("*stl_cat_ceiling_score.json"):
        out["cat"] = json.load(open(p)).get("cat_macro_f1_mean")
    for p in dest.glob("*cat_score.json"):
        out.setdefault("cat", json.load(open(p)).get("cat_macro_f1_mean"))
    for p in dest.glob("*a40_matched_score.json"):
        d = json.load(open(p))
        out["joint_cat"] = d.get("cat_macro_f1_mean")
        out["joint_reg"] = d.get("reg_full_top10_mean")
    for p in dest.glob("*region_head_*.json"):
        try:
            agg = json.load(open(p))["heads"]["next_stan_flow"]["aggregate"]
            out["reg"] = round(agg["top10_acc_mean"] * 100, 4)
        except Exception:
            pass
    return out


def sweep_sandboxes(host, reason: str = "post-run teardown") -> dict:
    """Terminate every idle sandbox for the target. Safe ONLY when no job is still running.

    `close()` is host-scoped: it kills every sandbox and cancels running jobs. So check first.
    The ledger under-reports -- a close on 2026-08-10 terminated six sandboxes when the ledger
    listed two -- which is exactly why this is called unconditionally rather than "if needed".
    """
    led = str(host.compute.ledger())
    busy = [l.strip() for l in led.splitlines()
            if "state=running" in l or "state=harvesting" in l]
    if busy:
        return {"skipped": True, "reason": "jobs still active", "active": busy}
    rep = host.compute.create("byoc:modal").close(intent=reason)
    return {"skipped": False, "report": str(rep)}


def submit(host, state: str, seed: int, gpu: str = "A100-40GB", parallel: bool = True,
           cells: str = "cat,reg,joint", timeout_s: int = 10800,
           memory: int = 65536, cpu: int = 8) -> dict:
    """Submit one lane and return immediately. DOES NOT wait, archive or tear anything down.

    Submission cannot block: the result arrives asynchronously as a compute_done notification,
    so the caller must park on wait_for_notification and then call finish(), which is where the
    archiving and the guaranteed teardown live. Naming this `submit` rather than `run_cell`
    keeps that contract visible -- an earlier version claimed to tear down here and did not.

        d = submit(host, state="florida", seed=7)
        # ... park on wait_for_notification, get `payload` ...
        out = finish(host, payload, state="florida", seed=7, gpu=d["gpu"])

    If you never reach finish(), the sandbox survives until Modal's 15-minute idle window
    closes. Call finish() in every path, including on failure.
    """
    c = host.compute.create("byoc:modal", provider_params={
        "image": IMAGE, "gpu": gpu, "cpu": cpu, "memory": memory,
        "volumes": {"/data": "poimtl-v18-data"}, "timeout": timeout_s + 600})
    job = c.submit_job(command=lane_command(state, seed, parallel, cells),
                       intent=f"{state} seed-{seed} lane ({cells}) on {gpu}",
                       run_timeout_s=timeout_s)
    return {"state": state, "seed": seed, "gpu": gpu, "parallel": parallel,
            "cells": cells, "job_id": job.job_id}


def push_to_host(host, dest: pathlib.Path, repo: str = "/home/vitor.oliveira/PoiMtlNet") -> dict:
    """Copy the harvested rundir scores to the GPU host, preserving their paths.

    score_all.py does NOT use a sidecar's numbers: it re-reads every value out of the rundir.
    A Modal cell leaves its rundir on the volume, so without this the aggregation reports
    "sidecar present but <score>.json unreadable" and the cell never enters the tables.

    run_lane.sh's harvest_rundir_scores() copies each score JSON to
    out/rundirs/<literal rundir path>/<name>, so the destination here is a straight mirror --
    no path guessing. An earlier version derived paths from each JSON's own `rundir` field and
    put a file in the wrong one of two same-family rundirs.
    """
    import base64, json as _json
    sent = {}
    rundirs = dest / "rundirs"
    if rundirs.is_dir():
        for f in rundirs.rglob("*.json"):
            sent[str(f.relative_to(rundirs))] = base64.b64encode(f.read_bytes()).decode()
    for f in dest.glob("*region_head_*.json"):      # the P1 result the reg sidecar points at
        sent[f"docs/results/P1/{f.name.split('__')[-1]}"] = base64.b64encode(f.read_bytes()).decode()
    # Os SIDECARS. score_all.py os usa para achar o rundir e o skip-guard os usa para nao
    # repetir a celula; sem eles a celula fica invisivel E seria refeita no A40.
    for f in dest.glob("*_s*_*.json"):
        if f.name.startswith("region_head") or "__" in f.name:
            continue
        d = _json.loads(f.read_text() or "{}")
        if {"state", "seed", "family"} <= set(d):
            sent[f"docs/results/closing_data/v18/{f.name}"] = base64.b64encode(f.read_bytes()).decode()
    if not sent:
        return {"pushed": 0, "files": [], "note": "no rundir scores in this archive"}
    # The payload travels INSIDE the program as a base64 literal. A heredoc (`python3 - <<'PY'`)
    # rebinds stdin, so a piped JSON never reaches the interpreter -- that bug shipped once and
    # wrote nothing while appearing to succeed.
    prog = ("import base64, json, pathlib\n"
            f"files = json.loads(base64.b64decode('{base64.b64encode(_json.dumps(sent).encode()).decode()}'))\n"
            "for dst, b64 in files.items():\n"
            "    p = pathlib.Path(dst); p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_bytes(base64.b64decode(b64))\n"
            "    print(f'{p.stat().st_size:9d}B {dst}')\n")
    r = host.compute.create("ssh:nespedgpu").call_command(
        f"cd {repo} && echo '{base64.b64encode(prog.encode()).decode()}' | base64 -d > /tmp/_push.py "
        f"&& python3 /tmp/_push.py",
        intent="mirror Modal rundir scores onto the GPU host")
    written = [l for l in (r.get("stdout") or "").splitlines() if l.strip().endswith(".json")]
    if len(written) != len(sent):
        return {"pushed": 0, "error": "transfer incomplete", "expected": len(sent),
                "written": len(written), "stderr": (r.get("stderr") or "")[-300:]}
    return {"pushed": len(written), "files": sorted(sent), "verified": True}


def finish(host, payload: dict, gpu: str = None, parallel: bool = None,
           state: str = None, seed=None, dest_root: str = "modal_runs",
           push: bool = True) -> dict:
    """Handle a compute_done payload: archive, parse, then sweep sandboxes. Always call this."""
    payload = dict(payload)
    payload["_state"], payload["_seed"] = state, seed
    payload["_gpu"], payload["_parallel"] = gpu, parallel
    try:
        dest = archive(payload, dest_root)
        vals = values_from(dest)
        pushed = push_to_host(host, dest) if push else {"pushed": 0, "skipped": True}
        ok = payload.get("state") == "succeeded" and payload.get("exit_code") == 0
        return {"ok": ok, "job_id": payload.get("job_id"), "state": payload.get("state"),
                "exit_code": payload.get("exit_code"), "wall_s": payload.get("job_wall_s"),
                "archive": str(dest), "n_files": payload.get("output_file_count"),
                "values": vals, "pushed_to_host": pushed,
                "sweep": sweep_sandboxes(host, f"teardown after {state} s{seed}")}
    finally:
        # even if archiving raises, the container must not survive this call
        try:
            sweep_sandboxes(host, "failsafe teardown")
        except Exception:
            pass
