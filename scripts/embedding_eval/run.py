"""CLI runner for the embedding-eval ladder — L0 (geometry) + L1 (linear probe).

Example:
    python scripts/embedding_eval/run.py \
        --engines hgi check2hgi check2hgi_resln_design_b \
        --states florida --tasks cat reg \
        --ref-engine hgi

Writes:
    docs/results/embedding_eval/metrics_long.csv   (one row per metric/seed)
    docs/results/embedding_eval/summary.md         (engine x task tables)

L2 (capacity ladder) / L3 (MTL) are GPU-hour sequence runs launched via
scripts/train.py; pass --emit-l2l3 to print the ready-to-run commands.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# repo root + src/ on path (mirror scripts/train.py)
_root = Path(__file__).resolve().parent.parent.parent
for p in (_root, _root / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from configs.paths import EmbeddingEngine

from scripts.embedding_eval.geometry import l0_metrics, linear_cka
from scripts.embedding_eval.labels import ItemTable, load_item_table
from scripts.embedding_eval.linear_probe import _fit_one

TASK_NAMES = {"cat": "next-cat", "reg": "next-reg", "poi": "next-poi"}


def _engine(name: str) -> EmbeddingEngine:
    return EmbeddingEngine(name)


def _records_for(
    tab: ItemTable,
    task: str,
    knn_k: int,
    sil_sample: int,
    seeds: list[int],
    probe_epochs: int,
    probe_lr: float,
    max_items: int | None = None,
) -> list[dict]:
    """L0 + L1 rows for one (engine, state, granularity, task)."""
    mask = tab.valid_mask(task)
    emb = tab.emb[mask]
    labels = tab.labels(task)[mask]
    rows: list[dict] = []
    base = dict(engine=tab.engine, state=tab.state, task=task, granularity=tab.granularity)

    if len(np.unique(labels)) < 2:
        return rows  # degenerate (e.g. poi @ poi-granularity) — skip

    # Subsample AFTER masking (m9) so realized N / class coverage are comparable
    # across engines regardless of each engine's label-coverage on this task.
    n_total = len(labels)
    if max_items is not None and n_total > max_items:
        rng = np.random.default_rng(seeds[0])
        sel = rng.choice(n_total, size=max_items, replace=False)
        emb, labels = emb[sel], labels[sel]

    # provenance: realized eval N and how many of THIS engine's items carry a
    # valid label for this task (M6 — surfaces silent per-engine drop).
    rows.append({**base, "level": "L0", "metric": "n_eval", "seed": -1, "value": float(len(labels))})
    rows.append({**base, "level": "L0", "metric": "label_coverage", "seed": -1,
                 "value": float(mask.mean())})

    # L0 — deterministic geometry (seed only feeds the silhouette subsample)
    for metric, value in l0_metrics(
        emb, labels, k=knn_k, silhouette_sample=sil_sample, seed=seeds[0]
    ).items():
        rows.append({**base, "level": "L0", "metric": metric, "seed": -1, "value": value})

    # L1 — multi-seed linear probe (densify labels to [0, C))
    uniq, dense = np.unique(labels, return_inverse=True)
    num_classes = len(uniq)
    for s in seeds:
        res = _fit_one(
            emb, dense.astype(np.int64), num_classes,
            seed=s, epochs=probe_epochs, lr=probe_lr,
            weight_decay=0.0, test_size=0.2,
        )
        for metric, value in res.items():
            rows.append({**base, "level": "L1", "metric": metric, "seed": s, "value": value})
    return rows


def _cka_rows(tables: dict, ref_engine: str) -> list[dict]:
    """Linear CKA of each engine vs ref, on placeid-aligned POI-pooled embeddings."""
    rows: list[dict] = []
    by_state: dict[str, dict[str, ItemTable]] = {}
    for (eng, state), tab in tables.items():
        if tab.granularity != "poi":
            continue
        by_state.setdefault(state, {})[eng] = tab
    for state, engs in by_state.items():
        if ref_engine not in engs:
            continue
        ref = engs[ref_engine]
        ref_idx = {int(p): i for i, p in enumerate(ref.placeid)}
        for eng, tab in engs.items():
            common = [int(p) for p in tab.placeid if int(p) in ref_idx]
            if len(common) < 10:
                continue
            tab_idx = {int(p): i for i, p in enumerate(tab.placeid)}
            xi = ref.emb[[ref_idx[p] for p in common]]
            yi = tab.emb[[tab_idx[p] for p in common]]
            rows.append({
                "engine": eng, "state": state, "task": "-", "granularity": "poi",
                "level": "L0", "metric": f"cka_vs_{ref_engine}", "seed": -1,
                "value": linear_cka(xi, yi),
            })
    return rows


def _md_table(piv: pd.DataFrame) -> str:
    """Render a pivot table (with index) as a markdown table — no tabulate dep."""
    piv = piv.reset_index()
    cols = [str(c) for c in piv.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in piv.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if not np.isnan(v) else "nan")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _write_summary(df: pd.DataFrame, out: Path) -> None:
    lines = ["# Embedding-eval ladder — summary\n"]
    # L0 geometry table per task
    for task in sorted(df[df.level == "L0"].task.unique()):
        if task == "-":
            continue
        sub = df[(df.level == "L0") & (df.task == task)]
        piv = sub.pivot_table(index="engine", columns="metric", values="value")
        lines.append(f"\n## L0 geometry — {TASK_NAMES.get(task, task)}\n")
        lines.append(_md_table(piv.round(4)))
    # CKA table
    cka = df[df.metric.str.startswith("cka_vs_")]
    if len(cka):
        lines.append("\n## L0 — linear CKA vs reference\n")
        lines.append(_md_table(cka.pivot_table(index="engine", columns="metric", values="value").round(4)))
    # L1 probe tables (mean ± std over seeds) per task
    for task in sorted(df[df.level == "L1"].task.unique()):
        sub = df[(df.level == "L1") & (df.task == task)]
        agg = sub.groupby(["engine", "metric"])["value"].agg(["mean", "std"]).reset_index()
        agg["cell"] = agg.apply(lambda r: f"{r['mean']:.4f}±{r['std']:.4f}", axis=1)
        piv = agg.pivot_table(index="engine", columns="metric", values="cell", aggfunc="first")
        lines.append(
            f"\n## L1 linear probe — {TASK_NAMES.get(task, task)} "
            f"(mean±SD over seeds; SD is a spread estimate, n=seeds, NOT a 95% CI)\n"
        )
        lines.append(_md_table(piv))
    out.write_text("\n".join(lines) + "\n")


def _emit_l2l3(engines: list[str], states: list[str]) -> None:
    print("\n# ---- L2 capacity ladder (STL sequence task) ----")
    for eng in engines:
        for st in states:
            for head in ("next_gru", "next_stan_flow"):
                print(
                    f"python scripts/train.py --task next --state {st} --engine {eng} "
                    f"--cat-head {head} --epochs 50 --folds 5 --seed 0 --no-checkpoints"
                )
    print("\n# ---- L3 MTL (deployment) — H3-alt small-state recipe; see CLAUDE.md for B9/large-state ----")
    print("# NOTE: bare defaults DO NOT reproduce paper numbers. Full canonical flags below.")
    for st in states:
        print(
            f"python scripts/train.py --task mtl --task-set check2hgi_next_region \\\n"
            f"    --state {st} --engine check2hgi --seed 0 \\\n"
            f"    --epochs 50 --folds 5 --batch-size 2048 --model mtlnet_crossattn \\\n"
            f"    --mtl-loss static_weight --category-weight 0.75 --scheduler constant \\\n"
            f"    --cat-head next_gru --reg-head next_getnext_hard \\\n"
            f"    --task-a-input-type checkin --task-b-input-type region \\\n"
            f"    --per-fold-transition-dir output/check2hgi/{st} --no-checkpoints"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding-eval ladder L0+L1 runner")
    ap.add_argument("--engines", nargs="+", required=True)
    ap.add_argument("--states", nargs="+", default=["florida"])
    ap.add_argument("--tasks", nargs="+", default=["cat", "reg"], choices=["cat", "reg", "poi"])
    ap.add_argument("--granularity", default="poi", choices=["poi", "checkin"])
    ap.add_argument("--max-items", type=int, default=None,
                    help="subsample items (needed for checkin-granularity L0)")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--silhouette-sample", type=int, default=10000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 7, 100])
    ap.add_argument("--probe-epochs", type=int, default=300,
                    help="max epochs; early-stops on a train-val slice well before this")
    ap.add_argument("--probe-lr", type=float, default=1e-2)
    ap.add_argument("--ref-engine", default="hgi", help="reference engine for CKA")
    ap.add_argument("--out", default="docs/results/embedding_eval")
    ap.add_argument("--emit-l2l3", action="store_true")
    args = ap.parse_args()

    out_dir = _root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict = {}
    all_rows: list[dict] = []
    for name in args.engines:
        eng = _engine(name)
        for state in args.states:
            print(f"[load] {name} / {state} ({args.granularity}) ...", flush=True)
            # load FULL table; subsample per-task AFTER masking (fairness, m9)
            tab = load_item_table(state, eng, granularity=args.granularity, seed=args.seeds[0])
            tables[(name, state)] = tab
            for task in args.tasks:
                print(f"  [eval] task={task}", flush=True)
                all_rows += _records_for(
                    tab, task, args.knn_k, args.silhouette_sample,
                    args.seeds, args.probe_epochs, args.probe_lr,
                    max_items=args.max_items,
                )

    if args.granularity == "poi":
        all_rows += _cka_rows(tables, args.ref_engine)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "metrics_long.csv"
    df.to_csv(csv_path, index=False)
    _write_summary(df, out_dir / "summary.md")
    print(f"\n[done] {len(df)} rows -> {csv_path}")
    print(f"[done] summary -> {out_dir / 'summary.md'}")

    if args.emit_l2l3:
        _emit_l2l3(args.engines, args.states)


if __name__ == "__main__":
    main()
