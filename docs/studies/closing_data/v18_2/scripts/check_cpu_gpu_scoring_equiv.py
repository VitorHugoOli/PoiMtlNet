#!/usr/bin/env python3
"""CPU-vs-CUDA equivalence for p1's val scoring, on the card that will actually run the cells.

WHY THIS EXISTS
  `MTL_CHUNK_VAL_METRIC=1` forces p1 to move the full [N_val x C] val logit to CPU and score it
  there. Dropping that flag for small states returns them to GPU scoring, so the two paths must
  agree at reporting precision. The repo's own test
  (tests/test_scripts/test_p1_val_chunk_guard.py::TestCpuGpuEquivalence) asserts this but is
  `skipif(not torch.cuda.is_available())` and therefore never runs on the dev box.

WHAT THE FIRST VERSION OF THIS SCRIPT GOT WRONG (Fable review, 2026-08-10) — all fixed here:
  1. It asked for the key `f1_macro`; compute_classification_metrics emits `f1`
     (src/tracking/metrics.py:180). Guarded by `if k in m_cpu`, so **f1 was silently skipped** and
     the run reported coverage it did not have. Now: the key list is asserted against what the
     function actually returns, and a missing key is a hard failure, never a skip.
  2. It reported the margin as 3350x. Reporting precision is 4 dp on a PERCENTAGE = 1e-4 on the
     0-100 scale = **1e-6 on the 0-1 scale these metrics live on**. The real margin is ~34x.
  3. The headline reg metric is `top10_acc`, and `_top_k_accuracy` (metrics.py:102-110) uses
     `logits.topk(k).indices` — whose tie-break AT THE K-BOUNDARY is a kernel/arch detail, NOT the
     device-independent strict-`>` rank used by mrr/ndcg. So top-k needs *empirical* evidence on
     the arch that will run it, not a by-construction argument. Hence the boundary-tie case below
     and the --gpu flag.
"""
import sys, time, argparse

sys.path.insert(0, "/data/repo/src")
sys.path.insert(0, "src")
import torch
from tracking.metrics import compute_classification_metrics

REPORTED = ("top10_acc", "top5_acc", "mrr", "ndcg_5", "ndcg_10",
            "accuracy", "accuracy_macro", "f1", "f1_weighted")
# 4 dp on a percentage == 1e-4 on 0-100 == 1e-6 on the 0-1 scale the metrics use.
BAR = 1e-6


def make(kind, n, c, k=10):
    g = torch.Generator().manual_seed(0)
    if kind == "float":
        return torch.randn(n, c, generator=g)
    if kind == "ties":                      # heavy exact ties everywhere
        return torch.randint(0, 5, (n, c), generator=g).float()
    if kind == "kboundary":
        # Worst case for topk: force FIVE columns to exactly the k-th largest value in every row,
        # so the k-th and (k+1)-th scores tie and which index lands inside top-k is decided purely
        # by the kernel's tie-break. Verified below: 100% of rows get an exact boundary tie.
        x = torch.randn(n, c, generator=g)
        v, _ = x.sort(dim=-1, descending=True)
        tie = v[:, k - 1].unsqueeze(-1)
        x.scatter_(1, torch.arange(k - 1, k + 4).repeat(n, 1), tie.expand(n, 5))
        return x
    raise ValueError(kind)


def boundary_tie_frac(x, k=10):
    """Fraction of rows whose k-th and (k+1)-th largest scores are EXACTLY equal.

    Reported per case so the evidence is auditable: a 'tie stress' case that produces no ties
    would make the whole run vacuous, which is exactly the sort of silent hole the first version
    of this script shipped with."""
    v, _ = x.sort(dim=-1, descending=True)
    return float((v[:, k - 1] == v[:, k]).float().mean())


def topk_setdiff(cpu_logits, gpu_logits, k=10):
    """Rows where the top-k INDEX SET differs between devices. This isolates the tie-break
    mechanism from the metric: a nonzero count here with a zero metric delta means the kernels
    disagree about WHICH tied class is in the set, but the target happened not to be one of
    them — informative, and not something the metric alone would reveal."""
    a = cpu_logits.topk(k, dim=-1).indices.sort(dim=-1).values
    b = gpu_logits.topk(k, dim=-1).indices.sort(dim=-1).values.cpu()
    return int((a != b).any(dim=-1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-label", default="?")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA"
    dev = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__} | {dev} | requested label {args.gpu_label}")

    # Fail loudly if the metric dict does not contain exactly what we think it does.
    probe = compute_classification_metrics(torch.randn(64, 50), torch.randint(0, 50, (64,)),
                                           num_classes=50, top_k=(5, 10))
    missing = [k for k in REPORTED if k not in probe]
    assert not missing, f"metric keys missing from compute_classification_metrics: {missing}"
    print(f"  key check: all {len(REPORTED)} reported metrics present\n")

    CASES = [
        (20_000, 1109, "float",     "alabama-scale C"),
        (40_179, 1547, "float",     "ARIZONA val fold (measured)"),
        (54_334,  520, "float",     "ISTANBUL val fold (measured C=520)"),
        (40_000, 6553, "float",     "texas-scale C"),
        (20_000, 1109, "ties",      "heavy exact ties"),
        (40_179, 1547, "ties",      "ARIZONA, heavy ties"),
        (20_000, 1109, "kboundary", "TOP-K BOUNDARY ties (worst case for topk)"),
        (40_179, 1547, "kboundary", "ARIZONA, top-k boundary ties"),
    ]
    worst, worst_where, fails = 0.0, "", 0
    for n, c, kind, label in CASES:
        logits = make(kind, n, c)
        targets = torch.randint(0, c, (n,), generator=torch.Generator().manual_seed(1))
        t0 = time.time()
        m_cpu = compute_classification_metrics(logits.cpu(), targets.cpu(), num_classes=c, top_k=(5, 10))
        t_cpu = time.time() - t0
        lg, tg = logits.cuda(), targets.cuda()
        torch.cuda.synchronize(); t0 = time.time()
        m_gpu = compute_classification_metrics(lg, tg, num_classes=c, top_k=(5, 10))
        torch.cuda.synchronize(); t_gpu = time.time() - t0
        btf = boundary_tie_frac(logits)
        sd = topk_setdiff(logits.cpu(), lg)
        mx, mk = 0.0, ""
        for k in REPORTED:
            d = abs(float(m_cpu[k]) - float(m_gpu[k]))   # KeyError here is a real failure
            if d > mx:
                mx, mk = d, k
            if d > BAR:
                fails += 1
                print(f"    !! OVER BAR {k}: cpu={m_cpu[k]!r} gpu={m_gpu[k]!r} d={d:.3e}")
        if mx > worst:
            worst, worst_where = mx, f"{label}/{mk}"
        print(f"  {label:42} N={n:>6} C={c:>5} boundary_ties={btf:5.1%} "
              f"topk_setdiff={sd:>6}/{n}  max|Δ|={mx:.3e} ({mk:12}) "
              f"cpu {t_cpu:5.2f}s gpu {t_gpu:5.2f}s")

    print(f"\n  worst deviation      : {worst:.3e}   at {worst_where}")
    print(f"  reporting quantum    : {BAR:.0e}  (4 dp on a percentage = 1e-6 on the 0-1 scale)")
    print(f"  margin               : {BAR/worst:.0f}x" if worst else "  margin: exact")
    print(f"  VERDICT: {'PASS' if fails == 0 else 'FAIL'}  ({fails} metric-case(s) over the bar)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
