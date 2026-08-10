"""`StreamingClsMetrics` == the full-logit path, and == the inline loops it replaced.

Three properties are pinned here, and they are not the same property:

1. **Streamed == full** on the same device, for any batch size. Every metric is a per-row
   reduction or an additive per-class count, so batching cannot move it. Includes shapes with
   forced exact ties AT the top-k boundary, which is the only place the claim could break.
2. **Class == the inline loops it replaced.** `mtl_eval`, `mtl_cv` and `p1_region_head_ablation`
   each carried their own copy of the accumulation; those copies are reproduced verbatim below as
   golden references so the refactor is pinned rather than trusted. `mtl_cv`'s variant reduces on
   the logits' device and stores on CPU, which is a *different* thing from p1's "move the logits
   to CPU first" — the two knobs are tested separately for exactly that reason.
3. **The loop actually runs.** The p1 port shipped a read-before-assignment that killed every
   streamed run instantly, and a unit test of `_streamed_cls_metrics` could never have caught it
   because it does not execute the caller's loop. `test_driven_by_a_loop` drives the class the way
   a training loop does.
"""
import pytest
import torch

from tracking.metrics import (
    StreamingClsMetrics,
    _rank_of_target,
    _streamed_cls_metrics,
    _CARDINALITY_HAND_ROLLED_THRESHOLD,
    compute_classification_metrics,
)

KEYS = ("top5_acc", "top10_acc", "mrr", "ndcg_5", "ndcg_10",
        "accuracy", "accuracy_macro", "f1", "f1_weighted")


def _make(kind, n, c, k=10):
    """`kboundary` forces the k-th..(k+4)-th logits to be EXACTLY equal, so the top-k cut falls
    inside a tie group — the pathological case for any topk-based metric."""
    g = torch.Generator().manual_seed(0)
    if kind == "float":
        return torch.randn(n, c, generator=g)
    if kind == "ties":
        return torch.randint(0, 5, (n, c), generator=g).float()
    x = torch.randn(n, c, generator=g)
    v, _ = x.sort(dim=-1, descending=True)
    x.scatter_(1, torch.arange(k - 1, k + 4).repeat(n, 1), v[:, k - 1].unsqueeze(-1).expand(n, 5))
    return x


def _targets(n, c):
    return torch.randint(0, c, (n,), generator=torch.Generator().manual_seed(1))


# (n, c, kind, batch_size) — batch 1 and a batch that does not divide n are deliberate.
CASES = [
    (5000, 1109, "float", 512),
    (5000, 1109, "float", 4096),
    (5000, 1109, "float", 1),
    (5000, 1547, "ties", 512),
    (5000, 1109, "kboundary", 512),
    (5000, 1109, "kboundary", 997),
    (3000, 520, "float", 256),
    (3000, 6553, "float", 512),
]


@pytest.mark.parametrize("n,c,kind,bs", CASES)
def test_streamed_equals_full(n, c, kind, bs):
    logits, targets = _make(kind, n, c), _targets(n, c)
    acc = StreamingClsMetrics(c, top_k=(5, 10))
    for i in range(0, n, bs):
        acc.update(logits[i:i + bs], targets[i:i + bs])
    got = acc.compute()
    want = compute_classification_metrics(logits, targets, num_classes=c, top_k=(5, 10))
    for k in KEYS:
        assert float(got[k]) == float(want[k]), f"{k} diverged at {kind} bs={bs}"


def test_driven_by_a_loop():
    """Exercises the class through a DataLoader-shaped loop, which is what the three call sites
    do and what a direct test of `_streamed_cls_metrics` cannot reach."""
    n, c = 1024, 700
    logits, targets = _make("float", n, c), _targets(n, c)
    loader = [(logits[i:i + 128], targets[i:i + 128]) for i in range(0, n, 128)]
    acc = StreamingClsMetrics(c, top_k=(5, 10), diagnose_ties=True)
    with torch.no_grad():
        for xb, yb in loader:
            acc.update(xb, yb)
    m = acc.compute()
    assert acc.n_rows == n
    assert set(KEYS).issubset(m)
    assert 0.0 <= m["top10_acc"] <= 1.0


def test_golden_mtl_eval_inline_loop():
    """Verbatim pre-refactor `mtl_eval` S2 loop (reduce and store on the logits' device)."""
    n, c, ks = 2000, 1109, (3, 5)
    logits, targets = _make("float", n, c), _targets(n, c)
    sv_preds, sv_tgts, sv_rank = [], [], []
    sv_hit = {k: [] for k in ks}
    for i in range(0, n, 256):
        _no = logits[i:i + 256].detach()
        y_next = targets[i:i + 256]
        sv_preds.append(_no.argmax(dim=-1))
        sv_tgts.append(y_next)
        sv_rank.append(_rank_of_target(_no, y_next))
        for _k in ks:
            _ke = min(_k, _no.shape[-1])
            sv_hit[_k].append((_no.topk(_ke, dim=-1).indices == y_next.unsqueeze(-1)).any(dim=-1))
    golden = _streamed_cls_metrics(torch.cat(sv_preds), torch.cat(sv_tgts), torch.cat(sv_rank),
                                   {k: torch.cat(sv_hit[k]) for k in ks}, c, top_k=ks)

    acc = StreamingClsMetrics(c, top_k=ks)
    for i in range(0, n, 256):
        acc.update(logits[i:i + 256], targets[i:i + 256])
    assert acc.compute() == golden


def test_golden_mtl_cv_inline_loop_stores_on_cpu():
    """Verbatim pre-refactor `mtl_cv` S1 loop: reduce on the logits' device, `.cpu()` the small
    results. `store_on='cpu'` must reproduce it exactly — and must NOT be confused with moving
    the logits, which is a different (and not bit-identical) operation."""
    n, c, ks = 2000, 1109, (3, 5)
    logits, targets = _make("float", n, c), _targets(n, c)
    s1_preds, s1_tgts, s1_rank = [], [], []
    s1_hit = {k: [] for k in ks}
    for i in range(0, n, 256):
        _lb, _tb = logits[i:i + 256].detach(), targets[i:i + 256]
        s1_preds.append(_lb.argmax(dim=-1).cpu())
        s1_tgts.append(_tb.cpu())
        s1_rank.append(_rank_of_target(_lb, _tb).cpu())
        for _k in ks:
            _ke = min(_k, _lb.shape[-1])
            s1_hit[_k].append((_lb.topk(_ke, dim=-1).indices == _tb.unsqueeze(-1)).any(dim=-1).cpu())
    golden = _streamed_cls_metrics(torch.cat(s1_preds), torch.cat(s1_tgts), torch.cat(s1_rank),
                                   {k: torch.cat(s1_hit[k]) for k in ks}, c, top_k=ks)

    acc = StreamingClsMetrics(c, top_k=ks, store_on="cpu")
    for i in range(0, n, 256):
        acc.update(logits[i:i + 256], targets[i:i + 256])
    assert acc.compute() == golden


def test_store_on_cpu_is_value_preserving():
    """Parking accumulators in host memory is placement, not arithmetic."""
    n, c = 1500, 800
    logits, targets = _make("float", n, c), _targets(n, c)
    a = StreamingClsMetrics(c, top_k=(5, 10))
    b = StreamingClsMetrics(c, top_k=(5, 10), store_on="cpu")
    for i in range(0, n, 300):
        a.update(logits[i:i + 300], targets[i:i + 300])
        b.update(logits[i:i + 300], targets[i:i + 300])
    assert a.compute() == b.compute()


def test_tie_diagnostic_counts_the_right_boundary():
    """A tie forced at the 10/11 boundary must be seen by top10 and NOT by top5."""
    n, c = 400, 300
    logits, targets = _make("kboundary", n, c, k=10), _targets(n, c)
    acc = StreamingClsMetrics(c, top_k=(5, 10), diagnose_ties=True)
    acc.update(logits, targets)
    assert acc.tie_counts[10] == n
    assert acc.tie_counts[5] == 0
    assert acc.has_ties
    assert "top10 (10th==11th): 400/400 (100.0000%)" in acc.tie_summary()


def test_tie_diagnostic_clean_on_continuous_logits():
    acc = StreamingClsMetrics(1109, top_k=(5, 10), diagnose_ties=True)
    acc.update(_make("float", 2000, 1109), _targets(2000, 1109))
    assert not acc.has_ties


def test_tie_diagnostic_skips_boundaries_past_c():
    """C smaller than k+1 has no k-boundary to tie at; it must not crash or miscount."""
    acc = StreamingClsMetrics(4, top_k=(5, 10), diagnose_ties=True)
    acc.update(torch.randn(50, 4), torch.randint(0, 4, (50,)))
    assert acc.tie_counts == {5: 0, 10: 0}


def test_should_stream_tracks_the_shared_threshold():
    """The gate must move with `compute_classification_metrics`' torchmetrics cutoff, not with a
    literal re-typed per call site (there were four such literals before this class)."""
    t = _CARDINALITY_HAND_ROLLED_THRESHOLD
    assert StreamingClsMetrics.should_stream(t + 1)
    assert not StreamingClsMetrics.should_stream(t)
    assert not StreamingClsMetrics.should_stream(7)
    assert not StreamingClsMetrics.should_stream(None)
