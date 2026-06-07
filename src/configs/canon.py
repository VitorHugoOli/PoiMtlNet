"""Canonical version recipe bundles for the ``--canon`` selector (traceback to v11..v16).

The ``mtl_improvement`` study repeatedly flipped train.py's effective recipe (v11 paper
canon → v12 log_T-KD → v15 C25-unweighted → v16 champion **G**). Each version is a *bundle*
of CLI flags. ``--canon vXX`` (default **v16**, the champion) injects that bundle BEFORE the
user's own flags so that **explicit flags always override the bundle** (argparse "last value
wins" for store actions). This makes the champion the default while keeping every prior
version reproducible forever with a single ``--canon vNN``.

Scope: ``--canon`` only injects for ``--task mtl`` (the versions are MTL recipes) and is a
no-op when ``--config`` is given. ``--canon none`` disables injection (bare smoke defaults).

**Append-only.** Never edit a frozen bundle (it would silently break that version's
reproduction). Add a new key when a new version is pinned, and update
``docs/results/CANONICAL_VERSIONS.md`` + ``tests/test_configs/test_canon.py``.

Authoritative source for each bundle: ``docs/results/CANONICAL_VERSIONS.md`` (the §vNN blocks
and reproduction maps) + ``docs/NORTH_STAR.md`` (the B9 invocation). The regression test
asserts each bundle resolves to the documented config field-by-field.
"""

from __future__ import annotations

from typing import List

DEFAULT_CANON = "v16"

# The shared cross-attn "B9" recipe underlying v11/v12/v15 (NORTH_STAR §Champion / CANONICAL_VERSIONS §v12).
_CROSSATTN_B9: List[str] = [
    "--task-set", "check2hgi_next_region",
    "--model", "mtlnet_crossattn",
    "--mtl-loss", "static_weight", "--category-weight", "0.75",
    "--scheduler", "cosine", "--max-lr", "3e-3",
    "--cat-lr", "1e-3", "--reg-lr", "3e-3", "--shared-lr", "1e-3",
    "--alternating-optimizer-step", "--alpha-no-weight-decay", "--min-best-epoch", "5",
    "--cat-head", "next_gru", "--reg-head", "next_getnext_hard",
    "--task-a-input-type", "checkin", "--task-b-input-type", "region",
    "--engine", "check2hgi",                       # frozen GCN paper substrate
    "--folds", "5", "--epochs", "50", "--batch-size", "2048",
]

CANON_BUNDLES: dict[str, List[str]] = {
    # v11 — BRACIS paper canon: GCN substrate, log_T-KD OFF, CLASS-WEIGHTED (pre-C25),
    # old joint selector (0.5*(cat+reg)). CANONICAL_VERSIONS §v11 + §v12 reproduction map.
    "v11": _CROSSATTN_B9 + [
        "--log-t-kd-weight", "0.0",
        "--checkpoint-selector", "joint_f1_mean",
        "--reg-class-weights", "--cat-class-weights",
    ],
    # v12 — v11 + log_T-KD W=0.2 ON; still class-WEIGHTED (predates C25); geom_simple selector (default).
    "v12": _CROSSATTN_B9 + [
        "--log-t-kd-weight", "0.2",
        "--reg-class-weights", "--cat-class-weights",
    ],
    # v15 — v12 recipe + the C25 fix: BOTH heads UNWEIGHTED (the current bare code default).
    "v15": _CROSSATTN_B9 + [
        "--log-t-kd-weight", "0.2",
        "--no-reg-class-weights", "--no-cat-class-weights",
    ],
    # v16 — CHAMPION "G": reg-private dual-tower (aux fusion, α·log_T prior OFF), v14 substrate,
    # onecycle (NO alt-opt), unweighted, KD OFF, geom_simple selector. CANONICAL_VERSIONS §v16.
    "v16": [
        "--task-set", "check2hgi_next_region",
        "--model", "mtlnet_crossattn_dualtower",
        "--reg-head", "next_stan_flow_dualtower",
        "--reg-head-param", "raw_embed_dim=64",
        "--reg-head-param", "fusion_mode=aux",
        "--reg-head-param", "freeze_alpha=True",
        "--reg-head-param", "alpha_init=0.0",
        "--cat-head", "next_gru",
        "--mtl-loss", "static_weight", "--category-weight", "0.75",
        "--scheduler", "onecycle", "--max-lr", "3e-3",
        "--cat-lr", "1e-3", "--reg-lr", "3e-3", "--shared-lr", "1e-3",
        "--log-t-kd-weight", "0.0",
        "--no-reg-class-weights", "--no-cat-class-weights",
        "--task-a-input-type", "checkin", "--task-b-input-type", "region",
        "--engine", "check2hgi_design_k_resln_mae_l0_1",   # v14 substrate
        "--folds", "5", "--epochs", "50", "--batch-size", "2048",
    ],
}

CANON_CHOICES = sorted(CANON_BUNDLES) + ["none"]


def resolve_canon_argv(canon: str | None, argv: List[str]) -> List[str]:
    """Return the effective argv = ``bundle(canon) + argv``.

    The bundle is PREPENDED so the user's own flags (later in argv) override it via
    argparse last-wins. ``canon`` in ``{None, "none"}`` returns argv unchanged.
    """
    if canon in (None, "none"):
        return list(argv)
    if canon not in CANON_BUNDLES:
        raise SystemExit(
            f"--canon: unknown version {canon!r}; choose from {sorted(CANON_BUNDLES)} or 'none'"
        )
    return list(CANON_BUNDLES[canon]) + list(argv)


__all__ = ["DEFAULT_CANON", "CANON_BUNDLES", "CANON_CHOICES", "resolve_canon_argv"]
