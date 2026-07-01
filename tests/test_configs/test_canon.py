"""Regression guard for the ``--canon`` version selector (src/configs/canon.py + train.py).

The whole point of ``--canon`` is that flipping the default (v16=G) does NOT lose the ability
to reproduce v11..v15. This test pins each bundle field-by-field through the REAL train.py
arg parser, so a future default change that silently breaks a version's reproduction fails CI.

If you intentionally change a frozen version's recipe, you have introduced a reproduction bug:
canon bundles are append-only (add a new version), never edit a frozen one.
"""

import importlib.util
from pathlib import Path

import pytest

_TRAIN = Path(__file__).resolve().parents[2] / "scripts" / "train.py"


def _load_train():
    spec = importlib.util.spec_from_file_location("_train_for_canon_test", _TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def train():
    return _load_train()


def _parse(train, ver, *extra):
    base = ["--task", "mtl", "--state", "florida", "--seed", "42"]
    if ver is not None:
        base += ["--canon", ver]
    return train._parse_args(base + list(extra))


# Expected canon-controlled fields per version (the documented recipe, CANONICAL_VERSIONS).
EXPECTED = {
    "v11": dict(model_name="mtlnet_crossattn", engine="check2hgi", scheduler="cosine",
                log_t_kd_weight=0.0, checkpoint_selector="joint_f1_mean",
                use_class_weights_reg=True, use_class_weights_cat=True,
                alternating_optimizer_step=True, reg_head="next_getnext_hard"),
    "v12": dict(model_name="mtlnet_crossattn", engine="check2hgi", scheduler="cosine",
                log_t_kd_weight=0.2, checkpoint_selector="geom_simple",
                use_class_weights_reg=True, use_class_weights_cat=True,
                reg_head="next_getnext_hard"),
    "v15": dict(model_name="mtlnet_crossattn", engine="check2hgi", scheduler="cosine",
                log_t_kd_weight=0.2, checkpoint_selector="geom_simple",
                use_class_weights_reg=False, use_class_weights_cat=False,
                reg_head="next_getnext_hard"),
    "v16": dict(model_name="mtlnet_crossattn_dualtower", engine="check2hgi_design_k_resln_mae_l0_1",
                scheduler="onecycle", log_t_kd_weight=0.0, checkpoint_selector="geom_simple",
                use_class_weights_reg=False, use_class_weights_cat=False,
                reg_head="next_stan_flow_dualtower"),
    # v17 — champion candidate: v16 + bs8192 + --onecycle-per-head-lr (per-head cat-lr 1e-3 effective).
    "v17": dict(model_name="mtlnet_crossattn_dualtower", engine="check2hgi_design_k_resln_mae_l0_1",
                scheduler="onecycle", log_t_kd_weight=0.0, checkpoint_selector="geom_simple",
                use_class_weights_reg=False, use_class_weights_cat=False,
                reg_head="next_stan_flow_dualtower", batch_size=8192,
                onecycle_per_head_lr=True),
}


@pytest.mark.parametrize("ver", ["v11", "v12", "v15", "v16", "v17"])
def test_canon_bundle_resolves_to_documented_recipe(train, ver):
    args = _parse(train, ver)
    for field, want in EXPECTED[ver].items():
        got = getattr(args, field)
        assert got == want, f"--canon {ver}: {field} = {got!r}, expected {want!r}"
    assert getattr(args, "_canon_active") is True


def test_default_canon_is_v17(train):
    """No --canon flag → the champion recipe (v17 = v16 + bs8192 + per-head cat-lr) is the default."""
    args = train._parse_args(["--task", "mtl", "--state", "florida", "--seed", "42"])
    assert args.canon == "v17"
    assert args.model_name == "mtlnet_crossattn_dualtower"
    assert args.scheduler == "onecycle"
    assert args.batch_size == 8192
    assert args.onecycle_per_head_lr is True


def test_explicit_flag_overrides_bundle(train):
    """Precedence: explicit flags win over the injected bundle (argparse last-wins)."""
    args = _parse(train, "v16", "--engine", "hgi", "--max-lr", "0.001")
    assert args.engine == "hgi"           # user override beats v16's v14 engine
    assert float(args.max_lr) == 0.001    # user override beats the bundle's 3e-3
    assert args.model_name == "mtlnet_crossattn_dualtower"  # un-overridden bundle field stays


def test_canon_none_disables_injection(train):
    """--canon none → no bundle; bare smoke defaults (not the dual-tower)."""
    args = train._parse_args(["--task", "mtl", "--state", "florida", "--seed", "42", "--canon", "none"])
    assert args._canon_active is False
    assert args.model_name != "mtlnet_crossattn_dualtower"


def test_canon_scoped_to_mtl(train):
    """--canon is a no-op for non-mtl tasks (the versions are MTL recipes)."""
    args = train._parse_args(["--task", "next", "--state", "florida", "--canon", "v16"])
    assert args._canon_active is False
    assert args.model_name != "mtlnet_crossattn_dualtower"
