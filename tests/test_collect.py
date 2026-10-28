from pathlib import Path

import numpy as np
import pytest

from supermimic_yai import seeds
from supermimic_yai.arm import load_arm
from supermimic_yai.collect import Collector
from supermimic_yai.config import load_config
from supermimic_yai.product import load_product
from supermimic_yai.replay import ReplayBuffer
from supermimic_yai.startbank import collect_start_states, resolve_recipe, save_bank
from supermimic_yai.task import TOKEN_KINDS
from supermimic_yai.tokens import load_recipe

ROOT = Path(__file__).parent.parent
ARM_DIR = ROOT / "arms" / "panda"
A = ROOT / "products" / "product_a"


@pytest.fixture(scope="module")
def collector(tmp_path_factory):
    arm = load_arm(ARM_DIR)
    product = load_product(A)
    bank = collect_start_states(arm, product, load_recipe(resolve_recipe(A), product, arm), seeds.TRAIN[:1])
    path = tmp_path_factory.mktemp("bank") / "bank.npz"
    save_bank(path, bank)
    cfg = load_config("laptop")
    cfg["collect"]["workers"] = 2
    with Collector(cfg, ARM_DIR, [A], path) as c:
        yield c, cfg, arm


def test_collector_no_torch(collector):
    c, _, _ = collector
    assert not any(c.pool.map(__import__("supermimic_yai.collect", fromlist=["_has_torch"])._has_torch, range(4)))


def test_scripted_seed_data_shapes(collector):
    c, cfg, arm = collector
    eps = c.collect(4, None, rng_seed=0)
    assert len(eps) == 4
    n = len(arm.arm_joints)
    for e in eps:
        T = len(e["a"])
        assert e["s"].shape == (T + 1, 2 * n + 17) and e["a"].shape == (T, n + 1)
        assert e["r"].shape == e["done"].shape == e["unstable"].shape == (T,)
        assert e["goal"].shape == (3,) and e["ctx"].shape == (2 + n,)
        assert e["done"][-1] and not e["done"][:-1].any()
        assert e["terminal_kind"] in ("success", "unstable", "dropped", "force", "timeout")
        assert TOKEN_KINDS[e["kind"]] and e["product"]
        assert np.abs(e["a"][:, :n]).max() <= 1.0
    buf = ReplayBuffer(10_000, 2 * n + 17, n + 1, 3, 2 + n)
    for e in eps:
        buf.add(e)
    assert buf.size == sum(len(e["a"]) for e in eps)
