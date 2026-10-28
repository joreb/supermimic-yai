from pathlib import Path

import json
import shutil

import numpy as np
import pytest

from supermimic_yai import seeds
from supermimic_yai.arm import load_arm
from supermimic_yai import collect as C
from supermimic_yai.collect import Collector
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.config import load_config
from supermimic_yai.product import load_product
from supermimic_yai.replay import ReplayBuffer
from supermimic_yai.startbank import collect_start_states, resolve_recipe, save_bank
from supermimic_yai.task import TOKEN_KINDS
from supermimic_yai.tokens import load_recipe

ROOT = Path(__file__).parent.parent
ARM_DIR = ROOT / "arms" / "panda"
A = ROOT / "products" / "product_a"


ARM = load_arm(ARM_DIR)


def _bank(product_dir):
    product = load_product(product_dir)
    return product, load_recipe(resolve_recipe(product_dir), product, ARM), None


@pytest.fixture(scope="module")
def a_bank(tmp_path_factory):
    product, recipe, _ = _bank(A)
    bank = collect_start_states(ARM, product, recipe, seeds.TRAIN[:1])
    path = tmp_path_factory.mktemp("bank") / "bank.npz"
    save_bank(path, bank)
    return product, recipe, bank, path


@pytest.fixture(scope="module")
def collector(a_bank):
    cfg = load_config("laptop")
    cfg["collect"]["workers"] = 2
    with Collector(cfg, ARM_DIR, [A], a_bank[3]) as c:
        yield c, cfg, ARM


def test_collector_no_torch(collector, a_bank):
    c, _, _ = collector
    job = [(next(s for s in a_bank[2] if s.token == "ALIGN"), 0)]
    # each task runs a real episode, then holds its worker so every worker takes part
    got = c.pool.map(C._probe, [(job, None, 0.0)] * 6, chunksize=1)
    assert len({pid for pid, *_ in got}) == 2
    assert not any(torch for _, torch, *_ in got)


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
        assert not e["done"][:-1].any()
        assert e["terminal_kind"].split(":")[0] in ("success", "unstable", "dropped", "force", "timeout",
                                                    "script_abort", "script_ok_unregistered")
        assert e["done"][-1] == (e["terminal_kind"] in ("success", "unstable", "dropped", "force"))
        assert TOKEN_KINDS[e["kind"]] and e["product"]
        assert np.abs(e["a"][:, :n]).max() <= 1.0
        assert set(np.unique(e["a"][:, n])) <= {0.0, 1.0}  # no noise on the gripper
        assert e["noise"] in cfg["collect"]["action_noise"]
    buf = ReplayBuffer(10_000, 2 * n + 17, n + 1, 3, 2 + n)
    for e in eps:
        buf.add(e)
    assert buf.size == sum(len(e["a"]) for e in eps)


def test_noise_is_sampled_from_list(collector):
    c, cfg, _ = collector
    eps = c.collect(12, None, rng_seed=1)
    assert {e["noise"] for e in eps} <= set(cfg["collect"]["action_noise"]) and len({e["noise"] for e in eps}) > 1


def _tok(a_bank, kind):
    _, recipe, bank, _ = a_bank
    st = next(s for s in bank if s.token == kind)
    return AssemblyEnv(ARM, a_bank[0]), recipe.steps[st.step], st


def test_script_abort_is_truncation(a_bank, monkeypatch):
    env, tok, st = _tok(a_bank, "GRASP")
    monkeypatch.setattr("supermimic_yai.scripted._budget", lambda env, tok: 0.05)  # scripted gives up, episode budget intact
    e = C.run_episode(env, tok, st, 0.05, 0.0, np.random.default_rng(0))
    assert e["terminal_kind"] == "script_abort:timeout" and not e["done"].any()


def test_script_ok_unregistered(a_bank, monkeypatch):
    env, tok, st = _tok(a_bank, "ALIGN")
    monkeypatch.setattr("supermimic_yai.task.success", lambda *a: False)
    e = C.run_episode(env, tok, st, 0.05, 0.0, np.random.default_rng(0))
    assert e["terminal_kind"] == "script_ok_unregistered" and not e["done"].any()


def test_policy_gripper_clipped_and_noiseless(a_bank):
    env, tok, st = _tok(a_bank, "ALIGN")
    n = env.n_arm
    for b, want in ((5.0, np.tanh(5.0)), (-0.5, 0.0)):
        w = [(np.zeros((2 * n + 17 + 3, n + 1)), np.append(np.zeros(n), b))]
        e = C.run_episode(env, tok, st, 0.05, 0.5, np.random.default_rng(0), w)
        assert (e["a"][:, n] == want).all() and (np.abs(e["a"][:, :n]) <= 1).all()


def test_timeout_is_truncation(a_bank, monkeypatch):
    env, tok, st = _tok(a_bank, "ALIGN")
    n = env.n_arm
    monkeypatch.setattr(C, "budget_steps", lambda env, tok: 3)
    w = [(np.zeros((2 * n + 17 + 3, n + 1)), np.zeros(n + 1))]
    for weights in (w, None):  # policy loop and scripted recorder both stop on terminal_kind
        e = C.run_episode(env, tok, st, 0.05, 0.0, np.random.default_rng(0), weights)
        assert e["terminal_kind"] == "timeout" and len(e["a"]) == 3 and not e["done"].any()


def test_env_lru_cap(a_bank, tmp_path):
    product, recipe, bank, path = a_bank
    d = tmp_path / "products" / "copy_b"
    shutil.copytree(A, d)
    m = json.loads((d / "assembly.json").read_text())
    m["name"] = "copy_b"
    (d / "assembly.json").write_text(json.dumps(m))
    (tmp_path / "recipes").mkdir()
    r = json.loads((ROOT / "recipes" / "product_a.json").read_text())
    r["product"] = "copy_b"
    (tmp_path / "recipes" / "copy_b.json").write_text(json.dumps(r))
    p2, r2, _ = _bank(d)
    save_bank(tmp_path / "bank.npz", bank + collect_start_states(ARM, p2, r2, seeds.TRAIN[:1], kinds=("ALIGN",)))
    cfg = load_config("laptop")
    cfg["collect"].update(workers=2, envs_per_worker=1)
    with Collector(cfg, ARM_DIR, [A, d], tmp_path / "bank.npz") as c:
        eps = c.collect(12, None, rng_seed=0)
        assert {e["product"] for e in eps} == {product.name, "copy_b"}
        job = [(bank[0], 0)]
        got = c.pool.map(C._probe, [(job, None, 0.0)] * 6, chunksize=1)
        assert all(n <= 1 and peak <= 1 for _, _, n, peak in got)
