import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supermimic_yai import seeds
from supermimic_yai.arm import load_arm
from supermimic_yai.collect import Collector
from supermimic_yai.config import load_config
from supermimic_yai.product import load_product
from supermimic_yai.replay import save_episodes
from supermimic_yai.startbank import collect_start_states, resolve_recipe, save_bank
from supermimic_yai.task import TOKEN_KINDS
from supermimic_yai.tokens import load_recipe

ROOT = Path(__file__).parent.parent
ARM_DIR = ROOT / "arms" / "panda"
A, B = str(ROOT / "products" / "product_a"), str(ROOT / "products" / "product_b")
TINY = ["policy.width=16", "policy.depth=1", "policy.batch=32", "policy.bc_iters=3", "critic.width=16", "critic.depth=1",
        "wm.width=16", "wm.depth=1", "wm.batch=32", "collect.workers=2", "replay.capacity=20000",
        "train.episodes_per_iter=2", "train.min_updates=1", "train.max_updates=2", "train.log_every=1",
        "train.eval_every=2", "train.eval_n=1", "train.eval_recipe_seeds=1", "train.video_every=1000", "train.ckpt_every=1"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("train")
    arm = load_arm(ARM_DIR)
    product = load_product(Path(A))
    recipe = load_recipe(resolve_recipe(A), product, arm)
    save_bank(d / "bank.npz", collect_start_states(arm, product, recipe, seeds.TRAIN[:1]))
    cfg = load_config("laptop")
    cfg["collect"]["workers"] = 2
    with Collector(cfg, ARM_DIR, [A], d / "bank.npz") as c:
        eps = c.collect(8, None)
    init = d / "init"
    init.mkdir()
    save_episodes(init / "episodes.npz", eps)
    import torch
    from supermimic_yai.wm import WorldModel
    sd, ad, cd = eps[0]["s"].shape[1], eps[0]["a"].shape[1], eps[0]["ctx"].shape[0]
    m = WorldModel(sd, ad, cd, len(TOKEN_KINDS), 16, 1)
    ok = np.concatenate([e["s"][:-1] for e in eps]), np.concatenate([e["s"][1:] for e in eps])
    m.set_norm(ok[0], ok[1], np.concatenate([np.tile(e["ctx"], (len(e["a"]), 1)) for e in eps]))
    torch.save(m.state_dict(), init / "model.pt")
    return d


def args(setup, tmp, iters, *extra, products=(A,), heldout=(B,)):
    a = ["--config", "laptop", "--init", str(setup / "init"), "--products", *products, "--bank", str(setup / "bank.npz"),
         "--eval-banks", str(setup / "bank.npz"), "--heldout", *heldout, "--arm", str(ARM_DIR), "--logdir", str(tmp), "--iters", str(iters)]
    for t in TINY:
        a += ["--set", t]
    return a + list(extra)


def test_train_smoke(setup, tmp_path):
    from supermimic_yai.train import main
    assert main(args(setup, tmp_path, 2)) == 2
    for f in ("ckpt.pt", "actor.npz", "actor_bc.npz"):
        assert (tmp_path / f).exists()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(tmp_path))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    assert {"policy/loss", "sys/utd", "wm/loss", "critic/loss", "sys/collect_s", "eval/recipe_success/product_a",
            "eval/recipe_success/product_b", "eval/token_success/product_a/GRASP"} <= tags
    assert any(t.startswith("policy/imagined_reward_sum/") for t in tags)
    assert any(t.startswith("curriculum/weight/") for t in tags)
    assert "mujoco.viewer" not in sys.modules


def _scalar_steps(path, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    return [e.step for e in acc.Scalars(tag)]


def test_resume_continues_iteration(setup, tmp_path):
    import torch
    from supermimic_yai.train import main
    assert main(args(setup, tmp_path, 1, "--set", "train.eval_every=1000")) == 1
    assert main(args(setup, tmp_path, 2, "--set", "train.eval_every=1000", "--resume")) == 2
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["it"] == 2
    assert _scalar_steps(tmp_path, "wm/loss") == [1, 2]  # without --resume iteration 1 would run twice


def test_rejects_eval_product_in_training(setup, tmp_path):
    from supermimic_yai.train import main
    with pytest.raises(SystemExit, match="product_b"):
        main(args(setup, tmp_path, 1, products=(A, B)))


def test_rejects_heldout_state_in_bank(setup, tmp_path):
    import dataclasses
    from supermimic_yai.startbank import load_bank
    from supermimic_yai.train import main
    save_bank(tmp_path / "bank.npz", [dataclasses.replace(s, product="product_b") for s in load_bank(setup / "bank.npz")[:2]])
    a = args(setup, tmp_path, 1)
    a[a.index("--bank") + 1] = str(tmp_path / "bank.npz")
    with pytest.raises(SystemExit, match="--bank"):
        main(a)


def test_abort_keeps_last_good_ckpt(setup, tmp_path, monkeypatch):
    import torch
    from supermimic_yai import improve, train
    assert train.main(args(setup, tmp_path, 1, "--set", "train.eval_every=1000")) == 1
    good = (tmp_path / "ckpt.pt").read_bytes()
    monkeypatch.setattr(improve, "out_of_bounds", lambda m, h: True)
    monkeypatch.setattr(train, "ABORT_STREAK", 1)
    with pytest.raises(SystemExit, match="ABORT"):
        train.main(args(setup, tmp_path, 3, "--set", "train.eval_every=1000", "--resume"))
    assert (tmp_path / "actor_abort.npz").exists()
    assert (tmp_path / "ckpt.pt").read_bytes() == good
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["it"] == 1


def test_import_pulls_no_torch():
    out = subprocess.run([sys.executable, "-c", "import sys, supermimic_yai.train; print('torch' in sys.modules, 'mujoco.viewer' in sys.modules)"],
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]
