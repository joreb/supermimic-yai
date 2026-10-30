from pathlib import Path

import numpy as np
import pytest

from supermimic_yai import runner as R
from supermimic_yai.arm import load_arm
from supermimic_yai.cli import main
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import load_product
from supermimic_yai.score import score
from supermimic_yai.scripted import execute, run_token
from supermimic_yai.task import TokenEpisode
from supermimic_yai.tokens import load_recipe

ROOT = Path(__file__).parent.parent
ARM = load_arm(ROOT / "arms" / "panda")


def _setup(name="a", seed=0):
    p = load_product(ROOT / "products" / f"product_{name}")
    recipe = load_recipe(ROOT / "recipes" / f"product_{name}.json", p, ARM)
    env = AssemblyEnv(ARM, p)
    env.reset(seed=seed)
    return env, recipe


def _to_first(env, recipe, kind):
    held = None
    for i, tok in enumerate(recipe.steps):
        if tok.kind == kind:
            return tok, held, i
        _, held = run_token(env, tok, held)


def test_live_mode_no_restore():
    env, recipe = _setup()
    tok, held, _ = _to_first(env, recipe, "GRASP")
    q = env.data.qpos.copy()
    ep = TokenEpisode.live(env, tok, held, 10, 0.05)
    s = ep.reset()
    assert np.array_equal(env.data.qpos, q) and ep.held == held and s.shape == (2 * env.n_arm + 17,)


def test_held_handoff(monkeypatch):
    env, recipe = _setup()
    tok, held, i = _to_first(env, recipe, "GRASP")
    ctrl = R.LearnedController(None, 0.05)
    res, held = ctrl(env, tok, held)
    assert res.ok and held == tok.part
    seen = []
    monkeypatch.setattr(R, "run_token", lambda env, t, h: seen.append(h) or (None, h))
    ctrl(env, recipe.steps[i + 1], held)
    assert seen == [tok.part]


def test_learned_kinds_routes_scripted(monkeypatch):
    w, seen = {"w": 1}, []
    monkeypatch.setattr(R, "run_contact_token", lambda env, tok, held, weights, scale: seen.append((tok.kind, weights)) or ("r", held))
    monkeypatch.setattr(R, "run_token", lambda env, tok, held: seen.append((tok.kind, "token")) or ("r", held))
    env, recipe = _setup()
    ctrl = R.LearnedController(w, 0.05)
    for kind in ("GRASP", "ALIGN", "MOVE_ABOVE"):
        ctrl(env, next(t for t in recipe.steps if t.kind == kind), None)
    assert seen == [("GRASP", w), ("ALIGN", None), ("MOVE_ABOVE", "token")]
    assert ctrl.tags == ["learned", "scripted", "scripted"]


def test_cli_learned_requires_actor():
    a = ["run-recipe", str(ROOT / "products" / "product_a"), str(ROOT / "recipes" / "product_a.json"),
         "--arm", str(ROOT / "arms" / "panda"), "--controller", "learned"]
    assert main(a) == 2


def _wins(name, ctrl):
    n = 0
    for seed in range(5):
        env, recipe = _setup(name, seed)
        log = execute(env, recipe, controller=ctrl)
        n += log.stopped_at is None and score(env.product, env.obs()["part_poses"]).success
    return n


@pytest.mark.slow
@pytest.mark.parametrize("name", "abc")
def test_scripted_through_runner_matches_execute(name):
    assert abs(_wins(name, R.LearnedController(None, 0.05)) - _wins(name, None)) <= 1
