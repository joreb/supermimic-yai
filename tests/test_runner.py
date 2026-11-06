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
    assert ep.z0 == env.obs()["part_poses"][tok.part][2]
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
    ctrl = R.LearnedController(w, 0.05, ("GRASP",))
    for kind in ("GRASP", "ALIGN", "MOVE_ABOVE"):
        ctrl(env, next(t for t in recipe.steps if t.kind == kind), None)
    assert seen == [("GRASP", w), ("ALIGN", None), ("MOVE_ABOVE", "token")]
    assert ctrl.tags == ["learned", "scripted", "scripted"]


class _FakeRenderer:
    def update_scene(self, data):
        pass

    def render(self):
        return np.zeros((2, 2, 3), np.uint8)


def test_learned_path_timeout_and_frames(monkeypatch):
    import torch
    from supermimic_yai.policy import Actor
    env, recipe = _setup()
    tok, held, _ = _to_first(env, recipe, "GRASP")
    torch.manual_seed(0)
    n = env.n_arm
    w = Actor(2 * n + 17, 3, n + 1, 3, 4, 8, 1).export()
    monkeypatch.setattr(R, "budget_steps", lambda env, tok: 5)
    env._rec = {"r": _FakeRenderer(), "frames": [], "every": 1, "i": 0}
    res, h = R.run_contact_token(env, tok, held, w, 0.05)
    assert not res.ok and res.reason == "timeout" and h == held
    assert res.sim_time == pytest.approx(5 * env.substeps * env.model.opt.timestep)
    assert len(env._rec["frames"]) == 5


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


def test_held_move_converges_when_part_slips_in_grip(monkeypatch):
    """A part that yaws ~3.5 deg in the grip mid-move must still be driven to the target (tf is live, not stale)."""
    import mujoco
    from supermimic_yai import scripted as S
    env, recipe = _setup()
    held = None
    for tok in recipe.steps[:4]:  # MOVE_TO, MOVE_TO, GRASP, RETREAT
        res, held = run_token(env, tok, held)
        assert res.ok
    adr, tick, n = env.index.part_qpos_adr[held], S._Run.tick, [0]

    def slip(self, *a, **k):
        tick(self, *a, **k)
        n[0] += 1
        if n[0] == 10:
            q = np.empty(4)
            mujoco.mju_mulQuat(q, [np.cos(np.radians(1.75)), 0, 0, np.sin(np.radians(1.75))], env.data.qpos[adr + 3:adr + 7])
            env.data.qpos[adr + 3:adr + 7] = q
            mujoco.mj_forward(env.model, env.data)

    monkeypatch.setattr(S._Run, "tick", slip)
    res, held = run_token(env, recipe.steps[4], held)
    assert res.ok, res.reason


def test_learned_kinds_default_to_actor_kinds(monkeypatch):
    w, seen = {"kinds": ("GRASP", "ALIGN")}, []
    monkeypatch.setattr(R, "run_contact_token", lambda env, tok, held, weights, scale: seen.append((tok.kind, weights)) or ("r", held))
    env, recipe = _setup()
    ctrl = R.LearnedController(w, 0.05)
    for kind in ("GRASP", "ALIGN", "INSERT"):
        ctrl(env, next(t for t in recipe.steps if t.kind == kind), None)
    assert seen == [("GRASP", w), ("ALIGN", w), ("INSERT", None)]
    assert R.LearnedController(w, 0.05, ("INSERT",)).kinds == ("INSERT",)
