from pathlib import Path

import numpy as np
import pytest

from supermimic_yai import seeds
from supermimic_yai.arm import load_arm
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import load_product
from supermimic_yai import task
from supermimic_yai.scripted import _yaw, run_token
from supermimic_yai.startbank import collect_start_states
from supermimic_yai.task import (TokenEpisode, involved_context, reward, success, token_goal, token_state)
from supermimic_yai.tokens import load_recipe

ROOT = Path(__file__).parent.parent
ARM = load_arm(ROOT / "arms" / "panda")


def _load(name):
    product = load_product(ROOT / "products" / name)
    return product, load_recipe(ROOT / "recipes" / f"{name}.json", product, ARM)


@pytest.fixture(scope="module")
def a():
    product, recipe = _load("product_a")
    return product, recipe, collect_start_states(ARM, product, recipe, [seeds.TRAIN[0]])


def test_state_fixed_size():
    sizes = set()
    for name in ("product_a", "product_b", "product_c"):
        product, recipe = _load(name)
        env = AssemblyEnv(ARM, product)
        obs, _ = env.reset(seed=0)
        for kind in ("GRASP", "INSERT"):
            tok = next(t for t in recipe.steps if t.kind == kind)
            held = None if kind == "GRASP" else tok.part
            s = token_state(obs, env, tok, held, obs["part_poses"][tok.part][2])
            sizes.add(s.shape)
            assert np.isfinite(s).all()
        assert token_goal(tok, product).shape == (3,)
        assert involved_context(env.ctx, held, env.part_order).shape == (2 + env.n_arm,)
    assert sizes == {(2 * env.n_arm + 17,)}


def test_success_matches_scripted(a):
    product, recipe, bank = a
    env = AssemblyEnv(ARM, product)
    for k in bank:
        tok = recipe.steps[k.step]
        env.restore(k)
        goal = token_goal(tok, product)
        z0 = env.obs()["part_poses"][k.held or tok.part][2]
        s0 = token_state(env.obs(), env, tok, k.held, z0)
        assert not success(tok.kind, s0, goal)  # also: INSERT starts >= depth above its target
        res, held = run_token(env, tok, k.held)
        s = token_state(env.obs(), env, tok, held if tok.kind == "GRASP" else k.held, z0)
        assert res.ok and success(tok.kind, s, goal), (k.step, tok.kind)


def test_grasp_requires_z0():
    product, recipe = _load("product_a")
    env = AssemblyEnv(ARM, product)
    obs, _ = env.reset(seed=0)
    tok = next(t for t in recipe.steps if t.kind == "GRASP")
    with pytest.raises(ValueError, match="z0"):
        token_state(obs, env, tok, None)


def test_reward_pure(a):
    product, recipe, bank = a
    env = AssemblyEnv(ARM, product)
    k = next(x for x in bank if x.token == "INSERT")
    tok = recipe.steps[k.step]
    env.restore(k)
    s = token_state(env.obs(), env, tok, k.held)
    goal = token_goal(tok, product)
    s2 = s.copy()
    s2[-1] = 3.0  # force 3x the limit
    before = (s.copy(), s2.copy())
    r = reward("INSERT", s, s2, goal)
    assert r == reward("INSERT", s, s2, goal) and r < 0
    assert all(np.array_equal(x, y) for x, y in zip(before, (s, s2)))


def test_episode_terminal_kinds(a):
    product, recipe, bank = a
    env = AssemblyEnv(ARM, product)
    k = next(x for x in bank if x.token == "ALIGN")
    tok = recipe.steps[k.step]
    ep = TokenEpisode(env, k, tok, budget_steps=3)
    s = ep.reset()
    hold = np.append(s[:env.n_arm] * 0, 0.0)  # zero motion, gripper closed
    for _ in range(2):
        s, r, done, info = ep.step(hold)
        assert not done and info == {}
    s, r, done, info = ep.step(hold)
    assert not done and info["terminal_kind"] == "timeout"  # truncation, not termination
    # success: start from the state after a scripted ALIGN
    env.restore(k)
    run_token(env, tok, k.held)
    after = type(k)(k.product, k.step, k.token, k.held, k.context,
                    _state(env))
    ep = TokenEpisode(env, after, tok, budget_steps=50)
    ep.reset()
    _, _, done, info = ep.step(hold)
    assert done and info["terminal_kind"] == "success"


def _state(env):
    import mujoco
    from supermimic_yai.env import START_SPEC
    s = np.empty(mujoco.mj_stateSize(env.model, START_SPEC))
    mujoco.mj_getState(env.model, env.data, s, START_SPEC)
    return s


def _at(a, kind):
    product, recipe, bank = a
    env = AssemblyEnv(ARM, product)
    k = next(x for x in bank if x.token == kind)
    tok = recipe.steps[k.step]
    ep = TokenEpisode(env, k, tok, budget_steps=100)
    ep.reset()
    return env, ep, tok, np.append(np.zeros(env.n_arm), 0.0)


def test_align_needs_held_flag(a):
    env, ep, tok, hold = _at(a, "ALIGN")
    s = ep.s.copy()
    n = env.n_arm
    s[2 * n + 1:2 * n + 8] = s[2 * n + 9:2 * n + 16]  # part exactly on target
    g = token_goal(tok, env.product)
    s[2 * n + 8] = 1
    assert success("ALIGN", s, g)
    s[2 * n + 8] = 0
    assert not success("ALIGN", s, g)


def test_dropped_debounced(a, monkeypatch):
    env, ep, tok, hold = _at(a, "ALIGN")
    seq = iter([False, True, False, False, False, False, False])  # flicker, then 5 consecutive
    monkeypatch.setattr(task, "fingers_hold", lambda e, p: next(seq))
    kinds = []
    for _ in range(7):
        _, _, done, info = ep.step(hold)
        kinds.append(info.get("terminal_kind"))
        if done:
            break
    assert kinds == [None] * 6 + ["dropped"]


def test_yaw_numpy_matches_and_normalises(a):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q1, q2 = rng.normal(size=4), rng.normal(size=4)
        q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
        c, t = np.r_[0, 0, 0, q1], np.r_[0, 0, 0, q2]
        assert abs(task._yaw_np(c[3:], t[3:]) - _yaw(c, t)) < 1e-9
        assert abs(task._yaw_np(1.3 * c[3:], 0.7 * t[3:]) - _yaw(c, t)) < 1e-9


def test_scaled_quats_do_not_change_result(a):
    env, ep, tok, _ = _at(a, "ALIGN")
    n, s = env.n_arm, ep.s.copy()
    s2 = s.copy()
    s2[2 * n + 4:2 * n + 8] *= 1.3
    s2[2 * n + 12:2 * n + 16] *= 1.3
    g = token_goal(tok, env.product)
    assert success("ALIGN", s, g) == success("ALIGN", s2, g)
    assert reward("ALIGN", s, s, g) == pytest.approx(reward("ALIGN", s2, s2, g))


def test_grasp_z0_baseline(a):
    env, ep, tok, hold = _at(a, "GRASP")
    assert ep.z0 == ep.env.obs()["part_poses"][tok.part][2]
    assert ep.s[2 * env.n_arm + 8] == 0


def test_force_terminal_and_ratio(a, monkeypatch):
    env, ep, tok, hold = _at(a, "INSERT")
    monkeypatch.setattr(task, "contact_force", lambda e, p: 2.5 * tok.max_force)
    _, _, done, info = ep.step(hold)
    assert done and info["terminal_kind"] == "force"
    assert info["max_force_ratio"] == pytest.approx(2.5)


def test_reward_sanity(a):
    env, ep, tok, _ = _at(a, "GRASP")
    n, g = env.n_arm, np.zeros(3)
    s, near = ep.s.copy(), ep.s.copy()
    near[2 * n + 9:2 * n + 12] *= 0.5
    assert reward("GRASP", s, near, g) > reward("GRASP", s, s, g)
    env, ep, tok, _ = _at(a, "ALIGN")
    s, near = ep.s.copy(), ep.s.copy()
    g = token_goal(tok, env.product)
    near[2 * n + 1:2 * n + 3] = near[2 * n + 9:2 * n + 11] + 0.001
    far = near.copy()
    far[2 * n + 1:2 * n + 3] += 0.01
    assert reward("ALIGN", s, near, g) > reward("ALIGN", s, far, g)
