from pathlib import Path

import numpy as np
import pytest

from supermimic_yai import seeds
from supermimic_yai.arm import load_arm
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import load_product
from supermimic_yai.scripted import run_token
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
            s = token_state(obs, env, tok, held)
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
        s0 = token_state(env.obs(), env, tok, k.held)
        assert not success(tok.kind, s0, goal)  # also: INSERT starts >= depth above its target
        res, held = run_token(env, tok, k.held)
        s = token_state(env.obs(), env, tok, held if tok.kind == "GRASP" else k.held)
        assert res.ok and success(tok.kind, s, goal), (k.step, tok.kind)
    # failure side: ALIGN/INSERT/GRASP states that scripted would reject are not successes
    k = next(x for x in bank if x.token == "INSERT")
    env.restore(k)
    tok = recipe.steps[k.step]
    assert not success("INSERT", token_state(env.obs(), env, tok, k.held), token_goal(tok, product))


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
    assert done and info["terminal_kind"] == "timeout"
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
