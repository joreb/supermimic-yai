from pathlib import Path

import mujoco
import numpy as np

from supermimic_yai.arm import load_arm
from supermimic_yai.context import nominal_context
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import load_product
from supermimic_yai.scripted import _Thing, _down, _ee, _mat, _part, _pose, execute, run_token
from supermimic_yai.tokens import Grasp, Insert, MoveAbove, MoveTo, Press, Recipe, Release, Retreat

ROOT = Path(__file__).parent.parent
ARM = load_arm(ROOT / "arms" / "panda")
PRODUCT = load_product(ROOT / "products" / "product_a")
ENV = AssemblyEnv(ARM, PRODUCT)


def fresh():
    ENV.reset(options={"context": nominal_context(PRODUCT, 7)})
    return ENV


def peg_grasp(env, part="peg1"):
    """Grasp pose in the part frame: 1 cm above centre, gripper pointing down."""
    T = np.linalg.inv(_mat(env._obs()["part_poses"][part])) @ _mat(np.concatenate([[0, 0, 0], _down(env)]))
    T[:3, 3] = [0, 0, 0.01]
    return tuple(_pose(T))


def test_move_to_reaches():
    env = fresh()
    ee = env._obs()["ee_pose"]
    tgt = np.concatenate([ee[:3] + [0.05, 0.05, -0.05], ee[3:]])
    res, _ = run_token(env, MoveTo(tuple(tgt)), None)
    assert res.ok, res.reason
    assert np.linalg.norm(env._obs()["ee_pose"][:3] - tgt[:3]) < 0.003


def test_unreachable_fails_fast():
    env = fresh()
    res, _ = run_token(env, MoveTo((2.0, 0, 0.5, *_down(env))), None)
    assert not res.ok and res.reason == "unreachable" and res.sim_time <= 4


def test_grasp_lifts_peg():
    env = fresh()
    z0 = env._obs()["part_poses"]["peg1"][2]
    res, held = run_token(env, Grasp("peg1", peg_grasp(env)), None)
    assert res.ok, res.reason
    assert held == "peg1" and env._obs()["part_poses"]["peg1"][2] > z0 + 0.015


def test_unstable_reported(monkeypatch):
    env = fresh()
    monkeypatch.setattr(env, "step", lambda a: (env._obs(), 0.0, True, False, {"unstable": True}))
    log = execute(env, Recipe("product_a", (MoveTo((0.4, 0, 0.4, *_down(env))),)))
    assert log.results[-1].reason == "simulation unstable" and log.stopped_at == 0


def test_stops_at_first_failure():
    env = fresh()
    down = tuple(_down(env))
    steps = (Retreat(0.01), MoveTo((2.0, 0, 0.5, *down)), Release())
    log = execute(env, Recipe("product_a", steps))
    assert len(log.results) == 2 and log.stopped_at == 1


def _grasp_peg1(env):
    res, held = run_token(env, Grasp("peg1", peg_grasp(env)), None)
    assert res.ok, res.reason
    return held


def test_insert_succeeds_with_tiny_max_force_while_gripping():
    env = fresh()
    held = _grasp_peg1(env)
    _, held = run_token(env, MoveAbove("final:peg1", 0.05), held)
    res, _ = run_token(env, Insert("peg1", "final:peg1", 0.03, 1e-3), held)
    assert res.ok, res.reason  # grip force is excluded from the limit


def test_insert_jam_reports_force_limit():
    env = fresh()
    held = _grasp_peg1(env)
    _, held = run_token(env, MoveAbove("final:cover", 0.05), held)  # peg lands on the base top
    res, _ = run_token(env, Insert("peg1", "final:cover", 0.03, 0.3), held)
    assert not res.ok and res.reason == "force limit"


def test_press_blocked_reports_force_limit_or_misaligned():
    env = fresh()
    ee = env._obs()["ee_pose"]
    top = env._obs()["part_poses"]["peg2"]
    run_token(env, MoveTo((top[0], top[1], 0.1, *ee[3:])), None)
    res, _ = run_token(env, Press("peg2", (0, 0, -1), 5.0), None)  # peg2 rests on the table
    assert not res.ok and res.reason in ("force limit", "misaligned")


def test_held_transform_follows_grip_shift():
    env = fresh()
    _grasp_peg1(env)
    adr = env.index.part_qpos_adr["peg1"]
    env.data.qpos[adr] += 0.002  # the part slides 2 mm in the grip
    mujoco.mj_forward(env.model, env.data)
    thing = _Thing(env, "peg1")
    assert np.allclose(thing.ee_target(_part(env, "peg1"))[:3], _ee(env)[:3], atol=1e-6)
