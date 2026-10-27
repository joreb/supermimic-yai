from pathlib import Path

import dataclasses

import mujoco
import numpy as np
import pytest

from supermimic_yai.context import nominal_context, sample_context
from supermimic_yai.arm import load_arm
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import load_product

ROOT = Path(__file__).parent.parent
ARM = load_arm(ROOT / "arms" / "panda")
ENV = AssemblyEnv(ARM, load_product(ROOT / "products" / "product_a"))


def test_reset_deterministic():
    o1, i1 = ENV.reset(seed=7)
    o2, i2 = ENV.reset(seed=7)
    assert i1["context"] == i2["context"]
    for n in o1["part_poses"]:
        assert np.array_equal(o1["part_poses"][n], o2["part_poses"][n])


def test_hold_home_is_stable():
    obs, _ = ENV.reset(options={"context": nominal_context(ENV.product, 7)})  # offsets would shift ee by design
    start = obs["ee_pose"][:3]
    act = np.append(ARM.home_qpos, 1.0)
    for _ in range(100):
        obs, r, term, trunc, info = ENV.step(act)
        assert not term
    assert np.linalg.norm(obs["ee_pose"][:3] - start) < 1e-3


def test_unstable_terminates():
    ENV.reset(seed=2)
    ENV.data.qvel[:] = 1e9
    _, _, term, _, info = ENV.step(np.append(ARM.home_qpos, 1.0))
    assert term and info["unstable"]


def test_step_before_reset_raises_clear_error():
    env = AssemblyEnv(ARM, load_product(ROOT / "products" / "product_a"))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.append(ARM.home_qpos, 1.0))


def test_obs_arm_qpos_is_encoder_frame():
    ctx = nominal_context(ENV.product, 7)
    ctx = dataclasses.replace(ctx, joint_offsets=(0.03,) * 7)
    ENV.reset(options={"context": ctx})
    act = np.append(ARM.home_qpos, 1.0)
    for _ in range(50):
        obs, *_ = ENV.step(act)
    assert np.abs(obs["arm_qpos"] - ARM.home_qpos).max() < 2e-3
    assert np.abs(ENV.data.qpos[ENV.index.arm_qpos_adr] - (np.array(ARM.home_qpos) + 0.03)).max() < 2e-3


def _flat(o):
    return np.concatenate([o["arm_qpos"], o["arm_qvel"], o["gripper"], o["ee_pose"],
                           *[o["part_poses"][n] for n in sorted(o["part_poses"])]])


def _rebuilt_env(ctx):
    from supermimic_yai.scene import build_scene
    e = AssemblyEnv(ARM, ENV.product)
    e.model, e.index = build_scene(ARM, ENV.product, ctx)
    e.data = mujoco.MjData(e.model)
    e.ctx = ctx
    return e


def test_inplace_matches_rebuild():
    rng = np.random.default_rng(0)
    ctxs = [sample_context(rng, ENV.product, 7) for _ in range(2)]
    ctxs.append(ctxs[0])
    for k, ctx in enumerate(ctxs):
        o1, _ = ENV.reset(options={"context": ctx})
        ref = _rebuilt_env(ctx)
        o2, _ = ref.reset(options={"context": ctx})
        assert np.abs(_flat(o1) - _flat(o2)).max() < 1e-9
        for t in range(50):
            act = np.append(ARM.home_qpos + 0.05 * np.sin(0.2 * t + np.arange(7)), 0.5 + 0.5 * np.sin(0.1 * t))
            o1 = ENV.step(act)[0]
            o2 = ref.step(act)[0]
        assert np.abs(_flat(o1) - _flat(o2)).max() < 1e-6


def test_context_applied_inplace():
    ctx = sample_context(np.random.default_rng(3), ENV.product, 7)
    ENV.reset(options={"context": nominal_context(ENV.product, 7)})
    m0 = ENV._base.body_mass.copy()
    ENV.reset(options={"context": ctx})
    m, ix = ENV.model, ENV.index
    assert m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table"), 0] == ctx.friction
    for n, b in ix.part_body.items():
        assert np.allclose(m.geom_friction[m.geom_bodyid == b, 0], ctx.friction)
        assert np.isclose(m.body_mass[b], m0[b] * ctx.mass_scale[n])
    base = ENV._base
    for n, adr in ix.part_qpos_adr.items():
        dx, dy, dyaw = ctx.start_pose_noise[n]
        assert np.allclose(m.qpos0[adr:adr + 2], base.qpos0[adr:adr + 2] + [dx, dy])
        q, q0 = m.qpos0[adr + 3:adr + 7], base.qpos0[adr + 3:adr + 7]
        rel = np.empty(4)  # q * conj(q0) = yaw rotation about z
        mujoco.mju_mulQuat(rel, q, q0 * [1, -1, -1, -1])
        assert np.isclose(2 * np.arctan2(rel[3], rel[0]), dyaw)


def test_action_shape_checked():
    ENV.reset(seed=1)
    with pytest.raises(ValueError):
        ENV.step(np.zeros(3))


def test_obs_public():
    ENV.reset(seed=1)
    assert set(ENV.obs()) == {"arm_qpos", "arm_qvel", "gripper", "ee_pose", "part_poses", "context"}
