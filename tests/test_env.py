from pathlib import Path

import dataclasses

import numpy as np
import pytest

from supermimic_yai.context import nominal_context
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
