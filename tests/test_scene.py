import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from supermimic_yai.arm import load_arm
from supermimic_yai.collision import decompose
from supermimic_yai.context import CONTEXT_RANGES, nominal_context, sample_context
from supermimic_yai.product import load_product
from supermimic_yai.scene import build_scene

ROOT = Path(__file__).parent.parent
CACHE = ROOT / ".cache" / "coacd"
ARM = load_arm(ROOT / "arms" / "panda")


def prod(x):
    return load_product(ROOT / "products" / f"product_{x}")


def test_decompose_caches():
    mesh = next(iter(prod("a").parts.values())).mesh
    a = decompose(mesh, CACHE)
    t = time.perf_counter()
    b = decompose(mesh, CACHE)
    assert a == b and a and time.perf_counter() - t < 0.1


@pytest.mark.parametrize("x", "abc")
def test_scene_builds_for_all_products(x):
    p = prod(x)
    model, idx = build_scene(ARM, p, nominal_context(p, 7), CACHE)
    assert set(idx.part_body) == set(p.parts)
    assert set(idx.part_qpos_adr) == {n for n, s in p.parts.items() if not s.fixed}
    assert idx.finger_geoms and len(idx.arm_qpos_adr) == 7


def test_parts_settle():
    p = prod("a")
    model, idx = build_scene(ARM, p, nominal_context(p, 7), CACHE)
    d = mujoco.MjData(model)
    d.qpos[idx.arm_qpos_adr] = ARM.home_qpos
    d.ctrl[idx.arm_ctrl_idx] = ARM.home_qpos
    d.ctrl[idx.gripper_ctrl_idx] = ARM.gripper_open_ctrl
    mujoco.mj_forward(model, d)
    start = {n: d.xpos[b].copy() for n, b in idx.part_body.items()}
    for _ in range(int(1 / model.opt.timestep)):
        mujoco.mj_step(model, d)
    for n in idx.part_qpos_adr:
        assert np.linalg.norm(d.xpos[idx.part_body[n]] - start[n]) < 0.002, n
    assert np.isfinite(d.qacc).all()


def test_context_in_ranges():
    p = prod("a")
    rng = np.random.default_rng(0)
    r = CONTEXT_RANGES
    for _ in range(50):
        c = sample_context(rng, p, 7)
        assert r["friction"][0] <= c.friction <= r["friction"][1]
        assert all(r["mass_scale"][0] <= s <= r["mass_scale"][1] for s in c.mass_scale.values())
        for noise in c.start_pose_noise.values():
            assert all(lo <= v <= hi for v, (lo, hi) in zip(noise, r["start_pose_noise"]))
        assert len(c.joint_offsets) == 7
        assert all(r["joint_offsets"][0] <= o <= r["joint_offsets"][1] for o in c.joint_offsets)
    assert c.to_vector(list(p.parts)).shape == (1 + 4 * len(p.parts) + 7,)


def test_context_applied():
    p = prod("a")
    ctx = nominal_context(p, 7)
    ctx = type(ctx)(0.5, ctx.mass_scale, ctx.start_pose_noise, ctx.joint_offsets)
    model, idx = build_scene(ARM, p, ctx, CACHE)
    geoms = [g for b in idx.part_body.values() for g in np.flatnonzero(model.geom_bodyid == b)]
    assert geoms and np.all(model.geom_friction[geoms, 0] == 0.5)


def test_relative_paths(monkeypatch):
    monkeypatch.chdir(ROOT)
    p = load_product(Path("products/product_a"))
    model, idx = build_scene(ARM, p, nominal_context(p, 7))
    assert set(idx.part_body) == set(p.parts)


def test_contact_is_stiff():
    p = prod("a")
    model, idx = build_scene(ARM, p, nominal_context(p, 7), CACHE)
    d = mujoco.MjData(model)
    d.qpos[idx.arm_qpos_adr] = ARM.home_qpos
    d.ctrl[idx.arm_ctrl_idx] = ARM.home_qpos
    d.ctrl[idx.gripper_ctrl_idx] = ARM.gripper_open_ctrl
    b = idx.part_body["peg1"]
    deepest = 0.0
    for i in range(int(1 / model.opt.timestep)):
        d.xfrc_applied[b, 2] = -20.0
        mujoco.mj_step(model, d)
        if i > 50 and d.ncon:  # skip the first 0.1 s impact transient of the suddenly applied load
            deepest = max(deepest, -min(c.dist for c in d.contact[:d.ncon]))
    assert deepest < 0.002 and np.isfinite(d.qacc).all()
