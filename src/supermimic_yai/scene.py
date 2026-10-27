import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from supermimic_yai.arm import ArmSpec
from supermimic_yai.collision import decompose
from supermimic_yai.context import SimContext
from supermimic_yai.product import Product


@dataclass
class SceneIndex:
    part_body: dict[str, int]
    part_qpos_adr: dict[str, int]
    arm_qpos_adr: list[int]
    arm_ctrl_idx: list[int]
    gripper_ctrl_idx: int
    ee_body: int
    ee_offset: np.ndarray
    finger_geoms: set[int]


# Stiff contacts: default soft constraints scale with effective mass, so light parts sink under load.
SOLREF = (0.004, 1.0)
SOLIMP = (0.95, 0.99, 0.001, 0.5, 2)


def _start_pose(pose, noise):
    dx, dy, dyaw = noise
    q = np.empty(4)
    h = dyaw / 2
    mujoco.mju_mulQuat(q, np.array([np.cos(h), 0, 0, np.sin(h)]), np.array(pose[3:]))
    return np.array([pose[0] + dx, pose[1] + dy, pose[2]]), q


def build_scene(arm: ArmSpec, product: Product, ctx: SimContext,
                cache_dir: Path | None = None) -> tuple[mujoco.MjModel, SceneIndex]:
    cache_dir = Path(cache_dir or os.environ.get("SMY_CACHE_DIR", ".cache/coacd"))
    spec = mujoco.MjSpec.from_file(str(arm.mjcf))
    for k in list(spec.keys):  # arm keyframes have the wrong qpos size once free joints are added
        spec.delete(k)
    spec.memory = 256 * 1024 * 1024  # many convex pieces x multi-point contacts need a big arena
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 2
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD  # mesh pairs otherwise yield one contact point, so resting boxes drift
    for b in spec.bodies:  # position actuators sag under gravity otherwise; parts are added after, so unaffected
        b.gravcomp = 1.0
    world = spec.worldbody

    table = world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1])
    table.friction[0] = ctx.friction
    table.solref, table.solimp = SOLREF, SOLIMP

    for name, part in product.parts.items():
        pieces = decompose(part.mesh, cache_dir)
        vols = [trimesh.load(p, force="mesh").volume for p in pieces]
        density = part.mass * ctx.mass_scale[name] / sum(vols)

        if part.fixed:
            body = world.add_body(name=f"part:{name}", pos=part.final_pose[:3], quat=part.final_pose[3:])
        else:
            pos, quat = _start_pose(part.start_pose, ctx.start_pose_noise[name])
            body = world.add_body(name=f"part:{name}", pos=pos, quat=quat)
            body.add_freejoint(name=f"free:{name}")

        spec.add_mesh(name=f"visual:{name}", file=str(part.mesh.resolve()))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"visual:{name}",
                      contype=0, conaffinity=0, group=2, density=0).friction[0] = ctx.friction
        for i, p in enumerate(pieces):
            spec.add_mesh(name=f"col:{name}:{i}", file=str(p.resolve()))
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"col:{name}:{i}",
                              density=density, group=3)
            g.friction[0] = ctx.friction
            g.solref, g.solimp = SOLREF, SOLIMP
            g.condim = 4  # torsional friction: a pinched part otherwise pivots freely about the pinch axis

    model = spec.compile()

    def jid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)

    def aid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    body_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"part:{n}") for n in product.parts}
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)

    def under_ee(b):
        while b > 0:
            b = model.body_parentid[b]
            if b == ee:
                return True
        return False

    finger_bodies = {b for b in range(model.nbody) if under_ee(b) and model.body_jntnum[b] > 0}
    index = SceneIndex(
        part_body=body_ids,
        part_qpos_adr={n: int(model.jnt_qposadr[jid(f"free:{n}")])
                       for n, p in product.parts.items() if not p.fixed},
        arm_qpos_adr=[int(model.jnt_qposadr[jid(j)]) for j in arm.arm_joints],
        arm_ctrl_idx=[aid(a) for a in arm.arm_actuators],
        gripper_ctrl_idx=aid(arm.gripper_actuator),
        ee_body=ee,
        ee_offset=np.array(arm.ee_offset),
        finger_geoms={g for g in range(model.ngeom) if model.geom_bodyid[g] in finger_bodies},
    )
    return model, index


def apply_context(model: mujoco.MjModel, index: SceneIndex, ctx: SimContext, base_model: mujoco.MjModel) -> None:
    """Apply ctx to a model built with the nominal context, in place. base_model is an untouched copy of it."""
    m = model
    m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table"), 0] = ctx.friction
    for name, b in index.part_body.items():
        m.geom_friction[m.geom_bodyid == b, 0] = ctx.friction
        m.body_mass[b] = base_model.body_mass[b] * ctx.mass_scale[name]
        m.body_inertia[b] = base_model.body_inertia[b] * ctx.mass_scale[name]
    for name, adr in index.part_qpos_adr.items():
        dx, dy, dyaw = ctx.start_pose_noise[name]
        q = base_model.qpos0[adr:adr + 7]
        pos, quat = _start_pose(np.concatenate([q[:3], q[3:]]), (dx, dy, dyaw))
        m.qpos0[adr:adr + 3], m.qpos0[adr + 3:adr + 7] = pos, quat
        m.qpos_spring[adr:adr + 3], m.qpos_spring[adr + 3:adr + 7] = pos, quat
        b = index.part_body[name]
        m.body_pos[b], m.body_quat[b] = pos, quat
    mujoco.mj_setConst(m, mujoco.MjData(m))  # recompute mass-derived constants (invweight0, dof_M0, subtreemass...)
