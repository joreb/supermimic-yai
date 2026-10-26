import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from supermimic_yai.arm import ArmSpec
from supermimic_yai.context import SimContext, nominal_context, sample_context
from supermimic_yai.product import Product
from supermimic_yai.scene import build_scene

_WARNS = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
          mujoco.mjtWarning.mjWARN_BADQPOS)


class AssemblyEnv(gym.Env):
    def __init__(self, arm: ArmSpec, product: Product, control_hz: int = 50):
        self.arm, self.product = arm, product
        self.n_arm = len(arm.arm_joints)
        self.part_order = list(product.parts)
        self.ctx = nominal_context(product, self.n_arm)
        self._build(self.ctx)
        self.substeps = round(1 / (control_hz * self.model.opt.timestep))

        rng = self.model.jnt_range
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm.arm_joints]
        lo = np.append(rng[jids, 0], 0.0)
        hi = np.append(rng[jids, 1], 1.0)
        self.action_space = spaces.Box(lo, hi, dtype=np.float64)
        inf = lambda n: spaces.Box(-np.inf, np.inf, (n,), np.float64)  # noqa: E731
        self.observation_space = spaces.Dict({
            "arm_qpos": inf(self.n_arm), "arm_qvel": inf(self.n_arm), "gripper": inf(1),
            "ee_pose": inf(7),
            "part_poses": spaces.Dict({n: inf(7) for n in self.part_order}),
            "context": inf(len(self.ctx.to_vector(self.part_order))),
        })

    def _build(self, ctx: SimContext):
        self.model, self.index = build_scene(self.arm, self.product, ctx)
        self.data = mujoco.MjData(self.model)
        self.ctx = ctx

    def _set_ctrl(self, target, grip):
        i, a = self.index, self.arm
        self.data.ctrl[i.arm_ctrl_idx] = np.asarray(target) + self.ctx.joint_offsets
        self.data.ctrl[i.gripper_ctrl_idx] = a.gripper_closed_ctrl + grip * (a.gripper_open_ctrl - a.gripper_closed_ctrl)

    def _obs(self):
        d, i = self.data, self.index
        ee_rot = d.xmat[i.ee_body].reshape(3, 3)
        return {
            "arm_qpos": d.qpos[i.arm_qpos_adr].copy(),
            "arm_qvel": d.qvel[self._arm_dof].copy(),
            "gripper": np.array([d.actuator_length[i.gripper_ctrl_idx]]),
            "ee_pose": np.concatenate([d.xpos[i.ee_body] + ee_rot @ i.ee_offset, d.xquat[i.ee_body]]),
            "part_poses": {n: np.concatenate([d.xpos[b], d.xquat[b]]) for n, b in i.part_body.items()},
            "context": self.ctx.to_vector(self.part_order),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ctx = (options or {}).get("context") or sample_context(self.np_random, self.product, self.n_arm)
        if ctx != self.ctx:
            self._build(ctx)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)  # parts keep their built start poses via qpos0
        d.qpos[self.index.arm_qpos_adr] = self.arm.home_qpos
        self._arm_dof = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
                         for j in self.arm.arm_joints]
        self._set_ctrl(self.arm.home_qpos, 1.0)
        mujoco.mj_forward(m, d)
        return self._obs(), {"context": ctx}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        self._set_ctrl(a[:self.n_arm], float(np.clip(a[self.n_arm], 0.0, 1.0)))
        before = [self.data.warning[w].number for w in _WARNS]
        mujoco.mj_step(self.model, self.data, self.substeps)
        d = self.data
        unstable = (any(d.warning[w].number > b for w, b in zip(_WARNS, before))
                    or not (np.isfinite(d.qacc).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qpos).all()))
        info = {"unstable": True} if unstable else {}
        return self._obs(), 0.0, unstable, False, info
