import mujoco
import numpy as np

from supermimic_yai.product import Pose


def ee_point(env, d=None) -> np.ndarray:
    d, i = env.data if d is None else d, env.index
    return d.xpos[i.ee_body] + d.xmat[i.ee_body].reshape(3, 3) @ i.ee_offset


def rot_error(q_target, q_current) -> np.ndarray:
    """Axis-angle (world frame) of q_target * q_current^-1, shortest path."""
    inv, dq, out = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_negQuat(inv, np.asarray(q_current, dtype=np.float64))
    mujoco.mju_mulQuat(dq, np.asarray(q_target, dtype=np.float64), inv)
    if dq[0] < 0:
        dq = -dq
    mujoco.mju_quat2Vel(out, dq, 1.0)
    return out


MAX_POS_SPEED, MAX_ROT_SPEED, MAX_DQ_SPEED = 1.0, 5.0, 2.5  # m/s, rad/s, rad/s; times the tick, they cap each step (keeps the linearisation valid on long moves)


def _cap(v, limit):
    n = np.linalg.norm(v)
    return v if n <= limit else v * limit / n


IK_HZ = 50  # the controller's iteration rate; slower control ticks run several kinematic iterations per tick


def ik_step(env, target_pose: Pose, damping: float = 1e-2, max_dq: float | None = None) -> np.ndarray:
    """Damped-least-squares steps on the 6-D pose error at the end-effector point, IK_HZ iterations per second:
    n = IK_HZ / control_hz of them per tick, the later ones from the kinematically predicted pose, each capped by
    the per-iteration speed limits. -> the arm command (encoder frame)."""
    m, i = env.model, env.index
    n = max(1, round(IK_HZ / env.control_hz))
    dt = 1 / (env.control_hz * n)
    max_dq = MAX_DQ_SPEED * dt if max_dq is None else max_dq
    if n > 1:
        d = env.__dict__.setdefault("_ik_data", mujoco.MjData(m))
        d.qpos[:] = env.data.qpos
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
    else:
        d = env.data
    lo, hi = env.action_space.low[:env.n_arm], env.action_space.high[:env.n_arm]
    off = env.ctx.joint_offsets
    for k in range(n):
        point = ee_point(env, d)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, point, i.ee_body)
        J = np.vstack([jacp, jacr])[:, env._arm_dof]
        err = np.concatenate([_cap(np.asarray(target_pose[:3]) - point, MAX_POS_SPEED * dt),
                              _cap(rot_error(target_pose[3:], d.xquat[i.ee_body]), MAX_ROT_SPEED * dt)])
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err), -max_dq, max_dq)
        q = np.clip(d.qpos[i.arm_qpos_adr] - off + dq, lo, hi)  # env adds the offsets back onto the command
        if k < n - 1:
            d.qpos[i.arm_qpos_adr] = q + off
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
    return q
