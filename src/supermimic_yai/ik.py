import mujoco
import numpy as np

from supermimic_yai.product import Pose


def ee_point(env) -> np.ndarray:
    d, i = env.data, env.index
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


def ik_step(env, target_pose: Pose, damping: float = 1e-2, max_dq: float | None = None) -> np.ndarray:
    """One damped-least-squares step on the 6-D pose error at the end-effector point."""
    m, d, i = env.model, env.data, env.index
    dt = 1 / env.control_hz
    max_dq = MAX_DQ_SPEED * dt if max_dq is None else max_dq
    point = ee_point(env)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, point, i.ee_body)
    J = np.vstack([jacp, jacr])[:, env._arm_dof]
    err = np.concatenate([_cap(np.asarray(target_pose[:3]) - point, MAX_POS_SPEED * dt),
                          _cap(rot_error(target_pose[3:], d.xquat[i.ee_body]), MAX_ROT_SPEED * dt)])
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
    dq = np.clip(dq, -max_dq, max_dq)
    q = d.qpos[i.arm_qpos_adr] - env.ctx.joint_offsets + dq  # env adds the offsets back onto the command
    return np.clip(q, env.action_space.low[:env.n_arm], env.action_space.high[:env.n_arm])
