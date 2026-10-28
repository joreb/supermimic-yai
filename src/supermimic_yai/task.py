"""Token task interface: fixed-size state, goal, analytic reward/success, short episode per contact token.

State layout (n = arm joints), all float64:
  [0:n] encoder-frame arm qpos | [n:2n] arm qvel | [2n] gripper
  [2n+1:2n+8] held-part pose rel. EE | [2n+8] held flag
  [2n+9:2n+16] target-frame pose rel. EE | [2n+16] contact force on the part
"Rel. EE" is world-axis-aligned: position = p - p_ee, quaternion = q * q_ee^-1. So differences of two such
poses are exact world-frame errors, which is what success/reward need (xy error, yaw).
Held flag: ALIGN/INSERT -> fingers hold the held part; GRASP -> fingers hold it AND it is lifted GRASP_LIFT
above its z at episode start, passed as z0 (the scripted grasp check; the state has no memory, so the flag carries it).
INSERT success: the scripted check also requires a descent of depth - tol from the token's start z; that is
implied by being at the final pose in recipes where the MOVE_ABOVE offset >= INSERT depth - tol (not checked
here), so success only needs the product tolerance, carried in goal[2].
reward/success are mujoco-free numpy; quaternions are normalised before use.
"""

import mujoco
import numpy as np

from supermimic_yai.scripted import (ALIGN_POS_TOL, ALIGN_ROT_TOL, _budget, _mat, _pose, contact_force,
                                     fingers_hold)
from supermimic_yai.tokens import resolve_target

TOKEN_KINDS = ("GRASP", "ALIGN", "INSERT")
FORCE_SCALE = 20.0
DROP_STEPS = 5  # consecutive steps without a grip before a part counts as dropped
GRASP_LIFT = 0.015  # same as the scripted grasp's lift check
_EXTRA = 1 + 8 + 7 + 1


def _rel(p, ee):
    q = np.empty(4)
    mujoco.mju_mulQuat(q, np.ascontiguousarray(p[3:], dtype=np.float64),
                       np.array([ee[3], -ee[4], -ee[5], -ee[6]], dtype=np.float64))
    return np.concatenate([np.asarray(p[:3]) - ee[:3], q])


def _rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _yaw_np(q_cur, q_tgt):
    R = _rot(q_tgt) @ _rot(q_cur).T
    return abs(np.arctan2(R[1, 0], R[0, 0]))


def _slices(s):
    n = (len(s) - _EXTRA) // 2
    return s[2 * n + 1:2 * n + 8], s[2 * n + 8], s[2 * n + 9:2 * n + 16], s[2 * n + 16]


def token_state(obs, env, tok, held, z0=None) -> np.ndarray:
    ee, part = obs["ee_pose"], held or tok.part
    pose = obs["part_poses"][part]
    if tok.kind == "GRASP":
        if z0 is None:
            raise ValueError("GRASP needs z0, the part's z at token start (lift baseline)")
        target = _pose(_mat(pose) @ _mat(tok.grasp_pose))
        flag = fingers_hold(env, part) and pose[2] - z0 >= GRASP_LIFT
    else:
        target = np.asarray(resolve_target(tok.target, env.product), dtype=np.float64)
        flag = held is not None and fingers_hold(env, held)
    f = contact_force(env, part) / (tok.max_force if tok.kind == "INSERT" else FORCE_SCALE)
    return np.concatenate([obs["arm_qpos"], obs["arm_qvel"], obs["gripper"], _rel(pose, ee), [float(flag)],
                           _rel(target, ee), [f]])


def token_goal(tok, product) -> np.ndarray:
    """INSERT: [depth, max_force/20, tol_pos]; otherwise zeros."""
    if tok.kind != "INSERT":
        return np.zeros(3)
    return np.array([tok.depth, tok.max_force / FORCE_SCALE, product.tol_pos])


def involved_context(ctx, held, part_order) -> np.ndarray:
    return np.array([ctx.friction, ctx.mass_scale[held] if held else 1.0, *ctx.joint_offsets])


def _align_err(s):
    part, _, tgt, _ = _slices(s)
    return np.linalg.norm(tgt[:2] - part[:2]), _yaw_np(part[3:], tgt[3:])


def reward(kind, s, s_next, goal) -> float:
    part, flag, tgt, f = _slices(s_next)
    if kind == "GRASP":
        return float(-np.linalg.norm(tgt[:3]) + flag)
    if kind == "ALIGN":
        xy, yaw = _align_err(s_next)
        return float(-(xy + 0.05 * yaw))
    rem = lambda x: _slices(x)[0][2] - _slices(x)[2][2]  # remaining descent  # noqa: E731
    mf = goal[1] * FORCE_SCALE
    return float((rem(s) - rem(s_next)) / goal[0] - 0.1 * max(0.0, f * mf - mf))


def success(kind, s, goal) -> bool:
    part, flag, tgt, _ = _slices(s)
    if kind == "GRASP":
        return bool(flag)
    if kind == "ALIGN":
        xy, yaw = _align_err(s)
        return bool(flag and xy <= ALIGN_POS_TOL and yaw <= ALIGN_ROT_TOL)
    return bool(np.linalg.norm(tgt[:3] - part[:3]) <= goal[2])


def budget_steps(env, tok) -> int:
    return round(_budget(env, tok) / (env.substeps * env.model.opt.timestep))


class TokenEpisode:
    def __init__(self, env, start, tok, budget_steps, action_scale=0.05):
        self.env, self.start, self.tok, self.budget = env, start, tok, budget_steps
        self.scale, self.n = action_scale, env.n_arm
        self.goal = token_goal(tok, env.product)
        self.lo, self.hi = env.action_space.low[:self.n], env.action_space.high[:self.n]

    def _state(self, obs):
        return token_state(obs, self.env, self.tok, self.held, self.z0)

    def reset(self):
        self.held, self.t, self.lost, self.ratio = self.start.held, 0, 0, 0.0
        obs = self.env.restore(self.start)
        self.z0 = obs["part_poses"][self.held or self.tok.part][2]
        self.s = self._state(obs)
        return self.s

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        q = np.clip(self.s[:self.n] + self.scale * a[:self.n], self.lo, self.hi)
        obs, _, unstable, _, _ = self.env.step(np.append(q, np.clip(a[self.n], 0.0, 1.0)))
        s2, kind, self.t = self._state(obs), self.tok.kind, self.t + 1
        r = reward(kind, self.s, s2, self.goal)
        flag = s2[2 * self.n + 8]
        self.lost = self.lost + 1 if flag == 0 and (self.lost or self.s[2 * self.n + 8] == 1) else 0
        if kind == "INSERT":
            self.ratio = max(self.ratio, s2[-1])
        why = None
        if unstable:
            why = "unstable"
        elif success(kind, s2, self.goal):
            why = "success"
        elif kind != "GRASP" and self.lost >= DROP_STEPS:
            why = "dropped"
        elif kind == "INSERT" and s2[-1] > 2:
            why = "force"
        elif self.t >= self.budget:
            why = "timeout"
        self.s = s2
        info = {"terminal_kind": why} if why else {}
        if kind == "INSERT":
            info["max_force_ratio"] = self.ratio
        return s2, r, why is not None, info
