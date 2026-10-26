"""Scripted token executor: IK-driven arm control with contact-aware token semantics."""

from dataclasses import dataclass, field

import mujoco
import numpy as np
import trimesh

from supermimic_yai.ik import ee_point, ik_step, rot_error
from supermimic_yai.tokens import (Align, Grasp, Insert, MoveAbove, MoveTo, Place, Press,
                                   Recipe, Release, Retreat, Token, resolve_target)

POS_TOL, ROT_TOL = 0.003, np.radians(3.0)
ALIGN_POS_TOL, ALIGN_ROT_TOL = 0.001, np.radians(2.0)
APPROACH_SPEED = 0.05   # m/s, grasp descent / lift
INSERT_SPEED = 0.01     # m/s
PRESS_SPEED = 0.005     # m/s
DAMP = 1e-3
LEAD = 0.003            # max distance the commanded point may lead the measured one (bounds contact force)
IK = {"damping": DAMP}
PLATEAU = (0.5, 5e-4)   # window [s], minimum improvement [m]


class _Fail(Exception):
    pass


@dataclass
class TokenResult:
    ok: bool
    reason: str
    sim_time: float


@dataclass
class ExecutionLog:
    results: list = field(default_factory=list)
    stopped_at: int | None = None
    frames: list = field(default_factory=list)


# ---- pose helpers (pose = [x, y, z, qw, qx, qy, qz]) ----
def _mat(p):
    T, R = np.eye(4), np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(p[3:], dtype=np.float64))
    T[:3, :3], T[:3, 3] = R.reshape(3, 3), p[:3]
    return T


def _pose(T):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(T[:3, :3]).ravel())
    return np.concatenate([T[:3, 3], q])


def _ee(env):
    return np.concatenate([ee_point(env), env.data.xquat[env.index.ee_body]])


def _part(env, name):
    b = env.index.part_body[name]
    return np.concatenate([env.data.xpos[b], env.data.xquat[b]])


def _down(env):
    """EE orientation at the arm's home configuration (the generic 'down')."""
    if getattr(env, "_down_quat", None) is None or env._down_for is not env.model:
        d = mujoco.MjData(env.model)
        d.qpos[env.index.arm_qpos_adr] = env.arm.home_qpos
        mujoco.mj_kinematics(env.model, d)
        env._down_quat, env._down_for = d.xquat[env.index.ee_body].copy(), env.model
    return env._down_quat


def _err(cur, tgt):
    return np.linalg.norm(np.asarray(tgt[:3]) - cur[:3]), np.linalg.norm(rot_error(tgt[3:], cur[3:]))


def _yaw(cur, tgt):
    R = _mat(tgt)[:3, :3] @ _mat(cur)[:3, :3].T
    return abs(np.arctan2(R[1, 0], R[0, 0]))


# ---- contacts ----
def _part_geoms(env, name):
    return set(np.flatnonzero(env.model.geom_bodyid == env.index.part_body[name]))


def contact_force(env, name) -> float:
    """Summed contact normal force on a part, excluding finger contacts."""
    geoms, f, tot = _part_geoms(env, name), np.zeros(6), 0.0
    for k in range(env.data.ncon):
        c = env.data.contact[k]
        a, b = c.geom1, c.geom2
        if (a in geoms) == (b in geoms) or (b if a in geoms else a) in env.index.finger_geoms:
            continue
        mujoco.mj_contactForce(env.model, env.data, k, f)
        tot += f[0]
    return tot


def _fingers_hold(env, name) -> bool:
    geoms, touched = _part_geoms(env, name), set()
    fb = {g: env.model.geom_bodyid[g] for g in env.index.finger_geoms}
    for k in range(env.data.ncon):
        c = env.data.contact[k]
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if a in fb and b in geoms:
                touched.add(fb[a])
    return len(touched) == len(set(fb.values())) >= 2


# ---- what is being controlled: the EE point, or the held part via its fixed EE-in-part transform ----
class _Thing:
    def __init__(self, env, held):
        self.env, self.held = env, held
        if held is None:
            self.tf = None
        else:
            saved = getattr(env, "_held_tf", None)
            self.tf = saved[1] if saved and saved[0] == held else np.linalg.inv(_mat(_part(env, held))) @ _mat(_ee(env))

    def pose(self):
        return _ee(self.env) if self.held is None else _part(self.env, self.held)

    def ee_target(self, p):
        return np.asarray(p) if self.held is None else _pose(_mat(p) @ self.tf)


class _Run:
    def __init__(self, env, budget):
        self.env = env
        self.dt = env.substeps * env.model.opt.timestep
        self.limit, self.n = round(budget / self.dt), 0

    def tick(self, ee_target, grip, lead=False):
        env, tgt = self.env, np.array(ee_target, dtype=np.float64)
        if lead:
            gap = tgt[:3] - ee_point(env)
            n = np.linalg.norm(gap)
            if n > LEAD:
                tgt[:3] = ee_point(env) + gap * LEAD / n
        _, _, term, _, _ = env.step(np.append(ik_step(env, tgt, **IK), grip))
        self.n += 1
        rec = getattr(env, "_rec", None)
        if rec is not None:
            rec["i"] += 1
            if rec["i"] % rec["every"] == 0:
                rec["r"].update_scene(env.data)
                rec["frames"].append(rec["r"].render().copy())
        if term:
            raise _Fail("simulation unstable")
        if self.n > self.limit:
            raise _Fail("timeout")

    def converge(self, thing, tgt, grip, pos_tol=POS_TOL, rot_tol=ROT_TOL, yaw_only=False):
        """Drive `thing` to pose tgt; fail 'unreachable' if the error plateaus, else 'timeout'."""
        w, hist = round(PLATEAU[0] / self.dt), []
        wr = pos_tol / rot_tol
        while True:
            cur = thing.pose()
            p, r = _err(cur, tgt)
            if yaw_only:
                p, r = np.linalg.norm(np.asarray(tgt[:2]) - cur[:2]), _yaw(cur, tgt)
            if p <= pos_tol and r <= rot_tol:
                return
            hist.append(p + wr * r)
            if len(hist) > w and hist[-w - 1] - hist[-1] < PLATEAU[1]:
                raise _Fail("unreachable")
            self.tick(thing.ee_target(tgt), grip)

    def ramp(self, thing, start, direction, speed, dist, grip, check=None):
        """Move `thing` from pose `start` along unit `direction` at `speed`; returns when
        check() is true, or at the end of the ramp when check is None."""
        start, direction, t = np.asarray(start, dtype=np.float64), np.asarray(direction, dtype=np.float64), 0.0
        while True:
            if check is not None and check():
                return
            if check is None and t * speed >= dist:
                return
            t += self.dt
            p = start.copy()
            p[:3] += direction * min(t * speed, dist)
            self.tick(thing.ee_target(p), grip, lead=True)


def _at(env, name, tgt, rot=True):
    p, r = _err(_part(env, name), tgt)
    return p <= env.product.tol_pos and (not rot or r <= np.radians(env.product.tol_rot_deg))


Z = np.array([0.0, 0.0, 1.0])


def _grasp(run, tok):
    env = run.env
    z0 = _part(env, tok.part)[2]
    E = _pose(_mat(_part(env, tok.part)) @ _mat(tok.grasp_pose))
    pre = E.copy()
    pre[2] += 0.08
    ee = _Thing(env, None)
    run.converge(ee, pre, 1.0)
    run.ramp(ee, pre, -Z, APPROACH_SPEED, 0.08, 1.0)
    run.converge(ee, E, 1.0)
    for _ in range(round(0.5 / run.dt)):
        run.tick(E, 0.0)
    if not _fingers_hold(env, tok.part):
        raise _Fail("grasp failed")
    lift = E.copy()
    lift[2] += 0.02
    run.ramp(ee, E, Z, APPROACH_SPEED, 0.02, 0.0)
    run.converge(ee, lift, 0.0)
    if not _fingers_hold(env, tok.part) or _part(env, tok.part)[2] - z0 < 0.015:
        raise _Fail("grasp failed")
    env._held_tf = (tok.part, np.linalg.inv(_mat(_part(env, tok.part))) @ _mat(_ee(env)))
    return tok.part


def _above(env, target, offset):
    p = np.array(resolve_target(target, env.product), dtype=np.float64)
    p[2] += offset
    return p


def _descent(run, env, thing, held, target, start_z, check):
    tgt = np.array(resolve_target(target, env.product), dtype=np.float64)
    start = tgt.copy()
    start[2] = start_z
    run.ramp(thing, start, -Z, INSERT_SPEED, max(start_z - tgt[2], 0.0), 0.0, check)


def _press(run, tok):
    env, part = run.env, tok.part
    verts = trimesh.load(env.product.parts[part].mesh, force="mesh").vertices
    T = _mat(_part(env, part))
    world = verts @ T[:3, :3].T + T[:3, 3]
    p = np.concatenate([(world.min(0)[:2] + world.max(0)[:2]) / 2, [world[:, 2].max() + 0.01], _down(env)])
    ee = _Thing(env, None)
    run.converge(ee, p, 0.0)
    final = np.array(env.product.parts[part].final_pose, dtype=np.float64)
    d = np.asarray(tok.direction, dtype=np.float64)
    hist, w = [], round(1.0 / run.dt)

    def check():
        if _at(env, part, final):
            return True
        if contact_force(env, part) > tok.max_force:
            raise _Fail("force limit")
        hist.append(ee_point(env) @ d)
        if len(hist) > w and hist[-1] - hist[-w - 1] < 0.001:
            raise _Fail("misaligned")
        return False

    run.ramp(ee, p, d, PRESS_SPEED, 0.1, 0.0, check)


def _run(env, tok, held, run):
    """Returns the part held afterwards."""
    ee = _Thing(env, None)
    if isinstance(tok, Grasp):
        return _grasp(run, tok)
    thing = _Thing(env, held)
    grip = 0.0 if held else 1.0
    if isinstance(tok, MoveTo):
        run.converge(ee, np.array(tok.pose), grip)
    elif isinstance(tok, MoveAbove):
        if held:
            run.converge(thing, _above(env, tok.target, tok.offset), grip)
        else:
            p = _above(env, tok.target, tok.offset)
            p[3:] = _ee(env)[3:]
            run.converge(ee, p, grip)
    elif isinstance(tok, Retreat):
        p = _ee(env)
        p[2] += tok.distance
        run.converge(ee, p, grip)
    elif isinstance(tok, Release):
        for _ in range(round(0.5 / run.dt)):
            run.tick(_ee(env), 1.0)
        env._held_tf = None
        return None
    elif isinstance(tok, Align):
        p = np.array(resolve_target(tok.target, env.product), dtype=np.float64)
        p[2] = _part(env, held)[2]
        run.converge(thing, p, grip, ALIGN_POS_TOL, ALIGN_ROT_TOL, yaw_only=True)
    elif isinstance(tok, Insert):
        tgt = np.array(resolve_target(tok.target, env.product), dtype=np.float64)
        z0, tol = _part(env, held)[2], env.product.tol_pos

        def check():
            if _at(env, held, tgt, rot=False) and z0 - _part(env, held)[2] >= tok.depth - tol:
                return True  # arrival wins over residual contact force
            if contact_force(env, held) > tok.max_force:
                raise _Fail("force limit")
            return False

        _descent(run, env, thing, held, tok.target, z0, check)
        if not check():
            raise _Fail("timeout")
    elif isinstance(tok, Place):
        run.converge(thing, _above(env, tok.target, 0.02), grip)
        tgt = np.array(resolve_target(tok.target, env.product), dtype=np.float64)
        _descent(run, env, thing, held, tok.target, tgt[2] + 0.02, lambda: _at(env, held, tgt))
        if not _at(env, held, tgt):
            raise _Fail("timeout")
    elif isinstance(tok, Press):
        _press(run, tok)
        if not _at(env, tok.part, np.array(env.product.parts[tok.part].final_pose, dtype=np.float64)):
            raise _Fail("timeout")
    return held


def _budget(env, tok):
    return {Grasp: 8.0, Place: 6.0, Press: 5.0, Release: 1.0}.get(
        type(tok), tok.depth / INSERT_SPEED + 2 if isinstance(tok, Insert) else 4.0)


def run_token(env, token: Token, held: str | None) -> tuple[TokenResult, str | None]:
    run = _Run(env, _budget(env, token))
    try:
        held = _run(env, token, held, run)
    except _Fail as e:
        return TokenResult(False, str(e), run.n * run.dt), held
    return TokenResult(True, "", run.n * run.dt), held


def execute(env, recipe: Recipe, record: bool = False, fps: int = 25) -> ExecutionLog:
    log, held = ExecutionLog(), None
    env._held_tf = None
    if record:
        hz = 1 / (env.substeps * env.model.opt.timestep)
        env._rec = {"r": mujoco.Renderer(env.model, 480, 640), "frames": log.frames,
                    "every": max(1, round(hz / fps)), "i": 0}
    try:
        for i, tok in enumerate(recipe.steps):
            res, held = run_token(env, tok, held)
            log.results.append(res)
            if not res.ok:
                log.stopped_at = i
                break
    finally:
        rec = getattr(env, "_rec", None)
        if rec is not None:
            rec["r"].close()
            env._rec = None
    return log
