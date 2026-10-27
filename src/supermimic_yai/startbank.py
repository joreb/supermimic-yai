"""Bank of start states snapshotted while the scripted recipe runs."""

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from supermimic_yai.context import SimContext
from supermimic_yai.env import START_SPEC, AssemblyEnv
from supermimic_yai.scripted import execute


@dataclass(frozen=True, eq=False)
class StartState:
    product: str
    step: int
    token: str
    held: str | None
    context: SimContext
    state: np.ndarray


def collect_start_states(arm, product, recipe, seeds, kinds=("GRASP", "ALIGN", "INSERT")):
    env, out = AssemblyEnv(arm, product), []

    def snap(env, i, tok, held):
        if tok.kind in kinds:
            s = np.empty(mujoco.mj_stateSize(env.model, START_SPEC))
            mujoco.mj_getState(env.model, env.data, s, START_SPEC)
            out.append(StartState(product.name, i, tok.kind, held, env.ctx, s))

    for seed in seeds:
        env.reset(seed=seed)
        execute(env, recipe, on_token_start=snap)  # a failed token just ends that seed's snapshots
    return out


def _ctx_json(c):
    return {"friction": c.friction, "mass_scale": c.mass_scale, "start_pose_noise": c.start_pose_noise,
            "joint_offsets": list(c.joint_offsets)}


def _ctx_from(d):
    return SimContext(d["friction"], d["mass_scale"], {k: tuple(v) for k, v in d["start_pose_noise"].items()},
                      tuple(d["joint_offsets"]))


def save_bank(path, states):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"s{i}": s.state for i, s in enumerate(states)})
    meta = [{"product": s.product, "step": s.step, "token": s.token, "held": s.held, "context": _ctx_json(s.context)}
            for s in states]
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_bank(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as z:
        return [StartState(m["product"], m["step"], m["token"], m["held"], _ctx_from(m["context"]), z[f"s{i}"])
                for i, m in enumerate(meta)]
