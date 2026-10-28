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
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"s{i}": s.state for i, s in enumerate(states)})
    meta = [{"product": s.product, "step": s.step, "token": s.token, "held": s.held, "context": _ctx_json(s.context)}
            for s in states]
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_bank(path):
    path = Path(path).with_suffix(".npz")
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as z:
        return [StartState(m["product"], m["step"], m["token"], m["held"], _ctx_from(m["context"]), z[f"s{i}"])
                for i, m in enumerate(meta)]


def resolve_recipe(pdir):
    """<dir>/recipe.json if present, else recipes/<dir name>.json beside the products directory."""
    d = Path(pdir)
    return d / "recipe.json" if (d / "recipe.json").exists() else d.parent.parent / "recipes" / f"{d.name}.json"


def main(argv=None):
    import argparse

    from supermimic_yai import seeds as seedmod
    from supermimic_yai.arm import load_arm
    from supermimic_yai.product import load_product
    from supermimic_yai.tokens import load_recipe
    ap = argparse.ArgumentParser(description="Snapshot start states for several products into one bank.")
    ap.add_argument("--products", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, required=True, help="number of TRAIN seeds per product")
    ap.add_argument("--out", required=True, help="bank path, ending in .npz")
    ap.add_argument("--arm", required=True, help="arm directory")
    a = ap.parse_args(argv)
    arm, states = load_arm(Path(a.arm)), []
    for d in a.products:
        product = load_product(Path(d))
        recipe = load_recipe(resolve_recipe(d), product, arm)
        states += collect_start_states(arm, product, recipe, seedmod.TRAIN[:a.seeds])
        print(f"{product.name}: {len(states)} states so far", flush=True)
    save_bank(a.out, states)


if __name__ == "__main__":
    main()
