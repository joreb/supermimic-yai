"""Per-token and full-recipe evaluation of the learned controller against the scripted one (EVAL seeds, zero noise)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from supermimic_yai import seeds as seedmod
from supermimic_yai.collect import Collector
from supermimic_yai.startbank import load_bank, resolve_recipe


def _name(d):
    return json.loads((Path(d) / "assembly.json").read_text())["name"]


def eval_tokens(cfg, arm_dir, pdirs, bank_path, weights, n_per_kind, rng_seed=0, learned_kinds=("GRASP",)):
    """-> {(product, kind): {"success", "n", "terminal"}}. Kinds outside learned_kinds run scripted (weights=None)."""
    names = [_name(d) for d in pdirs]
    kinds = sorted({s.token for s in load_bank(bank_path) if s.product in names})
    out = {}
    for kind in kinds:
        w = weights if kind in learned_kinds else None
        with Collector(cfg, arm_dir, pdirs, bank_path, kinds=[kind]) as c:
            rng, picks = np.random.default_rng(rng_seed), []
            for p in names:  # exactly n_per_kind distinct start states per product, or all of them if fewer
                idx = [i for i, s in enumerate(c.bank) if s.product == p]
                picks += list(rng.choice(idx, min(n_per_kind, len(idx)), replace=False)) if idx else []
            it = iter(picks)
            eps = c.collect(len(picks), w, rng_seed=rng_seed, sampler=lambda r, bank: int(next(it)), noise=[0.0])
        for p in names:
            cnt = Counter(e["terminal_kind"] for e in eps if e["product"] == p)
            n = sum(cnt.values())
            if n:
                out[(p, kind)] = {"success": cnt["success"] / n, "n": n, "terminal": dict(cnt)}
    return out


def eval_recipes(arm_dir, pdirs, weights, seeds, record_first=False, action_scale=0.05, learned_kinds=("GRASP",)):
    """-> {product: {"success", "per_token", "frames"}}; weights=None runs the plain scripted recipe."""
    from supermimic_yai.arm import load_arm
    from supermimic_yai.env import AssemblyEnv
    from supermimic_yai.product import load_product
    from supermimic_yai.runner import LearnedController
    from supermimic_yai.score import score
    from supermimic_yai.scripted import execute
    from supermimic_yai.tokens import load_recipe
    seeds = list(seeds)
    if bad := [s for s in seeds if s not in seedmod.EVAL]:
        raise ValueError(f"seeds outside seeds.EVAL: {bad}")
    arm, out = load_arm(Path(arm_dir)), {}
    for d in pdirs:
        product = load_product(Path(d))
        recipe, env = load_recipe(resolve_recipe(d), product, arm), AssemblyEnv(arm, product)
        ok, per, frames = 0, np.zeros(len(recipe.steps)), None
        for i, s in enumerate(seeds):
            env.reset(seed=s)
            ctrl = None if weights is None else LearnedController(weights, action_scale, learned_kinds)
            log = execute(env, recipe, record=record_first and i == 0, controller=ctrl)
            n_ok = sum(r.ok for r in log.results)
            per[:n_ok] += 1
            ok += log.stopped_at is None and score(product, env.obs()["part_poses"]).success
            if record_first and i == 0:
                frames = log.frames
        env.close()
        out[product.name] = {"success": ok / len(seeds), "per_token": list(per / len(seeds)), "frames": frames}
    return out


def results_markdown(tokens, recipes, scripted_kinds=("PLACE", "PRESS"), learned_kinds=("GRASP",)):
    """Tables with a controller column: 'learned' only for kinds in learned_kinds (and never a scripted kind)."""
    ctl = lambda k: "learned" if k in learned_kinds and k not in scripted_kinds else "scripted"  # noqa: E731
    rows = ["| product | token kind | controller | success | n |", "|---|---|---|---|---|"]
    rows += [f"| {p} | {k} | {ctl(k)} | {v['success']:.2f} | {v['n']} |" for (p, k), v in sorted(tokens.items())]
    rows += [f"| {p} | {k} | scripted | - | - |" for p in sorted({p for p, _ in tokens}) for k in scripted_kinds]
    rows += ["", "| product | full-recipe success | per-token ok |", "|---|---|---|"]
    rows += [f"| {p} | {v['success']:.2f} | {' '.join(f'{x:.2f}' for x in v['per_token'])} |" for p, v in sorted(recipes.items())]
    return "\n".join(rows) + "\n"


def log_video(writer, tag, frames, step, fps=25):
    """uint8 frames (T, H, W, C) -> video summary, downscaled to 240x320.
    Skips SummaryWriter.add_video: its _prepare_video calls np.reshape(newshape=), removed in numpy 2."""
    import torch
    import torch.nn.functional as F
    from tensorboard.compat.proto.summary_pb2 import Summary
    from torch.utils.tensorboard.summary import make_video
    x = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(240, 320), mode="area").round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    writer._get_file_writer().add_summary(Summary(value=[Summary.Value(tag=tag, image=make_video(x, fps))]), step)
