"""Deterministic geometry variants of Product A. Run:
python tools/make_variants.py --n 32 --seed 0 --out products_train/"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from make_products import product_a_params  # noqa: E402

ROOT = Path(__file__).parent.parent
KEYS = ["peg_d", "clearance", "peg_len", "base_t", "cover_t", "spacing"]


def sample(n, seed, cfg_path=ROOT / "configs" / "variants.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    rng = np.random.default_rng(seed)
    return [{k: round(float(rng.uniform(*cfg[k])), 6) for k in KEYS} for _ in range(n)]


def variant_recipe(name, p):
    """A's recipe; the numbers that depend on geometry are recomputed from the params."""
    r = json.loads((ROOT / "recipes" / "product_a.json").read_text())
    r["product"] = name
    for s in r["steps"]:
        if s["token"] == "MOVE_TO" and s["pose"][2] == 0.12 and s["pose"][1] == -0.22:
            s["pose"][2] = round(p["peg_len"] + 0.08, 6)  # hover 80 mm over the peg tops
        if s["token"] == "GRASP" and s["part"] == "cover":
            s["grasp_pose"][2] = round(p["cover_t"] / 2 + 0.004, 6)  # 4 mm above the cover top
    return r


def write_variant(i, p, out):
    name = f"a_v{i:03d}"
    product_a_params(**p, name=name, out=out)
    (out / name / "recipe.json").write_text(json.dumps(variant_recipe(name, p), indent=1))
    return name


def check(args):
    """Scripted run on seeds.TRAIN[0]; returns (name, ok, detail)."""
    from supermimic_yai import seeds
    from supermimic_yai.arm import load_arm
    from supermimic_yai.env import AssemblyEnv
    from supermimic_yai.product import load_product
    from supermimic_yai.score import score
    from supermimic_yai.scripted import execute
    from supermimic_yai.tokens import describe, load_recipe
    d = Path(args)
    arm = load_arm(ROOT / "arms" / "panda")
    try:
        product = load_product(d)
        recipe = load_recipe(d / "recipe.json", product, arm)
    except Exception as e:  # invalid variant (e.g. too wide to grasp)
        return d.name, False, f"load: {e}"
    env = AssemblyEnv(arm, product)
    env.reset(seed=seeds.TRAIN[0])
    log = execute(env, recipe)
    if log.stopped_at is not None:
        return d.name, False, f"step {log.stopped_at} ({describe(recipe)[log.stopped_at]}): {log.results[-1].reason}"
    return d.name, score(product, env.obs()["part_poses"]).success, "score"


def make(n, seed, out, verify=True, workers=None):
    out = Path(out)
    params = sample(n, seed)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = [write_variant(i, p, out) for i, p in enumerate(params)]
    if not verify:
        return names, []
    with ProcessPoolExecutor(workers or min(4, os.cpu_count() or 1)) as pool:
        res = list(pool.map(check, [str(out / nm) for nm in names]))
    rejected = [(nm, detail) for nm, ok, detail in res if not ok]
    for nm, _ in rejected:
        shutil.rmtree(out / nm)
    return [nm for nm, ok, _ in res if ok], rejected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="products_train/")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    kept, rejected = make(a.n, a.seed, a.out, not a.no_verify)
    print(f"kept {len(kept)} rejected {len(rejected)}")
    allp = sample(a.n, a.seed)
    for nm, why in rejected:
        print(nm, allp[int(nm[3:])], why)
