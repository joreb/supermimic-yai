import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from supermimic_yai.arm import load_arm
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import load_product
from supermimic_yai.score import score
from supermimic_yai.scripted import execute
from supermimic_yai.tokens import describe, load_recipe

ROOT = Path(__file__).parent.parent
ARM = load_arm(ROOT / "arms" / "panda")


def run_seed(env, recipe, seed):
    """Returns (success, detail) for one sample_context seed."""
    env.reset(seed=seed)
    log = execute(env, recipe)
    report = score(env.product, env._obs()["part_poses"])
    if log.stopped_at is not None:
        step = describe(recipe)[log.stopped_at]
        return False, f"step {log.stopped_at} ({step}): {log.results[-1].reason}"
    if not report.success:
        return False, "score: " + ", ".join(
            f"{s.part} {s.pos_err * 1000:.1f}mm/{s.rot_err_deg:.1f}deg" for s in report.parts if not s.ok)
    return True, "ok"


def _seed(args):
    x, seed = args
    product = load_product(ROOT / "products" / f"product_{x}")
    recipe = load_recipe(ROOT / "recipes" / f"product_{x}.json", product, ARM)
    return run_seed(AssemblyEnv(ARM, product), recipe, seed)


@pytest.mark.slow
@pytest.mark.parametrize("x", ["a", "b", "c"], ids=lambda x: f"product_{x}")
def test_reference_recipe_succeeds(x, capsys):
    with ProcessPoolExecutor(min(10, os.cpu_count() or 1)) as pool:  # one sim per seed, in parallel
        outcomes = list(pool.map(_seed, [(x, s) for s in range(10)]))
    with capsys.disabled():
        for s, (ok, detail) in enumerate(outcomes):
            print(f"\n[product_{x} seed {s}] {'PASS' if ok else 'FAIL'} {detail}", end="")
    assert sum(ok for ok, _ in outcomes) >= 8, outcomes
