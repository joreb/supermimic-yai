"""Benchmark env throughput: python -m tools.bench_env <product_dir> [--workers N] [--steps S] [--arm arms/panda]"""
import argparse
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def work(args):
    product_dir, arm_dir, steps = args
    from supermimic_yai.arm import load_arm
    from supermimic_yai.env import AssemblyEnv
    from supermimic_yai.product import load_product
    arm = load_arm(Path(arm_dir))
    env = AssemblyEnv(arm, load_product(Path(product_dir)))
    env.reset(seed=0)
    t0 = time.perf_counter()
    for s in range(1, 6):
        env.reset(seed=s)
    reset_t = (time.perf_counter() - t0) / 5
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for i in range(steps):
        if i % 100 == 0:
            env.reset(seed=1000 + i)
        a = np.append(arm.home_qpos + 0.02 * rng.standard_normal(env.n_arm), 1.0)
        env.step(a)
    dt = time.perf_counter() - t0
    return steps / dt, reset_t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("product_dir")
    ap.add_argument("--arm", default="arms/panda")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, [(a.product_dir, a.arm, a.steps)] * a.workers))
    for i, (sps, rt, rss) in enumerate(res):
        print(f"worker {i}: {sps:.1f} steps/s, reset {rt * 1000:.1f} ms, RSS {rss:.0f} MB")
    print(f"total: {sum(r[0] for r in res):.1f} steps/s over {a.workers} workers")


if __name__ == "__main__":
    main()
