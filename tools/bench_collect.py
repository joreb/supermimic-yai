"""Scripted-noise collection benchmark: python -m tools.bench_collect [--episodes 200] [--config laptop] [--seeds 5]"""
import argparse
import collections
import time
from pathlib import Path

from supermimic_yai import seeds
from supermimic_yai.arm import load_arm
from supermimic_yai.collect import Collector
from supermimic_yai.config import load_config
from supermimic_yai.product import load_product
from supermimic_yai.startbank import collect_start_states, resolve_recipe, save_bank
from supermimic_yai.task import TOKEN_KINDS
from supermimic_yai.tokens import load_recipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--config", default="laptop")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--product", default="products/product_a")
    ap.add_argument("--arm", default="arms/panda")
    a = ap.parse_args()
    arm, prod = load_arm(Path(a.arm)), Path(a.product)
    product = load_product(prod)
    bank = collect_start_states(arm, product, load_recipe(resolve_recipe(prod), product, arm), seeds.TRAIN[:a.seeds])
    print(f"bank {len(bank)} states", flush=True)
    save_bank("out/bench_bank.npz", bank)
    with Collector(load_config(a.config), a.arm, [prod], "out/bench_bank.npz") as c:
        t0 = time.perf_counter()
        eps = c.collect(a.episodes, None, rng_seed=0)
        dt = time.perf_counter() - t0
    steps = sum(len(e["a"]) for e in eps)
    print(f"{len(eps)} episodes, {steps} transitions, {dt:.1f} s wall (incl. env build) -> {steps / dt:.0f} transitions/s")
    for k, name in enumerate(TOKEN_KINDS):
        sub = [e for e in eps if e["kind"] == k]
        why = collections.Counter(e["terminal_kind"] for e in sub)
        print(f"{name}: n={len(sub)} success={why['success'] / max(1, len(sub)):.2f} {dict(why)}")


if __name__ == "__main__":
    main()
