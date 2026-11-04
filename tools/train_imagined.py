"""Imagination-only actor-critic in the frozen world model, then real-sim eval.
python tools/train_imagined.py --config laptop --wm runs/wm --kinds GRASP --iters 5000 --logdir runs/imagined --eval-bank runs/eval/bank_a.npz

Torch is imported only after the first Collector pool exists (workers are spawned from this module and must never
import torch). The scripted reference (policy_weights=None) runs before training; the learned actor runs after.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from supermimic_yai.collect import Collector
from supermimic_yai.replay import buffer_from, load_episodes


def evaluate(cfg, arm, products, bank, kind, n, weights):
    with Collector(cfg, arm, products, bank, kinds=[kind]) as c:
        eps = c.collect(n, weights, rng_seed=0, noise=[0.0])
    cnt = Counter(e["terminal_kind"] for e in eps)
    return cnt["success"] / len(eps), dict(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="laptop")
    ap.add_argument("--wm", default="runs/wm")
    ap.add_argument("--kinds", nargs="+", default=["GRASP"])
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--logdir", default="runs/imagined")
    ap.add_argument("--eval-bank", default="runs/eval/bank_a.npz")
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--no-ref", action="store_true", help="skip the scripted reference eval")
    ap.add_argument("--products", nargs="+", default=["products/product_a"])
    ap.add_argument("--arm", default="arms/panda")
    a = ap.parse_args()
    from supermimic_yai.config import load_config
    cfg, out = load_config(a.config), Path(a.logdir)
    out.mkdir(parents=True, exist_ok=True)
    ref = {k: evaluate(cfg, a.arm, a.products, a.eval_bank, k, a.eval_episodes, None) for k in a.kinds if not a.no_ref}
    for k, (sr, cnt) in ref.items():
        print(f"scripted {k}: success {sr:.3f} {cnt}", flush=True)

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from supermimic_yai.imagine import R_MAX
    from supermimic_yai.improve import Learner, out_of_bounds
    from supermimic_yai.npmlp import save_actor_npz
    from supermimic_yai.task import TOKEN_KINDS
    from supermimic_yai.wm import load_wm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load_episodes(Path(a.wm) / "episodes.npz")
    buf = buffer_from(eps)
    sd, ad, gd, cd = buf.s.shape[1], buf.a.shape[1], buf.goal.shape[1], buf.ctx.shape[1]
    wm = load_wm(Path(a.wm) / "model.pt", sd, ad, cd, dev)
    wm.requires_grad_(False)
    L, rng = Learner(cfg, sd, ad, gd, cd, wm, dev), np.random.default_rng(0)
    L.set_norm(buf)
    kinds, name = tuple(TOKEN_KINDS.index(k) for k in a.kinds), "+".join(a.kinds)
    tw = SummaryWriter(str(out))
    t0 = time.perf_counter()
    for it in range(cfg["policy"]["bc_iters"]):
        m = L.bc_update(buf, rng, kinds)
        if it % 100 == 0:
            tw.add_scalar("policy/bc_loss", m["bc_loss"], it)
    curve, over, bound = [], 0, cfg["policy"]["horizon"]
    for it in range(1, a.iters + 1):
        m = L.update(buf, rng, kinds)
        curve.append(m["imagined_return"])
        over = over + 1 if out_of_bounds(m, cfg["policy"]["horizon"]) else 0
        if over >= 20:
            print(f"ABORT it {it}: reward sum {m['imagined_reward_sum']:.3g} (max {bound * R_MAX}) or mean V {m['v_mean']:.3g} "
                  f"(range {m['v_lo']:.3g}..{m['v_hi']:.3g}) out of bounds for 20 iterations", flush=True)
            save_actor_npz(out / "actor_abort.npz", L.actor.export())
            torch.save(L.actor.state_dict(), out / "actor_abort.pt")
            raise SystemExit(1)
        if it % 50 == 0:
            tw.add_scalar(f"policy/imagined_return/{name}", m["imagined_return"], it)
            tw.add_scalar(f"policy/imagined_reward_sum/{name}", m["imagined_reward_sum"], it)
            tw.add_scalar("policy/grad_norm", m["grad_norm"], it)
            tw.add_scalar("policy/clamp_frac", m["clamp_frac"], it)
            tw.add_scalar("policy/held_frac", m["held_frac"], it)
            tw.add_scalar("critic/loss", m["critic_loss"], it)
            tw.add_scalar("sys/it_per_s", it / (time.perf_counter() - t0), it)
            tw.add_scalar("sys/gpu_mem_mb", torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0, it)
        if it % 500 == 0:
            save_actor_npz(out / f"actor_{it}.npz", L.actor.export())
            print(f"it {it} return {np.mean(curve[-50:]):.3f} rsum {m['imagined_reward_sum']:.2f} held>.5 {m['held_frac']:.2f} clamp {m['clamp_frac']:.3f} critic {m['critic_loss']:.4f} gn {m['grad_norm']:.2f}", flush=True)
    tw.close()
    train_wall = time.perf_counter() - t0
    torch.save(L.actor.state_dict(), out / "actor.pt")
    w = L.actor.export()
    save_actor_npz(out / "actor.npz", w)
    if curve:
        print(f"imagined return start {np.mean(curve[:50]):.3f} end {np.mean(curve[-50:]):.3f}; train wall {train_wall:.0f} s")
    for k in a.kinds:
        sr, cnt = evaluate(cfg, a.arm, a.products, a.eval_bank, k, a.eval_episodes, w)
        print(f"learned {k}: success {sr:.3f} {cnt} | scripted {ref[k][0] if k in ref else float('nan'):.3f}")
    print(f"total wall {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
