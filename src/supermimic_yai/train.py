"""Dyna training loop: collect real-sim episodes asynchronously while the GPU updates the world model and the actor/critic.

python -m supermimic_yai.train --config laptop --init runs/wm --arm <arm dir> --products <train product dirs> \
  --bank runs/wm/bank.npz --eval-banks runs/eval/bank_a.npz runs/eval/bank_b.npz runs/eval/bank_c.npz \
  --heldout <held-out product dirs> --logdir runs/train --iters N [--resume]

Torch is imported inside run(): spawned collector workers re-import this module (as __main__) and must never import torch.
--resume restores actor, critic, world model, optimisers, sampler and iteration from <logdir>/ckpt.pt; replay restarts from the
--init episodes. Held-out products are only ever evaluated on, never trained on.
"""
import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import yaml

from supermimic_yai import seeds as seedmod
from supermimic_yai.collect import Collector
from supermimic_yai.config import load_config
from supermimic_yai.curriculum import ResetSampler
from supermimic_yai.replay import ReplayBuffer, load_episodes
from supermimic_yai.startbank import load_bank
from supermimic_yai.task import TOKEN_KINDS

TRAINED = ("GRASP", "ALIGN", "INSERT")  # PLACE/PRESS stay scripted (ruling P1)
ABORT_STREAK = 20  # consecutive out-of-bounds updates of one kind


def _names(d):
    return {Path(d).name, json.loads((Path(d) / "assembly.json").read_text())["name"]}


def check_heldout(products, heldout, bank):
    held = set().union(*map(_names, heldout))
    if bad := sorted(held & set().union(*map(_names, products))):
        raise SystemExit(f"refusing to train: held-out product(s) {bad} appear in --products")
    if bad := sorted({s.product for s in load_bank(bank)} & held):
        raise SystemExit(f"refusing to train: --bank has start states of held-out product(s) {bad}")


def _evaluate(it, cfg, a, dirs, w, tw, record):
    from supermimic_yai.evaluate import eval_recipes, eval_tokens, log_video
    t, arm, learned = cfg["train"], a.arm, TRAINED
    if it % t["eval_every"] == 0:
        for bank in a.eval_banks:
            for (p, k), v in eval_tokens(cfg, arm, dirs, bank, w, t["eval_n"], 0, learned).items():
                tw.add_scalar(f"eval/token_success/{p}/{k}", v["success"], it)
    seeds = seedmod.EVAL[:t["eval_recipe_seeds"]]
    for i, d in enumerate([a.products[0], *a.heldout]):
        r = eval_recipes(arm, [d], w, seeds, record_first=record and i < 2, action_scale=cfg["action_scale"], learned_kinds=learned)
        for p, v in r.items():
            tw.add_scalar(f"eval/recipe_success/{p}", v["success"], it)
            if v["frames"]:
                log_video(tw, f"eval/video/{p}", v["frames"], it)
    tw.flush()


def run(a):
    def _term(*_):
        raise KeyboardInterrupt  # docker stop: unwind through the loop's finally, which saves the last completed iteration
    old = signal.signal(signal.SIGTERM, _term)
    try:
        return _run(a)
    finally:
        signal.signal(signal.SIGTERM, old)


def _run(a):
    cfg = load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        *path, last = k.split(".")
        d = cfg
        for p in path:
            d = d[p]
        d[last] = yaml.safe_load(v)
    check_heldout(a.products, a.heldout, a.bank)
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from supermimic_yai.improve import Learner, out_of_bounds
    from supermimic_yai.npmlp import save_actor_npz
    from supermimic_yai.wm import load_wm, wm_loss
    t, w_cfg, p_cfg = cfg["train"], cfg["wm"], cfg["policy"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.logdir)
    out.mkdir(parents=True, exist_ok=True)
    eps = load_episodes(Path(a.init) / "episodes.npz")
    sd, ad, gd, cd = eps[0]["s"].shape[1], eps[0]["a"].shape[1], eps[0]["goal"].shape[0], eps[0]["ctx"].shape[0]
    buf = ReplayBuffer(cfg["replay"]["capacity"], sd, ad, gd, cd)
    for e in eps:
        buf.add(e)
    wm = load_wm(Path(a.init) / "model.pt", sd, ad, cd, dev)
    opt_w = torch.optim.Adam(wm.parameters(), w_cfg["lr"])
    L, rng = Learner(cfg, sd, ad, gd, cd, wm, dev), np.random.default_rng(0)
    L.set_norm(buf)
    kinds = [TOKEN_KINDS.index(k) for k in TRAINED]
    tw = SummaryWriter(str(out))
    dirs = list(dict.fromkeys([*a.products, *a.heldout]))
    start = 0
    ckpt_path = out / "ckpt.pt"

    def save_ckpt(it):
        torch.save({"it": it, "actor": L.actor.state_dict(), "critic": L.critic.state_dict(), "wm": wm.state_dict(),
                    "opt": [o.state_dict() for o in (L.opt_a, L.opt_c, opt_w)], "sched": [s.state_dict() for s in L.sched],
                    "sampler": sampler.state_dict()}, ckpt_path)

    with Collector(cfg, a.arm, a.products, a.bank, kinds=TRAINED) as coll:
        sampler = ResetSampler(coll.bank, p_cfg["gamma"])
        if a.resume:
            ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
            L.actor.load_state_dict(ck["actor"]), L.critic.load_state_dict(ck["critic"]), wm.load_state_dict(ck["wm"])
            for o, s in zip((L.opt_a, L.opt_c, opt_w), ck["opt"]):
                o.load_state_dict(s)
            for o, s in zip(L.sched, ck["sched"]):
                o.load_state_dict(s)
            sampler.load_state_dict(ck["sampler"])
            start = ck["it"]
        else:  # every fresh run starts from a working (behaviour-cloned) actor (controller ruling 1)
            for i in range(p_cfg["bc_iters"]):
                m = L.bc_update(buf, rng, kinds)
                if i % 100 == 0:
                    tw.add_scalar("policy/bc_loss", m["bc_loss"], i)
            save_actor_npz(out / "actor_bc.npz", L.actor.export())
        tens = lambda b: {k: torch.as_tensor(v, device=dev, dtype=torch.int64 if k == "kind" else torch.float32)  # noqa: E731
                          for k, v in b.items() if k in ("s", "a", "ctx", "kind", "done")}
        t0, streak, it, aborted = time.perf_counter(), dict.fromkeys(kinds, 0), start, False
        done = start
        try:
            for it in range(start + 1, a.iters + 1):
                wts = L.actor.export()
                tc = time.perf_counter()
                pending = coll.collect_async(t["episodes_per_iter"], wts, rng_seed=it, sampler=sampler)
                n, last = 0, {}
                while n < t["max_updates"] and (n < t["min_updates"] or not pending.ready()):
                    wm.requires_grad_(True)
                    wl, _ = wm_loss(wm, tens(buf.sample_windows(w_cfg["k"], w_cfg["batch"], rng)), w_cfg["k"])
                    opt_w.zero_grad(set_to_none=True)
                    wl.backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), w_cfg["grad_clip"])
                    opt_w.step()
                    wm.requires_grad_(False)  # imagination must not leave gradients on the world model
                    k = kinds[n % len(kinds)]
                    m = last[k] = L.update(buf, rng, (k,))
                    streak[k] = streak[k] + 1 if out_of_bounds(m, p_cfg["horizon"]) else 0
                    if streak[k] >= ABORT_STREAK:  # diverged weights go only to actor_abort.npz; ckpt.pt keeps the last good state
                        aborted = True
                        save_actor_npz(out / "actor_abort.npz", L.actor.export())
                        raise SystemExit(f"ABORT it {it}: imagined reward sum or mean V out of bounds for {ABORT_STREAK} updates "
                                         f"(rsum {m['imagined_reward_sum']:.3g}, V {m['v_mean']:.3g} in {m['v_lo']:.3g}..{m['v_hi']:.3g})")
                    n += 1
                new = pending.get()
                collect_s = time.perf_counter() - tc
                for e in new:
                    buf.add(e)
                sampler.update(new)
                save_actor_npz(out / "actor.npz", L.actor.export())
                if it % t["log_every"] == 0:
                    steps = sum(len(e["a"]) for e in new)
                    for ki, k in zip(kinds, TRAINED):
                        es = [e for e in new if e["kind"] == ki]
                        if es:
                            tw.add_scalar(f"collect/success/{k}", np.mean([e["terminal_kind"] == "success" for e in es]), it)
                        if ki in last:
                            tw.add_scalar(f"policy/imagined_return/{k}", last[ki]["imagined_return"], it)
                            tw.add_scalar(f"policy/imagined_reward_sum/{k}", last[ki]["imagined_reward_sum"], it)
                    ms = list(last.values())
                    for tag, key in (("policy/loss", "policy_loss"), ("critic/loss", "critic_loss"), ("policy/grad_norm", "grad_norm"),
                                     ("policy/held_frac", "held_frac"), ("policy/clamp_frac", "clamp_frac")):
                        tw.add_scalar(tag, float(np.mean([m[key] for m in ms])), it)
                    with torch.no_grad():
                        _, wmm = wm_loss(wm, tens(buf.sample_windows(w_cfg["k"], 2048, rng)), w_cfg["k"], metrics=True)
                    tw.add_scalar("wm/loss", wl.item(), it)
                    for key, v in wmm.items():
                        if key.startswith("kstep_err_pt/"):
                            tw.add_scalar(f"wm/{key}", v, it)
                    for b, wt in sampler.weights().items():
                        tw.add_scalar(f"curriculum/weight/{b}", wt, it)
                    tw.add_scalar("sys/gpu_mem_mb", torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0, it)
                    tw.add_scalar("sys/it_per_s", (it - start) / (time.perf_counter() - t0), it)
                    tw.add_scalar("sys/utd", n / max(steps, 1), it)  # policy updates per collected transition
                    tw.add_scalar("sys/updates", n, it)
                    tw.add_scalar("sys/collect_s", collect_s, it)
                rec = it % t["video_every"] == 0
                if it % t["eval_every"] == 0 or rec:
                    _evaluate(it, cfg, a, dirs, L.actor.export(), tw, rec)
                done = it
                if it % t["ckpt_every"] == 0:
                    save_ckpt(it)
        finally:
            if not aborted:
                save_ckpt(done)
            tw.close()
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="laptop")
    ap.add_argument("--init", required=True, help="dir with episodes.npz and model.pt (Plan 2a world model)")
    ap.add_argument("--products", nargs="+", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--eval-banks", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--logdir", default="runs/train")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. policy.batch=512")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
