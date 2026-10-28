"""Multi-process token-episode collector. Workers are spawned and never import torch (policy weights are numpy).

Episode dict (one token episode, T transitions):
  s (T+1, s_dim) token states | a (T, a_dim) learner actions actually executed (noise included; a[:n] in [-1, 1]
  scaled by action_scale, a[n] gripper) | r (T,) | done (T,) bool, True only on the last step | unstable (T,) bool
  kind int index into TOKEN_KINDS | goal (g_dim,) | ctx (c_dim,) involved_context (world model / critic only)
  terminal_kind str: success / unstable / dropped / force / timeout (analytic, done=True on the last step), or for a
  scripted run that ended first: "script_abort:<reason>" (scripted failed) / "script_ok_unregistered" (scripted ok,
  success() not yet true). Those are truncations: done stays False everywhere.
  noise float: sigma of the arm-action noise for this episode, drawn from cfg collect.action_noise (a list)
  product str (name)
Scripted seed data (policy_weights=None, ruling R2): run_token drives the env through a wrapper that turns each absolute
arm target into a learner action, adds N(0, action_noise^2), and executes it through TokenEpisode.step (so state,
reward and terminal conditions are exactly the policy-rollout ones). The gripper channel gets no noise in either path.
Policy rollouts feed mlp_forward(weights, [s, goal]) (+ the same arm noise).
"""
import json
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path

import numpy as np

from supermimic_yai.npmlp import mlp_forward
from supermimic_yai.startbank import load_bank, resolve_recipe
from supermimic_yai.task import TOKEN_KINDS, TokenEpisode, budget_steps, involved_context

_G = {}  # per-worker state


class _Stop(Exception):
    pass


class _Recorder:
    """Looks like the env to run_token; every env.step becomes a noisy learner action through the TokenEpisode."""

    def __init__(self, ep, noise, rng, log):
        object.__setattr__(self, "ep", ep)
        object.__setattr__(self, "_p", (noise, rng, log))

    def __getattr__(self, k):
        return getattr(self.ep.env, k)

    def __setattr__(self, k, v):  # executor stashes (_down_quat, ...) on the env
        setattr(self.ep.env, k, v)

    def step(self, action):
        ep, (noise, rng, log) = self.ep, self._p
        n, action = ep.n, np.asarray(action, dtype=np.float64)
        a = np.clip((action[:n] - ep.s[:n]) / ep.scale + noise * rng.standard_normal(n), -1.0, 1.0)
        a = np.append(a, np.clip(action[n], 0.0, 1.0))
        out = ep.step(a)
        log(a, *out)
        if out[2]:
            raise _Stop
        return None, 0.0, False, False, {}


def run_episode(env, tok, start, scale, noise, rng, weights=None):
    """-> episode dict, or None if it had no steps."""
    from supermimic_yai.scripted import run_token
    ep = TokenEpisode(env, start, tok, budget_steps(env, tok), scale)
    S, rec = [ep.reset()], {"a": [], "r": [], "done": [], "unstable": []}
    why, res = [None], None

    def log(a, s2, r, done, info):
        S.append(s2)
        rec["a"].append(a), rec["r"].append(r), rec["done"].append(done)
        rec["unstable"].append(info.get("terminal_kind") == "unstable")
        if done:
            why[0] = info["terminal_kind"]

    if weights is None:
        try:
            res, _ = run_token(_Recorder(ep, noise, rng, log), tok, start.held)
        except _Stop:
            pass
    else:
        done = False
        while not done:
            a = mlp_forward(weights, np.concatenate([ep.s, ep.goal]))
            a[:ep.n] = np.clip(a[:ep.n] + noise * rng.standard_normal(ep.n), -1.0, 1.0)
            a[ep.n] = np.clip(a[ep.n], 0.0, 1.0)
            _, r, done, info = ep.step(a)
            log(a, ep.s, r, done, info)
    if not rec["a"]:
        return None
    if why[0] is None:
        why[0] = f"script_abort:{res.reason}" if not res.ok else "script_ok_unregistered"
    return {"s": np.array(S), "a": np.array(rec["a"]), "r": np.array(rec["r"]), "done": np.array(rec["done"]),
            "unstable": np.array(rec["unstable"]), "kind": TOKEN_KINDS.index(tok.kind), "goal": ep.goal,
            "ctx": involved_context(start.context, start.held, env.part_order), "terminal_kind": why[0],
            "noise": noise, "product": start.product}


def _init(arm_dir, products, envs_per_worker, scale):
    _G.update(arm_dir=arm_dir, products=products, cap=envs_per_worker, scale=scale, envs=OrderedDict())


def _env_for(name):
    envs = _G["envs"]
    if name in envs:
        envs.move_to_end(name)
        return envs[name]
    from supermimic_yai.arm import load_arm
    from supermimic_yai.env import AssemblyEnv
    from supermimic_yai.product import load_product
    from supermimic_yai.tokens import load_recipe
    while len(envs) >= _G["cap"]:  # LRU (ruling R1); evict before building so memory never exceeds the cap
        envs.popitem(last=False)
    arm = load_arm(Path(_G["arm_dir"]))
    pdir, rpath = _G["products"][name]
    product = load_product(Path(pdir))
    envs[name] = (AssemblyEnv(arm, product), load_recipe(Path(rpath), product, arm))
    _G["peak"] = max(_G.get("peak", 0), len(envs))
    return envs[name]


def _run_chunk(args):
    jobs, weights = args
    out = []
    for start, seed, noise in jobs:
        env, recipe = _env_for(start.product)
        out.append(run_episode(env, recipe.steps[start.step], start, _G["scale"], noise,
                               np.random.default_rng(seed), weights))
    return out


def _probe(args):
    """Test hook: run the chunk, hold this worker briefly so others get tasks, report (pid, torch imported, envs, peak)."""
    import os
    import sys
    import time
    jobs, weights, noise = args
    _run_chunk(([(s, sd, noise) for s, sd in jobs], weights))
    time.sleep(0.5)
    return os.getpid(), "torch" in sys.modules, len(_G["envs"]), _G["peak"]


class Collector:
    def __init__(self, cfg, arm_dir, dirs, bank_path):
        self.noises, self.workers = cfg["collect"]["action_noise"], cfg["collect"]["workers"]
        products = {json.loads((Path(d) / "assembly.json").read_text())["name"]: (str(d), str(resolve_recipe(d)))
                    for d in dirs}
        self.bank = [s for s in load_bank(bank_path) if s.product in products]
        if not self.bank:
            raise ValueError("start-state bank has no states for these products")
        self.pool = mp.get_context("spawn").Pool(
            self.workers, _init, (str(arm_dir), products, cfg["collect"]["envs_per_worker"], cfg["action_scale"]))

    def collect(self, n_episodes, policy_weights=None, rng_seed=0, sampler=None):
        """sampler(rng, bank) -> index into bank; default uniform (Plan 2b swaps in return-based reset sampling)."""
        rng = np.random.default_rng(rng_seed)
        pick = sampler or (lambda r, bank: int(r.integers(len(bank))))
        seeds = rng.integers(0, 2**63 - 1, n_episodes)
        jobs = sorted(((self.bank[pick(rng, self.bank)], int(sd), float(rng.choice(self.noises))) for sd in seeds),
                      key=lambda j: j[0].product)
        m = max(1, -(-n_episodes // (self.workers * 4)))  # same-product jobs stay together so envs are reused
        res = self.pool.map(_run_chunk, [(jobs[i:i + m], policy_weights) for i in range(0, len(jobs), m)])
        return [e for chunk in res for e in chunk if e is not None]

    def close(self):
        self.pool.terminate()

    __enter__ = lambda self: self  # noqa: E731
    __exit__ = lambda self, *a: self.close()  # noqa: E731
