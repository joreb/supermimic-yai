import numpy as np
import torch

from supermimic_yai.imagine import gae_targets, imagine, policy_loss, value_range
from supermimic_yai.improve import Learner
from supermimic_yai.replay import ReplayBuffer
from supermimic_yai.wm import WorldModel

N_ARM = 7
SD, AD, GD, CD, NK = 2 * N_ARM + 17, N_ARM + 1, 3, 2 + N_ARM, 3
CFG = {"policy": {"z_dim": 8, "width": 32, "depth": 2, "lr": 3e-3, "betas": [0.9, 0.95], "grad_clip": 1.0,
                  "grad_clip_elem": 0.1, "gamma": 0.99, "horizon": 6, "batch": 16, "bc_iters": 0},
       "critic": {"width": 32, "depth": 2, "lam": 0.95, "epochs": 2, "minibatches": 2},
       "train": {"lr_decay_iters": 1000}}


def toy_buf():
    g = np.random.default_rng(0)
    buf = ReplayBuffer(16, SD, AD, GD, CD)
    for kind, T in ((0, 6), (1, 5), (2, 5)):
        goal = np.zeros(GD) if kind < 2 else np.array([0.01, 0.5, 0.002])
        buf.add({"s": g.normal(size=(T + 1, SD)), "a": np.clip(g.normal(size=(T, AD)), -1, 1) * 0.5 + np.r_[np.zeros(AD - 1), 0.5], "r": np.zeros(T),
                 "done": np.zeros(T, bool), "unstable": np.zeros(T, bool), "kind": kind, "goal": goal,
                 "ctx": g.normal(size=CD)})
    return buf


def learner(buf):
    torch.manual_seed(0)
    wm = WorldModel(SD, AD, CD, NK, 32, 2)
    lr = Learner(CFG, SD, AD, GD, CD, wm, "cpu")
    lr.set_norm(buf)
    return lr


def test_learner_update_improves_imagined_return():
    buf = toy_buf()
    lr = learner(buf)
    t = lambda x, dt=torch.float32: torch.as_tensor(x[:16], dtype=dt)  # noqa: E731
    s0, goal, ctx, kind = t(buf.s), t(buf.goal), t(buf.ctx), t(buf.kind, torch.int64)

    def ret():
        with torch.no_grad():
            return -policy_loss(imagine(lr.wm, lr.actor, s0, goal, ctx, kind, 6), lr.critic, goal, ctx, kind, 0.99).item()

    before, rng = ret(), np.random.default_rng(0)
    for _ in range(50):
        m = lr.update(buf, rng, (0, 1, 2))
    assert set(m) == {"held_frac", "v_mean", "v_lo", "v_hi", "imagined_reward_sum", "imagined_return", "policy_loss", "critic_loss", "grad_norm"}
    assert ret() > before


def test_learner_kind_filter():
    buf = toy_buf()
    lr = learner(buf)
    for kinds in ((0,), (1, 2)):
        i, *_, k = lr._sample(buf, np.random.default_rng(0), kinds)
        assert set(k.tolist()) <= set(kinds) and set(buf.kind[i]) == set(kinds) & {0, 1, 2}


def test_bc_update_reduces_mse():
    buf = toy_buf()
    lr = learner(buf)
    rng = np.random.default_rng(0)
    first = lr.bc_update(buf, rng, (0, 1, 2))["bc_loss"]
    for _ in range(200):
        last = lr.bc_update(buf, rng, (0, 1, 2))["bc_loss"]
    assert last < 0.5 * first


def test_imagined_objective_bounded():
    """A wildly drifting world model must not yield rewards above the env's physical maximum (1/step)."""
    buf = toy_buf()
    lr = learner(buf)
    lr.wm.ds_sd.fill_(50.0)  # huge residuals: unclamped states leave the physical range within a few steps
    t = lambda x, dt=torch.float32: torch.as_tensor(x[:16], dtype=dt)  # noqa: E731
    s0, goal, ctx, kind = t(buf.s), t(buf.goal), t(buf.ctx), t(buf.kind, torch.int64)
    out = imagine(lr.wm, lr.actor, s0, goal, ctx, kind, 6, *([lr.lo, lr.hi] if hasattr(lr, "lo") else []))
    assert out["r"].max() <= 1.0 and out["r"].sum(1).max() <= 6 * 1.0
    assert out["c"].min() >= 0 and out["c"].max() <= 1
    assert out["r"].min() > -50  # finite and bounded below too


def test_critic_clamped_to_value_range():
    buf = toy_buf()
    lr = learner(buf)
    t = lambda x, dt=torch.float32: torch.as_tensor(x[:16], dtype=dt)  # noqa: E731
    s0, goal, ctx, kind = t(buf.s), t(buf.goal), t(buf.ctx), t(buf.kind, torch.int64)
    out = imagine(lr.wm, lr.actor, s0, goal, ctx, kind, 6, lr.lo, lr.hi)
    lr.critic.net[-1].bias.data.fill_(1e6)  # a diverged critic
    lo, hi = value_range(kind, goal, 0.99)
    tg = gae_targets(out, lr.critic, goal, ctx, kind, 0.99, 0.95)
    assert (tg <= hi[:, None]).all() and (tg >= lo[:, None]).all()
    assert -policy_loss(out, lr.critic, goal, ctx, kind, 0.99).item() <= hi.max().item() + 1e-3
