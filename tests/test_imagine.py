import numpy as np
import pytest
import torch

from supermimic_yai import task
from supermimic_yai.imagine import (critic_loss, gae_targets, imagine, policy_loss, policy_step, reward_t,
                                    success_t)
from supermimic_yai.policy import Actor, Critic
from supermimic_yai.wm import WorldModel
from test_collect import a_bank, collector  # noqa: F401  (fixtures)

N_ARM = 7
SD, AD, GD, CD, NK = 2 * N_ARM + 17, N_ARM + 1, 3, 2 + N_ARM, 3


def _nets(seed=0):
    torch.manual_seed(seed)
    wm = WorldModel(SD, AD, CD, NK, 32, 2)
    return wm, Actor(SD, GD, AD, NK, 8, 32, 2), Critic(SD, GD, CD, NK, 32, 2)


def _batch(B=6):
    g = torch.Generator().manual_seed(1)
    s = torch.randn(B, SD, generator=g)
    goal = torch.rand(B, GD, generator=g) + 0.5
    goal[torch.arange(B) % NK < 2] = 0  # GRASP/ALIGN goals are zeros
    return s, goal, torch.randn(B, CD, generator=g), torch.arange(B) % NK


def test_reward_twin_parity(collector):  # noqa: F811
    c, _, _ = collector
    eps = c.collect(6, None, rng_seed=0)
    S, S2, G, K = [], [], [], []
    for e in eps:
        S.append(e["s"][:-1]), S2.append(e["s"][1:])
        G += [e["goal"]] * len(e["a"])
        K += [e["kind"]] * len(e["a"])
    S, S2, G, K = np.concatenate(S), np.concatenate(S2), np.array(G), np.array(K)
    i = np.random.default_rng(0).choice(len(S), 512)
    S, S2, G, K = S[i], S2[i], G[i], K[i]
    # mixed kinds: evaluate every kind on every state, as the twin must handle any pairing
    for k, name in enumerate(task.TOKEN_KINDS):
        kk = np.full(len(S), k)
        ref_s = [task.success(name, a, g) for a, g in zip(S, G)]
        t = lambda x: torch.as_tensor(x, dtype=torch.float64)  # noqa: E731
        g = t(G).clone()
        g[:, 0] = g[:, 0].clamp_min(1e-2)
        ref_r = [task.reward(name, a, b, gg) for a, b, gg in zip(S, S2, g.numpy())]
        assert np.allclose(reward_t(torch.as_tensor(kk), t(S), t(S2), g).numpy(), ref_r, atol=1e-5)
        assert (success_t(torch.as_tensor(kk), t(S), g).numpy() == np.array(ref_s)).all()
    # force the branches real data may not hit: INSERT force above its limit, rotated held part
    S2f = S2.copy()
    n = (S.shape[1] - 17) // 2
    S2f[:, 2 * n + 16] = 1 + np.random.default_rng(1).random(len(S))
    ang = np.random.default_rng(2).uniform(0.05, 1.0, len(S))
    qz = np.stack([np.cos(ang / 2), 0 * ang, 0 * ang, np.sin(ang / 2)], 1)
    S2f[:, 2 * n + 4:2 * n + 8] = qz
    g = G.copy()
    g[:, 0], g[:, 1] = 0.01, 0.5
    for k, name in enumerate(task.TOKEN_KINDS):
        ref = np.array([task.reward(name, a, b, gg) for a, b, gg in zip(S, S2f, g)])
        got = reward_t(torch.full((len(S),), k), torch.as_tensor(S), torch.as_tensor(S2f), torch.as_tensor(g))
        assert np.allclose(got.numpy(), ref, atol=1e-5)
        if name == "INSERT":  # penalty and yaw branches are active: differs from the no-penalty value
            base = np.array([task.reward(name, a, np.where(np.arange(len(b)) == 2 * n + 16, 0, b), gg)
                             for a, b, gg in zip(S, S2f, g)])
            assert (ref != base).all()
        if name == "ALIGN":
            assert task._align_err(S2f[0])[1] > 0.04
    # mixed batch picks the matching branch
    kk = torch.as_tensor(K)
    got = reward_t(kk, torch.as_tensor(S), torch.as_tensor(S2), torch.as_tensor(G))
    ref = [task.reward(task.TOKEN_KINDS[k], a, b, g) for k, a, b, g in zip(K, S, S2, G)]
    assert np.allclose(got.numpy(), ref, atol=1e-5)


def test_imagine_shapes_and_survival():
    wm, actor, _ = _nets()
    s, goal, ctx, kind = _batch()
    out = imagine(wm, actor, s, goal, ctx, kind, 5)
    assert out["s"].shape == (6, 6, SD) and out["r"].shape == out["d"].shape == (6, 5) and out["c"].shape == (6, 6)
    assert (out["c"][:, 0] == 1).all() and (out["c"][:, 1:] <= out["c"][:, :-1]).all()
    assert all(p.requires_grad for p in wm.parameters())


def test_policy_step_leaves_wm_and_critic():
    wm, actor, critic = _nets()
    s, goal, ctx, kind = _batch()
    before = [p.detach().clone() for m in (wm, critic) for p in m.parameters()]
    a0 = [p.detach().clone() for p in actor.parameters()]
    out = imagine(wm, actor, s, goal, ctx, kind, 4)
    policy_step(actor, torch.optim.Adam(actor.parameters(), 1e-2), policy_loss(out, critic, goal, ctx, kind, 0.99), 1.0, 0.1)
    assert all(torch.equal(b, p) for b, p in zip(before, [p for m in (wm, critic) for p in m.parameters()]))
    assert any(not torch.equal(b, p) for b, p in zip(a0, actor.parameters()))
    assert all(p.grad is None for p in wm.parameters())


def test_policy_grad_reaches_first_action():
    wm, actor, critic = _nets()
    s, goal, ctx, kind = _batch()
    out = imagine(wm, actor, s, goal, ctx, kind, 4)
    policy_loss(out, critic, goal, ctx, kind, 0.99).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.dec.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)


def test_gae_targets_detached_and_match_reference():
    _, _, critic = _nets()
    H, B = 3, 4
    s, goal, ctx, kind = _batch(B)
    kind = torch.where(kind == 1, 0, kind)  # ALIGN rewards are <= 0; this test feeds positive ones
    out = {"s": torch.randn(B, H + 1, SD, requires_grad=True), "r": torch.randn(B, H), "d": torch.rand(B, H) * 0.3}
    tg = gae_targets(out, critic, goal, ctx, kind, 0.99, 0.95)
    assert tg.grad_fn is None and tg.shape == (B, H)
    with torch.no_grad():
        V = np.stack([critic(out["s"][:, t], goal, ctx, kind).numpy() for t in range(H + 1)], 1)
    r, d, ref = out["r"].numpy(), out["d"].numpy(), np.zeros((B, H))
    R = V[:, H]
    for t in (2, 1, 0):
        R = r[:, t] + 0.99 * (1 - d[:, t]) * (0.05 * V[:, t + 1] + 0.95 * R)
        ref[:, t] = R
    assert np.allclose(tg.numpy(), ref, atol=1e-5)
    assert critic_loss(critic, out["s"][:, :H], goal, ctx, kind, tg).item() >= 0
