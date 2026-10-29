"""Imagined rollouts through the frozen world model and the actor-critic losses (paper Eq. 2, 3).

reward_t/success_t are torch twins of task.reward/task.success for mixed-kind batches (kind: 0 GRASP, 1 ALIGN, 2 INSERT).
Indexing: out["d"][:, t] and out["r"][:, t] belong to the transition s_t -> s_{t+1}; c_{t+1} = c_t (1 - d_t).
"""
import torch

from supermimic_yai.scripted import ALIGN_POS_TOL, ALIGN_ROT_TOL
from supermimic_yai.task import _EXTRA, FORCE_SCALE


def _slices(s):
    n = (s.shape[-1] - _EXTRA) // 2
    return s[..., 2 * n + 1:2 * n + 8], s[..., 2 * n + 8], s[..., 2 * n + 9:2 * n + 16], s[..., 2 * n + 16]


def _rot(q):
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def _align_err(s):
    part, _, tgt, _ = _slices(s)
    R = _rot(tgt[..., 3:]) @ _rot(part[..., 3:]).transpose(-1, -2)
    return (tgt[..., :2] - part[..., :2]).norm(dim=-1), torch.atan2(R[..., 1, 0], R[..., 0, 0]).abs()


def reward_t(kind, s, s_next, goal):
    part, flag, tgt, f = _slices(s_next)
    xy, yaw = _align_err(s_next)
    rem = lambda x: _slices(x)[0][..., 2] - _slices(x)[2][..., 2]  # noqa: E731
    mf = goal[:, 1] * FORCE_SCALE
    depth = torch.where(kind == 2, goal[:, 0], 1.0)  # safe denominator: zero goals would NaN the where-grads
    r = [-tgt[..., :3].norm(dim=-1) + flag, -(xy + 0.05 * yaw),
         (rem(s) - rem(s_next)) / depth - 0.1 * (f * mf - mf).clamp_min(0)]
    return torch.where(kind == 0, r[0], torch.where(kind == 1, r[1], r[2]))


def success_t(kind, s, goal):
    part, flag, tgt, _ = _slices(s)
    xy, yaw = _align_err(s)
    ok = [flag > 0.5, (flag > 0.5) & (xy <= ALIGN_POS_TOL) & (yaw <= ALIGN_ROT_TOL),
          (tgt[..., :3] - part[..., :3]).norm(dim=-1) <= goal[:, 2]]
    return torch.where(kind == 0, ok[0], torch.where(kind == 1, ok[1], ok[2]))


def imagine(wm, actor, s0, goal, ctx, kind, H):
    """-> {s (B,H+1,s_dim), r (B,H), d (B,H), c (B,H+1)}. WM params get no grad; states and actions do."""
    flags = [p.requires_grad for p in wm.parameters()]
    wm.requires_grad_(False)
    try:
        s, S, R, D = s0, [s0], [], []
        for _ in range(H):
            s2, d = wm.step(s, actor(s, goal, kind), ctx, kind)
            R.append(reward_t(kind, s, s2, goal)), D.append(d), S.append(s2)
            s = s2
    finally:
        for p, f in zip(wm.parameters(), flags):
            p.requires_grad_(f)
    d = torch.stack(D, 1)
    c = torch.cat([torch.ones_like(d[:, :1]), torch.cumprod(1 - d, 1)], 1)
    return {"s": torch.stack(S, 1), "r": torch.stack(R, 1), "d": d, "c": c}


def _values(out, critic, goal, ctx, kind):
    B, T, sd = out["s"].shape
    rep = lambda x: x[:, None].expand(B, T, *x.shape[1:]).reshape(B * T, *x.shape[1:])  # noqa: E731
    return critic(out["s"].reshape(B * T, sd), rep(goal), rep(ctx), rep(kind)).reshape(B, T)


def policy_loss(out, critic, goal, ctx, kind, gamma):
    """Eq. 2: -mean_b sum_{t<H} gamma^t c_t r_t + gamma^H c_H V(s_H). Critic params get no grad (frozen inside)."""
    H = out["r"].shape[1]
    flags = [p.requires_grad for p in critic.parameters()]
    critic.requires_grad_(False)
    try:
        V = _values(out, critic, goal, ctx, kind)
    finally:
        for p, f in zip(critic.parameters(), flags):
            p.requires_grad_(f)
    g = gamma ** torch.arange(H + 1, device=V.device, dtype=V.dtype)
    ret = (g[:H] * out["c"][:, :H] * out["r"]).sum(1) + g[H] * out["c"][:, H] * V[:, H]
    return -ret.mean()


@torch.no_grad()
def gae_targets(out, critic, goal, ctx, kind, gamma, lam):
    """R_t = r_t + gamma (1 - d_t)[(1 - lam) V_{t+1} + lam R_{t+1}], R_H = V_H -> (B,H)."""
    V, H = _values(out, critic, goal, ctx, kind), out["r"].shape[1]
    R, T = V[:, H], []
    for t in reversed(range(H)):
        R = out["r"][:, t] + gamma * (1 - out["d"][:, t]) * ((1 - lam) * V[:, t + 1] + lam * R)
        T.append(R)
    return torch.stack(T[::-1], 1).detach()


def critic_loss(critic, s, goal, ctx, kind, targets):
    """MSE of V on states (B,H,s_dim) against targets (B,H); states are detached."""
    out = {"s": s.detach()}
    return ((_values(out, critic, goal, ctx, kind) - targets) ** 2).mean()


def policy_step(actor, opt, loss, clip, clip_elem):
    """-> global grad norm before clipping."""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), clip)
    for p in actor.parameters():
        if p.grad is not None:
            p.grad.clamp_(-clip_elem, clip_elem)
    opt.step()
    return norm.item()
