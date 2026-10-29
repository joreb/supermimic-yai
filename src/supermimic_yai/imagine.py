"""Imagined rollouts through the frozen world model and the actor-critic losses (paper Eq. 2, 3).

reward_t/success_t are torch twins of task.reward/task.success for mixed-kind batches (kind: 0 GRASP, 1 ALIGN, 2 INSERT).
Indexing: out["d"][:, t] and out["r"][:, t] belong to the transition s_t -> s_{t+1}; c_{t+1} = c_t (1 - d_t).
"""
import torch

from supermimic_yai.scripted import ALIGN_POS_TOL, ALIGN_ROT_TOL
from supermimic_yai.task import _EXTRA, FORCE_SCALE


# Physical bounds of the env reward (task.reward): flag in {0,1}; relative distances < D_MAX [m] (arm reach);
# INSERT per-step progress is a fraction of the depth, |.| <= 1; force ratio f < 2 (the env terminates above 2).
D_MAX, F_MAX, R_MAX = 2.0, 2.0, 1.0


def reward_range(kind, goal):
    """(lo, hi) per row that reward_t can produce: GRASP [-D_MAX, 1], ALIGN [-(D_MAX + 0.05 pi), 0], INSERT [-1 - 0.1 mf (F_MAX - 1), 1]."""
    lo = torch.stack([torch.full_like(goal[:, 0], -D_MAX), torch.full_like(goal[:, 0], -D_MAX - 0.05 * torch.pi),
                      -1 - 0.1 * goal[:, 1] * FORCE_SCALE * (F_MAX - 1)], 1).gather(1, kind[:, None])[:, 0]
    return lo, torch.where(kind == 1, torch.zeros_like(lo), torch.full_like(lo, R_MAX))


# Soft grip box on (gripper width, held-part position rel. EE). The env's held flag needs finger contacts (not in the
# token state); the WM's flag head is unconstrained off-distribution and was exploited (closing ~2 cm away from the part).
# Edges = 0.1/99.9 percentiles over all held=1 rows (s and s_next, GRASP+ALIGN+INSERT, Plan 2a seed episodes) widened
# by a margin; each side is a sigmoid of width margin/3, so the box is 0.95 at a fitted edge and ~0 a few cm outside.
GRIP_LO, GRIP_HI, GRIP_M = 0.0075, 0.0250, 0.0015  # gripper width [m]
POS_LO, POS_HI, POS_M = (-0.0038, -0.0004, -0.0119), (0.0053, 0.0037, 0.0106), 0.003  # part pos rel. EE [m] (x, y, z)


def grip_box_t(s):
    n = (s.shape[-1] - _EXTRA) // 2
    sg = torch.sigmoid
    lo = torch.tensor((GRIP_LO, *POS_LO), dtype=s.dtype, device=s.device)
    hi = torch.tensor((GRIP_HI, *POS_HI), dtype=s.dtype, device=s.device)
    m = torch.tensor((GRIP_M, *(POS_M,) * 3), dtype=s.dtype, device=s.device)
    x = torch.cat([s[..., 2 * n:2 * n + 1], s[..., 2 * n + 1:2 * n + 4]], -1)
    return (sg(3 * (x - lo + m) / m) * sg(3 * (hi + m - x) / m)).prod(-1)


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
    flag, f = flag.clamp(0, 1), f.clamp(0, F_MAX)  # WM outputs can leave the physical range; the env's cannot
    xy, yaw = _align_err(s_next)
    xy = xy.clamp_max(D_MAX)
    rem = lambda x: _slices(x)[0][..., 2] - _slices(x)[2][..., 2]  # noqa: E731
    mf = goal[:, 1] * FORCE_SCALE
    depth = torch.where(kind == 2, goal[:, 0], 1.0)  # safe denominator: zero goals would NaN the where-grads
    r = [-tgt[..., :3].norm(dim=-1).clamp_max(D_MAX) + flag, -(xy + 0.05 * yaw),
         ((rem(s) - rem(s_next)) / depth).clamp(-1, 1) - 0.1 * (f * mf - mf).clamp_min(0)]
    return torch.where(kind == 0, r[0], torch.where(kind == 1, r[1], r[2]))


def success_t(kind, s, goal):
    part, flag, tgt, _ = _slices(s)
    xy, yaw = _align_err(s)
    ok = [flag > 0.5, (flag > 0.5) & (xy <= ALIGN_POS_TOL) & (yaw <= ALIGN_ROT_TOL),
          (tgt[..., :3] - part[..., :3]).norm(dim=-1) <= goal[:, 2]]
    return torch.where(kind == 0, ok[0], torch.where(kind == 1, ok[1], ok[2]))


def _set_held(s):
    """held := clamp(WM held, 0, 1) * grip_box(s): the WM judges lift/contact, the box blocks implausible grips."""
    i = (s.shape[-1] - _EXTRA) // 2 * 2 + 8
    new = s[..., i].clamp(0, 1) * grip_box_t(s)
    return torch.cat([s[..., :i], new[..., None], s[..., i + 1:]], -1)


def imagine(wm, actor, s0, goal, ctx, kind, H, lo=None, hi=None):
    """-> {s (B,H+1,s_dim), r (B,H), d (B,H), c (B,H+1)}. WM params get no grad; states and actions do.
    lo/hi (s_dim,): per-dimension clamp on every predicted state, keeping the rollout inside the data range."""
    flags = [p.requires_grad for p in wm.parameters()]
    wm.requires_grad_(False)
    try:
        s, S, R, D = s0, [s0], [], []
        for _ in range(H):
            s2, d = wm.step(s, actor(s, goal, kind), ctx, kind)
            if lo is not None:
                s2 = torch.maximum(torch.minimum(s2, hi), lo)
            s2 = _set_held(s2)
            R.append(reward_t(kind, s, s2, goal)), D.append(d), S.append(s2)
            s = s2
    finally:
        for p, f in zip(wm.parameters(), flags):
            p.requires_grad_(f)
    d = torch.stack(D, 1)
    c = torch.cat([torch.ones_like(d[:, :1]), torch.cumprod(1 - d.clamp(0, 1), 1)], 1)
    return {"s": torch.stack(S, 1), "r": torch.stack(R, 1), "d": d, "c": c}


def _values(out, critic, goal, ctx, kind):
    B, T, sd = out["s"].shape
    rep = lambda x: x[:, None].expand(B, T, *x.shape[1:]).reshape(B * T, *x.shape[1:])  # noqa: E731
    return critic(out["s"].reshape(B * T, sd), rep(goal), rep(ctx), rep(kind)).reshape(B, T)


def value_range(kind, goal, gamma):
    """(lo, hi) per row of any discounted return: r_range / (1 - gamma). Critic output and targets are clamped to it."""
    lo, hi = reward_range(kind, goal)
    return lo / (1 - gamma), hi / (1 - gamma)


def _clamped_values(out, critic, goal, ctx, kind, gamma):
    lo, hi = value_range(kind, goal, gamma)
    return torch.maximum(torch.minimum(_values(out, critic, goal, ctx, kind), hi[:, None]), lo[:, None])


def policy_loss(out, critic, goal, ctx, kind, gamma):
    """Eq. 2: -mean_b sum_{t<H} gamma^t c_t r_t + gamma^H c_H V(s_H). Critic params get no grad (frozen inside)."""
    H = out["r"].shape[1]
    flags = [p.requires_grad for p in critic.parameters()]
    critic.requires_grad_(False)
    try:
        V = _clamped_values(out, critic, goal, ctx, kind, gamma)
    finally:
        for p, f in zip(critic.parameters(), flags):
            p.requires_grad_(f)
    g = gamma ** torch.arange(H + 1, device=V.device, dtype=V.dtype)
    ret = (g[:H] * out["c"][:, :H] * out["r"]).sum(1) + g[H] * out["c"][:, H] * V[:, H]
    return -ret.mean()


@torch.no_grad()
def gae_targets(out, critic, goal, ctx, kind, gamma, lam):
    """R_t = r_t + gamma (1 - d_t)[(1 - lam) V_{t+1} + lam R_{t+1}], R_H = V_H -> (B,H)."""
    V, H = _clamped_values(out, critic, goal, ctx, kind, gamma), out["r"].shape[1]
    lo, hi = value_range(kind, goal, gamma)
    R, T = V[:, H], []
    for t in reversed(range(H)):
        R = out["r"][:, t] + gamma * (1 - out["d"][:, t]) * ((1 - lam) * V[:, t + 1] + lam * R)
        R = torch.maximum(torch.minimum(R, hi), lo)
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
