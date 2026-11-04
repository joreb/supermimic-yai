"""Learner: actor and critic optimised through imagined rollouts in a frozen world model (paper Eq. 2, 3)."""
import math

import numpy as np
import torch

from supermimic_yai.imagine import (R_MAX, _values, critic_loss, fit_grip_box, gae_targets, held_index, imagine, policy_loss,
                                    policy_step, value_range)
from supermimic_yai.policy import Actor, Critic
from supermimic_yai.task import TOKEN_KINDS


class Learner:
    def __init__(self, cfg, s_dim, a_dim, g_dim, c_dim, wm, device):
        p, c, n = cfg["policy"], cfg["critic"], len(TOKEN_KINDS)
        self.p, self.c, self.dev, self.wm = p, c, device, wm
        self.actor = Actor(s_dim, g_dim, a_dim, n, p["z_dim"], p["width"], p["depth"]).to(device)
        self.critic = Critic(s_dim, g_dim, c_dim, n, c["width"], c["depth"]).to(device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), p["lr"], betas=tuple(p["betas"]))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), p["lr"], betas=tuple(p["betas"]))
        N = cfg["train"]["lr_decay_iters"]
        cos = lambda i: 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(i / N, 1)))  # noqa: E731
        self.sched = [torch.optim.lr_scheduler.LambdaLR(o, cos) for o in (self.opt_a, self.opt_c)]

    def set_norm(self, buf):
        ok = buf.stable_rows()
        self.actor.set_norm(buf.s[ok], self.p["std_floor"])
        self.critic.set_norm(buf.s[ok], buf.ctx[ok])
        x = np.concatenate([buf.s[ok], buf.s2[ok]])  # imagined states stay within the data range + 5% of it
        lo, hi = x.min(0), x.max(0)
        m = 0.05 * (hi - lo)
        self.grip = fit_grip_box(buf)
        self.lo, self.hi = (torch.as_tensor(v, dtype=torch.float32, device=self.dev) for v in (lo - m, hi + m))

    def _sample(self, buf, rng, kinds):
        rows = np.flatnonzero(~buf.unstable[:buf.size] & np.isin(buf.kind[:buf.size], kinds))
        i = rng.choice(rows, self.p["batch"])
        t = lambda x, dt=torch.float32: torch.as_tensor(x[i], dtype=dt, device=self.dev)  # noqa: E731
        return i, t(buf.s), t(buf.goal), t(buf.ctx), t(buf.kind, torch.int64)

    def update(self, buf, rng, kinds) -> dict:
        p, c = self.p, self.c
        _, s0, goal, ctx, kind = self._sample(buf, rng, kinds)
        out = imagine(self.wm, self.actor, s0, goal, ctx, kind, p["horizon"], self.lo, self.hi, self.grip)
        loss = policy_loss(out, self.critic, goal, ctx, kind, p["gamma"])
        gn = policy_step(self.actor, self.opt_a, loss, p["grad_clip"], p["grad_clip_elem"])
        B, H = out["r"].shape
        tg = gae_targets(out, self.critic, goal, ctx, kind, p["gamma"], c["lam"]).reshape(-1, 1)
        S = out["s"][:, :H].detach().reshape(B * H, 1, -1)
        rep = lambda x: x[:, None].expand(B, H, *x.shape[1:]).reshape(B * H, *x.shape[1:])  # noqa: E731
        G, C, K = rep(goal), rep(ctx), rep(kind)
        for _ in range(c["epochs"]):
            for mb in torch.randperm(B * H, device=self.dev).chunk(c["minibatches"]):
                cl = critic_loss(self.critic, S[mb], G[mb], C[mb], K[mb], tg[mb])
                self.opt_c.zero_grad(set_to_none=True)
                cl.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), p["grad_clip"])
                self.opt_c.step()
        for s in self.sched:
            s.step()
        with torch.no_grad():  # raw critic output vs the physical value range, for the abort check
            v, (vlo, vhi) = _values(out, self.critic, goal, ctx, kind).mean().item(), value_range(kind, goal, p["gamma"])
        held = (out["s"][:, 1:, held_index(out["s"].shape[-1])] > 0.5).float().mean().item()
        rs = (out["c"][:, :H] * out["r"]).sum(1).mean().item()  # bounded by H * r_max, unlike the bootstrapped return
        return {"clamp_frac": out["sat"], "held_frac": held, "v_mean": v, "v_lo": vlo.mean().item(), "v_hi": vhi.mean().item(), "imagined_reward_sum": rs, "imagined_return": -loss.item(), "policy_loss": loss.item(), "critic_loss": cl.item(), "grad_norm": gn}

    def bc_update(self, buf, rng, kinds) -> dict:
        i, s, goal, _, kind = self._sample(buf, rng, kinds)
        a = torch.as_tensor(buf.a[i], dtype=torch.float32, device=self.dev)
        loss = ((self.actor(s, goal, kind) - a) ** 2).mean()
        gn = policy_step(self.actor, self.opt_a, loss, self.p["grad_clip"], self.p["grad_clip_elem"])
        return {"bc_loss": loss.item(), "grad_norm": gn}


def out_of_bounds(m, horizon) -> bool:
    """Abort criterion on a Learner.update result: imagined reward sum above horizon * r_max, or mean V outside its physical range."""
    return m["imagined_reward_sum"] > horizon * R_MAX + 1e-3 or not m["v_lo"] <= m["v_mean"] <= m["v_hi"]
