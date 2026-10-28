"""World model: MLP predicting the normalized state residual and a termination logit; k-step autoregressive loss.

Reward and success are analytic (ruling P2), so there is no reward head. Normalisation stats are buffers (set_norm).
s layout (task.py): [qpos n, qvel n, gripper, held pose 7, held flag, target pose 7, force]; n = (s_dim - 17) / 2.
"""
import torch
from torch import nn

from supermimic_yai.task import TOKEN_KINDS

_EXTRA = 17


class WorldModel(nn.Module):
    def __init__(self, s_dim, a_dim, c_dim, n_kinds, width, depth):
        super().__init__()
        self.n_kinds = n_kinds
        layers, d = [], s_dim + a_dim + c_dim + n_kinds
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, s_dim + 1))
        for name, n in (("s", s_dim), ("c", c_dim), ("ds", s_dim)):
            self.register_buffer(f"{name}_mu", torch.zeros(n))
            self.register_buffer(f"{name}_sd", torch.ones(n))

    def set_norm(self, s, s_next, c):
        """Stats from numpy arrays (N, ·): states, their successors, contexts. Floor on std keeps constant dims finite."""
        t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
        s, ds, c = t(s), t(s_next) - t(s), t(c)
        for name, x in (("s", s), ("c", c), ("ds", ds)):
            getattr(self, f"{name}_mu").copy_(x.mean(0))
            getattr(self, f"{name}_sd").copy_(x.std(0).clamp_min(1e-3))

    def step(self, s, a, c, kind):
        """Raw s (B,s_dim), a, c, kind (B,) int -> (s_next raw, done_prob). Differentiable."""
        s2, logit = self._logit_step(s, a, c, kind)
        return s2, torch.sigmoid(logit)

    def _logit_step(self, s, a, c, kind):
        """-> (s_next, done logit); the logit keeps BCE stable."""
        x = torch.cat([(s - self.s_mu) / self.s_sd, a, (c - self.c_mu) / self.c_sd,
                       nn.functional.one_hot(kind, self.n_kinds).to(s.dtype)], -1)
        out = self.net(x)
        return s + out[:, :-1] * self.ds_sd + self.ds_mu, out[:, -1]


def rollout(model, batch, k):
    """Seed at logged s_0, roll on logged actions, feeding back predictions. -> s_hat (B,k+1,s_dim), logits (B,k)."""
    s, S, L = batch["s"][:, 0], [batch["s"][:, 0]], []
    for j in range(k):
        s, logit = model._logit_step(s, batch["a"][:, j], batch["ctx"], batch["kind"])
        S.append(s), L.append(logit)
    return torch.stack(S, 1), torch.stack(L, 1)


def gate_errors(s_hat, s, j, n):
    """Per-window unnormalised errors (m) at step j: held-part position rel EE, and part-to-target vector."""
    h, t = slice(2 * n + 1, 2 * n + 4), slice(2 * n + 9, 2 * n + 12)
    held = (s_hat[:, j, h] - s[:, j, h]).norm(dim=-1)
    pt = ((s_hat[:, j, h] - s_hat[:, j, t]) - (s[:, j, h] - s[:, j, t])).norm(dim=-1)
    return held, pt


def wm_loss(model, batch, k, metrics=False):
    """Paper Eq. 1 minus the reward term: sum_j |s_hat_j - s_j|^2 (normalised) + BCE(done). Batch values are tensors.
    -> (loss, {}) ; with metrics=True the dict has loss/state_loss/done_loss and per-kind errors (.item() syncs, eval only)."""
    s_hat, logits = rollout(model, batch, k)
    s = batch["s"]
    sq = (((s_hat[:, 1:] - s[:, 1:]) / model.s_sd) ** 2).sum(-1).sum(1).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, batch["done"].float(), reduction="none").sum(1).mean()
    loss = sq + bce
    if not metrics:
        return loss, {}
    n = (s.shape[-1] - _EXTRA) // 2
    out = {"loss": loss.item(), "state_loss": sq.item(), "done_loss": bce.item()}
    with torch.no_grad():
        for name, j in (("kstep_err", k), ("step1_err", 1)):
            held, pt = gate_errors(s_hat, s, j, n)
            for ki, kn in enumerate(TOKEN_KINDS):
                m = batch["kind"] == ki
                if m.any():
                    out[f"{name}/{kn}"] = held[m].mean().item()
                    out[f"{name.replace('err', 'err_pt')}/{kn}"] = pt[m].mean().item()
    return loss, out
