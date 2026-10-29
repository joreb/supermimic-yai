"""Per-kind actor and shared-trunk critic (rulings Q1, Q2). Actor input is raw s; stats are buffers (set_norm)."""
import numpy as np
import torch
from torch import nn


def _mlp(d_in, d_out, width, depth):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ELU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def _stats(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return x.mean(0), x.std(0).clamp_min(1e-3)


class Actor(nn.Module):
    def __init__(self, s_dim, g_dim, a_dim, n_kinds, z_dim, width, depth):
        super().__init__()
        self.enc = nn.ModuleList(_mlp(g_dim, z_dim, width, depth) for _ in range(n_kinds))
        self.dec = _mlp(s_dim + z_dim, a_dim, width, depth)
        self.register_buffer("s_mu", torch.zeros(s_dim))
        self.register_buffer("s_sd", torch.ones(s_dim))

    def set_norm(self, s, floor=1e-2):
        """floor: raw-unit std floor (1 cm / 0.01 rad: finer than any scale the policy can act on). Without it, dims
        with ~1 mm data spread blow up to 100s of sigma once the world model drifts, and the actor exploits that."""
        mu, sd = _stats(s)
        self.s_mu.copy_(mu), self.s_sd.copy_(sd.clamp_min(floor))

    def forward(self, s, goal, kind):
        """s (B,s_dim) raw, goal (B,g_dim), kind (B,) int -> a (B,a_dim): arm in (-1,1), gripper in (0,1)."""
        z = torch.zeros(s.shape[0], self.enc[0][-1].out_features, dtype=s.dtype, device=s.device)
        for k, enc in enumerate(self.enc):
            m = kind == k
            if m.any():
                z = z.masked_scatter(m[:, None], torch.tanh(enc(goal[m])))
        a = torch.tanh(self.dec(torch.cat([(s - self.s_mu) / self.s_sd, z], -1)))
        return torch.cat([a[:, :-1], (a[:, -1:] + 1) / 2], -1)

    def export(self) -> dict:
        f = lambda m: [(l.weight.detach().cpu().double().numpy().T.copy(), l.bias.detach().cpu().double().numpy().copy())  # noqa: E731
                       for l in m if isinstance(l, nn.Linear)]
        return {"s_mu": self.s_mu.double().cpu().numpy(), "s_sd": self.s_sd.double().cpu().numpy(),
                "enc": [f(e) for e in self.enc], "dec": f(self.dec)}


class Critic(nn.Module):
    def __init__(self, s_dim, g_dim, c_dim, n_kinds, width, depth):
        super().__init__()
        self.net = _mlp(s_dim + g_dim + c_dim, n_kinds, width, depth)
        for n, d in (("s", s_dim), ("c", c_dim)):
            self.register_buffer(f"{n}_mu", torch.zeros(d))
            self.register_buffer(f"{n}_sd", torch.ones(d))

    def set_norm(self, s, ctx):
        for n, x in (("s", s), ("c", ctx)):
            mu, sd = _stats(x)
            getattr(self, f"{n}_mu").copy_(mu), getattr(self, f"{n}_sd").copy_(sd)

    def forward(self, s, goal, ctx, kind):
        x = torch.cat([(s - self.s_mu) / self.s_sd, goal, (ctx - self.c_mu) / self.c_sd], -1)
        return self.net(x).gather(1, kind[:, None])[:, 0]
