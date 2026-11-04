"""Return-based reset sampling over recipe-step bins (ruling Q8). numpy only: workers may call it."""
import numpy as np


class ResetSampler:
    def __init__(self, bank, gamma, ema=0.1, temp=25.0, mix=(0.5, 0.4, 0.1)):
        self.gamma, self.ema, self.temp, self.mix = gamma, ema, temp, mix
        self.n_bank = len(bank)
        self.kind = {}
        self.members = {}
        for i, s in enumerate(bank):
            self.kind[s.step] = s.token
            self.members.setdefault(s.step, []).append(i)
        self.bins = sorted(self.members)
        self.G, self.wsum, self.n = {}, {b: 0.0 for b in self.bins}, 0

    def _soft(self):
        """Per-kind softmax(-temp * G_hat), each kind block scaled by its share of bins."""
        w = {}
        for k in set(self.kind.values()):
            bs = [b for b in self.bins if self.kind[b] == k]
            g = np.array([self.G.get(b, np.nan) for b in bs])
            seen = ~np.isnan(g)
            gh = np.zeros(len(bs))
            if seen.any():
                lo, hi = g[seen].min(), g[seen].max()
                gh[seen] = (g[seen] - lo) / (hi - lo + 1e-8)
            e = np.exp(-self.temp * gh)
            for b, x in zip(bs, e / e.sum() * len(bs) / len(self.bins)):
                w[b] = float(x)
        return w

    def weights(self):
        u, a, c = self.mix
        soft = self._soft()
        tot = sum(self.wsum.values())
        return {b: u / len(self.bins) + a * soft[b] + c * (self.wsum[b] / tot if tot else soft[b])
                for b in self.bins}

    def update(self, episodes):
        for e in episodes:
            r = np.asarray(e["r"], float)
            g = float(r @ self.gamma ** np.arange(len(r)))
            b = e["step"]
            if b not in self.members:
                raise ValueError(f"episode step {b} has no bin in the bank")
            self.G[b] = g if b not in self.G else (1 - self.ema) * self.G[b] + self.ema * g
        for b, x in self._soft().items():
            self.wsum[b] += x
        self.n += 1

    def __call__(self, rng, bank):
        assert len(bank) == self.n_bank, "sampler was built for a different bank"  # indices come from construction
        w = self.weights()
        b = self.bins[rng.choice(len(self.bins), p=[w[x] for x in self.bins])]
        return self.members[b][int(rng.integers(len(self.members[b])))]

    def state_dict(self):
        return {"G": dict(self.G), "wsum": dict(self.wsum), "n": self.n}

    def load_state_dict(self, d):
        self.G, self.wsum, self.n = dict(d["G"]), dict(d["wsum"]), d["n"]
