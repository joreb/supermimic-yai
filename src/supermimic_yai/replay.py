"""FIFO ring of transitions that samples k-step windows.

add() takes an episode dict (see collect.py): s (T+1,s_dim), a (T,a_dim), r/done/unstable (T,), kind, goal, ctx.
Capacity counts transitions. A window is k consecutive transitions of one live episode; the episode's last
transition may be the window's last, but a window never runs past an episode end or past an unstable step.
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, s_dim, a_dim, g_dim, c_dim):
        z = lambda *sh, dt=np.float64: np.zeros((capacity, *sh), dt)  # noqa: E731
        self.cap, self.size, self.head, self.next_id = capacity, 0, 0, 0
        self.s, self.s2, self.a, self.r = z(s_dim), z(s_dim), z(a_dim), z(dt=np.float64)
        self.done, self.unstable = z(dt=bool), z(dt=bool)
        self.goal, self.ctx, self.kind, self.eid = z(g_dim), z(c_dim), z(dt=np.int64), np.full(capacity, -1, np.int64)
        self._valid = {}

    def add(self, ep):
        T = len(ep["a"])
        assert T <= self.cap, "episode longer than the buffer"
        rows = (self.head + np.arange(T)) % self.cap
        self.s[rows], self.s2[rows], self.a[rows], self.r[rows] = ep["s"][:-1], ep["s"][1:], ep["a"], ep["r"]
        self.done[rows], self.unstable[rows] = ep["done"], ep["unstable"]
        self.goal[rows], self.ctx[rows], self.kind[rows], self.eid[rows] = ep["goal"], ep["ctx"], ep["kind"], self.next_id
        self.next_id += 1
        self.head, self.size = (self.head + T) % self.cap, min(self.size + T, self.cap)
        self._valid = {}

    def _starts(self, k):
        if k not in self._valid:
            start = (self.head - self.size) % self.cap  # oldest row
            i = (start + np.arange(self.size)) % self.cap
            rows = (i[:, None] + np.arange(k)) % self.cap
            ok = np.arange(self.size) + k <= self.size
            same = (self.eid[rows] == self.eid[i][:, None]).all(1)
            self._valid[k] = i[ok & same & ~self.unstable[rows].any(1)]
        return self._valid[k]

    def stable_rows(self):
        """Row indices of stored transitions that are not unstable (their s_next is an auto-reset state)."""
        return np.flatnonzero(~self.unstable[:self.size])

    def sample_windows(self, k, n, rng) -> dict:
        starts = self._starts(k)
        if not len(starts):
            raise ValueError("no complete k-step window in the buffer")
        i = rng.choice(starts, n)
        rows = (i[:, None] + np.arange(k)) % self.cap
        last = rows[:, -1]
        return {"s": np.concatenate([self.s[rows], self.s2[last][:, None]], 1), "a": self.a[rows], "r": self.r[rows],
                "done": self.done[rows], "unstable": self.unstable[rows],
                "goal": self.goal[i], "ctx": self.ctx[i], "kind": self.kind[i]}
