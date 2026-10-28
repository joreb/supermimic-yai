import numpy as np

from supermimic_yai.config import load_config
from supermimic_yai.npmlp import mlp_forward
from supermimic_yai.replay import ReplayBuffer


def ep(T, tag, unstable_at=None):
    s = np.full((T + 1, 2), float(tag)) + np.arange(T + 1)[:, None] * 0.01
    u = np.zeros(T, bool)
    d = np.zeros(T, bool)
    d[-1] = True
    if unstable_at is not None:
        u[unstable_at] = True
    return {"s": s, "a": np.full((T, 1), float(tag)), "r": np.zeros(T), "done": d, "unstable": u,
            "kind": 1, "goal": np.zeros(3), "ctx": np.zeros(4)}


def buf(cap):
    return ReplayBuffer(cap, 2, 1, 3, 4)


def test_windows_respect_boundaries():
    b, rng = buf(100), np.random.default_rng(0)
    for tag in (1, 2, 3):
        b.add(ep(6, tag))
    w = b.sample_windows(4, 200, rng)
    assert w["s"].shape == (200, 5, 2) and w["a"].shape == (200, 4, 1) and w["done"].shape == (200, 4)
    assert (np.round(w["s"][:, :, 0]) == w["a"][:, :1, 0]).all()  # one episode per window
    assert (np.diff(w["s"][:, :, 0], axis=1) > 0).all()  # consecutive states


def test_fifo_overwrites_oldest():
    b, rng = buf(10), np.random.default_rng(0)
    for tag in (1, 2, 3):
        b.add(ep(4, tag))  # 12 transitions into 10 rows: episode 1 is partly gone
    assert b.size == 10
    w = b.sample_windows(3, 100, rng)
    assert set(np.round(w["s"][:, 0, 0])) == {2, 3}  # the damaged oldest episode yields no window
    assert (np.diff(w["s"][:, :, 0], axis=1) > 0).all()


def test_unstable_masked():
    b = buf(50)
    b.add(ep(8, 1, unstable_at=3))
    w = b.sample_windows(2, 200, np.random.default_rng(0))
    assert not w["unstable"][:, :-1].any()  # nothing follows an unstable step
    starts = np.round((w["s"][:, 0, 0] - 1) * 100)
    assert set(starts) == {0, 1, 2, 4, 5, 6}  # the window that would start at the unstable step is excluded


def test_mlp_and_config():
    w = [(np.eye(2), np.zeros(2)), (np.eye(2), np.zeros(2))]
    assert np.allclose(mlp_forward(w, np.array([1.0, -50.0])), np.tanh([1.0, -1.0]))
    c = load_config("laptop")
    assert c["wm"]["width"] == 512 and c["wm"]["k"] == 8 and c["collect"]["workers"] >= 1
    assert load_config("big-gpu")["wm"]["batch"] == 2048 * 10
