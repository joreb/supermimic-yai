import numpy as np
import torch

from supermimic_yai.npmlp import actor_forward, load_actor_npz, save_actor_npz
from supermimic_yai.policy import Actor, Critic

S, G, A, K = 31, 3, 8, 3


def _actor(seed=0):
    torch.manual_seed(seed)
    a = Actor(S, G, A, K, 32, 64, 3)
    a.set_norm(np.random.default_rng(seed).normal(2, 3, (100, S)))
    return a


def _inputs(n, rng, scale=1.0):
    return rng.normal(size=(n, S)) * scale, rng.normal(size=(n, G))


def test_actor_numpy_parity():
    a, rng = _actor(), np.random.default_rng(1)
    w = a.export()
    for k in range(K):
        s, g = _inputs(32, rng)
        want = a(torch.tensor(s, dtype=torch.float32), torch.tensor(g, dtype=torch.float32), torch.full((32,), k)).detach().numpy()
        got = np.stack([actor_forward(w, si, gi, k) for si, gi in zip(s, g)])
        assert np.abs(got - want).max() < 1e-6


def test_actor_output_ranges():
    a, rng = _actor(), np.random.default_rng(2)
    s, g = _inputs(64, rng, 100.0)
    out = a(torch.tensor(s, dtype=torch.float32), torch.tensor(g * 100, dtype=torch.float32), torch.randint(0, K, (64,))).detach()
    assert (out[:, :-1].abs() < 1).all() and (out[:, -1] > 0).all() and (out[:, -1] < 1).all()


def test_actor_kind_matters():
    a, s = _actor(), torch.randn(1, S)
    z = torch.zeros(1, G)
    assert (a(s, z, torch.tensor([0])) - a(s, z, torch.tensor([1]))).abs().max() > 1e-4


def test_critic_shape():
    c = Critic(S, G, 5, K, 16, 2)
    c.set_norm(np.random.randn(20, S), np.random.randn(20, 5))
    assert c(torch.randn(7, S), torch.randn(7, G), torch.randn(7, 5), torch.randint(0, K, (7,))).shape == (7,)


def test_npz_roundtrip(tmp_path):
    w = _actor().export()
    save_actor_npz(tmp_path / "a.npz", w)
    w2 = load_actor_npz(tmp_path / "a.npz")
    s, g = np.random.randn(S), np.random.randn(G)
    assert all(np.array_equal(actor_forward(w, s, g, k), actor_forward(w2, s, g, k)) for k in range(K))
