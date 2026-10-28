import numpy as np
import torch

from supermimic_yai.wm import WorldModel, gate_errors, rollout, wm_loss

N = 7
SD, AD, CD, K = 2 * N + 17, N + 1, 5, 4


def _batch(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *sh: torch.randn(*sh, generator=g)  # noqa: E731
    s = r(n, K + 1, SD)
    return {"s": s, "a": r(n, K, AD), "ctx": r(n, CD), "kind": torch.randint(0, 3, (n,), generator=g),
            "done": (torch.arange(K) == K - 1).float().expand(n, K).clone()}


def _model():
    torch.manual_seed(0)
    return WorldModel(SD, AD, CD, 3, 64, 2)


def test_overfits_tiny_batch():
    m, b = _model(), _batch()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    l0 = wm_loss(m, b, K)[0].item()
    for _ in range(300):
        loss, _ = wm_loss(m, b, K)
        opt.zero_grad(), loss.backward(), opt.step()
    assert loss.item() < l0 / 10


def test_autoregressive_rollout_uses_predictions():
    m, b, seen, outs = _model(), _batch(), [], []
    orig = m._logit_step

    def spy(s, *a):
        seen.append(s)
        out = orig(s, *a)
        outs.append(out[0])
        return out
    m._logit_step = spy
    rollout(m, b, K)
    assert torch.equal(seen[0], b["s"][:, 0])
    for j in range(1, K):
        assert torch.equal(seen[j], outs[j - 1]) and not torch.allclose(seen[j], b["s"][:, j])


def test_gradients_flow_through_rollout():
    m, b = _model(), _batch()
    b["a"].requires_grad_(True)
    rollout(m, b, K)[0][:, K].sum().backward()
    assert b["a"].grad[:, 0].abs().sum() > 0


def test_runs_without_cuda():
    m, b = _model(), _batch()
    s2, p = m.step(b["s"][:, 0], b["a"][:, 0], b["ctx"], b["kind"])
    assert s2.device.type == "cpu" and s2.shape == (16, SD) and ((p >= 0) & (p <= 1)).all()
    m.set_norm(np.random.randn(50, SD), np.random.randn(50, SD), np.random.randn(50, CD))
    _, met = wm_loss(m, b, K, metrics=True)
    assert "kstep_err/GRASP" in met or "kstep_err/INSERT" in met


def test_gate_errors_known_values():
    s = torch.zeros(1, 2, SD)
    s_hat = s.clone()
    h, t = slice(2 * N + 1, 2 * N + 4), slice(2 * N + 9, 2 * N + 12)
    s_hat[0, 1, h] = torch.tensor([0.003, 0.004, 0.0])  # held part off by 5 mm
    held, pt = gate_errors(s_hat, s, 1, N)
    assert torch.allclose(held, torch.tensor([0.005])) and torch.allclose(pt, torch.tensor([0.005]))
    s_hat[0, 1, t] = torch.tensor([0.003, 0.004, 0.0])  # target moves with it: part-to-target error vanishes
    held, pt = gate_errors(s_hat, s, 1, N)
    assert torch.allclose(held, torch.tensor([0.005])) and pt.item() < 1e-7
