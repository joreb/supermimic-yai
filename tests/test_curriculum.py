import numpy as np

from supermimic_yai.curriculum import ResetSampler


class S:
    def __init__(self, step, token):
        self.step, self.token = step, token


def mk(spec, per=3):
    """spec: {step: kind}"""
    return [S(st, k) for st, k in spec.items() for _ in range(per)]


def ep(step, ret, kind="INSERT"):
    return {"step": step, "kind": kind, "r": np.array([ret])}


def shares(sm, bank, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    c = np.bincount([bank[sm(rng, bank)].step for _ in range(n)])
    return c / n


def test_uniform_before_updates():
    bank = mk({0: "INSERT", 1: "INSERT", 2: "INSERT", 3: "INSERT"})
    p = shares(ResetSampler(bank, 0.99), bank)
    assert np.allclose(p, 0.25, atol=0.02)


def test_low_return_bin_sampled_more():
    bank = mk({0: "INSERT", 1: "INSERT"})
    sm = ResetSampler(bank, 0.99)
    for _ in range(30):
        sm.update([ep(0, -5.0), ep(1, 0.0)])
    assert shares(sm, bank)[0] > 0.5
    assert abs(sum(sm.weights().values()) - 1) < 1e-9


def test_learned_bin_keeps_practice():
    bank = mk({0: "INSERT", 1: "INSERT", 2: "INSERT"})
    sm = ResetSampler(bank, 0.99)
    off = ResetSampler(bank, 0.99, mix=(0.5, 0.5, 0.0))
    for _ in range(20):
        for x in (sm, off):
            x.update([ep(0, -5.0), ep(1, 0.0), ep(2, -1.0)])
    for _ in range(12):
        for x in (sm, off):
            x.update([ep(0, 3.0), ep(1, 0.0), ep(2, -1.0)])
    G = sm.state_dict()["G"]
    assert G[0] > G[1] > G[2]  # bin 0 is now the highest-return bin
    assert sm.weights()[0] > off.weights()[0] + 0.005  # only the 10 % term keeps practice on it
    p = shares(sm, bank)
    assert p[0] > shares(off, bank)[0]


def test_unvisited_bin_counts_lowest():
    bank = mk({0: "INSERT", 1: "INSERT", 2: "INSERT"})
    sm = ResetSampler(bank, 0.99)
    sm.update([ep(0, -5.0), ep(1, 0.0)])
    w = sm.weights()
    assert w[2] == w[0] > w[1]  # unvisited ties the lowest-return bin (G_hat = 0)


def test_kind_without_visits_is_uniform():
    bank = mk({0: "GRASP", 1: "GRASP", 2: "INSERT", 3: "INSERT"})
    sm = ResetSampler(bank, 0.99)
    sm.update([ep(2, -5.0), ep(3, 0.0)])
    w = sm.weights()
    assert abs(w[0] - w[1]) < 1e-12 and w[2] > w[3]


def test_unknown_step_raises():
    import pytest
    sm = ResetSampler(mk({0: "INSERT"}), 0.99)
    with pytest.raises(ValueError):
        sm.update([ep(7, 0.0)])


def test_per_kind_normalisation():
    bank = mk({0: "GRASP", 1: "GRASP", 2: "INSERT", 3: "INSERT"})
    a, b = ResetSampler(bank, 0.99), ResetSampler(bank, 0.99)
    for _ in range(10):
        a.update([ep(0, -1, "GRASP"), ep(1, 0, "GRASP"), ep(2, -5), ep(3, 0)])
        b.update([ep(0, -100, "GRASP"), ep(1, 0, "GRASP"), ep(2, -5), ep(3, 0)])
    assert abs(a.weights()[2] - b.weights()[2]) < 1e-9
    assert abs(a.weights()[0] + a.weights()[1] - 0.5) < 1e-9  # kinds keep their bin share


def test_discounted_return():
    bank = mk({0: "INSERT", 1: "INSERT"})
    sm = ResetSampler(bank, 0.5)
    sm.update([{"step": 0, "kind": "INSERT", "r": np.array([1.0, 1.0, 1.0])}])
    assert abs(sm.state_dict()["G"][0] - 1.75) < 1e-9  # first visit initialises the EMA


def test_state_roundtrip():
    import pickle
    bank = mk({0: "INSERT", 1: "INSERT"})
    a = ResetSampler(bank, 0.99)
    a.update([ep(0, -5.0), ep(1, 0.0)])
    b = ResetSampler(bank, 0.99)
    b.load_state_dict(pickle.loads(pickle.dumps(a.state_dict())))
    assert a.weights() == b.weights()


def test_state_dict_has_no_dead_counter_and_loads_old():
    sm = ResetSampler(mk({0: "INSERT"}), 0.99)
    assert set(sm.state_dict()) == {"G", "wsum"}
    sm.load_state_dict({"G": {}, "wsum": {0: 0.0}, "n": 7})
