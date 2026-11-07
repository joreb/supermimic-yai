import numpy as np
import pytest

from supermimic_yai.config import load_config
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
    assert set(np.round(w["s"][:, 0, 0])) == {2, 3}  # the 2 surviving rows of overwritten episode 1 are fewer than k, so no window starts there
    assert (np.diff(w["s"][:, :, 0], axis=1) > 0).all()


def test_unstable_masked():
    b = buf(50)
    b.add(ep(8, 1, unstable_at=3))
    w = b.sample_windows(2, 200, np.random.default_rng(0))
    assert not w["unstable"].any()  # no window contains an unstable step (its s_next is an auto-reset state)
    starts = np.round((w["s"][:, 0, 0] - 1) * 100)
    assert set(starts) == {0, 1, 4, 5, 6}


def test_norm_stats_skip_unstable():
    b = buf(50)
    b.add(ep(8, 1, unstable_at=3))
    assert len(b.stable_rows()) == 7 and not b.unstable[b.stable_rows()].any()


def test_config():
    c = load_config("laptop")
    assert c["wm"]["width"] == 512 and c["wm"]["k"] == 8 and c["collect"]["workers"] >= 1
    assert load_config("big-gpu")["wm"]["batch"] == 2048 * 10
    assert c["replay"]["capacity"] == 400000 and load_config("big-gpu")["policy"]["batch"] == 4096


def test_product_roundtrips_and_legacy_loads(tmp_path):
    from supermimic_yai.replay import load_episodes, save_episodes
    e = {"s": np.zeros((3, 2)), "a": np.zeros((2, 1)), "r": np.zeros(2), "done": np.zeros(2, bool), "unstable": np.zeros(2, bool),
         "goal": np.zeros(3), "ctx": np.zeros(4), "kind": 0, "terminal_kind": "timeout", "product": "p"}
    save_episodes(tmp_path / "e.npz", [e, e], 10)
    assert [x["product"] for x in load_episodes(tmp_path / "e.npz", 10)] == ["p", "p"]
    z = {k: v for k, v in np.load(tmp_path / "e.npz").items() if k != "product"}
    np.savez(tmp_path / "old.npz", **z)
    assert "product" not in load_episodes(tmp_path / "old.npz", 10)[0]


def test_rate_mismatch_refused(tmp_path):
    import shutil

    import torch
    from supermimic_yai.context import SimContext
    from supermimic_yai.npmlp import load_actor_npz, save_actor_npz
    from supermimic_yai.policy import Actor
    from supermimic_yai.replay import load_episodes, save_episodes
    from supermimic_yai.startbank import StartState, load_bank, save_bank
    from supermimic_yai.wm import WorldModel, load_wm, save_wm
    e = {"s": np.zeros((3, 2)), "a": np.zeros((2, 1)), "r": np.zeros(2), "done": np.zeros(2, bool), "unstable": np.zeros(2, bool),
         "goal": np.zeros(3), "ctx": np.zeros(4), "kind": 0, "terminal_kind": "timeout", "product": "p"}
    st = StartState("p", 0, "GRASP", None, SimContext(1.0, 1.0, {}, (0.0,)), np.zeros(3))
    for name, save, load in [
        ("episodes.npz", lambda p, hz: save_episodes(p, [e], hz), lambda p, hz: load_episodes(p, control_hz=hz)),
        ("bank.npz", lambda p, hz: save_bank(p, [st], hz), lambda p, hz: load_bank(p, control_hz=hz)),
        ("actor.npz", lambda p, hz: save_actor_npz(p, Actor(2, 3, 1, 3, 4, 8, 1).export(), ("GRASP",), hz),
         lambda p, hz: load_actor_npz(p, control_hz=hz)),
        ("wm.pt", lambda p, hz: save_wm(p, WorldModel(2, 1, 4, 3, 8, 1), hz), lambda p, hz: load_wm(p, 2, 1, 4, control_hz=hz)),
    ]:
        new, old = tmp_path / name, tmp_path / ("old_" + name)
        save(new, 50)
        load(new, 50)
        # legacy file: same content without the field
        if name.endswith(".pt"):
            sd = torch.load(new)
            del sd["control_hz"]
            torch.save(sd, old)
        else:
            np.savez(old, **{k: v for k, v in np.load(new).items() if k != "control_hz"})
            if name == "bank.npz":
                shutil.copy(new.with_suffix(".json"), old.with_suffix(".json"))
        for f in (new, old):
            with pytest.raises(ValueError, match="50 Hz"):
                load(f, 10)
            load(f, 50)
