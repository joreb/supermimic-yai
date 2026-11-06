"""Numpy-only actor forward + npz io (workers never import torch). Layers = [(W, b), ...], W shaped (in, out)."""
import numpy as np


def _mlp(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
    return x


def actor_forward(w, s, goal, kind) -> np.ndarray:
    """w = Actor.export(). z = tanh(enc_kind(goal)); a = dec([norm(s), z]); arm tanh, last channel (tanh+1)/2."""
    s, goal = np.asarray(s, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    z = np.tanh(_mlp(w["enc"][kind], goal))
    a = np.tanh(_mlp(w["dec"], np.concatenate([(s - w["s_mu"]) / w["s_sd"], z], -1)))
    a[..., -1] = (a[..., -1] + 1) / 2
    return a


def save_actor_npz(path, w, kinds):
    """kinds: the token kinds this actor was trained for (the default learned kinds when it is evaluated or watched)."""
    d = {"s_mu": w["s_mu"], "s_sd": w["s_sd"], "kinds": np.array(kinds)}
    for name, layers in (("dec", w["dec"]),) + tuple((f"enc{k}", l) for k, l in enumerate(w["enc"])):
        for i, (W, b) in enumerate(layers):
            d[f"{name}_W{i}"], d[f"{name}_b{i}"] = W, b
    np.savez(path, **d)


def load_actor_npz(path) -> dict:
    z = np.load(path)

    def layers(name):
        n = sum(k.startswith(f"{name}_W") for k in z.files)
        return [(z[f"{name}_W{i}"], z[f"{name}_b{i}"]) for i in range(n)]

    n_kinds = sum(k.startswith("enc") and "_W0" in k for k in z.files)
    return {"s_mu": z["s_mu"], "s_sd": z["s_sd"], "enc": [layers(f"enc{k}") for k in range(n_kinds)], "dec": layers("dec"),
            "kinds": tuple(map(str, z["kinds"])) if "kinds" in z.files else ("GRASP",)}  # legacy files: the BC actor was GRASP-only


def learned_kinds(w, override=None):
    """Kinds the actor drives: the override, else the ones stored with it; none without an actor."""
    return tuple(override) if override is not None else tuple(w.get("kinds", ())) if w else ()
