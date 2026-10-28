"""Fit the world model on scripted-noise data. python tools/fit_wm.py --config laptop --episodes 2000 --iters 20000 --logdir runs/wm

Needs runs/wm/bank.npz (python -m supermimic_yai.startbank ...). Torch is imported only after collection: Collector
workers are spawned from this module and must never import torch. --reuse skips collection (loads episodes.npz).
"""
import argparse
import time
from pathlib import Path

import numpy as np


def save_eps(path, eps):
    d = {k: np.concatenate([e[k] for e in eps]) for k in ("s", "a", "r", "done", "unstable")}
    d["len"] = np.array([len(e["a"]) for e in eps])
    for k in ("goal", "ctx"):
        d[k] = np.stack([e[k] for e in eps])
    d["kind"] = np.array([e["kind"] for e in eps])
    d["terminal_kind"] = np.array([e["terminal_kind"] for e in eps])
    np.savez_compressed(path, **d)


def load_eps(path):
    d, eps, i = np.load(path), [], 0
    for j, T in enumerate(d["len"]):
        sl, s0 = slice(i, i + T), i + j
        eps.append({"s": d["s"][s0:s0 + T + 1], **{k: d[k][sl] for k in ("a", "r", "done", "unstable")},
                    "goal": d["goal"][j], "ctx": d["ctx"][j], "kind": int(d["kind"][j]),
                    "terminal_kind": str(d["terminal_kind"][j])})
        i += T
    return eps


def auc(score, y):
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    p = y.sum()
    return float((r[y].sum() - p * (p + 1) / 2) / (p * (len(y) - p))) if 0 < p < len(y) else float("nan")


def buffer(eps, k):
    from supermimic_yai.replay import ReplayBuffer
    sd, ad = eps[0]["s"].shape[1], eps[0]["a"].shape[1]
    buf = ReplayBuffer(sum(len(e["a"]) for e in eps), sd, ad, eps[0]["goal"].shape[0], eps[0]["ctx"].shape[0])
    for e in eps:
        buf.add(e)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="laptop")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--logdir", default="runs/wm")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--products", nargs="*", default=["products/product_a", *sorted(map(str, Path("products_train").iterdir()))])
    ap.add_argument("--arm", default="arms/panda")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    from supermimic_yai.config import load_config
    cfg, out = load_config(a.config), Path(a.logdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if a.reuse:
        eps = load_eps(out / "episodes.npz")
    else:
        from supermimic_yai.collect import Collector
        with Collector(cfg, a.arm, a.products, a.bank or out / "bank.npz") as c:
            eps = c.collect(a.episodes, None, rng_seed=0)
        save_eps(out / "episodes.npz", eps)
    t_collect = time.perf_counter() - t0
    print(f"{len(eps)} episodes, {sum(len(e['a']) for e in eps)} transitions, collect/load {t_collect:.0f} s", flush=True)
    fit(cfg, eps, a.iters, out, t_collect)


def fit(cfg, eps, iters, out, t_collect=0.0):
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from supermimic_yai.task import TOKEN_KINDS
    from supermimic_yai.wm import WorldModel, rollout, wm_loss
    w, k = cfg["wm"], cfg["wm"]["k"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # no-op on Pascal
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(eps))
    nv = max(1, len(eps) // 10)
    val, train = [eps[i] for i in perm[:nv]], [eps[i] for i in perm[nv:]]
    tb, vb = buffer(train, k), buffer(val, k)
    sd, ad, cd = train[0]["s"].shape[1], train[0]["a"].shape[1], train[0]["ctx"].shape[0]
    model = WorldModel(sd, ad, cd, len(TOKEN_KINDS), w["width"], w["depth"]).to(dev)
    model.set_norm(tb.s[:tb.size], tb.s2[:tb.size], tb.ctx[:tb.size])
    opt = torch.optim.Adam(model.parameters(), w["lr"])
    tens = lambda b: {kk: torch.as_tensor(v, device=dev, dtype=torch.int64 if kk == "kind" else torch.float32)  # noqa: E731
                      for kk, v in b.items() if kk in ("s", "a", "ctx", "kind", "done")}
    veval = tens(vb.sample_windows(k, 4096, np.random.default_rng(1)))
    tw = SummaryWriter(str(out))

    def evaluate(it):
        model.eval()
        with torch.no_grad():
            _, m = wm_loss(model, veval, k)
            _, lg = rollout(model, veval, k)
        model.train()
        y, p = veval["done"].cpu().numpy().ravel() > 0.5, lg.cpu().numpy().ravel()
        m["done_acc"], m["done_auc"] = float(((p > 0) == y).mean()), auc(p, y)
        m["done_pos_rate"] = float(y.mean())
        for name, v in m.items():
            tw.add_scalar(f"wm_val/{name}", v, it)
        tw.add_scalar("wm/val_loss", m["loss"], it)
        for name, v in m.items():
            if name.startswith("kstep_err/"):
                tw.add_scalar(f"wm/{name}", v, it)
        return m

    t1 = time.perf_counter()
    for it in range(1, iters + 1):
        batch = tens(tb.sample_windows(k, w["batch"], rng))
        loss, m = wm_loss(model, batch, k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), w["grad_clip"])
        opt.step()
        if it % 100 == 0:
            tw.add_scalar("wm/loss", m["loss"], it)
            tw.add_scalar("sys/it_per_s", it / (time.perf_counter() - t1), it)
            tw.add_scalar("sys/gpu_mem_mb", torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0, it)
        if it % 1000 == 0 or it == iters:
            r = evaluate(it)
            print(f"it {it} train {m['loss']:.2f} val {r['loss']:.2f} "
                  + " ".join(f"{kn} {r.get(f'kstep_err/{kn}', float('nan')) * 1e3:.2f}/{r.get(f'kstep_err_pt/{kn}', float('nan')) * 1e3:.2f}mm"
                             for kn in TOKEN_KINDS) + f" acc {r['done_acc']:.3f} auc {r['done_auc']:.3f}", flush=True)
    tw.close()
    torch.save(model.state_dict(), out / "model.pt")
    wall = time.perf_counter() - t1
    gpu = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
    print(f"fit wall {wall:.0f} s ({iters / wall:.1f} it/s), gpu peak {gpu:.0f} MB (allocated)")
    print("FINAL (val, mm): " + ", ".join(f"{n} {v * (1e3 if 'err' in n else 1):.3f}" for n, v in sorted(r.items())))


if __name__ == "__main__":
    main()
