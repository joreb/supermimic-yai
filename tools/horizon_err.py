"""Held-out H-step world-model error. python tools/horizon_err.py --model runs/wm/model.pt --episodes runs/wm/episodes.npz --H 8 16 32

Held-out = last 10 % of episodes (as fit_wm). Mean over every complete, stable H-window, per kind, in mm:
part-to-target error (pt) and held-part error (held), seeded on logged actions.
"""
import argparse

import numpy as np
import torch
from fit_wm import buffer, load_eps

from supermimic_yai.task import TOKEN_KINDS
from supermimic_yai.wm import WorldModel, gate_errors, rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/wm/model.pt")
    ap.add_argument("--episodes", default="runs/wm/episodes.npz")
    ap.add_argument("--H", type=int, nargs="+", default=[8, 16, 32])
    a = ap.parse_args()
    eps = load_eps(a.episodes)
    val = eps[len(eps) - max(1, len(eps) // 10):]
    sd, ad, cd = val[0]["s"].shape[1], val[0]["a"].shape[1], val[0]["ctx"].shape[0]
    st = torch.load(a.model, map_location="cpu")
    lin = [k for k in st if k.startswith("net.") and k.endswith("weight")]
    model = WorldModel(sd, ad, cd, len(TOKEN_KINDS), st[lin[0]].shape[0], len(lin) - 1)
    model.load_state_dict(st)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    n = (sd - 17) // 2
    print(f"held-out episodes {len(val)}; errors in mm, pt = part-to-target, held = held part rel EE")
    print("H    windows  " + "  ".join(f"{k}:pt/held" for k in TOKEN_KINDS))
    for H in a.H:
        buf = buffer(val, H)
        rows = buf._starts(H)
        idx = (rows[:, None] + np.arange(H)) % buf.cap
        tot = {k: [[], []] for k in range(len(TOKEN_KINDS))}
        for i in range(0, len(rows), 4096):
            r, ix = rows[i:i + 4096], idx[i:i + 4096]
            t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)  # noqa: E731
            s = t(np.concatenate([buf.s[ix], buf.s2[ix[:, -1]][:, None]], 1))
            b = {"s": s, "a": t(buf.a[ix]), "ctx": t(buf.ctx[r]), "kind": t(buf.kind[r], torch.int64)}
            with torch.no_grad():
                s_hat, _ = rollout(model, b, H)
                held, pt = gate_errors(s_hat, s, H, n)
            for k in tot:
                m = b["kind"] == k
                tot[k][0].append(pt[m].cpu()), tot[k][1].append(held[m].cpu())
        cell = lambda k: (f"{torch.cat(tot[k][0]).mean() * 1e3:6.2f}/{torch.cat(tot[k][1]).mean() * 1e3:6.2f}"  # noqa: E731
                          if len(torch.cat(tot[k][0])) else "   n/a")
        print(f"{H:<4} {len(rows):<8} " + "  ".join(f"{cell(k):>{len(TOKEN_KINDS[k]) + 9}}" for k in tot), flush=True)


if __name__ == "__main__":
    main()
