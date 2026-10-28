"""One-shot generator for the frozen products. Run: python tools/make_products.py"""
import json
from pathlib import Path

import manifold3d as m3
import numpy as np
import trimesh

OUT = Path(__file__).parent.parent / "products"
SEG = 64
RHO = 1200.0
CX, CY = 0.5, 0.0  # base footprint centre, world
EPS = 0.001  # cutter overshoot


def box(sx, sy, z0, z1, cx=CX, cy=CY):
    return m3.Manifold.cube((sx, sy, z1 - z0), center=True).translate((cx, cy, (z0 + z1) / 2))


def cyl(d, z0, z1, cx=CX, cy=CY):
    c = m3.Manifold.cylinder(z1 - z0, d / 2, d / 2, SEG, center=True)
    return c.translate((cx, cy, (z0 + z1) / 2))


def hole(d, z0, z1, cx=CX, cy=CY):
    return cyl(d, z0 - EPS, z1 + EPS, cx, cy)


def to_mesh(man):
    mm = man.to_mesh()
    return trimesh.Trimesh(mm.vert_properties[:, :3].astype(float), mm.tri_verts, process=True)


def write(name, tol, parts, starts, novelty=None, out=None):
    """parts: {name: (manifold, fixed)}; starts: {name: (x, y)} for non-fixed parts."""
    root = (out or OUT) / name
    (root / "parts").mkdir(parents=True, exist_ok=True)
    spec = {}
    for pn, (man, fixed) in parts.items():
        mesh = to_mesh(man)
        centre = mesh.bounds.mean(axis=0)  # R9: origin at bbox centre
        mesh.apply_translation(-centre)
        mesh.export(root / "parts" / f"{pn}.stl")  # binary
        final = [*(round(float(v), 6) for v in centre), 1, 0, 0, 0]
        h = mesh.bounds[1, 2]
        start = final if fixed else [starts[pn][0], starts[pn][1], round(float(h), 6), 1, 0, 0, 0]
        spec[pn] = {"mesh": f"parts/{pn}.stl", "mass": round(float(mesh.volume) * RHO, 4),
                    "fixed": fixed, "start_pose": start, "final_pose": final}
    (root / "assembly.json").write_text(json.dumps(
        {"name": name, "units": "m", "tolerance": tol, "parts": spec}, indent=2) + "\n")
    if novelty:
        (root / "NOVELTY.md").write_text(novelty + "\n")


TOL = {"pos": 0.005, "rot_deg": 5.0}


A_PARAMS = dict(peg_d=.020, clearance=.002, peg_len=.040, base_t=.010, cover_t=.006, spacing=.060)


def product_a_params(peg_d, clearance, peg_len, base_t, cover_t, spacing,
                     name="product_a", out=None, novelty=None):
    hd, s = round(peg_d + clearance, 9), spacing / 2
    top = round(base_t + cover_t, 9)
    base = (box(.12, .06, 0, base_t) - hole(hd, 0, base_t, CX - s) - hole(hd, 0, base_t, CX + s))
    cover = (box(.10, .05, base_t, top) - hole(hd, base_t, top, CX - s)
             - hole(hd, base_t, top, CX + s))
    write(name, TOL, {
        "base": (base, True),
        "peg1": (cyl(peg_d, 0, peg_len, CX - s), False),
        "peg2": (cyl(peg_d, 0, peg_len, CX + s), False),
        "cover": (cover, False),
    }, {"peg1": (.42, -.22), "peg2": (.58, -.22), "cover": (.5, .22)}, novelty, out)


def product_a():
    product_a_params(**A_PARAMS, novelty="Baseline: two pegs through a base, then a cover over both.")


def product_b():
    # pocket centred at base-local (+30, +30) mm, floor at z=5 mm
    px, py = CX + .03, CY + .03
    base = (box(.10, .10, 0, .01) - hole(.022, 0, .01) - box(.04, .04, .005, .01 + EPS, px, py))
    # DEVIATION (flagged): cap OD 30 mm, not 40 -- a 40 mm cap overlaps the 38 mm block
    # in the corner pocket (block corner is 15.6 mm from the peg axis).
    cap = cyl(.03, .01, .025) - hole(.022, .01, .025)
    write("product_b", TOL, {
        "base": (base, True),
        "block": (box(.038, .038, .005, .025, px, py), False),
        "peg": (cyl(.02, 0, .05), False),
        "cap": (cap, False),
    }, {"block": (.40, .22), "cap": (.60, .22), "peg": (.5, -.22)},
        "pocket placement and a capped stack; neither join appears in A\n"
        "peg length 50 mm, outside the 30-45 mm range of A's variants")


def product_c():
    base = box(.08, .08, 0, .01) - hole(.022, 0, .01)
    parts = {"base": (base, True), "peg": (cyl(.02, 0, .06), False)}
    for n, d, z in (("ring_l", .06, .01), ("ring_m", .05, .02), ("ring_s", .04, .03)):
        parts[n] = (cyl(d, z, z + .01) - hole(.022, z, z + .01), False)
    write("product_c", TOL, parts,
          {"peg": (.5, -.22), "ring_l": (.38, .22), "ring_m": (.5, .22), "ring_s": (.62, .22)},
          "three-high stack whose order is fixed by size; A has no stacking deeper than one\n"
          "peg length 60 mm, outside the 30-45 mm range of A's variants")


if __name__ == "__main__":
    product_a(); product_b(); product_c()
