from pathlib import Path

from supermimic_yai.product import load_product
from supermimic_yai.score import score

P = load_product(Path(__file__).parent.parent / "products" / "product_a")
FINAL = {n: s.final_pose for n, s in P.parts.items()}


def test_score_exact_is_success():
    r = score(P, FINAL)
    assert r.success and all(s.pos_err == 0 and s.rot_err_deg < 1e-6 for s in r.parts)


def test_score_offset_fails():
    poses = dict(FINAL)
    x, *rest = poses["peg1"]
    poses["peg1"] = (x + 0.01, *rest)
    r = score(P, poses)
    assert not r.success and [s.part for s in r.parts if not s.ok] == ["peg1"]
