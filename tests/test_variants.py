import filecmp
import json
import sys
from pathlib import Path

import pytest
import yaml
from trimesh.collision import CollisionManager

from supermimic_yai.product import load_product
from test_product import PRODUCTS, placed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import make_products  # noqa: E402
import make_variants  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "variants.yaml").read_text())


def test_a_unchanged(tmp_path):
    make_products.OUT = tmp_path
    try:
        make_products.product_a()
    finally:
        make_products.OUT = ROOT / "products"
    cmp = filecmp.dircmp(tmp_path / "product_a", PRODUCTS / "product_a")
    assert not (cmp.diff_files or cmp.left_only or cmp.right_only)
    sub = filecmp.dircmp(tmp_path / "product_a" / "parts", PRODUCTS / "product_a" / "parts")
    assert sorted(sub.same_files) == sorted(sub.left_list) == sorted(sub.right_list)


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("v")
    make_variants.make(4, 0, out, verify=False)
    return out


@pytest.mark.parametrize("i", range(4))
def test_variant_valid(made, i):
    p = make_variants.sample(4, 0)[i]
    for k, (lo, hi) in CFG.items():
        assert lo <= p[k] <= hi
    assert p["clearance"] > 0
    prod = load_product(made / f"a_v{i:03d}")
    assert prod.name == f"a_v{i:03d}"
    cm = CollisionManager()
    for n, s in prod.parts.items():
        cm.add_object(n, placed(s, s.final_pose))
        assert s.mass == pytest.approx(placed(s, s.final_pose).volume * 1200, abs=1e-3)
    hit, _, contacts = cm.in_collision_internal(return_names=True, return_data=True)
    assert max((c.depth for c in contacts), default=0.0) < 0.0005, hit


def test_heldout_outside_ranges():
    lo, hi = CFG["peg_len"]
    for n in ("product_b", "product_c"):
        peg = load_product(PRODUCTS / n).parts["peg"]
        z = placed(peg, peg.start_pose).bounds[:, 2]
        assert not lo <= z[1] - z[0] <= hi


def test_deterministic(made, tmp_path):
    make_variants.make(4, 0, tmp_path / "x", verify=False)
    cmp = filecmp.dircmp(made, tmp_path / "x")
    assert not (cmp.diff_files or cmp.left_only or cmp.right_only)
    for d in cmp.common_dirs:
        assert not filecmp.dircmp(made / d, tmp_path / "x" / d).diff_files
        assert not filecmp.dircmp(made / d / "parts", tmp_path / "x" / d / "parts").diff_files


@pytest.mark.slow
def test_variant_recipe_succeeds(made):
    # a_v002 / a_v003 are kept by `make_variants.py --n 32 --seed 0`; others are legitimately rejected
    for i in (2, 3):
        name, ok, detail = make_variants.check((str(made / f"a_v{i:03d}"), 0))
        assert ok, (name, detail)


def test_peg_grasp_z():
    assert make_variants.peg_grasp_z(make_products.A_PARAMS) == 0
    assert make_variants.peg_grasp_z({**make_products.A_PARAMS, "peg_len": .030, "base_t": .012}) == 0.005
    # A's recipe is reproduced exactly
    r = make_variants.variant_recipe("product_a", make_products.A_PARAMS)
    assert r == json.loads((ROOT / "recipes" / "product_a.json").read_text())


def test_out_guard(tmp_path):
    d = tmp_path / "precious"
    d.mkdir()
    (d / "f.txt").write_text("x")
    with pytest.raises(SystemExit):
        make_variants.make(1, 0, d, verify=False)
    assert (d / "f.txt").read_text() == "x"
