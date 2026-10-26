import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh
from trimesh.collision import CollisionManager
from trimesh.transformations import quaternion_matrix

from supermimic_yai.product import ProductError, load_product

PRODUCTS = Path(__file__).parent.parent / "products"
NAMES = ["product_a", "product_b", "product_c"]


def placed(spec, pose):
    m = trimesh.load(spec.mesh, force="mesh")
    t = quaternion_matrix(pose[3:])
    t[:3, 3] = pose[:3]
    m.apply_transform(t)
    return m


@pytest.mark.parametrize("name", NAMES)
def test_load_all_products(name):
    p = load_product(PRODUCTS / name)
    assert sum(s.fixed for s in p.parts.values()) == 1


def test_non_watertight_mesh_raises(tmp_path):
    shutil.copytree(PRODUCTS / "product_a", tmp_path / "a")
    stl = next((tmp_path / "a" / "parts").glob("*.stl"))
    trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]).export(stl)
    with pytest.raises(ProductError, match=stl.name):
        load_product(tmp_path / "a")


def test_missing_field_raises(tmp_path):
    shutil.copytree(PRODUCTS / "product_a", tmp_path / "a")
    f = tmp_path / "a" / "assembly.json"
    d = json.loads(f.read_text())
    part = next(iter(d["parts"]))
    del d["parts"][part]["final_pose"]
    f.write_text(json.dumps(d))
    with pytest.raises(ProductError, match=rf"{part}.*final_pose"):
        load_product(tmp_path / "a")


@pytest.mark.parametrize("name", NAMES)
def test_final_poses_do_not_interpenetrate(name):
    p = load_product(PRODUCTS / name)
    cm = CollisionManager()
    for n, s in p.parts.items():
        cm.add_object(n, placed(s, s.final_pose))
    hit, _, contacts = cm.in_collision_internal(return_names=True, return_data=True)
    depth = max((c.depth for c in contacts), default=0.0)
    assert depth < 0.0005, (hit, depth)


@pytest.mark.parametrize("name", NAMES)
def test_start_poses_on_table_and_separated(name):
    p = load_product(PRODUCTS / name)
    boxes = {}
    for n, s in p.parts.items():
        if s.fixed:
            continue
        b = placed(s, s.start_pose).bounds
        assert -1e-6 <= b[0, 2] <= 0.002, n
        boxes[n] = b
    names = list(boxes)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            gap = np.maximum(boxes[a][0, :2] - boxes[c][1, :2], boxes[c][0, :2] - boxes[a][1, :2])
            assert gap.max() >= 0.03, (a, c)
