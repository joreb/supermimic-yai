import copy
import json
from pathlib import Path

import pytest
import trimesh

from supermimic_yai.arm import load_arm
from supermimic_yai.product import load_product
from supermimic_yai.tokens import Grasp, Insert, MoveAbove, Recipe, RecipeError, Release, describe, parse_recipe

ROOT = Path(__file__).parent.parent
PRODUCT = load_product(ROOT / "products" / "product_a")
ARM = load_arm(ROOT / "arms" / "panda")

VALID = {
    "product": "product_a",
    "steps": [
        {"token": "MOVE_ABOVE", "target": "final:peg1", "offset": 0.05},
        {"token": "GRASP", "part": "peg1", "grasp_pose": [0, 0, 0, 1, 0, 0, 0]},
        {"token": "INSERT", "part": "peg1", "target": "final:peg1", "depth": 0.01, "max_force": 20},
        {"token": "RELEASE"},
        {"token": "RETREAT", "distance": 0.05},
    ],
}


def recipe_with(**edits):
    d = copy.deepcopy(VALID)
    for idx, change in edits.items():
        d["steps"][int(idx)].update(change)
    return d


def test_parse_valid():
    r = parse_recipe(copy.deepcopy(VALID), PRODUCT, ARM)
    assert isinstance(r, Recipe)
    assert [type(s) for s in r.steps] == [MoveAbove, Grasp, Insert, Release, type(r.steps[4])]
    assert r.steps[1].part == "peg1"


def test_unknown_token():
    d = copy.deepcopy(VALID)
    d["steps"][2] = {"token": "FLY"}
    with pytest.raises(RecipeError, match=r"^step 2: unknown token"):
        parse_recipe(d, PRODUCT, ARM)


def test_unknown_part():
    d = recipe_with(**{"1": {"part": "nope"}})
    with pytest.raises(RecipeError, match="nope"):
        parse_recipe(d, PRODUCT, ARM)


def test_missing_param():
    d = copy.deepcopy(VALID)
    del d["steps"][2]["depth"]
    with pytest.raises(RecipeError, match="depth"):
        parse_recipe(d, PRODUCT, ARM)


def test_part_too_wide_for_gripper(tmp_path):
    base = trimesh.creation.box(extents=(0.1, 0.1, 0.02))
    big = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    prod = tmp_path / "wide"
    (prod / "parts").mkdir(parents=True)
    base.export(prod / "parts" / "base.stl")
    big.export(prod / "parts" / "big.stl")
    pose_base = [0, 0, 0.01, 1, 0, 0, 0]
    d = {
        "name": "wide",
        "tolerance": {"pos": 0.005, "rot_deg": 5.0},
        "parts": {
            "base": {"mesh": "parts/base.stl", "mass": 0.1, "fixed": True,
                     "start_pose": pose_base, "final_pose": pose_base},
            "big": {"mesh": "parts/big.stl", "mass": 0.1, "fixed": False,
                    "start_pose": [0.3, 0, 0.1, 1, 0, 0, 0], "final_pose": [0.3, 0.3, 0.1, 1, 0, 0, 0]},
        },
    }
    (prod / "assembly.json").write_text(json.dumps(d))
    product = load_product(prod)
    recipe = {"product": "wide", "steps": [{"token": "GRASP", "part": "big", "grasp_pose": [0, 0, 0, 1, 0, 0, 0]}]}
    with pytest.raises(RecipeError, match="big"):
        parse_recipe(recipe, product, ARM)


def test_grasp_without_release():
    grasp = {"token": "GRASP", "part": "peg1", "grasp_pose": [0, 0, 0, 1, 0, 0, 0]}
    d = {"product": "product_a", "steps": [grasp, dict(grasp, part="peg2")]}
    with pytest.raises(RecipeError, match=r"^step 1: .*RELEASE"):
        parse_recipe(d, PRODUCT, ARM)


def test_describe_one_line_per_step():
    r = parse_recipe(copy.deepcopy(VALID), PRODUCT, ARM)
    lines = describe(r)
    assert len(lines) == len(r.steps)
    assert all("\n" not in line for line in lines)
    assert lines[2].startswith("3. Insert peg1 into its final position, 10 mm deep")


def test_non_finite_number_rejected():
    d = recipe_with(**{"0": {"offset": float("nan")}})
    with pytest.raises(RecipeError, match="offset"):
        parse_recipe(d, PRODUCT, ARM)
