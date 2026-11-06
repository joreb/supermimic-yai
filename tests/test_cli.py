import json
from pathlib import Path

from supermimic_yai.cli import main

ROOT = Path(__file__).parent.parent


def test_invalid_recipe_exits_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"product": "nope", "steps": []}))
    assert main(["run-recipe", str(ROOT / "products" / "product_a"), str(bad), "--arm", str(ROOT / "arms" / "panda")]) == 2


def test_bad_arm_yaml_exits_2(tmp_path):
    (tmp_path / "arm.yaml").write_text("mjcf: x\n")  # missing keys
    assert main(["run-recipe", str(ROOT / "products" / "product_a"), str(ROOT / "recipes" / "product_a.json"), "--arm", str(tmp_path)]) == 2


def test_run_recipe_rejects_unknown_learned_kind(tmp_path, capsys):
    from supermimic_yai.cli import main
    root = Path(__file__).parent.parent
    rc = main(["run-recipe", str(root / "products/product_a"), str(root / "recipes/product_a.json"), "--arm", str(root / "arms/panda"),
               "--controller", "learned", "--actor", str(tmp_path / "x.npz"), "--learned-kinds", "NOPE"])
    assert rc == 2 and "NOPE" in capsys.readouterr().err
