import json
from pathlib import Path

from supermimic_yai.cli import main

ROOT = Path(__file__).parent.parent


def test_invalid_recipe_exits_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"product": "nope", "steps": []}))
    assert main(["run-recipe", str(ROOT / "products" / "product_a"), str(bad), "--arm", str(ROOT / "arms" / "panda")]) == 2
