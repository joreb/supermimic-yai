from pathlib import Path

import numpy as np
import pytest

from supermimic_yai import seeds
from supermimic_yai.evaluate import eval_recipes, log_video, results_markdown

ROOT = Path(__file__).parent.parent
A = str(ROOT / "products" / "product_a")


def test_eval_rejects_non_eval_seeds():
    with pytest.raises(ValueError):
        eval_recipes(ROOT / "arms" / "panda", [A], None, [0, seeds.EVAL[0]])


def test_log_video_smoke(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(str(tmp_path))
    log_video(w, "v", np.zeros((3, 480, 640, 3), np.uint8), 0)
    w.close()
    assert list(tmp_path.iterdir())


def test_results_markdown_labels_scripted():
    toks = {("a", k): {"success": 1.0, "n": 2, "terminal": {}} for k in ("GRASP", "ALIGN")}
    md = results_markdown(toks, {"a": {"success": 0.5, "per_token": [1.0, 0.5], "frames": None}})
    rows = {l.split("|")[2].strip(): l.split("|")[3].strip() for l in md.splitlines() if l.startswith("| a |")}
    assert rows["GRASP"] == "learned" and rows["ALIGN"] == "scripted"
    assert rows["PLACE"] == rows["PRESS"] == "scripted"
    assert "learned" not in results_markdown(toks, {}, learned_kinds=())


@pytest.mark.slow
def test_eval_recipes_scripted_baseline():
    r = eval_recipes(ROOT / "arms" / "panda", [A], None, seeds.EVAL[:2], record_first=True)["product_a"]
    from supermimic_yai.arm import load_arm
    from supermimic_yai.product import load_product
    from supermimic_yai.tokens import load_recipe
    n = len(load_recipe(ROOT / "recipes" / "product_a.json", load_product(Path(A)), load_arm(ROOT / "arms" / "panda")).steps)
    assert 0.0 <= r["success"] <= 1.0 and len(r["per_token"]) == n
    assert r["frames"]


def test_eval_bank_from_eval_seeds(tmp_path):
    from supermimic_yai.arm import load_arm
    from supermimic_yai.context import sample_context
    from supermimic_yai.env import AssemblyEnv
    from supermimic_yai.product import load_product
    from supermimic_yai.startbank import load_bank, main
    main(["--products", A, "--seeds", "2", "--seed-range", "eval", "--arm", str(ROOT / "arms" / "panda"),
          "--out", str(tmp_path / "b.npz")])
    product = load_product(Path(A))
    env = AssemblyEnv(load_arm(ROOT / "arms" / "panda"), product)
    want = [sample_context(np.random.default_rng(s), product, env.n_arm) for s in seeds.EVAL[:2]]
    got = load_bank(tmp_path / "b")
    assert got and all(s.context in want for s in got) and {want.index(s.context) for s in got} == {0, 1}
