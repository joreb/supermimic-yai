import dataclasses
from pathlib import Path

import numpy as np
import pytest

from supermimic_yai.arm import load_arm
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import load_product
from supermimic_yai.scripted import execute, run_token
from supermimic_yai.startbank import collect_start_states, load_bank, save_bank
from supermimic_yai.tokens import load_recipe

ROOT = Path(__file__).parent.parent
ARM = load_arm(ROOT / "arms" / "panda")


@pytest.fixture(scope="module")
def a():
    product = load_product(ROOT / "products" / "product_a")
    recipe = load_recipe(ROOT / "recipes" / "product_a.json", product, ARM)
    return product, recipe, collect_start_states(ARM, product, recipe, [0])


def test_bank_covers_contact_tokens(a):
    product, recipe, bank = a
    want = [i for i, t in enumerate(recipe.steps) if t.kind in ("GRASP", "ALIGN", "INSERT")]
    assert len(want) == 9 and [s.step for s in bank] == want
    assert all(s.product == product.name and s.token == recipe.steps[s.step].kind for s in bank)
    assert all(s.held is None for s in bank if s.token == "GRASP")


def test_restore_reproduces(a):
    product, recipe, bank = a
    k = next(s for s in bank if s.token == "INSERT")
    env = AssemblyEnv(ARM, product)
    env.reset(seed=5)  # different context/state first
    execute(env, recipe)
    env.restore(k)
    held, got = k.held, []
    for tok in recipe.steps[k.step:]:
        res, held = run_token(env, tok, held)
        got.append(res)
    env.reset(seed=0)
    ref = execute(env, recipe).results[k.step:]
    assert got == ref


def test_restore_mismatch_raises(a):
    product, _, bank = a
    env = AssemblyEnv(ARM, product)
    with pytest.raises(ValueError):
        env.restore(dataclasses.replace(bank[0], product="other"))
    with pytest.raises(ValueError):
        env.restore(dataclasses.replace(bank[0], state=bank[0].state[:-1]))


def test_save_load_roundtrip(a, tmp_path):
    _, _, bank = a
    save_bank(tmp_path / "bank", bank)
    back = load_bank(tmp_path / "bank")
    assert len(back) == len(bank)
    for x, y in zip(bank, back):
        assert (x.product, x.step, x.token, x.held, x.context) == (y.product, y.step, y.token, y.held, y.context)
        assert np.array_equal(x.state, y.state)
