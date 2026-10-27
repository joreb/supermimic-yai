# Supermimic YAI

Yet Another Implementation of [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026).

- `wiki/` — paper notes, a git submodule pointing at a private repo (`joreb/supermimic-wiki`). Collaborators with access: `git submodule update --init`.
- `src/supermimic_yai/` — implementation (work in progress).

## Docker

All tests run in the image (CPU, MuJoCo with OSMesa rendering):

```
docker build -t supermimic-yai .
docker run --rm -v "$PWD":/work -w /work supermimic-yai pytest -q
```

## Run a recipe

Execute a hand-written token recipe on a product and score the result (exit code 0 on success, 1 on failure, 2 on an invalid recipe):

```
docker run --rm -v "$PWD":/work -w /work supermimic-yai \
  python -m supermimic_yai.cli run-recipe products/product_a recipes/product_a.json --seed 0 --video out/product_a.mp4
```

Locally installed, the same command is `supermimic run-recipe ...`. Reference recipes for products A, B and C live in `recipes/`; `pytest -m slow` checks that each succeeds on at least 8 of 10 seeds.
