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
