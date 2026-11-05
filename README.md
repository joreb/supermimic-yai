# Supermimic YAI

Yet Another Implementation of [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026).

- `wiki/` — paper notes, a git submodule pointing at a private repo (`joreb/supermimic-wiki`). Collaborators with access: `git submodule update --init`.
- `src/supermimic_yai/` — implementation (work in progress).

## Demo: Product A assembled by scripted action tokens

![Franka Panda assembling Product A in MuJoCo (2x speed)](media/product_a.gif)

Scripted baseline (Plan 1), seed 0, played at 2x. Full-length video: [media/product_a.mp4](media/product_a.mp4). The learned SuperMimic-style controller replaces the contact tokens in Plan 2.

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

## Train

Dyna-style training of the learned controller from the Plan 2a world model (`--init` dir holds `episodes.npz` and `model.pt`). B and C are held out: never trained on, only evaluated. `--arm` is required; `--set key=val` overrides any config value. On the laptop GPU (GTX 1050, 4 GB) use `--set episodes_per_iter=8`.

```
docker run --rm --gpus all -p 6006:6006 -v "$PWD":/work -w /work supermimic-yai \
  python -m supermimic_yai.train --config laptop --arm arms/panda --init runs/wm \
  --products products/product_a --heldout products/product_b products/product_c \
  --bank runs/train/bank_a.npz --eval-banks runs/eval/bank_b.npz runs/eval/bank_c.npz \
  --iters 200 --set episodes_per_iter=8
```

Add `--resume` to continue from the last checkpoint. Watch curves and eval videos in a second container:

```
docker run --rm -p 6006:6006 -v "$PWD":/work -w /work supermimic-yai \
  tensorboard --logdir runs --bind_all --port 6006
```

then open http://localhost:6006 (publish the port on whichever container runs TensorBoard).

## Evaluate

Per-token and full-recipe success on EVAL seeds for products A, B and C; `--controller scripted` gives the baseline column (PLACE and PRESS are scripted in every table).

```
docker run --rm --gpus all -v "$PWD":/work -w /work supermimic-yai \
  python tools/eval.py --actor runs/train/actor.npz --seeds 20
```

Results land in `runs/eval/results.md`.

## Watch live

Open a MuJoCo window (WSLg) that follows the recipe in real time. Without `DISPLAY`/`WAYLAND_DISPLAY` the command exits 2.

```
docker run --rm --gpus all -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -v "$PWD":/work -w /work supermimic-yai \
  python -m supermimic_yai.cli run-recipe products/product_a recipes/product_a.json --viewer
```

Add `--controller learned --actor runs/train/actor.npz` to watch the trained policy. Rebuild the image once (`docker build -t supermimic-yai .`) to pick up `libglfw3`.
