"""Per-token and full-recipe eval on EVAL seeds.
python tools/eval.py --actor runs/bc/actor.npz --products products/product_a products/product_b products/product_c --seeds 20
--controller scripted gives the baseline column. Banks: runs/eval/bank_<last letter of product dir>.npz (startbank --seed-range eval).
"""
import argparse
import tempfile
from pathlib import Path

from supermimic_yai import seeds as seedmod
from supermimic_yai.config import load_config
from supermimic_yai.evaluate import eval_recipes, eval_tokens, log_video, results_markdown
from supermimic_yai.npmlp import learned_kinds, load_actor_npz
from supermimic_yai.startbank import load_bank, save_bank
from supermimic_yai.task import TOKEN_KINDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="runs/train/actor.npz")
    ap.add_argument("--controller", choices=["learned", "scripted"], default="learned")
    ap.add_argument("--learned-kinds", nargs="+", choices=TOKEN_KINDS, help="default: the kinds stored in the actor file")
    ap.add_argument("--products", nargs="+", default=["products/product_a", "products/product_b", "products/product_c"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--config", default="laptop")
    ap.add_argument("--arm", default="arms/panda")
    ap.add_argument("--bank-dir", default="runs/eval")
    ap.add_argument("--out", default="runs/eval/results.md")
    ap.add_argument("--video-dir")
    a = ap.parse_args()
    cfg = load_config(a.config)
    w = load_actor_npz(a.actor) if a.controller == "learned" else None
    kinds = learned_kinds(w, a.learned_kinds)
    states = [s for d in a.products for s in load_bank(Path(a.bank_dir) / f"bank_{Path(d).name.split('_')[-1]}.npz")]
    with tempfile.TemporaryDirectory() as t:
        save_bank(Path(t) / "b.npz", states)
        tokens = eval_tokens(cfg, a.arm, a.products, Path(t) / "b.npz", w, a.seeds, learned_kinds=kinds)
    recipes = eval_recipes(a.arm, a.products, w, seedmod.EVAL[:a.seeds], record_first=a.video_dir is not None,
                           action_scale=cfg["action_scale"], learned_kinds=kinds)
    md = results_markdown(tokens, recipes, learned_kinds=kinds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md)
    print(md)
    if a.video_dir:
        from torch.utils.tensorboard import SummaryWriter
        tw = SummaryWriter(a.video_dir)
        for p, v in recipes.items():
            if v["frames"]:
                log_video(tw, f"eval/{p}", v["frames"], 0)
        tw.close()


if __name__ == "__main__":
    main()
