import argparse
import sys
from pathlib import Path

from supermimic_yai.arm import ArmError, load_arm
from supermimic_yai.env import AssemblyEnv
from supermimic_yai.product import ProductError, load_product
from supermimic_yai.score import score
from supermimic_yai.scripted import execute
from supermimic_yai.tokens import RecipeError, describe, load_recipe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="supermimic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-recipe")
    r.add_argument("assembly_dir", type=Path)
    r.add_argument("recipe", type=Path)
    r.add_argument("--arm", type=Path, help="arm directory (default: the only directory under arms/)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--video", type=Path)
    a = ap.parse_args(argv)

    arm_dir = a.arm
    if arm_dir is None:
        found = [d for d in Path("arms").glob("*") if d.is_dir()]
        if len(found) != 1:
            print("error: pass --arm (expected exactly one directory under arms/)", file=sys.stderr)
            return 2
        arm_dir = found[0]
    try:
        arm, product = load_arm(arm_dir), load_product(a.assembly_dir)
        recipe = load_recipe(a.recipe, product, arm)
    except (ValueError, KeyError, TypeError, OSError) as e:  # incl. ArmError, ProductError, RecipeError
        print(f"error: {e}", file=sys.stderr)
        return 2

    lines = describe(recipe)
    print("\n".join(lines))
    env = AssemblyEnv(arm, product)
    env.reset(seed=a.seed)
    log = execute(env, recipe, record=a.video is not None)
    for line, res in zip(lines, log.results):
        print(f"{line}: {'ok' if res.ok else 'FAILED (' + res.reason + ')'} [{res.sim_time:.2f} s]")
    report = score(product, env._obs()["part_poses"])
    for s in report.parts:
        print(f"{s.part}: pos_err {s.pos_err * 1000:.1f} mm, rot_err {s.rot_err_deg:.1f} deg, {'ok' if s.ok else 'FAIL'}")
    if a.video is not None:
        import imageio
        a.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(a.video, log.frames, fps=25)
    ok = report.success and log.stopped_at is None
    print("SUCCESS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
