import argparse
import contextlib
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    from supermimic_yai.task import TOKEN_KINDS
    ap = argparse.ArgumentParser(prog="supermimic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-recipe")
    r.add_argument("assembly_dir", type=Path)
    r.add_argument("recipe", type=Path)
    r.add_argument("--arm", type=Path, help="arm directory (default: the only directory under arms/)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--video", type=Path)
    r.add_argument("--controller", choices=["scripted", "learned"], default="scripted")
    r.add_argument("--actor", type=Path, help="actor .npz (required for --controller learned)")
    r.add_argument("--learned-kinds", nargs="+", choices=TOKEN_KINDS, help="default: the kinds stored in the actor file")
    r.add_argument("--viewer", action="store_true", help="watch live in a MuJoCo window (needs a display, e.g. WSLg)")
    a = ap.parse_args(argv)
    if a.viewer:
        from supermimic_yai.viewer import has_display
        if not has_display():
            print("error: --viewer needs a display; pass -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR "
                  "-v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg (WSLg)", file=sys.stderr)
            return 2
        os.environ["MUJOCO_GL"] = "glfw"  # before any mujoco import: the image default (osmesa) would bind the GL loader
    if a.controller == "learned" and a.actor is None:
        print("error: --controller learned needs --actor PATH.npz", file=sys.stderr)
        return 2

    from supermimic_yai.arm import load_arm
    from supermimic_yai.config import load_config
    from supermimic_yai.env import AssemblyEnv
    from supermimic_yai.product import load_product
    from supermimic_yai.score import score
    from supermimic_yai.scripted import execute
    from supermimic_yai.tokens import describe, load_recipe

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
    ctrl = None
    if a.controller == "learned":
        from supermimic_yai.npmlp import load_actor_npz
        from supermimic_yai.runner import LearnedController
        ctrl = LearnedController(load_actor_npz(a.actor), load_config("laptop")["action_scale"], a.learned_kinds)
    with contextlib.ExitStack() as stack:
        if a.viewer:
            from supermimic_yai.viewer import watch
            stack.enter_context(watch(env))
        log = execute(env, recipe, record=a.video is not None, controller=ctrl)
    tags = ctrl.tags if ctrl else ["scripted"] * len(log.results)
    for line, res, tag in zip(lines, log.results, tags):
        print(f"{line} [{tag}]: {'ok' if res.ok else 'FAILED (' + res.reason + ')'} [{res.sim_time:.2f} s]")
    report = score(product, env.obs()["part_poses"])
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
