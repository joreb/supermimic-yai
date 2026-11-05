import contextlib
import os
import time

DISPLAY_VARS = ("DISPLAY", "WAYLAND_DISPLAY")


def has_display() -> bool:
    return any(os.environ.get(v) for v in DISPLAY_VARS)


@contextlib.contextmanager
def watch(env):
    """Open a passive viewer and sync it after every control step, paced to real time."""
    os.environ["MUJOCO_GL"] = "glfw"
    import mujoco.viewer

    dt = env.substeps * env.model.opt.timestep
    step = env.step
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        def synced(action):
            t = time.perf_counter()
            out = step(action)
            v.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t)))
            return out
        env.step = synced
        try:
            yield v
        finally:
            env.step = step
