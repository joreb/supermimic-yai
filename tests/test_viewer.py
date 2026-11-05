import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARGS = ["run-recipe", str(ROOT / "products" / "product_a"), str(ROOT / "recipes" / "product_a.json"),
        "--arm", str(ROOT / "arms" / "panda")]


def _env():
    e = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    e["PYTHONPATH"] = str(ROOT / "src")
    return e


def test_viewer_flag_without_display_exits_2():
    p = subprocess.run([sys.executable, "-m", "supermimic_yai.cli", *ARGS, "--viewer"],
                       env=_env(), capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 2
    assert "DISPLAY" in p.stderr and "WAYLAND_DISPLAY" in p.stderr


def test_headless_never_imports_viewer():
    code = ("import os, sys\nfrom supermimic_yai.cli import main\n"
            f"main({ARGS!r})\n")
    code += "assert 'mujoco.viewer' not in sys.modules\nassert os.environ.get('MUJOCO_GL') != 'glfw'\nassert 'DISPLAY' not in __import__('os').environ\n"
    p = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr


def test_viewer_selects_glfw_before_mujoco_loads():
    code = f"""
import contextlib, os, sys, types
m = types.ModuleType("supermimic_yai.viewer")
m.seen = {{}}
def has_display():
    m.seen["mujoco_loaded"] = "mujoco" in sys.modules
    return True
class Stop(Exception): pass
@contextlib.contextmanager
def watch(env):
    m.seen["gl"] = os.environ.get("MUJOCO_GL")
    raise Stop
    yield
m.has_display, m.watch = has_display, watch
sys.modules["supermimic_yai.viewer"] = m
from supermimic_yai.cli import main
try:
    main({ARGS!r} + ["--viewer"])
except Stop:
    pass
assert m.seen == {{"mujoco_loaded": False, "gl": "glfw"}}, m.seen
"""
    e = _env() | {"MUJOCO_GL": "osmesa"}
    p = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
