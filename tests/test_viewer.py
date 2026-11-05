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
    code = ("import sys\nfrom supermimic_yai.cli import main\n"
            f"main({ARGS!r})\n")
    code += "assert 'mujoco.viewer' not in sys.modules\nassert 'DISPLAY' not in __import__('os').environ\n"
    p = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
