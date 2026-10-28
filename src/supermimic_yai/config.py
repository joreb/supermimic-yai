"""YAML scale presets: load_config("laptop") reads configs/laptop.yaml; a path to a .yaml file also works."""
import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name_or_path) -> dict:
    p = Path(name_or_path)
    if not p.is_file():
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(p.read_text())
    if cfg["collect"].get("workers") is None:
        cfg["collect"]["workers"] = min(4, os.cpu_count() or 1)
    return cfg
