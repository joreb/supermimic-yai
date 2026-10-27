import json
import math
from dataclasses import dataclass
from pathlib import Path

import trimesh

Pose = tuple[float, float, float, float, float, float, float]


class ProductError(ValueError):
    pass


@dataclass(frozen=True)
class PartSpec:
    name: str
    mesh: Path
    mass: float
    start_pose: Pose
    final_pose: Pose
    fixed: bool


@dataclass(frozen=True)
class Product:
    name: str
    root: Path
    parts: dict[str, PartSpec]
    tol_pos: float
    tol_rot_deg: float


def _pose(part: str, key: str, raw) -> Pose:
    try:
        v = [float(x) for x in raw]
    except (TypeError, ValueError):
        raise ProductError(f"part {part!r}: {key} must be 7 numbers") from None
    if len(v) != 7:
        raise ProductError(f"part {part!r}: {key} must have 7 numbers, got {len(v)}")
    n = math.sqrt(sum(q * q for q in v[3:]))
    if n < 1e-9:
        raise ProductError(f"part {part!r}: {key} has zero-length quaternion")
    return (*v[:3], *(q / n for q in v[3:]))


def load_product(directory: Path) -> Product:
    root = Path(directory)
    try:
        data = json.loads((root / "assembly.json").read_text())
        tol = data["tolerance"]
        raw_parts = data["parts"]
        name = data["name"]
        tol_pos, tol_rot = float(tol["pos"]), float(tol["rot_deg"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ProductError(f"{root}: bad assembly.json ({e!r})") from None
    if not raw_parts:
        raise ProductError(f"{root}: no parts")

    parts = {}
    for pname, p in raw_parts.items():
        for key in ("mesh", "mass", "fixed", "start_pose", "final_pose"):
            if key not in p:
                raise ProductError(f"part {pname!r}: missing field {key!r}")
        mesh_path = root / p["mesh"]
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except Exception as e:
            raise ProductError(f"cannot load mesh {mesh_path.name}: {e}") from None
        if not mesh.is_watertight:
            raise ProductError(f"mesh {mesh_path.name} is not watertight")
        try:
            mass = float(p["mass"])
        except (TypeError, ValueError):
            raise ProductError(f"part {pname!r}: mass must be a number") from None
        if not isinstance(p["fixed"], bool):
            raise ProductError(f"part {pname!r}: fixed must be true or false")
        if not mass > 0:
            raise ProductError(f"part {pname!r}: mass must be positive")
        parts[pname] = PartSpec(
            pname, mesh_path, mass,
            _pose(pname, "start_pose", p["start_pose"]),
            _pose(pname, "final_pose", p["final_pose"]),
            p["fixed"],
        )

    fixed = [s for s in parts.values() if s.fixed]
    if len(fixed) != 1:
        raise ProductError(f"{root}: exactly one fixed part required, found {len(fixed)}")
    if fixed[0].start_pose != fixed[0].final_pose:
        raise ProductError(f"fixed part {fixed[0].name!r}: start_pose must equal final_pose")
    return Product(name, root, parts, tol_pos, tol_rot)
