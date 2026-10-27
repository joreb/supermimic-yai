"""Action tokens and recipe validation.

A recipe is JSON: {"product": "<name>", "steps": [{"token": "GRASP", ...}, ...]}.
Example step: {"token": "INSERT", "part": "p", "target": "final:p", "depth": 0.01,
"max_force": 20}. Poses are [x, y, z, qw, qx, qy, qz] in metres.
"""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, Union

import numpy as np
import trimesh
from trimesh.transformations import quaternion_matrix

from supermimic_yai.arm import ArmSpec
from supermimic_yai.product import Pose, PartSpec, Product


class RecipeError(ValueError):
    pass


@dataclass(frozen=True)
class Grasp:
    kind: ClassVar[str] = "GRASP"
    part: str
    grasp_pose: Pose


@dataclass(frozen=True)
class MoveTo:
    kind: ClassVar[str] = "MOVE_TO"
    pose: Pose


@dataclass(frozen=True)
class MoveAbove:
    kind: ClassVar[str] = "MOVE_ABOVE"
    target: str
    offset: float


@dataclass(frozen=True)
class Align:
    kind: ClassVar[str] = "ALIGN"
    part: str
    target: str


@dataclass(frozen=True)
class Insert:
    kind: ClassVar[str] = "INSERT"
    part: str
    target: str
    depth: float
    max_force: float


@dataclass(frozen=True)
class Place:
    kind: ClassVar[str] = "PLACE"
    part: str
    target: str


@dataclass(frozen=True)
class Press:
    kind: ClassVar[str] = "PRESS"
    part: str
    direction: tuple[float, float, float]
    max_force: float


@dataclass(frozen=True)
class Release:
    kind: ClassVar[str] = "RELEASE"


@dataclass(frozen=True)
class Retreat:
    kind: ClassVar[str] = "RETREAT"
    distance: float


Token = Union[Grasp, MoveTo, MoveAbove, Align, Insert, Place, Press, Release, Retreat]

FREE_SPACE = {MoveTo, MoveAbove, Release, Retreat}
CONTACT = {Grasp, Align, Insert, Place, Press}
_BY_KIND = {cls.kind: cls for cls in (Grasp, MoveTo, MoveAbove, Align, Insert, Place, Press, Release, Retreat)}


@dataclass(frozen=True)
class Recipe:
    product: str
    steps: tuple[Token, ...]


def resolve_target(target: str, product: Product) -> Pose:
    part = target[len("final:"):] if isinstance(target, str) and target.startswith("final:") else None
    if part not in product.parts:
        raise RecipeError(f"unknown target {target!r}")
    return product.parts[part].final_pose


def _number(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RecipeError(f"{name} must be a number, got {v!r}")
    if not math.isfinite(v):
        raise RecipeError(f"{name} must be finite, got {v!r}")
    return float(v)


def _positive(v, name: str) -> float:
    x = _number(v, name)
    if x <= 0:
        raise RecipeError(f"{name} must be > 0, got {x:g}")
    return x


def _vector(v, n: int, name: str) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise RecipeError(f"{name} must be {n} numbers")
    return [_number(x, name) for x in v]


def _pose(v, name: str) -> Pose:
    x = _vector(v, 7, name)
    n = math.sqrt(sum(q * q for q in x[3:]))
    if n < 1e-9:
        raise RecipeError(f"{name} has zero-length quaternion")
    return (*x[:3], *(q / n for q in x[3:]))


def _direction(v, name: str) -> tuple[float, float, float]:
    x = _vector(v, 3, name)
    n = math.sqrt(sum(c * c for c in x))
    if n < 1e-9:
        raise RecipeError(f"{name} must be non-zero")
    return (x[0] / n, x[1] / n, x[2] / n)


def _part(v, product: Product) -> str:
    if not isinstance(v, str) or v not in product.parts:
        raise RecipeError(f"unknown part {v!r}")
    return v


def _target(v, product: Product) -> str:
    resolve_target(v, product)  # raises on unknown target
    return v


def _convert(name: str, v, product: Product):
    if name == "part":
        return _part(v, product)
    if name == "target":
        return _target(v, product)
    if name in ("pose", "grasp_pose"):
        return _pose(v, name)
    if name == "direction":
        return _direction(v, name)
    if name in ("depth", "max_force", "distance"):
        return _positive(v, name)
    return _number(v, name)  # offset


def _check_grasp_width(spec: PartSpec, grasp_pose: Pose, arm: ArmSpec) -> None:
    # Approach axis = gripper +z in the part frame; caliper width is the
    # minimum extent of the mesh over in-plane directions.
    axis = quaternion_matrix(grasp_pose[3:])[:3, 2]
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    dirs = np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
    verts = trimesh.load(spec.mesh, force="mesh").vertices
    proj = verts @ dirs.T
    width = float((proj.max(axis=0) - proj.min(axis=0)).min())
    if width > arm.max_grip_width:
        raise RecipeError(
            f"part {spec.name!r} is too wide to grasp: {width * 1000:g} mm "
            f"> gripper max {arm.max_grip_width * 1000:g} mm"
        )


def _parse_step(raw, product: Product, arm: ArmSpec) -> Token:
    if not isinstance(raw, dict) or "token" not in raw:
        raise RecipeError("missing 'token'")
    cls = _BY_KIND.get(raw["token"])
    if cls is None:
        raise RecipeError(f"unknown token {raw['token']!r}")
    names = [f.name for f in fields(cls)]
    for n in names:
        if n not in raw:
            raise RecipeError(f"{cls.kind} missing field {n!r}")
    extra = set(raw) - set(names) - {"token"}
    if extra:
        raise RecipeError(f"{cls.kind} unexpected field {sorted(extra)[0]!r}")
    tok = cls(**{n: _convert(n, raw[n], product) for n in names})
    if isinstance(tok, Grasp):
        _check_grasp_width(product.parts[tok.part], tok.grasp_pose, arm)
    return tok


def _track(tok: Token, held: str | None, product: Product) -> str | None:
    """Sequencing rules; returns the part held after this step."""
    if isinstance(tok, Grasp):
        if product.parts[tok.part].fixed:
            raise RecipeError(f"cannot grasp fixed part {tok.part!r}")
        if held is not None:
            raise RecipeError(f"GRASP {tok.part!r} while {held!r} is still held; RELEASE first")
        return tok.part
    if isinstance(tok, Release):
        return None
    if isinstance(tok, (Align, Insert, Place)) and tok.part != held:
        raise RecipeError(f"{tok.kind} must name the held part, which is {held!r}, not {tok.part!r}")
    if isinstance(tok, Press) and product.parts[tok.part].fixed:
        raise RecipeError(f"cannot press fixed part {tok.part!r}")
    return held


def parse_recipe(data: dict, product: Product, arm: ArmSpec) -> Recipe:
    if not isinstance(data, dict) or data.get("product") != product.name:
        raise RecipeError(f"recipe must be for product {product.name!r}")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        raise RecipeError("steps must be a list")
    steps: list[Token] = []
    held = None
    for i, raw in enumerate(raw_steps):
        try:
            tok = _parse_step(raw, product, arm)
            held = _track(tok, held, product)
        except RecipeError as e:
            raise RecipeError(f"step {i}: {e}") from None
        steps.append(tok)
    return Recipe(product.name, tuple(steps))


def _where(target: str, part: str) -> str:
    tpart = target.split(":", 1)[1]
    return "its final position" if tpart == part else f"the final position of {tpart}"


def _mm(v: float) -> str:
    return f"{v * 1000:g} mm"


def _describe_step(tok: Token) -> str:
    if isinstance(tok, Grasp):
        return f"Grasp {tok.part}"
    if isinstance(tok, MoveTo):
        x, y, z = tok.pose[:3]
        return f"Move gripper to ({_mm(x)}, {_mm(y)}, {_mm(z)})"
    if isinstance(tok, MoveAbove):
        return f"Move {_mm(tok.offset)} above {tok.target.split(':', 1)[1]}"
    if isinstance(tok, Align):
        return f"Align {tok.part} with {_where(tok.target, tok.part)}"
    if isinstance(tok, Insert):
        return (f"Insert {tok.part} into {_where(tok.target, tok.part)}, "
                f"{_mm(tok.depth)} deep, stop above {tok.max_force:g} N")
    if isinstance(tok, Place):
        return f"Place {tok.part} at {_where(tok.target, tok.part)}"
    if isinstance(tok, Press):
        dx, dy, dz = tok.direction
        return (f"Press {tok.part} along ({dx:.2f}, {dy:.2f}, {dz:.2f}), "
                f"stop above {tok.max_force:g} N")
    if isinstance(tok, Release):
        return "Release gripper"
    return f"Retreat {_mm(tok.distance)}"


def describe(recipe: Recipe) -> list[str]:
    return [f"{i}. {_describe_step(tok)}" for i, tok in enumerate(recipe.steps, 1)]


def load_recipe(path: Path, product: Product, arm: ArmSpec) -> Recipe:
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise RecipeError(f"{path}: invalid JSON ({e})") from None
    return parse_recipe(data, product, arm)
