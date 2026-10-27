from dataclasses import dataclass

import numpy as np

from supermimic_yai.product import Pose, Product


@dataclass
class PartScore:
    part: str
    pos_err: float
    rot_err_deg: float
    ok: bool


@dataclass
class ScoreReport:
    parts: list[PartScore]
    success: bool


def score(product: Product, part_poses: dict[str, Pose]) -> ScoreReport:
    """Compare each non-fixed part to its final pose. Rotational symmetry is ignored in v1."""
    parts = []
    for name, spec in product.parts.items():
        if spec.fixed:
            continue
        p = np.asarray(part_poses[name], dtype=np.float64)
        f = np.asarray(spec.final_pose, dtype=np.float64)
        pos = float(np.linalg.norm(p[:3] - f[:3]))
        dot = min(1.0, abs(float(f[3:] @ p[3:]) / (np.linalg.norm(f[3:]) * np.linalg.norm(p[3:]))))
        rot = float(np.degrees(2 * np.arccos(dot)))
        parts.append(PartScore(name, pos, rot, pos <= product.tol_pos and rot <= product.tol_rot_deg))
    return ScoreReport(parts, all(s.ok for s in parts))
