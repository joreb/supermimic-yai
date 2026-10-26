from dataclasses import dataclass

import numpy as np

from supermimic_yai.product import Product

CONTEXT_RANGES = {
    "friction": (0.5, 1.2),
    "mass_scale": (0.8, 1.2),
    "start_pose_noise": ((-0.01, 0.01), (-0.01, 0.01), (-0.2, 0.2)),  # dx, dy [m], dyaw [rad]
    "joint_offsets": (-0.01, 0.01),
}


@dataclass(frozen=True)
class SimContext:
    friction: float
    mass_scale: dict[str, float]
    start_pose_noise: dict[str, tuple[float, float, float]]
    joint_offsets: tuple[float, ...]

    def to_vector(self, part_order: list[str]) -> np.ndarray:
        v = [self.friction]
        v += [self.mass_scale[p] for p in part_order]
        for p in part_order:
            v += self.start_pose_noise[p]
        return np.array([*v, *self.joint_offsets], dtype=np.float64)


def sample_context(rng: np.random.Generator, product: Product, n_joints: int) -> SimContext:
    r = CONTEXT_RANGES
    free = [n for n, p in product.parts.items() if not p.fixed]
    return SimContext(
        friction=float(rng.uniform(*r["friction"])),
        mass_scale={n: float(rng.uniform(*r["mass_scale"])) for n in product.parts},
        start_pose_noise={
            n: tuple(float(rng.uniform(*b)) for b in r["start_pose_noise"]) if n in free else (0.0, 0.0, 0.0)
            for n in product.parts
        },
        joint_offsets=tuple(float(x) for x in rng.uniform(*r["joint_offsets"], n_joints)),
    )


def nominal_context(product: Product, n_joints: int) -> SimContext:
    return SimContext(
        1.0,
        {n: 1.0 for n in product.parts},
        {n: (0.0, 0.0, 0.0) for n in product.parts},
        (0.0,) * n_joints,
    )
