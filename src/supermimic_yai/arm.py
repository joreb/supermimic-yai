import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import yaml


class ArmError(ValueError):
    pass


@dataclass(frozen=True)
class ArmSpec:
    name: str
    mjcf: Path
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    gripper_actuator: str
    gripper_open_ctrl: float
    gripper_closed_ctrl: float
    ee_body: str
    ee_offset: tuple[float, float, float]
    max_grip_width: float
    home_qpos: tuple[float, ...]
    base_pose: tuple[float, ...]


def load_arm(arm_dir: Path) -> ArmSpec:
    arm_dir = Path(arm_dir)
    data = yaml.safe_load((arm_dir / "arm.yaml").read_text())
    mjcf = os.path.expandvars(data["mjcf"])
    if "$" in mjcf:
        raise ArmError(f"unexpanded variable in mjcf path: {mjcf}")
    mjcf_path = Path(mjcf)
    if not mjcf_path.is_file():
        raise ArmError(f"mjcf file not found: {mjcf_path}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    checks = [
        (mujoco.mjtObj.mjOBJ_JOINT, data["arm_joints"]),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, [*data["arm_actuators"], data["gripper_actuator"]]),
        (mujoco.mjtObj.mjOBJ_BODY, [data["ee_body"]]),
    ]
    for kind, names in checks:
        for name in names:
            if mujoco.mj_name2id(model, kind, name) < 0:
                raise ArmError(f"{name!r} not found in {mjcf_path}")

    return ArmSpec(
        name=arm_dir.name,
        mjcf=mjcf_path,
        arm_joints=tuple(data["arm_joints"]),
        arm_actuators=tuple(data["arm_actuators"]),
        gripper_actuator=data["gripper_actuator"],
        gripper_open_ctrl=float(data["gripper_open_ctrl"]),
        gripper_closed_ctrl=float(data["gripper_closed_ctrl"]),
        ee_body=data["ee_body"],
        ee_offset=tuple(float(v) for v in data["ee_offset"]),
        max_grip_width=float(data["max_grip_width"]),
        home_qpos=tuple(float(v) for v in data["home_qpos"]),
        base_pose=tuple(float(v) for v in data["base_pose"]),
    )
