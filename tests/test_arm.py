from pathlib import Path

import mujoco
import pytest

from supermimic_yai.arm import ArmError, load_arm

ROOT = Path(__file__).resolve().parent.parent
PANDA_DIR = ROOT / "arms" / "panda"


def test_load_panda():
    arm = load_arm(PANDA_DIR)
    assert len(arm.arm_joints) == 7
    assert arm.gripper_actuator == "actuator8"
    assert arm.max_grip_width == 0.08


def test_names_exist_in_model():
    arm = load_arm(PANDA_DIR)
    model = mujoco.MjModel.from_xml_path(str(arm.mjcf))
    for name in arm.arm_joints:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0, name
    for name in (*arm.arm_actuators, arm.gripper_actuator):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0, name
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body) >= 0


def test_bad_joint_name_raises(tmp_path):
    text = (PANDA_DIR / "arm.yaml").read_text()
    assert "joint1" in text
    (tmp_path / "arm.yaml").write_text(text.replace("joint1", "nope", 1))
    with pytest.raises(ArmError, match="nope"):
        load_arm(tmp_path)


def _edit(tmp_path, old, new):
    text = (PANDA_DIR / "arm.yaml").read_text()
    assert old in text
    (tmp_path / "arm.yaml").write_text(text.replace(old, new))
    return tmp_path


@pytest.mark.parametrize("old,new,match", [
    ("home_qpos: [0, 0, 0,", "home_qpos: [0, 0,", "home_qpos"),
    ("ee_offset: [0, 0, 0.1034]", "ee_offset: [0, 0]", "ee_offset"),
    ("base_pose: [0, 0, 0, 1, 0, 0, 0]", "base_pose: [0, 0, 1, 1, 0, 0, 0]", "base_pose not supported"),
    ("ee_offset: [0, 0, 0.1034]", "ee_offset: 5", "bad arm.yaml"),
    ("gripper_actuator: actuator8\n", "", "bad arm.yaml"),
    ("ee_offset: [0, 0, 0.1034]", "ee_offset: [0, 0, 0.1034", "bad arm.yaml"),
])
def test_bad_arm_yaml_raises_arm_error(tmp_path, old, new, match):
    with pytest.raises(ArmError, match=match):
        load_arm(_edit(tmp_path, old, new))
