import os
from pathlib import Path

import mujoco

BOX = """
<mujoco><worldbody>
<light pos="0 0 3"/><camera name="c" pos="0 -1 0.5" xyaxes="1 0 0 0 0.3 1"/>
<geom type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
<body pos="0 0 0.5"><freejoint/><geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/></body>
</worldbody></mujoco>
"""


def test_mujoco_steps():
    model = mujoco.MjModel.from_xml_string(BOX)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert data.time > 0


def test_offscreen_render():
    model = mujoco.MjModel.from_xml_string(BOX)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, 64, 64) as r:
        r.update_scene(data, camera="c")
        frame = r.render()
    assert frame.shape == (64, 64, 3)
    assert frame.std() > 0


def test_menagerie_present():
    assert Path(os.environ["MENAGERIE_DIR"], "franka_emika_panda/panda.xml").exists()
