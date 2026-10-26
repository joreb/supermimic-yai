FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 git ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mujoco==3.3.5 gymnasium trimesh manifold3d coacd \
    imageio imageio-ffmpeg pyyaml numpy pytest python-fcl scipy
RUN git clone https://github.com/google-deepmind/mujoco_menagerie /opt/mujoco_menagerie \
    && cd /opt/mujoco_menagerie \
    && git checkout f3b6013629c7b94f8ee93fd977b161fa8350f9bc \
    && find . -mindepth 1 -maxdepth 1 ! -name franka_emika_panda -exec rm -rf {} +
ENV MUJOCO_GL=osmesa MENAGERIE_DIR=/opt/mujoco_menagerie PYTHONPATH=/work/src
