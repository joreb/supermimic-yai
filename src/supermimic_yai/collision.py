import hashlib
from pathlib import Path

import coacd
import numpy as np
import trimesh

THRESHOLD = 0.02


def decompose(mesh_path: Path, cache_dir: Path) -> list[Path]:
    """Convex-decompose a mesh with CoACD; pieces are cached as STLs."""
    h = hashlib.sha256(Path(mesh_path).read_bytes())
    h.update(repr(THRESHOLD).encode())
    out = Path(cache_dir) / h.hexdigest()
    done = out / "done"
    if not done.exists():
        mesh = trimesh.load(mesh_path, force="mesh")
        coacd.set_log_level("error")
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=THRESHOLD, seed=0)
        out.mkdir(parents=True, exist_ok=True)
        for i, (v, f) in enumerate(parts):
            trimesh.Trimesh(np.asarray(v), np.asarray(f)).export(out / f"piece_{i:03d}.stl")
        done.touch()
    return sorted(out.glob("piece_*.stl"))
