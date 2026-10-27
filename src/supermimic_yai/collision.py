import hashlib
from pathlib import Path

import coacd
import numpy as np
import trimesh

# Concavity thresholds (fraction of part size), finest first. Fine cuts keep hull facets from
# closing narrow holes, but every piece adds contact constraints, so take the finest one that
# stays under MAX_PIECES.
THRESHOLDS = (0.012, 0.015, 0.02, 0.03)
MAX_PIECES = 64


def _pieces(mesh, mesh_path: Path, threshold: float, cache_dir: Path) -> list[Path]:
    h = hashlib.sha256(Path(mesh_path).read_bytes())
    h.update(repr(threshold).encode())
    out = Path(cache_dir) / h.hexdigest()
    done = out / "done"
    if not done.exists():
        coacd.set_log_level("error")
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, seed=0)
        out.mkdir(parents=True, exist_ok=True)
        for i, (v, f) in enumerate(parts):
            trimesh.Trimesh(np.asarray(v), np.asarray(f)).export(out / f"piece_{i:03d}.stl")
        done.touch()
    return sorted(out.glob("piece_*.stl"))


def decompose(mesh_path: Path, cache_dir: Path) -> list[Path]:
    """Convex-decompose a mesh with CoACD; pieces are cached as STLs."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_convex:  # CoACD splits even convex solids at low thresholds
        return [Path(mesh_path)]
    for threshold in THRESHOLDS:
        pieces = _pieces(mesh, mesh_path, threshold, cache_dir)
        if len(pieces) <= MAX_PIECES:
            break
    return pieces
