"""Loading, saving and normalising meshes.

Real-world STLs arrive in every state imaginable: scenes with a dozen parts,
inverted normals, duplicate vertices, positioned a metre off the origin. Stage 1
assumes none of that, so everything gets normalised on the way in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

__all__ = ["load", "save", "concat", "drop_to_bed", "set_lift", "mesh_from_bytes", "summary"]

SUPPORTED_SUFFIXES = {".stl", ".obj", ".ply", ".3mf", ".off", ".glb", ".gltf"}


def load(path: str | Path, repair: bool = True) -> trimesh.Trimesh:
    """Load a mesh file and return a single normalised Trimesh."""
    path = Path(path)
    obj = trimesh.load(path, force="mesh", process=True)
    return _normalise(obj, repair=repair)


def mesh_from_bytes(data: bytes, suffix: str, repair: bool = True) -> trimesh.Trimesh:
    """Load a mesh from an in-memory upload."""
    import io

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    if suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type {suffix!r}")
    obj = trimesh.load(io.BytesIO(data), file_type=suffix.lstrip("."), force="mesh", process=True)
    return _normalise(obj, repair=repair)


def _normalise(obj, repair: bool = True) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        obj = obj.to_geometry() if hasattr(obj, "to_geometry") else obj.dump(concatenate=True)
    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"could not read a triangle mesh (got {type(obj).__name__})")
    if obj.is_empty or len(obj.faces) == 0:
        raise ValueError("mesh has no faces")

    mesh = obj.copy()
    if repair:
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        # Normals pointing inward would invert every overhang test downstream.
        mesh.fix_normals()
    return mesh


def save(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def concat(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    """Join meshes without a boolean union.

    Slicers handle overlapping shells fine, and a real union on a mesh with a
    few hundred support pillars is slow and fragile. See CLAUDE.md.
    """
    parts = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not parts:
        return trimesh.Trimesh()
    if len(parts) == 1:
        return parts[0].copy()
    return trimesh.util.concatenate(parts)


def drop_to_bed(
    mesh: trimesh.Trimesh, center_xy: bool = True, lift: float = 0.0
) -> trimesh.Trimesh:
    """Translate so the lowest point sits at ``lift``, optionally centred in XY.

    The build plate is always z=0 everywhere downstream, so a non-zero `lift`
    leaves the model floating with clear air underneath and the scaffold spans
    the gap. See ``SupportParams.lift_height``.
    """
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    shift = np.array([0.0, 0.0, -lo[2] + max(0.0, float(lift))])
    if center_xy:
        shift[0] = -(lo[0] + hi[0]) * 0.5
        shift[1] = -(lo[1] + hi[1]) * 0.5
    mesh.apply_translation(shift)
    return mesh


def set_lift(mesh: trimesh.Trimesh, lift: float) -> trimesh.Trimesh:
    """Move an already-posed mesh so its lowest point sits at ``lift``.

    Only the Z shift, so a model that has been oriented and centred keeps both.
    Cheap enough to re-run every time the lift slider moves, which is the point:
    changing the lift must not re-run the orientation search.
    """
    mesh = mesh.copy()
    mesh.apply_translation([0.0, 0.0, max(0.0, float(lift)) - float(mesh.bounds[0][2])])
    return mesh


def summary(mesh: trimesh.Trimesh) -> dict:
    """Cheap facts about a mesh, for logging and the UI."""
    lo, hi = mesh.bounds
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "size": [float(v) for v in (hi - lo)],
        "watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }
