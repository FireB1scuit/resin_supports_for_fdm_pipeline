"""Generate a synthetic 'miniature' to exercise the whole pipeline.

A real pre-supported mini is someone's copyrighted sculpt, so the repo ships a
generator instead of a model. This one is deliberately awkward in the ways that
break support generators:

  * a wide thin base disc          -> the orientation scorer should find it
  * an outstretched arm            -> a genuine mid-air island
  * a cape flaring outward         -> a long shallow overhang
  * a raised sword tip             -> a tall isolated pillar that needs bracing
  * a head off to one side         -> a ball in free air, over the rim of the base
  * bumpy detail on the front only -> supports should end up on the back

    python scripts/make_sample.py samples/synthetic_mini.stl
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _bumpy(mesh: trimesh.Trimesh, direction, amplitude=0.35, freq=2.2) -> trimesh.Trimesh:
    """Displace vertices facing `direction`, so one side reads as sculpted detail."""
    mesh = mesh.copy()
    n = mesh.vertex_normals
    facing = np.clip(n @ np.asarray(direction, dtype=float), 0, 1) ** 2
    v = mesh.vertices
    wave = np.sin(v[:, 2] * freq) * np.cos(v[:, 0] * freq * 1.7) * np.sin(v[:, 1] * freq * 0.9)
    mesh.vertices = v + n * (wave * amplitude * facing)[:, None]
    return mesh


def build() -> trimesh.Trimesh:
    parts = []

    # Base disc: 25 mm across, 2 mm thick.
    base = trimesh.creation.cylinder(radius=12.5, height=2.0, sections=64)
    base.apply_translation([0, 0, 1.0])
    parts.append(base)

    # Torso and legs, detailed on the +Y side (the "front").
    body = trimesh.creation.capsule(radius=5.0, height=18.0, count=[24, 24])
    body.apply_translation([0, 0, 8.0])
    parts.append(_bumpy(body, [0, 1, 0], amplitude=0.5))

    # Head, carried out to the front-left rather than stacked on the shoulders.
    # Sitting on the axis it had the torso under it and the cape's mouth under
    # that, so most of its lower hemisphere was in the shadow of something else
    # and a shaft reaching it only had to clear the model it was already beside.
    # Out here nothing is beneath it but the base disc, so it is a ball in free
    # air: contacts wrap the whole underside and the shafts holding them have to
    # lean out past the rim of the disc to find the plate.
    head = trimesh.creation.icosphere(subdivisions=3, radius=3.6)
    head.apply_translation([-10.0, 8.0, 31.0])
    parts.append(_bumpy(head, [0, 1, 0], amplitude=0.45, freq=4.0))

    # Outstretched arm: nothing beneath its far end, so its tip is an island.
    arm = trimesh.creation.cylinder(radius=1.6, height=13.0, sections=16)
    arm.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    arm.apply_translation([8.0, 0, 25.0])
    parts.append(arm)

    # Sword held up off the far end of the arm: tall, thin, isolated.
    blade = trimesh.creation.box(extents=[1.2, 3.0, 20.0])
    blade.apply_translation([14.0, 0, 34.0])
    parts.append(blade)

    # Cape: a cone shell flaring out behind, a long shallow overhang.
    cape = trimesh.creation.cone(radius=9.0, height=17.0, sections=32)
    cape.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    cape.apply_translation([0, -3.0, 24.0])
    parts.append(cape)

    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/synthetic_mini.stl")
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh = build()
    mesh.export(out)
    lo, hi = mesh.bounds
    print(f"wrote {out}")
    print(f"  {len(mesh.faces):,} faces, {(hi - lo)[0]:.1f} x {(hi - lo)[1]:.1f} x {(hi - lo)[2]:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
