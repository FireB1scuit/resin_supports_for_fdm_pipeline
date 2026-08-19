"""Getting the result out in a form a slicer can use.

Three modes, in increasing order of usefulness:

    combined   one STL, model and supports welded together. Slice with
               supports switched off. Works everywhere, but the whole plate
               prints at one layer height.
    separate   two STLs. You place them yourself; the slicer treats them as
               two objects, so each can have its own settings.
    3mf        one file, two objects, with per-object settings already set:
               the miniature at 0.08 mm and the supports at 0.16 mm with two
               walls. This is what the Resin2FDM docs recommend and it is the
               reason the 3MF path exists at all — coarse supports print in a
               fraction of the time and snap off no worse.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import trimesh

from .mesh_io import concat, save
from .types import SupportParams

__all__ = ["export_combined", "export_separate", "export_3mf", "export"]

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def export(
    model: trimesh.Trimesh,
    supports: trimesh.Trimesh,
    path: str | Path,
    params: SupportParams,
    mode: str = "auto",
) -> list[Path]:
    """Dispatch on `mode`, or on the file extension when mode is "auto"."""
    path = Path(path)
    if mode == "auto":
        mode = "3mf" if path.suffix.lower() == ".3mf" else "combined"

    if mode == "combined":
        return [export_combined(model, supports, path)]
    if mode == "separate":
        return export_separate(model, supports, path)
    if mode == "3mf":
        return [export_3mf(model, supports, path, params)]
    raise ValueError(f"unknown export mode {mode!r}")


def export_combined(
    model: trimesh.Trimesh, supports: trimesh.Trimesh, path: str | Path
) -> Path:
    return save(concat(model, supports), path)


def export_separate(
    model: trimesh.Trimesh, supports: trimesh.Trimesh, path: str | Path
) -> list[Path]:
    """Write `<stem>_model.stl` and `<stem>_supports.stl`."""
    path = Path(path)
    suffix = path.suffix or ".stl"
    stem = path.with_suffix("")
    out = [save(model, stem.with_name(stem.name + "_model").with_suffix(suffix))]
    if supports is not None and len(supports.faces):
        out.append(save(supports, stem.with_name(stem.name + "_supports").with_suffix(suffix)))
    return out


def export_3mf(
    model: trimesh.Trimesh,
    supports: trimesh.Trimesh,
    path: str | Path,
    params: SupportParams,
) -> Path:
    """Write a two-object 3MF with per-object slicer settings.

    The core 3MF part is standard and loads in any slicer. The per-object
    settings ride along in ``Metadata/Slic3r_PE_model.config``, which is the
    PrusaSlicer/OrcaSlicer/Bambu Studio convention. A slicer that does not
    know that file still opens the model correctly — it just shows two plain
    objects and you set the support layer height by hand.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    objects: list[tuple[int, str, trimesh.Trimesh, dict[str, str]]] = [
        (1, "miniature", model, {"layer_height": f"{params.layer_height:g}"}),
    ]
    if supports is not None and len(supports.faces):
        objects.append(
            (
                2,
                "supports",
                supports,
                {
                    # Supports carry no detail, so print them coarse and fast.
                    "layer_height": f"{params.support_layer_height:g}",
                    "perimeters": "2",
                    "fill_density": "0%",
                    # Support the supports? No. They are self-supporting by design.
                    "support_material": "0",
                },
            )
        )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("3D/3dmodel.model", _model_xml(objects))
        z.writestr("Metadata/Slic3r_PE_model.config", _config_xml(objects))
    return path


# --------------------------------------------------------------------- XML

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="config" ContentType="application/vnd.ms-printing.printticket+xml"/>
</Types>
"""

_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rel0" Target="/3D/3dmodel.model" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _model_xml(objects) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{_CORE_NS}">',
        " <resources>",
    ]
    for oid, name, mesh, _ in objects:
        parts.append(f'  <object id="{oid}" type="model" name="{escape(name)}">')
        parts.append("   <mesh>")
        parts.append("    <vertices>")
        # %g keeps the file small; 3MF is millimetres and we do not need more
        # than micron precision.
        for x, y, zz in np.asarray(mesh.vertices, dtype=np.float64):
            parts.append(f'     <vertex x="{x:.5g}" y="{y:.5g}" z="{zz:.5g}"/>')
        parts.append("    </vertices>")
        parts.append("    <triangles>")
        for a, b, c in np.asarray(mesh.faces, dtype=np.int64):
            parts.append(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>')
        parts.append("    </triangles>")
        parts.append("   </mesh>")
        parts.append("  </object>")
    parts.append(" </resources>")
    parts.append(" <build>")
    for oid, _, _, _ in objects:
        parts.append(f'  <item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>')
    parts.append(" </build>")
    parts.append("</model>")
    return "\n".join(parts)


def _config_xml(objects) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    for oid, name, mesh, settings in objects:
        parts.append(f'  <object id="{oid}">')
        parts.append(f'    <metadata type="object" key="name" value="{escape(name)}"/>')
        for key, value in settings.items():
            parts.append(
                f'    <metadata type="object" key="{escape(key)}" value="{escape(value)}"/>'
            )
        parts.append(f'    <volume firstid="0" lastid="{len(mesh.faces) - 1}">')
        parts.append(f'      <metadata type="volume" key="name" value="{escape(name)}"/>')
        parts.append('      <metadata type="volume" key="volume_type" value="ModelPart"/>')
        parts.append("    </volume>")
        parts.append("  </object>")
    parts.append("</config>")
    return "\n".join(parts)
