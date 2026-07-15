"""
Parser for Atlantis box-model geometry (.bgm) files.

Replicates the geometry-reading behaviour of CSIRO's Olive Java tool
(cmr.apps.olive.BMParameters / BMDataAccessor / BMPolygonPatch3D), which reads
box polygons, box centres, box bottom depth and the model bounding polygon out
of a plain-text .bgm file via a template (bm.tpl) that maps these tags:

    nbox                box count
    box<i>.label         box label (string)
    box<i>.inside         point known to be inside the box (box centre)
    box<i>.vert            polygon vertex, repeated (closed ring)
    box<i>.botz            bottom depth (m, negative down)
    box<i>.area             box surface area (m2)
    box<i>.nconn/.iface/.ibox   box connectivity (face indices / neighbour box ids)
    nface, face<i>.p1/p2/length/cs/lr   face (edge) geometry and connectivity
    bnd_vert                model bounding polygon vertex, repeated
    projection               proj4-style projection string

Unlike the Java tool (which only needs a generic template engine because it
supports several box-model variants), this reader is written directly against
the current .bgm format used by Atlantis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import Polygon


_TAG_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.(\w+)$")


@dataclass
class Box:
    index: int
    label: str
    inside: tuple
    botz: float = float("nan")
    area: float = float("nan")
    nconn: int = 0
    iface: list = field(default_factory=list)
    ibox: list = field(default_factory=list)
    vertices: list = field(default_factory=list)  # list of (x, y), closed ring

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.vertices)


@dataclass
class Face:
    index: int
    p1: tuple
    p2: tuple
    length: float
    cs: tuple
    lr: tuple  # (left_box, right_box)


@dataclass
class BgmGeometry:
    boxes: list  # list[Box], ordered by index
    faces: list  # list[Face], ordered by index
    boundary_vertices: list  # list[(x, y)]
    proj4: str | None
    crs: "pyproj.CRS | None"
    maxwcbotz: float | None
    source_path: str

    def __post_init__(self):
        self._by_label = {b.label: b for b in self.boxes}

    @property
    def nbox(self) -> int:
        return len(self.boxes)

    @property
    def nface(self) -> int:
        return len(self.faces)

    def box(self, index: int) -> Box:
        return self.boxes[index]

    def box_by_label(self, label: str) -> Box:
        return self._by_label[label]

    @property
    def boundary_polygon(self) -> Polygon:
        return Polygon(self.boundary_vertices)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Box polygons as a GeoDataFrame indexed by box index, in the bgm's
        native projected CRS (metres)."""
        recs = []
        for b in self.boxes:
            recs.append(
                {
                    "box": b.index,
                    "label": b.label,
                    "botz": b.botz,
                    "area": b.area,
                    "nconn": b.nconn,
                    "geometry": b.polygon,
                }
            )
        gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=self.crs)
        gdf = gdf.set_index("box", drop=False)
        return gdf

    def boundary_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            [{"geometry": self.boundary_polygon}], geometry="geometry", crs=self.crs
        )


def _split_bgm_value(tokens: list) -> list:
    return [float(t) for t in tokens]


def parse_bgm(path: str | Path) -> BgmGeometry:
    path = Path(path)
    boxes: dict[int, Box] = {}
    faces: dict[int, Face] = {}
    boundary_vertices: list = []
    proj4 = None
    maxwcbotz = None

    def get_box(i: int) -> Box:
        if i not in boxes:
            boxes[i] = Box(index=i, label=f"Box{i}", inside=(float("nan"), float("nan")))
        return boxes[i]

    def get_face(i: int) -> Face:
        if i not in faces:
            faces[i] = Face(index=i, p1=None, p2=None, length=float("nan"), cs=None, lr=None)
        return faces[i]

    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            rest = parts[1:]

            if key.lower() == "projection":
                proj4 = " ".join(rest)
                continue
            if key.lower() == "bnd_vert":
                x, y = float(rest[0]), float(rest[1])
                boundary_vertices.append((x, y))
                continue
            if key.lower() == "nbox":
                continue  # box count is implied by parsed boxes
            if key.lower() == "nface":
                continue
            if key.lower() == "maxwcbotz":
                maxwcbotz = float(rest[0])
                continue

            m = _TAG_RE.match(key)
            if not m:
                continue
            prefix, attr = m.group(1), m.group(2).lower()

            box_m = re.match(r"^box(\d+)$", prefix, re.IGNORECASE)
            face_m = re.match(r"^face(\d+)$", prefix, re.IGNORECASE)

            if box_m:
                b = get_box(int(box_m.group(1)))
                if attr == "label":
                    b.label = rest[0]
                elif attr == "inside":
                    b.inside = (float(rest[0]), float(rest[1]))
                elif attr == "vert":
                    b.vertices.append((float(rest[0]), float(rest[1])))
                elif attr == "botz":
                    b.botz = float(rest[0])
                elif attr == "area":
                    b.area = float(rest[0])
                elif attr == "nconn":
                    b.nconn = int(float(rest[0]))
                elif attr == "iface":
                    b.iface = [int(float(t)) for t in rest]
                elif attr == "ibox":
                    b.ibox = [int(float(t)) for t in rest]
                # vertmix, horizmix, relax_tol: hydro-only, not needed for viewing
                continue

            if face_m:
                f = get_face(int(face_m.group(1)))
                if attr == "p1":
                    f.p1 = (float(rest[0]), float(rest[1]))
                elif attr == "p2":
                    f.p2 = (float(rest[0]), float(rest[1]))
                elif attr == "length":
                    f.length = float(rest[0])
                elif attr == "cs":
                    f.cs = (float(rest[0]), float(rest[1]))
                elif attr == "lr":
                    f.lr = (int(float(rest[0])), int(float(rest[1])))
                continue

    box_list = [boxes[i] for i in sorted(boxes.keys())]
    face_list = [faces[i] for i in sorted(faces.keys())]

    crs = None
    if proj4:
        try:
            proj4_str = " ".join(f"+{tok}" if not tok.startswith("+") else tok for tok in proj4.split())
            crs = pyproj.CRS.from_proj4(proj4_str)
        except Exception:
            crs = None

    return BgmGeometry(
        boxes=box_list,
        faces=face_list,
        boundary_vertices=boundary_vertices,
        proj4=proj4,
        crs=crs,
        maxwcbotz=maxwcbotz,
        source_path=str(path),
    )
