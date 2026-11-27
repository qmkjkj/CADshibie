"""Build complete, traceable templates from selected training SVGs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from .svg import iter_primitives
from .templating import descriptors


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _record(primitive):
    points = [[_finite(x, "x"), _finite(y, "y")] for x, y in primitive.points]
    command = int(primitive.command)
    rgb = [int(value) for value in primitive.rgb]
    width, length = _finite(primitive.width, "width"), _finite(primitive.length, "length")
    if len(points) != 4 or command not in (0, 1, 2, 3):
        raise ValueError("Each primitive must have a valid command and four points")
    if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
        raise ValueError("Invalid primitive RGB")
    if width < 0 or length < 0:
        raise ValueError("Primitive width/length cannot be negative")
    return {"command": command, "points": points, "rgb": rgb,
            "width": width, "length": length, "layer_id": int(primitive.layer_id)}


def _metadata(root):
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("XML root must be svg")
    raw, viewbox = root.get("viewBox"), None
    if raw is not None:
        viewbox = [_finite(value, "viewBox") for value in re.split(r"[\s,]+", raw.strip())]
        if len(viewbox) != 4 or min(viewbox[2:]) <= 0:
            raise ValueError("Invalid SVG viewBox")
    return {"canvas": {"viewBox": raw, "width": root.get("width"), "height": root.get("height")},
            "viewbox": viewbox, "coordinate_space": "svg_native"}


def build_library(svg_dir: Path, *, allowed_stems: set[str] | None = None,
                  limit: int = 0) -> dict:
    """Fail on invalid selected sources; never include an unselected SVG.

    Only exactly equal complete primitive multisets with equal canvases merge.
    Descriptors select candidates; collisions cannot merge different geometry.
    Translated/scaled copies remain separate until equivalence is validated.
    """
    svg_dir = Path(svg_dir)
    if not svg_dir.is_dir():
        raise ValueError(f"SVG directory does not exist: {svg_dir}")
    paths = sorted((p for p in svg_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".svg"),
                   key=lambda p: p.relative_to(svg_dir).as_posix())
    if not paths:
        raise ValueError(f"No SVG files in {svg_dir}")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if allowed_stems is not None:
        if not allowed_stems or any(not isinstance(s, str) or not s.strip() for s in allowed_stems):
            raise ValueError("allowed_stems must contain nonempty source stems")
        available = {}
        for path in paths:
            available.setdefault(path.stem, []).append(path)
        missing = sorted(allowed_stems - available.keys())
        ambiguous = sorted(s for s in allowed_stems if len(available.get(s, [])) > 1)
        if missing or ambiguous:
            raise ValueError(f"Invalid split: missing SVG stems={missing}, ambiguous stems={ambiguous}")
        paths = [p for p in paths if p.stem in allowed_stems]
    if limit:
        paths = paths[:limit]
    templates, seen, source_files, empty_files, errors = [], {}, [], [], []
    instance_count = 0
    for path in paths:
        relative = path.relative_to(svg_dir).as_posix()
        try:
            raw = path.read_bytes()
            metadata = _metadata(ET.fromstring(raw))
            source_file = {"svg": relative, "sha256": hashlib.sha256(raw).hexdigest()}
            source_files.append(source_file)
            groups = {}
            for primitive in iter_primitives(path):
                semantic, instance = int(primitive.semantic_id), int(primitive.instance_id)
                if 0 <= semantic < 30 and instance >= 0:
                    groups.setdefault((semantic, instance), []).append(_record(primitive))
            if not groups:
                empty_files.append(relative)
            for (semantic, instance), items in sorted(groups.items()):
                canonical = {"class": semantic, "canvas": metadata,
                             "primitives": sorted(json.dumps(p, sort_keys=True, allow_nan=False) for p in items)}
                key = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
                source = {**source_file, "image_stem": path.stem,
                          "semantic_id": semantic, "instance_id": instance, **metadata}
                if key in seen:
                    template = templates[seen[key]]
                    template["count"] += 1
                    template["sources"].append(source)
                else:
                    points = np.asarray([p["points"] for p in items], dtype=np.float64).reshape(-1, 2)
                    shape = descriptors(points)
                    if shape.shape != (4, 256) or not np.isfinite(shape).all():
                        raise ValueError("Invalid template descriptors")
                    seen[key] = len(templates)
                    templates.append({"class": semantic, "count": 1, "primitives": items,
                                      "descriptors": shape.astype(np.float32), "sources": [source],
                                      "geometry_sha256": key})
                instance_count += 1
        except Exception as exc:
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError("Template build failed; no library written:\n" + "\n".join(errors))
    if not templates:
        raise ValueError("Selected SVGs have no eligible Thing instances")
    return {"schema_version": 2, "grid": 16, "rotations": 4, "templates": templates,
            "build": {"source_files": source_files, "source_instances": instance_count,
                      "template_count": len(templates), "files_without_instances": empty_files,
                      "allowed_stems": sorted(allowed_stems) if allowed_stems is not None else None,
                      "deduplication": "exact-raw-primitive-multiset-and-canvas-v1"}}
