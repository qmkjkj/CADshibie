#!/usr/bin/env python3
"""Validate PNG/JSON coverage and describe predictions without estimating PQ."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from cadvec.raster import render_data  # noqa: E402
from cadvec.schema import validate_output  # noqa: E402
from infer import find_pngs  # noqa: E402


def index_paths(paths):
    groups = defaultdict(list)
    for path in paths:
        groups[path.stem].append(path)
    return ({name: paths[0] for name, paths in groups.items() if len(paths) == 1},
            {name: [str(p) for p in paths] for name, paths in groups.items() if len(paths) > 1})


def describe(data, image_size):
    validate_output(data)
    width, height = float(data["width"]), float(data["height"])
    points = np.asarray(data["args"], dtype=float).reshape(-1, 4, 2)
    outside = ((points[..., 0] < 0) | (points[..., 0] > width)
               | (points[..., 1] < 0) | (points[..., 1] > height))
    things = {(int(c), int(i)) for c, i in zip(data["semanticIds"], data["instanceIds"])
              if 0 <= c < 30 and i >= 0}
    stuff = {int(c) for c in data["semanticIds"] if 30 <= c < 35}
    return {"empty": not data["commands"], "primitive_count": len(data["commands"]),
            "thing_instances": len(things), "stuff_classes": len(stuff),
            "per_class_primitives": dict(sorted(Counter(data["semanticIds"]).items())),
            "per_class_instances": dict(sorted(Counter(c for c, _ in things).items())),
            "stuff_class_ids": sorted(stuff), "canvas": [width, height],
            "png_size": list(image_size), "outside_point_count": int(outside.sum()),
            "outside_primitive_count": int(outside.any(axis=1).sum()),
            "aspect_ratio_relative_difference": abs((width / height) / (image_size[0] / image_size[1]) - 1)}


def select_samples(records):
    valid = [r for r in records if r.get("valid")]
    empty = sorted((r for r in valid if r["empty"]), key=lambda r: r["name"])
    nonempty = sorted((r for r in valid if not r["empty"]), key=lambda r: (r["primitive_count"], r["name"]))
    chosen, seen = [], set()
    dense = sorted(nonempty, key=lambda r: (-r["primitive_count"], r["name"]))[:2]
    for role, rows in [("empty", empty[:2]), ("sparse", nonempty[:2]), ("dense", dense)]:
        for row in rows:
            if row["name"] not in seen:
                chosen.append((role, row))
                seen.add(row["name"])
    return chosen


def preview(png, prediction, output, role):
    data = json.loads(prediction.read_text(encoding="utf-8"))
    with Image.open(png) as source:
        rgba = source.convert("RGBA")
    original = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    if original.size != (1000, 1000):
        raise ValueError("This preview requires native 1000x1000 PNGs to preserve the scoring raster")
    rendered = render_data(data, raster=1000)
    mask = np.zeros((1000, 1000), dtype=bool)
    for (semantic, instance), value in rendered.items():
        if semantic < 35 and (semantic >= 30 or instance >= 0):
            mask |= value
    mask_image = Image.fromarray(mask)
    prediction_image = Image.new("RGB", original.size, "white")
    prediction_image.paste((0, 0, 0), mask=mask_image)
    overlay = Image.blend(original, Image.new("RGB", original.size, "white"), 0.55)
    overlay.paste((220, 30, 30), mask=mask_image)
    sheet = Image.new("RGB", (3000, 1040), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, panel) in enumerate(zip(["Input", "Prediction (1 px)", "Overlay (red=prediction; no GT)"],
                                               [original, prediction_image, overlay])):
        sheet.paste(panel, (1000 * index, 40))
        draw.text((1000 * index + 10, 8), f"{png.stem} | {role} | {title}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def audit(input_path, prediction_dir, *, preview_dir=None):
    pngs, png_duplicates = index_paths(find_pngs(Path(input_path)))
    predictions, prediction_duplicates = index_paths(sorted(Path(prediction_dir).rglob("*.json")))
    missing = sorted(pngs.keys() - predictions.keys())
    extra = sorted(predictions.keys() - pngs.keys())
    records, failures = [], []
    for name, path in sorted(predictions.items()):
        row = {"name": name, "prediction": str(path), "valid": False}
        try:
            raw = path.read_bytes()
            row["sha256"] = hashlib.sha256(raw).hexdigest()
            data = json.loads(raw)
            validate_output(data)
            if name in pngs:
                with Image.open(pngs[name]) as image:
                    row.update(describe(data, image.size))
                row["png"] = str(pngs[name])
                row["valid"] = True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures.append({"name": name, "error": row["error"]})
        records.append(row)
    valid = [r for r in records if r["valid"]]
    classes, instances = Counter(), Counter()
    for row in valid:
        classes.update(row["per_class_primitives"])
        instances.update(row["per_class_instances"])
    previews = []
    if preview_dir:
        for role, row in select_samples(records):
            destination = Path(preview_dir) / f"{role}_{row['name']}.png"
            try:
                preview(pngs[row["name"]], predictions[row["name"]], destination, role)
                previews.append({"name": row["name"], "role": role, "file": str(destination)})
            except (OSError, ValueError) as exc:
                failures.append({"name": row["name"], "stage": "preview", "error": str(exc)})
    return {"ok": not (missing or extra or png_duplicates or prediction_duplicates or failures),
            "has_ground_truth": False, "PQ": None,
            "note": "Coverage/schema validation and prediction statistics only; no accuracy estimate",
            "input_images": len(pngs), "prediction_files": len(predictions),
            "missing": missing, "extra": extra, "duplicate_input_names": png_duplicates,
            "duplicate_prediction_names": prediction_duplicates, "failures": failures,
            "empty_images": sum(r["empty"] for r in valid), "valid_images": len(valid),
            "primitive_count": sum(r["primitive_count"] for r in valid),
            "thing_instance_count": sum(r["thing_instances"] for r in valid),
            "stuff_segment_count": sum(r["stuff_classes"] for r in valid),
            "outside_primitive_count": sum(r["outside_primitive_count"] for r in valid),
            "per_class_primitives": dict(sorted(classes.items())),
            "per_class_thing_instances": dict(sorted(instances.items())),
            "samples": previews, "per_image": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.input, args.pred_dir, preview_dir=args.preview_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report[key] for key in ["ok", "input_images", "prediction_files", "valid_images", "empty_images", "primitive_count", "has_ground_truth"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
