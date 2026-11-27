#!/usr/bin/env python3
"""CPU-only, fixed-detector probe of complete native training templates.

Candidate selection uses training SVGs, target PNGs, frozen detector records and
the current predictions. Evaluation labels are opened only after every output
has been written. This experiment is deliberately separate from infer.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import shutil
import sys
from types import SimpleNamespace
import zlib

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.layers import load_rgba
from cadvec.local_match import CropFrame, match_local
from cadvec.raster import render_window
from cadvec.schema import assemble_output, write_output
from cadvec.template_build import build_library
from cadvec.templating import TemplateLibrary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def audit_sources(training, development, groups):
    training, development = set(training), set(development)
    missing = (training | development) - set(groups)
    overlap = {groups[s] for s in training} & {groups[s] for s in development}
    if missing or training & development or overlap:
        raise ValueError(f"Unsafe template sources: missing={sorted(missing)}, source_overlap={sorted(overlap)}")
    return {"training_pages": len(training), "development_pages": len(development),
            "training_groups": len({groups[s] for s in training}),
            "development_groups": len({groups[s] for s in development}), "overlap_groups": []}


def native_from_image(viewbox, size):
    x, y, width, height = map(float, viewbox)
    iw, ih = size
    if min(width, height, iw, ih) <= 0:
        raise ValueError("Invalid native/image frame")
    return np.array([[width / iw, 0, x], [0, height / ih, y], [0, 0, 1]], dtype=float)


def transformed_items(items, matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 2x3 similarity")
    linear = matrix[:, :2]
    gram = linear.T @ linear
    if np.linalg.det(linear) <= 0 or not np.allclose(gram, np.eye(2) * np.trace(gram) / 2, rtol=1e-5, atol=1e-8):
        raise ValueError("Template transfer requires an orientation-preserving similarity")
    angle = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    # The output schema has axis-aligned ellipses. An arbitrary ellipse rotation
    # cannot be represented exactly, so this probe declines those candidates.
    if any(p["command"] == 3 for p in items) and abs((angle + 45) % 90 - 45) > 2:
        raise ValueError("Non-axis-aligned ellipse transfer is unsupported")
    scale = float(np.sqrt(np.trace(gram) / 2))
    return [dict(p, points=(np.asarray(p["points"]) @ linear.T + matrix[:, 2]).tolist(),
                 width=float(p["width"]) * scale, length=float(p["length"]) * scale)
            for p in items]


@lru_cache(maxsize=24)
def png_image(path):
    rgb, alpha = load_rgba(Path(path))
    return np.clip(rgb * alpha[..., None] + 255 * (1 - alpha[..., None]), 0, 255).astype(np.uint8)


def bbox(items):
    points = np.asarray([p for item in items for p in item["points"]])
    return np.r_[points.min(0), points.max(0)]


def iou_box(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    area = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - area
    return float(area / max(union, 1e-12))


def square_crop(image, box, *, side_pixels=256):
    box = np.asarray(box, float)
    side = max(16, int(math.ceil(max(box[2:] - box[:2]) * 1.5 + 16)))
    center = (box[:2] + box[2:]) / 2
    x, y = np.floor(center - side / 2).astype(int)
    crop = np.full((side, side, 3), 255, np.uint8)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + side, width), min(y + side, height)
    if x1 > x0 and y1 > y0:
        crop[y0-y:y1-y, x0-x:x1-x] = image[y0:y1, x0:x1]
    scale = side_pixels / side
    return cv2.resize(crop, (side_pixels, side_pixels), interpolation=cv2.INTER_LINEAR), (int(x), int(y)), scale


def support_metrics(items, image, canvas_size, detector_box):
    height, width = image.shape[:2]
    sx, sy = width / canvas_size[0], height / canvas_size[1]
    box = np.asarray(detector_box, float)
    pad = 4
    x0, y0 = np.maximum(0, np.floor(box[:2] - pad)).astype(int)
    x1, y1 = np.minimum([width, height], np.ceil(box[2:] + pad)).astype(int)
    if x1 <= x0 or y1 <= y0:
        return {"precision": 0., "recall": 0., "F1": 0., "pixels": 0}
    rendered = render_window(items, (y1-y0, x1-x0), (sx, sy), (int(x0), int(y0)))
    ink = image[y0:y1, x0:x1].min(2) < 245
    # A fixed 1.5 native-pixel support allowance handles anti-aliasing/line width;
    # it is a PNG acceptance proxy, never the strict 1000-pixel PQ evaluator.
    distance = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    reverse = cv2.distanceTransform((~rendered).astype(np.uint8), cv2.DIST_L2, 3)
    count = int(rendered.sum())
    precision = float(np.mean(distance[rendered] <= 1.5)) if count else 0.
    recall = float(np.mean(reverse[ink] <= 1.5)) if ink.any() and count else 0.
    return {"precision": precision, "recall": recall,
            "F1": 2 * precision * recall / max(precision + recall, 1e-12), "pixels": count}


def instance_items(data):
    groups = {}
    for i, instance in enumerate(data["instanceIds"]):
        args = data["args"][i]
        groups.setdefault(instance, []).append({"command": data["commands"][i],
            "points": list(map(list, zip(args[::2], args[1::2]))), "length": data["lengths"][i],
            "rgb": data["rgb"][i], "layer_id": data["layerIds"][i], "width": data["widths"][i],
            "semantic": data["semanticIds"][i]})
    return groups


def replace_instances(data, replacements):
    if not replacements:
        return data
    emitted, done = [], set()
    groups = instance_items(data)
    for i, instance in enumerate(data["instanceIds"]):
        if instance in replacements:
            if instance in done:
                continue
            done.add(instance)
            items = replacements[instance]
        else:
            items = [groups[instance].pop(0)]
        for p in items:
            emitted.append(SimpleNamespace(command=p["command"], points=p["points"], length=p["length"],
                semantic_id=p["semantic"], instance_id=instance, rgb=p["rgb"],
                layer_id=p["layer_id"], width=p["width"]))
    return assemble_output(emitted, width=data["width"], height=data["height"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-protocol", type=Path, required=True)
    p.add_argument("--source-splits", type=Path, required=True)
    p.add_argument("--svg-dir", type=Path, required=True)
    p.add_argument("--png-dir", type=Path, required=True)
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path)
    p.add_argument("--pages", type=int, default=20)
    p.add_argument("--seed", type=int, default=73)
    args = p.parse_args(argv)
    if not 1 <= args.pages <= 20:
        p.error("This registered probe is limited to 1..20 pages")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "protocol.json").exists():
        raise ValueError("Use a new output directory; this probe freezes its protocol once")
    training = read(args.training_protocol)["training"]["pages"]
    if len(training) != 800 or len(set(training)) != 800:
        raise ValueError("Expected the declared original native-head 800 training pages")
    development = sorted(path.stem for path in args.baseline.glob("*.json"))
    source_groups = read(args.source_splits)["groups"]["train_final"]
    audit = audit_sources(training, development, source_groups)
    selected = sorted(random.Random(args.seed).sample(development, min(args.pages, len(development))))
    protocol = {"schema_version": 1, "method": "complete native templates + local mutual RootSIFT similarity + PNG support",
        "seed": args.seed, "classes": [2, 3, 8, 18, 22], "selected_pages": selected,
        "training_pages": sorted(training), "source_audit": audit,
        "detectors": "frozen real YOLO records; boxes/classes/confidence unchanged; only replace existing same-class instances",
        "limits": {"detections_per_page": 8, "templates_per_detection": 8, "crop_side": 256},
        "matching": {"ratio": .75, "ransac_px": 2.5, "min_inliers": 3, "min_inlier_ratio": .4, "min_coverage": .02},
        "acceptance": {"precision": .85, "F1_gain": .03, "pixel_ratio": [.6, 1.8], "bbox_IoU": .35,
                       "scale_vs_box_ratio": [.5, 2.], "ellipse_axis_tolerance_degrees": 2.},
        "selection": "highest PNG support F1 among passing templates; no development labels used",
        "tool_sha256": sha(__file__), "training_protocol_sha256": sha(args.training_protocol),
        "source_splits_sha256": sha(args.source_splits),
        "baseline_sha256": {s: sha(args.baseline / (s + '.json')) for s in selected},
        "snapshot_sha256": {s: sha(args.snapshots / (s + '.json')) for s in selected},
        "target_png_sha256": {s: sha(args.png_dir / (s + '.png')) for s in selected}}
    dump(args.out / "protocol.json", protocol)
    print("Protocol frozen; building complete templates from 800 declared training pages", flush=True)
    payload = build_library(args.svg_dir, allowed_stems=set(training))
    payload["templates"] = [t for t in payload["templates"] if t["class"] in protocol["classes"]]
    raw = zlib.compress(pickle.dumps(payload, protocol=4), 6)
    (args.out / "native_templates.pkl").write_bytes(raw)
    library = TemplateLibrary(payload)
    dump(args.out / "library_provenance.json", {"library_sha256": hashlib.sha256(raw).hexdigest(),
         "class_templates": dict(Counter(t["class"] for t in payload["templates"])), **payload["build"]})
    print("Library templates:", len(library.templates), flush=True)

    @lru_cache(maxsize=128)
    def source_crop(index):
        template = library.templates[index]
        source = template["sources"][0]
        path = args.svg_dir / (source["image_stem"] + ".png")
        image = png_image(str(path))
        if source["viewbox"] is None:
            raise ValueError("An explicit source viewBox is required")
        matrix = native_from_image(source["viewbox"], (image.shape[1], image.shape[0]))
        points = library.points[index]
        pixels = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(matrix).T)[:, :2]
        box = np.r_[pixels.min(0), pixels.max(0)]
        crop, offset, scale = square_crop(image, box)
        return crop, CropFrame(offset, (scale, scale), matrix), box, sha(path)

    counts = Counter()
    for page_index, stem in enumerate(selected):
        snapshot = read(args.snapshots / (stem + ".json"))
        baseline = read(args.baseline / (stem + ".json"))
        image = png_image(str(args.png_dir / (stem + ".png")))
        height, width = image.shape[:2]
        if list(snapshot["support"]["size"]) != [height, width]:
            raise ValueError("Frozen detector frame and target PNG differ")
        canvas = (baseline["width"], baseline["height"])
        groups = instance_items(baseline)
        target_native = native_from_image((0, 0, *canvas), (width, height))
        scale_to_pixel = np.array([width / canvas[0], height / canvas[1]] * 2)
        boxes = {key: bbox(items) * scale_to_pixel for key, items in groups.items() if key >= 0}
        records = [r for r in snapshot["support"]["detections"] if r["class"] in protocol["classes"]]
        records = sorted(records, key=lambda r: (-r["conf"], r["class"], r["box"]))[:8]
        replacements, audit_rows, occupied = {}, [], set()
        for record in records:
            counts["detections"] += 1
            class_id, detector_box = int(record["class"]), np.asarray(record["box"], float)
            eligible = [(iou_box(box, detector_box), instance) for instance, box in boxes.items()
                        if groups[instance][0]["semantic"] == class_id and instance not in occupied]
            overlap, instance = max(eligible, default=(0., -1))
            row = {"detector": record, "baseline_instance": instance, "baseline_box_iou": overlap, "candidates": []}
            audit_rows.append(row)
            if overlap < .15:
                row["reason"] = "no_existing_same_class_instance"; counts[row["reason"]] += 1
                continue
            occupied.add(instance)
            baseline_support = support_metrics(groups[instance], image, canvas, detector_box)
            row["baseline_support"] = baseline_support
            target_crop, offset, scale = square_crop(image, detector_box)
            target_frame = CropFrame(offset, (scale, scale), target_native)
            query = np.asarray([point for item in groups[instance] for point in item["points"]])
            retrieved = library.retrieve(query, top_k=8, class_filter=class_id)
            candidates = []
            for distance, index, rotation in retrieved:
                counts["template_attempts"] += 1
                template = library.templates[index]
                crop, source_frame, source_box, png_sha = source_crop(index)
                found = match_local(crop, target_crop, source_frame=source_frame, target_frame=target_frame)
                detail = {"template_index": index, "descriptor_distance": distance,
                          "source": template["sources"][0], "source_png_sha256": png_sha,
                          "geometry_sha256": template["geometry_sha256"], "matching": found}
                row["candidates"].append(detail)
                if not found["ok"]:
                    counts["match_rejected:" + found["reason"]] += 1
                    continue
                counts["feature_matched"] += 1
                try:
                    items = transformed_items(template["primitives"], found["native_matrix"])
                except ValueError as error:
                    detail["rejected"] = str(error); counts["unsupported_transform"] += 1; continue
                transformed_box = bbox(items) * scale_to_pixel
                actual_scale = math.hypot(*np.asarray(found["image_matrix"])[:, 0])
                expected_scale = max(detector_box[2:] - detector_box[:2]) / max(max(source_box[2:] - source_box[:2]), 1e-9)
                detail["scale_vs_box"] = actual_scale / expected_scale
                detail["bbox_iou"] = iou_box(transformed_box, detector_box)
                quality = support_metrics(items, image, canvas, detector_box)
                detail["support"] = quality
                pixels_ratio = quality["pixels"] / max(baseline_support["pixels"], 1)
                if not (.5 <= detail["scale_vs_box"] <= 2 and detail["bbox_iou"] >= .35
                        and quality["precision"] >= .85 and .6 <= pixels_ratio <= 1.8
                        and quality["F1"] >= baseline_support["F1"] + .03):
                    detail["rejected"] = "PNG_or_geometry_acceptance"; counts["acceptance_rejected"] += 1; continue
                for item in items:
                    item["semantic"] = class_id
                candidates.append((quality["F1"], -index, items, detail))
                counts["passing_templates"] += 1
            if candidates:
                _, _, items, chosen = max(candidates, key=lambda c: c[:2])
                replacements[instance] = items
                row["selected_template"] = chosen["template_index"]
                counts["replaced_instances"] += 1
            else:
                row["reason"] = "no_passing_template"
        write_output(baseline, args.out / "baseline" / (stem + ".json"))
        write_output(replace_instances(baseline, replacements), args.out / "templates" / (stem + ".json"))
        dump(args.out / "audit" / (stem + ".json"), audit_rows)
        print("Template probe", page_index + 1, "/", len(selected), dict(counts), flush=True)
    dump(args.out / "prediction_audit.json", {"all_predictions_written_before_labels": True,
        "pages": len(selected), "counts": dict(counts), "prediction_sha256": {
            role: {s: sha(args.out / role / (s + '.json')) for s in selected} for role in ("baseline", "templates")}})
    if args.gt_dir:
        # No evaluation import or label I/O occurs above this phase boundary.
        from cadvec.evaluate import evaluate_directories
        from tools.summarize_ablation import aggregate
        label_dir = args.out / "evaluation_labels"
        label_dir.mkdir(exist_ok=True)
        for stem in selected:
            shutil.copyfile(args.gt_dir / (stem + ".json"), label_dir / (stem + ".json"))
        reports = {}
        for role in ("baseline", "templates"):
            report = evaluate_directories(label_dir, args.out / role)
            dump(args.out / (role + "_eval.json"), report)
            reports[role] = aggregate(list(report["per_image"].values()))
        dump(args.out / "comparison.json", {"metrics": reports, "counts": dict(counts),
            "scope": "20-page previously observed development probe; no independent confirmation", "default_changed": False})
        print(json.dumps(reports), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
