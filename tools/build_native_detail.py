#!/usr/bin/env python3
"""Build streamed native detail crops from clean TRAIN pages and real YOLO boxes.

Targets are complete same-class GT instance ink rendered by the scorer. GT
boxes supplement positives; deployment-shaped YOLO boxes include hard negatives.
No test/dev labels or images are used to construct the training examples.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("YOLO_OFFLINE", "1")

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.raster import render_window
from cadvec.layers import load_rgba
from cadvec.pipeline_segdet import _model
from cadvec.roi_ink import RoiInkRefiner
from cadvec.native_detail import tile_positions


def bbox_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(x1-x0, 0) * max(y1-y0, 0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-pages", type=int, default=1200)
    parser.add_argument("--val-pages", type=int, default=200)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--max-per-class", type=int, default=4)
    args = parser.parse_args()
    if args.tile < 64 or args.tile % 8:
        parser.error("--tile must be >=64 and divisible by 8")
    args.out.mkdir(parents=True, exist_ok=True)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    selected = {}
    for role, maximum in (("train", args.train_pages), ("val", args.val_pages)):
        names = list(splits["train_final"][role]); rng.shuffle(names)
        selected[role] = names[:maximum] if maximum > 0 else names
    if set(selected["train"]) & set(selected["val"]):
        raise ValueError("Training and validation pages overlap")
    models = [RoiInkRefiner.load("weights/roi_ink_c2.pt", device="cuda"),
              RoiInkRefiner.load("weights/roi_ink_512.pt", device="cuda"),
              RoiInkRefiner.load("weights/roi_ink_256.pt", device="cuda")]
    detector = _model("weights/yolo_seg.pt")
    routes = [{2}, {3, 4, 22, 26, 29}, {8, 12, 14, 15, 18, 19, 20, 21, 25, 27, 28}]
    rows = []; counts = Counter(); failures = []
    metadata = {"schema_version": 1, "seed": args.seed, "tile": args.tile,
                "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
                "selected_pages": selected,
                "coarse_weights": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in ("weights/roi_ink_c2.pt", "weights/roi_ink_512.pt",
                                             "weights/roi_ink_256.pt", "weights/yolo_seg.pt")}}
    (args.out/"metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    for role in ("train", "val"):
        directory = args.out / role; directory.mkdir(exist_ok=True)
        for page_number, stem in enumerate(selected[role]):
            png = args.data_root/"train_final"/f"{stem}.png"
            label = args.data_root/"train_json"/f"{stem}.json"
            gt = json.loads(label.read_text(encoding="utf-8"))
            rgb, alpha = load_rgba(png)
            image = np.clip(rgb * alpha[..., None] + 255*(1-alpha[..., None]), 0, 255).astype(np.uint8)
            height, width = image.shape[:2]
            native_groups = {}
            for semantic, instance, command, coordinates in zip(
                    gt["semanticIds"],gt["instanceIds"],gt["commands"],gt["args"]):
                if semantic not in set.union(*routes) or instance < 0:
                    continue
                native_groups.setdefault((semantic,instance),[]).append({
                    "command":command,"points":list(zip(coordinates[::2],coordinates[1::2]))})
            native_scale = (width/gt["width"],height/gt["height"])
            rendered = {key:render_window(items,(height,width),native_scale)
                        for key,items in native_groups.items()}
            truths = {}
            for (class_id, instance), mask in rendered.items():
                if class_id not in set.union(*routes) or instance < 0 or not mask.any():
                    continue
                ys, xs = np.nonzero(mask)
                box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                truths.setdefault(class_id, []).append((instance, mask, box))
            # The production detector reads the PNG path. Preserve its exact
            # alpha/color handling rather than substituting the ROI white RGB.
            detection = detector.predict(source=str(png), imgsz=640, conf=.2, iou=.5,
                                         verbose=False, device=0)[0]
            candidates = []
            per_class = Counter()
            if detection.boxes is not None and detection.masks is not None:
                for cls, confidence, polygon in zip(detection.boxes.cls.cpu().numpy(),
                        detection.boxes.conf.cpu().numpy(), detection.masks.xy):
                    class_id = int(cls)
                    if class_id not in set.union(*routes) or len(polygon) < 3:
                        continue
                    if per_class[class_id] >= args.max_per_class:
                        continue
                    per_class[class_id] += 1
                    box = (float(polygon[:, 0].min()), float(polygon[:, 1].min()),
                           float(polygon[:, 0].max()), float(polygon[:, 1].max()))
                    choices = sorted(((bbox_iou(box, truth[2]), i) for i, truth in
                                      enumerate(truths.get(class_id, []))), reverse=True)
                    # Ambiguous or partially overlapping detections are skipped
                    # rather than labeled as background despite a possible GT.
                    if choices and choices[0][0] >= .45:
                        if len(choices) > 1 and choices[1][0] > choices[0][0] - .1:
                            continue
                        target_index = choices[0][1]
                    elif not choices or choices[0][0] == 0:
                        target_index = None
                    else:
                        continue
                    candidates.append((class_id, float(confidence), box, target_index, "detector"))
            # A few GT-box positives recover examples the fixed detector misses;
            # these are explicitly recorded and never used as evaluation boxes.
            for class_id, entries in (truths.items() if role == "train" else []):
                for i, (_instance, _mask, box) in enumerate(entries[:2]):
                    if not any(c[0] == class_id and c[3] == i for c in candidates):
                        candidates.append((class_id, 1., box, i, "gt_supplement"))
            for classes, model in zip(routes, models):
                group = [c for c in candidates if c[0] in classes]
                if not group:
                    continue
                analyzed = []
                for start in range(0, len(group), 8):
                    analyzed.extend(model.analyze(
                        image, [(c[0], c[1], c[2]) for c in group[start:start+8]],
                        width, height, gt["width"], gt["height"], include_empty=True))
                if len(analyzed) != len(group):
                    raise ValueError("Coarse crop alignment mismatch")
                for number, (candidate, item) in enumerate(zip(group, analyzed)):
                    class_id, confidence, _box, target_index, source = candidate
                    left, top, right, bottom = item["box"]
                    crop_rgb = image[top:bottom, left:right]
                    probability = item["probability"]
                    target = (truths[class_id][target_index][1][top:bottom, left:right]
                              if target_index is not None else np.zeros(probability.shape, dtype=bool))
                    instance = truths[class_id][target_index][0] if target_index is not None else -1
                    # Tile instead of downscaling the native detail evidence.
                    positions = [(x,y) for y in tile_positions(bottom-top, args.tile)
                                 for x in tile_positions(right-left, args.tile)]
                    for tile_index, (x,y) in enumerate(positions):
                        valid_h = min(args.tile, bottom-top-y)
                        valid_w = min(args.tile, right-left-x)
                        sample = np.full((args.tile, args.tile, 4), 255, dtype=np.uint8)
                        sample[..., 3] = 0
                        sample[:valid_h, :valid_w, :3] = crop_rgb[y:y+valid_h, x:x+valid_w]
                        sample[:valid_h, :valid_w, 3] = np.clip(
                            probability[y:y+valid_h, x:x+valid_w]*255, 0, 255).astype(np.uint8)
                        truth = np.zeros((args.tile, args.tile), dtype=np.uint8)
                        truth[:valid_h, :valid_w] = target[y:y+valid_h, x:x+valid_w]
                        # Keep a bounded fraction of empty tiles from large boxes.
                        if not truth.any() and probability[y:y+valid_h, x:x+valid_w].max() < .05:
                            if tile_index > 0:
                                continue
                        name = f"{stem}_{class_id}_{number}_{tile_index}.npz"
                        np.savez_compressed(directory/name, image=sample, target=truth,
                                            valid_shape=np.array([valid_h,valid_w],dtype=np.int32))
                        rows.append({"path": f"{role}/{name}", "role": role, "stem": stem,
                                     "class": class_id, "instance": instance, "source": source,
                                     "confidence": confidence, "positive": bool(truth.any()),
                                     "coarse_threshold": float(model.threshold),
                                     "scale_xy": list(native_scale),
                                     "target_crop_coverage": (float(target.sum()) / max(int(
                                         truths[class_id][target_index][1].sum()), 1)
                                         if target_index is not None else None)})
                        counts[f"{role}_{'positive' if truth.any() else 'negative'}"] += 1
            if (page_number+1) % 25 == 0:
                print(role, page_number+1, "/", len(selected[role]), dict(counts), flush=True)
                (args.out/"index.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    (args.out/"index.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    print("finished", dict(counts), "samples", len(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
