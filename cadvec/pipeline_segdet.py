"""YOLO-seg 实例分割 + 图像笔画矢量化。

实例分割提供“完整实例候选区域”；输出时不用分割掩码本身（它是凸包），
而是把区域内的图像笔画矢量化，保证几何与 GT 的 4 点折线一致。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    IGNORE_CLASS,
    InferenceConfig,
    STUFF_CLASS_MAX,
    STUFF_CLASS_MIN,
)
from .dataset import component_scene
from .layers import load_palette, load_rgba
from .schema import assemble_output, empty_output, validate_output
from .templating import TemplateLibrary

YOLO_SEG_CACHE: dict[str, Any] = {}
TEMPLATE_CACHE: dict[str, Any] = {}


def _library(path):
    if path is None:
        return None
    key = str(Path(path).resolve())
    library = TEMPLATE_CACHE.get(key)
    if library is None:
        library = TemplateLibrary.load(path)
        if library is not None:
            TEMPLATE_CACHE[key] = library
    return library


def _model(path: str | Path):
    key = str(Path(path).resolve())
    model = YOLO_SEG_CACHE.get(key)
    if model is None:
        import os
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        model = YOLO(str(path))
        YOLO_SEG_CACHE[key] = model
    return model


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, stroke=None, semantic=35, instance=-1, payload=None):
        if payload is not None:
            self.command = int(payload["command"])
            self.points = payload["points"]
            self.length = float(payload["length"])
            self.semantic_id = int(payload["semantic"])
            self.instance_id = int(instance)
            self.rgb = [int(v) for v in payload["rgb"]]
            self.layer_id = int(payload["layer"])
            self.width = float(payload["width"])
            return
        self.command = 0
        self.points = stroke.points[:4]
        self.length = float(stroke.length)
        self.semantic_id = int(semantic)
        self.instance_id = int(instance)
        self.rgb = [int(v) for v in stroke.rgb]
        self.layer_id = int(stroke.layer_id)
        self.width = float(stroke.width)


def _dense_polyline(points, scale_x: float, scale_y: float,
                    step: float = 1.0) -> np.ndarray:
    """把画布坐标折线按约 ``step`` 像素间距加密采样，返回画布坐标数组。"""
    pixels = np.asarray(points, dtype=np.float64) * np.asarray(
        [scale_x, scale_y], dtype=np.float64)
    if len(pixels) < 2:
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pieces = [pixels[:1]]
    for start, end in zip(pixels[:-1], pixels[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(int(np.ceil(distance / max(step, 1e-6))), 1)
        ratios = np.arange(1, count + 1, dtype=np.float64) / count
        pieces.append(start[None, :] + (end - start)[None, :] * ratios[:, None])
    dense = np.concatenate(pieces, axis=0)
    return dense / np.asarray([scale_x, scale_y], dtype=np.float64)


def _resample_four(points: np.ndarray) -> list[list[float]]:
    """按弧长等距取 4 个点（与 skeleton.sample_polyline 的约定一致）。"""
    if len(points) == 0:
        return []
    if len(points) == 1:
        point = points[0]
        return [[float(point[0]), float(point[1])]] * 4
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        point = points[0]
        return [[float(point[0]), float(point[1])]] * 4
    out = []
    for index in range(4):
        target = total * index / 3.0
        position = int(np.searchsorted(cumulative, target, side="right") - 1)
        position = min(max(position, 0), len(points) - 2)
        segment = float(deltas[position])
        ratio = 0.0 if segment <= 1e-9 else (target - cumulative[position]) / segment
        point = points[position] + (points[position + 1] - points[position]) * ratio
        out.append([float(point[0]), float(point[1])])
    return out


def _local_attribution(detections, strokes, scale_x: float, scale_y: float,
                       width: int, height: int, min_length: float = 0.5):
    """把每条检测掩码内的图像笔画**局部切分**后归属给该实例。

    与整条笔画的“先到先得”不同：先沿骨架折线按像素采样，每个采样点只归属
    第一个（置信度最高的）覆盖它的检测；检测只拿到真正落在掩码内的连续片段。
    这样穿过门窗框的墙线不会被整条收进实例，也不会因为同一笔被两个实例
    重复占用。
    """
    dense_all: list[np.ndarray] = []
    for stroke in strokes:
        dense_all.append(_dense_polyline(stroke.points, scale_x, scale_y))
    if not any(len(item) for item in dense_all):
        return []
    lengths = [len(item) for item in dense_all]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    total = int(offsets[-1])
    all_pixels = np.concatenate([
        item * np.asarray([scale_x, scale_y]) for item in dense_all
    ], axis=0) if total else np.zeros((0, 2))
    stroke_ids = np.repeat(np.arange(len(strokes)), lengths)
    rows = np.clip(np.round(all_pixels[:, 1]).astype(int), 0, height - 1)
    columns = np.clip(np.round(all_pixels[:, 0]).astype(int), 0, width - 1)
    owned = np.zeros(total, dtype=bool)

    results = []
    for confidence, class_id, mask in detections:
        if class_id == IGNORE_CLASS or total == 0:
            continue
        inside = mask[rows, columns] & ~owned
        if not inside.any():
            continue
        # 连续片段：同一笔画内相邻采样点都 inside。
        previous_inside = np.concatenate([[False], inside[:-1]])
        same_stroke = np.concatenate([[False], stroke_ids[1:] == stroke_ids[:-1]])
        starts = np.flatnonzero(inside & ~(previous_inside & same_stroke))
        next_bad = np.concatenate([
            ~inside[1:] | (stroke_ids[1:] != stroke_ids[:-1]), [True],
        ])
        ends = np.flatnonzero(inside & next_bad)
        segments = []
        for start, end in zip(starts, ends):
            piece = dense_all[int(stroke_ids[start])][
                start - offsets[int(stroke_ids[start])]:
                end - offsets[int(stroke_ids[start])] + 1
            ]
            if len(piece) < 2:
                continue
            length = float(np.linalg.norm(np.diff(piece, axis=0), axis=1).sum())
            if length < min_length:
                continue
            source = strokes[int(stroke_ids[start])]
            segments.append({
                "command": 0,
                "points": _resample_four(piece),
                "polyline": piece,
                "length": length,
                "rgb": [int(v) for v in source.rgb],
                "layer": int(source.layer_id),
                "width": float(source.width),
            })
        owned |= inside
        if not segments:
            continue
        results.append((confidence, class_id, segments))
    return results


def run_image_segdet(
    png_path: str | Path,
    weights: str | Path,
    config: InferenceConfig,
    palette_path: str | Path,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    det_conf: float = 0.25,
    det_iou: float = 0.5,
    image_size: int = 640,
    max_instances: int = 12,
    inside_fraction: float = 0.5,
    templates_path: str | Path | None = None,
    template_max_score: float = 3.0,
    device: str | int | None = None,
    local_split: bool = False,
    min_segment_length: float = 0.5,
    support_classes: frozenset[int] = frozenset(),
    refiner: Any = None,
    refiner_classes: frozenset[int] = frozenset(),
    refiners: tuple[tuple[frozenset[int], Any], ...] = (),
) -> dict[str, Any]:
    started = time.perf_counter()
    palette = load_palette(palette_path)
    scene = component_scene(png_path, palette, config, canvas_width, canvas_height)
    if not scene.strokes or not scene.layers:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return {"data": data, "support": None,
                "stats": {"primitive_count": 0, "instances": 0,
                          "seconds": round(time.perf_counter() - started, 3),
                          "command_counts": {"line": 0, "arc": 0,
                                             "circle": 0, "ellipse": 0}}}
    height, width = scene.layers[0].mask.shape
    from scipy import ndimage
    stroke_mask = np.zeros((height, width), dtype=bool)
    for layer in scene.layers:
        stroke_mask |= layer.mask
    distance_transform = ndimage.distance_transform_edt(~stroke_mask)
    canvas_to_pixel = width / max(canvas_width, 1)
    library = _library(templates_path)
    model = _model(weights)
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    result = model.predict(source=str(png_path), imgsz=image_size, conf=det_conf,
                           iou=det_iou, verbose=False, device=device,
                           retina_masks=False)[0]
    detections = []
    detection_records = []
    support_polygons: dict[int, list[list[list[float]]]] = {}
    if result.masks is not None and result.boxes is not None and len(result.masks):
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        polygons = result.masks.xy
        for polygon, class_id, confidence in zip(polygons, classes, confidences):
            if polygon is None or len(polygon) < 3:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
            detections.append((float(confidence), int(class_id), mask.astype(bool)))
            ys, xs = np.nonzero(mask)
            if len(xs):
                detection_records.append({
                    "class": int(class_id),
                    "conf": float(confidence),
                    "box": [float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max())],
                })
            if int(class_id) in support_classes:
                support_polygons.setdefault(int(class_id), []).append(
                    np.round(np.asarray(polygon, dtype=np.float64), 1).tolist()
                )
    detections.sort(key=lambda item: -item[0])

    scale_x = width / max(canvas_width, 1)
    scale_y = height / max(canvas_height, 1)
    refiner_assigned: list[tuple[float, int, list[dict]]] = []
    plan = list(refiners)
    if not plan and refiner is not None and refiner_classes:
        plan = [(frozenset(refiner_classes), refiner)]
    handled_classes: set[int] = set()
    for classes, _ in plan:
        handled_classes.update(classes)
    stroke_detections = [item for item in detections
                         if item[1] not in handled_classes]
    if plan:
        rgb, alpha = load_rgba(png_path)
        image = np.clip(
            rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
            0, 255).astype(np.uint8)
        for classes, model in plan:
            boxes = []
            for confidence, class_id, mask in detections:
                if class_id not in classes:
                    continue
                ys, xs = np.nonzero(mask)
                if len(xs) == 0:
                    continue
                boxes.append((int(class_id), float(confidence),
                              (float(xs.min()), float(ys.min()),
                               float(xs.max()), float(ys.max()))))
            if not boxes:
                continue
            for _class_id, confidence, segments in model.refine(
                    image, boxes, width, height, canvas_width, canvas_height):
                refiner_assigned.append((confidence, _class_id, segments))
    if local_split:
        assigned = _local_attribution(
            stroke_detections, scene.strokes, scale_x, scale_y, width, height,
            min_length=min_segment_length,
        )
    else:
        claimed = [False] * len(scene.strokes)
        assigned = []
        for confidence, class_id, mask in stroke_detections:
            if class_id == IGNORE_CLASS:
                continue
            segments = []
            for index, stroke in enumerate(scene.strokes):
                if claimed[index]:
                    continue
                points = stroke.points
                inside = 0
                for x, y in points:
                    column = int(min(max(x * scale_x, 0), width - 1))
                    row = int(min(max(y * scale_y, 0), height - 1))
                    if mask[row, column]:
                        inside += 1
                if inside / max(len(points), 1) >= inside_fraction:
                    claimed[index] = True
                    segments.append({
                        "command": int(stroke.command),
                        "points": [list(point) for point in stroke.points],
                        "polyline": _dense_polyline(stroke.points, scale_x, scale_y),
                        "length": float(stroke.length),
                        "rgb": [int(v) for v in stroke.rgb],
                        "layer": int(stroke.layer_id),
                        "width": float(stroke.width),
                    })
            if segments:
                assigned.append((confidence, class_id, segments))
    groups = [(confidence, class_id, segments) for confidence, class_id, segments in assigned
              if not STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX]
    stuff_groups = [(class_id, segments) for _confidence, class_id, segments in assigned
                    if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX]

    if refiner_assigned:
        assigned = sorted(assigned + refiner_assigned, key=lambda item: -item[0])
        groups = [(confidence, class_id, segments)
                  for confidence, class_id, segments in assigned
                  if class_id in refiner_classes
                  and not STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX]
        groups += [(confidence, class_id, segments)
                   for confidence, class_id, segments in assigned
                   if class_id not in refiner_classes
                   and not STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX]
        # Stuff 没有实例身份：同一类别的细化段全部并入该类的 -1 段。
        stuff_groups.extend((class_id, segments)
                            for _confidence, class_id, segments in refiner_assigned
                            if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX)
    if max_instances > 0 and not refiner_assigned:
        groups = groups[:max_instances]
    elif max_instances > 0 and refiner_assigned:
        # 细化类别已按类内 NMS/上限处理，不再参与全局 top-K，否则门窗会被截断。
        refiner_groups = [item for item in groups if item[1] in refiner_classes]
        other_groups = [item for item in groups if item[1] not in refiner_classes]
        groups = refiner_groups + other_groups[:max_instances]
    emitted = []
    instance_id = 0
    for confidence, class_id, segments in groups:
        fit = None
        if library is not None:
            points = np.asarray([point for segment in segments
                                 for point in segment["polyline"]])
            local_mask = np.zeros((height, width), dtype=np.uint8)
            for segment in segments:
                pixels = np.round(
                    np.asarray(segment["polyline"]) * canvas_to_pixel
                ).astype(np.int32)
                cv2.polylines(local_mask, [pixels.reshape(-1, 1, 2)], False, 1,
                              thickness=2)
            bounds = points
            if len(points) >= 8 and len(bounds):
                box_scale = max(float(np.ptp(bounds[:, 0])), float(np.ptp(bounds[:, 1])), 0.5)
                center = (float(bounds[:, 0].mean()), float(bounds[:, 1].mean()))
                fit = library.fit(
                    points, distance_transform, canvas_to_pixel, top_k=8,
                    class_filter=class_id, center=center, scale_override=box_scale,
                    stroke_mask=local_mask.astype(bool),
                )
        if fit is not None and fit.class_id <= STUFF_CLASS_MAX and fit.score <= template_max_score:
            for payload in library.instantiate(fit, np.zeros((0, 2))):
                emitted.append(_Primitive(payload=payload, instance=instance_id))
        else:
            for segment in segments:
                payload = {key: segment[key] for key in
                           ("command", "points", "length", "rgb", "layer", "width")}
                payload["semantic"] = int(class_id)
                emitted.append(_Primitive(payload=payload, instance=instance_id))
        instance_id += 1
    for class_id, segments in stuff_groups:
        for segment in segments:
            payload = {key: segment[key] for key in
                       ("command", "points", "length", "rgb", "layer", "width")}
            payload["semantic"] = int(class_id)
            emitted.append(_Primitive(payload=payload, instance=-1))

    data = assemble_output(emitted, width=canvas_width, height=canvas_height)
    validate_output(data)
    return {
        "data": data,
        "support": {
            "size": [height, width],
            "polygons": support_polygons,
            "detections": detection_records,
        },
        "stats": {
            "primitive_count": len(emitted),
            "instances": instance_id,
            "detections": len(detections),
            "seconds": round(time.perf_counter() - started, 3),
            "command_counts": {"line": len(emitted), "arc": 0, "circle": 0, "ellipse": 0},
        },
    }
