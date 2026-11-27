"""按类别合并两个引擎的预测（hybrid engine）。

默认：v2/v3 像素管线作为主引擎，GAT 图元引擎只接管类 8（盲窗）——两个
验证半集都显示 GAT 在 class 8 上 PQ 更高，其余类别 v3 更好。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from .schema import assemble_output, validate_output
from .raster import render_items

RASTER = 400

_FIELDS = (
    "commands", "args", "lengths", "semanticIds", "instanceIds",
    "rgb", "layerIds", "widths",
)


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, data: dict[str, Any], index: int, instance_offset: int = 0):
        self.command = int(data["commands"][index])
        points = data["args"][index]
        self.points = [
            (float(points[position]), float(points[position + 1]))
            for position in range(0, 8, 2)
        ]
        self.length = float(data["lengths"][index])
        self.semantic_id = int(data["semanticIds"][index])
        self.instance_id = int(data["instanceIds"][index])
        if self.instance_id >= 0:
            self.instance_id += instance_offset
        self.rgb = [int(value) for value in data["rgb"][index]]
        self.layer_id = int(data["layerIds"][index])
        self.width = float(data["widths"][index])


def _class_instances(data: dict[str, Any], class_id: int,
                     instance_offset: int = 0):
    """把目标类的图元按实例分组并栅格化。"""
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(data["commands"])):
        if int(data["semanticIds"][index]) == class_id:
            groups[int(data["instanceIds"][index])].append(index)
    instances = []
    for instance, indices in groups.items():
        mask = np.zeros((RASTER, RASTER), dtype=np.uint8)
        for index in indices:
            points = np.round(
                np.asarray(data["args"][index]) * (RASTER / 100.0)
            ).astype(np.int32)
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, thickness=1)
        primitives = [
            _Primitive(data, index,
                       instance_offset if instance >= 0 else 0)
            for index in indices
        ]
        instances.append((mask.astype(bool), primitives))
    return instances


def _items(data: dict[str, Any], prefix: str) -> list[dict]:
    items = []
    for index in range(len(data["commands"])):
        points = data["args"][index]
        items.append({
            "command": int(data["commands"][index]),
            "points": [(float(points[k]), float(points[k + 1])) for k in range(0, 8, 2)],
            "length": float(data["lengths"][index]),
            "semantic": int(data["semanticIds"][index]),
            "instance": int(data["instanceIds"][index]),
            "key": (prefix, int(data["instanceIds"][index])),
            "rgb": [int(v) for v in data["rgb"][index]],
            "layer": int(data["layerIds"][index]),
            "width": float(data["widths"][index]),
        })
    return items


def _assemble(items: list[dict], width: int, height: int) -> dict[str, Any]:
    mapping: dict[tuple, int] = {}
    next_instance = 0

    def renumber(key):
        nonlocal next_instance
        if key[1] < 0:
            return -1
        if key not in mapping:
            mapping[key] = next_instance
            next_instance += 1
        return mapping[key]

    primitives = []
    for item in items:
        instance = renumber(item["key"])
        payload = {
            "commands": [item["command"]],
            "args": [[value for point in item["points"] for value in point]],
            "lengths": [item["length"]],
            "semanticIds": [item["semantic"]],
            "instanceIds": [instance],
            "rgb": [item["rgb"]],
            "layerIds": [item["layer"]],
            "widths": [item["width"]],
        }
        primitives.append(_Primitive(payload, 0))
    data = assemble_output(primitives, width=width, height=height)
    validate_output(data)
    return data


def merge_final(
    primary: dict[str, Any],
    gat: dict[str, Any] | None = None,
    segdet: dict[str, Any] | None = None,
    *,
    gat_classes: set[int] = frozenset({8}),
    segdet_classes: set[int] = frozenset(),
    segdet_union_classes: set[int] = frozenset(),
    suppress: set[int] = frozenset(),
    iou_threshold: float = 0.5,
    gate_classes: set[int] = frozenset(),
    gate_threshold: float = 0.1,
    gate_metric: str = "cover",
    support: dict[str, Any] | None = None,
    quality_rules: dict[int, tuple[str, str, float]] | None = None,
    raster_mode: str = "legacy400",
    source_identity: str = "scoped",
) -> dict[str, Any]:
    """最终融合：v3 ∪ GAT(class 8) → segdet 接管指定类别 → 抑制低精度类。"""
    width = int(primary.get("width", 100))
    height = int(primary.get("height", 100))
    items = _items(primary, "p")
    if raster_mode not in {"legacy400", "canonical1000"}:
        raise ValueError(f"Unknown fusion raster mode: {raster_mode}")
    if source_identity not in {"legacy", "scoped"}:
        raise ValueError(f"Unknown source identity mode: {source_identity}")
    raster_size = 1000 if raster_mode == "canonical1000" else RASTER

    def identity(item):
        return item["key"] if source_identity == "scoped" else item["key"][1]

    def raster(group: list[dict]) -> np.ndarray:
        if raster_mode == "canonical1000":
            return render_items(group, raster_size, width, height)
        import cv2
        mask = np.zeros((RASTER, RASTER), np.uint8)
        for item in group:
            pixels = np.round(
                np.asarray(item["points"]) * (RASTER / 100.0)
            ).astype(np.int32)
            cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], False, 1,
                          thickness=1)
        return mask.astype(bool)

    def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
        intersection = int(np.count_nonzero(first & second))
        if not intersection:
            return 0.0
        return intersection / max(int(np.count_nonzero(first | second)), 1)

    gat_by_class: dict[int, list[tuple[tuple, np.ndarray, list[dict]]]] = {}
    if gat is not None and gat_classes:
        groups: dict[int, dict[tuple, list[dict]]] = {}
        for item in _items(gat, "g"):
            class_id = item["semantic"]
            if class_id in gat_classes:
                groups.setdefault(class_id, {}).setdefault(item["key"], []).append(item)
        for class_id, per_key in groups.items():
            gat_by_class[class_id] = [
                (key, raster(group), group) for key, group in per_key.items()
            ]
    if gat_classes:
        for class_id in sorted(gat_classes):
            # 主引擎同类实例的掩码
            kept_masks = []
            primary_groups: dict[tuple, list[dict]] = {}
            for item in items:
                if item["semantic"] == class_id:
                    primary_groups.setdefault(identity(item), []).append(item)
            for group in primary_groups.values():
                kept_masks.append(raster(group))
            for _key, mask, group in gat_by_class.get(class_id, []):
                if not any(mask_iou(mask, existing) > iou_threshold
                           for existing in kept_masks):
                    kept_masks.append(mask)
                    items.extend(group)
    if segdet is not None and segdet_classes:
        items = [item for item in items if item["semantic"] not in segdet_classes]
        items.extend(item for item in _items(segdet, "s")
                     if item["semantic"] in segdet_classes)
    # 实例级门控：被门控类别的每个实例，必须落在 YOLO-seg 同类检测掩码内
    # 达到一定比例（默认 containment>0.1）。YOLO 的类 8 检测很准，而像素
    # 管线在无该类图纸上大量误报；全量实测 class 8 FP 2899→1272、TP 131→128，
    # 总 PQ +0.37，calib/holdout 两个半集方向一致。
    if gate_classes and support:
        import cv2
        height_px, width_px = support.get("size") or (raster_size, raster_size)
        scale_x = raster_size / max(float(width_px), 1.0)
        scale_y = raster_size / max(float(height_px), 1.0)
        support_masks: dict[int, list[np.ndarray]] = {}
        for raw_class, polygons in (support.get("polygons") or {}).items():
            class_id = int(raw_class)
            if class_id not in gate_classes:
                continue
            masks = []
            for polygon in polygons:
                if not polygon or len(polygon) < 3:
                    continue
                mask = np.zeros((raster_size, raster_size), np.uint8)
                points = np.asarray(polygon, dtype=np.float64)
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
                masks.append(mask.astype(bool))
            support_masks[class_id] = masks
        grouped: dict[tuple, list[dict]] = {}
        for item in items:
            if item["semantic"] in gate_classes:
                grouped.setdefault((item["semantic"], identity(item)), []).append(item)
        dropped: set[int] = set()
        for (class_id, _instance), group in grouped.items():
            mask = raster(group)
            area = int(np.count_nonzero(mask))
            best = 0.0
            for other in support_masks.get(class_id, []):
                intersection = int(np.count_nonzero(mask & other))
                if not intersection:
                    continue
                if gate_metric == "iou":
                    best = max(best, intersection /
                               max(int(np.count_nonzero(mask | other)), 1))
                else:
                    best = max(best, intersection /
                               max(min(area, int(np.count_nonzero(other))), 1))
            if best <= gate_threshold:
                dropped.update(id(item) for item in group)
        if dropped:
            items = [item for item in items if id(item) not in dropped]
    if suppress:
        items = [item for item in items if item["semantic"] not in suppress]
    # 与主引擎做实例级并集（保留两边，按掩码 IoU 去重）。ROI 细化器的
    # 实例用于 class 2/3/8/14/18/22/28：这些类别主引擎要么没有输出，
    # 要么召回低，并集比替换安全。
    if segdet is not None and segdet_union_classes:
        segdet_groups: dict[int, dict[tuple, list[dict]]] = {}
        for item in _items(segdet, "s"):
            if item["semantic"] in segdet_union_classes:
                segdet_groups.setdefault(item["semantic"], {}).setdefault(
                    item["key"], []).append(item)
        for class_id in sorted(segdet_union_classes):
            kept: list[tuple[np.ndarray, list[dict]]] = []
            primary_groups: dict[tuple, list[dict]] = {}
            for item in items:
                if item["semantic"] == class_id:
                    primary_groups.setdefault(identity(item), []).append(item)
            for group in primary_groups.values():
                kept.append((raster(group), group))
            for group in segdet_groups.get(class_id, {}).values():
                mask = raster(group)
                duplicate = None
                for index, (existing, existing_group) in enumerate(kept):
                    if mask_iou(mask, existing) > iou_threshold:
                        duplicate = index
                        break
                if duplicate is not None:
                    # 细化实例通常几何更准：重复时用细化实例替换主引擎实例。
                    dropped = {id(item) for item in kept[duplicate][1]}
                    items = [item for item in items if id(item) not in dropped]
                    kept.pop(duplicate)
                kept.append((mask, group))
                items.extend(group)
    # 实例几何质量门限：ROI 细化实例的 TP/FP 在墨迹长度、密度、长宽比上
    # 可分离（calib 选阈值、holdout 验证，两半集同向）。门限同时作用于
    # 主引擎与细化实例，直接裁剪输出以降低 FP。
    if quality_rules:
        features: dict[tuple[int, int], dict[str, float]] = {}
        grouped: dict[tuple, list[dict]] = {}
        for item in items:
            # 质量门控必须按完整来源键分组；只按 instance id 会把主引擎与
            # ROI 中恰好同号的实例拼在一起，导致密度/长度被稀释而误删。
            key = (item["semantic"], item["key"])
            grouped.setdefault(key, []).append(item)
        for key, group in grouped.items():
            xs = [point[0] for item in group for point in item["points"]]
            ys = [point[1] for item in group for point in item["points"]]
            lengths = [item["length"] for item in group]
            total = float(sum(lengths))
            width_box = max(xs) - min(xs)
            height_box = max(ys) - min(ys)
            features[key] = {
                "n": float(len(group)),
                "total": total,
                "mean": total / max(len(group), 1),
                "max": float(max(lengths)),
                "w": width_box,
                "h": height_box,
                "area": max(width_box * height_box, 1e-6),
                "aspect": min(width_box, height_box) /
                          max(width_box, height_box, 1e-6),
                "density": total / max(width_box + height_box, 1e-6),
            }
        dropped: set[int] = set()
        for key, group in grouped.items():
            rule = quality_rules.get(key[0])
            if not rule:
                continue
            feature, operator, threshold = rule
            value = features[key].get(feature)
            if value is None:
                continue
            keep = value >= threshold if operator == ">=" else value <= threshold
            if not keep:
                dropped.update(id(item) for item in group)
        if dropped:
            items = [item for item in items if id(item) not in dropped]
    return _assemble(items, width, height)


def merge_predictions(primary: dict[str, Any], secondary: dict[str, Any],
                      classes: set[int], iou_threshold: float = 0.5,
                      *, raster_mode: str = "legacy400") -> dict[str, Any]:
    """主引擎保留非目标类的图元；目标类取两个引擎的并集，按实例掩码
    IoU 去重（保留主引擎实例）。"""
    if not classes:
        return primary
    # Reuse source-aware, per-class suppression instead of a shared mask list
    # and a fixed ID offset (which could suppress another class or collide).
    return merge_final(primary, secondary, gat_classes=classes,
                       iou_threshold=iou_threshold, raster_mode=raster_mode)
