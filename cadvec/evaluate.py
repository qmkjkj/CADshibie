"""本地 PQ / SQ / RQ 评估。

实现与赛题 README 第 7.3 节参考脚本保持一致：
- 只读取 semanticIds / instanceIds / commands / args；
- 类别 35 跳过；Thing（0-29）实例号为 -1 的图元跳过；
- Stuff（30-34）按类别合并为一个段；
- 栅格掩码 IoU **严格大于 0.5** 才可匹配；
- 每个类别内用匈牙利算法一对一匹配；
- 数据集级指标用累计 TP/FP/FN/ΣIoU 计算，而不是逐图平均。

该脚本仅用于本地验证，不参与正式评测。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment

from .raster import render_items

FIELDS = ("semanticIds", "instanceIds", "commands", "args")
IGNORE_ID = 35
THING_CLASS_COUNT = 30
MATCH_THRESHOLD = 0.5


class EvaluationError(ValueError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvaluationError(f"{path}: 根节点必须是对象")
    missing = [field for field in FIELDS if field not in data]
    if missing:
        raise EvaluationError(f"{path}: 缺少字段 {', '.join(missing)}")
    n = len(data["commands"])
    if any(len(data[field]) != n for field in FIELDS):
        raise EvaluationError(f"{path}: 四个图元字段长度必须一致")
    for semantic_id, instance_id, command, coordinates in zip(*(data[k] for k in FIELDS)):
        if not isinstance(command, int) or command not in (0, 1, 2, 3):
            raise EvaluationError(f"{path}: 非法 command {command!r}")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 8:
            raise EvaluationError(f"{path}: args 每行必须 8 个数")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value))
                   for value in coordinates):
            raise EvaluationError(f"{path}: args 含非有限数值")
        if not isinstance(semantic_id, int) or not 0 <= semantic_id <= IGNORE_ID:
            raise EvaluationError(f"{path}: 非法 semantic ID {semantic_id!r}")
        if not isinstance(instance_id, int) or instance_id < -1:
            raise EvaluationError(f"{path}: 非法 instance ID {instance_id!r}")
    return {field: data[field] for field in FIELDS}


def _segments(
    data: dict[str, Any],
    canvas_size: tuple[float, float],
    raster_size: int,
    dilation: int = 0,
) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    skipped = 0
    for class_id, instance_id, command, coordinates in zip(*(data[k] for k in FIELDS)):
        if class_id == IGNORE_ID:
            skipped += 1
            continue
        if class_id < THING_CLASS_COUNT:
            if instance_id == -1:
                skipped += 1
                continue
            key = (class_id, instance_id)
        else:
            key = (class_id, -1)
        points = [
            (coordinates[index], coordinates[index + 1])
            for index in range(0, 8, 2)
        ]
        groups[key].append({"command": command, "points": points})
    # Drawing independent primitives of one segment onto a binary PIL canvas
    # is exactly their union, avoiding one full image allocation per primitive.
    result = {key: render_items(items, raster_size, *canvas_size)
              for key, items in groups.items()}
    if dilation > 0:
        result = {
            key: binary_dilation(mask, iterations=dilation)
            for key, mask in result.items()
        }
    return result, skipped


def _metrics(tp: int, fp: int, fn: int, iou_sum: float) -> dict[str, float]:
    denominator = tp + 0.5 * (fp + fn)
    return {
        "PQ": iou_sum / denominator if denominator else 0.0,
        "SQ": iou_sum / tp if tp else 0.0,
        "RQ": tp / denominator if denominator else 0.0,
    }


def evaluate(
    gt: dict[str, Any],
    pred: dict[str, Any],
    canvas_size: tuple[float, float] = (100, 100),
    raster_size: int = 1000,
    dilation: int = 0,
) -> dict[str, Any]:
    """对单张图按同类别段 IoU > 0.5 做一对一匹配。

    ``dilation`` 默认 0，与参考脚本完全一致；本地诊断时可设为 1-3 像素，
    用来区分「几何位置错误」和「1 像素栅格化过细导致的 IoU 抖动」。
    """
    if min(*canvas_size, raster_size) <= 0:
        raise EvaluationError("canvas size 与 raster size 必须为正")
    gt_segments, skipped_gt = _segments(gt, canvas_size, raster_size, dilation)
    pred_segments, skipped_pred = _segments(pred, canvas_size, raster_size, dilation)
    classes = sorted({key[0] for key in gt_segments} | {key[0] for key in pred_segments})

    totals = {"TP": 0, "FP": 0, "FN": 0, "sum_iou": 0.0}
    per_class: dict[int, dict[str, Any]] = {}
    for class_id in classes:
        gt_masks = [mask for (label, _), mask in gt_segments.items() if label == class_id]
        pred_masks = [mask for (label, _), mask in pred_segments.items() if label == class_id]
        matches: list[float] = []
        if gt_masks and pred_masks:
            ious = np.zeros((len(gt_masks), len(pred_masks)), dtype=float)
            for gt_index, gt_mask in enumerate(gt_masks):
                for pred_index, pred_mask in enumerate(pred_masks):
                    union = np.count_nonzero(gt_mask | pred_mask)
                    if union:
                        ious[gt_index, pred_index] = (
                            np.count_nonzero(gt_mask & pred_mask) / union
                        )
            eligible = np.where(ious > MATCH_THRESHOLD, ious, 0.0)
            size = len(gt_masks) + len(pred_masks)
            weights = np.zeros((size, size), dtype=float)
            weights[: len(gt_masks), : len(pred_masks)] = eligible
            rows, columns = linear_sum_assignment(weights, maximize=True)
            matches = [
                float(ious[row, column])
                for row, column in zip(rows, columns)
                if row < len(gt_masks) and column < len(pred_masks)
                and ious[row, column] > MATCH_THRESHOLD
            ]
        tp = len(matches)
        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp
        iou_sum = float(sum(matches))
        per_class[class_id] = {
            "TP": tp, "FP": fp, "FN": fn, "sum_iou": iou_sum,
            **_metrics(tp, fp, fn, iou_sum),
        }
        for key, value in (("TP", tp), ("FP", fp), ("FN", fn), ("sum_iou", iou_sum)):
            totals[key] += value

    micro = _metrics(totals["TP"], totals["FP"], totals["FN"], totals["sum_iou"])
    macro = {
        name: float(np.mean([row[name] for row in per_class.values()])) if classes else 0.0
        for name in ("PQ", "SQ", "RQ")
    }
    return {
        **totals, "micro": micro, "macro": macro, "per_class": per_class,
        "skipped_gt": skipped_gt, "skipped_pred": skipped_pred,
    }


def _empty_totals() -> dict[str, int | float]:
    return {"TP": 0, "FP": 0, "FN": 0, "sum_iou": 0.0}


def _accumulate(
    destination: dict[str, int | float], source: dict[str, Any],
) -> None:
    for key in ("TP", "FP", "FN", "sum_iou"):
        destination[key] += source[key]


def evaluate_directories(
    gt_dir: str | Path,
    pred_dir: str | Path,
    raster_size: int = 1000,
    *,
    canvas: tuple[float, float] | None = None,
    dilation: int = 0,
) -> dict[str, Any]:
    """逐图评估后按数据集累计量汇总。"""
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)
    if not gt_path.is_dir() or not pred_path.is_dir():
        raise EvaluationError("gt_dir 和 pred_dir 必须是目录")
    gt_files = sorted(gt_path.glob("*.json"), key=lambda path: path.name)
    if not gt_files:
        raise EvaluationError(f"{gt_path} 中没有 JSON")
    gt_names = {path.name for path in gt_files}
    pred_names = {path.name for path in pred_path.glob("*.json") if not path.name.startswith("_")}
    missing = sorted(gt_names - pred_names)
    extra = sorted(pred_names - gt_names)
    if missing:
        raise EvaluationError(f"预测缺少 {len(missing)} 个文件，例如 {missing[:3]}")
    if extra:
        raise EvaluationError(f"预测多出 {len(extra)} 个文件，例如 {extra[:3]}")

    totals = _empty_totals()
    per_class_acc: dict[int, dict[str, Any]] = {}
    skipped_gt = skipped_pred = 0
    per_image: dict[str, Any] = {}
    for gt_file in gt_files:
        gt = load_json(gt_file)
        pred = load_json(pred_path / gt_file.name)
        if canvas is None:
            raw_gt = json.loads(gt_file.read_text(encoding="utf-8"))
            width = float(raw_gt.get("width", 100))
            height = float(raw_gt.get("height", 100))
            canvas_size = (width, height)
        else:
            canvas_size = canvas
        result = evaluate(gt, pred, canvas_size, raster_size, dilation)
        per_image[gt_file.stem] = {
            "TP": result["TP"], "FP": result["FP"], "FN": result["FN"],
            "sum_iou": result["sum_iou"], **result["micro"],
        }
        _accumulate(totals, result)
        skipped_gt += result["skipped_gt"]
        skipped_pred += result["skipped_pred"]
        for class_id, row in result["per_class"].items():
            bucket = per_class_acc.setdefault(
                class_id, {**_empty_totals(), "images": 0},
            )
            _accumulate(bucket, row)
            bucket["images"] += 1

    per_class = {}
    for class_id, bucket in sorted(per_class_acc.items()):
        per_class[class_id] = {
            "TP": bucket["TP"], "FP": bucket["FP"], "FN": bucket["FN"],
            "sum_iou": bucket["sum_iou"], "images": bucket["images"],
            **_metrics(
                int(bucket["TP"]), int(bucket["FP"]), int(bucket["FN"]),
                float(bucket["sum_iou"]),
            ),
        }
    micro = _metrics(
        int(totals["TP"]), int(totals["FP"]), int(totals["FN"]),
        float(totals["sum_iou"]),
    )
    macro = {
        name: float(np.mean([row[name] for row in per_class.values()])) if per_class else 0.0
        for name in ("PQ", "SQ", "RQ")
    }
    return {
        "images": len(gt_files),
        **{key: int(totals[key]) if key != "sum_iou" else float(totals[key])
           for key in ("TP", "FP", "FN", "sum_iou")},
        "micro": micro, "macro": macro, "per_class": per_class,
        "per_image": per_image,
        "skipped_gt": skipped_gt, "skipped_pred": skipped_pred,
        "raster_size": raster_size, "dilation": dilation,
    }


def _format_metrics(label: str, values: dict[str, float]) -> str:
    return (
        f"{label}: PQ={values['PQ'] * 100:.3f}% "
        f"SQ={values['SQ'] * 100:.3f}% RQ={values['RQ'] * 100:.3f}%"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地 PQ/SQ/RQ 评估（参考脚本口径）")
    parser.add_argument("--gt", type=Path, help="单图真值 JSON")
    parser.add_argument("--pred", type=Path, help="单图预测 JSON")
    parser.add_argument("--gt-dir", type=Path, help="真值 JSON 目录")
    parser.add_argument("--pred-dir", type=Path, help="预测 JSON 目录")
    parser.add_argument("--canvas-width", type=float, default=None)
    parser.add_argument("--canvas-height", type=float, default=None)
    parser.add_argument("--raster-size", type=int, default=1000)
    parser.add_argument(
        "--dilate", type=int, default=0,
        help="本地诊断用：栅格掩码膨胀像素数；0 表示严格对齐参考脚本",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="完整结果写入文件")
    args = parser.parse_args(argv)

    if not args.raster_size > 0:
        parser.error("--raster-size 必须为正")
    if args.dilate < 0:
        parser.error("--dilate 必须为非负整数")
    canvas = None
    if args.canvas_width is not None or args.canvas_height is not None:
        if args.canvas_width is None or args.canvas_height is None:
            parser.error("--canvas-width 与 --canvas-height 需同时提供")
        canvas = (args.canvas_width, args.canvas_height)

    if args.gt_dir or args.pred_dir:
        if not args.gt_dir or not args.pred_dir:
            parser.error("--gt-dir 与 --pred-dir 需同时提供")
        result = evaluate_directories(
            args.gt_dir, args.pred_dir, args.raster_size,
            canvas=canvas, dilation=args.dilate,
        )
        print(
            f"images={result['images']} TP={result['TP']} FP={result['FP']} "
            f"FN={result['FN']}; skipped GT/pred={result['skipped_gt']}/"
            f"{result['skipped_pred']}"
        )
        for label in ("micro", "macro"):
            print(_format_metrics(label, result[label]))
        for class_id, row in result["per_class"].items():
            print(
                f"class {class_id:02d}: TP={row['TP']} FP={row['FP']} FN={row['FN']} "
                f"PQ={row['PQ'] * 100:.3f}% SQ={row['SQ'] * 100:.3f}% "
                f"RQ={row['RQ'] * 100:.3f}%"
            )
    else:
        if not args.gt or not args.pred:
            parser.error("需要 --gt/--pred 或 --gt-dir/--pred-dir")
        gt_data = load_json(args.gt)
        pred_data = load_json(args.pred)
        canvas_size = canvas or (100, 100)
        if canvas is None:
            raw = json.loads(args.gt.read_text(encoding="utf-8"))
            canvas_size = (float(raw.get("width", 100)), float(raw.get("height", 100)))
        result = evaluate(
            gt_data, pred_data, canvas_size, args.raster_size, args.dilate,
        )
        print(
            f"TP={result['TP']} FP={result['FP']} FN={result['FN']}; "
            f"skipped GT/pred={result['skipped_gt']}/{result['skipped_pred']}"
        )
        for label in ("micro", "macro"):
            print(_format_metrics(label, result[label]))
        for class_id, row in result["per_class"].items():
            print(
                f"class {class_id:02d}: TP={row['TP']} FP={row['FP']} FN={row['FN']} "
                f"PQ={row['PQ'] * 100:.3f}% SQ={row['SQ'] * 100:.3f}% "
                f"RQ={row['RQ'] * 100:.3f}%"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
