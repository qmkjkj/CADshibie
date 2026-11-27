## 工程表格检测数据集与评估

本节介绍本目录下 工程表格 数据集，以及用于比较模型预测结果和真实标注的评估脚本 [`evaluate.py`](TD-TSR_60-20-20_五位编号/evaluate.py)。学生可以直接运行该脚本，也可以阅读脚本了解匹配和指标计算过程。这部分是**目标检测/表格结构识别**任务；

通过网盘分享的文件：工程表格.zip
链接: https://pan.baidu.com/s/1CFqaIcwl6bu55GcmtmUlXA?pwd=dhx2 提取码: dhx2

### 数据集内容

图像来自建筑结构工程图，主要是钢筋明细表等工程表格。数据集包含以下划分：

| 划分 | 图片数 | 标注数 |
|---|---:|---:|
| train | 404 | 55,902 |
| dev | 135 | 20,251 |
| test | 135 | 18,430 |

图片位于 `images/{train,test}`，标签位于 `annotations/{train,test}.json`。图像在各划分内按五位数字命名，例如 `00001.png`。JSON 顶层只有 `categories` 和 `annotations`；每条标注中的 `image_id` 是图片文件名。

类别 ID 定义如下：

| category_id | 类别 | 含义 |
|---:|---|---|
| 1 | `table` | 整张表格区域 |
| 2 | `row` | 表格中的一行 |
| 3 | `cell` | 表格中的单元格 |
| 4 | `header` | 表头区域 |
| 5 | `column` | 表格中的一列 |

行、列、单元格、表头和整张表格是不同类别，空间上可能互相包含或重叠。模型预测类别必须使用与 `categories` 一致的 ID。

### 标注与预测 JSON

真实 JSON 中的标注框采用 COCO 的 `[x, y, width, height]` 格式，坐标是原图像素坐标：

```json
{
  "categories": [
    {"id": 1, "name": "table", "supercategory": ""}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": "00001.png",
      "category_id": 3,
      "bbox": [100.0, 200.0, 80.0, 40.0]
    }
  ]
}
```

预测 JSON 使用相同的顶层字段和框格式；每个预测框还必须带 `score`，表示模型置信度，范围为 0 到 1：

```json
{
  "categories": [
    {"id": 1, "name": "table", "supercategory": ""}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": "00001.png",
      "category_id": 3,
      "bbox": [101.0, 199.0, 81.0, 39.0],
      "score": 0.93
    }
  ]
}
```

真实和预测 JSON 的类别列表必须一致。若模型在缩放图像上推理，评估前要先将预测框坐标还原到原图像素坐标。

### 指标与匹配规则

匹配时先按图片 `image_id` 和 `category_id` 分组，再计算预测框与真实框的 IoU（交并比）。一个预测最多匹配一个真实框，一个真实框也最多被匹配一次；在同图、同类别且尚未匹配的真值中，选择 IoU 最大的候选。IoU 达到阈值时计为 TP；没有合格真值可匹配的预测计为 FP；未被匹配的真值计为 FN。类别错误的框不会匹配到真实目标：错误类别产生 FP，真实类别目标仍是 FN。

脚本输出：

- `AP`：在 IoU 0.50 到 0.95（每次增加 0.05）分别计算 AP，再对有真实标注的类别和 IoU 阈值取平均。
- `AP50`、`AP75`：分别在 IoU 0.50、0.75 下计算各类别 AP 后取平均。
- `precision`、`recall`、`f1`：在默认 `score >= 0.5` 且 `IoU >= 0.5` 时统计 TP/FP/FN 后计算。可用 `--score`、`--iou` 调整这两个阈值。这组阈值不会截断 AP 的预测排序计算。
- `per_class`：各类别的 AP、TP/FP/FN 和分类别 Precision/Recall/F1。
- `per_image`：逐图指标、成功匹配的真值/预测 ID 及 IoU、未匹配真值和预测 ID。

其中 `Precision = TP / (TP + FP)`，`Recall = TP / (TP + FN)`，`F1 = 2 × Precision × Recall / (Precision + Recall)`。`AP` 是 Precision-Recall 曲线下的面积近似；`mAP` 是多个类别 AP 的平均值。

### 运行评估

进入该数据集目录后，只需给出真实 JSON、预测 JSON 和结果文件路径；不需要把图片目录传给评估脚本。评估所有 JSON 中出现的图片：

```bash
python evaluate.py \
  --gt annotations/dev.json \
  --pred predictions/dev_predictions.json \
  --out metrics/dev_metrics.json
```

只评估一张图片时，真值和预测 JSON 都按同一个文件名筛选：

```bash
python evaluate.py \
  --gt annotations/dev.json \
  --pred predictions/dev_predictions.json \
  --image 00001.png \
  --out metrics/00001_metrics.json \
  --iou 0.5 \
  --score 0.5
```

运行 `python evaluate.py --help` 可查看全部参数。该脚本只依赖 Python 标准库。评估输出 JSON 包含总体指标、分类别指标和逐图匹配明细。

---

### 脚本

```
#!/usr/bin/env python3
"""Evaluate COCO-style table-detection JSON files (ground truth and predictions)."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
PRF_IOU_THRESHOLD = 0.50
PRF_SCORE_THRESHOLD = 0.50


def _load_json(path, label):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON must be an object")
    for field in ("categories", "annotations"):
        if field not in data:
            raise ValueError(f"{label} JSON is missing {field!r}")
    if not isinstance(data["categories"], list) or not isinstance(data["annotations"], list):
        raise ValueError(f"{label} JSON categories and annotations must be arrays")
    return data


def _category_map(categories, label):
    result = {}
    for category in categories:
        if not isinstance(category, dict) or "id" not in category or "name" not in category:
            raise ValueError(f"Each {label} category must contain id and name")
        category_id = category["id"]
        if category_id in result:
            raise ValueError(f"Duplicate category id in {label}: {category_id}")
        result[category_id] = str(category["name"])
    if not result:
        raise ValueError(f"{label} categories cannot be empty")
    return result


def _validate_annotations(annotations, category_ids, label, require_score=False):
    for index, annotation in enumerate(annotations):
        context = f"{label} annotation at index {index}"
        if not isinstance(annotation, dict):
            raise ValueError(f"{context} must be an object")
        for field in ("image_id", "category_id", "bbox"):
            if field not in annotation:
                raise ValueError(f"{context} is missing {field!r}")
        if not isinstance(annotation["image_id"], str) or not annotation["image_id"]:
            raise ValueError(f"{context} image_id must be a non-empty filename string")
        if annotation["category_id"] not in category_ids:
            raise ValueError(f"{context} has unknown category_id {annotation['category_id']!r}")
        bbox = annotation["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"{context} bbox must be [x, y, width, height]")
        try:
            box = [float(value) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} bbox must contain numeric values") from exc
        if not all(math.isfinite(value) for value in box):
            raise ValueError(f"{context} bbox values must be finite")
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"{context} bbox width and height must be positive")
        annotation["bbox"] = box
        if require_score:
            if "score" not in annotation:
                raise ValueError(f"{context} is missing prediction score")
            try:
                score = float(annotation["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context} score must be numeric") from exc
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{context} score must be finite and in [0, 1]")
            annotation["score"] = score


def iou_xywh(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _match_image(gt_annotations, pred_annotations, iou_threshold, score_threshold=None):
    predictions = [
        prediction for prediction in pred_annotations
        if score_threshold is None or prediction["score"] >= score_threshold
    ]
    predictions.sort(key=lambda prediction: -prediction.get("score", 1.0))
    matched_gt = set()
    matches, unmatched_predictions = [], []
    for prediction in predictions:
        best_iou, best_index = 0.0, None
        for index, truth in enumerate(gt_annotations):
            if index in matched_gt or truth["category_id"] != prediction["category_id"]:
                continue
            overlap = iou_xywh(prediction["bbox"], truth["bbox"])
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_index)
            matches.append({
                "gt_id": gt_annotations[best_index].get("id"),
                "prediction_id": prediction.get("id"),
                "category_id": prediction["category_id"],
                "iou": best_iou,
                "score": prediction.get("score"),
            })
        else:
            unmatched_predictions.append(prediction.get("id"))
    unmatched_gt = [
        truth.get("id") for index, truth in enumerate(gt_annotations)
        if index not in matched_gt
    ]
    return matches, unmatched_predictions, unmatched_gt


def _average_precision(tp_flags, fp_flags, num_gt):
    if num_gt == 0:
        return None
    if not tp_flags:
        return 0.0
    tp_total = fp_total = 0
    recalls, precisions = [], []
    for tp, fp in zip(tp_flags, fp_flags):
        tp_total += tp
        fp_total += fp
        recalls.append(tp_total / num_gt)
        precisions.append(tp_total / (tp_total + fp_total))
    # COCO-style 101-point interpolated AP.
    interpolated = []
    for recall_threshold_index in range(101):
        recall_threshold = recall_threshold_index / 100
        eligible = [
            precision for recall, precision in zip(recalls, precisions)
            if recall >= recall_threshold
        ]
        interpolated.append(max(eligible, default=0.0))
    return sum(interpolated) / 101


def _class_ap(gt_annotations, pred_annotations, image_names, category_id, iou_threshold):
    gt_by_image = defaultdict(list)
    for truth in gt_annotations:
        if truth["category_id"] == category_id:
            gt_by_image[truth["image_id"]].append(truth)
    num_gt = sum(len(items) for items in gt_by_image.values())
    if num_gt == 0:
        return None, 0, sum(p["category_id"] == category_id for p in pred_annotations)

    detections = [p for p in pred_annotations if p["category_id"] == category_id]
    detections.sort(key=lambda prediction: -prediction["score"])
    matched = {name: set() for name in image_names}
    tp_flags, fp_flags = [], []
    for prediction in detections:
        candidates = gt_by_image[prediction["image_id"]]
        best_iou, best_index = 0.0, None
        for index, truth in enumerate(candidates):
            if index in matched[prediction["image_id"]]:
                continue
            overlap = iou_xywh(prediction["bbox"], truth["bbox"])
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_index is not None and best_iou >= iou_threshold:
            matched[prediction["image_id"]].add(best_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    return _average_precision(tp_flags, fp_flags, num_gt), num_gt, len(detections)


def _precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(gt_data, prediction_data, iou_threshold=PRF_IOU_THRESHOLD,
             score_threshold=PRF_SCORE_THRESHOLD):
    """Evaluate two COCO-style objects containing categories and annotations."""
    if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be finite and in [0, 1]")
    if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("score threshold must be finite and in [0, 1]")

    gt_categories = _category_map(gt_data["categories"], "ground truth")
    pred_categories = _category_map(prediction_data["categories"], "prediction")
    if gt_categories != pred_categories:
        raise ValueError("Ground-truth and prediction category id/name mappings differ")
    category_ids = set(gt_categories)
    gt_annotations = gt_data["annotations"]
    predictions = prediction_data["annotations"]
    _validate_annotations(gt_annotations, category_ids, "ground truth")
    _validate_annotations(predictions, category_ids, "prediction", require_score=True)

    # The supplied aggregate GT file has no top-level `images` list. Infer the
    # evaluated image universe from its annotations; this dataset has GT on all
    # 135 images. Predictions for any other filename are rejected.
    image_names = sorted({truth["image_id"] for truth in gt_annotations})
    if not image_names:
        raise ValueError("Ground truth contains no annotations; image names cannot be inferred")
    unknown_images = sorted({p["image_id"] for p in predictions} - set(image_names))
    if unknown_images:
        raise ValueError(f"Predictions contain image_id absent from ground truth: {unknown_images[:10]}")

    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    for truth in gt_annotations:
        gt_by_image[truth["image_id"]].append(truth)
    for prediction in predictions:
        pred_by_image[prediction["image_id"]].append(prediction)

    per_image = {}
    total_tp = total_fp = total_fn = 0
    for image_name in image_names:
        matches, unmatched_predictions, unmatched_gt = _match_image(
            gt_by_image[image_name], pred_by_image[image_name], iou_threshold,
            score_threshold,
        )
        tp, fp, fn = len(matches), len(unmatched_predictions), len(unmatched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image[image_name] = {
            "gt_count": len(gt_by_image[image_name]),
            "prediction_count": sum(
                prediction["score"] >= score_threshold
                for prediction in pred_by_image[image_name]
            ),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **_precision_recall_f1(tp, fp, fn),
            "matches": matches,
            "unmatched_prediction_ids": unmatched_predictions,
            "unmatched_gt_ids": unmatched_gt,
        }

    per_class = {}
    ap_by_iou = defaultdict(list)
    for category_id, category_name in gt_categories.items():
        class_gt = [ann for ann in gt_annotations if ann["category_id"] == category_id]
        class_pred = [ann for ann in predictions if ann["category_id"] == category_id]
        class_tp = class_fp = class_fn = 0
        for image_name in image_names:
            matches, unmatched_predictions, unmatched_gt = _match_image(
                [ann for ann in gt_by_image[image_name]
                 if ann["category_id"] == category_id],
                [ann for ann in pred_by_image[image_name]
                 if ann["category_id"] == category_id],
                iou_threshold, score_threshold,
            )
            class_tp += len(matches)
            class_fp += len(unmatched_predictions)
            class_fn += len(unmatched_gt)

        class_result = {
            "ground_truth_count": len(class_gt),
            "prediction_count": len(class_pred),
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            **_precision_recall_f1(class_tp, class_fp, class_fn),
            "iou_threshold_for_prf": iou_threshold,
            "score_threshold_for_prf": score_threshold,
        }
        for threshold in IOU_THRESHOLDS:
            ap, num_gt, num_pred = _class_ap(
                gt_annotations, predictions, image_names, category_id, threshold
            )
            class_result[f"AP{int(round(threshold * 100))}"] = ap
            if ap is not None:
                ap_by_iou[threshold].append(ap)
        per_class[category_name] = class_result

    ap50_values = [result["AP50"] for result in per_class.values()
                   if result["AP50"] is not None]
    ap75_values = [result["AP75"] for result in per_class.values()
                   if result["AP75"] is not None]
    all_ap = [ap for values in ap_by_iou.values() for ap in values]
    summary = {
        "images": len(image_names),
        "ground_truth_annotations": len(gt_annotations),
        "prediction_annotations": len(predictions),
        # COCO-style AP: mean over categories with GT and IoU thresholds 0.50:0.05:0.95.
        "AP": sum(all_ap) / len(all_ap) if all_ap else None,
        "AP50": sum(ap50_values) / len(ap50_values) if ap50_values else None,
        "AP75": sum(ap75_values) / len(ap75_values) if ap75_values else None,
        **_precision_recall_f1(total_tp, total_fp, total_fn),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "iou_threshold_for_prf": iou_threshold,
        "score_threshold_for_prf": score_threshold,
    }
    return {"summary": summary, "per_class": per_class, "per_image": per_image}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_json", help="COCO-style ground-truth JSON")
    parser.add_argument("pred_json", help="COCO-style prediction JSON")
    args = parser.parse_args(argv)
    try:
        gt_data = _load_json(args.gt_json, "ground truth")
        prediction_data = _load_json(args.pred_json, "prediction")
        result = evaluate(gt_data, prediction_data)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

```

### 提交规则

每张 `dev/<id>.png` 对应一个 UTF-8 JSON：`results_table/<id>.json`。`<id>` 必须与 PNG 的五位文件名完全一致。一个 JSON 表示一张图中的所有预测结果。`results_table/` 目录只放预测 JSON，不放 PNG、训练标签或映射表；
