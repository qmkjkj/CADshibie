OCR转换脚本（文本提取）

```python
#!/usr/bin/env python3
"""Extract SVG text snippets and their native x/y anchors into per-image JSON."""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


TEXT_TAGS = {"text", "tspan", "textPath"}
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _first_number(value):
    if value is None:
        return None
    match = NUMBER_RE.search(value)
    if not match:
        return None
    result = float(match.group())
    if not math.isfinite(result):
        return None
    return result


def parse_svg_text_points(svg_source, image_width, image_height):
    """Parse bytes/path/string and retain each text tag's own SVG x/y values.

    Parent transforms are deliberately not applied. If a nested text tag omits
    x/y, the nearest ancestor text tag's anchor is inherited so its text is not
    lost. The original anchor numbers remain in the returned JSON.
    """
    if isinstance(svg_source, bytes):
        root = ET.fromstring(svg_source)
    elif isinstance(svg_source, Path):
        root = ET.parse(svg_source).getroot()
    elif isinstance(svg_source, str) and svg_source.lstrip().startswith("<"):
        root = ET.fromstring(svg_source)
    else:
        root = ET.parse(svg_source).getroot()

    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [_first_number(part) for part in re.split(r"[\s,]+", view_box.strip())]
        if len(values) != 4 or any(value is None for value in values):
            raise ValueError(f"Invalid SVG viewBox: {view_box!r}")
        view_box = values
    else:
        view_box = [0.0, 0.0,
                    _first_number(root.attrib.get("width")) or float(image_width),
                    _first_number(root.attrib.get("height")) or float(image_height)]

    annotations = []

    def text_subtree(element, inherited_x=None, inherited_y=None):
        """Collect positioned text runs in document order for one text root."""
        runs = []

        def walk(node, anchor_x, anchor_y, x_source, y_source):
            tag = local_name(node.tag)
            if tag in TEXT_TAGS:
                own_x = _first_number(node.attrib.get("x"))
                own_y = _first_number(node.attrib.get("y"))
                if own_x is not None:
                    anchor_x, x_source = own_x, "element"
                if own_y is not None:
                    anchor_y, y_source = own_y, "element"
                current_tag = tag
            else:
                current_tag = local_name(element.tag)

            if node.text and node.text.strip():
                runs.append((node.text, anchor_x, anchor_y, current_tag,
                             x_source, y_source))
            for child in node:
                walk(child, anchor_x, anchor_y, x_source, y_source)
                if child.tail and child.tail.strip():
                    runs.append((child.tail, anchor_x, anchor_y, current_tag,
                                 x_source, y_source))

        walk(element, inherited_x, inherited_y,
             "inherited" if inherited_x is not None else "missing",
             "inherited" if inherited_y is not None else "missing")
        merged = []
        for content, x, y, tag, x_source, y_source in runs:
            if merged and merged[-1]["x"] == x and merged[-1]["y"] == y:
                merged[-1]["text"] += content
            else:
                merged.append({"text": content, "x": x, "y": y, "source_tag": tag,
                               "x_source": x_source, "y_source": y_source})
        return merged

    # Find text roots and process each subtree once, preserving parent anchors.
    def find_roots(element, in_text=False, inherited_x=None, inherited_y=None):
        tag = local_name(element.tag)
        is_text = tag in TEXT_TAGS
        own_x = _first_number(element.attrib.get("x")) if is_text else None
        own_y = _first_number(element.attrib.get("y")) if is_text else None
        anchor_x = own_x if own_x is not None else inherited_x
        anchor_y = own_y if own_y is not None else inherited_y
        if is_text and not in_text:
            for item in text_subtree(element, inherited_x, inherited_y):
                item["text"] = " ".join(item["text"].split())
                if item["text"]:
                    item["has_coordinates"] = item["x"] is not None and item["y"] is not None
                    annotations.append(item)
            return
        for child in element:
            find_roots(child, in_text or is_text, anchor_x, anchor_y)

    find_roots(root)
    return {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "svg_viewBox": view_box,
        "coordinate_space": "svg_native",
        "anchor_definition": "SVG text insertion anchor; x/y are copied from SVG attributes; transforms are not applied",
        "annotations": annotations,
    }


def export_split(svg_dir, png_dir, output_dir):
    svg_dir, png_dir, output_dir = Path(svg_dir), Path(png_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"svg_files": 0, "with_text": 0, "text_entries": 0,
               "missing_coordinates": 0, "missing_png": [], "parse_errors": []}
    for svg_path in sorted(svg_dir.glob("*.svg")):
        summary["svg_files"] += 1
        png_path = png_dir / f"{svg_path.stem}.png"
        if not png_path.is_file():
            summary["missing_png"].append(svg_path.name)
            continue
        try:
            with Image.open(png_path) as image:
                width, height = image.size
            data = parse_svg_text_points(svg_path, width, height)
            data["image"] = png_path.name
            data["source_svg"] = svg_path.name
            out_path = output_dir / f"{svg_path.stem}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
            count = len(data["annotations"])
            summary["with_text"] += count > 0
            summary["text_entries"] += count
            summary["missing_coordinates"] += sum(not a["has_coordinates"]
                                                   for a in data["annotations"])
        except (ET.ParseError, OSError, ValueError) as exc:
            summary["parse_errors"].append({"file": svg_path.name, "error": str(exc)})
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, required=True)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = export_split(args.svg_dir, args.png_dir, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["parse_errors"] or summary["missing_png"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

指标评估

```
TP：预测点与真值点距离不超过阈值，且预测文本完全正确。
FP：没有匹配到真值的预测，包括多余预测、位置超出阈值的预测，以及文字识别错误的预测。
FN：没有被正确预测匹配到的真值，包括漏检，以及文字识别错误对应的真值。
```

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
端到端 F1 = 2 × Precision × Recall / (Precision + Recall)
          = 2TP / (2TP + FP + FN)
```

```
#!/usr/bin/env python3
"""Evaluate text anchor localization and end-to-end text spotting F1."""

import argparse
import json
import math
import unicodedata
from pathlib import Path

DEFAULT_THRESHOLDS = (5.0, 10.0, 20.0)


def _threshold_key(threshold):
    value = float(threshold)
    return str(int(value) if value.is_integer() else value)


def normalize_text(text):
    return unicodedata.normalize("NFC", str(text)).strip()


def _load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("image_width", "image_height", "svg_viewBox", "annotations"):
        if field not in data:
            raise ValueError(f"{path}: missing field {field!r}")
    if len(data["svg_viewBox"]) != 4:
        raise ValueError(f"{path}: svg_viewBox must have four numbers")
    data.setdefault("coordinate_space", "svg_native")
    if data["coordinate_space"] not in ("svg_native", "pixel"):
        raise ValueError(f"{path}: coordinate_space must be 'svg_native' or 'pixel'")
    return data


def _usable_annotations(data):
    valid, skipped = [], 0
    for annotation in data["annotations"]:
        x, y = annotation.get("x"), annotation.get("y")
        if x is None or y is None:
            skipped += 1
            continue
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            skipped += 1
            continue
        valid.append({**annotation, "x": x, "y": y})
    return valid, skipped


def _point_to_pixel(point, view_box, image_width, image_height, coordinate_space):
    min_x, min_y, view_width, view_height = map(float, view_box)
    if min(view_width, view_height, image_width, image_height) <= 0:
        raise ValueError("viewBox and image dimensions must be positive")
    if coordinate_space == "pixel":
        return float(point[0]), float(point[1])
    if coordinate_space != "svg_native":
        raise ValueError("coordinate space must be 'svg_native' or 'pixel'")
    return ((point[0] - min_x) * image_width / view_width,
            (point[1] - min_y) * image_height / view_height)


def point_distance_px(point_a, point_b, view_box, image_width, image_height,
                      coordinate_space_a="svg_native", coordinate_space_b="svg_native"):
    pixel_a = _point_to_pixel(point_a, view_box, image_width, image_height, coordinate_space_a)
    pixel_b = _point_to_pixel(point_b, view_box, image_width, image_height, coordinate_space_b)
    return math.hypot(pixel_a[0] - pixel_b[0], pixel_a[1] - pixel_b[1])


def match_points(gt, pred, view_box, image_width, image_height, threshold_px,
                 require_equal_text=False, gt_coordinate_space="svg_native",
                 pred_coordinate_space="svg_native"):
    """Maximum-cardinality, minimum-distance one-to-one radius matching."""
    if not gt or not pred:
        return []
    distances = []
    for i, truth in enumerate(gt):
        row = []
        for j, candidate in enumerate(pred):
            distance = point_distance_px(
                (truth["x"], truth["y"]), (candidate["x"], candidate["y"]),
                view_box, image_width, image_height,
                gt_coordinate_space, pred_coordinate_space)
            if require_equal_text and normalize_text(truth.get("text", "")) != normalize_text(
                    candidate.get("text", "")):
                distance = math.inf
            row.append(distance)
        distances.append(row)

    # Successive shortest augmenting paths on a bipartite residual graph.
    # Augment until no valid path remains: this maximizes match cardinality;
    # edge cost then minimizes total point distance for that cardinality.
    n_gt, n_pred = len(gt), len(pred)
    source, gt_base, pred_base, sink = 0, 1, 1 + n_gt, 1 + n_gt + n_pred
    graph = [[] for _ in range(sink + 1)]

    def add_edge(start, end, capacity, cost, pair=None):
        forward = [end, len(graph[end]), capacity, cost, pair]
        reverse = [start, len(graph[start]), 0, -cost, None]
        graph[start].append(forward)
        graph[end].append(reverse)

    for i in range(n_gt):
        add_edge(source, gt_base + i, 1, 0.0)
    for j in range(n_pred):
        add_edge(pred_base + j, sink, 1, 0.0)
    for i in range(n_gt):
        for j in range(n_pred):
            distance = distances[i][j]
            if distance <= threshold_px:
                add_edge(gt_base + i, pred_base + j, 1, distance, (i, j, distance))

    while True:
        dist = [math.inf] * len(graph)
        parent = [None] * len(graph)
        in_queue = [False] * len(graph)
        queue = [source]
        dist[source] = 0.0
        in_queue[source] = True
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                nxt, _, capacity, cost, _ = edge
                candidate = dist[node] + cost
                if capacity and candidate < dist[nxt] - 1e-12:
                    dist[nxt] = candidate
                    parent[nxt] = (node, edge_index)
                    if not in_queue[nxt]:
                        queue.append(nxt)
                        in_queue[nxt] = True
        if parent[sink] is None:
            break
        node = sink
        while node != source:
            previous, edge_index = parent[node]
            edge = graph[previous][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = previous

    matches = []
    for i in range(n_gt):
        for edge in graph[gt_base + i]:
            if edge[4] is not None and edge[2] == 0:
                matches.append((edge[4][0], edge[4][1], float(edge[4][2])))
    return sorted(matches)


def _f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_image(gt_data, pred_data, thresholds=DEFAULT_THRESHOLDS):
    if any(not math.isfinite(float(threshold)) or float(threshold) <= 0
           for threshold in thresholds):
        raise ValueError("pixel thresholds must be finite and positive")
    gt, skipped_gt = _usable_annotations(gt_data)
    pred, skipped_pred = _usable_annotations(pred_data)
    if skipped_pred:
        raise ValueError("prediction annotations must all have finite x/y coordinates")
    view_box = gt_data["svg_viewBox"]
    width, height = int(gt_data["image_width"]), int(gt_data["image_height"])
    if (int(pred_data["image_width"]) != width or
            int(pred_data["image_height"]) != height):
        raise ValueError("GT and prediction image dimensions differ")
    if any(abs(float(a) - float(b)) > 1e-8
           for a, b in zip(pred_data["svg_viewBox"], view_box)):
        raise ValueError("GT and prediction svg_viewBox differ")
    gt_space = gt_data.get("coordinate_space", "svg_native")
    pred_space = pred_data.get("coordinate_space", "svg_native")
    if gt_space not in ("svg_native", "pixel") or pred_space not in ("svg_native", "pixel"):
        raise ValueError("coordinate_space must be 'svg_native' or 'pixel'")

    results = {}
    for threshold in thresholds:
        matches = match_points(gt, pred, view_box, width, height, threshold,
                               gt_coordinate_space=gt_space,
                               pred_coordinate_space=pred_space)
        location_tp = len(matches)
        exact_matches = match_points(gt, pred, view_box, width, height, threshold,
                                     require_equal_text=True,
                                     gt_coordinate_space=gt_space,
                                     pred_coordinate_space=pred_space)
        exact_tp = len(exact_matches)
        results[_threshold_key(threshold)] = {
            "location": {"TP": location_tp, "FP": len(pred) - location_tp,
                         "FN": len(gt) - location_tp,
                         **_f1(location_tp, len(pred) - location_tp, len(gt) - location_tp)},
            "end_to_end": {"TP": exact_tp, "FP": len(pred) - exact_tp,
                           "FN": len(gt) - exact_tp,
                           **_f1(exact_tp, len(pred) - exact_tp, len(gt) - exact_tp)},
            "matched_text_correct": exact_tp,
            "matched_text_incorrect": location_tp - exact_tp,
            "mean_matched_distance_px": (sum(item[2] for item in matches) / location_tp
                                         if location_tp else None),
        }
    return {"gt_annotations": len(gt), "pred_annotations": len(pred),
            "skipped_gt_missing_coordinates": skipped_gt,
            "skipped_pred_missing_coordinates": skipped_pred,
            "thresholds": results}


def evaluate_dirs(gt_dir, pred_dir, thresholds=DEFAULT_THRESHOLDS):
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    gt_files = {p.name: p for p in gt_dir.glob("*.json") if p.name != "summary.json"}
    pred_files = {p.name: p for p in pred_dir.glob("*.json") if p.name != "summary.json"}
    missing = sorted(gt_files.keys() - pred_files.keys())
    extra = sorted(pred_files.keys() - gt_files.keys())
    if missing or extra:
        raise ValueError(f"JSON filename mismatch; missing predictions={missing[:10]}, "
                         f"extra predictions={extra[:10]}")

    per_image = {}
    totals = {_threshold_key(t): {
        "location": [0, 0, 0], "end_to_end": [0, 0, 0], "distance_sum": 0.0,
        "distance_count": 0, "correct_text": 0, "incorrect_text": 0}
        for t in thresholds}
    skipped_gt = skipped_pred = 0
    for name in sorted(gt_files):
        result = evaluate_image(_load(gt_files[name]), _load(pred_files[name]), thresholds)
        per_image[name] = result
        skipped_gt += result["skipped_gt_missing_coordinates"]
        skipped_pred += result["skipped_pred_missing_coordinates"]
        for threshold, metrics in result["thresholds"].items():
            total = totals[threshold]
            for key, index in (("TP", 0), ("FP", 1), ("FN", 2)):
                total["location"][index] += metrics["location"][key]
                total["end_to_end"][index] += metrics["end_to_end"][key]
            if metrics["mean_matched_distance_px"] is not None:
                count = metrics["location"]["TP"]
                total["distance_sum"] += metrics["mean_matched_distance_px"] * count
                total["distance_count"] += count
            total["correct_text"] += metrics["matched_text_correct"]
            total["incorrect_text"] += metrics["matched_text_incorrect"]

    summary = {}
    for threshold, total in totals.items():
        loc_tp, loc_fp, loc_fn = total["location"]
        e2e_tp, e2e_fp, e2e_fn = total["end_to_end"]
        summary[threshold] = {
            "location": {"TP": loc_tp, "FP": loc_fp, "FN": loc_fn,
                         **_f1(loc_tp, loc_fp, loc_fn)},
            "end_to_end": {"TP": e2e_tp, "FP": e2e_fp, "FN": e2e_fn,
                           **_f1(e2e_tp, e2e_fp, e2e_fn)},
            "matched_text_correct": total["correct_text"],
            "matched_text_incorrect": total["incorrect_text"],
            "mean_matched_distance_px": (total["distance_sum"] / total["distance_count"]
                                         if total["distance_count"] else None),
        }
    return {"images": len(per_image), "skipped_gt_missing_coordinates": skipped_gt,
            "skipped_pred_missing_coordinates": skipped_pred,
            "overall": summary, "per_image": per_image}


def evaluate_paths(gt_path, pred_path, thresholds=DEFAULT_THRESHOLDS):
    """Score one JSON pair or two directories of paired JSON files."""
    gt_path, pred_path = Path(gt_path), Path(pred_path)
    if gt_path.is_file() and pred_path.is_file():
        result = evaluate_image(_load(gt_path), _load(pred_path), thresholds)
        return {"images": 1,
                "skipped_gt_missing_coordinates": result["skipped_gt_missing_coordinates"],
                "skipped_pred_missing_coordinates": result["skipped_pred_missing_coordinates"],
                "overall": result["thresholds"], "per_image": {gt_path.name: result}}
    if gt_path.is_dir() and pred_path.is_dir():
        return evaluate_dirs(gt_path, pred_path, thresholds)
    raise ValueError("GT and prediction must both be JSON files or both be directories")


def format_console_summary(result):
    """Format only the fused end-to-end F1 as the primary reported metric."""
    lines = []
    for threshold, scores in result["overall"].items():
        row = scores["end_to_end"]
        lines.append(
            f"F1@{threshold}px: TP={row['TP']} FP={row['FP']} FN={row['FN']} "
            f"Precision={row['precision']:.6f} Recall={row['recall']:.6f} "
            f"F1={row['f1']:.6f}"
        )
    lines.append(
        f"Images={result['images']}; skipped missing-coordinate GT/pred="
        f"{result['skipped_gt_missing_coordinates']}/"
        f"{result['skipped_pred_missing_coordinates']}"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    result = evaluate_paths(args.gt_dir, args.pred_dir, tuple(args.thresholds))
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(format_console_summary(result))


if __name__ == "__main__":
    main()

```

运行命令：

```
python3 eval_text_point_f1.py \
  datasets/datasets_ori/dev_final_text_gt/pT476.json \
  datasets/datasets_ori/dev_final_text_pred_demo/pT476.json \
  --output datasets/datasets_ori/pT476_demo_ocr_metrics.json
```

### 提交规则

每张 `dev_final/<id>.png` 对应一个 UTF-8 JSON：`results_ocr/<id>.json`。`<id>` 必须与 PNG 的五位文件名完全一致。一个 JSON 表示一张图中的所有预测图元。`results_ocr/` 目录只放预测 JSON，不放 PNG、训练标签或映射表；
