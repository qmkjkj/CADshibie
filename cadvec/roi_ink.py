"""ROI 实例墨迹细化（GPU 可选，CPU 可运行）。

YOLO-seg 只提供实例位置（框/掩码），本模块在检测框内用一个小 U-Net 预测
该实例的精确墨迹，再把墨迹骨架切成短折线输出。相比“把框内整条笔画收进
实例”，它能保留门/窗符号本体、排除穿框墙线。

训练数据：train_final 的 GT 实例框 + 按评测器一致方式栅格化的实例墨迹。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.morphology import remove_small_objects, skeletonize

from .skeleton import trace_paths

CROP_SIZE = 192
PAD = 0.30
CHUNK = 6
MIN_LENGTH = 0.2


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class RoiInkUNet(nn.Module):
    def __init__(self, classes: int = 36, base: int = 32, class_dim: int = 64,
                 use_anchor: bool = False):
        super().__init__()
        self.use_anchor = bool(use_anchor)
        self.class_embed = nn.Sequential(
            nn.Linear(classes, class_dim), nn.SiLU(),
            nn.Linear(class_dim, base * 8),
        )
        self.anchor_embed = None
        if self.use_anchor:
            self.anchor_embed = nn.Sequential(
                nn.Linear(4, class_dim), nn.SiLU(),
                nn.Linear(class_dim, base * 8),
            )
        self.enc1 = _conv_block(3, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, image: torch.Tensor, class_onehot: torch.Tensor,
                anchor: torch.Tensor | None = None) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        condition = self.class_embed(class_onehot)
        if self.anchor_embed is not None:
            if anchor is None:
                anchor = torch.zeros((image.shape[0], 4), device=image.device)
            condition = condition + self.anchor_embed(anchor)
        e4 = e4 + condition.view(-1, e4.shape[1], 1, 1)
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def _resample_four(points: np.ndarray) -> list[list[float]]:
    if len(points) < 2:
        return []
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return []
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


def _point_segment_distance(point, first, second) -> float:
    vector = second - first
    norm = float(vector @ vector)
    if norm <= 1e-12:
        return float(np.linalg.norm(point - first))
    ratio = float((point - first) @ vector) / norm
    ratio = min(max(ratio, 0.0), 1.0)
    return float(np.linalg.norm(point - (first + ratio * vector)))


def _four_point_error(points: np.ndarray, four: list[list[float]]) -> float:
    corners = np.asarray(four, dtype=np.float64)
    best = 0.0
    for point in points:
        distance = min(_point_segment_distance(point, corners[index],
                                               corners[index + 1])
                       for index in range(3))
        best = max(best, distance)
    return best


def _adaptive_pieces(points: np.ndarray, eps: float,
                     max_depth: int = 12) -> list[np.ndarray]:
    """按四点折线与原路径的最大偏差递归切分；直线只保留一段。"""
    stack = [(points, 0)]
    pieces = []
    while stack:
        piece, depth = stack.pop()
        if len(piece) <= 4 or depth >= max_depth:
            pieces.append(piece)
            continue
        four = _resample_four(piece)
        if not four or _four_point_error(piece, four) <= eps:
            pieces.append(piece)
            continue
        errors = np.asarray([min(_point_segment_distance(
            point, np.asarray(four[index]), np.asarray(four[index + 1]))
            for index in range(3)) for point in piece])
        split = int(np.argmax(errors))
        split = min(max(split, 2), len(piece) - 2)
        stack.append((piece[split:], depth + 1))
        stack.append((piece[:split + 1], depth + 1))
    return pieces


def vectorize_mask(mask: np.ndarray, left: int, top: int,
                   crop_width: int, crop_height: int,
                   scale_x: float, scale_y: float,
                   threshold: float = 0.5,
                   chunk: int = CHUNK,
                   adaptive: bool = False,
                   eps: float = 0.10) -> list[tuple[list[list[float]], float]]:
    """把某一裁剪区域内的概率掩码转成若干 4 点短折线（画布坐标）。

    ``chunk`` 是每条折线最多使用的骨架点数；4 点重采样前保留更长路径可以
    减少碎片，但会增大弯曲处的四点误差。默认值保持与旧权重训练时一致。
    """
    local = remove_small_objects(mask > threshold, min_size=8, connectivity=2)
    if local.sum() < 8:
        return []
    skeleton = skeletonize(local)
    if not skeleton.any():
        return []
    step_x = crop_width / float(mask.shape[1])
    step_y = crop_height / float(mask.shape[0])
    segments = []
    chunk = max(int(chunk), 2)
    for path in trace_paths(skeleton, min_pixels=2):
        points = np.asarray(
            [((left + (column + 0.5) * step_x) / scale_x,
              (top + (row + 0.5) * step_y) / scale_y)
             for row, column in path], dtype=np.float64)
        if adaptive:
            pieces = _adaptive_pieces(points, eps)
        else:
            pieces = [points[start:start + chunk + 1]
                      for start in range(0, len(points) - 1, chunk)]
        for piece in pieces:
            if len(piece) < 2:
                continue
            length = float(np.linalg.norm(np.diff(piece, axis=0), axis=1).sum())
            if length < MIN_LENGTH:
                continue
            four = _resample_four(piece)
            if four:
                segments.append((four, length))
    return segments


class RoiInkRefiner:
    """加载训练好的 ROI 墨迹模型，把检测框转成精确实例图元。"""

    def __init__(self, model: RoiInkUNet, device: torch.device,
                 threshold: float = 0.5, max_instances: int = 15,
                 crop_size: int = CROP_SIZE, quality_model: dict | None = None,
                 native_pad: bool = False):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.max_instances = max_instances
        self.crop_size = int(crop_size)
        self.quality_model = quality_model
        self.native_pad = bool(native_pad)
        self.use_anchor = bool(getattr(model, "use_anchor", False))
        self.onehot = torch.eye(36, device=device)
        self.native_detail = None

    @classmethod
    def load(cls, path: str | Path, device: str | None = "auto",
             max_instances: int = 15,
             quality_path: str | Path | None = None) -> "RoiInkRefiner | None":
        if not str(path).strip():
            return None
        path = Path(path)
        if not path.exists():
            return None
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = RoiInkUNet(**checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device).eval()
        quality_model = None
        if quality_path is not None and Path(quality_path).exists():
            try:
                quality_model = json.loads(
                    Path(quality_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                quality_model = None
        return cls(model, torch.device(device),
                   threshold=float(checkpoint.get("threshold", 0.5)),
                   max_instances=max_instances,
                   crop_size=int(checkpoint.get("crop_size", CROP_SIZE)),
                   quality_model=quality_model,
                   native_pad=bool(checkpoint.get("native_pad", False)))

    def _crop(self, box, width: int, height: int):
        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = side * (0.5 + PAD)
        left = int(round(max(cx - half, 0)))
        top = int(round(max(cy - half, 0)))
        right = int(round(min(cx + half, width)))
        bottom = int(round(min(cy + half, height)))
        return left, top, max(right, left + 4), max(bottom, top + 4)

    FEATURES = (
        "conf", "prob_mean", "prob_max", "prob_p90", "mask_fraction",
        "n_seg", "total_len", "mean_len", "max_len", "w", "h",
        "aspect", "density", "log_box_area", "ink_box_ratio",
    )

    def _features(self, class_id: int, confidence: float, det_box, segments,
                  probability: np.ndarray, scale_x: float, scale_y: float,
                  canvas_width: int, canvas_height: int) -> dict[str, float]:
        mask = probability > self.threshold
        if mask.any():
            values = probability[mask]
            prob_mean = float(values.mean())
            prob_p90 = float(np.percentile(values, 90))
        else:
            prob_mean = prob_p90 = 0.0
        prob_max = float(probability.max()) if probability.size else 0.0
        mask_fraction = float(mask.mean()) if mask.size else 0.0
        lengths = [length for _points, length in segments]
        total = float(sum(lengths))
        xs = [point[0] for points, _length in segments for point in points]
        ys = [point[1] for points, _length in segments for point in points]
        width = max(xs) - min(xs) if xs else 0.0
        height = max(ys) - min(ys) if ys else 0.0
        x0, y0, x1, y1 = det_box
        box_w = max((x1 - x0) / scale_x, 1e-6)
        box_h = max((y1 - y0) / scale_y, 1e-6)
        box_area = box_w * box_h
        return {
            "conf": float(confidence),
            "prob_mean": prob_mean,
            "prob_max": prob_max,
            "prob_p90": prob_p90,
            "mask_fraction": mask_fraction,
            "n_seg": float(len(segments)),
            "total_len": total,
            "mean_len": total / max(len(segments), 1),
            "max_len": max(lengths) if lengths else 0.0,
            "w": width,
            "h": height,
            "aspect": min(width, height) / max(width, height, 1e-6),
            "density": total / max(width + height, 1e-6),
            "log_box_area": math.log(max(box_area, 1e-6)),
            "ink_box_ratio": (width * height) / max(box_area, 1e-6),
        }

    def _quality_score(self, class_id: int, features: dict[str, float]) -> float | None:
        if not self.quality_model:
            return None
        spec = (self.quality_model.get("classes") or {}).get(str(class_id))
        if not spec:
            return None
        names = self.quality_model.get("features") or self.FEATURES
        z = float(spec.get("intercept", 0.0))
        for name, coef, mean, scale in zip(names, spec["coef"], spec["mean"], spec["scale"]):
            z += float(coef) * (features.get(name, 0.0) - float(mean)) / max(float(scale), 1e-9)
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))

    @torch.no_grad()
    def analyze(self, image: np.ndarray, boxes: list[tuple[int, float, tuple]],
                width: int, height: int, canvas_width: int = 100,
                canvas_height: int = 100, *, include_empty: bool = False) -> list[dict]:
        """返回每个检测框的细化实例、概率统计与几何特征（不做 NMS/上限）。"""
        if not boxes:
            return []
        crops, meta = [], []
        for class_id, confidence, det_box in boxes:
            left, top, right, bottom = self._crop(det_box, width, height)
            crop_width = right - left
            crop_height = bottom - top
            if self.native_pad:
                valid_w, valid_h = crop_width, crop_height
                if valid_w > self.crop_size or valid_h > self.crop_size:
                    ratio = min(self.crop_size / valid_w,
                                self.crop_size / valid_h)
                    valid_w = max(1, int(round(valid_w * ratio)))
                    valid_h = max(1, int(round(valid_h * ratio)))
                    crop = cv2.resize(
                        image[top:bottom, left:right], (valid_w, valid_h),
                        interpolation=cv2.INTER_AREA)
                else:
                    crop = image[top:bottom, left:right]
                canvas = np.full((self.crop_size, self.crop_size, 3), 255,
                                 dtype=np.uint8)
                canvas[:valid_h, :valid_w] = crop
                crops.append(canvas)
                meta.append((class_id, confidence, det_box, left, top,
                             right, bottom, crop_width, crop_height,
                             valid_w, valid_h))
            else:
                crop = cv2.resize(image[top:bottom, left:right],
                                  (self.crop_size, self.crop_size),
                                  interpolation=cv2.INTER_AREA)
                crops.append(crop)
                meta.append((class_id, confidence, det_box, left, top,
                             right, bottom))
        batch = torch.from_numpy(np.stack(crops)).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        class_ids = torch.tensor([item[0] for item in meta], device=self.device)
        anchor_tensor = None
        if self.use_anchor:
            values = []
            for item in meta:
                _class_id, _conf, det_box = item[0], item[1], item[2]
                x0, y0, x1, y1 = det_box
                values.append([
                    (x0 + x1) / 2.0 / max(width, 1),
                    (y0 + y1) / 2.0 / max(height, 1),
                    (x1 - x0) / max(width, 1),
                    (y1 - y0) / max(height, 1),
                ])
            anchor_tensor = torch.tensor(values, dtype=torch.float32,
                                         device=self.device)
        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.model(batch, self.onehot[class_ids], anchor_tensor)
        probabilities = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        scale_x = width / max(canvas_width, 1)
        scale_y = height / max(canvas_height, 1)
        results = []
        for item, probability in zip(meta, probabilities):
            (class_id, confidence, det_box, left, top, right, bottom) = item[:7]
            if self.native_pad:
                crop_width, crop_height, valid_w, valid_h = item[7:11]
                resized = probability[:valid_h, :valid_w]
                if (valid_w, valid_h) != (crop_width, crop_height):
                    resized = cv2.resize(
                        resized, (crop_width, crop_height),
                        interpolation=cv2.INTER_LINEAR)
            else:
                crop_width = right - left
                crop_height = bottom - top
                resized = cv2.resize(probability, (crop_width, crop_height),
                                     interpolation=cv2.INTER_LINEAR)
            segments = vectorize_mask(
                resized, left, top, crop_width, crop_height,
                scale_x, scale_y, self.threshold)
            if not segments and not include_empty:
                continue
            results.append({
                "class_id": class_id,
                "confidence": confidence,
                "segments": [
                    {
                        "command": 0,
                        "points": points,
                        "polyline": np.asarray(points, dtype=np.float64),
                        "length": float(length),
                        "rgb": [0, 0, 0],
                        "layer": 0,
                        "width": 1.0,
                    }
                    for points, length in segments
                ],
                "features": self._features(
                    class_id, confidence, det_box, segments, resized,
                    scale_x, scale_y, canvas_width, canvas_height),
                "probability": resized,
                "box": (left, top, right, bottom),
                "det_box": det_box,
            })
        return results

    @torch.no_grad()
    def refine(self, image: np.ndarray, boxes: list[tuple[int, float, tuple]],
               width: int, height: int, canvas_width: int = 100,
               canvas_height: int = 100) -> list[tuple[int, float, list[dict]]]:
        """boxes: [(class_id, confidence, (x0,y0,x1,y1))]（像素坐标）。

        返回 [(class_id, confidence, segments)]，segments 与 pipeline_segdet
        的图元字典一致。
        """
        analyzed = self.analyze(image, boxes, width, height,
                                canvas_width, canvas_height)

        def box_iou(first, second):
            x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
            x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
            if x1 <= x0 or y1 <= y0:
                return 0.0
            inter = (x1 - x0) * (y1 - y0)
            area_a = (first[2] - first[0]) * (first[3] - first[1])
            area_b = (second[2] - second[0]) * (second[3] - second[1])
            return inter / max(area_a + area_b - inter, 1e-9)

        results = []
        counts: dict[int, int] = {}
        kept_boxes: dict[int, list[tuple]] = {}
        for item in analyzed:
            class_id = int(item["class_id"])
            if self.max_instances > 0 and counts.get(class_id, 0) >= self.max_instances:
                continue
            box = item["box"]
            if any(box_iou(box, other) > 0.5
                   for other in kept_boxes.get(class_id, [])):
                continue
            quality = self._quality_score(class_id, item["features"])
            if quality is not None:
                spec = self.quality_model["classes"][str(class_id)]
                if quality < float(spec.get("threshold", 0.5)):
                    continue
            kept_boxes.setdefault(class_id, []).append(box)
            counts[class_id] = counts.get(class_id, 0) + 1
            segments = item["segments"]
            if self.native_detail is not None:
                left, top, right, bottom = item["box"]
                prompt_args = {}
                if getattr(self.native_detail, "uses_instance_prompt", False):
                    x0,y0,x1,y1 = item["det_box"]
                    prompt_args["det_box"] = (x0-left,y0-top,x1-left,y1-top)
                probability = self.native_detail.correct(
                    image[top:bottom, left:right], item["probability"], class_id, **prompt_args)
                native_segments = vectorize_mask(
                    probability, left, top, right-left, bottom-top,
                    width/max(canvas_width,1), height/max(canvas_height,1), self.threshold)
                if native_segments:
                    segments = [{"command":0, "points":points, "polyline":np.asarray(points),
                                 "length":float(length), "rgb":[0,0,0], "layer":0,"width":1.}
                                for points,length in native_segments]
            results.append((class_id, float(item["confidence"]), segments))
        return results

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """训练用 soft Dice 损失（推理不调用）。"""
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()
