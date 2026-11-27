"""与正式评分器一致的几何栅格化（PIL、1px、Command 2/3 按外接椭圆）。

去重、门控和质量诊断都应使用同一渲染语义；训练可用软代理，但决策与验收
不应依赖不同的栅格器。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw


def _points(item, sx: float, sy: float):
    return [(float(x) * sx, float(y) * sy) for x, y in item["points"]]


def render_items(items, raster: int = 1000, canvas_width: float = 100.0,
                 canvas_height: float = 100.0) -> np.ndarray:
    sx = raster / max(float(canvas_width), 1e-6)
    sy = raster / max(float(canvas_height), 1e-6)
    return render_window(items, (raster, raster), (sx, sy))


def render_window(items, shape: tuple[int, int], scale_xy=(1.0, 1.0),
                  origin_xy=(0, 0)) -> np.ndarray:
    """Render a crop without changing global PIL pixel phase or primitive types.

    ``origin_xy`` is an integer pixel offset in the full raster, not a new
    canvas coordinate origin. This also supports rectangular source images.
    """
    height, width = shape
    sx, sy = scale_xy
    ox, oy = origin_xy
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for item in items:
        points = [(x - ox, y - oy) for x, y in _points(item, sx, sy)]
        if int(item.get("command", 0)) in (0, 1):
            draw.line(points, fill=1, width=1, joint="curve")
        else:
            xs, ys = zip(*points)
            draw.ellipse((min(xs), min(ys), max(xs), max(ys)),
                         outline=1, width=1)
    # PIL mode "1" exposes true bytes as 255. Normalize to standard 0/1
    # storage before passing these masks to native image kernels.
    return np.not_equal(np.asarray(image), 0)


def render_data(data: dict, raster: int = 1000,
                canvas_width: float | None = None,
                canvas_height: float | None = None):
    """按 (semantic, instance) 返回掩码；Stuff 按类别合并（instance=-1）。"""
    width = float(canvas_width if canvas_width is not None
                  else data.get("width", 100))
    height = float(canvas_height if canvas_height is not None
                   else data.get("height", 100))
    groups = defaultdict(list)
    for index in range(len(data["commands"])):
        semantic = int(data["semanticIds"][index])
        instance = int(data["instanceIds"][index])
        key = (semantic, -1 if semantic >= 30 else instance)
        points = data["args"][index]
        groups[key].append({
            "command": int(data["commands"][index]),
            "points": [(float(points[k]), float(points[k + 1]))
                       for k in range(0, 8, 2)],
        })
    return {key: render_items(items, raster, width, height)
            for key, items in groups.items()}
