"""Image evidence for experimental template candidates, never a PQ estimate."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .raster import render_items


def inspect_ink_support(items, foreground, *, canvas_width=100.0,
                        canvas_height=100.0, tolerance_px=1.5, reverse_band_px=3.0):
    """Describe supported and unsupported geometry using the PIL 1px renderer.

    Each primitive is rendered separately; no line connects adjacent primitives.
    Reverse distances use only nearby image ink, not the entire bounding box.
    Nearby neighboring symbols can still support a wrong candidate. Callers
    must separately verify identity, competing instances, and coordinate frames.
    This function has no calibrated acceptance threshold and does not edit items.
    """
    foreground = np.asarray(foreground)
    if (foreground.dtype != np.bool_ or foreground.ndim != 2 or not foreground.size
            or foreground.shape[0] != foreground.shape[1]):
        raise ValueError("Expected a square boolean foreground raster")
    values = [canvas_width, canvas_height, tolerance_px, reverse_band_px]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Canvas dimensions and pixel tolerances must be positive and finite")
    items = list(items)
    outside = 0
    for item in items:
        points = np.asarray(item["points"], dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Each primitive must have four finite sampled points")
        if item.get("command", 0) not in (0, 1, 2, 3):
            raise ValueError("Unknown primitive command")
        outside += int(((points[:, 0] < 0) | (points[:, 0] > canvas_width)
                        | (points[:, 1] < 0) | (points[:, 1] > canvas_height)).sum())
    raster = foreground.shape[0]

    def render(primitives):
        return render_items(primitives, raster, canvas_width, canvas_height)

    template = render(items)
    count = int(template.sum())
    result = {"has_ground_truth": False, "primitive_count": len(items),
              "template_pixels": count, "foreground_pixels": int(foreground.sum()),
              "outside_point_count": outside, "tolerance_px": float(tolerance_px),
              "reverse_band_px": float(reverse_band_px), "support_fraction": None,
              "mean_distance_px": None, "p90_distance_px": None,
              "maximum_primitive_unsupported_fraction": None,
              "nearby_image_pixels": 0, "nearby_image_mean_distance_px": None}
    if not count or not foreground.any():
        result["reason"] = "empty_template" if not count else "empty_foreground"
        return result
    image_distance = distance_transform_edt(~foreground)
    distances = image_distance[template]
    result.update(reason=None, support_fraction=float((distances <= tolerance_px).mean()),
                  mean_distance_px=float(distances.mean()),
                  p90_distance_px=float(np.quantile(distances, 0.9)))
    unsupported = []
    for item in items:
        mask = render([item])
        if mask.any():
            unsupported.append(float((image_distance[mask] > tolerance_px).mean()))
    result["maximum_primitive_unsupported_fraction"] = max(unsupported, default=None)
    template_distance = distance_transform_edt(~template)
    nearby = foreground & (template_distance <= reverse_band_px)
    result["nearby_image_pixels"] = int(nearby.sum())
    if nearby.any():
        result["nearby_image_mean_distance_px"] = float(template_distance[nearby].mean())
    return result
