"""Conservative instance translation against fixed, native-resolution PNG ink.

This is an optional route-A prototype. It changes geometry only; it never
creates, removes, relabels or regroups instances. Image agreement is an
acceptance signal, not a substitute for held-out PQ evaluation.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .raster import render_window
from .schema import validate_output


@dataclass(frozen=True)
class GeometryRefineConfig:
    max_shift_px: int = 2
    distance_cap_px: float = 2.0
    min_gain: float = 0.03
    ambiguity_margin: float = 0.01
    movement_penalty: float = 0.015
    min_support: float = 0.85
    min_primitive_support: float = 0.65
    tolerance_px: float = 1.0
    foreground_threshold: float = 0.08

    def __post_init__(self):
        if (not isinstance(self.max_shift_px, int)
                or not 0 <= self.max_shift_px <= 4):
            raise ValueError("max_shift_px must be an integer in [0, 4]")
        positive = (self.distance_cap_px, self.tolerance_px)
        nonnegative = (self.min_gain, self.ambiguity_margin, self.movement_penalty)
        fractions = (self.min_support, self.min_primitive_support,
                     self.foreground_threshold)
        if (not np.isfinite(positive + nonnegative + fractions).all()
                or min(positive) <= 0 or min(nonnegative) < 0
                or not all(0 <= value <= 1 for value in fractions)):
            raise ValueError("Invalid geometry refinement thresholds")


def image_ink(path, threshold: float = 0.08):
    """Recover visible ink, including colored and alpha-antialiased strokes."""
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    # Transparent RGB is not evidence. Using the darkest channel preserves
    # saturated yellow/red strokes that a luminance threshold might discard.
    coverage = rgba[..., 3] * (1.0 - rgba[..., :3].min(axis=2))
    foreground = coverage > threshold
    return foreground, skeletonize(foreground)


def _groups(data):
    groups = defaultdict(list)
    for index, (semantic, instance) in enumerate(zip(data["semanticIds"], data["instanceIds"])):
        if not 0 <= semantic < 30 or instance < 0:
            continue
        args = data["args"][index]
        groups[(semantic, instance)].append({
            "index": index, "command": data["commands"][index],
            "points": [(args[i], args[i + 1]) for i in range(0, 8, 2)],
        })
    return groups


def refine_geometry(data, foreground, centerline=None, *, config=None):
    """Return (output, decisions) using one bounded pass over frozen evidence.

    Pixel shifts are measured in the source PNG; canvas axes are mapped
    separately. A fixed evidence window and precomputed distance fields make
    every pose comparable. Integer translations preserve PIL raster phase.
    """
    config = config or GeometryRefineConfig()
    foreground = np.asarray(foreground)
    if foreground.ndim != 2 or foreground.dtype != np.bool_ or not foreground.size:
        raise ValueError("foreground must be a nonempty 2D boolean image")
    centerline = (skeletonize(np.not_equal(foreground, False))
                  if centerline is None else np.asarray(centerline))
    if centerline.shape != foreground.shape or centerline.dtype != np.bool_:
        raise ValueError("centerline must be boolean and match the source image")
    if np.any(centerline & ~foreground):
        raise ValueError("centerline must lie inside foreground")
    validate_output(data)
    height, width = foreground.shape
    sx, sy = width / data["width"], height / data["height"]
    groups = _groups(data)
    output = copy.deepcopy(data)
    decisions = []
    if not foreground.any() or not centerline.any() or config.max_shift_px == 0:
        return output, [{"reason": "empty_evidence" if not centerline.any() else "disabled",
                         "accepted": False, "class": key[0], "instance": key[1]}
                        for key in groups]
    image_dt = distance_transform_edt(~centerline)
    foreground_dt = distance_transform_edt(~foreground)
    # Frozen competing geometry prevents a translated instance from taking
    # over another existing object's ink. Stuff is intentionally excluded.
    occupancy = np.zeros((height, width), dtype=np.int32)
    bounds, original_masks = {}, {}
    pad = config.max_shift_px + int(np.ceil(config.distance_cap_px)) + 3
    for key, items in groups.items():
        points = np.asarray([p for item in items for p in item["points"]]) * (sx, sy)
        left, top = np.floor(points.min(axis=0)).astype(int) - pad
        right, bottom = np.ceil(points.max(axis=0)).astype(int) + pad + 1
        left, top = max(int(left), 0), max(int(top), 0)
        right, bottom = min(int(right), width), min(int(bottom), height)
        if left >= right or top >= bottom:
            continue
        bounds[key] = (left, top, right, bottom)
        mask = render_window(items, (bottom - top, right - left), (sx, sy), (left, top))
        original_masks[key] = mask
        occupancy[top:bottom, left:right] += mask

    accepted_masks = {}
    for key, items in groups.items():
        row = {"class": key[0], "instance": key[1], "accepted": False,
               "shift_px": [0, 0], "reason": "no_improvement"}
        decisions.append(row)
        if key not in bounds:
            row["reason"] = "outside_image"
            continue
        left, top, right, bottom = bounds[key]
        mask = original_masks[key]
        ys, xs = np.nonzero(mask)
        points = np.asarray([p for item in items for p in item["points"]]) * (sx, sy)
        if (len(xs) < 8 or np.any(points < 0)
                or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)):
            row["reason"] = "small_or_clipped_geometry"
            continue
        template_dt = distance_transform_edt(~mask)
        # Evidence is frozen from the ORIGINAL geometry, never selected by a
        # candidate. Distant walls cannot change the denominator per pose.
        evidence = (centerline[top:bottom, left:right]
                    & (template_dt <= config.max_shift_px + config.distance_cap_px))
        ey, ex = np.nonzero(evidence)
        if not len(ex):
            row["reason"] = "no_local_evidence"
            continue
        primitive_pixels = [np.nonzero(render_window(
            [item], mask.shape, (sx, sy), (left, top))) for item in items]
        local_dt = image_dt[top:bottom, left:right]
        local_fg_dt = foreground_dt[top:bottom, left:right]
        others = occupancy[top:bottom, left:right] - mask
        cap = config.distance_cap_px

        def assess(dx, dy):
            moved = points + (dx, dy)
            if (np.any(moved < 0) or np.any(moved[:, 0] >= width)
                    or np.any(moved[:, 1] >= height)):
                return None
            xx, yy = xs + dx, ys + dy
            if (xx.min() < 0 or yy.min() < 0 or xx.max() >= mask.shape[1]
                    or yy.max() >= mask.shape[0]):
                return None
            forward_values = local_dt[yy, xx]
            forward = float(np.minimum(forward_values / cap, 1.0).mean())
            rx, ry = ex - dx, ey - dy
            valid = ((rx >= 0) & (ry >= 0) & (rx < mask.shape[1]) & (ry < mask.shape[0]))
            reverse_values = np.full(len(ex), cap)
            reverse_values[valid] = template_dt[ry[valid], rx[valid]]
            backward = float(np.minimum(reverse_values / cap, 1.0).mean())
            supports = [float((local_dt[py + dy, px + dx] <= config.tolerance_px).mean())
                        for py, px in primitive_pixels if len(px)]
            cost = (forward + backward) / 2
            return {"dx": dx, "dy": dy, "forward": forward, "backward": backward,
                    "cost": cost, "objective": cost + config.movement_penalty * float(np.hypot(dx, dy)),
                    "support": float((forward_values <= config.tolerance_px).mean()),
                    "foreground_loss": float(np.minimum(local_fg_dt[yy, xx] / cap, 1.0).mean()),
                    "primitive_support": supports,
                    "conflict": float((others[yy, xx] > 0).mean())}

        baseline = assess(0, 0)
        row["before"] = baseline
        candidates = []
        for dy in range(-config.max_shift_px, config.max_shift_px + 1):
            for dx in range(-config.max_shift_px, config.max_shift_px + 1):
                candidate = assess(dx, dy)
                if candidate is not None:
                    candidates.append(candidate)
        candidates.sort(key=lambda c: (c["objective"], c["dx"] ** 2 + c["dy"] ** 2,
                                       c["dy"], c["dx"]))
        best = candidates[0]
        row["candidate"] = best
        dx, dy = best["dx"], best["dy"]
        if baseline["cost"] - best["objective"] < config.min_gain or (dx == 0 and dy == 0):
            continue
        if (best["forward"] > baseline["forward"] + 0.01
                or best["backward"] > baseline["backward"] + 0.01
                or best["foreground_loss"] > baseline["foreground_loss"] + 0.01
                or best["support"] < config.min_support
                or min(best["primitive_support"], default=0) < config.min_primitive_support
                or any(new < old - 0.02 for new, old in zip(
                    best["primitive_support"], baseline["primitive_support"]))):
            row["reason"] = "insufficient_ink_support"
            continue
        if best["conflict"] > baseline["conflict"] + 0.02:
            row["reason"] = "competing_instance"
            continue
        shifted = np.zeros_like(mask)
        shifted[ys + dy, xs + dx] = True
        ambiguous = False
        for other in candidates[1:]:
            if other["objective"] - best["objective"] > config.ambiguity_margin:
                break
            oy, ox = ys + other["dy"], xs + other["dx"]
            intersection = int(shifted[oy, ox].sum())
            # Endpoint-only differences on a long line are equivalent for this
            # ambiguity check; distinct competing shapes are not.
            iou = intersection / max(2 * len(xs) - intersection, 1)
            if iou < 0.95:
                ambiguous = True
                break
        if ambiguous:
            row["reason"] = "ambiguous_pose"
            continue
        moved_items = [{**item, "points": [((x * sx + dx) / sx, (y * sy + dy) / sy)
                                             for x, y in item["points"]]}
                       for item in items]
        if not np.array_equal(render_window(moved_items, mask.shape, (sx, sy),
                                             (left, top)), shifted):
            row["reason"] = "raster_phase_mismatch"
            continue
        row.update(accepted=True, reason="improved_fixed_ink", shift_px=[dx, dy])
        accepted_masks[key] = shifted

    # Two independently improved candidates may converge onto the same ink.
    # Resolve on frozen proposals by rejecting both, without order dependence.
    accepted_keys = list(accepted_masks)
    conflicts = set()
    for i, key in enumerate(accepted_keys):
        l1, t1, r1, b1 = bounds[key]
        for other in accepted_keys[i + 1:]:
            l2, t2, r2, b2 = bounds[other]
            l, t, r, b = max(l1, l2), max(t1, t2), min(r1, r2), min(b1, b2)
            if l >= r or t >= b:
                continue
            def overlap(masks):
                return int(np.count_nonzero(masks[key][t-t1:b-t1, l-l1:r-l1]
                                             & masks[other][t-t2:b-t2, l-l2:r-l2]))
            increase = overlap(accepted_masks) - overlap(original_masks)
            if increase > 0.02 * min(accepted_masks[key].sum(), accepted_masks[other].sum()):
                conflicts.update((key, other))
    for row in decisions:
        key = (row["class"], row["instance"])
        if key in conflicts:
            row.update(accepted=False, reason="converging_instances", shift_px=[0, 0])
        if not row["accepted"]:
            continue
        px, py = row["shift_px"]
        dx, dy = px / sx, py / sy
        for item in groups[key]:
            original = data["args"][item["index"]]
            output["args"][item["index"]] = [(value * sx + px) / sx if i % 2 == 0
                                             else (value * sy + py) / sy
                                             for i, value in enumerate(original)]
        for center, box in zip(output["obj_cts"], output["boxes"]):
            if center[3] == key[1]:
                center[0] += dx
                center[1] += dy
                box[0] += dx
                box[1] += dy
                box[2] += dx
                box[3] += dy
    validate_output(output)
    return output, decisions


def refine_image(data, png_path, *, config=None):
    config = config or GeometryRefineConfig()
    foreground, centerline = image_ink(png_path, config.foreground_threshold)
    return refine_geometry(data, foreground, centerline, config=config)
