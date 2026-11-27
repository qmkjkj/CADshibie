"""Experimental local RootSIFT registration; not part of default inference.

Matrices map source to target. Acceptance only establishes a feature candidate;
it does not verify symbol identity or whether transferred geometry is present.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropFrame:
    offset_xy: tuple[float, float] = (0.0, 0.0)
    # Resized-crop pixels per full-image pixel, using pixel-center coordinates.
    scale_xy: tuple[float, float] = (1.0, 1.0)
    native_from_image: np.ndarray | None = None


def _gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 image")
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected grayscale, RGB, or RGBA")
    rgb = image[..., :3]
    if image.shape[2] == 4:
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        rgb = np.rint(rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


def _affine(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
        raise ValueError("Expected an invertible 3x3 affine matrix")
    return matrix


def _local_to_image(frame):
    offset = np.asarray(frame.offset_xy, dtype=np.float64)
    scale = np.asarray(frame.scale_xy, dtype=np.float64)
    if (offset.shape != (2,) or scale.shape != (2,) or not np.isfinite(offset).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid crop offset or resize scale")
    if not np.isclose(scale[0], scale[1], rtol=1e-6, atol=0):
        raise ValueError("Similarity registration requires uniform crop resize scales")
    # cv2.resize maps pixel centers: original=(resized+0.5)/scale-0.5.
    shift = offset + 0.5 / scale - 0.5
    return np.array([[1 / scale[0], 0, shift[0]],
                     [0, 1 / scale[1], shift[1]], [0, 0, 1]])


def _features(sift, gray):
    keypoints, desc = sift.detectAndCompute(gray, None)
    if desc is not None:
        desc = desc.astype(np.float32)
        desc = np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-12))
    return keypoints, desc


def _ratio_matches(matcher, first, second, ratio):
    return {pair[0].queryIdx: pair[0].trainIdx
            for pair in matcher.knnMatch(first, second, k=2)
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}


def _coverage(points, shape):
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull) / (shape[0] * shape[1]))


def match_local(source, target, *, source_frame=CropFrame(), target_frame=CropFrame(),
                ratio=0.75, ransac_px=2.5, min_inliers=3,
                min_inlier_ratio=0.4, min_coverage=0.02):
    """Return candidate transforms and rejection diagnostics, never predictions.

    ransac_px is in target crop pixels. Native transforms require both explicit
    image-to-native matrices; no canvas or viewBox is guessed from the PNG size.
    CropFrame describes actual integer crop bounds followed by uniform resize,
    not an arbitrary warpAffine preprocessing step. Pad before resizing when
    necessary. Thresholds are experimental defaults, not calibrated settings.
    """
    values = [ratio, ransac_px, min_inliers, min_inlier_ratio, min_coverage]
    if (not np.isfinite(values).all() or not 0 < ratio < 1 or ransac_px <= 0
            or not isinstance(min_inliers, int) or min_inliers < 3
            or not 0 <= min_inlier_ratio <= 1 or not 0 <= min_coverage <= 1):
        raise ValueError("Invalid feature matching thresholds")
    source_to_image, target_to_image = _local_to_image(source_frame), _local_to_image(target_frame)
    native_source, native_target = source_frame.native_from_image, target_frame.native_from_image
    if (native_source is None) != (native_target is None):
        raise ValueError("Supply both native transforms or neither")
    if native_source is not None:
        native_source, native_target = _affine(native_source), _affine(native_target)
    source_gray, target_gray = _gray(source), _gray(target)
    sift = cv2.SIFT_create(nfeatures=2000)
    source_keys, source_desc = _features(sift, source_gray)
    target_keys, target_desc = _features(sift, target_gray)
    stats = {"source_keypoints": len(source_keys), "target_keypoints": len(target_keys)}

    def failure(reason):
        return {"ok": False, "reason": reason, "stats": stats,
                "local_matrix": None, "image_matrix": None, "native_matrix": None}

    if source_desc is None or target_desc is None or min(len(source_keys), len(target_keys)) < 3:
        return failure("insufficient_keypoints")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = _ratio_matches(matcher, source_desc, target_desc, ratio)
    reverse = _ratio_matches(matcher, target_desc, source_desc, ratio)
    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    stats.update(ratio_matches=len(forward), mutual_matches=len(pairs))
    if len(pairs) < min_inliers:
        return failure("insufficient_mutual_matches")
    src = np.float32([source_keys[i].pt for i, _ in pairs])
    dst = np.float32([target_keys[j].pt for _, j in pairs])
    matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_px,
                                             maxIters=3000, confidence=0.995, refineIters=10)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return failure("ransac_failed")
    keep = mask.ravel().astype(bool)
    count = int(keep.sum())
    stats.update(inliers=count, inlier_ratio=count / len(pairs))
    if count < min_inliers or stats["inlier_ratio"] < min_inlier_ratio:
        return failure("insufficient_inliers")
    stats["source_coverage"] = _coverage(src[keep], source_gray.shape)
    stats["target_coverage"] = _coverage(dst[keep], target_gray.shape)
    if min(stats["source_coverage"], stats["target_coverage"]) < min_coverage:
        return failure("insufficient_spatial_coverage")
    local = np.vstack([matrix, [0, 0, 1]])
    if abs(np.linalg.det(local[:2, :2])) < 1e-12:
        return failure("degenerate_transform")
    image = target_to_image @ local @ np.linalg.inv(source_to_image)
    native = native_target @ image @ np.linalg.inv(native_source) if native_source is not None else None
    residual = np.linalg.norm((src @ matrix[:, :2].T + matrix[:, 2]) - dst, axis=1)
    stats["median_inlier_error_px"] = float(np.median(residual[keep]))
    return {"ok": True, "reason": None, "stats": stats,
            "local_matrix": local[:2].tolist(), "image_matrix": image[:2].tolist(),
            "native_matrix": native[:2].tolist() if native is not None else None}
