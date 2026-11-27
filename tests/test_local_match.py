import cv2
import numpy as np
import pytest

from cadvec.local_match import CropFrame, _local_to_image, match_local


def textured_image():
    rng = np.random.default_rng(17)
    image = np.full((280, 320), 255, np.uint8)
    for x, y, radius in rng.integers([20, 20, 2], [300, 260, 9], size=(90, 3)):
        cv2.circle(image, (int(x), int(y)), int(radius), 0, 1)
    cv2.putText(image, "CAD 738", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return image


def test_known_similarity_recovers_points_and_composes_coordinate_frames():
    source = textured_image()
    known = cv2.getRotationMatrix2D((160, 140), 8, 0.93)
    known[:, 2] += [5, -3]
    target = cv2.warpAffine(source, known, (320, 280), borderValue=255)
    source_frame = CropFrame((100, 200), (2, 2), np.array([[0.1, 0, 3], [0, 0.1, 4], [0, 0, 1]]))
    target_frame = CropFrame((30, 70), (0.8, 0.8), np.array([[0.2, 0, -2], [0, 0.2, 1], [0, 0, 1]]))
    result = match_local(source, target, source_frame=source_frame, target_frame=target_frame)
    assert result["ok"], result
    points = np.array([[50, 50, 1], [200, 200, 1], [150, 80, 1]])
    expected = points @ known.T
    actual = points @ np.asarray(result["local_matrix"]).T
    assert np.max(np.linalg.norm(actual - expected, axis=1)) < 0.8
    source_image = points @ _local_to_image(source_frame).T
    actual_image = source_image @ np.asarray(result["image_matrix"]).T
    expected_image = np.column_stack([expected, np.ones(3)]) @ _local_to_image(target_frame).T
    assert np.max(np.linalg.norm(actual_image - expected_image[:, :2], axis=1)) < 1
    source_native = source_image @ source_frame.native_from_image.T
    actual_native = source_native @ np.asarray(result["native_matrix"]).T
    expected_native = expected_image @ target_frame.native_from_image.T
    assert np.max(np.linalg.norm(actual_native - expected_native[:, :2], axis=1)) < 0.2


def test_white_and_transparent_images_fail_without_inventing_transforms():
    white = np.full((100, 100, 3), 255, np.uint8)
    transparent = np.zeros((100, 100, 4), np.uint8)
    result = match_local(transparent, white)
    assert not result["ok"] and result["reason"] == "insufficient_keypoints"
    assert result["native_matrix"] is None and result["local_matrix"] is None


def test_spatially_concentrated_features_are_rejected():
    image = np.full((1000, 1000), 255, np.uint8)
    image[20:160, 20:180] = cv2.resize(textured_image(), (160, 140))
    result = match_local(image, image, min_coverage=0.1)
    assert not result["ok"] and result["reason"] == "insufficient_spatial_coverage"


def test_frames_require_valid_scales_and_explicit_paired_native_transforms():
    source = textured_image()
    with pytest.raises(ValueError, match="scale"):
        match_local(source, source, source_frame=CropFrame(scale_xy=(0, 1)))
    with pytest.raises(ValueError, match="both native"):
        match_local(source, source, source_frame=CropFrame(native_from_image=np.eye(3)))
    frame = CropFrame(offset_xy=(10, 20), scale_xy=(2, 2))
    assert np.allclose(_local_to_image(frame) @ [0, 0, 1], [9.75, 19.75, 1])


def test_actual_crop_and_resize_recovers_the_full_image_transform():
    source = textured_image()
    known = cv2.getRotationMatrix2D((160, 140), 4, 1.04)
    known[:, 2] += [50, 10]
    target = cv2.warpAffine(source, known, (400, 380), borderValue=255)
    source_crop = cv2.resize(source[20:250, 20:300], (560, 460))
    target_crop = cv2.resize(target[20:290, 40:360], (480, 405))
    result = match_local(source_crop, target_crop,
                         source_frame=CropFrame((20, 20), (2, 2)),
                         target_frame=CropFrame((40, 20), (1.5, 1.5)))
    assert result["ok"], result
    points = np.array([[70, 60, 1], [180, 140, 1], [240, 180, 1]])
    actual = points @ np.asarray(result["image_matrix"]).T
    assert np.max(np.linalg.norm(actual - points @ known.T, axis=1)) < 1


def test_nonuniform_resize_cannot_silently_use_a_similarity_model():
    source = textured_image()
    with pytest.raises(ValueError, match="uniform"):
        match_local(source, source, source_frame=CropFrame(scale_xy=(2, 1)))
