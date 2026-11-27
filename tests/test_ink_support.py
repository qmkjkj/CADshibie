import numpy as np
from PIL import Image, ImageDraw
import pytest

from cadvec.ink_support import inspect_ink_support


def line(x1, y1, x2, y2):
    return {"command": 0, "points": [[x1, y1], [x1, y1], [x2, y2], [x2, y2]]}


def foreground(*segments):
    image = Image.new("1", (1000, 1000), 0)
    draw = ImageDraw.Draw(image)
    for segment in segments:
        draw.line(segment, fill=1, width=1)
    return np.asarray(image, dtype=bool)


def test_disconnected_primitives_do_not_gain_an_artificial_bridge():
    ink = foreground((100, 100, 200, 100), (800, 800, 900, 800))
    items = [line(10, 10, 20, 10), line(80, 80, 90, 80)]
    result = inspect_ink_support(items, ink)
    assert result["template_pixels"] == 202
    assert result["support_fraction"] == 1 and result["mean_distance_px"] == 0
    assert result["maximum_primitive_unsupported_fraction"] == 0
    assert result["has_ground_truth"] is False


def test_unsupported_primitive_is_reported_even_when_other_ink_is_correct():
    ink = foreground((100, 100, 900, 100))
    result = inspect_ink_support([line(10, 10, 90, 10), line(50, 60, 51, 60)], ink)
    assert result["support_fraction"] > 0.95
    assert result["support_fraction"] < 1
    assert result["maximum_primitive_unsupported_fraction"] == 1


def test_distant_background_walls_do_not_count_as_unexplained_instance_ink():
    ink = foreground((100, 100, 200, 100), (0, 500, 999, 500))
    result = inspect_ink_support([line(10, 10, 20, 10)], ink)
    assert result["foreground_pixels"] == 1101
    assert result["nearby_image_pixels"] == 101
    assert result["nearby_image_mean_distance_px"] == 0


def test_empty_image_does_not_create_a_positive_support_score():
    result = inspect_ink_support([line(10, 10, 20, 10)], np.zeros((100, 100), bool))
    assert result["reason"] == "empty_foreground" and result["support_fraction"] is None
    with pytest.raises(ValueError, match="boolean"):
        inspect_ink_support([], np.zeros((100, 100), np.uint8))
