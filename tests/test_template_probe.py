import numpy as np
import pytest

from tools.probe_local_templates import (
    audit_sources, native_from_image, square_crop, support_metrics, transformed_items,
    replace_instances,
)


def test_training_sources_are_checked_by_full_source_group():
    with pytest.raises(ValueError, match="source_overlap"):
        audit_sources(["train"], ["dev"], {"train": "same", "dev": "same"})
    assert audit_sources(["train"], ["dev"], {"train": "a", "dev": "b"})["overlap_groups"] == []


def test_native_similarity_keeps_every_primitive_and_explicit_viewbox_origin():
    matrix = native_from_image((10, 20, 100, 50), (1000, 500))
    np.testing.assert_allclose(matrix @ [500, 250, 1], [60, 45, 1])
    items = [{"command": 0, "points": [[i, 0], [i, 1], [i, 2], [i, 3]],
              "width": .1, "length": 3} for i in range(73)]
    moved = transformed_items(items, [[0, -2, 5], [2, 0, 7]])
    assert len(moved) == 73 and moved[-1]["points"][0] == [5, 151]
    assert moved[-1]["length"] == 6
    with pytest.raises(ValueError, match="similarity"):
        transformed_items(items, [[2, 1, 0], [0, 2, 0]])


def test_png_support_rejects_unsupported_geometry_and_crop_is_uniform():
    image = np.full((100, 100, 3), 255, np.uint8)
    image[20, 10:51] = 0
    items = [{"command": 0, "points": [[10, 20], [20, 20], [30, 20], [50, 20]]}]
    good = support_metrics(items, image, (100, 100), [5, 10, 55, 40])
    wrong = [{"command": 0, "points": [[10, 30], [20, 30], [30, 30], [50, 30]]}]
    bad = support_metrics(wrong, image, (100, 100), [5, 10, 55, 40])
    assert good["precision"] == 1 and good["F1"] == 1
    assert bad["precision"] == 0
    crop, offset, scale = square_crop(image, [0, 0, 30, 10])
    assert crop.shape == (256, 256, 3) and offset[0] < 0 and scale > 0


def test_replacement_preserves_other_instances_and_rebuilds_valid_schema():
    from types import SimpleNamespace
    from cadvec.schema import assemble_output, validate_output
    primitives = [SimpleNamespace(command=0, points=[[0, i], [1, i], [2, i], [3, i]],
        length=3, semantic_id=2, instance_id=i, rgb=[0, 0, 0], layer_id=0, width=.1) for i in range(2)]
    data = assemble_output(primitives, width=100, height=100)
    replacement = [{"command": 0, "points": [[10, 10], [11, 10], [12, 10], [13, 10]],
                    "length": 3, "semantic": 2, "rgb": [0, 0, 0], "layer_id": 0, "width": .1}]
    result = replace_instances(data, {0: replacement})
    validate_output(result)
    assert result["args"][1] == data["args"][1]
    assert result["args"][0] == [10., 10., 11., 10., 12., 10., 13., 10.]
