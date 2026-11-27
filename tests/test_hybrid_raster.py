from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cadvec.evaluate import _segments, evaluate
from cadvec.hybrid import merge_final, merge_predictions
from cadvec.raster import render_data
from cadvec.schema import assemble_output


def _data(lines, width=100, height=100):
    primitives = []
    for semantic, instance, command, points in lines:
        primitives.append(SimpleNamespace(semantic_id=semantic, instance_id=instance,
                         command=command, points=points, length=20., width=.1,
                         rgb=[0, 0, 0], layer_id=1))
    return assemble_output(primitives, width=width, height=height)


def _line(y):
    return [(10., y), (10., y), (30., y), (30., y)]


def test_thin_adjacent_instances_survive_canonical_fusion():
    first = _data([(8, 0, 0, _line(10.))])
    second = _data([(8, 0, 0, _line(10.1))])
    gt = _data([(8, 0, 0, _line(10.)), (8, 1, 0, _line(10.1))])
    legacy = merge_final(first, second, gat_classes={8})
    canonical = merge_final(first, second, gat_classes={8}, raster_mode="canonical1000")
    assert len(legacy["obj_cts"]) == 1
    assert len(canonical["obj_cts"]) == 2
    assert evaluate(gt, canonical)["TP"] == 2
    assert evaluate(gt, legacy)["FN"] == 1


def test_canonical_distinguishes_ellipse_from_same_four_point_polyline():
    points = [(20, 30), (30, 20), (40, 30), (30, 40)]
    first = _data([(8, 0, 2, points)])
    second = _data([(8, 0, 0, points)])
    legacy = merge_final(first, second, gat_classes={8})
    canonical = merge_final(first, second, gat_classes={8}, raster_mode="canonical1000")
    assert len(legacy["obj_cts"]) == 1
    assert len(canonical["obj_cts"]) == 2
    # Duplicate circles still collapse correctly.
    assert len(merge_final(first, first, raster_mode="canonical1000")["obj_cts"]) == 1


def test_gate_does_not_mix_source_ids_and_is_invariant_to_renumbering():
    first = _data([(8, 0, 0, _line(10))])
    support = {"size": [1000, 1000], "polygons": {
        8: [[[90, 95], [310, 95], [310, 105], [90, 105]]]}}
    outputs = []
    for instance in (0, 97):
        second = _data([(8, instance, 0, _line(80))])
        outputs.append(merge_final(first, second, gate_classes={8}, support=support,
                                   raster_mode="canonical1000"))
    assert outputs[0] == outputs[1]
    assert len(outputs[0]["obj_cts"]) == 1
    assert outputs[0]["args"] == first["args"]


def test_simple_merge_keeps_different_classes_at_same_geometry_and_large_ids():
    first = _data([(3, 1_000_000, 0, _line(10))])
    second = _data([(8, 0, 0, _line(10))])
    output = merge_predictions(first, second, classes={3, 8})
    assert output["semanticIds"] == [3, 8]
    assert len(set(output["instanceIds"])) == 2


def test_canonical_renderer_matches_scoring_on_nondefault_canvas():
    data = _data([(8, 1, 0, _line(10.13)),
                  (3, 0, 3, [(30.1, 20.1), (40.2, 20.1), (40.2, 40.3), (30.1, 40.3)])],
                 width=200, height=80)
    masks = render_data(data)
    expected, _ = _segments(data, (200, 80), 1000)
    assert masks.keys() == expected.keys()
    for key in masks:
        assert np.array_equal(masks[key], expected[key])
    assert len(merge_final(data, data, gat_classes={3, 8},
                           raster_mode="canonical1000")["obj_cts"]) == 2
    with pytest.raises(ValueError):
        merge_final(data, raster_mode="bad")
