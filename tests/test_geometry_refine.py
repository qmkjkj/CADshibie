from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cadvec.evaluate import evaluate
from cadvec.geometry_refine import GeometryRefineConfig, image_ink, refine_geometry
from cadvec.raster import render_items, render_window
from cadvec.schema import assemble_output


def _line(a, b, instance=7, semantic=8):
    return SimpleNamespace(command=0, points=[a, a, b, b], length=float(np.linalg.norm(
        np.subtract(b, a))), semantic_id=semantic, instance_id=instance,
        rgb=[0, 0, 0], layer_id=1, width=0.1)


def _drawing(primitives=None, width=100, height=100):
    if primitives is None:
        primitives = [_line((20, 20), (40, 20)), _line((40, 20), (40, 43))]
    return assemble_output(primitives, width=width, height=height)


def _items(data):
    return [{"command": c, "points": list(zip(args[::2], args[1::2]))}
            for c, args in zip(data["commands"], data["args"])]


def _offset(data, dx, dy):
    shifted = copy.deepcopy(data)
    for args in shifted["args"]:
        for i in range(8):
            args[i] += dx if i % 2 == 0 else dy
    for center, box in zip(shifted["obj_cts"], shifted["boxes"]):
        center[0] += dx
        center[1] += dy
        for i in range(4):
            box[i] += dx if i % 2 == 0 else dy
    return shifted


def test_known_pixel_offset_recovers_strict_match_without_changing_identity():
    gt = _drawing()
    pred = _offset(gt, 0.1, 0.2)
    original = copy.deepcopy(pred)
    ink = render_items(_items(gt))
    result, decisions = refine_geometry(pred, ink, ink)
    assert decisions[0]["accepted"]
    assert decisions[0]["shift_px"] == [-1, -2]
    assert evaluate(gt, pred)["TP"] == 0
    assert evaluate(gt, result)["micro"]["PQ"] > 0.99
    assert result["instanceIds"] == gt["instanceIds"]
    assert result["semanticIds"] == gt["semanticIds"]
    assert result["commands"] == gt["commands"]
    assert result["boxes"] == gt["boxes"]
    assert result["obj_cts"] == gt["obj_cts"]
    assert pred == original


def test_aligned_geometry_is_unchanged_and_refinement_is_stable():
    gt = _drawing()
    ink = render_items(_items(gt))
    result, decisions = refine_geometry(gt, ink, ink)
    assert result == gt
    assert not decisions[0]["accepted"]
    result, _ = refine_geometry(_offset(gt, 0.1, 0.2), ink, ink)
    twice, decisions = refine_geometry(result, ink, ink)
    assert twice == result
    assert not decisions[0]["accepted"]


def test_empty_and_unsupported_short_primitive_keep_original_geometry():
    gt = _drawing()
    blank = np.zeros((1000, 1000), dtype=bool)
    result, decisions = refine_geometry(gt, blank)
    assert result == gt
    assert decisions[0]["reason"] == "empty_evidence"
    gt = _drawing([_line((20, 20), (60, 20)), _line((45, 45), (46, 45))])
    pred = _offset(gt, 0, 0.1)
    # Only the long primitive has image support.
    ink = render_items(_items(gt)[:1])
    result, decisions = refine_geometry(pred, ink, ink)
    assert result == pred
    assert decisions[0]["reason"] == "insufficient_ink_support"


def test_rectangular_image_and_canvas_map_each_axis_and_preserve_stuff():
    gt = _drawing([_line((20, 20), (40, 20)), _line((40, 20), (40, 43)),
                   _line((90, 60), (100, 60), instance=-1, semantic=32)],
                  width=200, height=80)
    ink = render_window(_items(gt), (480, 1000), (5, 6))
    pred = copy.deepcopy(gt)
    for args in pred["args"][:2]:
        for i in range(8):
            args[i] += 0.2 if i % 2 == 0 else 1 / 6
    result, decisions = refine_geometry(pred, ink, ink)
    assert decisions[0]["accepted"]
    assert decisions[0]["shift_px"] == [-1, -1]
    assert result["args"][-1] == gt["args"][-1]
    assert result["instanceIds"] == gt["instanceIds"]
    assert np.allclose(result["args"][:2], gt["args"][:2])


def test_curve_type_and_disconnected_primitives_do_not_gain_bridges():
    circle = _line((20, 20), (30, 30))
    circle.command = 2
    circle.points = [(20, 25), (25, 20), (30, 25), (25, 30)]
    gt = _drawing([circle, _line((40, 20), (50, 20))])
    ink = render_items(_items(gt))
    assert not ink[250, 350]
    result, decisions = refine_geometry(_offset(gt, 0.1, 0), ink, ink)
    assert decisions[0]["accepted"]
    assert result["commands"] == [2, 0]
    assert evaluate(gt, result)["micro"]["PQ"] > 0.99


def test_competing_instance_and_clipped_geometry_do_not_move():
    gt = _drawing([_line((20, 20), (60, 20), instance=1),
                   _line((20, 20.2), (60, 20.2), instance=2)])
    ink = render_items(_items(gt)[:1])
    result, decisions = refine_geometry(gt, ink, ink)
    assert result == gt
    assert decisions[1]["reason"] == "competing_instance"
    outside = _drawing([_line((-1, 20), (40, 20))])
    result, decisions = refine_geometry(outside, ink, ink)
    assert result == outside
    assert decisions[0]["reason"] == "small_or_clipped_geometry"


def test_equal_parallel_evidence_is_ambiguous():
    pred = _drawing([_line((20, 20), (40, 20))])
    ink = render_items(_items(_offset(pred, 0, -0.1)))
    ink = ink | render_items(_items(_offset(pred, 0, 0.1)))
    result, decisions = refine_geometry(pred, ink, ink)
    assert result == pred
    assert decisions[0]["reason"] == "ambiguous_pose"


def test_alpha_and_colored_ink_and_crop_raster_phase(tmp_path):
    rgba = np.zeros((40, 70, 4), dtype=np.uint8)
    rgba[15, 10:60] = [255, 0, 0, 255]
    rgba[25, 10:60] = [0, 0, 0, 0]
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba).save(path)
    foreground, centerline = image_ink(path)
    assert foreground[15, 20] and centerline[15, 20]
    assert not foreground[25, 20]
    items = [{"command": 3, "points": [(20.12, 15.18), (30.11, 15.18),
                                        (30.11, 32.27), (20.12, 32.27)]}]
    full = render_items(items)
    crop = render_window(items, (200, 130), (10, 10), (190, 140))
    assert np.array_equal(full[140:340, 190:320], crop)


def test_disabled_mode_keeps_output_exactly():
    gt = _drawing()
    ink = render_items(_items(gt))
    output, decisions = refine_geometry(gt, ink, config=GeometryRefineConfig(max_shift_px=0))
    assert output == gt
    assert decisions[0]["reason"] == "disabled"
