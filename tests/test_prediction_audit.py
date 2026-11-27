import json

from PIL import Image

from cadvec.schema import empty_output
from tools.audit_predictions import audit, select_samples


def test_valid_empty_prediction_does_not_claim_quality(tmp_path):
    images, predictions = tmp_path / "images", tmp_path / "predictions"
    images.mkdir()
    predictions.mkdir()
    Image.new("RGB", (1000, 1000), "white").save(images / "one.png")
    (predictions / "one.json").write_text(json.dumps(empty_output(100, 100)), encoding="utf-8")
    report = audit(images, predictions, preview_dir=tmp_path / "previews")
    assert report["ok"] and report["valid_images"] == 1
    assert report["empty_images"] == 1 and report["PQ"] is None
    assert report["has_ground_truth"] is False
    assert len(report["samples"]) == 1


def test_missing_extra_and_invalid_outputs_are_reported_together(tmp_path):
    images, predictions = tmp_path / "images", tmp_path / "predictions"
    images.mkdir()
    predictions.mkdir()
    for name in ["missing", "broken"]:
        Image.new("RGB", (10, 10), "white").save(images / f"{name}.png")
    (predictions / "broken.json").write_text('{"commands":[]}', encoding="utf-8")
    (predictions / "extra.json").write_text(json.dumps(empty_output(100, 100)), encoding="utf-8")
    report = audit(images, predictions)
    assert not report["ok"]
    assert report["missing"] == ["missing"] and report["extra"] == ["extra"]
    assert len(report["failures"]) == 1


def test_sampling_is_deterministic_and_covers_empty_sparse_dense():
    rows = [{"name": name, "primitive_count": count, "empty": count == 0, "valid": True}
            for name, count in [("z", 0), ("a", 0), ("mid", 5), ("sparse", 1), ("dense", 100)]]
    chosen = select_samples(list(reversed(rows)))
    assert [r["name"] for _, r in chosen][:2] == ["a", "z"]
    assert {role for role, _ in chosen} == {"empty", "sparse", "dense"}
    assert len(chosen) == len({r["name"] for _, r in chosen})
