from PIL import Image
import pytest

from infer import find_pngs
from tools.audit_dataset_overlap import audit_datasets


def test_extracted_macos_resource_files_are_ignored_by_inference_and_audit(tmp_path):
    actual = tmp_path / "dev" / "drawing.PNG"
    actual.parent.mkdir()
    Image.new("RGB", (10, 10), "white").save(actual)
    resource = tmp_path / "__MACOSX" / "dev" / "._drawing.png"
    resource.parent.mkdir(parents=True)
    resource.write_bytes(b"AppleDouble metadata")
    sibling = actual.parent / "._another.png"
    sibling.write_bytes(b"AppleDouble metadata")
    assert find_pngs(tmp_path) == [actual]
    report = audit_datasets([("dev", tmp_path)])
    assert report["ok"] and len(report["images"]) == 1
    assert report["datasets"][0]["png_count"] == 1
    with pytest.raises(SystemExit, match="macOS"):
        find_pngs(resource)


def test_corrupt_regular_png_is_kept_and_reported_as_failure(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert find_pngs(tmp_path) == [path]
    report = audit_datasets([("dev", tmp_path)])
    assert not report["ok"] and report["failures"][0]["stage"] == "decode"
