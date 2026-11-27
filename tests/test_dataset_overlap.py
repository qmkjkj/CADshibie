import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tools.audit_dataset_overlap import audit_datasets, main


def save(path, image, comment=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text("Comment", comment)
    image.save(path, pnginfo=metadata)


def test_identical_pixels_ignore_encoding_but_inventory_tracks_it(tmp_path):
    a, b = tmp_path / "train", tmp_path / "validation"
    image = Image.new("RGB", (16, 8), "red")
    save(a / "first.png", image, "first")
    save(b / "renamed.png", image, "other")
    report = audit_datasets([("train", a), ("validation", b)])
    assert report["ok"]
    assert len(report["exact_groups"]) == 1
    assert report["exact_groups"][0]["cross_dataset"]
    assert len({i["source_sha256"] for i in report["images"]}) == 2
    assert not report["near"]["pairs"]
    original = report["input_content_sha256"]
    save(b / "renamed.png", image, "new metadata")
    assert audit_datasets([("train", a), ("validation", b)])["input_content_sha256"] != original


def test_dhash_collision_and_same_name_do_not_imply_exact_duplicate(tmp_path):
    a, b = tmp_path / "train", tmp_path / "validation"
    save(a / "same.png", Image.new("RGB", (12, 12), "red"))
    save(b / "same.png", Image.new("RGB", (12, 12), "blue"))
    report = audit_datasets([("train", a), ("validation", b)])
    assert not report["exact_groups"]
    assert report["near"]["candidate_only"]
    pair, = report["near"]["pairs"]
    assert pair["hamming_distance"] == 0 and pair["cross_dataset"]


def test_hidden_transparent_rgb_matches_opaque_white(tmp_path):
    a, b = tmp_path / "train", tmp_path / "validation"
    save(a / "transparent.png", Image.new("RGBA", (10, 10), (255, 0, 0, 0)))
    save(b / "opaque.png", Image.new("RGB", (10, 10), "white"))
    report = audit_datasets([("train", a), ("validation", b)])
    assert len(report["exact_groups"]) == 1


def test_dimension_changes_are_not_exact_duplicates_and_pair_cap_is_explicit(tmp_path):
    root = tmp_path / "train"
    for index in range(4):
        save(root / f"{index}.png", Image.new("RGB", (8 + index, 8), "white"))
    report = audit_datasets([("train", root)], max_candidate_pairs=2)
    assert not report["exact_groups"]
    assert report["near"]["truncated"] and not report["ok"]
    assert len(report["near"]["pairs"]) == 2


def test_empty_and_corrupt_input_reports_failure_without_reading_labels(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    (root / "one.svg").write_text("invalid XML", encoding="utf-8")
    output = tmp_path / "audit.json"
    assert main(["--dataset", f"train={root}", "--output", str(output)]) == 1
    assert not json.loads(output.read_text(encoding="utf-8"))["ok"]
    (root / "one.png").write_bytes(b"broken PNG")
    report = audit_datasets([("train", root)])
    assert not report["ok"] and not report["input_content_hash_complete"]
    assert report["failures"][0]["stage"] == "decode"
