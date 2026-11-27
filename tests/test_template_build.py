import numpy as np
import pytest

from cadvec.template_build import build_library
from cadvec.templating import TemplateLibrary
from tools.build_native_templates import main, selected_stems


def svg(path, *, count=1, bend=0, instance=1):
    body = "".join(f'<path d="M {i} 10 L {i + 5} {10 + bend}" '
                   f'stroke="#000" fill="none" semantic-id="4" instance-id="{instance}"/>'
                   for i in range(count))
    path.write_text(f'<svg viewBox="0 0 100 100">{body}</svg>', encoding="utf-8")


def test_complete_roundtrip_keeps_all_primitives_and_provenance(tmp_path):
    svg(tmp_path / "large.svg", count=73)
    output = tmp_path / "templates.pkl"
    assert main(["--svg-dir", str(tmp_path), "--out", str(output), "--all-training-sources"]) == 0
    library = TemplateLibrary.load(output)
    assert library is not None and len(library.templates) == 1
    item = library.templates[0]
    assert len(item["primitives"]) == 73
    assert item["primitives"][-1]["points"][0] == [72, 10]
    source = item["sources"][0]
    assert source["svg"] == "large.svg" and source["instance_id"] == 1
    assert source["viewbox"] == [0, 0, 100, 100]
    assert len(source["sha256"]) == 64
    assert library.retrieve(library.points[0], class_filter=3)[0][1] == 0


def test_descriptor_collision_does_not_merge_distinct_geometry(tmp_path, monkeypatch):
    import cadvec.template_build as builder
    svg(tmp_path / "first.svg")
    svg(tmp_path / "second.svg", bend=0.001)
    monkeypatch.setattr(builder, "descriptors", lambda points: np.zeros((4, 256)))
    assert len(build_library(tmp_path)["templates"]) == 2


def test_exact_duplicates_keep_all_source_instances(tmp_path):
    svg(tmp_path / "first.svg", instance=1)
    svg(tmp_path / "second.svg", instance=7)
    item, = build_library(tmp_path)["templates"]
    assert item["count"] == 2
    assert {s["instance_id"] for s in item["sources"]} == {1, 7}


def test_split_excludes_bad_validation_and_rejects_missing_source(tmp_path):
    svg(tmp_path / "train.svg")
    (tmp_path / "val.svg").write_text("invalid SVG", encoding="utf-8")
    payload = build_library(tmp_path, allowed_stems={"train"})
    assert payload["build"]["allowed_stems"] == ["train"]
    with pytest.raises(ValueError, match="missing SVG stems"):
        build_library(tmp_path, allowed_stems={"missing"})
    with pytest.raises(ValueError, match="val.svg"):
        build_library(tmp_path)
    split = tmp_path / "splits.json"
    split.write_text('{"train_final":{"train":["train"],"val":["val"]}}', encoding="utf-8")
    assert selected_stems(split) == {"train"}
    split.write_text('{"train_final":{"train":["train"],"val":["train"]}}', encoding="utf-8")
    with pytest.raises(ValueError, match="more than one split"):
        selected_stems(split)


def test_invalid_build_and_missing_source_declaration_write_no_output(tmp_path):
    output = tmp_path / "templates.pkl"
    with pytest.raises(SystemExit):
        main(["--svg-dir", str(tmp_path), "--out", str(output), "--all-training-sources"])
    assert not output.exists()
    svg(tmp_path / "one.svg")
    with pytest.raises(SystemExit):
        main(["--svg-dir", str(tmp_path), "--out", str(output)])
    assert not output.exists()
