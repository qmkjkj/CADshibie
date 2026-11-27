import pytest

from tools.build_clean_splits import clean_splits


def test_joint_groups_exclude_train_test_overlap_and_isolate_train_validation():
    images = [{"dataset": role, "path": name + ".png"} for role, name in
              [("train", "a"), ("train", "b"), ("train", "c"),
               ("test", "x"), ("test", "y"), ("test", "z"), ("dev", "d")]]
    audit = {"ok": True, "images": images, "input_content_sha256": "hash",
             "exact_groups": [{"members": [0, 3]}],
             "near": {"truncated": False, "pairs": [
                 {"left": [1], "right": [2]}, {"left": [4], "right": [5]}]}}
    historical = {"train_final": {"train": ["a", "b"], "val": ["c"]},
                  "test_final": {"calib": ["x", "y"], "hold": [], "final": ["z"]}}
    result = clean_splits(audit, historical)
    assert result["excluded"] == {"train": ["a"], "test": ["x"]}
    assert result["train_final"] == {"train": [], "val": ["b", "c"]}
    assert result["test_final"]["final"] == ["y", "z"]
    assert result["exact_train_overlap_test_stems"] == ["x"]
    assert result["groups"]["test_final"]["y"] == result["groups"]["test_final"]["z"]


def test_dev_related_training_is_excluded_and_incomplete_audit_rejected():
    audit = {"ok": True, "images": [{"dataset": "train", "path": "a.png"},
                                      {"dataset": "dev", "path": "d.png"}],
             "input_content_sha256": "hash", "exact_groups": [],
             "near": {"truncated": False, "pairs": [{"left": [0], "right": [1]}]}}
    result = clean_splits(audit, {"train_final": {"train": ["a"], "val": []}, "test_final": {}})
    assert result["excluded"]["train"] == ["a"]
    audit["near"]["truncated"] = True
    with pytest.raises(ValueError):
        clean_splits(audit, {})


def test_original_page_family_separates_distinct_crops_across_train_test():
    audit = {"ok":True,"images":[{"dataset":"train","path":"a.png"},
                                 {"dataset":"test","path":"b.png"}],
             "input_content_sha256":"hash","exact_groups":[],
             "near":{"truncated":False,"pairs":[]}}
    history = {"train_final":{"train":["a"],"val":[]},
               "test_final":{"calib":["b"],"hold":[],"final":[]}}
    result = clean_splits(audit,history,{"train/a":"source/0003","test/b":"source/0003"})
    assert result['excluded'] == {'train':['a'],'test':['b']}
    assert result['source_page_prefix_grouping']
