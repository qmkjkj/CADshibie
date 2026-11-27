#!/usr/bin/env python3
"""Freeze conservative cross-dataset image-family splits from an overlap audit.

Near-hash edges are exclusion candidates, not confirmed duplicates. They are
used conservatively to prevent related pages crossing validation boundaries.
No source data or labels are deleted or modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def clean_splits(audit, historical, source_families=None):
    if not audit.get("ok") or audit.get("near", {}).get("truncated"):
        raise ValueError("Require a complete successful image overlap audit")
    images = audit["images"]
    parent = list(range(len(images)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(members):
        if members:
            root = find(members[0])
            for member in members[1:]:
                other = find(member)
                parent[other] = root

    for group in audit["exact_groups"]:
        union(group["members"])
    for pair in audit["near"]["pairs"]:
        union(pair["left"] + pair["right"])
    origin_groups = {}
    for i, image in enumerate(images):
        key = f"{image['dataset']}/{Path(image['path']).stem}"
        family = (source_families or {}).get(key)
        if family is not None:
            origin_groups.setdefault(family, []).append(i)
    for members in origin_groups.values():
        union(members)
    families = {}
    for i, image in enumerate(images):
        families.setdefault(find(i), []).append(image)
    old_train_val = set(historical["train_final"]["val"])
    old_test_role = {stem: role for role, stems in historical["test_final"].items()
                     for stem in stems}
    train = {"train": [], "val": []}
    test = {"calib": [], "hold": [], "final": []}
    excluded = {"train": [], "test": []}
    groups = {"train_final": {}, "test_final": {}}
    exact_test = set()
    for group in audit["exact_groups"]:
        members = [images[i] for i in group["members"]]
        roles = {image["dataset"] for image in members}
        if {"train", "test"} <= roles:
            exact_test.update(Path(image["path"]).stem for image in members
                              if image["dataset"] == "test")
    for members in families.values():
        roles = {image["dataset"] for image in members}
        family = min(f"{image['dataset']}/{image['path']}" for image in members)
        train_stems = sorted(Path(image["path"]).stem for image in members
                             if image["dataset"] == "train")
        test_stems = sorted(Path(image["path"]).stem for image in members
                            if image["dataset"] == "test")
        for stem in train_stems:
            groups["train_final"][stem] = family
        for stem in test_stems:
            groups["test_final"][stem] = family
        if roles & {"test", "dev"}:
            excluded["train"].extend(train_stems)
        else:
            role = "val" if old_train_val.intersection(train_stems) else "train"
            train[role].extend(train_stems)
        if "train" in roles:
            excluded["test"].extend(test_stems)
        elif test_stems:
            # If transitive grouping connects historical roles, preserve the
            # strictest role, so final/hold pages never migrate into calibration.
            role = max((old_test_role[stem] for stem in test_stems),
                       key={"calib": 0, "hold": 1, "final": 2}.__getitem__)
            test[role].extend(test_stems)
    for mapping in (train, test, excluded):
        for role in mapping:
            mapping[role].sort()
    return {
        "schema_version": 2,
        "method": "Joint exact-pixel, conservative dHash, and optional source/page-prefix connected components",
        "near_candidates_are_confirmed_duplicates": False,
        "source_page_prefix_grouping": bool(source_families),
        "source_page_prefix_note": "Conservative candidate family: manifest source + stem prefix before '-'; not a claim that all such crops are pixel duplicates.",
        "audit_input_content_sha256": audit["input_content_sha256"],
        "train_final": train, "test_final": test, "excluded": excluded,
        "exact_train_overlap_test_stems": sorted(exact_test), "groups": groups,
        "counts": {"train_final": {r: len(s) for r, s in train.items()},
                   "test_final": {r: len(s) for r, s in test.items()},
                   "excluded": {r: len(s) for r, s in excluded.items()}},
        "existing_weights_note": "Existing weights are frozen historical models; their prior label exposure cannot be undone by a new split.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="Optional original dataset manifests for source/page-prefix grouping")
    args = parser.parse_args()
    source_families = {}
    if args.data_root:
        for role, directory in (("train", "train_final"), ("test", "test_final")):
            with (args.data_root/directory/"manifest.csv").open(encoding="utf-8-sig",newline="") as stream:
                for row in csv.DictReader(stream):
                    source_families[f"{role}/{row['dest_stem']}"] = row["source"]+"/"+row["stem"].split("-")[0]
    result = clean_splits(json.loads(args.audit.read_text(encoding="utf-8")),
                          json.loads(args.historical.read_text(encoding="utf-8")), source_families)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
