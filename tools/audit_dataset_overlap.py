#!/usr/bin/env python3
"""Audit exact PNG overlap and dHash review candidates without reading labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import struct

from PIL import Image


def image_fingerprint(path):
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        if source.format != "PNG":
            raise ValueError("File is not PNG")
        rgba = source.convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    digest = hashlib.sha256(b"white-rgb-v1\0" + struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()
    pixels = list(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    dhash = 0
    for y in range(8):
        for x in range(8):
            dhash = (dhash << 1) | int(pixels[y * 9 + x + 1] > pixels[y * 9 + x])
    return {"source_sha256": hashlib.sha256(raw).hexdigest(), "pixel_sha256": digest,
            "dhash64": f"{dhash:016x}", "width": rgb.width, "height": rgb.height}


class HashIndex:
    """BK tree over distinct hash values; equal hashes share a node."""
    def __init__(self):
        self.root = None

    def add(self, value, key):
        if self.root is None:
            self.root = [value, [key], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(key)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [key], {}]
                return
            node = node[2][distance]

    def query(self, value, threshold):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                for key in node[1]:
                    yield key, distance
            pending.extend(child for edge, child in node[2].items()
                           if distance - threshold <= edge <= distance + threshold)


def audit_datasets(datasets, max_dhash_distance=4, max_candidate_pairs=100000):
    if not 0 <= max_dhash_distance <= 64 or max_candidate_pairs < 1:
        raise ValueError("Invalid dHash threshold or candidate limit")
    if not datasets or any(not name.strip() for name, _ in datasets):
        raise ValueError("At least one named dataset is required")
    if len({name for name, _ in datasets}) != len(datasets):
        raise ValueError("Dataset names must be unique")
    images, failures, summaries = [], [], []
    for name, directory in sorted(datasets):
        root = Path(directory).expanduser().resolve()
        paths = []
        discovery_errors = []
        if root.is_dir():
            for current, subdirs, filenames in os.walk(root, followlinks=False,
                                                       onerror=lambda e: discovery_errors.append(str(e))):
                subdirs[:] = sorted(name for name in subdirs if name != "__MACOSX")
                paths.extend(Path(current) / f for f in sorted(filenames)
                             if Path(f).suffix.lower() == ".png" and not f.startswith("._"))
        else:
            discovery_errors.append(f"Dataset directory does not exist: {root}")
        for error in discovery_errors:
            failures.append({"dataset": name, "error": error, "stage": "discovery"})
        if not paths and not discovery_errors:
            failures.append({"dataset": name, "error": "No PNG files", "stage": "input"})
        valid = 0
        for path in sorted(paths):
            item = {"dataset": name, "path": path.relative_to(root).as_posix()}
            try:
                item.update(image_fingerprint(path))
                valid += 1
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                failures.append({**item, "stage": "decode"})
            images.append(item)
        summaries.append({"dataset": name, "root": str(root), "png_count": len(paths), "decoded_count": valid})
    inventory = json.dumps(images, sort_keys=True, separators=(",", ":")).encode("utf-8")
    content = defaultdict(list)
    for index, item in enumerate(images):
        if "pixel_sha256" in item:
            content[item["pixel_sha256"]].append(index)
    exact = [{"pixel_sha256": key, "members": members,
              "cross_dataset": len({images[i]["dataset"] for i in members}) > 1}
             for key, members in sorted(content.items()) if len(members) > 1]
    candidates, tree, truncated = [], HashIndex(), False
    for key, members in sorted(content.items()):
        value = int(images[members[0]]["dhash64"], 16)
        for previous, distance in tree.query(value, max_dhash_distance):
            if len(candidates) == max_candidate_pairs:
                truncated = True
                break
            candidates.append({"left": content[previous], "right": members,
                               "hamming_distance": distance,
                               "cross_dataset": len({images[i]["dataset"]
                                                     for i in content[previous] + members}) > 1})
        if truncated:
            break
        tree.add(value, key)
    return {"schema_version": 1, "ok": not failures and not truncated,
            "input_content_sha256": hashlib.sha256(inventory).hexdigest(),
            "input_content_hash_complete": not failures,
            "normalization": "PNG decoded to RGBA, composited on white; RGB and dimensions hashed",
            "datasets": summaries, "images": images, "failures": failures,
            "exact_groups": exact,
            "near": {"candidate_only": True, "max_dhash_distance": max_dhash_distance,
                     "max_candidate_pairs": max_candidate_pairs, "truncated": truncated,
                     "note": "dHash similarity is a review candidate, not proof of a duplicate or drawing family",
                     "pairs": candidates}}


def dataset_argument(value):
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Use NAME=PATH")
    return name.strip(), path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_argument, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dhash-distance", type=int, default=4)
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        report = audit_datasets(args.dataset, args.max_dhash_distance, args.max_candidate_pairs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"ok": report["ok"], "images": len(report["images"]),
                      "exact_groups": len(report["exact_groups"]),
                      "near_candidates": len(report["near"]["pairs"]),
                      "truncated": report["near"]["truncated"], "failures": len(report["failures"])}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
