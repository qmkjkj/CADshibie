#!/usr/bin/env python3
"""Build complete native templates restricted to a declared training split."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.template_build import build_library  # noqa: E402


def selected_stems(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload.get("train_final", {})
    names = dataset.get("train")
    if not isinstance(names, list) or not names or any(not isinstance(s, str) or not s for s in names):
        raise ValueError("train_final/train must be a nonempty list of source stems")
    if any(Path(s).name != s or Path(s).suffix for s in names):
        raise ValueError("Split entries must be source stems, without paths or extensions")
    if len(set(names)) != len(names):
        raise ValueError("Training split contains duplicate stems")
    others = {s for role, values in dataset.items() if role != "train"
              and isinstance(values, list) for s in values}
    if set(names) & others:
        raise ValueError("Source stems occur in more than one split")
    return set(names)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--splits", type=Path, help="build_splits.py JSON; train_final/train only")
    selection.add_argument("--all-training-sources", action="store_true",
                           help="Explicitly declare this directory contains training sources only")
    args = parser.parse_args(argv)
    try:
        allowed = selected_stems(args.splits) if args.splits else None
        payload = build_library(args.svg_dir, allowed_stems=allowed, limit=args.limit)
        payload["build"]["selection"] = {"role": "training", "explicit_split": args.splits is not None}
        if args.splits:
            payload["build"]["selection"]["split_sha256"] = hashlib.sha256(args.splits.read_bytes()).hexdigest()
        encoded = zlib.compress(pickle.dumps(payload, protocol=4), 9)
        meta = {"schema_version": 2, "templates": len(payload["templates"]),
                "library_sha256": hashlib.sha256(encoded).hexdigest(), **payload["build"]}
        if args.out.suffix.lower() == ".json":
            raise ValueError("Library output must not use the metadata .json extension")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(args.out)
        args.out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Built {len(payload['templates'])} complete templates from {len(payload['build']['source_files'])} SVGs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
