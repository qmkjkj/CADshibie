#!/usr/bin/env python3
"""Replay frozen hybrid candidates; no neural inference and no labels read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.hybrid import merge_final
from cadvec.schema import write_output


def replay(snapshot, *, raster_mode=None, source_identity=None, merge_function=merge_final):
    options = dict(snapshot["merge_options"])
    for name in ("gat_classes", "segdet_classes", "gate_classes", "segdet_union_classes", "suppress"):
        options[name] = set(options[name])
    options["quality_rules"] = {int(k): tuple(v) for k, v in (options["quality_rules"] or {}).items()}
    if raster_mode is not None:
        options["raster_mode"] = raster_mode
    if source_identity is not None:
        options["source_identity"] = source_identity
    data = merge_function(snapshot["primary"], snapshot["gat"], snapshot["segdet"],
                          support=snapshot["support"], **options)
    assignment = snapshot["assignment"]
    support = snapshot["support"] or {}
    detections = support.get("detections") or []
    size = support.get("size") or (0, 0)
    if detections and min(size) > 0:
        if assignment["anchor_classes"]:
            from cadvec.anchor_assign import reassign
            data = reassign(data, detections, set(assignment["anchor_classes"]),
                            int(size[1]), int(size[0]), inside=assignment["anchor_inside"],
                            affinity=assignment["anchor_affinity"])
        if assignment["atom_classes"] and assignment["atom_affinity"]:
            from cadvec.atom_assign import assign_atoms
            data = assign_atoms(data, detections, set(assignment["atom_classes"]),
                                int(size[1]), int(size[0]), assignment["atom_affinity"])
    if snapshot.get("stuff_items"):
        from cadvec.stuff_branch import append_stuff
        data = append_stuff(data, snapshot["stuff_items"], data["width"], data["height"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raster", choices=("legacy400", "canonical1000"))
    parser.add_argument("--source-identity", choices=("legacy", "scoped"))
    parser.add_argument("--geometry-refine", action="store_true")
    parser.add_argument("--png-dir", type=Path)
    parser.add_argument("--verify-recorded", action="store_true")
    args = parser.parse_args()
    if args.geometry_refine and args.png_dir is None:
        parser.error("--geometry-refine requires --png-dir")
    paths = sorted(args.snapshots.glob("*.json"))
    if not paths:
        parser.error("No stage snapshots")
    args.output.mkdir(parents=True, exist_ok=True)
    accepted = 0
    for index, path in enumerate(paths):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        data = replay(snapshot, raster_mode=args.raster, source_identity=args.source_identity)
        if args.verify_recorded and data != snapshot["final"]:
            raise ValueError(f"Replay differs from recorded final output: {path.name}")
        if args.geometry_refine:
            from cadvec.geometry_refine import refine_image
            data, decisions = refine_image(data, args.png_dir / snapshot["file"])
            accepted += sum(row["accepted"] for row in decisions)
            diagnostic_dir = args.output.parent / f"{args.output.name}_geometry_refine"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            (diagnostic_dir / path.name).write_text(json.dumps(decisions), encoding="utf-8")
        write_output(data, args.output / path.name)
        if (index + 1) % 25 == 0:
            print(f"replayed {index + 1}/{len(paths)}", flush=True)
    print(json.dumps({"images": len(paths), "geometry_accepted": accepted}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
