#!/usr/bin/env python3
"""Compare native geometry heads with identical cached hybrid branches.

Only the ROI detail checkpoint changes. Detector, coarse ROI output, quality
gates, NMS, decoding and branch merging retain the production implementation.
Labels are read only after all prediction files have been written.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("YOLO_OFFLINE", "true")

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.config import InferenceConfig
from cadvec.evaluate import evaluate_directories
from cadvec.native_detail import NativeDetailRefiner
from cadvec.roi_ink import RoiInkRefiner
from cadvec.schema import write_output
import cadvec.pipeline_segdet as segdet
from infer import build_parser, _prepare_environment
from tools.replay_hybrid import replay
from tools.summarize_ablation import aggregate, paired_group_bootstrap


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="NAME=CHECKPOINT, baseline first")
    parser.add_argument("--verify-baseline", type=Path, help="Require exact reproduction of existing baseline JSON")
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--source-splits", type=Path)
    parser.add_argument("--group-role", choices=("train_final", "test_final"), default="train_final")
    parser.add_argument("--source-identity", choices=("legacy", "scoped"),
                        help="Explicit identity mode for older snapshots missing this field")
    args = parser.parse_args()
    defaults = build_parser().parse_args([])
    _prepare_environment(defaults.seed)
    paths = sorted(args.snapshots.glob("*.json"))
    if not paths:
        parser.error("No snapshots")
    variants = {}
    for entry in args.variant:
        name, separator, path = entry.partition("=")
        if not separator or not name or name in variants or Path(name).name != name:
            parser.error("Use unique NAME=CHECKPOINT entries")
        variants[name] = (Path(path), NativeDetailRefiner.load(path, "cuda"))
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": 1,
        "weights": {n:{"path":str(p), "sha256":sha256(p)} for n,(p,_) in variants.items()},
        "snapshots": {p.name:sha256(p) for p in paths},
        "pngs": {p.stem:sha256(args.png_dir/json.loads(p.read_text(encoding='utf-8'))['file']) for p in paths},
        "fixed_inputs": {str(p):sha256(p) for p in (
            "weights/yolo_seg.pt", "weights/roi_ink_c2.pt", "weights/roi_ink_512.pt",
            "weights/roi_ink_256.pt", defaults.roi_quality_model, defaults.palette)},
        "max_instances": 0,
        "source_identity_override":args.source_identity,
        "source_files": {p:sha256(p) for p in (
            "cadvec/detail_unet.py", "cadvec/native_detail.py", "cadvec/roi_ink.py",
            "cadvec/pipeline_segdet.py", "cadvec/hybrid.py", "tools/replay_hybrid.py",
            "tools/compare_native_geometry.py")},
        "note": "Frozen primary/GAT/stuff branches reused; YOLO and coarse ROI predictions cached only across variants on the same page. Production refine and merge execute without monkeypatching selection.",
    }
    provenance_path = args.out/"provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Output directory belongs to a different frozen experiment")
    provenance_path.write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    plans = []
    for classes, path in (({2}, "weights/roi_ink_c2.pt"),
                          ({3,4,22,26,29}, "weights/roi_ink_512.pt"),
                          ({8,12,14,15,18,19,20,21,25,27,28}, "weights/roi_ink_256.pt")):
        model = RoiInkRefiner.load(path, device="cuda", max_instances=0,
                                   quality_path=defaults.roi_quality_model)
        original_analyze = model.analyze
        cache = {}
        def analyze(image, *a, _cache=cache, _original=original_analyze, **kw):
            key = repr((a, kw))
            if key not in _cache:
                _cache[key] = _original(image, *a, **kw)
            return _cache[key]
        model.analyze = analyze
        plans.append((frozenset(classes), model, cache))
    handled = frozenset().union(*(c for c,_,_ in plans))
    detector = segdet._model("weights/yolo_seg.pt")
    original_predict, original_scene = detector.predict, segdet.component_scene
    predict_cache, scene_cache = {}, {}
    def predict(*a, **kw):
        key = repr((a, kw))
        if key not in predict_cache:
            predict_cache[key] = original_predict(*a, **kw)
        return predict_cache[key]
    def scene(*a, **kw):
        if "value" not in scene_cache:
            scene_cache["value"] = original_scene(*a, **kw)
        return scene_cache["value"]
    detector.predict, segdet.component_scene = predict, scene
    config = InferenceConfig()
    timings = {n:0. for n in variants}
    for page_number, path in enumerate(paths):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        predict_cache.clear(); scene_cache.clear()
        for _,_,cache in plans:
            cache.clear()
        for name, (_, native) in variants.items():
            started = time.perf_counter()
            for _, model, _ in plans:
                model.native_detail = native
            result = segdet.run_image_segdet(
                args.png_dir/snapshot["file"], "weights/yolo_seg.pt", config, defaults.palette,
                det_conf=.2, det_iou=.5, max_instances=10, inside_fraction=.35,
                image_size=640, device=0, support_classes=frozenset({8}),
                refiner_classes=handled, refiners=tuple((c,m) for c,m,_ in plans))
            updated = dict(snapshot, segdet=result["data"], support=result["support"])
            data = replay(updated,source_identity=args.source_identity)
            if name == next(iter(variants)) and args.verify_baseline:
                expected = json.loads((args.verify_baseline/path.name).read_text(encoding="utf-8"))
                if data != expected:
                    raise ValueError("Baseline replay differs: " + path.name)
            write_output(data, args.out/name/path.name)
            timings[name] += time.perf_counter()-started
        if (page_number+1) % 10 == 0:
            print("paired native heads", page_number+1, "/", len(paths), flush=True)
    (args.out/"timings.json").write_text(json.dumps({
        "seconds":timings,
        "note":"First variant includes detector/coarse/scene work shared by later variants; these totals cannot compare end-to-end latency."
    }, indent=1))
    if args.gt_dir:
        reports = {}
        for name in variants:
            reports[name] = evaluate_directories(args.gt_dir, args.out/name)
            (args.out/(name+"_eval.json")).write_text(json.dumps(reports[name], indent=1))
        summary = {"metrics": {n:aggregate(list(r["per_image"].values())) for n,r in reports.items()}}
        if args.source_splits:
            groups = json.loads(args.source_splits.read_text(encoding="utf-8"))["groups"][args.group_role]
            summary["uncertainty"] = paired_group_bootstrap(
                reports, {p.stem for p in paths}, groups, draws=5000)
        (args.out/"comparison.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
