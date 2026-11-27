#!/usr/bin/env python3
"""Fixed class-2 native post-merge gate probe, separate from production.

Fit/calibration use only original native validation pages. Every development
prediction is saved before development labels are opened, with no second fit.
"""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.hybrid import _assemble, _items, merge_final
from cadvec.schema import write_output
from tools.diagnose_fusion_losses import primitive_signatures, quality_features
from tools.diagnose_instance_stages import assign, segment_records, strict_matches
from tools.replay_hybrid import replay


FEATURES = ('n', 'total', 'mean', 'max', 'w', 'h', 'area', 'aspect', 'density')
SEED = 2735
Q = .3378543626
THRESHOLDS = tuple(i / 20 for i in range(1, 20))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def feature_vector(features):
    value = np.asarray([features[name] for name in FEATURES], dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Quality features must be finite and nonnegative')
    return np.asarray([v if name == 'aspect' else np.log1p(v) for name, v in zip(FEATURES, value)])


def quality_probability(features, model):
    if tuple(model['features']) != FEATURES or model['transform'] != 'log1p_nonnegative_except_aspect':
        raise ValueError('Unexpected portable quality feature mapping')
    vector = feature_vector(features)
    z = float(((vector - model['mean']) / model['scale']) @ model['coef'] + model['intercept'])
    return float(1 / (1 + np.exp(-np.clip(z, -50, 50))))


def make_merge(model=None, decisions=None):
    """Remove only c2's old rule, then gate whole native signatures and rebuild.

    The baseline mode restores the exact original c2 rule, including all fixed
    primary/fusion decisions. Source signatures include every command/point.
    """
    def custom(primary, gat, segdet, **options):
        rules = dict(options.get('quality_rules') or {})
        old_rule = rules.pop(2, None)
        if old_rule != ('total', '>=', 22.) and old_rule != ['total', '>=', 22.]:
            raise ValueError('Expected the frozen class-2 total >= 22 rule')
        native = {signature for key, signature in primitive_signatures(segdet).items() if key[0] == 2}
        merged = merge_final(primary, gat, segdet, **dict(options, quality_rules=rules))
        signatures = primitive_signatures(merged)
        drop = set()
        for key, signature in signatures.items():
            if key[0] != 2:
                continue
            features = quality_features(merged, key)
            is_native = signature in native
            probability = quality_probability(features, model) if is_native and model else None
            keep = probability >= model['threshold'] if probability is not None else features['total'] >= 22.
            if not keep:
                drop.add(key)
            if decisions is not None:
                decisions.append({'instance': key[1], 'native_signature': is_native, 'features': features,
                                  'probability': probability, 'keep': keep, 'old_keep': features['total'] >= 22.})
        items = [item for item in _items(merged, 'postmerge')
                 if (item['semantic'], item['instance']) not in drop]
        return _assemble(items, merged['width'], merged['height'])
    return custom


def split_families(pages, groups):
    families = sorted({groups[s] for s in pages})
    random.Random(SEED).shuffle(families)
    fit_groups = set(families[:int(len(families)*2/3)])
    fit = sorted(s for s in pages if groups[s] in fit_groups)
    calib = sorted(set(pages)-set(fit))
    if not fit or not calib or {groups[s] for s in fit} & {groups[s] for s in calib}:
        raise ValueError('Need disjoint nonempty fit/calibration source families')
    return fit, calib


def score_filtered(matrix, keep):
    matrix = np.asarray(matrix, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    selected = matrix[:, keep]
    matches = assign(selected, selected > .5)
    count = int(keep.sum())
    return {'TP': len(matches), 'FP': count-len(matches), 'FN': len(matrix)-len(matches),
            'sum_iou': sum(v[1] for v in matches.values()), 'predictions': count}


def calibrate(pages, probabilities):
    results = []; best = None
    for threshold in THRESHOLDS:
        counts = Counter()
        for stem, page in pages.items():
            matrix = np.asarray(page['ious'], dtype=float).reshape(page['gt_count'], page['pred_count'])
            counts.update(score_filtered(matrix, np.asarray(probabilities[stem]) >= threshold))
        objective = counts['sum_iou'] - Q*.5*counts['predictions']
        row = {'threshold': threshold, 'objective': objective, **dict(counts)}
        results.append(row)
        if best is None or objective > best['objective'] + 1e-12 or (
                abs(objective-best['objective']) <= 1e-12 and threshold > best['threshold']):
            best = row
    return best, results


def freeze(args):
    if (args.out/'protocol.json').exists():
        raise ValueError('Use a new experiment output; protocol is written only once')
    metadata = read(args.native_data/'metadata.json')
    pages = sorted(metadata['selected_pages']['val'])
    if len(pages) != 120 or len(set(pages)) != 120:
        raise ValueError('Expected exactly the original native 120 validation pages')
    groups = read(args.source_splits)['groups']['train_final']
    fit, calib = split_families(pages, groups)
    development = sorted(path.stem for path in args.traces.glob('*.json'))
    if len(development) != 155:
        raise ValueError('Expected the 155 verified current development traces')
    if {groups[s] for s in pages} & {groups[s] for s in development}:
        raise ValueError('Fit/calibration and development source families overlap')
    files = [Path(__file__), Path('cadvec/hybrid.py'), Path('cadvec/roi_ink.py'), Path('cadvec/native_detail.py'),
             Path('cadvec/detail_unet.py'), Path('cadvec/raster.py'), Path('tools/replay_hybrid.py'),
             Path('tools/diagnose_instance_stages.py'), Path('tools/diagnose_fusion_losses.py')]
    weights = [args.detector, args.coarse, args.native, args.coarse_quality]
    protocol = {'schema_version': 1, 'frozen_at_utc': now(), 'seed': SEED,
        'class': 2, 'features': list(FEATURES), 'transform': 'log1p_nonnegative_except_aspect',
        'learner': {'name': 'LogisticRegression', 'C': .1, 'class_weight': None, 'solver': 'lbfgs',
                    'max_iter': 3000, 'standardization': 'fit-only StandardScaler'},
        'calibration': {'thresholds': list(THRESHOLDS), 'q': Q,
            'objective': 'sum_matched_IoU - q*0.5*predicted_instances', 'tie': 'stricter threshold',
            'matching': 'full-page class2 strict IoU>.5 Hungarian recomputed after every threshold'},
        'minimum_each_label_per_role': 10, 'fit_pages': fit, 'calibration_pages': calib,
        'fit_groups': sorted({groups[s] for s in fit}), 'calibration_groups': sorted({groups[s] for s in calib}),
        'development_pages': development, 'fit_calibration_development_group_overlap': [],
        'candidate_settings': {'detector_imgsz': 640, 'conf': .2, 'iou': .5, 'retina_masks': False,
            'box': 'integer rasterized mask extrema, exactly production', 'canvas': [100, 100],
            'coarse_batch_limit': 8, 'coarse_gate': 'unchanged existing quality model',
            'nms': 'unchanged padded-box .5', 'max_instances': 0, 'native': 'current boundary default; no instance conditioning'},
        'development_rule': 'native-source class2 only uses learned gate; other class2 total>=22; all other rules unchanged',
        'data_metadata_sha256': sha(args.native_data/'metadata.json'), 'source_splits_sha256': sha(args.source_splits),
        'weight_sha256': {str(p): sha(p) for p in weights}, 'source_sha256': {str(p): sha(p) for p in files},
        'limitations': ['120 proxy pages have already been used for native checkpoint selection.',
                       '155 development pages have already been observed; this is no new independent confirmation.',
                       'Fit/calibration labels cannot alter inference candidates; development labels cannot alter model or threshold.']}
    dump(args.out/'protocol.json', protocol)
    dump(args.out/'protocol_extension.json', {'recorded_at_utc': now(), 'purpose': 'root-specified regression entry rule',
        'require_development_delta_PQ_points': .5, 'require_development_TP_not_fall': True,
        'subsequent_checks': 'If eligible, observed 108 and historical 1000 must both raise PQ without reducing TP; root performs those.',
        'default_changed': False})
    return protocol


def infer_fit_calibration(args, protocol):
    from cadvec.layers import load_rgba
    from cadvec.native_detail import NativeDetailRefiner
    from cadvec.pipeline_segdet import _model, _Primitive
    from cadvec.roi_ink import RoiInkRefiner
    from cadvec.schema import assemble_output
    from infer import _prepare_environment
    _prepare_environment(SEED)
    detector = _model(args.detector)
    coarse = RoiInkRefiner.load(args.coarse, device=args.device, max_instances=0, quality_path=args.coarse_quality)
    coarse.native_detail = NativeDetailRefiner.load(args.native, args.device)
    if coarse.native_detail.uses_instance_prompt:
        raise ValueError('The independent quality probe must use the frozen boundary head')
    original = coarse.analyze
    def batched(image, boxes, *values, **kw):
        result = []
        for start in range(0, len(boxes), 8):
            result.extend(original(image, boxes[start:start+8], *values, **kw))
        return result
    coarse.analyze = batched
    provenance = {}
    pages = sorted(protocol['fit_pages']+protocol['calibration_pages'])
    for index, stem in enumerate(pages):
        path = args.data_root/'train_final'/(stem+'.png')
        rgb, alpha = load_rgba(path)
        image = np.clip(rgb*alpha[..., None]+255*(1-alpha[..., None]), 0, 255).astype(np.uint8)
        height, width = image.shape[:2]
        result = detector.predict(source=str(path), imgsz=640, conf=.2, iou=.5,
            verbose=False, device=0 if args.device == 'cuda' else 'cpu', retina_masks=False)[0]
        boxes = []
        if result.boxes is not None and result.masks is not None:
            for polygon, class_id, confidence in zip(result.masks.xy,
                    result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
                if int(class_id) != 2 or polygon is None or len(polygon) < 3:
                    continue
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
                ys, xs = np.nonzero(mask)
                if len(xs):
                    boxes.append((2, float(confidence), (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))))
        boxes.sort(key=lambda b: -b[1])
        accepted = coarse.refine(image, boxes, width, height, 100, 100) if boxes else []
        emitted = []
        for instance, (class_id, confidence, segments) in enumerate(sorted(accepted, key=lambda a: -a[1])):
            emitted.extend(_Primitive(payload=dict(p, semantic=class_id), instance=instance) for p in segments)
        data = assemble_output(emitted, width=100, height=100)
        write_output(data, args.out/'candidates'/(stem+'.json'))
        provenance[stem] = {'png_sha256': sha(path), 'prediction_sha256': sha(args.out/'candidates'/(stem+'.json')),
                            'boxes': boxes, 'accepted': len(accepted)}
        if (index+1) % 20 == 0:
            print('Frozen boundary c2 candidates', index+1, '/', len(pages), flush=True)
    dump(args.out/'candidate_provenance.json', {'completed_at_utc': now(), 'pages': provenance,
        'all_candidates_written_before_label_fitting': True})
    del detector, coarse
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_and_calibrate(args, protocol):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows = []; page_matrices = {}; labels = {}
    dump(args.out/'fit_label_open.json', {'opened_at_utc': now(), 'candidate_provenance_sha256': sha(args.out/'candidate_provenance.json')})
    for stem in sorted(protocol['fit_pages']+protocol['calibration_pages']):
        gt_path = args.data_root/'train_json'/(stem+'.json')
        truth = read(gt_path); data = read(args.out/'candidates'/(stem+'.json'))
        canvas = (truth.get('width', 100), truth.get('height', 100))
        if tuple(canvas) != (100, 100):
            raise ValueError('Fixed prediction canvas differs from labels')
        truth_records = [r for r in segment_records(truth, canvas, 1000) if r.key[0] == 2]
        predictions = segment_records(data, canvas, 1000)
        matches, matrix = strict_matches(truth_records, predictions)
        inverse = {j: score for _, (j, score) in matches.items()}
        for j, candidate in enumerate(predictions):
            rows.append({'stem': stem, 'instance': candidate.key[1], 'features': quality_features(data, candidate.key),
                         'tp': j in inverse, 'matched_IoU': inverse.get(j, 0.), 'candidate_column': j})
        page_matrices[stem] = {'gt_count': len(truth_records), 'pred_count': len(predictions), 'ious': matrix.tolist()}
        labels[stem] = sha(gt_path)
    dump(args.out/'fit_rows.json', rows)
    dump(args.out/'fit_page_matrices.json', page_matrices)
    dump(args.out/'fit_label_sha256.json', labels)
    counts = {}
    for role, pages in (('fit', protocol['fit_pages']), ('calibration', protocol['calibration_pages'])):
        selected = [r for r in rows if r['stem'] in set(pages)]
        counts[role] = {'positive': sum(r['tp'] for r in selected), 'negative': sum(not r['tp'] for r in selected), 'pages': len(pages)}
    dump(args.out/'label_counts.json', counts)
    if any(value[label] < 10 for value in counts.values() for label in ('positive', 'negative')):
        dump(args.out/'result.json', {'status': 'stopped_insufficient_fit_or_calibration_labels', 'counts': counts, 'default_changed': False})
        return None
    fit_rows = [r for r in rows if r['stem'] in set(protocol['fit_pages'])]
    x = np.stack([feature_vector(r['features']) for r in fit_rows]); y = [int(r['tp']) for r in fit_rows]
    scaler = StandardScaler(); x = scaler.fit_transform(x)
    learner = LogisticRegression(C=.1, class_weight=None, solver='lbfgs', max_iter=3000, random_state=SEED)
    learner.fit(x, y)
    model = {'schema_version': 1, 'kind': 'native-postmerge-quality-v1', 'class_id': 2,
        'features': list(FEATURES), 'transform': 'log1p_nonnegative_except_aspect',
        'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': learner.coef_[0].tolist(),
        'intercept': float(learner.intercept_[0]), 'threshold': None, 'fallback_rule': ['total', '>=', 22.],
        'fit_pages': protocol['fit_pages'], 'calibration_pages': protocol['calibration_pages'],
        'protocol_sha256': sha(args.out/'protocol.json'), 'label_counts': counts,
        'label_sha256': labels, 'source_sha256': protocol['source_sha256'], 'weight_sha256': protocol['weight_sha256'],
        'fit_rows_sha256': sha(args.out/'fit_rows.json'), 'fit_page_matrices_sha256': sha(args.out/'fit_page_matrices.json')}
    probability_map = {stem: [0.]*page_matrices[stem]['pred_count'] for stem in protocol['calibration_pages']}
    for row in rows:
        if row['stem'] in probability_map:
            probability_map[row['stem']][row['candidate_column']] = quality_probability(row['features'], model)
    best, scans = calibrate({stem: page_matrices[stem] for stem in protocol['calibration_pages']}, probability_map)
    model['threshold'] = best['threshold']; model['calibration_selection'] = best
    dump(args.out/'calibration_scan.json', scans)
    dump(args.out/'model.json', model)
    dump(args.out/'model_freeze.json', {'frozen_at_utc': now(), 'model_sha256': sha(args.out/'model.json'),
        'calibration_scan_sha256': sha(args.out/'calibration_scan.json'), 'development_labels_opened': False})
    return model


def development(args, protocol, model):
    decisions = {}; hashes = {}
    # Wrapper baseline is checked independently before any learned output.
    for stem in protocol['development_pages']:
        snapshot = read(args.snapshots/(stem+'.json')); trace = read(args.traces/(stem+'.json'))
        reference = read(args.baseline/(stem+'.json'))
        if not trace['verified_current_baseline'] or trace['final'] != reference:
            raise ValueError('Development trace is not verified current baseline')
        updated = dict(snapshot, segdet=trace['segdet'])
        actual = replay(updated, source_identity='legacy', merge_function=make_merge())
        if actual != reference:
            raise ValueError('Original-gate wrapper differs from current production: '+stem)
        write_output(actual, args.out/'development'/'baseline'/(stem+'.json'))
        hashes[stem] = {'snapshot': sha(args.snapshots/(stem+'.json')), 'trace': sha(args.traces/(stem+'.json')),
                        'baseline': sha(args.baseline/(stem+'.json'))}
    dump(args.out/'wrapper_baseline_verification.json', {'pages': len(hashes), 'all_fields_exact': True, 'provenance': hashes})
    for index, stem in enumerate(protocol['development_pages']):
        snapshot = read(args.snapshots/(stem+'.json')); trace = read(args.traces/(stem+'.json'))
        rows = []
        updated = dict(snapshot, segdet=trace['segdet'])
        actual = replay(updated, source_identity='legacy', merge_function=make_merge(model, rows))
        write_output(actual, args.out/'development'/'learned'/(stem+'.json'))
        decisions[stem] = rows
        if (index+1) % 25 == 0:
            print('Fixed learned gate development', index+1, '/', len(protocol['development_pages']), flush=True)
    dump(args.out/'development_decisions.json', decisions)
    dump(args.out/'development_prediction_freeze.json', {'written_at_utc': now(), 'model_sha256': sha(args.out/'model.json'),
        'all_predictions_written_before_scoring': True, 'prediction_sha256': {
            role: {s: sha(args.out/'development'/role/(s+'.json')) for s in protocol['development_pages']}
            for role in ('baseline', 'learned')}})
    from cadvec.evaluate import evaluate_directories
    from tools.summarize_ablation import aggregate, paired_group_bootstrap
    reports = {}
    for role in ('baseline', 'learned'):
        report = evaluate_directories(args.dev_gt, args.out/'development'/role)
        dump(args.out/(role+'_eval.json'), report); reports[role] = report
    metrics = {name: aggregate(list(report['per_image'].values())) for name, report in reports.items()}
    baseline, learned = metrics['baseline'], metrics['learned']
    delta = 100*(learned['PQ']-baseline['PQ'])
    result = {'status': 'completed', 'metrics': metrics, 'delta_PQ_points': delta,
        'paired_uncertainty': paired_group_bootstrap(reports, set(protocol['development_pages']),
            read(args.source_splits)['groups']['train_final'], draws=5000, seed=SEED),
        'regression_entry_rule_passes': delta >= .5 and learned['TP'] >= baseline['TP'],
        'model_sha256': sha(args.out/'model.json'), 'default_changed': False,
        'scope': 'Previously observed 155-page development evidence; not independent confirmation'}
    dump(args.out/'result.json', result)
    print(json.dumps(result), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-data', 'source-splits', 'data-root', 'snapshots', 'traces', 'baseline', 'dev-gt', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--detector', type=Path, default=Path('weights/yolo_seg.pt'))
    p.add_argument('--coarse', type=Path, default=Path('weights/roi_ink_c2.pt'))
    p.add_argument('--native', type=Path, default=Path('weights/native_detail.pt'))
    p.add_argument('--coarse-quality', type=Path, default=Path('weights/roi_quality_512c_all.json'))
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args(argv); cv2.setNumThreads(1)
    protocol = freeze(args)
    print('Protocol frozen', len(protocol['fit_pages']), '/', len(protocol['calibration_pages']), 'pages', flush=True)
    infer_fit_calibration(args, protocol)
    model = fit_and_calibrate(args, protocol)
    if model is not None:
        development(args, protocol, model)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
