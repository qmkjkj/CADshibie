#!/usr/bin/env python3
"""Trace current verified segdet matches through freshly replayed fusion stages.

This is a label-using diagnostic of observed development pages. It changes no
production configuration and never reuses historical merged/final geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.anchor_assign import reassign
from cadvec.atom_assign import assign_atoms
from cadvec.evaluate import _metrics
from cadvec.hybrid import merge_final
from cadvec.stuff_branch import append_stuff
from tools.diagnose_instance_stages import metric_counts, segment_records, strict_matches


STAGES = ('segdet', 'merge_without_quality', 'merge_quality', 'anchor', 'atoms', 'stuff')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def recompute_stages(snapshot, trace, reference):
    if (not trace.get('verified_current_baseline') or trace['final'] != reference
            or trace.get('source_identity') != 'legacy'):
        raise ValueError('A verified current legacy-identity trace is required')
    options = dict(snapshot['merge_options'])
    for name in ('gat_classes', 'segdet_classes', 'gate_classes', 'segdet_union_classes', 'suppress'):
        options[name] = set(options[name])
    options['quality_rules'] = {int(k): tuple(v) for k, v in (options['quality_rules'] or {}).items()}
    options['source_identity'] = 'legacy'
    stages = {'segdet': trace['segdet']}
    stages['merge_without_quality'] = merge_final(snapshot['primary'], snapshot['gat'], trace['segdet'],
        support=snapshot['support'], **dict(options, quality_rules={}))
    data = merge_final(snapshot['primary'], snapshot['gat'], trace['segdet'], support=snapshot['support'], **options)
    stages['merge_quality'] = data
    assignment = snapshot['assignment']; support = snapshot.get('support') or {}
    detections = support.get('detections') or []; size = support.get('size') or (0, 0)
    if detections and min(size) > 0 and assignment['anchor_classes']:
        data = reassign(data, detections, set(assignment['anchor_classes']), int(size[1]), int(size[0]),
                        inside=assignment['anchor_inside'], affinity=assignment['anchor_affinity'])
    stages['anchor'] = data
    if detections and min(size) > 0 and assignment['atom_classes'] and assignment['atom_affinity']:
        data = assign_atoms(data, detections, set(assignment['atom_classes']), int(size[1]), int(size[0]),
                            assignment['atom_affinity'])
    stages['atoms'] = data
    if snapshot.get('stuff_items'):
        data = append_stuff(data, snapshot['stuff_items'], data['width'], data['height'])
    stages['stuff'] = data
    if data != reference:
        raise ValueError('Fresh current-segdet replay differs from the verified current final')
    return stages


def primitive_signatures(data):
    groups = defaultdict(list)
    for semantic, instance, command, args in zip(data['semanticIds'], data['instanceIds'], data['commands'], data['args']):
        groups[(int(semantic), int(instance))].append((int(command), tuple(map(float, args))))
    return {key: tuple(sorted(value)) for key, value in groups.items()}


def quality_features(data, key):
    selected = [i for i, values in enumerate(zip(data['semanticIds'], data['instanceIds'])) if tuple(values) == key]
    points = [data['args'][i][k:k+2] for i in selected for k in range(0, 8, 2)]
    lengths = [float(data['lengths'][i]) for i in selected]
    if not points:
        return {}
    width = max(p[0] for p in points) - min(p[0] for p in points)
    height = max(p[1] for p in points) - min(p[1] for p in points)
    total = sum(lengths)
    return {'n': float(len(selected)), 'total': total, 'mean': total / len(selected),
            'max': max(lengths), 'w': width, 'h': height, 'area': max(width*height, 1e-6),
            'aspect': min(width, height) / max(width, height, 1e-6),
            'density': total / max(width + height, 1e-6)}


def last_loss_stage(matched):
    losses = [STAGES[i] for i in range(1, len(STAGES)) if matched[STAGES[i-1]] and not matched[STAGES[i]]]
    return losses[-1] if losses else None


def diagnose_page(snapshot, trace, reference, raw_truth):
    outputs = recompute_stages(snapshot, trace, reference)
    canvas = (raw_truth.get('width', 100), raw_truth.get('height', 100))
    truths = segment_records(raw_truth, canvas, 1000)
    metrics = {}; details = {i: {} for i in range(len(truths))}
    matched_sets = {}; signatures = {}; per_class = {}
    for name in STAGES:
        data = outputs[name]
        predictions = segment_records(data, canvas, 1000)
        matches, matrix = strict_matches(truths, predictions)
        matched_sets[name] = set(matches)
        metrics[name] = {'all': metric_counts(truths, predictions, matches)}
        thing_truth = [r for r in truths if r.key[0] < 30]
        thing_predictions = [r for r in predictions if r.key[0] < 30]
        thing_matches = {i: value for i, value in matches.items() if truths[i].key[0] < 30}
        metrics[name]['things'] = metric_counts(thing_truth, thing_predictions, thing_matches)
        signatures[name] = primitive_signatures(data)
        per_class[name] = {}
        for class_id in sorted({r.key[0] for r in truths} | {r.key[0] for r in predictions}):
            selected_matches = {i: value for i, value in matches.items() if truths[i].key[0] == class_id}
            per_class[name][str(class_id)] = metric_counts([r for r in truths if r.key[0] == class_id],
                [r for r in predictions if r.key[0] == class_id], selected_matches)
        for i in range(len(truths)):
            detail = {'matched': i in matches, 'best_same_class_IoU': float(matrix[i].max()) if len(predictions) else 0.,
                      'strict_potential_count': int((matrix[i] > .5).sum())}
            if i in matches:
                j, score = matches[i]
                detail.update(prediction_key=list(predictions[j].key), IoU=score)
            details[i][name] = detail
    lost = sorted(i for i in matched_sets['segdet']-matched_sets['stuff'] if truths[i].key[0] < 30)
    cases = []
    for i in lost:
        row = {'gt_key': list(truths[i].key), 'stages': details[i]}
        source_key = tuple(details[i]['segdet']['prediction_key'])
        original = signatures['segdet'][source_key]
        for name in STAGES:
            row['stages'][name]['original_primitive_multiset_preserved'] = original in signatures[name].values()
        matches = {name: details[i][name]['matched'] for name in STAGES}
        stage = last_loss_stage(matches)
        row['last_loss_stage'] = stage
        row['loss_transitions'] = [STAGES[n] for n in range(1, len(STAGES))
                                  if matches[STAGES[n-1]] and not matches[STAGES[n]]]
        row['cause'] = ('quality_rule_deletion' if stage == 'merge_quality' else
                        'fusion_before_quality' if stage == 'merge_without_quality' else
                        'anchor_assignment' if stage == 'anchor' else
                        'atom_assignment' if stage == 'atoms' else 'stuff_replacement')
        if stage == 'merge_without_quality':
            options = snapshot['merge_options']; class_id = truths[i].key[0]
            row['fusion_rules'] = {'class_suppressed': class_id in options['suppress'],
                'segdet_replacement': class_id in options['segdet_classes'],
                'segdet_union_after_suppression': class_id in options['segdet_union_classes']}
            if row['fusion_rules']['class_suppressed'] and not row['fusion_rules']['segdet_union_after_suppression']:
                row['cause'] = 'class_suppression'
        if details[i][stage]['strict_potential_count']:
            row['cause'] += '_with_one_to_one_competition'
        if details[i]['merge_without_quality']['matched']:
            key = tuple(details[i]['merge_without_quality']['prediction_key'])
            features = quality_features(outputs['merge_without_quality'], key)
            rule = (snapshot['merge_options'].get('quality_rules') or {}).get(str(truths[i].key[0]))
            if rule is None:
                rule = (snapshot['merge_options'].get('quality_rules') or {}).get(truths[i].key[0])
            if rule:
                feature, operator, threshold = rule
                value = features[feature]
                row['quality_rule'] = {'feature': feature, 'operator': operator, 'threshold': threshold,
                                       'value': value, 'passes': value >= threshold if operator == '>=' else value <= threshold}
        cases.append(row)
    transitions = {}
    for before, after in zip(STAGES, STAGES[1:]):
        thing = {i for i, record in enumerate(truths) if record.key[0] < 30}
        transitions[after] = {'lost': len((matched_sets[before]-matched_sets[after]) & thing),
                              'gained': len((matched_sets[after]-matched_sets[before]) & thing)}
    return {'metrics': metrics, 'per_class': per_class, 'thing_transitions': transitions,
            'segdet_to_final': {'lost': len(lost), 'gained': len([i for i in matched_sets['stuff']-matched_sets['segdet']
                                                             if truths[i].key[0] < 30])},
            'lost_cases': cases, 'output_sha256': {name: digest(data) for name, data in outputs.items()}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--gt-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv); cv2.setNumThreads(1)
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    pages = {}; provenance = {}; causes = Counter(); by_class = defaultdict(Counter)
    totals = {name: {kind: Counter() for kind in ('all', 'things')} for name in STAGES}
    classes = {name: defaultdict(Counter) for name in STAGES}; transitions = defaultdict(Counter)
    segdet_to_final = Counter()
    paths = sorted(args.traces.glob('*.json'))
    if not paths:
        raise ValueError('No verified current traces')
    for index, path in enumerate(paths):
        inputs = {'snapshot': args.snapshots/path.name, 'trace': path,
                  'reference': args.baseline/path.name, 'raw_truth': args.gt_dir/path.name}
        provenance[path.name] = {key: sha(value) for key, value in inputs.items()}
        page = diagnose_page(**{key: read(value) for key, value in inputs.items()})
        pages[path.stem] = page
        for case in page['lost_cases']:
            causes[case['cause']] += 1; by_class[str(case['gt_key'][0])][case['cause']] += 1
        for name in STAGES:
            for kind in ('all', 'things'):
                totals[name][kind].update(page['metrics'][name][kind])
            for class_id, row in page['per_class'][name].items():
                classes[name][class_id].update(row)
        for name, row in page['thing_transitions'].items():
            transitions[name].update(row)
        segdet_to_final.update(page['segdet_to_final'])
        if (index+1) % 25 == 0:
            print('Fresh fusion stage replay', index+1, '/', len(paths), flush=True)
    def metrics(values):
        return {**dict(values), **_metrics(values['TP'], values['FP'], values['FN'], values['sum_iou'])}
    result = {'schema_version': 1, 'pages': len(pages), 'verified_current_final_pages': len(pages),
        'method': 'Replace historical snapshot segdet with verified current trace.segdet; replay each production stage afresh using legacy identity.',
        'raster_size': 1000, 'strict_IoU_threshold': .5,
        'segdet_to_final_things': dict(segdet_to_final), 'lost_causes': dict(causes),
        'lost_causes_per_class': {key: dict(value) for key, value in by_class.items()},
        'metrics': {name: {kind: metrics(value) for kind, value in kinds.items()} for name, kinds in totals.items()},
        'per_class': {name: {key: metrics(value) for key, value in rows.items()} for name, rows in classes.items()},
        'thing_transitions': {key: dict(value) for key, value in transitions.items()},
        'limitations': ['Observed development data; diagnostic only, not an independent confirmation.',
                       'merge_without_quality is a fixed counterfactual for attribution, not a proposed threshold change.',
                       'Deleting strict matches may also remove many false positives; loss counts alone cannot choose production rules.',
                       'last_loss_stage records the final matched-to-unmatched transition; all intermediate recoveries are retained.',
                       'Primitive multiset preservation is an exact command/coordinate check, not raster equivalence.'],
        'tool_sha256': sha(__file__), 'source_sha256': {name: sha(name) for name in
            ('cadvec/hybrid.py', 'cadvec/anchor_assign.py', 'cadvec/atom_assign.py', 'cadvec/stuff_branch.py',
             'tools/diagnose_instance_stages.py')}, 'provenance': provenance, 'per_page': pages}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('pages', 'segdet_to_final_things', 'lost_causes', 'lost_causes_per_class', 'thing_transitions')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
