#!/usr/bin/env python3
"""Diagnose observed CAD instance failures without assigning unsupported causes.

Boxes describe candidate availability, not strict vector IoU. Historical branch
snapshots are compared with their own final output separately from the current
prediction. Optional verified ROI traces make current-run stage analysis possible.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.evaluate import _metrics, _segments


@dataclass
class Segment:
    key: tuple[int, int]
    mask: np.ndarray
    area: int
    box: tuple[float, float, float, float]


def segment_records(data, canvas, raster_size):
    records = []
    for key, mask in _segments(data, canvas, raster_size)[0].items():
        ys, xs = np.nonzero(mask)
        # Use half-open raster bounds so thin horizontal/vertical symbols have
        # nonzero area. The detector box is transformed into this raster frame.
        box = (float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)) if xs.size else (0., 0., 0., 0.)
        records.append(Segment(key, mask, int(xs.size), box))
    return records


def box_iou(first, second):
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0., x1-x0) * max(0., y1-y0)
    first_area = max(0., first[2]-first[0]) * max(0., first[3]-first[1])
    second_area = max(0., second[2]-second[0]) * max(0., second[3]-second[1])
    union = first_area + second_area - intersection
    return intersection/union if union else 0.


def mask_iou(first, second):
    x0 = int(max(first.box[0], second.box[0]))
    y0 = int(max(first.box[1], second.box[1]))
    x1 = int(min(first.box[2], second.box[2]))
    y1 = int(min(first.box[3], second.box[3]))
    if x0 >= x1 or y0 >= y1:
        return 0.
    intersection = int(np.count_nonzero(first.mask[y0:y1,x0:x1] & second.mask[y0:y1,x0:x1]))
    union = first.area + second.area - intersection
    return intersection/union if union else 0.


def assign(ious, eligible, *, count_first=False):
    """One-to-one assignment with explicit unmatched rows/columns.

    Strict geometry uses the evaluator's sum-IoU objective. Box availability uses
    maximum eligible cardinality then IoU; this diagnostic is not the evaluator.
    """
    n, m = ious.shape
    if not n or not m:
        return {}
    weights = np.zeros((n+m, n+m), dtype=float)
    bonus = min(n,m)+1 if count_first else 0.
    weights[:n,:m] = np.where(eligible, ious+bonus, 0.)
    rows, columns = linear_sum_assignment(weights, maximize=True)
    return {int(i): (int(j), float(ious[i,j])) for i,j in zip(rows,columns)
            if i < n and j < m and eligible[i,j]}


def geometry_matrix(truth, predicted):
    matrix = np.zeros((len(truth), len(predicted)), dtype=float)
    for i, actual in enumerate(truth):
        for j, candidate in enumerate(predicted):
            if actual.key[0] == candidate.key[0]:
                matrix[i,j] = mask_iou(actual, candidate)
    return matrix


def strict_matches(truth, predicted):
    matrix = geometry_matrix(truth, predicted)
    return assign(matrix, matrix > .5), matrix


def detection_coverage(truth, detections, detector_size, raster_size, threshold=.5):
    height, width = detector_size
    if width <= 0 or height <= 0:
        raise ValueError('Invalid cached detector size')
    boxes = [(record['box'][0]*raster_size/width, record['box'][1]*raster_size/height,
              record['box'][2]*raster_size/width, record['box'][3]*raster_size/height)
             for record in detections]
    matrix = np.asarray([[box_iou(actual.box, box) for box in boxes] for actual in truth], dtype=float)
    matrix = matrix.reshape(len(truth), len(boxes))
    same = np.asarray([[actual.key[0] == int(record['class']) for record in detections]
                       for actual in truth], dtype=bool).reshape(matrix.shape)
    eligible = matrix >= threshold
    return matrix, same, assign(matrix, eligible & same, count_first=True), assign(matrix, eligible, count_first=True)


def metric_counts(truth, predicted, matches):
    return {'TP':len(matches), 'FP':len(predicted)-len(matches),
            'FN':len(truth)-len(matches), 'sum_iou':sum(value[1] for value in matches.values())}


def trace_analysis(trace, current, truth, canvas, raster_size, final_matches):
    if not trace.get('verified_current_baseline') or trace.get('final') != current:
        raise ValueError('ROI trace is not verified against current prediction')
    stage = segment_records(trace['segdet'], canvas, raster_size)
    stage_matches, _ = strict_matches(truth, stage)
    candidates = trace.get('candidates', [])
    native_records, decisions, candidate_ids = [], [], []
    for candidate in candidates:
        if not candidate.get('native'):
            continue
        records = segment_records(candidate['native'], canvas, raster_size)
        things = [record for record in records if record.key[0] < 30]
        if len(things) > 1:
            raise ValueError('Each ROI candidate must contain at most one Thing instance')
        for record in things:
            native_records.append(record)
            decisions.append(candidate['decision'])
            candidate_ids.append(candidate['candidate_id'])
    matrix = geometry_matrix(truth, native_records)
    result = {'candidates_by_decision':dict(Counter(c['decision'] for c in candidates)),
              'segdet':metric_counts(truth,stage,stage_matches), 'truth':{}}
    all_matches = assign(matrix, matrix>.5)
    accepted_columns = [i for i, decision in enumerate(decisions) if decision == 'accepted']
    rejected_columns = [i for i, decision in enumerate(decisions) if decision != 'accepted']
    accepted = matrix[:,accepted_columns]
    rejected = matrix[:,rejected_columns]
    accepted_matches = assign(accepted, accepted>.5)
    rejected_matches = assign(rejected, rejected>.5)
    result.update(native_all_one_to_one=len(all_matches), native_accepted_one_to_one=len(accepted_matches),
                  native_rejected_one_to_one=len(rejected_matches),
                  native_potential_by_decision={})
    # Potential counts are independent existential diagnostics, can overlap, and
    # must not be added across decisions to estimate obtainable TP.
    for decision in sorted(set(decisions)):
        indices = [i for i,d in enumerate(decisions) if d == decision]
        result['native_potential_by_decision'][decision] = int(np.any(matrix[:,indices]>.5,axis=1).sum())
    counters = Counter()
    for i, actual in enumerate(truth):
        if actual.key[0] >= 30:
            continue
        qualifying = np.flatnonzero(matrix[i]>.5)
        potential_decisions = sorted({decisions[j] for j in qualifying})
        accepted_potential = 'accepted' in potential_decisions
        rejected_potential = any(d != 'accepted' for d in potential_decisions)
        values = {'accepted_native_potential':accepted_potential, 'rejected_native_potential':rejected_potential,
                  'current_segdet_matched':i in stage_matches,
                  'current_segdet_match_lost_to_final':i in stage_matches and i not in final_matches,
                  'accepted_potential_but_segdet_unmatched':accepted_potential and i not in stage_matches,
                  'accepted_potential_but_final_unmatched':accepted_potential and i not in final_matches,
                  'rejected_only_potential_but_final_unmatched':rejected_potential and not accepted_potential and i not in final_matches}
        counters.update({key:int(value) for key,value in values.items()})
        if i not in final_matches:
            values['final_FN_stage_bucket'] = (
                'current_segdet_match_lost_to_final' if i in stage_matches else
                'accepted_native_potential_unmatched_in_segdet' if accepted_potential else
                'rejected_native_only_potential' if rejected_potential else
                'no_current_native_ROI_strict_potential')
        result['truth'][str(i)] = {**values, 'potential_decisions':potential_decisions,
                                   'potential_candidate_ids':[candidate_ids[j] for j in qualifying]}
    result['thing_counts'] = dict(counters)
    return result


def diagnose_page(snapshot, current, raw_truth, *, raster_size=1000, box_threshold=.5, trace=None):
    canvas = (raw_truth.get('width',100), raw_truth.get('height',100))
    truth = segment_records(raw_truth, canvas, raster_size)
    latest = segment_records(current, canvas, raster_size)
    final_matches, final_matrix = strict_matches(truth, latest)
    old_segdet = segment_records(snapshot['segdet'], canvas, raster_size)
    old_final = segment_records(snapshot['final'], canvas, raster_size)
    old_segdet_matches, _ = strict_matches(truth, old_segdet)
    old_final_matches, _ = strict_matches(truth, old_final)
    things = [(i,record) for i,record in enumerate(truth) if record.key[0]<30]
    support = snapshot.get('support') or {}
    detections = support.get('detections') or []
    coverage, same, same_assignment, any_assignment = detection_coverage(
        [record for _,record in things], detections, support.get('size',(raster_size,raster_size)), raster_size, box_threshold)
    counters, per_class, details = Counter(), defaultdict(Counter), []
    old_lost = set(old_segdet_matches)-set(old_final_matches)
    for local_i, (i, actual) in enumerate(things):
        same_values = coverage[local_i][same[local_i]]
        wrong_values = coverage[local_i][~same[local_i]]
        best_same = float(same_values.max()) if same_values.size else 0.
        best_wrong = float(wrong_values.max()) if wrong_values.size else 0.
        if i in final_matches:
            bucket = 'final_matched'
        elif best_same >= box_threshold:
            bucket = 'final_FN_same_class_box_available'
        elif best_wrong >= box_threshold:
            bucket = 'final_FN_only_other_class_box_available'
        else:
            bucket = 'final_FN_no_box_at_threshold'
        row = {'gt_key':list(actual.key), 'gt_raster_box':actual.box, 'bucket':bucket,
               'best_same_class_box_iou':best_same, 'best_other_class_box_iou':best_wrong,
               'same_class_box_assignment':same_assignment.get(local_i),
               'class_agnostic_box_assignment':any_assignment.get(local_i),
               'best_current_same_class_geometry_iou':float(final_matrix[i].max()) if latest else 0.,
               'historical_segdet_matched':i in old_segdet_matches,
               'historical_final_matched':i in old_final_matches,
               'historical_segdet_match_lost_to_own_final':i in old_lost,
               'current_final_matched':i in final_matches}
        if i in final_matches:
            pred_index, iou = final_matches[i]
            row['current_prediction_key'] = list(latest[pred_index].key)
            row['current_match_iou'] = iou
        counts = {'GT':1, bucket:1, 'same_class_box_one_to_one':int(local_i in same_assignment),
                  'class_agnostic_box_one_to_one':int(local_i in any_assignment),
                  'same_class_box_potential':int(best_same >= box_threshold),
                  'other_class_box_potential':int(best_wrong >= box_threshold),
                  'historical_segdet_TP':int(i in old_segdet_matches),
                  'historical_final_TP':int(i in old_final_matches),
                  'historical_segdet_match_lost_to_own_final':int(i in old_lost)}
        if i not in final_matches:
            best = row['best_current_same_class_geometry_iou']
            interval = ('zero' if best == 0 else '0_to_0.25' if best < .25 else
                        '0.25_to_0.4' if best < .4 else '0.4_to_0.5' if best <= .5 else 'over_0.5_competition')
            counts['final_FN_geometry_'+interval] = 1
        counters.update(counts); per_class[actual.key[0]].update(counts)
        details.append(row)
    result = {'file':snapshot.get('file'), 'current':metric_counts(truth,latest,final_matches),
              'historical_segdet':metric_counts(truth,old_segdet,old_segdet_matches),
              'historical_final':metric_counts(truth,old_final,old_final_matches),
              'thing_counts':dict(counters), 'per_class':{str(k):dict(v) for k,v in per_class.items()},
              'detections':len(detections), 'instances':details}
    if trace is not None:
        analysis = trace_analysis(trace,current,truth,canvas,raster_size,final_matches)
        for local_i,(i,_actual) in enumerate(things):
            values = analysis['truth'][str(i)]
            details[local_i]['roi_trace'] = values
            numeric = {key:int(value) for key,value in values.items() if isinstance(value,bool)}
            if 'final_FN_stage_bucket' in values:
                numeric['FN_stage_'+values['final_FN_stage_bucket']] = 1
            per_class[truth[i].key[0]].update(numeric)
        result['per_class'] = {str(k):dict(v) for k,v in per_class.items()}
        result['roi_trace'] = {key:value for key,value in analysis.items() if key!='truth'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages',type=Path,required=True)
    parser.add_argument('--gt-dir',type=Path,required=True)
    parser.add_argument('--pred-dir',type=Path,required=True)
    parser.add_argument('--roi-traces',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--raster-size',type=int,default=1000)
    parser.add_argument('--box-threshold',type=float,default=.5)
    args = parser.parse_args()
    pages, counters, per_class = {}, Counter(), defaultdict(Counter)
    stage_totals = {key:Counter() for key in ('current','historical_segdet','historical_final')}
    trace_totals = Counter(); potential_totals = Counter(); candidate_totals = Counter()
    fn_stage_by_boxes = defaultdict(Counter)
    provenance = {}
    paths = sorted(args.gt_dir.glob('*.json'))
    if not paths:
        raise ValueError('No ground truth JSON files')
    for index,path in enumerate(paths):
        input_paths = {'truth':path,'stage':args.stages/path.name,'current':args.pred_dir/path.name}
        if args.roi_traces:
            input_paths['trace'] = args.roi_traces/path.name
        values = {key:json.loads(value.read_text(encoding='utf-8')) for key,value in input_paths.items()}
        provenance[path.name] = {key:hashlib.sha256(value.read_bytes()).hexdigest() for key,value in input_paths.items()}
        result = diagnose_page(values['stage'],values['current'],values['truth'],raster_size=args.raster_size,
                               box_threshold=args.box_threshold,trace=values.get('trace'))
        pages[path.stem] = result; counters.update(result['thing_counts'])
        for key in stage_totals:
            stage_totals[key].update(result[key])
        for class_id,row in result['per_class'].items():
            per_class[class_id].update(row)
        if 'roi_trace' in result:
            row = result['roi_trace']; trace_totals.update(row['thing_counts'])
            trace_totals.update({key:row[key] for key in ('native_all_one_to_one','native_accepted_one_to_one','native_rejected_one_to_one')})
            potential_totals.update(row['native_potential_by_decision']); candidate_totals.update(row['candidates_by_decision'])
            for key,value in row['segdet'].items():
                trace_totals['current_segdet_'+key] += value
            for instance in result['instances']:
                if not instance['current_final_matched']:
                    fn_stage_by_boxes[instance['bucket']][instance['roi_trace']['final_FN_stage_bucket']] += 1
        if (index+1)%25 == 0:
            print(f'Diagnosed {index+1}/{len(paths)}',flush=True)
    report = {'schema_version':1,'pages':len(pages),'raster_size':args.raster_size,
              'box_threshold':args.box_threshold,'thing_counts':dict(counters),
              'metrics':{key:{**dict(value),**_metrics(int(value['TP']),int(value['FP']),int(value['FN']),value['sum_iou'])}
                         for key,value in stage_totals.items()},
              'per_class':{key:dict(value) for key,value in sorted(per_class.items(),key=lambda item:int(item[0]))},
              'limits':['Observed development data only; not independent generalization evidence.',
                        'Box threshold coverage is diagnostic and cannot prove detector miss causality.',
                        'Cached detector boxes are post detector-internal confidence/NMS filtering; no raw pre-NMS trace.',
                        'Historical segdet to historical final changes belong to the historical model run.',
                        'Historical segdet versus current final must not be attributed to current postprocessing.',
                        'Without verified current ROI traces, empty/quality/NMS rejection causes cannot be separated.',
                        'Independent native potentials may overlap across decisions; they are not additive obtainable TP.'],
              'provenance':provenance,'per_page':pages}
    if args.roi_traces:
        report['roi_trace'] = {'counts':dict(trace_totals),'native_potential_by_decision':dict(potential_totals),
                               'candidates_by_decision':dict(candidate_totals),
                               'final_FN_by_box_and_current_stage':{key:dict(value) for key,value in fn_stage_by_boxes.items()}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=1),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('pages','thing_counts','metrics','roi_trace') if key in report}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
