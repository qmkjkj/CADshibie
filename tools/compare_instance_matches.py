#!/usr/bin/env python3
"""Identify old ground-truth matches lost and new matches gained by a change."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.evaluate import _segments, load_json


def matched_truth(gt_segments, pred_segments):
    matched = set()
    for class_id in sorted({key[0] for key in gt_segments}):
        truth = [(key,mask) for key,mask in gt_segments.items() if key[0]==class_id]
        predictions = [mask for key,mask in pred_segments.items() if key[0]==class_id]
        if not predictions:
            continue
        ious = np.zeros((len(truth),len(predictions)))
        for i,(_key,actual) in enumerate(truth):
            for j,predicted in enumerate(predictions):
                union = np.count_nonzero(actual | predicted)
                if union:
                    ious[i,j] = np.count_nonzero(actual & predicted)/union
        weights = np.zeros((len(truth)+len(predictions),)*2)
        weights[:len(truth),:len(predictions)] = np.where(ious>.5,ious,0.)
        rows,columns = linear_sum_assignment(weights,maximize=True)
        matched.update(truth[i][0] for i,j in zip(rows,columns)
                       if i<len(truth) and j<len(predictions) and ious[i,j]>.5)
    return matched


def compare(gt_dir, baseline_dir, candidate_dir):
    rows = []; totals = Counter(); per_class = {}
    for path in sorted(gt_dir.glob('*.json')):
        raw = json.loads(path.read_text(encoding='utf-8'))
        canvas = (raw.get('width',100),raw.get('height',100))
        truth,_ = _segments(load_json(path),canvas,1000)
        old,_ = _segments(load_json(baseline_dir/path.name),canvas,1000)
        new,_ = _segments(load_json(candidate_dir/path.name),canvas,1000)
        before,after = matched_truth(truth,old),matched_truth(truth,new)
        values = {'old_TP':len(before),'new_TP':len(after),
                  'retained':len(before & after),'lost':len(before-after),
                  'gained':len(after-before)}
        totals.update(values)
        for name,keys in (('retained',before&after),('lost',before-after),('gained',after-before)):
            for class_id,_instance_id in keys:
                per_class.setdefault(class_id,Counter())[name] += 1
        if before != after:
            rows.append({'page':path.stem,'lost':sorted(before-after),'gained':sorted(after-before)})
    return {**dict(totals),'per_class':{str(c):dict(v) for c,v in sorted(per_class.items())},
            'changed_pages':rows,'note':'Ground-truth matches under the same strict IoU > .5 Hungarian matching; prediction instance IDs are not compared.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--candidate-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = compare(args.gt_dir,args.baseline_dir,args.candidate_dir)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=1),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_class','changed_pages')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
