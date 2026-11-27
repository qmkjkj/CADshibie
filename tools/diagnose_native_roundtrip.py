#!/usr/bin/env python3
"""Measure the crop proxy lost by vectorizing perfect native target masks.

This offline representation diagnostic uses labels. Its numbers are neither
deployable predictions nor page-level PQ and cannot be reported as a score.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_native_detail import vector_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--crops', type=int, default=500)
    parser.add_argument('--seed', type=int, default=73)
    args = parser.parse_args()
    rows = [r for r in json.loads((args.data/'index.json').read_text()) if r['role']=='val']
    random.Random(args.seed).shuffle(rows)
    if args.crops > 0:
        rows = rows[:args.crops]
    totals = {'perfect_target':np.zeros(4), 'coarse':np.zeros(4)}
    classes = {}; failed = []
    for r in rows:
        with np.load(args.data/r['path']) as d:
            h,w=d['valid_shape']
            target=d['target'].astype(bool)
            coarse=d['image'][...,3]/255.
        valid = np.zeros_like(target); valid[:h,:w]=True
        scores = {n:vector_score(p,target,valid,r['coarse_threshold'],r['scale_xy'])
                  for n,p in (('perfect_target',target.astype(float)),('coarse',coarse))}
        entry = classes.setdefault(str(r['class']),{'crops':0,'perfect_target':np.zeros(4),'coarse':np.zeros(4)})
        entry['crops'] += 1
        for n,s in scores.items():
            totals[n] += s; entry[n] += s
        if target.any() and not scores['perfect_target'][0]:
            failed.append({'path':r['path'],'class':r['class'],'target_pixels':int(target.sum())})
    def metrics(a):
        tp,fp,fn,iou=a
        return {'TP':int(tp),'FP':int(fp),'FN':int(fn),'sum_iou':float(iou),
                'crop_proxy_PQ':float(iou/max(tp+.5*(fp+fn),1))}
    result={'crops':len(rows),'positive_crops':sum(r['positive'] for r in rows),
            'metrics':{n:metrics(t) for n,t in totals.items()},
            'per_class':{c:{'crops':v['crops'],**{n:metrics(v[n]) for n in totals}} for c,v in classes.items()},
            'perfect_target_failures':failed,
            'note':'Label-using representation oracle on partial ROI tiles, not page PQ or deployable output.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=1))
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_class','perfect_target_failures')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
