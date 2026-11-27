#!/usr/bin/env python3
"""Compare raw detector box coverage and strict final matches on labeled pages.

Box coverage is a localization diagnostic, not an estimate of achievable PQ.
"""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.evaluate import load_json,_segments
from tools.compare_instance_matches import matched_truth


def bbox(mask):
    y,x=np.nonzero(mask)
    return (x.min()-.5,y.min()-.5,x.max()+.5,y.max()+.5) if len(x) else None


def iou(a,b):
    if a is None:return 0.
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-9)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    classes={};pages=[]
    for path in sorted((a.base/'stages').glob('*.json')):
        stage=json.loads(path.read_text());raw=json.loads((a.base/'gt'/path.name).read_text())
        canvas=(raw.get('width',100),raw.get('height',100))
        truth,_=_segments(load_json(a.base/'gt'/path.name),canvas,1000)
        predicted,_=_segments(load_json(a.base/'baseline'/path.name),canvas,1000)
        final=matched_truth(truth,predicted)
        height,width=stage['support']['size']
        dets=stage['support']['detections']
        boxes=[((d['box'][0]-.5)*1000/width,(d['box'][1]-.5)*1000/height,
                (d['box'][2]+.5)*1000/width,(d['box'][3]+.5)*1000/height) for d in dets]
        gtkeys=[k for k in truth if k[0]<30]
        scores=np.array([[iou(bbox(truth[k]),b) for b in boxes] for k in gtkeys]).reshape(len(gtkeys),len(boxes))
        local=Counter()
        for index,key in enumerate(gtkeys):
            c=key[0];row=classes.setdefault(c,Counter());row['gt']+=1
            same=[scores[index,j] for j,d in enumerate(dets) if d['class']==c]
            correct=max(same,default=0)>.5;any_class=max(scores[index],default=0)>.5
            row['correct_class_box_coverage']+=int(correct);row['any_class_box_coverage']+=int(any_class)
            bucket=('final_matched' if key in final else
                    'unmatched_with_correct_class_box' if correct else
                    'unmatched_with_only_other_class_box' if any_class else 'unmatched_without_box')
            row[bucket]+=1;local[bucket]+=1
        pages.append({'page':path.stem,**local})
    total=sum(classes.values(),Counter())
    output={'totals':dict(total),'per_class':{str(c):dict(s) for c,s in sorted(classes.items())},'pages':pages,
            'note':'Independent GT box coverage at IoU > .5; proposals may cover multiple GT. Final matches use strict 1px scorer. Unmatched_without_box is a heuristic bucket, not proven detector causality.'}
    (a.base/'proposal_diagnosis.json').write_text(json.dumps(output,indent=1))
    print(json.dumps(output['totals']))
    return 0

if __name__=='__main__':raise SystemExit(main())
