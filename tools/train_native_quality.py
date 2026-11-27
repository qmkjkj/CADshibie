#!/usr/bin/env python3
"""Fit a native-aware quality gate on source-grouped old validation pages only."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
os.environ.setdefault('YOLO_OFFLINE','true')
import argparse,hashlib,json,random,sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.evaluate import _segments,load_json
from cadvec.layers import load_rgba
from cadvec.native_detail import NativeDetailRefiner
from cadvec.roi_ink import RoiInkRefiner,vectorize_mask
from cadvec.roi_quality import combined_features
from cadvec.raster import render_items
from cadvec.pipeline_segdet import _model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--metadata',type=Path,default=Path('_work/native_detail_data/metadata.json'))
    p.add_argument('--splits',type=Path,default=Path('_work/clean_splits.json'))
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    meta=json.loads(a.metadata.read_text());splits=json.loads(a.splits.read_text())
    pages=meta['selected_pages']['val'];families=splits['groups']['train_final']
    assert set(pages)<=set(splits['train_final']['val'])
    assert not {families[x] for x in pages}&{families[x] for x in meta['selected_pages']['train']}
    groups=sorted({families[x] for x in pages});random.Random(2712).shuffle(groups)
    fit_groups=set(groups[:round(len(groups)*.75)])
    native=NativeDetailRefiner.load('weights/native_detail.pt','cuda')
    plans=[(classes,RoiInkRefiner.load(path,device='cuda')) for classes,path in
        [({2},'weights/roi_ink_c2.pt'),({3,4,22,26,29},'weights/roi_ink_512.pt'),
         ({8,12,14,15,18,19,20,21,25,27,28},'weights/roi_ink_256.pt')]]
    detector=_model('weights/yolo_seg.pt');rows=[]
    rows_path=a.out.with_suffix('.rows.json')
    fingerprints={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path('weights/native_detail.pt'),Path('weights/yolo_seg.pt'),Path('weights/roi_ink_c2.pt'),
         Path('weights/roi_ink_512.pt'),Path('weights/roi_ink_256.pt'),a.splits,a.metadata,Path(__file__)]}
    if rows_path.exists():
        cached=json.loads(rows_path.read_text())
        if cached.get('fingerprints')==fingerprints:rows=cached['rows']
    if not rows:
        for page_number,stem in enumerate(pages):
            png=a.data_root/'train_final'/(stem+'.png');label=a.data_root/'train_json'/(stem+'.json')
            raw=json.loads(label.read_text());cw,ch=raw['width'],raw['height']
            truth,_=_segments(load_json(label),(cw,ch),1000)
            rgb,alpha=load_rgba(png)
            image=np.clip(rgb*alpha[...,None]+255*(1-alpha[...,None]),0,255).astype(np.uint8)
            h,w=image.shape[:2]
            result=detector.predict(source=str(png),imgsz=640,conf=.2,iou=.5,verbose=False,device=0)[0]
            boxes=[]
            if result.masks is not None and result.boxes is not None:
                for c,conf,polygon in zip(result.boxes.cls.cpu().numpy().astype(int),result.boxes.conf.cpu().numpy(),result.masks.xy):
                    if len(polygon)<3:continue
                    mask=np.zeros((h,w),dtype=np.uint8)
                    cv2.fillPoly(mask,[np.round(polygon).astype(np.int32)],1)
                    y,x=np.nonzero(mask)
                    if len(x):boxes.append((int(c),float(conf),(float(x.min()),float(y.min()),float(x.max()),float(y.max()))))
            boxes.sort(key=lambda b:-b[1]);by_class={}
            for classes,model in plans:
                subset=[b for b in boxes if b[0] in classes]
                if not subset:continue
                analyzed=model.analyze(image,subset,w,h,cw,ch)
                for item in analyzed:
                    c=item['class_id'];l,t,r,b=item['box']
                    probability=native.correct(image[t:b,l:r],item['probability'],c)
                    values=vectorize_mask(probability,l,t,r-l,b-t,w/cw,h/ch,model.threshold)
                    segments=[{'command':0,'points':xy,'length':float(length)} for xy,length in values] or item['segments']
                    features=combined_features(model,item,segments,probability,w,h,cw,ch)
                    candidate={'stem':stem,'family':families[stem],'role':'fit' if families[stem] in fit_groups else 'calib',
                               'class':int(c),'features':features,'tp':False,'iou':0.}
                    by_class.setdefault(c,[]).append((candidate,render_items(segments,1000,cw,ch)))
            for c,items in by_class.items():
                actual=[mask for (label,_),mask in truth.items() if label==c]
                scores=np.zeros((len(actual),len(items)))
                for i,gt in enumerate(actual):
                    for j,(_row,pred) in enumerate(items):
                        inter=np.count_nonzero(gt & pred)
                        if inter:scores[i,j]=inter/np.count_nonzero(gt | pred)
                weights=np.zeros((len(actual)+len(items),)*2)
                weights[:len(actual),:len(items)]=np.where(scores>.5,scores,0.)
                ri,ci=linear_sum_assignment(weights,maximize=True)
                for i,j in zip(ri,ci):
                    if i<len(actual) and j<len(items) and scores[i,j]>.5:
                        items[j][0]['tp']=True;items[j][0]['iou']=float(scores[i,j])
                rows.extend(row for row,_ in items)
            if (page_number+1)%10==0:print('native quality data',page_number+1,'/',len(pages),'rows',len(rows),flush=True)
        rows_path.write_text(json.dumps({'rows':rows,'fingerprints':fingerprints}))
    fit=[r for r in rows if r['role']=='fit'];calib=[r for r in rows if r['role']=='calib']
    names=sorted(k for k in fit[0]['features'] if k.startswith(('coarse_','native_','class_')) or k=='probability_change')
    support=[]
    for c in sorted({r['class'] for r in fit}):
        sample=[r for r in fit if r['class']==c];positive=sum(r['tp'] for r in sample)
        if len(sample)>=30 and positive>=8 and len(sample)-positive>=8:support.append(c)
    fit=[r for r in fit if r['class'] in support];calib=[r for r in calib if r['class'] in support]
    x=np.asarray([[r['features'][k] for k in names] for r in fit]);y=np.asarray([r['tp'] for r in fit])
    scaler=StandardScaler().fit(x);model=LogisticRegression(C=.1,max_iter=3000).fit(scaler.transform(x),y)
    cx=np.asarray([[r['features'][k] for k in names] for r in calib]);prob=model.predict_proba(scaler.transform(cx))[:,1]
    candidates=[]
    for threshold in np.arange(.05,.801,.05):
        kept=[r for r,pr in zip(calib,prob) if pr>=threshold]
        utility=sum(r['iou'] for r in kept)-.33*.5*len(kept)
        candidates.append((utility,float(threshold),len(kept),sum(r['tp'] for r in kept)))
    best=max(candidates,key=lambda v:v[0])
    # Actual default quality filename is resolved through the inference parser.
    from infer import build_parser
    baseline=json.loads(Path(build_parser().parse_args([]).roi_quality_model).read_text())
    payload={'schema_version':2,'features':names,'classes':{},'coarse_fallback':baseline,
             'requires_native_detail':True,'training':{'source':'old native validation pages only',
             'fit_pages':len({r['stem'] for r in fit}),'calib_pages':len({r['stem'] for r in calib}),
             'fit_rows':len(fit),'calib_rows':len(calib),'supported_classes':support,'C':.1,
             'threshold_utility':'sum(matched IoU) - 0.33*0.5*predictions; GT count constant',
             'threshold_candidates':candidates,'selected_threshold':best[1],
             'native_sha256':hashlib.sha256(Path('weights/native_detail.pt').read_bytes()).hexdigest(),
             'splits_sha256':hashlib.sha256(a.splits.read_bytes()).hexdigest()}}
    for c,spec in baseline['classes'].items():payload['classes'][c]=dict(spec,features=baseline.get('features',list(RoiInkRefiner.FEATURES)))
    for c in support:
        payload['classes'][str(c)]={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),
            'threshold':best[1],'features':names,'feature_stage':'native'}
    a.out.write_text(json.dumps(payload,indent=1));print('Trained native quality',payload['training'],flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
