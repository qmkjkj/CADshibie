#!/usr/bin/env python3
"""Replay frozen hybrid branches with paired ROI selection experiments.

Offline experiment only. Neural predictions are cached across selection modes
on each page; labels are opened only after every prediction has been written.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
os.environ.setdefault('YOLO_OFFLINE','true')
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.config import InferenceConfig
from cadvec.native_detail import NativeDetailRefiner
from cadvec.roi_ink import RoiInkRefiner, vectorize_mask
from cadvec.raster import render_items
from cadvec.roi_quality import combined_features,score as quality_score
from cadvec.schema import write_output
from cadvec.evaluate import evaluate_directories
from tools.replay_hybrid import replay
from infer import build_parser, _prepare_environment
import cadvec.pipeline_segdet as segdet


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection,1e-9)


def experimental_refine(self,image,boxes,width,height,canvas_width=100,canvas_height=100):
    key=repr((boxes,width,height,canvas_width,canvas_height))
    if key not in self.page_analysis:
        self.page_analysis[key]=self.original_analyze(image,boxes,width,height,canvas_width,canvas_height)
    analyzed=self.page_analysis[key]
    stats=self.counts[self.policy];stats['proposals']+=len(analyzed)
    results=[];kept={};counts=Counter()
    for item in analyzed:
        c=int(item['class_id']);stats[f'proposals_c{c}']+=1
        limit=10**9 if self.policy=='unlimited' else 60 if self.policy=='larger_limit' else self.max_instances
        if counts[c]>=limit:
            stats['limit_rejected']+=1;continue
        def detail():
            threshold=float(self.policy.removeprefix('decode'))/100 if self.policy.startswith('decode') else self.threshold
            item_key=(id(item),threshold)
            if item_key not in self.page_detail:
                l,t,r,b=item['box']
                if id(item) not in self.page_probability:
                    self.page_probability[id(item)]=self.native_detail.correct(image[t:b,l:r],item['probability'],c)
                probability=self.page_probability[id(item)]
                values=vectorize_mask(probability,l,t,r-l,b-t,width/canvas_width,height/canvas_height,threshold)
                segments=[{'command':0,'points':p,'polyline':np.asarray(p),
                           'length':float(length),'rgb':[0,0,0],'layer':0,'width':1.} for p,length in values]
                self.page_detail[item_key]=(segments or item['segments'],probability)
            return self.page_detail[item_key]
        frame=item['det_box'] if self.policy=='detector_nms' else item['box']
        duplicates=[old for old in kept.get(c,[]) if box_iou(frame,old[0])>.5]
        if self.policy=='ink_nms' and duplicates:
            mask=render_items(detail()[0],1000,canvas_width,canvas_height)
            duplicates=[old for old in duplicates if
                        np.count_nonzero(mask & old[1])/max(np.count_nonzero(mask | old[1]),1)>.5]
        if duplicates:
            stats['nms_rejected']+=1;continue
        features=item['features']
        if self.policy=='native_quality':
            segments,probability=detail()
            features=self._features(c,item['confidence'],item['det_box'],
                [(s['points'],s['length']) for s in segments],probability,
                width/canvas_width,height/canvas_height,canvas_width,canvas_height)
        quality=self._quality_score(c,features)
        payload=self.quality_model
        if self.policy in ('learned_quality','boosted_quality'):
            payload=self.learned_quality
            spec=payload.get('classes',{}).get(str(c),{})
            if spec.get('feature_stage')=='native':
                segments,probability=detail()
                features=combined_features(self,item,segments,probability,width,height,canvas_width,canvas_height)
            quality=quality_score(payload,c,features)
        if quality is not None and quality<float(payload['classes'][str(c)].get('threshold',.5)):
            stats['quality_rejected']+=1;continue
        segments,_=detail()
        mask=render_items(segments,1000,canvas_width,canvas_height) if self.policy=='ink_nms' else None
        kept.setdefault(c,[]).append((frame,mask));counts[c]+=1
        results.append((c,float(item['confidence']),segments));stats['accepted']+=1
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True,help='PNG, GT, stages and baseline under this role directory')
    parser.add_argument('--modes',default='legacy,detector_nms,ink_nms,native_quality,extra_classes')
    parser.add_argument('--image-size',type=int,default=640)
    parser.add_argument('--quality-model',type=Path)
    args=parser.parse_args();defaults=build_parser().parse_args([]);_prepare_environment(defaults.seed)
    modes=args.modes.split(',')
    if any(m not in ('legacy','detector_nms','ink_nms','native_quality','extra_classes','hires','larger_limit','learned_quality','boosted_quality','unlimited','decode40','decode50','decode70','decode80') for m in modes):
        parser.error('Unknown experiment')
    native=NativeDetailRefiner.load('weights/native_detail.pt','cuda')
    plans=[]
    for classes,path in [({2},'weights/roi_ink_c2.pt'),({3,4,22,26,29},'weights/roi_ink_512.pt'),
                        ({8,12,14,15,18,19,20,21,25,27,28},'weights/roi_ink_256.pt')]:
        model=RoiInkRefiner.load(path,device='cuda',quality_path=defaults.roi_quality_model)
        model.native_detail=native;model.original_analyze=model.analyze;model.counts={m:Counter() for m in modes}
        if args.quality_model:model.learned_quality=json.loads(args.quality_model.read_text())
        model.refine=types.MethodType(experimental_refine,model);plans.append((frozenset(classes),model))
    detector=segdet._model('weights/yolo_seg.pt');original_predict=detector.predict
    original_scene=segdet.component_scene
    predict_cache={};scene_cache={}
    def predict(*a,**k):
        key=repr(k)
        if key not in predict_cache:predict_cache[key]=original_predict(*a,**k)
        return predict_cache[key]
    def scene(*a,**k):
        if 'value' not in scene_cache:scene_cache['value']=original_scene(*a,**k)
        return scene_cache['value']
    detector.predict=predict;segdet.component_scene=scene
    config=InferenceConfig()
    paths=sorted((args.base/'stages').glob('*.json'))
    for page_number,path in enumerate(paths):
        snapshot=json.loads(path.read_text(encoding='utf-8'))
        predict_cache.clear();scene_cache.clear()
        for _classes,model in plans:model.page_analysis={};model.page_detail={};model.page_probability={}
        for mode in modes:
            for _classes,model in plans:model.policy=mode
            active_plans=list(plans)
            if mode=='extra_classes':
                active_plans[1]=(active_plans[1][0]|{9,10,11,13,16,17,23,24},active_plans[1][1])
            handled=frozenset().union(*(c for c,_ in active_plans))
            result=segdet.run_image_segdet(args.base/'png'/snapshot['file'],'weights/yolo_seg.pt',config,
                defaults.palette,det_conf=.2,det_iou=.5,max_instances=10,inside_fraction=.35,
                image_size=args.image_size if mode=='hires' else 640,
                device=0,support_classes=frozenset({8}),refiner_classes=handled,
                refiners=tuple(active_plans))
            if mode=='legacy' and result['data']!=snapshot['segdet']:
                raise ValueError('Legacy branch must reproduce cached stage: '+path.name)
            updated=dict(snapshot,segdet=result['data'],support=result['support'])
            updated['merge_options']=dict(snapshot['merge_options'],segdet_union_classes=sorted(handled))
            data=replay(updated)
            if mode=='legacy' and data!=snapshot['final']:
                raise ValueError('Legacy replay must reproduce final: '+path.name)
            write_output(data,args.base/mode/path.name)
        if (page_number+1)%10==0:print('paired policies',page_number+1,'/',len(paths),flush=True)
    counts={mode:dict(sum((model.counts[mode] for _,model in plans),Counter())) for mode in modes}
    count_path=args.base/'selection_counts.json'
    previous=json.loads(count_path.read_text()) if count_path.exists() else {}
    previous.update(counts)
    count_path.write_text(json.dumps(previous,indent=1))
    for mode in modes:
        report=evaluate_directories(args.base/'gt',args.base/mode)
        (args.base/(mode+'_eval.json')).write_text(json.dumps(report,indent=1))
        print(mode,report['micro'],report['TP'],report['FP'],report['FN'],counts[mode],flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
