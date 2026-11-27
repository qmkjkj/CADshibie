#!/usr/bin/env python3
"""Offline native-head replay using frozen detections and other-class geometry.

Every baseline must reproduce its full-inference prediction exactly. Optional
candidate references additionally verify the accelerated replay on known pages.
This tool does not change production inference or use labels to make outputs.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.layers import load_rgba
from cadvec.native_detail import NativeDetailRefiner
from cadvec.pipeline_segdet import _Primitive
from cadvec.roi_ink import RoiInkRefiner
from cadvec.schema import assemble_output,write_output
from cadvec.evaluate import evaluate_directories
from infer import build_parser,_prepare_environment
from tools.replay_hybrid import replay
from tools.summarize_ablation import aggregate,paired_group_bootstrap


def replace_roi_stage(snapshot,image,plans):
    original=snapshot['segdet'];support=snapshot.get('support') or {}
    records=support.get('detections') or []
    if not records:return original
    height,width=image.shape[:2]
    if list(support['size'])!=[height,width]:
        raise ValueError('Cached detector and PNG frames differ')
    cw,ch=original['width'],original['height']
    handled=set().union(*(set(c) for c,_ in plans))
    records=sorted(records,key=lambda r:-r['conf'])
    refined=[]
    for classes,model in plans:
        boxes=[(int(r['class']),float(r['conf']),tuple(r['box']))
               for r in records if r['class'] in classes]
        if boxes:
            refined.extend((conf,c,segments) for c,conf,segments
                           in model.refine(image,boxes,width,height,cw,ch))
    if not refined:return original
    # Production emits refined classes first in confidence order, followed by
    # the fixed selected other-class groups, then Stuff. Candidate acceptance
    # depends on coarse geometry/quality only; empty native results fall back
    # to coarse. Thus the other-class selection is invariant to this head.
    emitted=[];instance=0
    for _,class_id,segments in sorted(refined,key=lambda item:-item[0]):
        for segment in segments:
            payload={key:segment[key] for key in ('command','points','length','rgb','layer','width')}
            emitted.append(_Primitive(payload=dict(payload,semantic=class_id),instance=instance))
        instance+=1
    other_ids={}
    for i,(class_id,old_id) in enumerate(zip(original['semanticIds'],original['instanceIds'])):
        if class_id in handled:continue
        if old_id>=0:
            if old_id not in other_ids:
                other_ids[old_id]=instance;instance+=1
            new_id=other_ids[old_id]
        else:new_id=old_id
        args=original['args'][i]
        payload={'command':original['commands'][i],'points':list(zip(args[::2],args[1::2])),
                 'length':original['lengths'][i],'rgb':original['rgb'][i],
                 'layer':original['layerIds'][i],'width':original['widths'][i],'semantic':class_id}
        emitted.append(_Primitive(payload=payload,instance=new_id))
    return assemble_output(emitted,width=cw,height=ch)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--png-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--verify-baseline',type=Path,required=True)
    p.add_argument('--verify-candidate',type=Path)
    p.add_argument('--source-identity',choices=('legacy','scoped'),default='legacy')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--gt-dir',type=Path)
    p.add_argument('--source-splits',type=Path)
    p.add_argument('--group-role',choices=('train_final','test_final'),default='test_final')
    args=p.parse_args();defaults=build_parser().parse_args([]);_prepare_environment(defaults.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    native={n:NativeDetailRefiner.load(path,'cuda') for n,path in
            (('baseline',args.baseline),('boundary',args.candidate))}
    plans=[];caches=[]
    for classes,path in (({2},'weights/roi_ink_c2.pt'),({3,4,22,26,29},'weights/roi_ink_512.pt'),
                         ({8,12,14,15,18,19,20,21,25,27,28},'weights/roi_ink_256.pt')):
        model=RoiInkRefiner.load(path,device='cuda',max_instances=0,quality_path=defaults.roi_quality_model)
        original=model.analyze;cache={};caches.append(cache)
        def analyze(image,*a,_original=original,_cache=cache,**kw):
            key=repr((a,kw))
            if key not in _cache:_cache[key]=_original(image,*a,**kw)
            return _cache[key]
        model.analyze=analyze;plans.append((frozenset(classes),model))
    paths=sorted(args.snapshots.glob('*.json'))
    if args.limit:paths=paths[:args.limit]
    if not paths:raise ValueError('No stage snapshots')
    provenance={'method':'ROI-only replay; fixed detector, scene/other-class geometry and primary/GAT/stuff cached',
                'baseline_sha256':sha(args.baseline),'candidate_sha256':sha(args.candidate),
                'tool_sha256':sha(__file__),'source_identity':args.source_identity,
                'verified_baseline_pages':0,'verified_candidate_pages':0,
                'snapshots':{path.name:sha(path) for path in paths},
                'fixed_inputs':{str(path):sha(path) for path in ('weights/roi_ink_c2.pt',
                    'weights/roi_ink_512.pt','weights/roi_ink_256.pt',defaults.roi_quality_model)},
                'source_files':{path:sha(path) for path in ('cadvec/detail_unet.py','cadvec/native_detail.py',
                    'cadvec/roi_ink.py','cadvec/schema.py','cadvec/hybrid.py','tools/replay_hybrid.py')},
                'png_sha256':{}}
    for index,path in enumerate(paths):
        snapshot=json.loads(path.read_text(encoding='utf-8'))
        png=args.png_dir/snapshot['file'];provenance['png_sha256'][path.name]=sha(png)
        rgb,alpha=load_rgba(png)
        image=np.clip(rgb*alpha[...,None]+255*(1-alpha[...,None]),0,255).astype(np.uint8)
        for cache in caches:cache.clear()
        for name,head in native.items():
            for _,model in plans:model.native_detail=head
            updated=dict(snapshot,segdet=replace_roi_stage(snapshot,image,plans))
            data=replay(updated,source_identity=args.source_identity)
            reference=(args.verify_baseline/path.name if name=='baseline' else
                       args.verify_candidate/path.name if args.verify_candidate else None)
            if reference is not None and reference.exists():
                if data!=json.loads(reference.read_text(encoding='utf-8')):
                    raise ValueError(f'Accelerated {name} replay differs: {path.name}')
                provenance['verified_'+('baseline' if name=='baseline' else 'candidate')+'_pages']+=1
            elif name=='baseline':raise FileNotFoundError(reference)
            write_output(data,args.out/name/path.name)
        if (index+1)%25==0:print('native-only replay',index+1,'/',len(paths),flush=True)
    (args.out/'replay_provenance.json').write_text(json.dumps(provenance,indent=1),encoding='utf-8')
    if args.gt_dir:
        reports={}
        for name in native:
            report=evaluate_directories(args.gt_dir,args.out/name);reports[name]=report
            (args.out/(name+'_eval.json')).write_text(json.dumps(report,indent=1),encoding='utf-8')
        summary={'metrics':{n:aggregate(list(r['per_image'].values())) for n,r in reports.items()}}
        if args.source_splits:
            groups=json.loads(args.source_splits.read_text(encoding='utf-8'))['groups'][args.group_role]
            summary['uncertainty']=paired_group_bootstrap(reports,{x.stem for x in paths},groups,draws=5000)
        (args.out/'comparison.json').write_text(json.dumps(summary,indent=1),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
