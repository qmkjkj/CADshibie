#!/usr/bin/env python3
"""Trace frozen ROI gates and hypothetical native masks on rejected candidates.

Labels are never loaded. Every final output must reproduce the current baseline
before its trace is written. Rejected native masks are diagnostics, not outputs.
"""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.layers import load_rgba
from cadvec.native_detail import NativeDetailRefiner
from cadvec.pipeline_segdet import _Primitive
from cadvec.roi_ink import RoiInkRefiner, vectorize_mask
from cadvec.schema import assemble_output
from infer import build_parser, _prepare_environment
from tools.replay_hybrid import replay
from tools.replay_native_geometry import replace_roi_stage


def box_iou(first, second):
    dx = max(0, min(first[2],second[2])-max(first[0],second[0]))
    dy = max(0, min(first[3],second[3])-max(first[1],second[1]))
    inter = dx*dy
    a = (first[2]-first[0])*(first[3]-first[1])
    b = (second[2]-second[0])*(second[3]-second[1])
    return inter / max(a+b-inter, 1e-9)


def gate_decisions(model, analyzed):
    """Explain the fixed production order; empty masks never reach its gates."""
    kept = {}; counts = {}; result = []
    for item in analyzed:
        c = int(item['class_id'])
        quality = model._quality_score(c,item['features'])
        spec = ((model.quality_model or {}).get('classes') or {}).get(str(c),{})
        threshold = float(spec.get('threshold',.5)) if quality is not None else None
        if not item['segments']:
            reason = 'empty_coarse'
        elif model.max_instances > 0 and counts.get(c,0) >= model.max_instances:
            reason = 'cap'
        elif any(box_iou(item['box'],b)>.5 for b in kept.get(c,[])):
            reason = 'nms'
        elif quality is not None and quality < threshold:
            reason = 'quality'
        else:
            reason = 'accepted'; counts[c] = counts.get(c,0)+1
            kept.setdefault(c,[]).append(item['box'])
        result.append((reason,quality,threshold))
    return result


def output_for_segments(segments, class_id, cw, ch):
    primitives = []
    for segment in segments:
        payload = {k:segment[k] for k in ('command','points','length','rgb','layer','width')}
        primitives.append(_Primitive(payload=dict(payload,semantic=class_id),instance=0))
    return assemble_output(primitives,width=cw,height=ch)


class CachedNative:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def correct(self,image,coarse,class_id):
        return self.probabilities[id(coarse)]


def trace_page(snapshot, image, plans, head):
    height,width = image.shape[:2]
    cw,ch = snapshot['segdet']['width'],snapshot['segdet']['height']
    records = sorted(snapshot['support']['detections'],key=lambda x:-x['conf'])
    rows = []
    originals = []
    try:
        for group,(classes,model) in enumerate(plans):
            original = model.analyze; originals.append((model,original,model.native_detail))
            boxes = [(int(d['class']),float(d['conf']),tuple(d['box']))
                     for d in records if d['class'] in classes]
            analyzed = original(image,boxes,width,height,cw,ch,include_empty=True)
            if len(analyzed) != len(boxes):
                raise ValueError('Detector/crop trace alignment differs')
            probabilities = {}
            for number,(item,(reason,quality,threshold)) in enumerate(zip(analyzed,gate_decisions(model,analyzed))):
                c = int(item['class_id']); left,top,right,bottom = item['box']
                native = item['segments']; computed = bool(native); fallback = False
                if computed:
                    kwargs = {}
                    if getattr(head,'uses_instance_prompt',False):
                        x0,y0,x1,y1=item['det_box']
                        kwargs['det_box']=(x0-left,y0-top,x1-left,y1-top)
                    probability = head.correct(image[top:bottom,left:right],item['probability'],c,**kwargs)
                    probabilities[id(item['probability'])] = probability
                    vectors = vectorize_mask(probability,left,top,right-left,bottom-top,
                                             width/max(cw,1),height/max(ch,1),model.threshold)
                    fallback = not bool(vectors)
                    if vectors:
                        native = [{'command':0,'points':p,'length':float(length),
                                   'rgb':[0,0,0],'layer':0,'width':1.} for p,length in vectors]
                rows.append({'candidate_id':f'roi{group}:{number}','class_id':c,
                             'confidence':float(item['confidence']),
                             'det_box':list(item['det_box']),'crop_box':list(item['box']),
                             'decision':reason,'quality_score':quality,'quality_threshold':threshold,
                             'native_computed':computed,'native_empty_fallback':fallback,
                             'coarse':output_for_segments(item['segments'],c,cw,ch),
                             'native':output_for_segments(native,c,cw,ch)})
            # Feed the same nonempty coarse objects back into production refine.
            # This shares computation, without replacing its selection logic.
            selected = [item for item in analyzed if item['segments']]
            model.analyze = lambda *a,_items=selected,**kw:_items
            model.native_detail = CachedNative(probabilities)
        updated = dict(snapshot,segdet=replace_roi_stage(snapshot,image,plans))
        result = replay(updated,source_identity=snapshot.get('source_identity','legacy'))
        return rows,updated['segdet'],result
    finally:
        for model,analyze,native in originals:
            model.analyze = analyze; model.native_detail = native


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--png-dir',type=Path,required=True)
    p.add_argument('--verify-final',type=Path,required=True)
    p.add_argument('--native-weights',type=Path,default=Path('weights/native_detail.pt'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(); defaults=build_parser().parse_args([]);_prepare_environment(defaults.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    head=NativeDetailRefiner.load(a.native_weights,'cuda')
    plans=[(frozenset(classes),RoiInkRefiner.load(path,device='cuda',max_instances=0,
                quality_path=defaults.roi_quality_model)) for classes,path in (
                ({2},'weights/roi_ink_c2.pt'),({3,4,22,26,29},'weights/roi_ink_512.pt'),
                ({8,12,14,15,18,19,20,21,25,27,28},'weights/roi_ink_256.pt'))]
    paths=sorted(a.snapshots.glob('*.json'))
    if a.limit:paths=paths[:a.limit]
    if not paths:raise ValueError('No snapshots')
    manifest={'native_sha256':sha(a.native_weights),'tool_sha256':sha(__file__),
              'verified_pages':0,'snapshots':{},'png_sha256':{},
              'note':'All nonempty coarse candidates receive native diagnostics, including rejected ones. '
                     'Production acceptance is unchanged; every final is verified before writing. No labels read.'}
    for index,path in enumerate(paths):
        snapshot=json.loads(path.read_text(encoding='utf-8'))
        png=a.png_dir/snapshot['file']; rgb,alpha=load_rgba(png)
        image=np.clip(rgb*alpha[...,None]+255*(1-alpha[...,None]),0,255).astype(np.uint8)
        if list(image.shape[:2])!=list(snapshot['support']['size']):
            raise ValueError('PNG and cached detector frames differ')
        rows,segdet,final=trace_page(snapshot,image,plans,head)
        reference=json.loads((a.verify_final/path.name).read_text(encoding='utf-8'))
        if final!=reference:raise ValueError(f'Trace replay differs from current baseline: {path.name}')
        data={'schema_version':1,'file':snapshot['file'],
              'source_identity':snapshot.get('source_identity','legacy'),
              'native_sha256':manifest['native_sha256'],'candidates':rows,
              'segdet':segdet,'final':final,'verified_current_baseline':True}
        (a.out/path.name).write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
        manifest['verified_pages']+=1;manifest['snapshots'][path.name]=sha(path)
        manifest['png_sha256'][path.name]=sha(png)
        if (index+1)%10==0:print('traced',index+1,'/',len(paths),flush=True)
    (a.out.parent/(a.out.name+'_manifest.json')).write_text(json.dumps(manifest,indent=1),encoding='utf-8')
    print('completed',manifest['verified_pages'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
