"""Quality features for coarse/native instance geometry; no labels at runtime."""
from __future__ import annotations
import math
import numpy as np


def combined_features(refiner,item,segments,probability,width,height,canvas_width=100,canvas_height=100):
    coarse=item['features'];c=int(item['class_id'])
    native=refiner._features(c,item['confidence'],item['det_box'],
        [(s['points'],s['length']) for s in segments],probability,
        width/canvas_width,height/canvas_height,canvas_width,canvas_height)
    features=dict(coarse)
    features.update({'coarse_'+k:float(v) for k,v in coarse.items()})
    features.update({'native_'+k:float(v) for k,v in native.items()})
    before=item['probability']>refiner.threshold;after=probability>refiner.threshold
    features['coarse_native_mask_iou']=float(np.count_nonzero(before & after)/max(np.count_nonzero(before | after),1))
    features['probability_change']=float(np.abs(item['probability']-probability).mean())
    features.update({'class_'+str(k):float(c==k) for k in range(30)})
    return features


def score(payload,class_id,features):
    spec=(payload.get('classes') or {}).get(str(class_id))
    if not spec:return None
    if spec.get('model_kind')=='histogram-boosting':
        names=payload['features'];values=[float(features.get(k,0.)) for k in names]
        z=float(payload['boosting']['baseline'])
        for nodes in payload['boosting']['trees']:
            index=0
            while not nodes[index][5]:
                feature,threshold,left,right,_value,_leaf=nodes[index]
                index=left if values[feature]<=threshold else right
            z+=nodes[index][4]
        return 1/(1+math.exp(-min(max(z,-50.),50.)))
    names=spec.get('features') or payload.get('features') or ()
    z=float(spec.get('intercept',0.))
    for name,coef,mean,scale in zip(names,spec['coef'],spec['mean'],spec['scale']):
        z+=float(coef)*(features.get(name,0.)-float(mean))/max(float(scale),1e-9)
    return 1/(1+math.exp(-min(max(z,-50.),50.)))
