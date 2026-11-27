#!/usr/bin/env python3
"""Compare a fixed small boosted gate on the same frozen quality fit/calib rows."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.roi_quality import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    base=json.loads(a.base_model.read_text());rows=json.loads(a.rows.read_text())['rows']
    support=base['training']['supported_classes'];names=base['features']
    fit=[r for r in rows if r['role']=='fit' and r['class'] in support]
    calib=[r for r in rows if r['role']=='calib' and r['class'] in support]
    x=np.asarray([[r['features'][k] for k in names] for r in fit]);y=np.asarray([r['tp'] for r in fit])
    model=HistGradientBoostingClassifier(max_iter=100,max_depth=3,min_samples_leaf=20,
             l2_regularization=1.,learning_rate=.05,random_state=2712).fit(x,y)
    cx=np.asarray([[r['features'][k] for k in names] for r in calib]);probs=model.predict_proba(cx)[:,1]
    candidates=[]
    for threshold in np.arange(.05,.801,.05):
        kept=[r for r,pr in zip(calib,probs) if pr>=threshold]
        candidates.append((sum(r['iou'] for r in kept)-.33*.5*len(kept),float(threshold),len(kept),sum(r['tp'] for r in kept)))
    best=max(candidates,key=lambda v:v[0]);trees=[]
    for predictors in model._predictors:
        assert len(predictors)==1
        nodes=predictors[0].nodes
        assert not nodes['is_categorical'].any()
        trees.append([[int(n['feature_idx']),float(n['num_threshold']),int(n['left']),int(n['right']),
                       float(n['value']),bool(n['is_leaf'])] for n in nodes])
    payload=dict(base,boosting={'baseline':float(model._baseline_prediction[0,0]),'trees':trees})
    payload['classes']=dict(base['classes'])
    for c in support:payload['classes'][str(c)]={'model_kind':'histogram-boosting','threshold':best[1],'feature_stage':'native'}
    payload['training']=dict(base['training'],learner='HistGradientBoostingClassifier',max_depth=3,
        min_samples_leaf=20,l2_regularization=1.,learning_rate=.05,max_iter=100,
        threshold_candidates=candidates,selected_threshold=best[1])
    # Verify the portable runtime against sklearn on every fit/calibration row.
    expected=model.predict_proba(np.concatenate([x,cx]))[:,1]
    actual=np.asarray([score(payload,r['class'],r['features']) for r in fit+calib])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    a.out.write_text(json.dumps(payload,indent=1));print('Portable boosted gate verified',len(actual),
        'rows, threshold',best[1],'nodes',sum(map(len,trees)),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
