import numpy as np
import torch

from cadvec.roi_ink import RoiInkRefiner
from tools.trace_roi_stages import gate_decisions


def test_trace_rejected_boxes_do_not_suppress_later_candidates():
    model=RoiInkRefiner(torch.nn.Identity(),torch.device('cpu'),max_instances=0)
    model.quality_model={'classes':{'8':{'threshold':.5}}}
    model._quality_score=lambda c,f:f['quality']
    segment={'command':0,'points':[[0,0],[1,0],[2,0],[3,0]],'length':3.}
    rows=[{'class_id':8,'confidence':.9,'box':(0,0,20,20),'features':{'quality':q},'segments':[segment]}
          for q in (.1,.9,.9)]
    assert [r[0] for r in gate_decisions(model,rows)]==['quality','accepted','nms']
    model.analyze=lambda *a,**kw:rows
    assert len(model.refine(np.zeros((30,30,3),dtype=np.uint8),[],30,30))==1


def test_trace_empty_coarse_does_not_consume_cap_or_suppress():
    model=RoiInkRefiner(torch.nn.Identity(),torch.device('cpu'),max_instances=1)
    empty={'class_id':8,'box':(0,0,20,20),'features':{},'segments':[]}
    full=dict(empty,segments=[{'points':[]}])
    last=dict(full,box=(30,0,40,20))
    assert [r[0] for r in gate_decisions(model,[empty,full,last])]==['empty_coarse','accepted','cap']
