import numpy as np
import pytest

from cadvec.pipeline_segdet import _Primitive
from cadvec.schema import assemble_output
from tools.replay_native_geometry import replace_roi_stage


def payload(class_id,x):
    return {'command':0,'points':[[x,1],[x+1,1],[x+2,1],[x+3,1]],
            'length':3.,'rgb':[0,10,20],'layer':2,'width':1.,'semantic':class_id}


def test_roi_replay_keeps_other_geometry_and_remaps_ids_after_new_instances():
    original=assemble_output([_Primitive(payload=payload(c,x),instance=i)
        for c,x,i in ((2,0,0),(8,10,1),(10,20,2),(30,30,-1))],width=100,height=100)
    snapshot={'segdet':original,'support':{'size':[48,48],'detections':[
        {'class':2,'conf':.8,'box':[0,0,10,10]},
        {'class':8,'conf':.9,'box':[10,0,20,10]}]}}
    class Model:
        def refine(self,*args):
            return [(8,.9,[payload(8,11)]),(8,.85,[payload(8,15)]),(2,.8,[payload(2,1)])]
    actual=replace_roi_stage(snapshot,np.zeros((48,48,3),dtype=np.uint8),[({2,8},Model())])
    assert actual['semanticIds']==[8,8,2,10,30]
    assert actual['instanceIds']==[0,1,2,3,-1]
    for key in ('commands','args','lengths','rgb','layerIds','widths'):
        assert actual[key][-2:]==original[key][-2:]


def test_roi_replay_preserves_empty_branch_and_rejects_cached_frame_mismatch():
    original=assemble_output([],width=100,height=100)
    snapshot={'segdet':original,'support':{'size':[32,32],'detections':[
        {'class':2,'conf':.9,'box':[0,0,10,10]}]}}
    class Empty:
        def refine(self,*args):return []
    image=np.zeros((32,32,3),dtype=np.uint8)
    assert replace_roi_stage(snapshot,image,[({2},Empty())]) is original
    with pytest.raises(ValueError,match='frames differ'):
        replace_roi_stage(snapshot,image[:16],[({2},Empty())])
