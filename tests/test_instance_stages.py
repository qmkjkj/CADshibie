import numpy as np
import pytest

from tools.diagnose_instance_stages import (
    Segment, assign, detection_coverage, diagnose_page, strict_matches,
)


def segment(class_id, instance_id, pixels):
    mask = np.zeros((10,10),dtype=bool)
    for y,x in pixels:
        mask[y,x] = True
    ys,xs = np.nonzero(mask)
    return Segment((class_id,instance_id),mask,int(mask.sum()),
                   (float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)))


def output(class_id=2, instance_id=0):
    return {'commands':[0],'args':[[1,1,2,1,3,1,4,1]],
            'semanticIds':[class_id],'instanceIds':[instance_id]}


def test_strict_geometry_respects_class_and_exclusive_threshold():
    truth = [segment(2,0,[(1,1),(1,2)])]
    predictions = [segment(2,9,[(1,1)]),segment(3,3,[(1,1),(1,2)])]
    matches,matrix = strict_matches(truth,predictions)
    assert matrix.tolist() == [[.5,0.]]
    assert matches == {}


def test_box_assignment_is_one_to_one_and_wrong_class_is_separate():
    truth = [segment(2,0,[(1,1),(1,2)]),segment(2,1,[(1,1),(1,2)])]
    detections = [{'class':2,'box':[1,1,3,2]}, {'class':3,'box':[1,1,3,2]}]
    matrix,same,correct,agnostic = detection_coverage(truth,detections,(10,10),10)
    assert matrix.tolist() == [[1.,1.],[1.,1.]]
    assert same.tolist() == [[True,False],[True,False]]
    assert len(correct) == 1
    assert len(agnostic) == 2


def test_box_assignment_prioritizes_cardinality_but_geometry_matches_evaluator():
    matrix = np.array([[1.,.51],[.51,0.]])
    assert len(assign(matrix,matrix>.5,count_first=True)) == 2
    matrix = np.array([[1.,.5001,0.],[0.,1.,.5001],[.5001,0.,0.]])
    assert len(assign(matrix,matrix>.5,count_first=True)) == 3
    assert len(assign(matrix,matrix>.5)) == 2


def test_historical_stage_loss_is_not_assigned_to_current_run_and_trace_must_verify():
    empty = {key:[] for key in ('commands','args','semanticIds','instanceIds')}
    truth = {**output(),'width':10,'height':10}
    snapshot = {'file':'example.png','segdet':output(),'final':empty,
                'support':{'size':[10,10],'detections':[]}}
    result = diagnose_page(snapshot,output(),truth,raster_size=10)
    assert result['thing_counts']['final_matched'] == 1
    assert result['thing_counts']['historical_segdet_match_lost_to_own_final'] == 1
    with pytest.raises(ValueError,match='not verified'):
        diagnose_page(snapshot,output(),truth,raster_size=10,
                      trace={'verified_current_baseline':False,'final':output()})


def test_rejected_candidate_potential_is_independent_of_accepted_assignment():
    empty = {key:[] for key in ('commands','args','semanticIds','instanceIds')}
    truth = {**output(),'width':10,'height':10}
    snapshot = {'file':'example.png','segdet':empty,'final':empty,
                'support':{'size':[10,10],'detections':[]}}
    trace = {'verified_current_baseline':True,'final':empty,'segdet':empty,'candidates':[
        {'candidate_id':'a','decision':'quality','native':output()},
        {'candidate_id':'b','decision':'nms','native':output()}]}
    result = diagnose_page(snapshot,empty,truth,raster_size=10,trace=trace)
    assert result['roi_trace']['native_all_one_to_one'] == 1
    assert result['roi_trace']['native_potential_by_decision'] == {'nms':1,'quality':1}
    assert result['roi_trace']['thing_counts']['rejected_only_potential_but_final_unmatched'] == 1
    assert result['instances'][0]['roi_trace']['final_FN_stage_bucket'] == 'rejected_native_only_potential'
