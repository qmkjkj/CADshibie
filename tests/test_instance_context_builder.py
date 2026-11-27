from pathlib import Path

import numpy as np
import pytest

from tools.build_instance_context import context_for_page, render_truths, select_candidates


def polygon(box):
    x0,y0,x1,y1 = box
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],dtype=float)


def truth(instance,box,shape=(32,32)):
    return instance,np.zeros(shape,dtype=bool),box


def test_ambiguity_consumes_detection_limit_and_validation_never_adds_gt_boxes():
    truths = {2:[truth(7,(2,2,12,12)),truth(8,(2,2,12,12))]}
    detections = [(2,.9,polygon((2,2,12,12))), (2,.8,polygon((20,20,28,28)))]
    assert select_candidates(truths,detections,'val',max_per_class=1) == []
    train = select_candidates(truths,detections,'train',max_per_class=1)
    assert [(row[0],row[3],row[4]) for row in train] == [(2,0,'gt_supplement'),(2,1,'gt_supplement')]


def test_wrong_class_detection_remains_negative_and_does_not_steal_gt_class():
    truths = {2:[truth(7,(2,2,12,12))]}
    candidates = select_candidates(truths,[(3,.9,polygon((2,2,12,12)))],'train')
    assert candidates[0][0] == 3 and candidates[0][3] is None
    assert candidates[1][0] == 2 and candidates[1][4] == 'gt_supplement'


def test_context_uses_route_group_number_and_preserves_indexed_supplement_negative(tmp_path):
    # Class 22 and 3 share one route, so 3 receives candidate number 1.
    candidates = [(22,.8,(1.,1.,5.,5.),None,'detector'),
                  (3,1.,(8.,8.,12.,12.),0,'gt_supplement')]
    truths = {3:[truth(12,(8.,8.,12.,12.))]}
    row = {'path':'train/example_3_1_0.npz','role':'train','stem':'example',
           'class':3,'instance':12,'source':'gt_supplement','confidence':1.,'positive':False}
    path = tmp_path/row['path']; path.parent.mkdir(parents=True)
    np.savez(path,target=np.zeros((16,16),dtype=np.uint8),valid_shape=np.array([6,6]))
    records,counts,error = context_for_page([row],truths,candidates,32,32,16,tmp_path)
    assert records[row['path']] == {'target_box':[1.,1.,5.,5.],'tile_origin':[0,0]}
    assert counts['train_gt_supplement_negative'] == 1 and error == 0
    altered = dict(row,instance=13)
    with pytest.raises(ValueError,match='identity differs'):
        context_for_page([altered],truths,candidates,32,32,16,tmp_path)


def test_rendered_gt_boxes_use_inclusive_native_extrema():
    gt = {'semanticIds':[2],'instanceIds':[9],'commands':[0],
          'args':[[1.,1.,2.,1.,3.,1.,4.,1.]],'width':10,'height':10}
    truths,scale = render_truths(gt,100,100)
    assert scale == (10.,10.)
    assert truths[2][0][0] == 9
    assert truths[2][0][2] == (10.,10.,40.,10.)


def test_indexed_nonzero_origin_tile_is_kept_without_replaying_empty_tile_filter(tmp_path):
    row = {'path':'val/example_2_0_4.npz','role':'val','stem':'example',
           'class':2,'instance':-1,'source':'detector','confidence':.75,'positive':False}
    path = tmp_path/row['path']; path.parent.mkdir(parents=True)
    np.savez(path,target=np.zeros((16,16),dtype=np.uint8),valid_shape=np.array([16,16]))
    records,counts,_error = context_for_page([row],{},[(2,.75,(2.,2.,30.,30.),None,'detector')],
                                             32,32,16,tmp_path)
    assert records == {row['path']:{'target_box':[2.,2.,30.,30.],'tile_origin':[8,8]}}
    assert counts['val_detector_negative'] == 1
