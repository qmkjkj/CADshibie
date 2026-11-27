import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[1]/'_work/ultralytics'))
from cadvec.query_instances import (
    QueryInstanceLoss, QueryInstanceModel, dense_mask_logits, sample_mask_logits,
    sine_position_2d,
)


def known_output(batch=1):
    # Two queries of the same semantic class own disjoint strokes. A third query
    # is null. Features give an exact identity basis on these small native masks.
    pixels = torch.zeros(batch,2,4,4)
    pixels[:,0,0,:] = 1; pixels[:,1,3,:] = 1
    embeddings = torch.tensor([[[15.,-15.],[-15.,15.],[-15.,-15.]]]).expand(batch,-1,-1).clone()
    classes = torch.full((batch,3,31),-8.)
    classes[:,:2,2] = 8.; classes[:,2,30] = 8.
    boxes = torch.tensor([[[.5,.125,1.,.25],[.5,.875,1.,.25],[.5,.5,.2,.2]]]).expand(batch,-1,-1).clone()
    return {'pixel_features':pixels.requires_grad_(),'mask_embed':embeddings.requires_grad_(),
            'class_logits':classes.requires_grad_(),'boxes':boxes.requires_grad_()}


def known_targets():
    masks = torch.zeros(2,4,4,dtype=torch.bool); masks[0,0,:] = True; masks[1,3,:] = True
    return {'labels':torch.tensor([2,2]),'masks':masks,
            'boxes':torch.tensor([[.5,.125,1.,.25],[.5,.875,1.,.25]])}


def test_same_class_instances_are_matched_one_to_one_and_target_permutation_is_invariant():
    output,target = known_output(),known_targets(); criterion = QueryInstanceLoss()
    first = criterion(output,[target],torch.arange(16))
    query_ids,target_ids = first['assignment'][0]
    assert query_ids.tolist() == [0,1] and target_ids.tolist() == [0,1]
    permuted = {key:value.flip(0) for key,value in target.items()}
    second = criterion(output,[permuted],torch.arange(16))
    assert second['assignment'][0][1].tolist() == [1,0]
    for key in ('total','class','mask_bce','mask_dice','bbox_l1','giou'):
        torch.testing.assert_close(first[key],second[key])


def test_sampled_logits_equal_selected_dense_pixels_and_preserve_both_gradients():
    torch.manual_seed(3)
    output = {'pixel_features':torch.randn(2,5,4,6,requires_grad=True),
              'mask_embed':torch.randn(2,3,5,requires_grad=True)}
    points = torch.tensor([[0,5,9,23],[3,6,10,20]])
    sampled = sample_mask_logits(output,points)
    dense = dense_mask_logits(output,[0,1,2]).flatten(2)
    expected = dense.gather(2,points[:,None].expand(-1,3,-1))
    torch.testing.assert_close(sampled,expected)
    sampled.square().mean().backward()
    assert output['pixel_features'].grad.abs().sum()>0
    assert output['mask_embed'].grad.abs().sum()>0
    # Unselected native pixels do not receive mask-point-loss gradients.
    assert output['pixel_features'].grad[0,:,2,0].abs().sum() == 0


def test_empty_gt_trains_null_and_mixed_batch_mask_losses_are_finite():
    output = known_output(batch=2)
    empty = {'labels':torch.empty(0,dtype=torch.long),'masks':torch.empty(0,4,4,dtype=torch.bool),
             'boxes':torch.empty(0,4)}
    criterion = QueryInstanceLoss()
    result = criterion(output,[empty,known_targets()],torch.arange(16))
    assert len(result['assignment'][0][0]) == 0
    assert all(torch.isfinite(result[key]) for key in ('total','class','mask_bce','mask_dice','bbox_l1','giou'))
    result['total'].backward()
    assert output['class_logits'].grad[0].abs().sum()>0
    only_empty = criterion(known_output(),[empty],torch.arange(16))
    assert only_empty['class']>0 and only_empty['mask_bce'] == 0 and only_empty['giou'] == 0


def test_cpu_mask_targets_and_zero_height_boxes_have_finite_loss():
    output,target = known_output(),known_targets()
    target['boxes'][:,3] = 0
    result = QueryInstanceLoss()(output,[target],torch.arange(16))
    result['total'].backward()
    assert torch.isfinite(result['total']) and torch.isfinite(output['boxes'].grad).all()


def test_model_keeps_native_shape_learns_cad_encoder_and_uses_xy_positions():
    torch.set_num_threads(1)
    path = Path(__file__).resolve().parents[1]/'weights/yolo_seg.pt'
    model = QueryInstanceModel(backbone_path=path,num_queries=100)
    assert model.backbone_initialization['verified_sequential_routing']
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1,3,64,96))
    assert output['class_logits'].shape == (1,100,31)
    assert output['pixel_features'].shape == (1,32,64,96)
    assert output['mask_embed'].shape == (1,100,32)
    assert output['boxes'].shape == (1,100,4)
    assert (output['boxes']>0).all() and (output['boxes']<1).all()
    assert dense_mask_logits(output,[2,7]).shape == (1,2,64,96)
    position = sine_position_2d(2,3,128,'cpu',torch.float32).reshape(2,3,128)
    assert not torch.equal(position[0,0],position[0,1])
    assert not torch.equal(position[0,0],position[1,0])
    # Random offline construction can restore the complete learned state without
    # reopening/downloading the source pretrained checkpoint.
    restored = QueryInstanceModel(backbone_path=None,num_queries=100)
    restored.load_state_dict(model.state_dict(),strict=True)


def test_invalid_pixel_indices_are_rejected_before_matching():
    output = known_output()
    with pytest.raises(ValueError,match='outside'):
        sample_mask_logits(output,[16])
    with pytest.raises(ValueError,match='integer'):
        sample_mask_logits(output,[1.5])
    target = known_targets()
    over_budget = {key:torch.cat((value,value),0) for key,value in target.items()}
    with pytest.raises(ValueError,match='query budget'):
        QueryInstanceLoss()(output,[over_budget],torch.arange(16))
