import numpy as np
import torch
import pytest

from cadvec.detail_unet import NativeDetailUNet
from cadvec.native_detail import NativeDetailRefiner, tile_positions
from tools.build_native_detail import bbox_iou
from cadvec.roi_ink import RoiInkRefiner, vectorize_mask


def test_zero_initialized_residual_preserves_coarse_across_native_tiles():
    model = NativeDetailUNet(base=2).eval()
    coarse = np.random.default_rng(4).uniform(.01,.99,(93,147)).astype(np.float32)
    image = np.full((93,147,3),255,dtype=np.uint8)
    refiner = NativeDetailRefiner(model, "cpu", tile=80)
    actual = refiner.correct(image, coarse, 8)
    np.testing.assert_allclose(actual, coarse, rtol=1e-5,atol=1e-6)
    # Classes without training coverage must retain their probability exactly.
    refiner.classes = {2}
    assert refiner.correct(image,coarse,8) is coarse


def test_native_tile_coverage_and_empty_crop_matching():
    for length in (32,80,81,200,1000):
        covered = np.zeros(length,dtype=bool)
        for start in tile_positions(length,80):
            covered[start:start+80] = True
        assert covered.all()
    assert bbox_iou((0,0,10,10),(0,0,10,10)) == 1
    assert bbox_iou((0,0,10,10),(11,0,20,10)) == 0


def test_boundary_prior_softens_both_sides_without_changing_distant_background():
    coarse = torch.zeros((1,1,24,24))
    coarse[:,:,8:16,8:16] = 1
    fixed = NativeDetailUNet(base=2).coarse_prior(coarse)
    boundary = NativeDetailUNet(base=2,prior_mode='boundary').coarse_prior(coarse)
    assert boundary[0,0,8,10] == 2
    assert boundary[0,0,7,10] == -2
    assert boundary[0,0,4,10] == fixed[0,0,4,10]
    assert boundary[0,0,11,11] == fixed[0,0,11,11]
    assert torch.equal(torch.sigmoid(boundary)>.6,coarse>.6)
    # Uniform foreground has no artificial frame boundary from max-pool pad.
    uniform = torch.ones_like(coarse)
    model = NativeDetailUNet(base=2,prior_mode='boundary')
    assert torch.equal(model.coarse_prior(uniform),torch.logit(uniform.clamp(.001,.999)))


def test_checkpoint_preserves_prior_formula_and_rejects_unversioned_change(tmp_path):
    model = NativeDetailUNet(base=2,prior_mode='boundary').eval()
    path = tmp_path/'boundary.pt'
    payload = {'architecture':'native-residual-v2','model_kwargs':{
        'base':2,'prior_mode':'boundary','boundary_radius':2,'boundary_logit':2.},
        'state_dict':model.state_dict(),'tile':80,'classes':[8]}
    torch.save(payload,path)
    restored = NativeDetailRefiner.load(path,'cpu')
    image = np.full((48,48,3),255,dtype=np.uint8)
    coarse = np.zeros((48,48),dtype=np.float32);coarse[20:25,10:40]=1
    expected = NativeDetailRefiner(model,'cpu',tile=80,classes=[8]).correct(image,coarse,8)
    np.testing.assert_array_equal(restored.correct(image,coarse,8),expected)
    payload['architecture']='native-residual-v1';torch.save(payload,path)
    with pytest.raises(ValueError,match='version 2'):
        NativeDetailRefiner.load(path,'cpu')


def test_zero_residual_preserves_full_roi_geometry_and_empty_falls_back():
    coarse = np.zeros((64,96),dtype=np.float32)
    coarse[25:28,10:85] = .9
    segments = [{"command":0,"points":p,"polyline":np.asarray(p),"length":length,
                 "rgb":[0,0,0],"layer":0,"width":1.}
                for p,length in vectorize_mask(coarse,10,20,96,64,10.,10.,.6)]
    refiner = RoiInkRefiner(torch.nn.Identity(),torch.device('cpu'),threshold=.6)
    refiner.analyze = lambda *a,**k: [{"class_id":8,"confidence":.9,
                                      "box":(10,20,106,84),"features":{},
                                      "segments":segments,"probability":coarse}]
    image = np.full((120,140,3),255,dtype=np.uint8)
    original = refiner.refine(image,[(8,.9,(10,20,106,84))],140,120,14,12)
    refiner.native_detail = NativeDetailRefiner(NativeDetailUNet(base=2), 'cpu',tile=80)
    actual = refiner.refine(image,[(8,.9,(10,20,106,84))],140,120,14,12)
    assert [(p['points'],p['length']) for p in actual[0][2]] == [
        (p['points'],p['length']) for p in original[0][2]]
    class Empty:
        def correct(self, image, coarse, class_id):
            return np.zeros_like(coarse)
    refiner.native_detail = Empty()
    assert refiner.refine(image,[(8,.9,(10,20,106,84))],140,120,14,12)[0][2] is segments


def test_dense_instances_are_retained_without_a_per_class_cap_but_duplicates_still_suppressed():
    image = np.full((32,1340,3),255,dtype=np.uint8)
    segments = [{"command":0,"points":[[0,0],[1,0],[2,0],[3,0]],"length":3.}]
    candidates = [{"class_id":8,"confidence":.9,"box":(20*i,0,20*i+8,8),
                   "features":{},"segments":segments} for i in range(65)]
    candidates.append(dict(candidates[-1]))
    model = RoiInkRefiner(torch.nn.Identity(),torch.device('cpu'),max_instances=15)
    model.analyze = lambda *a,**k:candidates
    assert len(model.refine(image,[],1340,32)) == 15
    model.max_instances = 0
    assert len(model.refine(image,[],1340,32)) == 65
