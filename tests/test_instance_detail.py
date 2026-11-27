import numpy as np
import pytest
import torch

from cadvec.detail_unet import NativeDetailUNet
from cadvec.instance_prompt import coordinate_prompt, augment_prompt
from cadvec.native_detail import NativeDetailRefiner
from cadvec.roi_ink import RoiInkRefiner


def initialized_pair():
    torch.manual_seed(32)
    old=NativeDetailUNet(base=2,prior_mode='boundary').eval()
    torch.nn.init.normal_(old.head.weight,std=.02)
    new=NativeDetailUNet(base=2,prior_mode='boundary',conditioning='instance').eval()
    missing=new.load_state_dict(old.state_dict(),strict=False)
    assert set(missing.missing_keys)=={'class_embed.weight','prompt_projection.weight'}
    assert not missing.unexpected_keys
    return old,new


def test_box_prompt_uses_global_roi_coordinates_across_tile_boundaries():
    box=(12,9,45,56)
    full=coordinate_prompt((60,80),box)
    tile=coordinate_prompt((32,32),box,(23,18),(22,29))
    np.testing.assert_array_equal(tile[:,:22,:29],full[:,18:40,23:52])
    assert not tile[:,22:,:].any() and not tile[:,:,29:].any()
    with pytest.raises(ValueError):coordinate_prompt((32,32),(1,1,0,5))
    assert np.isfinite(coordinate_prompt((32,32),(1,4,20,4))).all()


def test_augmentation_rotates_prompt_vectors_with_the_target_box():
    box=(1,2,5,6);original=coordinate_prompt((8,8),box)
    # np.rot90 is counterclockwise in image coordinates: (x,y)->(y,8-x).
    turned=coordinate_prompt((8,8),(2,3,6,7))
    np.testing.assert_array_equal(augment_prompt(original,1),turned)
    mirrored=coordinate_prompt((8,8),(3,2,7,6))
    np.testing.assert_array_equal(augment_prompt(original,0,True),mirrored)


def test_zero_prompt_parameters_preserve_trained_outputs_and_both_receive_gradients():
    old,new=initialized_pair()
    image=torch.rand(2,4,32,32);classes=torch.tensor([2,22])
    prompt=torch.stack([torch.from_numpy(coordinate_prompt((32,32),b))
                        for b in ((4,3,25,28),(10,8,20,23))])
    expected=old(image);actual=new(image,classes,prompt)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().mean().backward()
    assert new.class_embed.weight.grad.abs().sum()>0
    assert new.prompt_projection.weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='per-sample'):new(image)
    with pytest.raises(ValueError,match='global class'):new(image,torch.tensor([2,36]),prompt)


def test_conditioning_is_per_sample_and_versioned_at_load(tmp_path):
    _,model=initialized_pair()
    with torch.no_grad():
        model.class_embed.weight[8].fill_(.5)
        model.prompt_projection.weight.fill_(.1)
    image=torch.rand(2,4,32,32);classes=torch.tensor([2,8]);prompt=torch.randn(2,2,32,32)
    actual=model(image,classes,prompt)
    for i in range(2):
        torch.testing.assert_close(actual[i:i+1],model(image[i:i+1],classes[i:i+1],prompt[i:i+1]),rtol=1e-5,atol=1e-6)
    assert not torch.equal(actual[0],actual[1])
    payload={'architecture':'native-residual-v3','model_kwargs':{
        'base':2,'prior_mode':'boundary','conditioning':'instance'},
        'state_dict':model.state_dict(),'tile':32,'classes':[2,8]}
    path=tmp_path/'instance.pt';torch.save(payload,path)
    loaded=NativeDetailRefiner.load(path,'cpu')
    rgb=np.full((37,51,3),255,np.uint8);coarse=np.full((37,51),.2,np.float32)
    with pytest.raises(ValueError,match='detector box'):loaded.correct(rgb,coarse,8)
    result=loaded.correct(rgb,coarse,8,det_box=(5,6,44,31))
    assert result.shape==coarse.shape and np.isfinite(result).all()
    payload['architecture']='native-residual-v2';torch.save(payload,path)
    with pytest.raises(ValueError,match='version 3'):NativeDetailRefiner.load(path,'cpu')


def test_roi_passes_detector_box_in_native_crop_frame():
    segment={'command':0,'points':[[1,1],[2,1],[3,1],[4,1]],'length':3.}
    item={'class_id':8,'confidence':.9,'box':(10,20,70,80),
          'det_box':(23,31,59,66),'features':{},'segments':[segment],
          'probability':np.zeros((60,60),np.float32)}
    class Capture:
        uses_instance_prompt=True
        def correct(self,image,coarse,class_id,*,det_box):
            assert image.shape==(60,60,3) and class_id==8
            assert det_box==(13,11,49,46)
            return coarse
    refiner=RoiInkRefiner(torch.nn.Identity(),torch.device('cpu'))
    refiner.analyze=lambda *a,**kw:[item];refiner.native_detail=Capture()
    result=refiner.refine(np.full((100,100,3),255,np.uint8),[],100,100)
    assert result[0][2] is item['segments']
