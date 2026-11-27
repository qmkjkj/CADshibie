import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from cadvec.schema import validate_output
from tools.infer_query_instances import decode_output, input_paths, load_model, load_png_tensor


def prediction(classes, masks):
    queries, height, width = masks.shape
    logits = torch.full((1, queries, 31), -12.)
    for query, class_id in enumerate(classes):
        logits[0, query, class_id] = 12.
    return {'class_logits': logits, 'mask_embed': torch.zeros(1, queries, 32),
            'pixel_features': torch.zeros(1, 32, 1, 1).expand(1, 32, height, width),
            'boxes': torch.zeros(1, queries, 4), '_test_masks': masks[None]}


def dense(output, query_indices):
    return output['_test_masks'][:, query_indices]


def test_blank_png_with_null_queries_has_empty_valid_output(tmp_path):
    path = tmp_path/'blank.png'; Image.new('RGB', (35, 19), 'white').save(path)
    tensor, shape = load_png_tensor(path)
    assert tensor.shape == (1, 3, 32, 64) and torch.all(tensor == 1)
    output = prediction([30, 30], torch.full((2, 32, 64), -10.))
    data, metadata = decode_output(output, shape, dense)
    validate_output(data)
    assert data['commands'] == [] and data['width'] == data['height'] == 100
    assert metadata['selected_queries'] == metadata['emitted_instances'] == 0


def test_rgb_and_rgba_composite_to_white_with_right_bottom_padding(tmp_path):
    rgba = np.zeros((7, 5, 4), np.uint8)
    rgba[2, 3] = [255, 0, 0, 128]
    path = tmp_path/'rgba.png'; Image.fromarray(rgba).save(path)
    tensor, shape = load_png_tensor(path)
    assert shape == (7, 5) and tensor.shape == (1, 3, 32, 32)
    torch.testing.assert_close(tensor[0, :, 2, 3], torch.tensor([1., 127/255, 127/255]))
    assert torch.all(tensor[0, :, 0, 0] == 1)
    assert torch.all(tensor[:, :, 7:, :] == 1) and torch.all(tensor[:, :, :, 5:] == 1)
    rgb_path = tmp_path/'rgb.png'; Image.new('RGB', (5, 7), (0, 64, 255)).save(rgb_path)
    rgb, _ = load_png_tensor(rgb_path)
    torch.testing.assert_close(rgb[0, :, 0, 0], torch.tensor([0., 64/255, 1.]))


def test_same_class_overlapping_queries_remain_separate_instances():
    masks = torch.full((3, 32, 32), -10.)
    masks[:2, 12, 3:28] = 10.
    output = prediction([2, 2, 30], masks)
    data, metadata = decode_output(output, (32, 32), dense)
    validate_output(data)
    assert set(data['instanceIds']) == {0, 1} and set(data['semanticIds']) == {2}
    assert metadata['emitted_instances'] == 2
    first = [args for args, instance in zip(data['args'], data['instanceIds']) if instance == 0]
    second = [args for args, instance in zip(data['args'], data['instanceIds']) if instance == 1]
    assert first == second


def test_native_1000px_coordinates_and_padding_are_cropped_before_vectorization():
    masks = torch.full((1, 1024, 1024), -10.)
    masks[0, 999, 900:1000] = 10.
    masks[0, 1000:, :] = 10.; masks[0, :, 1000:] = 10.
    data, metadata = decode_output(prediction([8], masks), (1000, 1000), dense)
    assert data['commands'] and metadata['query_results'][0]['mask_pixels'] == 100
    xs = [row[i] for row in data['args'] for i in (0, 2, 4, 6)]
    ys = [row[i] for row in data['args'] for i in (1, 3, 5, 7)]
    assert min(xs) == 90.05 and max(xs) == 99.95
    assert all(y == 99.95 for y in ys)
    assert data['width'] == data['height'] == 100


def test_threshold_is_strict_and_null_ties_are_excluded():
    output = prediction([2], torch.full((1, 32, 32), 10.))
    output['class_logits'].fill_(-100.)
    output['class_logits'][0, 0, 2] = output['class_logits'][0, 0, 30] = 0.
    data, metadata = decode_output(output, (32, 32), dense, score_threshold=.1)
    assert metadata['selected_queries'] == 0 and not data['commands']


def test_checkpoint_restores_without_loading_pretrained_backbone(tmp_path, monkeypatch):
    class Fake(torch.nn.Module):
        def __init__(self, backbone_path, num_queries):
            super().__init__()
            assert backbone_path is None and num_queries == 100
            self.weight = torch.nn.Parameter(torch.zeros(1))
    monkeypatch.setitem(sys.modules, 'cadvec.query_instances', SimpleNamespace(QueryInstanceModel=Fake, dense_mask_logits=dense))
    path = tmp_path/'model.pt'
    torch.save({'architecture': 'query-ink-set-v1', 'model_kwargs': {'backbone_path': 'must-not-open.pt', 'num_queries': 100},
                'state_dict': {'weight': torch.tensor([3.])}}, path)
    model, decoder = load_model(path, torch.device('cpu'))
    assert not model.training and model.weight.item() == 3. and decoder is dense
    Image.new('RGB', (4, 4)).save(tmp_path/'input.png')
    (tmp_path/'ignored.svg').write_text('not an input', encoding='utf-8')
    assert [p.name for p in input_paths(tmp_path)] == ['input.png']
