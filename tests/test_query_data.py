import numpy as np
import torch

from cadvec.query_data import page_targets, sample_training_points


def test_full_masks_keep_touching_same_class_instances_and_scoring_geometry():
    data = {'semanticIds': [2, 2, 30, 2], 'instanceIds': [0, 1, -1, -1],
            'commands': [0, 0, 0, 0], 'width': 100, 'height': 100,
            'args': [[10, 20, 20, 20, 30, 20, 40, 20],
                     [20, 10, 20, 20, 20, 30, 20, 40],
                     [10, 20, 20, 20, 30, 20, 40, 20],
                     [10, 20, 20, 20, 30, 20, 40, 20]]}
    target = page_targets(data, (100, 100))
    assert target['labels'].tolist() == [2, 2]
    assert target['masks'].shape == (2, 100, 100)
    assert target['masks'][:, 20, 20].all()
    assert target['boxes'].tolist() == [[10, 20, 41, 21], [20, 10, 21, 41]]


def test_training_sampling_includes_each_instance_and_visible_hard_ink():
    image = torch.ones(3, 32, 32)
    image[:, 4, :] = 0
    masks = torch.zeros(2, 32, 32, dtype=torch.bool)
    masks[0, 1, 1] = True
    masks[1, 30, 30] = True
    points = sample_training_points(image, {'masks': masks}, np.random.default_rng(3), 128)
    assert points.shape == (128,)
    assert 33 in points and (30 * 32 + 30) in points
    assert sum((points // 32 == 4).tolist()) >= 32
    assert points.min() >= 0 and points.max() < 1024


def test_empty_page_targets_and_sampling_remain_valid():
    target = page_targets({'semanticIds': [], 'instanceIds': [], 'commands': [],
                           'args': [], 'width': 100, 'height': 100}, (32, 32))
    assert target['masks'].shape == (0, 32, 32)
    points = sample_training_points(torch.ones(3, 32, 32),
                                   {'masks': torch.from_numpy(target['masks'])},
                                   np.random.default_rng(1), 64)
    assert points.shape == (64,)


def _stored_query_dataset(tmp_path, masks):
    import json
    from PIL import Image
    from cadvec.query_data import QueryPageDataset

    count, height, width = masks.shape
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if count:
        image[masks.any(0)] = 0
    Image.fromarray(image).save(tmp_path/'page.png')
    np.savez(tmp_path/'page.npz', shape=(height, width), labels=np.arange(count, dtype=np.int64),
             boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
             packed_masks=np.packbits(masks.reshape(count, height*width), axis=1))
    (tmp_path/'manifest.json').write_text(json.dumps({'pages': [
        {'role': 'train', 'stem': 'page', 'png': str(tmp_path/'page.png'), 'target': 'page.npz'}]}), encoding='utf-8')
    return QueryPageDataset(tmp_path, 'train')


class _FixedAugmentation:
    def __init__(self, turns, flip):
        self.turns, self.flip = turns, flip

    def integers(self, low, high):
        assert (low, high) == (0, 4)
        return self.turns

    def random(self):
        return 0. if self.flip else 1.


def test_nonsquare_augmentation_keeps_one_pixel_targets_boxes_and_padding_aligned(tmp_path):
    masks = np.zeros((2, 35, 67), dtype=bool)
    masks[0, 2, 1:10] = True
    masks[1, 34, 66] = True
    dataset = _stored_query_dataset(tmp_path, masks)
    for turns in range(4):
        for flip in (False, True):
            image, target = dataset.get(0, _FixedAugmentation(turns, flip))
            expected = np.rot90(masks, turns, axes=(1, 2))
            if flip:
                expected = expected[:, :, ::-1]
            height, width = expected.shape[-2:]
            pad_h, pad_w = image.shape[-2:]
            assert target['valid_shape'] == (height, width)
            assert pad_h % 32 == pad_w % 32 == 0
            np.testing.assert_array_equal(target['masks'][:, :height, :width].numpy(), expected)
            assert target['masks'][1].sum().item() == 1
            assert not target['masks'][:, height:, :].any() and not target['masks'][:, :, width:].any()
            assert torch.all(image[:, height:, :] == 1) and torch.all(image[:, :, width:] == 1)
            np.testing.assert_array_equal(image[:, :height, :width].numpy().min(0) < .5, expected.any(0))
            for index, mask in enumerate(expected):
                ys, xs = np.nonzero(mask)
                center, extent = target['boxes'][index, :2].numpy(), target['boxes'][index, 2:].numpy()
                recovered = np.r_[center-extent/2, center+extent/2] * [pad_w, pad_h, pad_w, pad_h]
                np.testing.assert_allclose(recovered, [xs.min(), ys.min(), xs.max()+1, ys.max()+1], atol=1e-5)


def test_empty_nonsquare_page_augmentation_preserves_empty_targets(tmp_path):
    dataset = _stored_query_dataset(tmp_path, np.zeros((0, 35, 67), dtype=bool))
    image, target = dataset.get(0, _FixedAugmentation(1, True))
    assert image.shape == (3, 96, 64) and torch.all(image == 1)
    assert target['valid_shape'] == (67, 35)
    assert target['labels'].shape == (0,) and target['boxes'].shape == (0, 4)
    assert target['masks'].shape == (0, 96, 64)
