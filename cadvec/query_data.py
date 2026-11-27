"""Training-only full-page targets for the experimental instance query decoder.

Targets retain individual one-pixel scorer masks, including overlapping strokes.
No detector proposals or predictions are used to construct this dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .layers import load_rgba
from .raster import render_window


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def page_targets(data, shape):
    """Return complete Thing masks, class IDs and native exclusive-edge boxes."""
    height, width = shape
    groups = {}
    for cls, inst, cmd, coords in zip(data['semanticIds'], data['instanceIds'],
                                     data['commands'], data['args']):
        if 0 <= cls < 30 and inst >= 0:
            groups.setdefault((cls, inst), []).append({
                'command': cmd, 'points': list(zip(coords[::2], coords[1::2]))})
    labels, masks, boxes, instances = [], [], [], []
    for (cls, inst), items in sorted(groups.items()):
        mask = render_window(items, shape, (width / data['width'], height / data['height']))
        if not mask.any():
            raise ValueError(f'Empty scorer target {(cls, inst)}')
        ys, xs = np.nonzero(mask)
        labels.append(cls)
        instances.append(inst)
        masks.append(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return {
        'labels': np.asarray(labels, dtype=np.int64),
        'instances': np.asarray(instances, dtype=np.int64),
        'masks': np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool),
        'boxes': np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
    }


def white_rgb(path):
    rgb, alpha = load_rgba(path)
    return np.clip(rgb * alpha[..., None] + 255 * (1 - alpha[..., None]), 0, 255).astype(np.uint8)


class QueryPageDataset:
    def __init__(self, directory, role, *, limit=0):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        self.rows = [row for row in self.manifest['pages'] if row['role'] == role]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f'No pages for {role}')

    def __len__(self):
        return len(self.rows)

    def get(self, index, rng=None):
        row = self.rows[index]
        image = white_rgb(row['png'])
        with np.load(self.directory / row['target']) as stored:
            count = len(stored['labels'])
            height, width = map(int, stored['shape'])
            masks = np.unpackbits(stored['packed_masks'], axis=1,
                                  count=height * width).reshape(count, height, width).astype(bool)
            labels = stored['labels'].copy()
            boxes = stored['boxes'].copy()
        if image.shape[:2] != (height, width):
            raise ValueError('Image/target dimensions changed')
        if rng is not None:
            turns = int(rng.integers(0, 4))
            for _ in range(turns):
                image = np.rot90(image)
                masks = np.rot90(masks, axes=(1, 2))
                x0, y0, x1, y1 = boxes.T.copy()
                boxes = np.stack((y0, width - x1, y1, width - x0), axis=1)
                height, width = width, height
            if rng.random() < .5:
                image = image[:, ::-1]
                masks = masks[:, :, ::-1]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        pad_h, pad_w = (height + 31) // 32 * 32, (width + 31) // 32 * 32
        padded = np.ones((3, pad_h, pad_w), dtype=np.float32)
        padded[:, :height, :width] = image.transpose(2, 0, 1) / 255.
        padded_masks = np.zeros((count, pad_h, pad_w), dtype=bool)
        padded_masks[:, :height, :width] = masks
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        extent = boxes[:, 2:] - boxes[:, :2]
        normalized_boxes = np.concatenate((center, extent), axis=1) / np.array(
            [pad_w, pad_h, pad_w, pad_h], dtype=np.float32)
        target = {'labels': torch.from_numpy(labels), 'masks': torch.from_numpy(padded_masks),
                  'boxes': torch.from_numpy(normalized_boxes.astype(np.float32)),
                  'valid_shape': (height, width), 'stem': row['stem']}
        return torch.from_numpy(padded), target


def sample_training_points(image, target, rng, count=8192):
    """Balance instances, visible input ink and uniform background; GT only here."""
    masks = target['masks'].numpy()
    height, width = image.shape[-2:]
    foreground = []
    foreground_budget = count // 4
    per_instance = max(1, foreground_budget // max(len(masks), 1))
    for mask in masks:
        positions = np.flatnonzero(mask)
        if len(positions):
            foreground.append(rng.choice(positions, size=per_instance, replace=True))
    foreground = np.concatenate(foreground) if foreground else np.empty(0, dtype=np.int64)
    ink = np.flatnonzero(image.numpy().min(axis=0) < .92)
    hard = (rng.choice(ink, size=count // 4, replace=True) if len(ink)
            else rng.integers(0, height * width, size=count // 4))
    uniform = rng.integers(0, height * width, size=count - len(foreground) - len(hard))
    indices = np.concatenate((foreground, hard, uniform))
    rng.shuffle(indices)
    return torch.from_numpy(indices.astype(np.int64))
