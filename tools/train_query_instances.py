#!/usr/bin/env python3
"""Train a shared full-page feature map and a set of instance mask queries.

This is a separate architecture experiment. It does not modify hybrid weights,
use proposal boxes as inputs, or select a checkpoint on the 155-page development set.
"""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.query_data import QueryPageDataset, sample_training_points, sha256
from cadvec.query_instances import QueryInstanceModel, QueryInstanceLoss, dense_mask_logits


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def move_target(target, device):
    # Full native GT remains on CPU. Criterion transfers only sampled pixels.
    return {**target, 'labels': target['labels'].to(device), 'boxes': target['boxes'].to(device)}


@torch.no_grad()
def measure(model, dataset, device, limit):
    """Strict pixel-mask PQ, before vector export; never presented as final CAD PQ."""
    model.eval()
    tp = fp = fn = 0
    sum_iou = 0.
    pages = []
    for index in range(min(limit, len(dataset))):
        image, target = dataset.get(index)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = model(image[None].to(device))
        probabilities = output['class_logits'][0].float().softmax(-1)
        scores, classes = probabilities[:, :30].max(-1)
        queries = torch.where((scores > .5) & (scores > probabilities[:, 30]))[0]
        predicted = []
        valid_h, valid_w = target['valid_shape']
        for start in range(0, len(queries), 8):
            masks = dense_mask_logits(output, queries[start:start + 8])[0] > 0
            predicted.extend(mask[:valid_h, :valid_w].flatten().cpu().numpy() for mask in masks)
        gt = target['masks'][:, :valid_h, :valid_w].flatten(1).numpy()
        gt_classes = target['labels'].numpy()
        pred_classes = classes[queries].cpu().numpy()
        gt_areas = gt.sum(1)
        pred_areas = np.asarray([p.sum() for p in predicted])
        ious = np.zeros((len(gt), len(predicted)), dtype=np.float64)
        for gi, mask in enumerate(gt):
            for pi, pred in enumerate(predicted):
                if gt_classes[gi] == pred_classes[pi]:
                    intersection = np.count_nonzero(mask & pred)
                    ious[gi, pi] = intersection / max(gt_areas[gi] + pred_areas[pi] - intersection, 1)
        eligible = np.where(ious > .5, ious, 0.)
        rows, cols = linear_sum_assignment(eligible, maximize=True)
        values = [ious[r, c] for r, c in zip(rows, cols) if ious[r, c] > .5]
        page = {'stem': target['stem'], 'TP': len(values), 'FP': len(predicted) - len(values),
                'FN': len(gt) - len(values), 'sum_iou': float(sum(values)),
                'predictions': len(predicted), 'max_predicted_class_score': float(scores.max()),
                'best_same_class_mask_iou_mean': float(ious.max(axis=1).mean())
                if len(gt) and len(predicted) else 0.}
        pages.append(page)
        tp += page['TP']; fp += page['FP']; fn += page['FN']; sum_iou += page['sum_iou']
    denominator = tp + .5 * (fp + fn)
    return {'images': len(pages), 'TP': tp, 'FP': fp, 'FN': fn, 'sum_iou': sum_iou,
            'mask_PQ_before_vector_export': sum_iou / denominator if denominator else 0.,
            'note': 'Native mask diagnostic, not final vector CAD PQ. Fixed confidence/mask thresholds .5.',
            'per_image': pages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, default=Path('weights/yolo_seg.pt'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--train-pages', type=int, default=0)
    parser.add_argument('--evaluate-pages', type=int, default=8)
    parser.add_argument('--overfit', action='store_true', help='Training capacity probe only, no augmentation')
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--points', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or args.points < args.queries:
        parser.error('Positive steps and at least one point per query required')
    if args.out.exists():
        parser.error('Refuse to overwrite experiment directory')
    args.out.mkdir(parents=True)
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    train = QueryPageDataset(args.data, 'train', limit=args.train_pages)
    validation = train if args.overfit else QueryPageDataset(args.data, 'val')
    if max(row['instances'] for row in train.rows + validation.rows) > args.queries:
        parser.error('Dataset exceeds query capacity')
    kwargs = {'num_queries': args.queries, 'dim': 128, 'mask_dim': 32, 'decoder_layers': 3, 'heads': 4}
    model = QueryInstanceModel(backbone_path=None if args.resume else str(args.backbone), **kwargs)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['architecture'] != 'query-ink-set-v1' or checkpoint['model_kwargs'] != kwargs:
            raise ValueError('Incompatible resume architecture')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    criterion = QueryInstanceLoss()
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) in backbone_ids], 'lr': args.lr * .1},
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': args.lr},
    ], weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    protocol = {'architecture': 'query-ink-set-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset_manifest_sha256': sha256(args.data / 'manifest.json'),
                'backbone_sha256': sha256(args.backbone), 'resume_sha256': sha256(args.resume) if args.resume else None,
                'model_kwargs': kwargs, 'parameters': sum(p.numel() for p in model.parameters()),
                'train_stems': [r['stem'] for r in train.rows], 'validation_stems': [r['stem'] for r in validation.rows],
                'steps': args.steps, 'lr': args.lr, 'backbone_lr': args.lr * .1,
                'points': args.points, 'batch': 1, 'seed': args.seed, 'overfit_capacity_probe': args.overfit,
                'augmentation': 'none' if args.overfit else 'random right-angle rotation and horizontal flip',
                'selection': 'Last step only. No repeated dev selection or confidence tuning.',
                'source_files': {p: sha256(p) for p in ('cadvec/query_instances.py', 'cadvec/query_data.py',
                                                       'tools/train_query_instances.py', 'tools/build_query_instances.py')}}
    json_write(args.out / 'protocol.json', protocol)
    initial = measure(model, validation, device, args.evaluate_pages)
    json_write(args.out / 'initial_mask_diagnostic.json', initial)
    print('initial', json.dumps({k: v for k, v in initial.items() if k != 'per_image'}), flush=True)
    losses = []
    order = rng.permutation(len(train)); cursor = 0; started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if cursor == len(order):
            order = rng.permutation(len(train)); cursor = 0
        index = int(order[cursor]); cursor += 1
        image, target = train.get(index, rng=None if args.overfit else rng)
        indices = sample_training_points(image, target, rng, count=args.points).to(device)
        target_device = move_target(target, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = model(image[None].to(device))
            report = criterion(output, [target_device], indices)
            loss = report['total']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite step {step}: {report}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        row = {'step': step, 'stem': target['stem'],
               **{key: float(value.detach()) for key, value in report.items()
                  if torch.is_tensor(value) and value.numel() == 1}}
        losses.append(row)
        if step == 1 or step % 25 == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({'step': step, 'loss': row, 'seconds_per_step': elapsed / step,
                              'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2**20
                              if device.type == 'cuda' else None}), flush=True)
            json_write(args.out / 'losses.json', losses)
    seconds = time.perf_counter() - started
    checkpoint = {'architecture': 'query-ink-set-v1', 'model_kwargs': kwargs,
                  'state_dict': {key: val.detach().cpu() for key, val in model.state_dict().items()},
                  'protocol_sha256': sha256(args.out / 'protocol.json'), 'steps': args.steps}
    torch.save(checkpoint, args.out / 'last.pt')
    json_write(args.out / 'losses.json', losses)
    final = measure(model, validation, device, args.evaluate_pages)
    json_write(args.out / 'final_mask_diagnostic.json', final)
    result = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'training_seconds': seconds,
              'seconds_per_step': seconds / args.steps,
              'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
              'weight_sha256': sha256(args.out / 'last.pt'), 'initial': initial, 'final': final,
              'production_adopted': False,
              'note': 'Overfit probe measures training capacity only.' if args.overfit
              else 'Observed validation mask diagnostic only; vector export and full-page CAD regression required.'}
    json_write(args.out / 'result.json', result)
    print('final', json.dumps({k: v for k, v in final.items() if k != 'per_image'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
