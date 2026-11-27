#!/usr/bin/env python3
"""PNG-only inference for the experimental native query instance model.

This entry point is separate from infer.py. Every query uses the same class and
mask thresholds; masks may overlap, without NMS or the existing quality gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.layers import load_rgba
from cadvec.pipeline_segdet import _Primitive
from cadvec.roi_ink import vectorize_mask
from cadvec.schema import assemble_output, validate_output, write_output


ARCHITECTURE = 'query-ink-set-v1'
CANVAS_SIZE = 100
MASK_BATCH = 8


def load_png_tensor(path):
    """Composite on white and pad only the right/bottom to multiples of32."""
    path = Path(path)
    if path.suffix.lower() != '.png':
        raise ValueError('Query inference accepts PNG files only')
    rgb, alpha = load_rgba(path)
    image = np.clip(rgb.astype(np.float32)*alpha[..., None] + 255*(1-alpha[..., None]), 0, 255)
    height, width = image.shape[:2]
    if min(height, width) <= 0:
        raise ValueError('Expected a nonempty PNG')
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255.
    tensor = F.pad(tensor, (0, (-width) % 32, 0, (-height) % 32), value=1.)
    return tensor, (height, width)


def validate_threshold(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name+' must be finite and within[0,1]')
    return value


def decode_output(output, original_shape, dense_function, *, score_threshold=.5, mask_threshold=.5):
    """Vectorize each retained query independently in the original PNG frame."""
    score_threshold = validate_threshold(score_threshold, 'score threshold')
    mask_threshold = validate_threshold(mask_threshold, 'mask threshold')
    height, width = map(int, original_shape)
    logits = output['class_logits']
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != 31:
        raise ValueError('Expected batch1 query logits with30 Thing classes and null class30')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite query class logits')
    spatial_shape = tuple(output['pixel_features'].shape[-2:])
    if min(height, width) <= 0 or height > spatial_shape[0] or width > spatial_shape[1]:
        raise ValueError('Original PNG does not fit the native pixel feature frame')
    probabilities = logits.float().softmax(-1)[0]
    scores, classes = probabilities[:, :30].max(-1)
    null = probabilities[:, 30]
    # Strict comparisons exclude the null winner, including a tie with null.
    selected = torch.nonzero((scores > score_threshold) & (scores > null), as_tuple=False).flatten()
    primitives = []; entries = []; emitted = 0
    for start in range(0, len(selected), MASK_BATCH):
        indices = selected[start:start+MASK_BATCH]
        masks = dense_function(output, indices)
        if tuple(masks.shape) != (1, len(indices), *spatial_shape):
            raise ValueError('Dense query masks must match the full native padded feature frame')
        if not torch.isfinite(masks).all():
            raise ValueError('Nonfinite dense query mask logits')
        # Crop before thinning/vectorization, so padding cannot extend a path.
        native = masks[0, :, :height, :width].float().sigmoid().cpu().numpy()
        for query_id, probability in zip(indices.tolist(), native):
            class_id = int(classes[query_id]); score = float(scores[query_id])
            segments = vectorize_mask(probability, 0, 0, width, height,
                width/CANVAS_SIZE, height/CANVAS_SIZE, threshold=mask_threshold, chunk=6, adaptive=False)
            instance_id = emitted if segments else None
            for points, length in segments:
                payload = {'command': 0, 'points': points, 'length': float(length),
                           'semantic': class_id, 'rgb': [0, 0, 0], 'layer': 0, 'width': 1.}
                primitives.append(_Primitive(payload=payload, instance=emitted))
            entries.append({'query': int(query_id), 'class': class_id, 'score': score,
                'null_score': float(null[query_id]), 'instance_id': instance_id,
                'mask_pixels': int(np.count_nonzero(probability > mask_threshold)), 'primitives': len(segments)})
            emitted += bool(segments)
    data = assemble_output(primitives, width=CANVAS_SIZE, height=CANVAS_SIZE)
    validate_output(data)
    metadata = {'original_size': [height, width], 'padded_size': list(spatial_shape),
                'queries': int(logits.shape[1]), 'selected_queries': len(selected),
                'emitted_instances': emitted, 'score_threshold': score_threshold,
                'mask_threshold': mask_threshold, 'query_results': entries}
    return data, metadata


def load_model(path, device):
    # Importing lazily also makes decoding testable without a trained model.
    from cadvec.query_instances import QueryInstanceModel, dense_mask_logits
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected a '+ARCHITECTURE+' checkpoint')
    kwargs = dict(checkpoint['model_kwargs'])
    # Inference restores every parameter from the checkpoint, never a backbone
    # package, pretrained download or a second local model file.
    kwargs['backbone_path'] = None
    model = QueryInstanceModel(**kwargs)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device).eval()
    return model, dense_mask_logits


def input_paths(path, limit=0):
    path = Path(path)
    if limit < 0:
        raise ValueError('Limit must be nonnegative')
    if path.is_file():
        paths = [path] if path.suffix.lower() == '.png' else []
    elif path.is_dir():
        paths = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.png'),
                       key=lambda p: (p.name.lower(), p.name))
    else:
        raise ValueError('Input does not exist: '+str(path))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError('No PNG files in the input')
    stems = [p.stem.lower() for p in paths]
    if len(stems) != len(set(stems)):
        raise ValueError('Input PNG names would overwrite prediction files')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--score-threshold', type=float, default=.5)
    parser.add_argument('--mask-threshold', type=float, default=.5)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    validate_threshold(args.score_threshold, 'score threshold')
    validate_threshold(args.mask_threshold, 'mask threshold')
    paths = input_paths(args.input, args.limit)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    model, dense_function = load_model(args.weights, device)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output.parent/(args.output.resolve().name+'_metadata')
    metadata_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); instances = 0
    with torch.inference_mode():
        for index, path in enumerate(paths):
            page_started = time.perf_counter()
            tensor, shape = load_png_tensor(path)
            output = model(tensor.to(device))
            data, metadata = decode_output(output, shape, dense_function,
                score_threshold=args.score_threshold, mask_threshold=args.mask_threshold)
            write_output(data, args.output/(path.stem+'.json'))
            metadata.update(file=path.name, seconds=round(time.perf_counter()-page_started, 4))
            (metadata_dir/(path.stem+'.json')).write_text(json.dumps(metadata, indent=1, allow_nan=False)+'\n', encoding='utf-8')
            instances += metadata['emitted_instances']
            print(f'query inference {index+1}/{len(paths)}: {path.name}, {metadata["emitted_instances"]} instances', flush=True)
    summary = {'architecture': ARCHITECTURE, 'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
               'images': len(paths), 'emitted_instances': instances, 'device': str(device),
               'seconds': round(time.perf_counter()-started, 4), 'score_threshold': args.score_threshold,
               'mask_threshold': args.mask_threshold, 'output': str(args.output), 'metadata': str(metadata_dir)}
    (metadata_dir/'_summary.json').write_text(json.dumps(summary, indent=1)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
