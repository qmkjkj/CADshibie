#!/usr/bin/env python3
"""Build exact complete full-page masks for the experimental query instance model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.query_data import page_targets, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True,
                        help='Frozen native metadata, selected_pages train/val')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=100)
    args = parser.parse_args()
    if (args.out / 'manifest.json').exists():
        parser.error('Refuse to overwrite completed dataset')
    selection = json.loads(args.selection.read_text(encoding='utf-8'))['selected_pages']
    if set(selection['train']) & set(selection['val']):
        parser.error('Train/val page overlap')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for role in ('train', 'val'):
        (args.out / role).mkdir(exist_ok=True)
        for index, stem in enumerate(selection[role]):
            png = args.data_root / 'train_final' / (stem + '.png')
            label = args.data_root / 'train_json' / (stem + '.json')
            with Image.open(png) as image:
                shape = (image.height, image.width)
            target = page_targets(json.loads(label.read_text(encoding='utf-8')), shape)
            if len(target['labels']) > args.queries:
                raise ValueError(f'{stem}: {len(target["labels"])} GT exceed query capacity')
            output = args.out / role / (stem + '.npz')
            flattened = target['masks'].reshape(len(target['labels']), shape[0] * shape[1])
            np.savez_compressed(output, labels=target['labels'], boxes=target['boxes'],
                                instances=target['instances'], shape=shape,
                                packed_masks=np.packbits(flattened, axis=1))
            rows.append({'role': role, 'stem': stem, 'png': str(png.resolve()),
                         'target': output.relative_to(args.out).as_posix(),
                         'instances': len(target['labels']), 'shape': shape,
                         'label_sha256': sha256(label), 'png_sha256': sha256(png),
                         'target_sha256': sha256(output)})
            if (index + 1) % 100 == 0:
                print(role, index + 1, '/', len(selection[role]), flush=True)
    report = {'architecture': 'query-ink-set-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'selection_sha256': sha256(args.selection), 'max_instances': max(r['instances'] for r in rows),
              'queries': args.queries, 'pages': rows,
              'target': 'Complete native PIL1px Thing masks; overlapping instances remain separate; no detector/coarse inputs',
              'source_labels': 'Official train only. Previously observed source groups, not fresh confirmation.'}
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('built', len(rows), 'max_instances', report['max_instances'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
