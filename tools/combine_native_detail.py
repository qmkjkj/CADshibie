#!/usr/bin/env python3
"""Join native TRAIN crops while preserving the original validation index order."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--additional',type=Path,required=True)
    parser.add_argument('--source-splits',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    bm,am=(read(p/'metadata.json') for p in (args.base,args.additional))
    base,extra=(read(p/'index.json') for p in (args.base,args.additional))
    if not extra or any(r['role']!='train' for r in extra) or am['selected_pages']['val']:
        raise ValueError('Additional data must contain training crops only')
    if bm['coarse_weights']!=am['coarse_weights'] or bm['tile']!=am['tile']:
        raise ValueError('Coarse models and native tile scale must be identical')
    groups=read(args.source_splits)['groups']['train_final']
    train_pages=bm['selected_pages']['train']+am['selected_pages']['train']
    val_pages=bm['selected_pages']['val']
    if len(set(train_pages))!=len(train_pages):
        raise ValueError('Duplicated training pages')
    if {groups[n] for n in train_pages}&{groups[n] for n in val_pages}:
        raise ValueError('Training and validation source families intersect')
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Do not shuffle or regenerate validation rows: the same seed must select
    # exactly the original fixed 500 crops, in exactly the same order.
    for directory,index in ((args.base,base),(args.additional,extra)):
        for row in index:
            path=(directory/row['path']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(dict(row,path=Path(os.path.relpath(path,args.out.resolve())).as_posix()))
    metadata=dict(bm,selected_pages={'train':train_pages,'val':val_pages},
        expansion={'base_index_sha256':sha(args.base/'index.json'),
                   'additional_index_sha256':sha(args.additional/'index.json'),
                   'additional_metadata':am,
                   'source_splits_sha256':sha(args.source_splits),
                   'source_group_counts':{'train':len({groups[n] for n in train_pages}),
                                          'val':len({groups[n] for n in val_pages})},
                   'note':'Official TRAIN labels only; original validation crop rows/order preserved.'})
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=1),encoding='utf-8')
    (args.out/'index.json').write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
    print(json.dumps({'train_pages':len(train_pages),'val_pages':len(val_pages),
                      'train_crops':sum(r['role']=='train' for r in rows),
                      'val_crops':sum(r['role']=='val' for r in rows)}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
