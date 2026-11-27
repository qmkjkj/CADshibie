#!/usr/bin/env python3
"""Select source-balanced additional official TRAIN pages, protecting local roles.

The original joint image/source audit already includes every train/test/dev
page. Its complete family closure is used without opening evaluation labels.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-splits',type=Path,required=True)
    parser.add_argument('--native-metadata',type=Path,required=True)
    parser.add_argument('--protected-protocol',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2731)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source=read(args.source_splits);metadata=read(args.native_metadata)
    protocol=read(args.protected_protocol)
    groups=source['groups']['train_final']
    # Protect the whole original validation role, not just its viewed subset.
    protected_pages=sorted(set(source['train_final']['val'])
                           |set(protocol['incremental_confirmation'])|{'3UElq'})
    protected_groups={groups[n] for n in protected_pages}
    original_groups={groups[n] for n in metadata['selected_pages']['train']}
    if original_groups & protected_groups:
        raise ValueError('Existing native training intersects a protected family')
    with (args.data_root/'train_final/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        manifest={r['dest_stem'] for r in csv.DictReader(f)}
    if manifest!=set(groups):
        raise ValueError('Joint audit must cover exactly the official train manifest')
    eligible=defaultdict(list)
    for stem in sorted(manifest):
        family=groups[stem]
        if family not in protected_groups|original_groups:
            eligible[family].append(stem)
    # One page per additional family avoids amplifying the largest projects.
    rng=random.Random(args.seed)
    selected=sorted(rng.choice(sorted(eligible[g])) for g in sorted(eligible))
    test_groups=set(source['groups']['test_final'].values())
    payload={
        'schema_version':1,'seed':args.seed,
        'train_final':{'train':selected,'val':[]},
        'groups':{'train_final':{n:groups[n] for n in selected}},
        'selected_pages':len(selected),'selected_source_groups':len(eligible),
        'eligible_pages_before_family_balance':sum(map(len,eligible.values())),
        'protected_pages':protected_pages,'protected_source_groups':len(protected_groups),
        'test_related_selected_source_groups':len(set(eligible)&test_groups),
        'test_related_note':'These are official train pages sharing candidate source families with historical test. Historical test is regression only; its labels are never read for training. Fresh incremental confirmation remains source disjoint.',
        'method':'One seed-fixed random page from each additional family in the precomputed full joint audit. Protect all original train-validation families, round3 confirmation families and the previously labeled 3UElq family; exclude already trained native families.',
        'source_splits_sha256':sha(args.source_splits),
        'native_metadata_sha256':sha(args.native_metadata),
        'protected_protocol_sha256':sha(args.protected_protocol),
        'official_train_manifest_sha256':sha(args.data_root/'train_final/manifest.csv'),
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=1),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k in (
        'selected_pages','selected_source_groups','eligible_pages_before_family_balance',
        'protected_source_groups','test_related_selected_source_groups')}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
