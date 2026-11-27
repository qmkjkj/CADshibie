#!/usr/bin/env python3
"""Recover target-box metadata for an unchanged native-detail crop index.

Only the frozen detector is evaluated. Coarse masks and NPZ pixels are reused;
candidate enumeration and GT supplementation mirror build_native_detail.py.
"""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
os.environ.setdefault('YOLO_OFFLINE','1')

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.native_detail import tile_positions
from cadvec.pipeline_segdet import _model
from cadvec.raster import render_window
from cadvec.roi_ink import RoiInkRefiner
from tools.build_native_detail import bbox_iou


ROUTES = ({2},{3,4,22,26,29},{8,12,14,15,18,19,20,21,25,27,28})
ROUTED_CLASSES = set.union(*ROUTES)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_truths(gt,width,height):
    groups = {}
    for semantic,instance,command,coordinates in zip(
            gt['semanticIds'],gt['instanceIds'],gt['commands'],gt['args']):
        if semantic not in ROUTED_CLASSES or instance < 0:
            continue
        groups.setdefault((semantic,instance),[]).append({
            'command':command,'points':list(zip(coordinates[::2],coordinates[1::2]))})
    native_scale = (width/gt['width'],height/gt['height'])
    rendered = {key:render_window(items,(height,width),native_scale)
                for key,items in groups.items()}
    truths = {}
    for (class_id,instance),mask in rendered.items():
        if not mask.any():
            continue
        ys,xs = np.nonzero(mask)
        box = (float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max()))
        truths.setdefault(class_id,[]).append((instance,mask,box))
    return truths,native_scale


def select_candidates(truths,detections,role,max_per_class=4):
    """Preserve detector order and pre-ambiguity class-count consumption."""
    candidates = []; per_class = Counter()
    for class_id,confidence,polygon in detections:
        class_id = int(class_id)
        if class_id not in ROUTED_CLASSES or len(polygon)<3:
            continue
        if per_class[class_id] >= max_per_class:
            continue
        per_class[class_id] += 1
        polygon = np.asarray(polygon)
        box = (float(polygon[:,0].min()),float(polygon[:,1].min()),
               float(polygon[:,0].max()),float(polygon[:,1].max()))
        choices = sorted(((bbox_iou(box,truth[2]),i) for i,truth in
                          enumerate(truths.get(class_id,[]))),reverse=True)
        if choices and choices[0][0]>=.45:
            if len(choices)>1 and choices[1][0]>choices[0][0]-.1:
                continue
            target_index = choices[0][1]
        elif not choices or choices[0][0]==0:
            target_index = None
        else:
            continue
        candidates.append((class_id,float(confidence),box,target_index,'detector'))
    for class_id,entries in (truths.items() if role=='train' else []):
        for i,(_instance,_mask,box) in enumerate(entries[:2]):
            if not any(c[0]==class_id and c[3]==i for c in candidates):
                candidates.append((class_id,1.,box,i,'gt_supplement'))
    return candidates


def context_for_page(rows,truths,candidates,width,height,tile,base_dir,*,confidence_tolerance=0.):
    if len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Duplicate original crop paths')
    wanted = {row['path']:row for row in rows}
    records = {}; counts = Counter(); maximum_confidence_error = 0.
    for classes in ROUTES:
        group = [candidate for candidate in candidates if candidate[0] in classes]
        for number,candidate in enumerate(group):
            class_id,confidence,box,target_index,source = candidate
            # _crop only uses box/frame and the fixed PAD constant, so no coarse
            # model is loaded or evaluated to recover these exact coordinates.
            left,top,right,bottom = RoiInkRefiner._crop(None,box,width,height)
            instance = truths[class_id][target_index][0] if target_index is not None else -1
            positions = [(x,y) for y in tile_positions(bottom-top,tile)
                         for x in tile_positions(right-left,tile)]
            for tile_index,(x,y) in enumerate(positions):
                # The old coarse-probability empty-tile filter is already
                # represented by the existing index. Never create new samples.
                if not rows:
                    continue
                role,stem = rows[0]['role'],rows[0]['stem']
                relative = f'{role}/{stem}_{class_id}_{number}_{tile_index}.npz'
                row = wanted.get(relative)
                if row is None:
                    continue
                if any(row[key] != value for key,value in (
                        ('class',class_id),('instance',instance),('source',source),('role',role),('stem',stem))):
                    raise ValueError(f'Candidate identity differs: {relative}')
                confidence_error = abs(float(row['confidence'])-confidence)
                maximum_confidence_error = max(maximum_confidence_error,confidence_error)
                if confidence_error > confidence_tolerance:
                    raise ValueError(f'Candidate confidence differs: {relative}: {confidence} != {row["confidence"]}')
                valid_h,valid_w = min(tile,bottom-top-y),min(tile,right-left-x)
                if target_index is None:
                    target = np.zeros((valid_h,valid_w),dtype=bool)
                else:
                    target = truths[class_id][target_index][1][top+y:top+y+valid_h,left+x:left+x+valid_w]
                with np.load(base_dir/relative,allow_pickle=False) as sample:
                    if list(sample['valid_shape']) != [valid_h,valid_w]:
                        raise ValueError(f'Tile valid shape differs: {relative}')
                    original_target = sample['target']
                    expected_target = np.zeros((tile,tile),dtype=np.uint8)
                    expected_target[:valid_h,:valid_w] = target
                    if not np.array_equal(original_target,expected_target):
                        raise ValueError(f'Tile target pixels differ: {relative}')
                if bool(target.any()) != row['positive']:
                    raise ValueError(f'Tile positivity differs: {relative}')
                records[relative] = {
                    'target_box':[box[0]-left,box[1]-top,box[2]-left,box[3]-top],
                    'tile_origin':[x,y]}
                counts[f'{role}_{source}_{"positive" if row["positive"] else "negative"}'] += 1
    missing = set(wanted)-set(records)
    if missing:
        raise ValueError(f'{len(missing)} original crop paths unmatched: {sorted(missing)[:5]}')
    return records,counts,maximum_confidence_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-data',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--detector',type=Path,default=Path('weights/yolo_seg.pt'))
    parser.add_argument('--max-per-class',type=int,default=4)
    parser.add_argument('--confidence-tolerance',type=float,default=0.)
    parser.add_argument('--limit-pages',type=int,default=0,help='Probe first N pages per role; 0 builds all')
    args = parser.parse_args()
    index_path,metadata_path = args.base_data/'index.json',args.base_data/'metadata.json'
    index_hash,metadata_hash = sha256(index_path),sha256(metadata_path)
    rows = json.loads(index_path.read_text(encoding='utf-8'))
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if sha256(args.detector) != metadata['coarse_weights'][str(args.detector).replace('\\','/')]:
        raise ValueError('Detector weights differ from original crop builder')
    tile = int(metadata['tile']); selected = metadata['selected_pages']
    pages = [(role,stem) for role in ('train','val') for stem in
             (selected[role][:args.limit_pages] if args.limit_pages else selected[role])]
    if len(set(pages)) != len(pages):
        raise ValueError('Duplicate selected pages')
    if set(selected['train']) & set(selected['val']):
        raise ValueError('Training and validation pages overlap')
    row_groups = defaultdict(list)
    for row in rows:
        row_groups[(row['role'],row['stem'])].append(row)
    if not set(row_groups).issubset(set((role,stem) for role in selected for stem in selected[role])):
        raise ValueError('Original rows refer to unselected pages')
    detector = _model(args.detector)
    records = {}; counts = Counter(); provenance_pages = {}; max_confidence_error = 0.
    for number,(role,stem) in enumerate(pages):
        png = args.data_root/'train_final'/f'{stem}.png'
        label = args.data_root/'train_json'/f'{stem}.json'
        with Image.open(png) as image:
            width,height = image.size
        gt = json.loads(label.read_text(encoding='utf-8'))
        truths,scale = render_truths(gt,width,height)
        if any(row['scale_xy'] != list(scale) for row in row_groups[(role,stem)]):
            raise ValueError(f'Original native scale differs: {stem}')
        detection = detector.predict(source=str(png),imgsz=640,conf=.2,iou=.5,verbose=False,device=0)[0]
        detections = []
        if detection.boxes is not None and detection.masks is not None:
            detections = zip(detection.boxes.cls.cpu().numpy(),detection.boxes.conf.cpu().numpy(),detection.masks.xy)
        candidates = select_candidates(truths,detections,role,args.max_per_class)
        recovered,page_counts,error = context_for_page(row_groups[(role,stem)],truths,candidates,width,height,
                                                       tile,args.base_data,confidence_tolerance=args.confidence_tolerance)
        if set(records)&set(recovered):
            raise ValueError('Recovered paths overlap between pages')
        records.update(recovered); counts.update(page_counts); max_confidence_error = max(max_confidence_error,error)
        provenance_pages[f'{role}/{stem}'] = {'png_sha256':sha256(png),'gt_sha256':sha256(label),
                                              'rows':len(recovered),'candidates':len(candidates)}
        if (number+1)%25 == 0:
            print(f'Context {number+1}/{len(pages)} pages; {len(records)} crops',flush=True)
    expected_rows = [row for row in rows if (row['role'],row['stem']) in set(pages)]
    if len(records) != len(expected_rows) or set(records) != {row['path'] for row in expected_rows}:
        raise ValueError('Recovered context does not cover original index exactly')
    if sha256(index_path) != index_hash or sha256(metadata_path) != metadata_hash:
        raise ValueError('Original index or metadata changed during recovery')
    result = {'schema_version':1,'base_index_sha256':index_hash,'base_metadata_sha256':metadata_hash,
              'records':{row['path']:records[row['path']] for row in expected_rows},
              'provenance':{'detector_sha256':sha256(args.detector),'tile':tile,'max_per_class':args.max_per_class,
                            'detector_settings':{'imgsz':640,'conf':.2,'iou':.5,'source':'PNG path','device':0},
                            'confidence_tolerance':args.confidence_tolerance,'maximum_confidence_error':max_confidence_error,
                            'pages':provenance_pages,'counts':dict(counts),'row_count':len(records),
                            'complete':not bool(args.limit_pages),'original_row_order_preserved':True,
                            'all_target_pixels_and_valid_shapes_verified':True,
                            'source_files_sha256':{path:sha256(path) for path in (
                                __file__,'tools/build_native_detail.py','cadvec/raster.py','cadvec/roi_ink.py')},
                            'note':'No original crop pixels, coarse probabilities, index rows or order were changed. '
                                   'Detector class is retained even for negatives. All original indexed GT-supplement negatives are included.'}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=1),encoding='utf-8')
    print(json.dumps({'pages':len(pages),'rows':len(records),'counts':dict(counts),
                      'max_confidence_error':max_confidence_error,'complete':not bool(args.limit_pages)}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
