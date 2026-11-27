#!/usr/bin/env python3
"""Label-using query geometry oracle on the exact observed capacity train pages.

This does not tune a threshold, produce inference predictions, or evaluate val/
dev. All-query geometry can expose a capacity failure hidden by low class scores.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('YOLO_OFFLINE','true')
os.environ.setdefault('YOLO_AUTOINSTALL','false')

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.query_data import QueryPageDataset,sha256
from cadvec.query_instances import QueryInstanceModel,dense_mask_logits
from tools.diagnose_instance_stages import assign


def matching_summary(matrix,classes=None,labels=None):
    matrix = np.asarray(matrix,dtype=np.float64)
    allowed = np.ones(matrix.shape,dtype=bool)
    if classes is not None:
        allowed = np.asarray(labels)[:,None] == np.asarray(classes)[None,:]
    eligible = (matrix>.5)&allowed
    matches = assign(matrix,eligible)
    values = [score for _query,score in matches.values()]
    best = np.where(allowed,matrix,0).max(1) if matrix.shape[1] else np.zeros(matrix.shape[0])
    return {'GT':matrix.shape[0],'one_to_one_TP':len(matches),
            'missed_GT':matrix.shape[0]-len(matches),'sum_matched_IoU':float(sum(values)),
            'sum_best_IoU':float(best.sum()),'mean_best_IoU':float(best.mean()) if len(best) else None,
            'independent_GT_with_IoU_over_half':int((best>.5).sum()),
            'assignment':[{'GT_row':int(i),'query':int(j),'IoU':float(score)} for i,(j,score) in matches.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    if not protocol.get('overfit_capacity_probe') or len(protocol['train_stems']) != 8:
        raise ValueError('This diagnosis is restricted to the frozen eight capacity training pages')
    dataset = QueryPageDataset(args.data,'train')
    by_stem = {row['stem']:i for i,row in enumerate(dataset.rows)}
    stems = protocol['train_stems']
    if len(set(stems)) != 8 or not set(stems)<=set(by_stem):
        raise ValueError('Capacity training pages differ from available original train pages')
    weight = args.run/'last.pt'; weight_hash = sha256(weight)
    checkpoint = torch.load(weight,map_location='cpu',weights_only=False)
    if checkpoint['architecture'] != 'query-ink-set-v1':
        raise ValueError('Unexpected checkpoint architecture')
    device = torch.device(args.device)
    model = QueryInstanceModel(backbone_path=None,**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['state_dict'],strict=True); model.to(device).eval()
    torch.set_num_threads(1)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    pages = []
    sums = {name:{'GT':0,'one_to_one_TP':0,'missed_GT':0,'sum_matched_IoU':0.,
                  'sum_best_IoU':0.,'independent_GT_with_IoU_over_half':0}
            for name in ('geometry_any_class','geometry_best_nonnull_class')}
    free_totals = {'TP':0,'FP':0,'FN':0,'predictions':0}
    with torch.no_grad():
        for stem in stems:
            image,target = dataset.get(by_stem[stem])
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                output = model(image[None].to(device))
            probability = output['class_logits'][0].float().softmax(-1)
            scores,classes = probability[:,:30].max(-1)
            accepted = (scores>.5)&(scores>probability[:,30])
            queries = output['mask_embed'].shape[1]
            h,w = target['valid_shape']
            truth = target['masks'][:,:h,:w].flatten(1).to(device=device,dtype=torch.float32)
            gt_areas = truth.sum(1)
            matrix = np.zeros((len(truth),queries),dtype=np.float64)
            areas = np.zeros(queries,dtype=np.int64)
            intersections = np.zeros((len(truth),queries),dtype=np.int64)
            for start in range(0,queries,8):
                selected = torch.arange(start,min(start+8,queries),device=device)
                mask = (dense_mask_logits(output,selected)[0,:,:h,:w]>0).flatten(1)
                area = mask.sum(1)
                # Multiplication sees only exact 0/1 values. Counts never exceed
                # 1M, so these float32 sums are exact integers before division.
                intersection = mask.float()@truth.T
                union = area[:,None].float()+gt_areas[None,:]-intersection
                ious = torch.where(union>0,intersection/union.clamp_min(1),torch.zeros_like(union))
                matrix[:,start:start+len(selected)] = ious.T.cpu().numpy()
                intersections[:,start:start+len(selected)] = intersection.T.cpu().numpy().astype(np.int64)
                areas[start:start+len(selected)] = area.cpu().numpy()
            class_ids = classes.cpu().numpy(); gt_labels = target['labels'].numpy()
            any_class = matching_summary(matrix)
            same_class = matching_summary(matrix,class_ids,gt_labels)
            keep = accepted.cpu().numpy()
            free = matching_summary(matrix[:,keep],class_ids[keep],gt_labels)
            free_counts = {'TP':free['one_to_one_TP'],'FP':int(keep.sum())-free['one_to_one_TP'],
                           'FN':free['missed_GT'],'predictions':int(keep.sum())}
            instances = []
            for i in range(len(truth)):
                best = int(matrix[i].argmax())
                overlap = int(intersections[i,best]); gt_area = int(gt_areas[i].item())
                instances.append({'GT_row':i,'class':int(gt_labels[i]),'best_geometry_query':best,
                    'IoU':float(matrix[i,best]),'GT_pixels':gt_area,'query_mask_pixels':int(areas[best]),
                    'intersection_pixels':overlap,'precision':overlap/max(int(areas[best]),1),
                    'recall':overlap/max(gt_area,1),
                    'true_class_probability':float(probability[best,int(gt_labels[i])]),
                    'null_probability':float(probability[best,30]),
                    'best_nonnull_class':int(class_ids[best]),'best_nonnull_probability':float(scores[best]),
                    'passes_unchanged_free_gate':bool(accepted[best])})
            page = {'stem':stem,'valid_shape':[h,w],'queries':queries,'GT':len(truth),
                    'geometry_any_class':any_class,'geometry_best_nonnull_class':same_class,
                    'free_gate_unchanged':free_counts,'instances':instances,
                    'query_mask_area_summary':{'min':int(areas.min()),'max':int(areas.max()),
                                               'mean':float(areas.mean()),'empty_queries':int((areas==0).sum())},
                    'checkpoint_mode':'eval','input_sha256':{
                        'png':sha256(dataset.rows[by_stem[stem]]['png']),
                        'cached_target':sha256(args.data/dataset.rows[by_stem[stem]]['target'])}}
            pages.append(page)
            for name,value in (('geometry_any_class',any_class),('geometry_best_nonnull_class',same_class)):
                for key in sums[name]:
                    sums[name][key] += value[key]
            for key in free_totals:
                free_totals[key] += free_counts[key]
            print(stem,'GT',len(truth),'any-class TP',any_class['one_to_one_TP'],
                  'pred-class TP',same_class['one_to_one_TP'],'free',free_counts,flush=True)
    for values in sums.values():
        values['mean_best_IoU'] = values['sum_best_IoU']/values['GT'] if values['GT'] else None
    if sha256(weight) != weight_hash:
        raise ValueError('Checkpoint changed during read-only diagnosis')
    result = {'schema_version':1,'scope':'Exact eight observed training capacity pages only; no val/dev accessed',
              'not_a_deployable_score':True,'no_oracle_PQ_reported':True,
              'fixed_mask_threshold':'logit>0 (sigmoid>.5)','matching':'Strict native-mask IoU>.5, sum-IoU Hungarian, one-to-one',
              'class_rule':'Best non-null argmax among 0..29; ignores confidence/null only for labeled oracle diagnosis',
              'all_queries_considered':True,'no_parameter_threshold_or_checkpoint_change':True,
              'geometry_any_class':sums['geometry_any_class'],
              'geometry_best_nonnull_class':sums['geometry_best_nonnull_class'],
              'free_gate_unchanged':free_totals,'pages':pages,
              'checkpoint_sha256':weight_hash,'capacity_protocol_sha256':sha256(args.run/'protocol.json'),
              'data_manifest_sha256':sha256(args.data/'manifest.json'),'tool_sha256':sha256(__file__),
              'limitations':['Oracle query-to-GT assignment uses labels and is not inference.',
                             'Native pixel-mask overlap precedes vector export and is not official CAD PQ.',
                             'Per-GT best-query values may reuse a query; only one-to-one_TP enforces unique assignment.',
                             'Oversampled training-point Dice does not imply dense pixel accuracy.',
                             'Both GT and predicted masks were cropped to the same original valid native frame.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=1),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('geometry_any_class','geometry_best_nonnull_class','free_gate_unchanged')}),flush=True)


if __name__=='__main__':
    main()
