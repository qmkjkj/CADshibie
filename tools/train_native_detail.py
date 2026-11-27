#!/usr/bin/env python3
"""Train a bounded-memory native residual head on disjoint training pages."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.detail_unet import NativeDetailUNet
from cadvec.instance_prompt import coordinate_prompt, augment_prompt
from cadvec.raster import render_window
from cadvec.roi_ink import vectorize_mask


class DetailDataset(Dataset):
    def __init__(self, root, rows, augment=False, context=None):
        self.root, self.rows, self.augment = root, rows, augment
        self.context = context

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=256)
    def _read(self, index):
        with np.load(self.root/self.rows[index]["path"]) as data:
            return data["image"].copy(), data["target"].copy(), data["valid_shape"].copy()

    def __getitem__(self, index):
        image, target, (height, width) = self._read(index)
        valid = np.zeros_like(target); valid[:height, :width] = 1
        prompt = None
        if self.context is not None:
            record = self.context[self.rows[index]['path']]
            prompt = coordinate_prompt(target.shape, record['target_box'],record['tile_origin'],
                                       (height,width))
        if self.augment:
            rotation = random.randrange(4)
            image, target, valid = (np.rot90(a, rotation).copy() for a in (image,target,valid))
            flip = random.random() < .5
            if flip:
                image, target, valid = (np.flip(a, 1).copy() for a in (image,target,valid))
            if prompt is not None:
                prompt = augment_prompt(prompt,rotation,flip)
        result = (torch.from_numpy(image.copy()).permute(2,0,1).float()/255,
                torch.from_numpy(target.copy())[None].float(),
                torch.from_numpy(valid.copy())[None].float(),
                float(self.rows[index].get("coarse_threshold",.5)),
                torch.tensor(self.rows[index].get("scale_xy",[10.,10.])))
        if self.context is not None:
            result += (int(self.rows[index]['class']),torch.from_numpy(prompt.copy()))
        return result


def vector_score(probability, target, valid, threshold=.5, scale_xy=(10.,10.)):
    height, width = target.shape
    probability = probability.copy(); probability[~valid] = 0
    sx,sy = map(float,scale_xy)
    segments = vectorize_mask(probability, 0, 0, width, height, sx, sy, threshold)
    rendered = render_window([{"command": 0, "points": points}
                              for points, _ in segments], (height, width), (sx,sy))
    intersection = np.count_nonzero(rendered & target)
    union = np.count_nonzero(rendered | target)
    iou = intersection / max(union, 1)
    if target.any():
        return (1,0,0,iou) if iou > .5 else (0,int(rendered.any()),1,0.)
    return (0,int(rendered.any()),0,0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--validation-crops", type=int, default=500)
    parser.add_argument("--prior-mode", choices=("fixed", "boundary"), default="fixed")
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--boundary-logit", type=float, default=2.)
    parser.add_argument("--conditioning", choices=("none", "instance"), default="none")
    parser.add_argument("--instance-context", type=Path,
                        help="Audited detector-box/tile metadata; required for instance conditioning")
    parser.add_argument("--init", type=Path, help="Warmstart compatible weights; epoch 0 is also eligible")
    parser.add_argument("--train-crops-per-epoch", type=int, default=0,
                        help="Fixed optimizer budget; randomly sample without replacement each epoch")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((args.data/"metadata.json").read_text(encoding="utf-8"))
    rows = json.loads((args.data/"index.json").read_text(encoding="utf-8"))
    context = None
    if args.instance_context:
        context_payload = json.loads(args.instance_context.read_text(encoding='utf-8'))
        if (context_payload['base_index_sha256'] != hashlib.sha256((args.data/'index.json').read_bytes()).hexdigest()
                or context_payload['base_metadata_sha256'] != hashlib.sha256((args.data/'metadata.json').read_bytes()).hexdigest()):
            raise ValueError('Instance context belongs to a different native crop dataset')
        context = context_payload['records']
        if set(context) != {r['path'] for r in rows}:
            raise ValueError('Instance context must cover exactly the original crop index')
    if args.conditioning == 'instance' and context is None:
        parser.error('--conditioning instance requires --instance-context')
    train = [r for r in rows if r["role"] == "train"]
    val = [r for r in rows if r["role"] == "val"]
    if not train or not val or {r["stem"] for r in train} & {r["stem"] for r in val}:
        raise ValueError("Need nonempty disjoint train/validation pages")
    # A fixed random crop sample selects experimental checkpoints. This is
    # a proxy; final deployment is selected only by end-to-end page PQ.
    rng = random.Random(args.seed)
    rng.shuffle(val)
    val = val[:args.validation_crops] if args.validation_crops > 0 else val
    train_dataset = DetailDataset(args.data, train, True, context)
    sampler = None
    if args.train_crops_per_epoch:
        if not 0 < args.train_crops_per_epoch <= len(train):
            parser.error("--train-crops-per-epoch must be positive and <= available crops")
        sampler = RandomSampler(train_dataset, replacement=False,
                                num_samples=args.train_crops_per_epoch,
                                generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, shuffle=sampler is None,
                              num_workers=0, pin_memory=device.type=="cuda",
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(DetailDataset(args.data, val, context=context), batch_size=args.batch_size,
                            num_workers=0, pin_memory=device.type=="cuda",
                            generator=torch.Generator().manual_seed(args.seed+1))
    model_kwargs = dict(base=args.base, prior_mode=args.prior_mode,
                        boundary_radius=args.boundary_radius, boundary_logit=args.boundary_logit,
                        conditioning=args.conditioning,class_count=36)
    model = NativeDetailUNet(**model_kwargs).to(device)
    init_sha256 = None
    if args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        initial_kwargs = checkpoint["model_kwargs"]
        architecture = checkpoint.get('architecture')
        if ((architecture == 'native-residual-v1' and initial_kwargs.get('prior_mode','fixed') != 'fixed')
                or ((initial_kwargs.get('conditioning','none') == 'instance') != (architecture == 'native-residual-v3'))):
            raise ValueError('Warmstart checkpoint version disagrees with its prior/conditioning metadata')
        if (checkpoint.get("architecture") not in ("native-residual-v1", "native-residual-v2", "native-residual-v3")
                or initial_kwargs.get("base") != args.base
                or initial_kwargs.get("prior_mode", "fixed") != args.prior_mode
                or initial_kwargs.get("boundary_radius", 2) != args.boundary_radius
                or initial_kwargs.get("boundary_logit", 2.) != args.boundary_logit
                or checkpoint["tile"] != metadata["tile"]):
            raise ValueError("Warmstart must preserve architecture, native scale and prior formula")
        initial_conditioning = initial_kwargs.get('conditioning','none')
        if initial_conditioning == args.conditioning:
            model.load_state_dict(checkpoint['state_dict'])
        elif initial_conditioning == 'none' and args.conditioning == 'instance':
            changes = model.load_state_dict(checkpoint['state_dict'],strict=False)
            if set(changes.missing_keys) != {'class_embed.weight','prompt_projection.weight'} or changes.unexpected_keys:
                raise ValueError('Instance warmstart may add only the zero-initialized prompt parameters')
        else:
            raise ValueError('Warmstart cannot discard learned instance conditions')
        init_sha256 = hashlib.sha256(args.init.read_bytes()).hexdigest()
    # Optional modules consume constructor RNG even when subsequently zeroed.
    # Reset after construction so paired runs share sampling and augmentation.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("native residual", device, "train", len(train), "val", len(val),
          "classes", dict(Counter(r['class'] for r in train)), flush=True)
    best = -1.; log = []
    for epoch in range(args.epochs+1):
        started = time.perf_counter(); loss_sum = 0.
        if epoch:
            model.train()
            for batch_number, batch in enumerate(train_loader):
                image,target,valid = batch[:3]
                image, target, valid = (a.to(device, non_blocking=True) for a in (image,target,valid))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type=="cuda"):
                    logits = (model(image,batch[5].to(device),batch[6].to(device))
                              if args.conditioning == 'instance' else model(image))
                    bce = (F.binary_cross_entropy_with_logits(logits.float(), target,
                            reduction="none", pos_weight=torch.tensor(3.,device=device))*valid).sum()/valid.sum().clamp_min(1)
                    probability = torch.sigmoid(logits.float())*valid
                    inter = (probability*target).sum((1,2,3))
                    dice = 1-(2*inter+1)/((probability+target*valid).sum((1,2,3))+1)
                    loss = bce + .5*dice.mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); loss_sum += float(loss.detach())
                if (batch_number+1) % 100 == 0:
                    print(f"epoch {epoch} batch {batch_number+1}/{len(train_loader)} "
                          f"loss {loss_sum/(batch_number+1):.4f}",flush=True)
            scheduler.step()
        model.eval(); total = np.zeros(4,dtype=float); coarse_total = np.zeros(4,dtype=float)
        class_stats = {}; validation_index = 0
        with torch.no_grad():
            for batch in val_loader:
                image,target,valid,thresholds,scales = batch[:5]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type=="cuda"):
                    logits = (model(image.to(device),batch[5].to(device),batch[6].to(device))
                              if args.conditioning == 'instance' else model(image.to(device)))
                    probability = torch.sigmoid(logits.float()).cpu().numpy()[:,0]
                truth, validity = target.numpy()[:,0]>.5, valid.numpy()[:,0]>.5
                for p,t,v,c,threshold,scale in zip(probability,truth,validity,image[:,3].numpy(),thresholds,scales):
                    score = vector_score(p,t,v,float(threshold),scale)
                    coarse_score = vector_score(c,t,v,float(threshold),scale)
                    total += score
                    coarse_total += coarse_score
                    class_id = int(val[validation_index]["class"]); validation_index += 1
                    entry = class_stats.setdefault(class_id,{"n":0,"detail":np.zeros(4),"coarse":np.zeros(4)})
                    entry["n"] += 1; entry["detail"] += score; entry["coarse"] += coarse_score
        def pq(t): return float(t[3]/max(t[0]+.5*(t[1]+t[2]),1))
        per_class = {str(c):{"n":s["n"],"detail_proxy_PQ":pq(s["detail"]),
                             "coarse_proxy_PQ":pq(s["coarse"]),
                             "detail_TP":int(s["detail"][0]),"coarse_TP":int(s["coarse"][0])}
                     for c,s in sorted(class_stats.items())}
        eligible = [c for c,s in class_stats.items() if s["n"]>=15
                    and pq(s["detail"])>pq(s["coarse"])+.01 and s["detail"][0]>=s["coarse"][0]]
        row = {"epoch":epoch,"loss":loss_sum/max(len(train_loader),1),
               "crop_proxy_PQ":pq(total),"coarse_crop_proxy_PQ":pq(coarse_total),
               "TP":int(total[0]),"FP":int(total[1]),"FN":int(total[2]),
               "seconds":round(time.perf_counter()-started,2),
               "validation_supported_classes":sorted(eligible),"per_class":per_class}
        print(json.dumps({k:v for k,v in row.items() if k!='per_class'}),flush=True); log.append(row)
        if (epoch or args.init) and row["crop_proxy_PQ"] > best:
            best = row["crop_proxy_PQ"]
            architecture = ('native-residual-v3' if args.conditioning == 'instance' else
                            'native-residual-v2' if args.prior_mode != 'fixed' else 'native-residual-v1')
            torch.save({"architecture":architecture,
                        "state_dict":model.state_dict(),
                        "model_kwargs":model_kwargs,"tile":metadata["tile"],
                        "classes":sorted({r["class"] for r in train}),"epoch":epoch,
                        "validation_supported_classes":sorted(eligible),"per_class_validation":per_class,
                        "crop_proxy_PQ":best,"data_metadata":metadata,
                        "training_config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                        "warmstart_sha256":init_sha256,
                        "instance_context_sha256":hashlib.sha256(args.instance_context.read_bytes()).hexdigest() if args.instance_context else None,
                        "training_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "index_sha256":hashlib.sha256((args.data/"index.json").read_bytes()).hexdigest()}, args.out)
        args.out.with_suffix(".training.json").write_text(json.dumps(log,indent=1),encoding="utf-8")
    print("completed; experimental crop proxy only, requires held-out page PQ",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
