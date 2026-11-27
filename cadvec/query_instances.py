"""Direct PNG-to-instance set prediction with shared native-pixel features.

This is a new lightweight query architecture, not a reproduction of another
model. It does not consume detector boxes, coarse masks, or legacy quality gates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F


class _ConvGN(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel=3):
        groups = min(8,out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(nn.Conv2d(in_channels,out_channels,kernel,padding=kernel//2,bias=False),
                         nn.GroupNorm(groups,out_channels),nn.SiLU())


def _new_cad_backbone():
    # These are local installed modules, not YOLO(path/model-name) constructors.
    # No download, prediction engine, or original segmentation head is invoked.
    from ultralytics.nn.modules import C2f,Conv,SPPF
    modules = nn.ModuleList([
        Conv(3,16,3,2),Conv(16,32,3,2),C2f(32,32,1,True),
        Conv(32,64,3,2),C2f(64,64,2,True),
        Conv(64,128,3,2),C2f(128,128,2,True),
        Conv(128,256,3,2),C2f(256,256,1,True),SPPF(256,256,5),
    ])
    # YOLO's model initialization changes BN settings from module defaults;
    # those attributes are not stored in state_dict and must also be restored.
    for module in modules.modules():
        if isinstance(module,nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = .03
    return modules


def sine_position_2d(height,width,dim,device,dtype):
    if dim % 4:
        raise ValueError('Position dimension must be divisible by four')
    quarter = dim//4
    frequency = 10000. ** (torch.arange(quarter,device=device,dtype=torch.float32)/max(quarter,1))
    x = (torch.arange(width,device=device,dtype=torch.float32)+.5)/width*(2*torch.pi)
    y = (torch.arange(height,device=device,dtype=torch.float32)+.5)/height*(2*torch.pi)
    x,y = x[:,None]/frequency[None,:],y[:,None]/frequency[None,:]
    px,py = torch.cat((x.sin(),x.cos()),1),torch.cat((y.sin(),y.cos()),1)
    encoding = torch.cat((px[None,:,:].expand(height,-1,-1),py[:,None,:].expand(-1,width,-1)),2)
    return encoding.reshape(height*width,dim).to(dtype=dtype)


class QueryInstanceModel(nn.Module):
    """A CAD encoder, shared native mask features, and joint instance queries.

    ``backbone_path=None`` creates the same encoder without loading a checkpoint,
    for inference checkpoint restoration. Neither mode downloads any weights.
    Inputs are floating RGB in [0,1]; callers retain original size/padding data.
    """
    def __init__(self,backbone_path='weights/yolo_seg.pt',num_queries=100,dim=128,
                 mask_dim=32,decoder_layers=3,heads=4,num_classes=30):
        super().__init__()
        if (num_queries<1 or dim<4 or dim%4 or dim%heads or mask_dim<1
                or decoder_layers<1 or num_classes<1):
            raise ValueError('Invalid query model dimensions')
        self.num_queries,self.num_classes = int(num_queries),int(num_classes)
        self.dim,self.mask_dim = int(dim),int(mask_dim)
        self.backbone = _new_cad_backbone()
        self.backbone_initialization = {'kind':'random-local-architecture','checkpoint_sha256':None,
                                        'layers':list(range(10)),'feature_layers':[4,6,9],
                                        'feature_channels':[64,128,256],'routing':'all f=-1'}
        if backbone_path is not None:
            path = Path(backbone_path)
            if not path.is_file():
                raise FileNotFoundError(f'Local CAD backbone checkpoint required: {path}')
            payload = torch.load(path,map_location='cpu',weights_only=False)
            source = payload.get('ema')
            if source is None:
                source = payload.get('model')
            layers = getattr(source,'model',None)
            if layers is None or len(layers)<10:
                raise ValueError('Expected the local YOLO CAD checkpoint model')
            for index,(actual,expected) in enumerate(zip(layers[:10],self.backbone)):
                if getattr(actual,'f',None) != -1 or type(actual).__name__ != type(expected).__name__:
                    raise ValueError(f'Unsupported CAD encoder routing/type at layer {index}')
                expected.load_state_dict(actual.float().state_dict(),strict=True)
            self.backbone_initialization.update(kind='local-trained-CAD-YOLO-layers-0-9',
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path),
                verified_sequential_routing=True,verified_state_shapes=True)
            del payload,source,layers
        # Stripped inference checkpoints often have requires_grad=False.
        self.backbone.requires_grad_(True)
        self.projections = nn.ModuleList([_ConvGN(ch,dim,1) for ch in (64,128,256)])
        self.fpn_smooth = _ConvGN(dim,dim)
        self.native_rgb = nn.Sequential(_ConvGN(3,16),_ConvGN(16,mask_dim))
        self.fpn_mask = _ConvGN(dim,mask_dim,1)
        self.mask_features = _ConvGN(mask_dim+2,mask_dim)
        layer = nn.TransformerDecoderLayer(d_model=dim,nhead=heads,dim_feedforward=dim*4,
                                           dropout=0.,batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,num_layers=decoder_layers,norm=nn.LayerNorm(dim))
        self.queries = nn.Embedding(num_queries,dim)
        nn.init.normal_(self.queries.weight,std=.02)
        self.class_head = nn.Linear(dim,num_classes+1)
        self.mask_head = nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,mask_dim))
        self.box_head = nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim),
                                      nn.SiLU(),nn.Linear(dim,4))

    def forward(self,image):
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError('Expected floating RGB Bx3xHxW in [0,1]')
        features = []; value = image
        for index,layer in enumerate(self.backbone):
            value = layer(value)
            if index in (4,6,9):
                features.append(value)
        p8,p16,p32 = [projection(value) for projection,value in zip(self.projections,features)]
        p16 = p16+F.interpolate(p32,size=p16.shape[-2:],mode='bilinear',align_corners=False)
        shared = self.fpn_smooth(p8+F.interpolate(p16,size=p8.shape[-2:],mode='bilinear',align_corners=False))
        memory = F.adaptive_avg_pool2d(shared,(min(32,shared.shape[-2]),min(32,shared.shape[-1])))
        h,w = memory.shape[-2:]
        memory = memory.flatten(2).transpose(1,2)
        memory = memory+sine_position_2d(h,w,self.dim,memory.device,memory.dtype)[None]
        query = self.queries.weight[None].expand(image.shape[0],-1,-1)
        decoded = self.decoder(query,memory)
        # Project BEFORE native upsampling: there is no full-resolution 128ch
        # tensor, and no QxHxW prediction tensor in this forward pass.
        native = self.native_rgb(image)
        low_mask = self.fpn_mask(shared)
        fused = native+F.interpolate(low_mask,size=image.shape[-2:],mode='bilinear',align_corners=False)
        # Shared convolutional features alone cannot distinguish translated
        # copies of identical symbols. Fuse native XY nonlinearly with RGB/FPN,
        # rather than adding positions only to the query-attention memory.
        height,width = image.shape[-2:]
        x = ((torch.arange(width,device=image.device,dtype=torch.float32)+.5)/width*2-1).to(fused.dtype)
        y = ((torch.arange(height,device=image.device,dtype=torch.float32)+.5)/height*2-1).to(fused.dtype)
        xy = torch.stack((x[None,:].expand(height,-1),y[:,None].expand(-1,width)))[None]
        pixels = self.mask_features(torch.cat((fused,xy.expand(image.shape[0],-1,-1,-1)),1))
        return {'class_logits':self.class_head(decoded),'mask_embed':self.mask_head(decoded),
                'pixel_features':pixels,'boxes':self.box_head(decoded).sigmoid()}


def _indices(values,batch,length,device,name):
    indices = torch.as_tensor(values,device=device)
    if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise ValueError(f'{name} must contain integer indices')
    indices = indices.long()
    if indices.ndim == 1:
        indices = indices[None].expand(batch,-1)
    if indices.ndim != 2 or indices.shape[0] != batch:
        raise ValueError(f'{name} must have shape P or BxP')
    if indices.numel() and (indices.min()<0 or indices.max()>=length):
        raise ValueError(f'{name} outside available range')
    return indices


def sample_mask_logits(output,flat_indices):
    """Evaluate only selected flat pixels, returning BxQxP float32 logits."""
    pixels,embeddings = output['pixel_features'],output['mask_embed']
    if pixels.ndim != 4 or embeddings.ndim != 3 or pixels.shape[:2] != (embeddings.shape[0],embeddings.shape[2]):
        raise ValueError('Incompatible mask embeddings and pixel features')
    batch,channels,height,width = pixels.shape
    indices = _indices(flat_indices,batch,height*width,pixels.device,'Pixel indices')
    selected = pixels.flatten(2).gather(2,indices[:,None].expand(-1,channels,-1))
    with torch.autocast(device_type=pixels.device.type,enabled=False):
        return torch.einsum('bqc,bcp->bqp',embeddings.float(),selected.float())


def dense_mask_logits(output,query_indices):
    """Inference-only selected-query masks, BxKxHxW; no thresholding/GT input."""
    pixels,embeddings = output['pixel_features'],output['mask_embed']
    batch,queries,channels = embeddings.shape
    indices = _indices(query_indices,batch,queries,embeddings.device,'Query indices')
    selected = embeddings.gather(1,indices[:,:,None].expand(-1,-1,channels))
    with torch.autocast(device_type=pixels.device.type,enabled=False):
        return torch.einsum('bkc,bchw->bkhw',selected.float(),pixels.float())


def box_cxcywh_to_xyxy(boxes):
    center,extent = boxes[...,:2],boxes[...,2:]
    return torch.cat((center-extent/2,center+extent/2),-1)


def generalized_box_iou(first,second):
    """Pairwise GIoU, including finite behavior for zero-extent line boxes."""
    first,second = box_cxcywh_to_xyxy(first),box_cxcywh_to_xyxy(second)
    first_area = (first[:,2:]-first[:,:2]).clamp_min(0).prod(-1)
    second_area = (second[:,2:]-second[:,:2]).clamp_min(0).prod(-1)
    left = torch.maximum(first[:,None,:2],second[None,:,:2])
    right = torch.minimum(first[:,None,2:],second[None,:,2:])
    intersection = (right-left).clamp_min(0).prod(-1)
    union = first_area[:,None]+second_area[None,:]-intersection
    iou = intersection/union.clamp_min(1e-7)
    enclosing_left = torch.minimum(first[:,None,:2],second[None,:,:2])
    enclosing_right = torch.maximum(first[:,None,2:],second[None,:,2:])
    enclosing_area = (enclosing_right-enclosing_left).clamp_min(0).prod(-1)
    return iou-(enclosing_area-union)/enclosing_area.clamp_min(1e-7)


class QueryInstanceLoss(nn.Module):
    """Hungarian set loss on sampled pixels, with independent sigmoid masks.

    Cost weights: class 2 / mask BCE 5 / Dice 5 / bbox L1 5 / GIoU 2.
    Training weights: CE 1 (null class weight .1), then BCE 5 / Dice 5 /
    bbox L1 5 / GIoU 2. Targets may keep full masks on CPU; only selected
    NxP pixels are transferred to the prediction device.
    """
    def __init__(self,num_classes=30,null_weight=.1):
        super().__init__()
        if num_classes<1 or null_weight<=0:
            raise ValueError('Invalid class/null weighting')
        self.num_classes = int(num_classes)
        weights = torch.ones(num_classes+1); weights[-1] = float(null_weight)
        self.register_buffer('class_weights',weights)

    def forward(self,output,targets,sample_indices):
        classes = output['class_logits'].float()
        boxes = output['boxes'].float()
        batch,queries,class_count = classes.shape
        if class_count != self.num_classes+1 or boxes.shape != (batch,queries,4) or len(targets) != batch:
            raise ValueError('Incompatible query outputs or target batch')
        height,width = output['pixel_features'].shape[-2:]
        indices = _indices(sample_indices,batch,height*width,classes.device,'Pixel indices')
        if indices.shape[1] == 0:
            raise ValueError('Set loss requires at least one sampled pixel')
        mask_logits = sample_mask_logits(output,indices)
        target_classes = torch.full((batch,queries),self.num_classes,device=classes.device,dtype=torch.long)
        assignment = []; matched_predictions = []; matched_targets = []
        matched_boxes = []; target_boxes = []
        for b,target in enumerate(targets):
            labels = torch.as_tensor(target['labels'],device=classes.device,dtype=torch.long)
            actual_boxes = torch.as_tensor(target['boxes'],device=classes.device,dtype=torch.float32)
            masks = torch.as_tensor(target['masks'])
            n = len(labels)
            if n>queries:
                raise ValueError('Target instance count exceeds the query budget')
            if labels.shape != (n,) or actual_boxes.shape != (n,4) or masks.shape != (n,height,width):
                raise ValueError('Targets need labels N, boxes Nx4, masks NxHxW in padded frame')
            if (torch.any(labels<0) or torch.any(labels>=self.num_classes)
                    or not torch.isfinite(actual_boxes).all() or torch.any(actual_boxes[:,2:]<0)):
                raise ValueError('Invalid target classes/boxes')
            if not n:
                empty = torch.empty(0,device=classes.device,dtype=torch.long)
                assignment.append((empty,empty))
                continue
            actual_masks = masks.flatten(1).index_select(1,indices[b].to(masks.device)).to(
                device=classes.device,dtype=torch.float32)
            if not torch.isfinite(actual_masks).all() or torch.any(actual_masks<0) or torch.any(actual_masks>1):
                raise ValueError('Target masks must be binary or probabilities in [0,1]')
            predicted = mask_logits[b]
            with torch.no_grad(),torch.autocast(device_type=classes.device.type,enabled=False):
                probability = predicted.sigmoid()
                points = predicted.shape[1]
                # Matrix identities avoid allocating QxNxP pairwise masks.
                bce_cost = F.softplus(predicted).mean(1)[:,None] - predicted@actual_masks.T/points
                dice_cost = 1-(2*(probability@actual_masks.T)+1)/(
                    probability.sum(1)[:,None]+actual_masks.sum(1)[None,:]+1)
                class_cost = -classes[b].softmax(-1)[:,labels]
                bbox_cost = torch.cdist(boxes[b],actual_boxes,p=1)
                giou_cost = -generalized_box_iou(boxes[b],actual_boxes)
                cost = 2*class_cost+5*bce_cost+5*dice_cost+5*bbox_cost+2*giou_cost
                if not torch.isfinite(cost).all():
                    raise ValueError('Nonfinite Hungarian matching cost')
                query_ids,target_ids = linear_sum_assignment(cost.detach().cpu().numpy())
            query_ids = torch.as_tensor(query_ids,device=classes.device,dtype=torch.long)
            target_ids = torch.as_tensor(target_ids,device=classes.device,dtype=torch.long)
            assignment.append((query_ids,target_ids))
            target_classes[b,query_ids] = labels[target_ids]
            matched_predictions.append(predicted[query_ids]); matched_targets.append(actual_masks[target_ids])
            matched_boxes.append(boxes[b,query_ids]); target_boxes.append(actual_boxes[target_ids])
        classification = F.cross_entropy(classes.transpose(1,2),target_classes,
                                          weight=self.class_weights.to(classes.device))
        zero = mask_logits.sum()*0+boxes.sum()*0
        if matched_predictions:
            predicted = torch.cat(matched_predictions); actual = torch.cat(matched_targets)
            predicted_boxes,actual_boxes = torch.cat(matched_boxes),torch.cat(target_boxes)
            count = len(predicted)
            bce = F.binary_cross_entropy_with_logits(predicted,actual,reduction='none').mean(1).mean()
            probability = predicted.sigmoid()
            dice = (1-(2*(probability*actual).sum(1)+1)/(probability.sum(1)+actual.sum(1)+1)).mean()
            l1 = F.l1_loss(predicted_boxes,actual_boxes,reduction='sum')/count
            giou = (1-generalized_box_iou(predicted_boxes,actual_boxes).diag()).mean()
        else:
            bce,dice,l1,giou = zero,zero,zero,zero
        total = classification+5*bce+5*dice+5*l1+2*giou
        return {'total':total,'class':classification,'mask_bce':bce,'mask_dice':dice,
                'bbox_l1':l1,'giou':giou,'assignment':assignment}
