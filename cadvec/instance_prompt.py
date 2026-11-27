"""Target-box coordinates derived solely from the detector's pixel frame."""
from __future__ import annotations

import numpy as np


def coordinate_prompt(shape, target_box, tile_origin=(0,0), valid_shape=None):
    """Return normalized dx/dy from a box center, with zero padded regions."""
    height,width=map(int,shape)
    x0,y0,x1,y1=map(float,target_box)
    ox,oy=map(float,tile_origin)
    if min(height,width)<=0 or not np.isfinite([x0,y0,x1,y1,ox,oy]).all() or x1<x0 or y1<y0:
        raise ValueError('Invalid target-box coordinate frame')
    vh,vw=(height,width) if valid_shape is None else map(int,valid_shape)
    if not (0<vh<=height and 0<vw<=width):
        raise ValueError('Invalid valid tile shape')
    result=np.zeros((2,height,width),dtype=np.float32)
    # A GT-box training supplement can be a single horizontal/vertical line.
    # Give zero pixel extent a 1px denominator, without dropping that sample.
    dx=(np.arange(vw,dtype=np.float32)+.5+ox-(x0+x1)/2)/max(x1-x0,1.)
    dy=(np.arange(vh,dtype=np.float32)+.5+oy-(y0+y1)/2)/max(y1-y0,1.)
    result[0,:vh,:vw]=np.clip(dx,-2,2)[None,:]
    result[1,:vh,:vw]=np.clip(dy,-2,2)[:,None]
    return result


def augment_prompt(prompt, rotations, flip_horizontal=False):
    """Rotate positions and vector components together, then optionally flip."""
    value=np.rot90(prompt,int(rotations)%4,axes=(1,2)).copy()
    x,y=value[0].copy(),value[1].copy()
    k=int(rotations)%4
    if k==1:value[0],value[1]=y,-x
    elif k==2:value[0],value[1]=-x,-y
    elif k==3:value[0],value[1]=-y,x
    if flip_horizontal:
        value=np.flip(value,axis=2).copy();value[0]*=-1
    return value
