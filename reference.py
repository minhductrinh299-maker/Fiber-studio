"""CPU oracle for final source sampling, separate from GPU shader execution."""
from copy import copy
import math
import numpy as np
from PIL import Image
from ..engine_v2.dependencies import digest,channel_key
from ..engine_v2.samples import sample_origin,BLOCK

def supersample(truth,bins,size,channels,keys,settings,sample_cache,block_cache,cancel,progress=None):
    from ..engine_v2.bake import bake_with_info,empty_map,IDENTITY
    count=max(1,min(16,int(settings.bake_samples)));side=math.ceil(math.sqrt(count));policy=copy(settings);policy.bake_samples=1
    width,height=size;maps={}
    for base in range(0,len(channels),4):
        group=channels[base:base+4];requested=tuple(dict.fromkeys((*group,'alpha')))
        sums={c:np.zeros((height,width,3),np.float32) for c in group if c not in IDENTITY and c!='alpha'}
        weights=np.zeros((height,width),np.float32);votes={c:[] for c in group if c in IDENTITY}
        for sample in range(count):
            if cancel():raise RuntimeError('Render cancelled')
            offset=((sample%side+.5)/side,(sample//side+.5)/side);sample_keys=dict(keys);sample_keys['sampling']=digest((keys['sampling'],offset))
            with sample_origin(offset):images,_,_=bake_with_info(truth,bins,size,requested,sample_keys,policy,sample_cache,block_cache,cancel,prefer_gpu=False)
            coverage=np.asarray(images['alpha'],np.float32)/255.;weights+=coverage
            for channel in group:
                raw=np.asarray(images[channel])
                if channel in IDENTITY:
                    value=(raw[...,0].astype(np.uint32)<<16)|(raw[...,1].astype(np.uint32)<<8)|raw[...,2].astype(np.uint32);votes[channel].append(value);continue
                if channel=='alpha':continue
                value=raw[...,:3].astype(np.float32)/255 if raw.ndim==3 else np.repeat((raw.astype(np.float32)/255)[...,None],3,axis=-1)
                if channel in ('normal','flow'):value=value*2-1
                sums[channel]+=value*coverage[...,None]
            if progress:progress((base+(sample+1)/count*len(group))/max(1,len(channels)))
        for channel in group:
            if channel in IDENTITY:
                chosen=np.zeros((height,width),np.uint32);best=np.zeros((height,width),np.int16)
                for value in votes[channel]:
                    score=sum((other==value).astype(np.int16) for other in votes[channel]);use=(value!=0)&(score>best);chosen[use]=value[use];best[use]=score[use]
                maps[channel]=Image.fromarray(np.stack(((chosen>>16)&255,(chosen>>8)&255,chosen&255),-1).astype(np.uint8));continue
            if channel=='alpha':maps[channel]=Image.fromarray(np.rint(weights/count*255).clip(0,255).astype(np.uint8));continue
            value=sums[channel]/np.maximum(weights[...,None],1e-8)
            if channel in ('normal','flow'):
                value/=np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-8)
                value[weights<=1e-6]=[0,0,1] if channel=='normal' else [0,0,0];value=value*.5+.5
            else:
                fallback=np.asarray(empty_map(channel,(1,1),settings)).reshape(-1).astype(np.float32)/255
                value[weights<=1e-6]=fallback[:3] if len(fallback)>=3 else fallback[0]
            data=np.rint(value*255).clip(0,255).astype(np.uint8)
            if channel=='color':data=np.dstack((data,np.rint(weights/count*255).clip(0,255).astype(np.uint8)))
            elif channel not in ('normal','flow','rgb_mask','material_mask'):data=data[...,0]
            maps[channel]=Image.fromarray(data)
    versions={c:{block:channel_key(c,keys,size) for block in bins[0]} for c in channels}
    return maps,versions,'cpu-source-supersampling'
