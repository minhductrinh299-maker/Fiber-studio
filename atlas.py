"""Lazy production atlas: source snapshots + GPU tile/page composition.

Normal viewport publication never downloads maps. Pillow is used only by an
explicit CPU consumer/export adapter after a GPU readback on its owner worker.
"""
from collections.abc import Mapping
from copy import deepcopy
from threading import RLock
from weakref import WeakValueDictionary
import ctypes as C
import json
import numpy as np

_registry=WeakValueDictionary();_lock=RLock()

def register(descriptor):
 with _lock:
  for channel,version in descriptor.versions.items():_registry[(descriptor.tile.id,channel,version)]=descriptor

def source_for(tile_id,channel,version):
 with _lock:return _registry.get((tile_id,channel,version))

class TileDescriptor:
 def __init__(self,tile,keys,size,channels):
  from ..engine_v2.dependencies import channel_key
  self.tile=deepcopy(tile);self.keys=dict(keys);self.size=tuple(size);self.channels=tuple(channels)
  self.versions={c:channel_key(c,keys,size) for c in channels};register(self)
 def ensure(self,gpu,channels,cancel=lambda:False,pin=False):
  from ..engine_v2.engine import ENGINE
  from .service import prepared_truth,geometry_metadata
  pooled=[gpu.raw_versions.get((self.tile.id,self.size,c,self.versions[c])) for c in channels]
  alive={id(texture) for texture in gpu.pool}
  if all(record is not None and id(record) in alive and record.version==self.versions[c] for c,record in zip(channels,pooled)):
   from .gpu import ResidentResult
   if pin:
    for record in pooled:record.pins+=1
   return ResidentResult(self.tile.id,self.tile.revision,self.size,{c:record.name for c,record in zip(channels,pooled)},{c:self.versions[c] for c in channels},{},tuple(pooled) if pin else ())
  existing=gpu.entries.get((self.tile.id,self.size))
  if existing is not None and all(c in existing.textures and c in existing.versions and __import__('fiberstrand.engine_v2.dependencies',fromlist=['channel_key']).channel_key(c,existing.versions[c],self.size)==self.versions[c] for c in channels):
   from .gpu import ResidentResult
   records=tuple(existing.textures[c] for c in channels)
   if pin:
    for record in records:record.pins+=1
   return ResidentResult(self.tile.id,self.tile.revision,self.size,{c:existing.textures[c].name for c in channels},{c:self.versions[c] for c in channels},{},records if pin else ())
  truth,keys=ENGINE.truth(self.tile,cancel=cancel)
  prepared=prepared_truth(truth,keys);data=geometry_metadata(prepared,keys,self.size,cancel)
  return gpu.render(self.tile,prepared,keys,data,self.size,channels,cancel,pin=pin)
 def image(self,channel):
  from PIL import Image
  from .service import device
  d=device(False)
  def get(gpu):return gpu.readback(self.ensure(gpu,(channel,)))[channel]
  data=d.call(get).result()
  if channel=='color':return Image.fromarray(data,'RGBA')
  if channel in ('normal','flow','rgb_mask','id','cluster_id','material_id','material_mask'):return Image.fromarray(np.ascontiguousarray(data[:,:,:3]),'RGB')
  return Image.fromarray(np.ascontiguousarray(data[:,:,0]),'L')

class LazyMaps(Mapping):
 def __init__(self,descriptor):self.descriptor=descriptor
 def __iter__(self):return iter(self.descriptor.channels)
 def __len__(self):return len(self.descriptor.channels)
 def __getitem__(self,key):
  if key not in self.descriptor.channels:raise KeyError(key)
  return self.descriptor.image(key)


def prepare_production(tile,channels,cancel=lambda:False,progress=None):
 from ..engine_v2.engine import ENGINE
 from ..raster import RenderResult
 from .service import render_resident
 truth,keys=ENGINE.truth(tile,cancel=cancel);size=(tile.width,tile.height)
 # Only a bounded pass is required to stay resident at once. The descriptor can
 # reconstruct an evicted channel from its exact immutable source when needed.
 metrics={}
 for start in range(0,len(channels),2):
  if cancel():raise RuntimeError('Render cancelled')
  _,result=render_resident(tile,truth,keys,None,size,channels[start:start+2],cancel)
  metrics=result.metrics
  if progress:progress(min(1.,(start+2)/max(1,len(channels))))
 descriptor=TileDescriptor(tile,keys,size,channels)
 return RenderResult(tile.id,tile.revision,*size,LazyMaps(descriptor),[],dict(engine='FiberBake X',backend='fiberbake-x-resident',gpu_descriptor=descriptor,source_versions=keys,channel_versions=descriptor.versions,dirty_regions={c:[(0,0,*size)] for c in channels},fiberbake_x=metrics))

COMPOSE=r'''#version 430 core
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba8) writeonly uniform image2D uPage;
uniform sampler2D uTile;uniform ivec2 uOrigin;uniform ivec2 uExtent;uniform ivec2 uPlacement;
uniform ivec2 uSourceSize;uniform ivec2 uSize;uniform int uGutter;uniform float uRotation;uniform int uClear;uniform int uColor;uniform vec4 uBackground;
vec4 sampleTile(vec2 uv){if(uColor==0)return texture(uTile,uv);ivec2 size=textureSize(uTile,0);vec2 q=uv*vec2(size)-.5;ivec2 p=ivec2(floor(q));vec2 f=fract(q);vec4 sum=vec4(0);for(int y=0;y<2;++y)for(int x=0;x<2;++x){vec4 v=texelFetch(uTile,clamp(p+ivec2(x,y),ivec2(0),size-1),0);v.rgb*=v.a;float w=(x==0?1.-f.x:f.x)*(y==0?1.-f.y:f.y);sum+=v*w;}if(sum.a>1e-8)sum.rgb/=sum.a;return sum;}
void main(){ivec2 local=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(local,uExtent)))return;ivec2 target=uOrigin+local;if(any(lessThan(target,ivec2(0)))||any(greaterThanEqual(target,uSize)))return;
if(uClear!=0){imageStore(uPage,target,uBackground);return;}
vec2 uv=(clamp(vec2(local-uGutter)+.5,vec2(.5),vec2(uPlacement)-.5))/vec2(uPlacement);
float c=cos(uRotation),s=sin(uRotation);vec2 v=uv-.5;
float quarter=abs(mod(uRotation,1.57079632679));vec2 coord=(quarter<1e-5?mat2(c,s,-s,c)*v:mat2(c,s,-s,c)*(v*vec2(uSourceSize))/vec2(uSourceSize))+.5;
vec4 color=any(lessThan(coord,vec2(0)))||any(greaterThan(coord,vec2(1)))?vec4(0):sampleTile(coord);imageStore(uPage,target,color);
}'''

def compose_page(cache,channel,page):
 from PIL import Image
 from .service import device
 import math
 d=device(False);size=cache.layout.size;cancel=getattr(cache,'_cancel',lambda:False)
 def work(gpu):
  g=gpu.g;gpu.assert_owner()
  if not hasattr(gpu,'compose_program'):
   gpu.compose_program=gpu.compile(COMPOSE)
   gpu.compose_image=getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)(None,C.c_uint,C.c_uint,C.c_int,C.c_ubyte,C.c_int,C.c_uint,C.c_uint)(gpu.context.get_proc('glBindImageTexture'))
  program=gpu.compose_program;set2=getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)(None,C.c_int,C.c_int,C.c_int)(gpu.context.get_proc('glUniform2i'))
  # The output has an independent lease so evicting a source tile cannot remove it.
  target=gpu._texture((size,size),None);target.version="";target.pins+=1
  try:
   def uniforms(origin,extent,placement=(1,1),source=(1,1),gutter=0,rotation=0,clear=False):
    g.glUseProgram(program);loc=lambda n:g.glGetUniformLocation(program,n.encode())
    for name,value in [('uOrigin',origin),('uExtent',extent),('uPlacement',placement),('uSourceSize',source),('uSize',(size,size))]:set2(loc(name),*value)
    for name,value in [('uGutter',gutter),('uClear',int(clear)),('uTile',0),('uColor',int(channel=='color'))]:g.glUniform1i(loc(name),value)
    g.glUniform1f(loc('uRotation'),math.radians(rotation));bg=1. if channel=='ao' else 0.;g.glUniform4f(loc('uBackground'),bg,bg,bg,0. if channel=='color' else 1.)
    gpu.compose_image(0,target.name,0,False,0,0x88B9,0x8058);gpu.f['glDispatchCompute']((extent[0]+7)//8,(extent[1]+7)//8,1);gpu.f['glMemoryBarrier'](0x20|0x8)
   uniforms((0,0),(size,size),clear=True)
   base=getattr(cache,'_cpu_pages',{}).get((channel,page))
   if base is not None:
    payload=np.ascontiguousarray(base.convert('RGBA'));g.glBindTexture(0x0DE1,target.name);g.glTexSubImage2D(0x0DE1,0,0,0,size,size,0x1908,0x1401,payload.ctypes.data)
   for tid,placement in cache.layout.placements.items():
    descriptor=getattr(cache,'_gpu_tiles',{}).get(tid)
    if placement.page!=page or descriptor is None or channel not in descriptor.channels:continue
    if cancel():raise RuntimeError('Export cancelled')
    result=descriptor.ensure(gpu,(channel,),cancel,pin=True)
    try:
     g.glActiveTexture(0x84C0);g.glBindTexture(0x0DE1,result.textures[channel]);pad=placement.gutter
     uniforms((placement.x-pad,placement.y-pad),(placement.width+pad*2,placement.height+pad*2),(placement.width,placement.height),descriptor.size,pad,placement.rotation)
     # Source sampling and any later reuse are ordered in this same context.
     # Retire the page once below, rather than synchronizing after every tile.
     gpu.f['glMemoryBarrier'](0x20|0x8|0x100)
    finally:gpu.release(result)
   gpu.wait(cancel);raw=np.empty((size,size,4),np.uint8);g.glBindTexture(0x0DE1,target.name);g.glGetTexImage(0x0DE1,0,0x1908,0x1401,raw.ctypes.data);return raw
  finally:target.pins-=1
 data=d.call(work).result();image=Image.fromarray(data,'RGBA')
 return image if channel=='color' else image.convert('RGB' if channel in ('normal','flow','id','cluster_id','material_id','material_mask','rgb_mask') else 'L')


def stage(cache,result):
 from ..atlas import AtlasCache
 descriptor=result.metadata['gpu_descriptor'];target=object.__new__(AtlasCache)
 target.layout=cache.layout;target._pages=dict(cache._pages);target._cpu_pages=dict(getattr(cache,'_cpu_pages',cache._pages))
 target._gpu_tiles=dict(getattr(cache,'_gpu_tiles',{}));target._gpu_tiles[result.tile_id]=descriptor
 target._tile_version=dict(cache._tile_version);target._tile_version[result.tile_id]=(result.revision,result.width,result.height)
 target._channel_versions=dict(cache._channel_versions);target.changed_regions=[]
 placement=cache.layout.placements.get(result.tile_id)
 if placement:
  for channel,version in descriptor.versions.items():
   target._channel_versions[result.tile_id,channel]=version;target._pages.pop((channel,placement.page),None)
   target.changed_regions.append((channel,placement.page,(placement.x,placement.y,placement.x+placement.width,placement.y+placement.height)))
 return target


def compatibility_tile(descriptor,placement,channels):
 """Explicit worker-only CPU adapter for unshared/unsupported render contexts."""
 from PIL import Image
 result={}
 for channel in channels:
  image=descriptor.image(channel);angle=placement.rotation%360
  if abs(angle-90)<1e-5:image=image.transpose(Image.Transpose.ROTATE_90)
  elif abs(angle-180)<1e-5:image=image.transpose(Image.Transpose.ROTATE_180)
  elif abs(angle-270)<1e-5:image=image.transpose(Image.Transpose.ROTATE_270)
  elif abs(angle)>1e-5:image=image.rotate(angle,resample=Image.Resampling.BILINEAR,expand=False)
  if image.size!=(placement.width,placement.height):image=image.resize((placement.width,placement.height),Image.Resampling.NEAREST if channel in ('id','cluster_id','material_id') else Image.Resampling.BILINEAR)
  result[channel]=image
 return result
