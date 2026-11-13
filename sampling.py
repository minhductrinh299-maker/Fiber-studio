"""GPU tile-isolated linear premultiplied mips for RenderCore.

Raw weighted source values are reduced before normalization. This avoids alpha
halos and preserves the same normal/flow/roughness rules as CPU tile_mips.
"""
import ctypes as C
import math
import numpy as np
from .gpu import Texture,ResidentResult

SOURCE=r'''#version 430 core
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) writeonly uniform image2D uRawOut;
layout(binding=1,rgba16f) writeonly uniform image2D uEncoded;
uniform sampler2D uSource;uniform sampler2D uAlpha;uniform sampler2D uPrevious;
uniform int uLevel;uniform int uChannel;uniform ivec2 uSize;uniform ivec2 uPreviousSize;
uniform float uRotation;uniform ivec2 uOrigin;uniform ivec2 uDispatchSize;uniform ivec2 uSourceSize;uniform int uQuarterTurn;
vec3 linearize(vec3 c){return mix(c/12.92,pow((c+.055)/1.055,vec3(2.4)),step(vec3(.04045),c));}
void main(){
 ivec2 local=ivec2(gl_GlobalInvocationID.xy);ivec2 p=local+uOrigin;if(any(greaterThanEqual(local,uDispatchSize))||any(greaterThanEqual(p,uSize)))return;
 vec4 weighted=vec4(0);
 if(uLevel==0){
  vec2 uv=(vec2(p)+.5)/vec2(uSize);float c=cos(uRotation),s=sin(uRotation);
  // Counter-clockwise image rotation, then resize to placement rectangle.
  vec2 delta=uv-.5;vec2 source=(uQuarterTurn!=0?mat2(c,s,-s,c)*delta:(mat2(c,s,-s,c)*(delta*vec2(uSourceSize)))/vec2(uSourceSize))+.5;
  vec4 value=texture(uSource,source);float a=texture(uAlpha,source).r;
  if(any(lessThan(source,vec2(0)))||any(greaterThan(source,vec2(1)))){value=vec4(0);a=0.;}
  if(uChannel==0)weighted=vec4(linearize(value.rgb)*a,a);
  else if(uChannel==1)weighted=vec4(a,0,0,a);
  else if(uChannel==2||uChannel==3){vec3 v=value.rgb*2.-1.;if(uChannel==3)v.z=0.;weighted=vec4(v*a,a);}
  else weighted=vec4(value.r*value.r*a,0,0,a);
 }else{
  // Area weighted reduction also covers odd/NPOT boundary texels.
  vec2 low=vec2(p)*vec2(uPreviousSize)/vec2(uSize),high=vec2(p+1)*vec2(uPreviousSize)/vec2(uSize);
  float area=0.;
  for(int y=int(floor(low.y));y<int(ceil(high.y));++y)for(int x=int(floor(low.x));x<int(ceil(high.x));++x){
   float w=max(0.,min(float(x+1),high.x)-max(float(x),low.x))*max(0.,min(float(y+1),high.y)-max(float(y),low.y));
   weighted+=texelFetch(uPrevious,ivec2(x,y),uLevel-1)*w;area+=w;
  }
  weighted/=max(area,1e-8);
 }
 imageStore(uRawOut,p,weighted);
 vec4 encoded=weighted;
 if(uChannel==1)encoded=vec4(weighted.x);
 else if(uChannel==2||uChannel==3){vec3 v=weighted.xyz;float len=length(v);v=len>1e-8?v/len:(uChannel==2?vec3(0,0,1):vec3(0,1,0));encoded=vec4(v*.5+.5,1);}
 else if(uChannel==4){float r=weighted.a>1e-8?sqrt(max(0.,weighted.r/weighted.a)):.5;encoded=vec4(r,r,r,1);}
 imageStore(uEncoded,ivec2(p.x,uSize.y-1-p.y),encoded);
}'''

class SamplingPacket:
 def __init__(self,device,result,levels):self.device=device;self.result=result;self.levels=levels;self.closed=False
 def close(self,fence=None):
  if self.closed:return
  self.closed=True;self.device.release(self.result,fence)
 def __del__(self):
  try:self.close()
  except Exception:pass


def _alive(gpu,texture):
 return texture is not None and any(record is texture for record in gpu.pool)


def _dirty_rects(entry,channel,previous,expected,source_size,size,rotation):
 """Conservative source -> rotated placement footprint, expanded for filtering."""
 if previous is None:return None
 required=(channel,'alpha') if channel!='alpha' else ('alpha',)
 rectangles=[]
 for name in required:
  old=previous['versions'][name];new=expected[name]
  if old==new:continue
  delta=entry.channel_deltas.get(name)
  if delta is None or delta[0]!=old or delta[1]!=new:return None
  rectangles.extend(delta[2])
 if not rectangles:return []
 angle=math.radians(-rotation);c=math.cos(angle);sn=math.sin(angle);sw,sh=source_size;w,h=size
 points=[]
 for x0,y0,x1,y1 in rectangles:
  for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1)):
   a=x/sw-.5;b=y/sh-.5
   if abs(rotation%90)<1e-5:px=c*a-sn*b;py=sn*a+c*b
   else:px=(c*(a*sw)-sn*(b*sh))/sw;py=(sn*(a*sw)+c*(b*sh))/sh
   points.append(((px+.5)*w,(py+.5)*h))
 x0=max(0,math.floor(min(q[0] for q in points))-2);y0=max(0,math.floor(min(q[1] for q in points))-2)
 x1=min(w,math.ceil(max(q[0] for q in points))+2);y1=min(h,math.ceil(max(q[1] for q in points))+2)
 if x0>=x1 or y0>=y1:return []
 return None if (x1-x0)*(y1-y0)>w*h*.55 else [(x0,y0,x1,y1)]


def create_sampling(gpu,tile_id,expected,size,rotation=0):
 gpu.assert_owner();g=gpu.g;channels=('color','alpha','normal','flow','roughness')
 from ..engine_v2.dependencies import channel_key,dirty_mips
 entry=None;source_size=None
 for (tid,dimensions),candidate in reversed(gpu.entries.items()):
  if tid==tile_id and all(c in candidate.textures and c in candidate.versions and expected.get(c)==channel_key(c,candidate.versions[c],dimensions) for c in channels):entry=candidate;source_size=dimensions;break
 if entry is None:
  from .atlas import source_for
  descriptor=source_for(tile_id,'color',expected.get('color'))
  if descriptor is None:return None
  descriptor.ensure(gpu,channels)
  entry=gpu.entries.get((tile_id,descriptor.size));source_size=descriptor.size
  if entry is None or not all(c in entry.textures and c in entry.versions and channel_key(c,entry.versions[c],source_size)==expected[c] for c in channels):
   from types import SimpleNamespace
   textures={c:gpu.raw_versions.get((tile_id,source_size,c,expected[c])) for c in channels}
   alive={id(texture) for texture in gpu.pool}
   if not all(record is not None and id(record) in alive and record.version==expected[c] for c,record in textures.items()):return None
   entry=SimpleNamespace(textures=textures,channel_deltas={})
 if not hasattr(gpu,'sampling_program'):
  gpu.sampling_program=gpu.compile(SOURCE)
  prototype=getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)
  gpu.bind_image=prototype(None,C.c_uint,C.c_uint,C.c_int,C.c_ubyte,C.c_int,C.c_uint,C.c_uint)(gpu.context.get_proc('glBindImageTexture'))
 width,height=map(int,size);levels=int(math.floor(math.log2(max(width,height))))+1
 amount=sum(max(1,width>>i)*max(1,height>>i)*8 for i in range(levels))
 keep=(tile_id,source_size);gpu.active_channels=set(channels);records=[];names={};updates={};base_stamps={};held=[]
 cache=getattr(gpu,'sampling_cache',None)
 if cache is None:gpu.sampling_cache=cache={}
 pending_cache={}
 stamps={c:(tile_id,tuple(size),float(rotation),c,expected[c],expected['alpha']) for c in channels}
 source_records=[entry.textures[c] for c in channels]
 for record in source_records:record.pins+=1
 def allocate(kind):
  gpu._reserve(amount,keep);name=g.name('Textures');g.glBindTexture(0x0DE1,name);g.glBindBuffer(0x88EC,0)
  for parameter,value in ((0x2801,0x2703),(0x2800,0x2601),(0x2802,0x812F),(0x2803,0x812F),(0x813D,levels-1)):g.glTexParameteri(0x0DE1,parameter,value)
  for level in range(levels):g.glTexImage2D(0x0DE1,level,0x881A,max(1,width>>level),max(1,height>>level),0,0x1908,0x1406,None)
  record=Texture(name,(width,height),amount,pins=1,kind=kind);gpu.pool.append(record);gpu.used+=amount;held.append(record);return record
 try:
  for ci,channel in enumerate(channels):
   identity=(tile_id,tuple(size),float(rotation),channel);previous=cache.get(identity)
   if previous is not None and (not _alive(gpu,previous['encoded']) or not _alive(gpu,previous['weighted'])):previous=None
   if previous is not None:
    previous['encoded'].pins+=1;previous['weighted'].pins+=1;held.extend((previous['encoded'],previous['weighted']))
   if previous is not None and previous['stamp']==stamps[channel]:
    record=previous['encoded'];record.pins+=1;records.append(record);names[channel]=record.name;updates[channel]=[];base_stamps[channel]=stamps[channel];continue
   dirty=_dirty_rects(entry,channel,previous,expected,source_size,size,rotation)
   weighted=allocate('weighted-mips');encoded=allocate('sampling')
   if previous is not None and dirty is not None:
    for level in range(levels):
     w=max(1,width>>level);h=max(1,height>>level)
     for old,new in ((previous['weighted'],weighted),(previous['encoded'],encoded)):
      gpu.f['glCopyImageSubData'](old.name,0x0DE1,level,0,0,0,new.name,0x0DE1,level,0,0,0,w,h,1)
    base_stamps[channel]=previous['stamp']
   else:base_stamps[channel]=None
   rectangles=[(0,0,width,height)] if dirty is None else dirty
   regions=[[] for _ in range(levels)]
   for rect in rectangles:
    for level,expanded in enumerate(dirty_mips(rect,(width,height))):regions[level].append(expanded)
   g.glUseProgram(gpu.sampling_program);loc=lambda n:g.glGetUniformLocation(gpu.sampling_program,n.encode())
   for unit,texture in ((0,entry.textures[channel].name),(1,entry.textures['alpha'].name),(2,weighted.name)):
    g.glActiveTexture(0x84C0+unit);g.glBindTexture(0x0DE1,texture)
   for name,value in [('uSource',0),('uAlpha',1),('uPrevious',2),('uChannel',ci),('uQuarterTurn',int(abs(rotation%90)<1e-5))]:g.glUniform1i(loc(name),value)
   g.glUniform1f(loc('uRotation'),math.radians(rotation))
   set2=getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)(None,C.c_int,C.c_int,C.c_int)(gpu.context.get_proc('glUniform2i'));set2(loc('uSourceSize'),*source_size)
   for level in range(levels):
    w=max(1,width>>level);h=max(1,height>>level)
    g.glUniform1i(loc('uLevel'),level);set2(loc('uSize'),w,h);set2(loc('uPreviousSize'),max(1,width>>max(0,level-1)),max(1,height>>max(0,level-1)))
    gpu.bind_image(0,weighted.name,level,False,0,0x88B9,0x881A);gpu.bind_image(1,encoded.name,level,False,0,0x88B9,0x881A)
    for x0,y0,x1,y1 in regions[level]:
     if x1<=x0 or y1<=y0:continue
     set2(loc('uOrigin'),x0,y0);set2(loc('uDispatchSize'),x1-x0,y1-y0);gpu.f['glDispatchCompute']((x1-x0+7)//8,(y1-y0+7)//8,1)
    gpu.f['glMemoryBarrier'](0x20|0x8|0x100)
   encoded.pins+=1;records.append(encoded);names[channel]=encoded.name;updates[channel]=regions
   pending_cache[identity]=dict(encoded=encoded,weighted=weighted,stamp=stamps[channel],versions=dict(expected))
  # Publish the complete five-channel mip packet after one owner fence.
  gpu.wait()
  cache.update(pending_cache)
  # Stale cache descriptors do not retain evicted resources or grow without bound.
  for key in list(cache):
   if not _alive(gpu,cache[key]['encoded']) or not _alive(gpu,cache[key]['weighted']):cache.pop(key,None)
  while len(cache)>256:cache.pop(next(iter(cache)))
  result=ResidentResult(tile_id,0,(width,height),names,dict(expected),dict(gpu_mips=True,readback_bytes=0,channel_stamps=stamps,base_stamps=base_stamps,mip_regions=updates),tuple(records));return result,levels
 except Exception:
  for record in records:record.pins=max(0,record.pins-1)
  raise
 finally:
  for record in source_records+held:record.pins=max(0,record.pins-1)


def request_sampling(device,tile_id,expected,size,rotation=0):
 def run(gpu):
  try:
   result=create_sampling(gpu,tile_id,expected,size,rotation)
   return SamplingPacket(device,*result) if result else None
  except Exception as error:
   device.error=str(error);return None
 return device.call(run)
