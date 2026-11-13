"""GL 4.3 raster backend. No Qt, Pillow or per-segment pixel grids in GPU dispatch.

CPU generates canonical curves; compute evaluates analytic capsule footprints and
source attributes deterministically per pixel. Output maps never feed each other.
"""
import ctypes as C
import threading,time,math,os
from dataclasses import dataclass,field
from collections import OrderedDict
import numpy as np
from ..engine_v2 import gpu_bake as old
from ..engine_v2.samples import BLOCK,AO_RADIUS
from ..engine_v2.dependencies import channel_key,CHANNEL_SOURCES
from .dependencies import plan,plan_bounds
from .buffers import changed_ranges

BAKE_CHANNELS=old.ALL_BAKE_CHANNELS+("material_id","material_mask")
STRAND_DTYPE=np.dtype(old.STRAND_DTYPE.descr+[("material_mask",np.float32,(4,))])
CHANNEL_INDEX={name:i for i,name in enumerate(BAKE_CHANNELS)}

SSBO=0x90D2;UNPACK=0x88EC;TEXTURE=0x0DE1;RGBA8=0x8058;RGBA=0x1908;BYTE=0x1401
BARRIER=0x00002000|0x00000080|0x00000100|0x00000200
EMPTY_BUFFER=np.zeros(1,np.uint32);EMPTY_BUFFER.setflags(write=False)

@dataclass
class Texture:
 name:int
 size:tuple
 bytes:int
 pins:int=0
 fence:object=None
 stamp:int=0
 kind:str="raw"
 version:str=""

@dataclass
class TileResources:
 buffers:list=field(default_factory=list)
 capacities:list=field(default_factory=lambda:[0]*8)
 arrays:list=field(default_factory=lambda:[None]*8)
 upload_sources:dict=field(default_factory=dict)
 source_data:object=None
 geometry_key:object=None
 packed:object=None
 keys:dict=field(default_factory=dict)
 bins:dict=field(default_factory=dict)
 signatures:object=None
 ranges:dict=field(default_factory=dict)
 boxes:tuple=()
 channel_deltas:dict=field(default_factory=dict)
 textures:dict=field(default_factory=dict)
 versions:dict=field(default_factory=dict)

@dataclass(frozen=True)
class ResidentResult:
 tile_id:str
 revision:int
 size:tuple
 textures:dict
 versions:dict
 metrics:dict
 records:tuple=()


def shader_source():
 s=old.COMPUTE_SOURCE.replace('uChannelSlots[15]','uChannelSlots[17]')
 s=s.replace('uvec4 ids;','uvec4 ids;vec4 material_mask;')
 s=s.replace('vec3 acc_color = vec3(0.0);','vec3 acc_color = vec3(0.0);vec3 acc_material_mask=vec3(0);')
 s=s.replace('float over = 1.0 - op_a;','float over = 1.0 - op_a;if((uChannelMask&(1u<<16u))!=0u)acc_material_mask=sh.material_mask.rgb*op_a+acc_material_mask*over;')
 s=s.replace('uniform int uWidth;','layout(std430,binding=7) readonly buffer FrameBuf {vec4 frames[];};\nuniform int uWidth;')
 s=s.replace('uint acc_cluster = 0u;','uint acc_cluster = 0u; uint acc_material=0u;')
 s=s.replace('if ((uChannelMask & (1u << 7u))', 'if((uChannelMask&(1u<<15u))!=0u && op_a>.26)acc_material=uint(round(sh.extra2.w));\n        if ((uChannelMask & (1u << 7u))')
 # Material labels originate in canonical data, never reconstructed from RGB.
 end=s.rfind('}')
 s=s[:end]+'if(uChannelSlots[15]>=0)out_pixels[uint(uChannelSlots[15])*npix+pidx]=pack_u24_rgb(acc_material);if(uChannelSlots[16]>=0){vec3 m=empty?vec3(0):acc_material_mask*inv;out_pixels[uint(uChannelSlots[16])*npix+pidx]=pack_byte4(m.r,m.g,m.b,1.); }\n'+s[end:]
 s=s.replace('float prox = 0.0;', 'float prox = 0.0;float near_depth=sh.ct_depth.w;')
 s=s.replace('prox = max(prox, near_v);','if(near_v>prox)near_depth=mix(attrs[(s0+j)*4u+1u].w,attrs[(s0+j)*4u+3u].w,u);prox = max(prox, near_v);')
 s=s.replace('int zbin = clamp(int(sh.ct_depth.w * 8.0), 0, 7);','int zbin = clamp(int(near_depth * 8.0), 0, 7);',1)
 s=s.replace('uint bin_indices[]','uvec2 bin_indices[]')
 s=s.replace('uint s_idx = bin_indices[bin_start + k];','uvec2 item=bin_indices[bin_start+k]; uint s_idx=item.x;')
 s=s.replace('uint s0 = sh.ids.x;', 'uint s0=sh.ids.x+(item.y&65535u);')
 s=s.replace('uint sc = sh.ids.y;', 'uint sc=item.y>>16u;')
 s=s.replace('uniform int uWidth;', 'layout(std430,binding=6) readonly buffer AttrBuf {vec4 attrs[];};\nuniform int uWidth;')
 s=s.replace('float prox = 0.0;', 'float prox = 0.0; float near_op = sh.cm_op.w;')
 s=s.replace('prox = max(prox, near_v);','if(near_v>prox){near_op=mix(attrs[(s0+j)*4u].x,attrs[(s0+j)*4u+2u].x,u);prox=near_v;}')
 s=s.replace('prox * sh.cm_op.w','prox * near_op')
 s=s.replace('float op_a = best_cover * sh.cm_op.w;', '''vec4 a0=mix(attrs[best_seg*4u],attrs[best_seg*4u+2u],best_u);
        vec4 a1=mix(attrs[best_seg*4u+1u],attrs[best_seg*4u+3u],best_u);
        float op_a = best_cover * a0.x;''')
 for before,after in [('acc_rough = sh.cr_rough.w','acc_rough = a0.y'),('acc_spec = sh.extra1.y','acc_spec = a0.z'),('acc_aniso = sh.extra1.z','acc_aniso = a0.w'),('acc_trans = sh.extra1.w','acc_trans = a1.x'),('acc_scat = sh.extra2.x','acc_scat = a1.y'),('acc_irid = sh.extra2.y','acc_irid = a1.z')]:s=s.replace(before,after)
 s=s.replace('acc_depth = sh.ct_depth.w', 'acc_depth = a1.w')
 s=s.replace('int zbin = clamp(int(sh.ct_depth.w * 8.0), 0, 7);','int zbin=clamp(int(a1.w*8.),0,7);')
 s=s.replace('/ 8.0 - sh.ct_depth.w','/ 8.0 - a1.w')
 s=s.replace('vec3 nrm = vec3(-best_ty * lat, best_tx * lat, sqrt(max(0.03, 1.0 - lat * lat)));',
  'vec3 nrm=vec3(-best_ty*lat,best_tx*lat,sqrt(max(.03,1.-lat*lat))); if(frames[best_seg*2u].w>.5){float tz=mix(frames[best_seg*2u].z,frames[best_seg*2u+1u].z,best_u);vec3 side=vec3(-best_ty,best_tx,0);vec3 front=vec3(-best_tx*tz,-best_ty*tz,sqrt(max(0.,1.-tz*tz)));nrm=side*lat+front*sqrt(max(.03,1.-lat*lat));}')
 s=s.replace('acc_flow = vec3(best_tx, best_ty, 0.0) * op_a + acc_flow * over;',
  'vec3 flow=frames[best_seg*2u].w>.5?mix(frames[best_seg*2u].xyz,frames[best_seg*2u+1u].xyz,best_u):vec3(best_tx,best_ty,0);acc_flow=flow*op_a+acc_flow*over;')
 from .quality import extend
 return extend(s,len(BAKE_CHANNELS))


def pack_pigment(batch):
 colors=batch.sampled_colors
 if colors is None:return None
 if batch.uniform_detail>=2:
  values=colors.reshape(batch.count,batch.uniform_detail,3)
  first=values[:,:-1].reshape(-1,3);last=values[:,1:].reshape(-1,3)
 else:
  valid=np.ones(max(0,len(colors)-1),bool)
  edges=batch.offsets[1:-1]-1;edges=edges[(edges>=0)&(edges<len(valid))];valid[edges]=False
  first=colors[:-1][valid];last=colors[1:][valid]
 result=np.zeros(len(first),old.SAMPLED_SEG_DTYPE)
 result['c0'][:,:3]=first;result['c1'][:,:3]=last
 return result


def pack_frames(truth):
    pieces=[]
    for strand in truth:
        length=len(strand.points)
        if length<2:continue
        data=np.zeros((length,4),np.float32)
        if strand.depth_profile is not None:
            tangent=np.gradient(strand.positions_3d,axis=0)
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
            data[:,:3]=tangent;data[:,3]=1
        pieces.append(np.stack((data[:-1],data[1:]),axis=1))
    return np.concatenate(pieces) if pieces else np.zeros((0,2,4),np.float32)


def pack_attributes(truth):
 lengths=np.asarray([max(0,len(s.t)-1) for s in truth],np.int32)
 n=int(lengths.sum());out=np.zeros((n,4,4),np.float32);cursor=0
 fields=('opacity','roughness','specular','anisotropy','transmission','scatter','iridescence')
 batch=getattr(truth,'batch',None)
 if batch is not None and not any(s.attributes or s.depth_profile is not None for s in truth):
  values=np.column_stack([getattr(batch,key) for key in fields]+[batch.depth]).astype(np.float32)
  per_segment=np.repeat(values,lengths,axis=0).reshape(n,2,4)
  out[:,:2]=per_segment;out[:,2:]=per_segment
  return out
 # This loop packs source vertices once; it never creates any pixel grid.
 for strand,length in zip(truth,lengths):
  if not length:continue
  samples=np.empty((length+1,8),np.float32)
  for i,key in enumerate(fields):samples[:,i]=strand.attribute(key,strand.t)
  samples[:,7]=strand.depth_at(strand.t)
  out[cursor:cursor+length,:2]=samples[:-1].reshape(length,2,4)
  out[cursor:cursor+length,2:]=samples[1:].reshape(length,2,4);cursor+=length
 return out


class GLBackend:
 def __init__(self,context,budget_mb=512):
  from ..rendercore.glapi import GL
  from .backend import BackendCapabilities
  self.capabilities=BackendCapabilities('OpenGL',True,True,True,int(budget_mb*1024*1024))
  self.context=context;self.g=GL(context);self.owner=threading.get_ident();self.budget=int(budget_mb*1024*1024)
  self.entries=OrderedDict();self.raw_versions=OrderedDict();self.pool=[];self.used=0;self.clock=0;self.scratch=0;self.scratch_cap=0
  self.f={};conv=getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)
  for name,ret,args in (
   ('glDispatchCompute',None,[C.c_uint,C.c_uint,C.c_uint]),
   ('glMemoryBarrier',None,[C.c_uint]),('glBindBufferBase',None,[C.c_uint,C.c_uint,C.c_uint]),
   ('glClearBufferSubData',None,[C.c_uint,C.c_uint,C.c_ssize_t,C.c_ssize_t,C.c_uint,C.c_uint,C.c_void_p]),
   ('glUniform1ui',None,[C.c_int,C.c_uint]),
   ('glCopyImageSubData',None,[C.c_uint,C.c_uint,C.c_int,C.c_int,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int])):
   ptr=context.get_proc(name)
   if not ptr:raise RuntimeError('FiberBake X missing '+name)
   self.f[name]=conv(ret,*args)(ptr)
  self.program=self.compile(shader_source());self.locations={name:self.g.glGetUniformLocation(self.program,name.encode()) for name in ('uWidth','uHeight','uBlockBase','uBlockCount','uChannelMask','uAntialias','uHalo','uRoundness','uPattern','uBandCount','uBandStrength','uSpeckle','uDefaults0','uDefaults1','uSamples','uExactAO','uDepthRange')}
  self.slots=[self.g.glGetUniformLocation(self.program,f'uChannelSlots[{i}]'.encode()) for i in range(len(BAKE_CHANNELS))]
  self.scratch=self.g.name('Buffers');self.last_metrics={};self.error=''
  from .profiling import TimestampProfiler
  self.profiler=TimestampProfiler(self.g) if os.environ.get('FSD_FIBERBAKE_PROFILE')=='1' else None
 def assert_owner(self):
  if threading.get_ident()!=self.owner:raise RuntimeError('GPU resource accessed outside its context owner')
 def compile(self,source):
  g=self.g;shader=g.glCreateShader(0x91B9);code=source.encode();ptr=C.c_char_p(code);length=C.c_int(len(code));ok=C.c_int()
  g.glShaderSource(shader,1,C.byref(ptr),C.byref(length));g.glCompileShader(shader);g.glGetShaderiv(shader,0x8B81,C.byref(ok))
  if not ok.value:
   log=C.create_string_buffer(16384);g.glGetShaderInfoLog(shader,len(log),None,log);g.glDeleteShader(shader);raise RuntimeError(log.value.decode(errors='replace'))
  program=g.glCreateProgram();g.glAttachShader(program,shader);g.glLinkProgram(program);g.glDeleteShader(shader);g.glGetProgramiv(program,0x8B82,C.byref(ok))
  if not ok.value:
   log=C.create_string_buffer(16384);g.glGetProgramInfoLog(program,len(log),None,log);g.glDeleteProgram(program);raise RuntimeError(log.value.decode(errors='replace'))
  return program
 def _ready(self,texture):
  if texture.pins:return False
  if texture.fence:
   result=self.g.glClientWaitSync(texture.fence,0,0)
   if result==0x911D:raise RuntimeError('Consumer fence failed')
   if result not in (0x911A,0x911C):return False
   self.g.glDeleteSync(texture.fence);texture.fence=None
  return True
 def _delete_entry(self,key):
  e=self.entries.pop(key)
  if e.buffers:
   arr=(C.c_uint*len(e.buffers))(*e.buffers);self.g.glDeleteBuffers(len(e.buffers),arr)
  if getattr(e,'bin_buffers',None):
   names=list(e.bin_buffers.values());arr=(C.c_uint*len(names))(*names);self.g.glDeleteBuffers(len(names),arr)
  self.used-=sum(e.capacities)+sum(getattr(e,'bin_capacities',{}).values())
  # Texture pool owns texture lifetime independently of tile metadata.
 def _reserve(self,amount,keep=None):
  if amount>self.budget:raise MemoryError('FiberBake X single allocation exceeds GPU budget')
  # Sorting/scan workspace is expendable after bins have been built. Retain
  # canonical buffers and final spatial index before evicting an entire tile.
  for entry in self.entries.values():
   if self.used+amount<=self.budget:break
   if getattr(entry,'bin_building',False):continue
   buffers=getattr(entry,'bin_buffers',{});capacities=getattr(entry,'bin_capacities',{})
   for name in list(buffers):
    if name in ('blocks','indices'):continue
    handle=C.c_uint(buffers.pop(name));self.g.glDeleteBuffers(1,C.byref(handle));self.used-=capacities.pop(name,0)
    if self.used+amount<=self.budget:break
  for key in list(self.entries):
   if self.used+amount<=self.budget:break
   if key!=keep:self._delete_entry(key)
  if self.used+amount>self.budget and keep in self.entries:
   entry=self.entries[keep]
   for channel in list(entry.textures):
    if channel not in getattr(self,'active_channels',()) and not entry.textures[channel].pins:
     entry.textures.pop(channel);entry.versions.pop(channel,None)
  protected={id(t) for e in self.entries.values() for t in e.textures.values()}
  for t in list(self.pool):
   if self.used+amount<=self.budget:break
   if id(t) not in protected and self._ready(t):
    arr=C.c_uint(t.name);self.g.glDeleteTextures(1,C.byref(arr));self.used-=t.bytes;self.pool.remove(t)
  if self.used+amount>self.budget:raise MemoryError('FiberBake X GPU budget exhausted; release/evict resident leases')
 def _texture(self,size,keep):
  protected={id(t) for e in self.entries.values() for t in e.textures.values()}
  for t in self.pool:
   if t.kind=='raw' and t.size==size and id(t) not in protected and self._ready(t):return t
  amount=size[0]*size[1]*4;self._reserve(amount,keep)
  tex=self.g.name('Textures');self.g.glBindTexture(TEXTURE,tex)
  self.g.glTexParameteri(TEXTURE,0x2801,0x2601);self.g.glTexParameteri(TEXTURE,0x2800,0x2601)
  self.g.glTexParameteri(TEXTURE,0x2802,0x812F);self.g.glTexParameteri(TEXTURE,0x2803,0x812F)
  self.g.glBindBuffer(UNPACK,0);self.g.glTexImage2D(TEXTURE,0,RGBA8,*size,0,RGBA,BYTE,None)
  t=Texture(tex,size,amount);self.pool.append(t);self.used+=amount;return t
 def _upload(self,e,slot,array,keep):
  if array is None:array=EMPTY_BUFFER
  array=np.ascontiguousarray(array);g=self.g;g.glBindBuffer(SSBO,e.buffers[slot]);n=array.nbytes
  if e.upload_sources.get(slot) is array:
   self.f['glBindBufferBase'](SSBO,slot,e.buffers[slot]);return 0,0
  uploaded=0;allocation=0
  if max(n,32)>e.capacities[slot]:
   capacity=max(4096,1<<(max(n,32)-1).bit_length());self._reserve(capacity-e.capacities[slot],keep)
   g.glBufferData(SSBO,capacity,None,0x88E8);self.used+=capacity-e.capacities[slot];e.capacities[slot]=capacity
   ranges=[(0,n)] if n else [];allocation=1
  else:ranges=changed_ranges(e.arrays[slot],array)
  for offset,length in ranges:g.glBufferSubData(SSBO,offset,length,array.ctypes.data+offset);uploaded+=length
  e.arrays[slot]=array.copy();e.upload_sources[slot]=array;self.f['glBindBufferBase'](SSBO,slot,e.buffers[slot]);return uploaded,allocation
 def wait(self,cancel=lambda:False):
  fence=self.g.glFenceSync(0x9117,0);self.g.glFlush();start=time.perf_counter()
  try:
   while True:
    status=self.g.glClientWaitSync(fence,0,0)
    if status in (0x911A,0x911C):break
    if status==0x911D:raise RuntimeError('GPU fence wait failed')
    # Always retire submitted GPU commands before reusing their resources.
    if time.perf_counter()-start>120:raise RuntimeError('GPU fence timeout')
    time.sleep(.001)
  finally:self.g.glDeleteSync(fence)
  error=self.g.glGetError()
  if error:raise RuntimeError('FiberBake X OpenGL error '+hex(error))
  if cancel():raise RuntimeError('Render cancelled')
  return (time.perf_counter()-start)*1000
 def render(self,tile,truth,keys,bins_data,size,channels,cancel=lambda:False,progress=None,pin=False):
  self.assert_owner();self.context.activate()
  batch=getattr(truth,'batch',None)
  if batch is None:raise NotImplementedError('Unbatched canonical truth')
  self.active_channels=set(channels)
  started=time.perf_counter();key=(tile.id,tuple(size));g=self.g;self.clock+=1
  if key not in self.entries:
   e=TileResources();e.buffers=[g.name('Buffers') for _ in range(8)];self.entries[key]=e
  e=self.entries[key];self.entries.move_to_end(key)
  if all(c in e.textures and c in e.versions and channel_key(c,keys,size)==channel_key(c,e.versions[c],size) for c in channels):
   records=tuple(e.textures[c] for c in channels)
   if pin:
    for record in records:record.pins+=1
   metrics=dict(backend='FiberBake X / OpenGL compute',renderer=g.renderer,version=g.version,uploaded_bytes=0,allocations=0,dirty_channels=[],dirty_blocks=0,passes=0,gpu_bytes=self.used,budget_bytes=self.budget,prepare_ms=0,fence_wait_ms=0,total_ms=(time.perf_counter()-started)*1000,readback_bytes=0)
   self.last_metrics=metrics
   return ResidentResult(tile.id,tile.revision,tuple(size),{c:e.textures[c].name for c in channels},{c:channel_key(c,keys,size) for c in channels},metrics,records if pin else ())
  geometry_key=(keys['geometry'],float(tile.settings.antialias),tuple(size));geometry_changed=e.geometry_key!=geometry_key
  if geometry_changed:
   packed=old._pack_gpu_geometry(batch,bins_data,size,tile.settings.antialias)
  else:packed=e.packed
  starts,counts,boxes,segs,sampled,_,_=packed
  # Sampled pigment belongs to MATERIAL, not to the geometry cache.
  if batch.sampled_colors is not None and (geometry_changed or keys.get('color')!=e.keys.get('color')):
   sampled=pack_pigment(batch)
  gpu_bins=not bool(bins_data[0])
  if geometry_changed and not gpu_bins:
   e.ranges={}
   # Store a conservative contiguous segment interval per strand/bin. Sparse
   # curves no longer scan every segment in every covered pixel.
   for block,ids in bins_data[0].items():
    bx,by=block;packed_indices=[]
    if batch.uniform_detail>=2:
     detail=batch.uniform_detail-1;local=segs.reshape(batch.count,detail)[np.asarray(ids,np.int32)]
     pad=np.maximum(local['rt'][...,0],local['rt'][...,1])+max(AO_RADIUS*size[0]/512.,.5*max(.3,tile.settings.antialias))+1
     ab=local['ab'];valid=(np.minimum(ab[...,0],ab[...,2])-pad<(bx+1)*BLOCK)&(np.maximum(ab[...,0],ab[...,2])+pad>bx*BLOCK)&(np.minimum(ab[...,1],ab[...,3])-pad<(by+1)*BLOCK)&(np.maximum(ab[...,1],ab[...,3])+pad>by*BLOCK)
     keep=valid.any(axis=1);first=valid.argmax(axis=1);last=detail-1-valid[:,::-1].argmax(axis=1);length=last-first+1
     e.ranges[block]=np.column_stack((np.asarray(ids,np.uint32)[keep],(first[keep]|(length[keep]<<16)).astype(np.uint32)))
     continue
    for index in ids:
     first=int(starts[index]);count=int(counts[index]);local=segs[first:first+count]
     pad=np.maximum(local['rt'][:,0],local['rt'][:,1])+max(AO_RADIUS*size[0]/512.,.5*max(.3,tile.settings.antialias))+1
     ab=local['ab'];valid=(np.minimum(ab[:,0],ab[:,2])-pad<(bx+1)*BLOCK)&(np.maximum(ab[:,0],ab[:,2])+pad>bx*BLOCK)&(np.minimum(ab[:,1],ab[:,3])-pad<(by+1)*BLOCK)&(np.maximum(ab[:,1],ab[:,3])+pad>by*BLOCK)
     selected=np.flatnonzero(valid)
     if len(selected):
      start=int(selected[0]);length=int(selected[-1])-start+1
      if start>65535 or length>65535:raise NotImplementedError('Segment interval exceeds packed GPU index')
      packed_indices.append((int(index),start|(length<<16)))
    e.ranges[block]=np.asarray(packed_indices,np.uint32).reshape(-1,2)
  if e.keys==keys and not geometry_changed and e.source_data is not None:
   strand_data,attributes,frames=e.source_data
  else:
   packed_strands=old._pack_gpu_strands(batch,starts,counts,boxes);strand_data=np.zeros(len(packed_strands),dtype=STRAND_DTYPE)
   for name in packed_strands.dtype.names:strand_data[name]=packed_strands[name]
   strand_data['material_mask'][:,:3]=np.asarray([s.material_mask for s in truth],np.float32)
   attribute_sources=('geometry','opacity','roughness','specular','anisotropy','transmission','scatter','iridescence')
   attributes=e.source_data[1] if e.source_data is not None and not geometry_changed and all(keys.get(k)==e.keys.get(k) for k in attribute_sources) else pack_attributes(truth)
   strand_data['extra2'][:,3]=np.asarray([s.material_id for s in truth],np.float32)
   frames=pack_frames(truth) if geometry_changed or e.source_data is None else e.source_data[2]
  depth_low=min(0.,float(np.min(attributes[:,(1,3),3]))) if len(attributes) else 0.
  depth_high=max(1.,float(np.max(attributes[:,(1,3),3]))) if len(attributes) else 1.
  depth_range=(depth_low,depth_high-depth_low)
  old_signatures=e.signatures;new_signatures=bins_data[1];changed=None
  if geometry_changed and old_signatures is not None and len(old_signatures)==len(new_signatures):changed=[i for i,(a,b) in enumerate(zip(old_signatures,new_signatures)) if a!=b]
  local_changed=changed if geometry_changed and all(keys.get(k)==e.keys.get(k) for k in keys if k not in ('geometry',)) else None
  uniforms_now={name:getattr(tile.settings,name,None) for name in ('roughness','specular','anisotropy','transmission','scatter','iridescence','fiber_roundness','pattern','band_count','band_strength','speckle','antialias','bake_samples','ao_mode')}
  if e.keys==keys and not geometry_changed:
   local_changed=[]
  elif not geometry_changed and e.arrays[0] is not None and e.arrays[0].shape==strand_data.shape and e.arrays[6] is not None and e.arrays[6].shape==attributes.shape:
   changed_rows=np.any(e.arrays[0].view(np.uint8).reshape(len(strand_data),-1)!=strand_data.view(np.uint8).reshape(len(strand_data),-1),axis=1)
   changed_segments=np.any(e.arrays[6]!=attributes,axis=(1,2));owners=np.repeat(np.arange(len(counts)),counts.astype(np.int64))
   if len(owners):changed_rows[np.unique(owners[changed_segments])]=True
   if sampled is not None and e.arrays[2] is not None and e.arrays[2].dtype==sampled.dtype and e.arrays[2].shape==sampled.shape:
    pigment_segments=np.any(e.arrays[2].view(np.uint8).reshape(len(sampled),-1)!=sampled.view(np.uint8).reshape(len(sampled),-1),axis=1)
    if len(owners):changed_rows[np.unique(owners[pigment_segments])]=True
   previous_uniforms=getattr(e,'uniform_values',{})
   if previous_uniforms==uniforms_now:local_changed=np.flatnonzero(changed_rows).tolist()

  plans={}
  for c in channels:
   previous=e.versions.get(c)
   # Incremental masks describe the immediately preceding buffer state only.
   # A channel not requested for several edits must not reuse that newer mask.
   contiguous=previous is not None and e.keys and channel_key(c,previous,size)==channel_key(c,e.keys,size)
   changed_for_channel=local_changed if contiguous else None
   if c=='ao' and getattr(e,'ao_range',None)!=depth_range:changed_for_channel=None
   plans[c]=plan_bounds(keys,previous,bins_data[2],e.boxes,size,(c,),changed_for_channel) if gpu_bins else plan(keys,previous,bins_data[0],e.bins,size,(c,),changed_for_channel)
  dirty=[c for c in channels if c not in e.textures or plans[c].channels]
  previous_versions={c:channel_key(c,e.versions[c],size) if c in e.versions else None for c in dirty}
  upload=allocations=0
  for slot,array in ((0,strand_data),(1,segs),(2,sampled),(6,attributes),(7,frames)):
   # Slot 5 is scratch output; slot 3 is the per-pass dirty block list.
   n,a=self._upload(e,slot,array,key);upload+=n;allocations+=a
  if gpu_bins:
   if geometry_changed or not hasattr(e,'gpu_bin_data'):
    from .binning import GPUBins
    if self.profiler:self.profiler.begin('bins')
    e.bin_building=True
    try:e.gpu_bin_data=GPUBins(self,e,key).build(segs,starts,counts,size,max(AO_RADIUS*size[0]/512.,.5*max(.3,tile.settings.antialias))+3.,cancel)
    finally:e.bin_building=False
    if self.profiler:self.profiler.end('bins')
   for slot in (0,1,2,6,7):self.f['glBindBufferBase'](SSBO,slot,e.buffers[slot])
   self.f['glBindBufferBase'](SSBO,3,e.gpu_bin_data[0]);self.f['glBindBufferBase'](SSBO,4,e.gpu_bin_data[1])
  prepared=time.perf_counter();blocks_total=0;pass_count=0;fence_ms=0
  # Bound scratch independently of the requested channel set (two/pass at 4K).
  pass_width=2 if size[0]*size[1]>=4096*4096 else 4
  if self.profiler:self.profiler.begin('raster-resolve')
  for begin in range(0,len(dirty),pass_width):
   if cancel():raise RuntimeError('Render cancelled')
   group=dirty[begin:begin+pass_width];npix=size[0]*size[1];nbytes=len(group)*npix*4
   if nbytes>self.scratch_cap:
    self._reserve(nbytes-self.scratch_cap,key);g.glBindBuffer(SSBO,self.scratch);g.glBufferData(SSBO,nbytes,None,0x88E8);self.used+=nbytes-self.scratch_cap;self.scratch_cap=nbytes;allocations+=1
   g.glBindBuffer(SSBO,self.scratch);self.f['glBindBufferBase'](SSBO,5,self.scratch)
   blocks=set().union(*(set(plans[c].blocks) for c in group));blocks_total+=len(blocks)
   if not gpu_bins:
    packed_bins=bins_data[0];cursor=0;indices=[];records=[]
    for bx,by in sorted(blocks):
     ids=e.ranges.get((bx,by),());records.append((bx,by,cursor,len(ids)));indices.extend(ids);cursor+=len(ids)
    # Repack indices for old-only blocks too: they must clear pixels after movement.
    for slot,array in ((3,np.asarray(records,dtype=np.uint32).reshape(-1,4)),(4,np.asarray(indices,dtype=np.uint32).reshape(-1,2))):
     n,a=self._upload(e,slot,array,key);upload+=n;allocations+=a
   g.glBindBuffer(SSBO,self.scratch)
   mask=0;slots=[-1]*len(BAKE_CHANNELS)
   for i,c in enumerate(group):
    slots[CHANNEL_INDEX[c]]=i;mask|=1<<CHANNEL_INDEX[c]
   # Every invocation writes every selected channel, including empty bins.
   # Only dispatched dirty blocks are copied below: no full-image scratch clear.
   g.glUseProgram(self.program);loc=self.locations
   for name,value in [('uWidth',size[0]),('uHeight',size[1]),('uPattern',old.PATTERN_MAP.get(tile.settings.pattern,0)),('uSamples',max(1,min(16,int(getattr(tile.settings,'bake_samples',1))))),('uExactAO',int(getattr(tile.settings,'ao_mode','proximity')=='geometry'))]:g.glUniform1i(loc[name],value)
   self.f['glUniform1ui'](loc['uChannelMask'],mask)
   g.glUniform2f(loc['uDepthRange'],*depth_range)
   for name,value in [('uAntialias',tile.settings.antialias),('uHalo',max(.5,AO_RADIUS*size[0]/512)),('uRoundness',tile.settings.fiber_roundness),('uBandCount',tile.settings.band_count),('uBandStrength',tile.settings.band_strength),('uSpeckle',tile.settings.speckle)]:g.glUniform1f(loc[name],float(value))
   g.glUniform4f(loc['uDefaults0'],tile.settings.roughness,tile.settings.specular,tile.settings.anisotropy,tile.settings.transmission);g.glUniform4f(loc['uDefaults1'],tile.settings.scatter,tile.settings.iridescence,0,0)
   for i,location in enumerate(self.slots):g.glUniform1i(location,slots[i])
   self.f['glMemoryBarrier'](BARRIER)
   if gpu_bins:
    columns=e.gpu_bin_data[2];ordered=sorted(by*columns+bx for bx,by in blocks);runs=[]
    max_chunk=max(1,min(64,4000000//max(1,batch.count*max(1,int(np.max(counts)))*max(1,int(getattr(tile.settings,'bake_samples',1))))))
    for value in ordered:
     if runs and value==runs[-1][0]+runs[-1][1] and runs[-1][1]<max_chunk:runs[-1]=(runs[-1][0],runs[-1][1]+1)
     else:runs.append((value,1))
   else:runs=[(base,min(64,len(records)-base)) for base in range(0,len(records),64)]
   for dispatch_number,(base,count) in enumerate(runs):
    if cancel():raise RuntimeError('Render cancelled')
    g.glUniform1i(loc['uBlockBase'],base);g.glUniform1i(loc['uBlockCount'],count);self.f['glDispatchCompute'](count*4,1,1)
    if dispatch_number%8==7:fence_ms+=self.wait(cancel)
   self.f['glMemoryBarrier'](BARRIER)
   # Copy GPU buffer -> GPU texture. No CPU round trip or Pillow conversion.
   g.glPixelStorei(0x0CF5,1);g.glPixelStorei(0x0CF2,size[0])
   pending={}
   for i,c in enumerate(group):
    previous=e.textures.get(c);full=plans[c].full or previous is None
    target=previous if previous is not None and self._ready(previous) else self._texture(size,key)
    # Invalidate old aliases before any write, including a cancelled update.
    target.version=""
    if target is previous:e.versions.pop(c,None)
    if previous is not None and target is not previous and not full:self.f['glCopyImageSubData'](previous.name,TEXTURE,0,0,0,0,target.name,TEXTURE,0,0,0,0,*size,1)
    g.glBindTexture(TEXTURE,target.name)
    filtering=0x2600 if c in ('id','cluster_id','material_id') else 0x2601
    g.glTexParameteri(TEXTURE,0x2801,filtering);g.glTexParameteri(TEXTURE,0x2800,filtering)
    g.glBindBuffer(UNPACK,self.scratch)
    rects=[(0,0,*size)] if full else [(bx*BLOCK,by*BLOCK,min(BLOCK,size[0]-bx*BLOCK),min(BLOCK,size[1]-by*BLOCK)) for bx,by in plans[c].blocks]
    for x,y,width,height in rects:
     offset=(i*npix+y*size[0]+x)*4;g.glTexSubImage2D(TEXTURE,0,x,y,width,height,RGBA,BYTE,C.c_void_p(offset))
    pending[c]=target;target.stamp=self.clock;target.pins+=1
   g.glBindBuffer(UNPACK,0);g.glPixelStorei(0x0CF2,0)
   try:fence_ms+=self.wait(cancel)
   finally:
    for texture in pending.values():texture.pins-=1
   pass_count+=1
   # Commit only after the GPU finishes; cancelled versions are never cache hits.
   for c in group:
    e.textures[c]=pending[c];e.versions[c]=dict(keys)
    version=channel_key(c,keys,size);pending[c].version=version;self.raw_versions[(tile.id,tuple(size),c,version)]=pending[c]
    if c=='ao':e.ao_range=depth_range
    e.channel_deltas[c]=(previous_versions[c],channel_key(c,keys,size),tuple((bx*BLOCK,by*BLOCK,min(size[0],(bx+1)*BLOCK),min(size[1],(by+1)*BLOCK)) for bx,by in plans[c].blocks))
   if progress:progress(min(1.,(begin+len(group))/max(1,len(dirty))))
  gpu_timestamps=None
  if self.profiler:
   self.profiler.end('raster-resolve');self.wait(cancel);gpu_timestamps=self.profiler.collect()
  packed=(starts,counts,boxes,segs,sampled,packed[5],packed[6])
  e.source_data=(strand_data,attributes,frames)
  e.packed=packed;e.geometry_key=geometry_key;e.keys=dict(keys);e.bins=dict(bins_data[0]);e.signatures=new_signatures;e.boxes=bins_data[2];e.uniform_values=uniforms_now
  while len(self.raw_versions)>8192:self.raw_versions.popitem(last=False)
  records=tuple(e.textures[c] for c in channels)
  if pin:
   for record in records:record.pins+=1
  metrics=dict(backend='FiberBake X / OpenGL compute',renderer=g.renderer,version=g.version,uploaded_bytes=upload,allocations=allocations,dirty_channels=dirty,dirty_blocks=blocks_total,passes=pass_count,gpu_bytes=self.used,budget_bytes=self.budget,prepare_ms=(prepared-started)*1000,fence_wait_ms=fence_ms,total_ms=(time.perf_counter()-started)*1000,readback_bytes=0,gpu_timestamps_ms=gpu_timestamps,bin_backend='gpu' if gpu_bins else 'cpu',dirty_regions={c:[(bx*BLOCK,by*BLOCK,min(size[0],(bx+1)*BLOCK),min(size[1],(by+1)*BLOCK)) for bx,by in plans[c].blocks] for c in dirty})
  self.last_metrics=metrics
  return ResidentResult(tile.id,tile.revision,tuple(size),{c:e.textures[c].name for c in channels},{c:channel_key(c,keys,size) for c in channels},metrics,records if pin else ())
 def readback(self,result):
  self.assert_owner();raw={};g=self.g;g.glBindBuffer(0x88EB,0)
  for channel,texture in result.textures.items():
   data=np.empty((result.size[1],result.size[0],4),np.uint8);g.glBindTexture(TEXTURE,texture);g.glGetTexImage(TEXTURE,0,RGBA,BYTE,data.ctypes.data);raw[channel]=data
  result.metrics['readback_bytes']=sum(v.nbytes for v in raw.values());return raw
 def release(self,result,fence=None):
  self.assert_owner()
  if fence:
   status=self.g.glClientWaitSync(fence,0,0)
   if status==0x911D:raise RuntimeError('Consumer GPU fence failed; leases quarantined')
   if status not in (0x911A,0x911C):return False
   self.g.glDeleteSync(fence)
  for record in result.records:record.pins=max(0,record.pins-1)
  return True
 def close(self):
  self.assert_owner();self.context.activate()
  if self.profiler:self.profiler.close()
  for key in list(self.entries):self._delete_entry(key)
  for texture in self.pool:
   arr=C.c_uint(texture.name);self.g.glDeleteTextures(1,C.byref(arr))
  for program in getattr(self,'evaluation_programs',{}).values():self.g.glDeleteProgram(program)
  names=list(getattr(self,'evaluation_buffers',{}).values())
  if names:self.g.glDeleteBuffers(len(names),(C.c_uint*len(names))(*names))
  for program in getattr(self,'bin_programs',{}).values():self.g.glDeleteProgram(program)
  if hasattr(self,'sampling_program'):self.g.glDeleteProgram(self.sampling_program)
  if hasattr(self,'compose_program'):self.g.glDeleteProgram(self.compose_program)
  self.pool.clear();arr=C.c_uint(self.scratch);self.g.glDeleteBuffers(1,C.byref(arr));self.g.glDeleteProgram(self.program);self.used=0
