"""GPU canonical point evaluation with stable CPU-authored random streams.

Seeds/topology remain source metadata. Dense curve formulas run on GPU. Legacy
CPU callers receive one canonical readback; raster consumes the same coordinates.
"""
import ctypes as C
import os,threading
from hashlib import blake2b
from ..engine_v2.cache import ByteLRU
from .buffers import changed_ranges

_results=ByteLRU(48*1024*1024)
import numpy as np

_disabled=threading.local()

def enabled():
 return os.name=='nt' and os.environ.get('FSD_FIBERBAKE_BACKEND','auto') not in ('cpu','legacy-gpu') and not getattr(_disabled,'value',False) and threading.current_thread() is not threading.main_thread()

def execute(name,source,arrays,output_shape,uniforms,count):
 from .service import device
 # Content addressing also reuses geometry after Undo/Redo and graph reevaluation.
 # Hash contiguous views without creating a second bytes copy of large inputs.
 arrays=[np.ascontiguousarray(array) for array in arrays]
 digest=blake2b(digest_size=24)
 digest.update(repr((name,source,tuple(output_shape),sorted(uniforms.items()),int(count))).encode())
 for array in arrays:
  digest.update(repr((array.shape,array.dtype.str)).encode())
  if array.nbytes:digest.update(memoryview(array).cast('B'))
 key=digest.digest();cached=_results.get(key)
 if cached is not None:return cached.copy()
 d=device(False)
 def run(gpu):
  gpu.assert_owner();g=gpu.g
  if not hasattr(gpu,'evaluation_programs'):gpu.evaluation_programs={};gpu.evaluation_buffers={};gpu.evaluation_capacities={}
  if name not in gpu.evaluation_programs:gpu.evaluation_programs[name]=gpu.compile(source)
  if not hasattr(gpu,'evaluation_inputs'):gpu.evaluation_inputs={};gpu.evaluation_locations={}
  program=gpu.evaluation_programs[name];data=list(arrays)+[None];output_bytes=int(np.prod(output_shape))*4
  for slot,array in enumerate(data):
   amount=max(32,output_bytes if array is None else array.nbytes);old=gpu.evaluation_capacities.get(slot,0)
   if slot not in gpu.evaluation_buffers:gpu.evaluation_buffers[slot]=g.name('Buffers')
   buffer=gpu.evaluation_buffers[slot];g.glBindBuffer(0x90D2,buffer)
   if amount>old:
    capacity=max(4096,1<<(amount-1).bit_length());gpu._reserve(capacity-old,None);g.glBufferData(0x90D2,capacity,None,0x88E8);gpu.used+=capacity-old;gpu.evaluation_capacities[slot]=capacity
    gpu.evaluation_inputs.pop(slot,None)
   if array is not None:
    previous=gpu.evaluation_inputs.get(slot)
    for offset,length in changed_ranges(previous,array):g.glBufferSubData(0x90D2,offset,length,array.ctypes.data+offset)
    gpu.evaluation_inputs[slot]=array.copy()
   else:
    # This slot will become shader output, invalidating any former input shadow.
    gpu.evaluation_inputs.pop(slot,None)
   gpu.f['glBindBufferBase'](0x90D2,slot,buffer)
  g.glUseProgram(program)
  locations=gpu.evaluation_locations.setdefault(program,{})
  def loc(name):
   if name not in locations:locations[name]=g.glGetUniformLocation(program,name.encode())
   return locations[name]
  for key,value in uniforms.items():
   if isinstance(value,tuple):
    if len(value)==4:g.glUniform4f(loc(key),*value)
    elif len(value)==3:g.glUniform3f(loc(key),*value)
    elif len(value)==2:g.glUniform2f(loc(key),*value)
   elif isinstance(value,int):g.glUniform1i(loc(key),value)
   else:g.glUniform1f(loc(key),float(value))
  if count:gpu.f['glDispatchCompute']((count+63)//64,1,1)
  gpu.f['glMemoryBarrier'](0x2000|0x200);gpu.wait()
  output=np.empty(output_shape,np.float32);g.glBindBuffer(0x90D2,gpu.evaluation_buffers[len(arrays)])
  reader=getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)(None,C.c_uint,C.c_ssize_t,C.c_ssize_t,C.c_void_p)(gpu.context.get_proc('glGetBufferSubData'))
  if output_bytes:reader(0x90D2,0,output_bytes,output.ctypes.data)
  if not np.isfinite(output).all():raise ValueError('GPU geometry produced non-finite canonical data')
  return output
 output=d.call(run,priority=0).result()
 output.setflags(write=False);_results.put(key,output)
 return output.copy()

FIBER=r'''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Recipes{vec4 recipes[];};
layout(std430,binding=1) readonly buffer Noise{float noise[];};
layout(std430,binding=2) readonly buffer Samples{vec4 samples[];};
layout(std430,binding=3) buffer Output{vec4 vertices[];};
uniform int uCount,uDetail,uNoiseCount;uniform vec4 uShape;uniform vec4 uCurl;uniform vec4 uWidth;uniform vec4 uProfile;uniform float uProfileLast;
float radius(float t){float profile=t<.25?mix(uProfile.x,uProfile.y,t*4.):t<.5?mix(uProfile.y,uProfile.z,(t-.25)*4.):t<.75?mix(uProfile.z,uProfile.w,(t-.5)*4.):mix(uProfile.w,uProfileLast,(t-.75)*4.);return max(0.,(uWidth.y+(1.-uWidth.y)*pow(1.-t,uWidth.x))*max(.02,1.+uWidth.z*4.*t*(1.-t)))*profile;}
vec2 point(uint id,int j){vec4 p=recipes[id*6u],phase=recipes[id*6u+1u],extra=recipes[id*6u+2u];vec4 knot=samples[j];float t=knot.x,sp=sin(3.141592653589793*t),wave=6.28318530718*uCurl.y*t;
float rnd=noise[id*uint(uNoiseCount)+uint(knot.w)]*knot.y+noise[id*uint(uNoiseCount)+uint(knot.w)+1u]*knot.z;
float x=p.x+(p.y-p.x)*t+uShape.x*.18*sp*p.z+uCurl.x*uCurl.z*(sin(wave)*phase.x+cos(wave)*phase.y)*sp+uShape.z*.07*sin(6.28318530718*(uCurl.y*.5)*t)*pow(t,.8)+uShape.w*.085*pow(max(0.,sp),.4)*rnd;
float y=p.w-p.z*t+uShape.y*.16*t*t+.004*uCurl.x*(sin(wave)*phase.z+cos(wave)*phase.w);
float f=t*4.;int a=min(3,int(f));vec2 wa=vec2(recipes[id*6u+3u+uint(a/2)][(a%2)*2],recipes[id*6u+3u+uint(a/2)][(a%2)*2+1]);int b=a+1;vec2 wb=b==4?recipes[id*6u+5u].xy:vec2(recipes[id*6u+3u+uint(b/2)][(b%2)*2],recipes[id*6u+3u+uint(b/2)][(b%2)*2+1]);
vec2 warp=mix(wa,wb,f-float(a));return vec2(x,y)+warp+extra.yz+vec2(extra.w*sp*t,0);}
void main(){uint id=gl_GlobalInvocationID.x;if(id>=uint(uCount))return;vec2 low=vec2(1e30),high=vec2(-1e30);
for(int j=0;j<uDetail;++j){vec2 p=point(id,j);low=min(low,p);high=max(high,p);vertices[id*uint(uDetail)+uint(j)]=vec4(p,recipes[id*6u+2u].x*radius(samples[j].x),0);}
for(int j=0;j<uDetail;++j){vec4 raw=vertices[id*uint(uDetail)+uint(j)];vec2 p=raw.xy;float span=high.x-low.x;p.x=span>.90?.5+(p.x-(low.x+high.x)*.5)*(.90/max(span,1e-8)):p.x+max(0.,.05-low.x)-max(0.,high.x-.95);p.y+=max(0.,.035-low.y)-max(0.,high.y-.965);p=clamp(p,vec2(.035,.030),vec2(.965,.970));vertices[id*uint(uDetail)+uint(j)]=vec4(p,raw.zw);}}
'''

def fiber_points(settings,kind,t,root_x,tip_x,length,root_y,cph,sph,cyph,syph,noise,rad_mult,overrides,j_idx,f_noise):
 if not enabled():return None
 count=len(root_x);recipes=np.zeros((count,24),np.float32)
 recipes[:,:4]=np.column_stack((root_x,tip_x,length,root_y));recipes[:,4:8]=np.column_stack((cph,sph,cyph,syph));recipes[:,8]=rad_mult
 for index,override in overrides:
  recipes[index,9]=float(override.get('offset_x',0));recipes[index,10]=float(override.get('offset_y',0));recipes[index,11]=float(override.get('bend_add',0))
  warp=override.get('curve_nodes')
  if isinstance(warp,list) and len(warp)==5:
   try:
    values=np.asarray(warp,np.float32)
    if values.shape==(5,2) and np.isfinite(values).all():recipes[index,12:22]=values.reshape(-1)
   except (ValueError,TypeError):pass
 samples=np.column_stack((t,1-f_noise,f_noise,j_idx)).astype(np.float32)
 uniforms=dict(uCount=count,uDetail=len(t),uNoiseCount=noise.shape[1],uShape=(settings.bend,settings.sag,settings.twist,settings.frizz),uCurl=(settings.curl,settings.curl_frequency,.28 if kind=='hair' else .48,0.),uWidth=(settings.taper,settings.tip_width,settings.mid_bulge,0.),uProfile=tuple(settings.width_profile[:4]),uProfileLast=settings.width_profile[4])
 try:
  data=execute('fiber_geometry',FIBER,[recipes,np.ascontiguousarray(noise,np.float32),samples],(count,len(t),4),uniforms,count)
  return np.ascontiguousarray(data[...,:2]),np.ascontiguousarray(data[...,2])
 except Exception:
  if os.environ.get('FSD_FIBERBAKE_BACKEND')=='gpu':raise
  return None
