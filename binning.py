"""Deterministic GPU sparse bin construction: scan -> emit -> merge sort -> compact.

No atomics choose strand order. Every producer has an exclusive prefix range;
lexicographic sorting and stable compaction recover canonical painter order.
"""
import ctypes as C
import math
import numpy as np

HEADER='#version 430 core\nlayout(local_size_x=256) in;\n'
SCAN=HEADER+r'''
layout(std430,binding=0) readonly buffer Input{uint inputData[];};
layout(std430,binding=1) writeonly buffer Output{uint outputData[];};
layout(std430,binding=2) writeonly buffer Sum{uint sums[];};
uniform uint uCount;shared uint temp[256];
void main(){uint lane=gl_LocalInvocationID.x,i=gl_GlobalInvocationID.x;uint v=i<uCount?inputData[i]:0u;temp[lane]=v;barrier();
for(uint step=1u;step<256u;step<<=1u){uint add=lane>=step?temp[lane-step]:0u;barrier();temp[lane]+=add;barrier();}
if(i<uCount)outputData[i]=temp[lane]-v;if(lane==255u)sums[gl_WorkGroupID.x]=temp[lane];}'''
ADD=HEADER+r'''
layout(std430,binding=0) buffer Data{uint data[];};layout(std430,binding=1) readonly buffer Offsets{uint offsets[];};uniform uint uCount;
void main(){uint i=gl_GlobalInvocationID.x;if(i<uCount)data[i]+=offsets[i/256u];}'''
SEGMENTS=r'''
struct Segment{vec4 ab;vec4 rt;};layout(std430,binding=0) readonly buffer Segments{Segment segments[];};
uniform uint uCount;uniform ivec2 uGrid;uniform ivec2 uSize;uniform float uHalo;
ivec4 extent(uint i){Segment s=segments[i];float pad=max(s.rt.x,s.rt.y)+uHalo;
ivec2 lo=max(ivec2(0),ivec2(floor((min(s.ab.xy,s.ab.zw)-pad)/32.)));
ivec2 hi=min(uGrid,ivec2(ceil((max(s.ab.xy,s.ab.zw)+pad)/32.)));
return ivec4(min(lo,uGrid),max(hi,ivec2(0)));}
'''
COUNT=HEADER+SEGMENTS+r'''
layout(std430,binding=1) writeonly buffer Counts{uint counts[];};
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uCount)return;ivec4 r=extent(i);counts[i]=uint(max(0,r.z-r.x)*max(0,r.w-r.y));}'''
INITIALIZE=HEADER+r'''
layout(std430,binding=0) writeonly buffer Pairs{uvec4 pairs[];};uniform uint uCount;
void main(){uint i=gl_GlobalInvocationID.x;if(i<uCount)pairs[i]=uvec4(0xffffffffu);}'''
EMIT=HEADER+SEGMENTS+r'''
layout(std430,binding=1) readonly buffer Offsets{uint offsets[];};
layout(std430,binding=2) readonly buffer Owners{uvec2 owners[];};
layout(std430,binding=3) writeonly buffer Pairs{uvec4 pairs[];};uniform uint uCapacity;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uCount)return;ivec4 r=extent(i);uint target=offsets[i];uvec2 owner=owners[i];
for(int y=r.y;y<r.w;++y)for(int x=r.x;x<r.z;++x){if(target<uCapacity)pairs[target]=uvec4(uint(y*uGrid.x+x),owner.x,owner.y,owner.y);++target;}}'''
COMPARE=r'''
bool lessKey(uvec4 a,uvec4 b){return a.x<b.x||(a.x==b.x&&(a.y<b.y||(a.y==b.y&&a.z<b.z)));}
'''
LOCAL_SORT=HEADER+COMPARE+r'''
layout(std430,binding=0) buffer Pairs{uvec4 pairs[];};uniform uint uCount;shared uvec4 values[256];
void main(){uint lane=gl_LocalInvocationID.x,i=gl_GlobalInvocationID.x;values[lane]=i<uCount?pairs[i]:uvec4(0xffffffffu);barrier();
for(uint width=2u;width<=256u;width<<=1u)for(uint stride=width>>1u;stride>0u;stride>>=1u){uvec4 a=values[lane],b=values[lane^stride];bool low=((lane&width)==0u)==((lane&stride)==0u);uvec4 next=low?(lessKey(b,a)?b:a):(lessKey(a,b)?b:a);barrier();values[lane]=next;barrier();}
if(i<uCount)pairs[i]=values[lane];}'''
MERGE=HEADER+COMPARE+r'''
layout(std430,binding=0) readonly buffer Input{uvec4 source[];};layout(std430,binding=1) writeonly buffer Output{uvec4 target[];};uniform uint uCount;uniform uint uWidth;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uCount)return;uint base=(i/(2u*uWidth))*(2u*uWidth);bool left=i-base<uWidth;
uint other=left?base+uWidth:base;uint end=min(other+uWidth,uCount);uint lo=other,hi=end;uvec4 value=source[i];
while(lo<hi){uint mid=(lo+hi)/2u;bool before=lessKey(source[mid],value)||(!left&&!lessKey(value,source[mid]));if(before)lo=mid+1u;else hi=mid;}
uint own=i-(left?base:base+uWidth);target[base+own+lo-other]=value;}'''
MARK=HEADER+r'''
layout(std430,binding=0) readonly buffer Pairs{uvec4 pairs[];};layout(std430,binding=1) writeonly buffer Marks{uint marks[];};uniform uint uCount;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uCount)return;uvec4 p=pairs[i];marks[i]=(p.x!=0xffffffffu&&(i==0u||any(notEqual(p.xy,pairs[i-1u].xy))))?1u:0u;}'''
BLOCKS=HEADER+r'''
layout(std430,binding=0) writeonly buffer Blocks{uvec4 blocks[];};uniform uint uCount;uniform uint uColumns;
void main(){uint i=gl_GlobalInvocationID.x;if(i<uCount)blocks[i]=uvec4(i%uColumns,i/uColumns,0,0);}'''
COMPACT=HEADER+r'''
layout(std430,binding=0) readonly buffer Pairs{uvec4 pairs[];};layout(std430,binding=1) readonly buffer Marks{uint marks[];};
layout(std430,binding=2) readonly buffer Positions{uint positions[];};layout(std430,binding=3) writeonly buffer Index{uvec2 indices[];};
layout(std430,binding=4) buffer Blocks{uvec4 blocks[];};layout(std430,binding=5) readonly buffer Starts{uint starts[];};uniform uint uCount;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uCount||marks[i]==0u)return;uvec4 p=pairs[i];uint last=p.z;
uint next=i+1u;while(next<uCount&&all(equal(pairs[next].xy,p.xy))){last=max(last,pairs[next].z);++next;}
uint position=positions[i],relative=p.z-starts[p.y];indices[position]=uvec2(p.y,relative|((last-p.z+1u)<<16u));
if(i==0u||pairs[i-1u].x!=p.x)blocks[p.x].z=position;
atomicAdd(blocks[p.x].w,1u);}'''

class GPUBins:
 def __init__(self,gpu,entry,keep):
  self.gpu=gpu;self.entry=entry;self.keep=keep
  if not hasattr(entry,'bin_buffers'):entry.bin_buffers={};entry.bin_capacities={}
  if not hasattr(gpu,'bin_programs'):gpu.bin_programs={}
 def buffer(self,key,bytes):
  gpu=self.gpu;g=gpu.g;amount=max(32,int(bytes));old=self.entry.bin_capacities.get(key,0)
  if key not in self.entry.bin_buffers:self.entry.bin_buffers[key]=g.name('Buffers')
  name=self.entry.bin_buffers[key];g.glBindBuffer(0x90D2,name)
  if amount>old:
   capacity=max(4096,1<<(amount-1).bit_length());gpu._reserve(capacity-old,self.keep);g.glBufferData(0x90D2,capacity,None,0x88E8);gpu.used+=capacity-old;self.entry.bin_capacities[key]=capacity
  return name
 def program(self,name,source):
  programs=self.gpu.bin_programs
  if name not in programs:programs[name]=self.gpu.compile(source)
  return programs[name]
 def dispatch(self,name,source,count,bindings,uniforms):
  gpu=self.gpu;g=gpu.g;program=self.program(name,source);g.glUseProgram(program)
  for slot,buffer in enumerate(bindings):gpu.f['glBindBufferBase'](0x90D2,slot,buffer)
  set2=getattr(C,'WINFUNCTYPE',C.CFUNCTYPE)(None,C.c_int,C.c_int,C.c_int)(gpu.context.get_proc('glUniform2i'))
  for key,value in {'uCount':int(count),**uniforms}.items():
   location=g.glGetUniformLocation(program,key.encode())
   if isinstance(value,tuple):set2(location,*value)
   elif isinstance(value,float):g.glUniform1f(location,value)
   else:gpu.f['glUniform1ui'](location,int(value))
  if count:gpu.f['glDispatchCompute']((count+255)//256,1,1)
  gpu.f['glMemoryBarrier'](0x2000|0x200)
 def scan(self,source,target,count,level=0):
  groups=(count+255)//256;sums=self.buffer(f'scan_sums_{level}',groups*4)
  self.dispatch('scan',SCAN,count,[source,target,sums],{})
  if groups>1:
   offsets=self.buffer(f'scan_offsets_{level}',groups*4);self.scan(sums,offsets,groups,level+1)
   self.dispatch('scan_add',ADD,count,[target,offsets],{})
 def build(self,segs,starts,counts,size,halo,cancel):
  gpu=self.gpu;g=gpu.g;n=len(segs);columns=(size[0]+31)//32;rows=(size[1]+31)//32
  # Host computes only an allocation bound; all bin membership/order/intervals
  # are produced on GPU. Padding protects float32/float64 boundary rounding.
  ab=segs['ab'];pad=np.maximum(segs['rt'][:,0],segs['rt'][:,1])+halo
  # Expand float32 endpoints by one ULP before quantizing. This bounds GPU
  # rounding at block edges without padding EVERY segment by two whole bins.
  lower=np.nextafter(np.minimum(ab[:,:2],ab[:,2:])-pad[:,None],np.float32(-np.inf))
  upper=np.nextafter(np.maximum(ab[:,:2],ab[:,2:])+pad[:,None],np.float32(np.inf))
  lo=np.maximum(0,np.floor(lower/32)).astype(np.int64)
  hi=np.minimum((columns,rows),np.ceil(upper/32)).astype(np.int64)
  extent=np.maximum(0,hi-lo);bound=int(np.sum(extent[:,0]*extent[:,1]))+1
  if bound>0x7fffffff or (len(counts) and int(np.max(counts))>65535):raise MemoryError('GPU bin index capacity exceeded')
  capacity=max(256,1<<(max(1,bound)-1).bit_length())
  # Two sorting buffers + compact records + scans stay in the global budget.
  pairs0=self.buffer('pairs0',capacity*16);pairs1=self.buffer('pairs1',capacity*16)
  marks=self.buffer('marks',capacity*4);positions=self.buffer('positions',capacity*4)
  segment_counts=self.buffer('counts',n*4);offsets=self.buffer('offsets',n*4)
  owner_buffer=self.buffer('owners',n*8);start_buffer=self.buffer('starts',len(starts)*4)
  owner=np.repeat(np.arange(len(counts),dtype=np.uint32),counts.astype(np.int64))
  local=np.arange(n,dtype=np.uint32);owners=np.column_stack((owner,local)).astype(np.uint32)
  for name,array in ((owner_buffer,owners),(start_buffer,np.asarray(starts,np.uint32))):
   g.glBindBuffer(0x90D2,name)
   if array.nbytes:g.glBufferSubData(0x90D2,0,array.nbytes,array.ctypes.data)
  uniforms=dict(uGrid=(columns,rows),uSize=tuple(size),uHalo=float(halo))
  self.dispatch('count',COUNT,n,[self.entry.buffers[1],segment_counts],uniforms);self.scan(segment_counts,offsets,n)
  self.dispatch('initialize',INITIALIZE,capacity,[pairs0],{})
  self.dispatch('emit',EMIT,n,[self.entry.buffers[1],offsets,owner_buffer,pairs0],{**uniforms,'uCapacity':capacity})
  self.dispatch('local_sort',LOCAL_SORT,capacity,[pairs0],{})
  width=256;source,target=pairs0,pairs1
  while width<capacity:
   if cancel():gpu.wait();raise RuntimeError('Render cancelled')
   self.dispatch('merge',MERGE,capacity,[source,target],dict(uWidth=width));source,target=target,source;width*=2
  self.dispatch('mark',MARK,capacity,[source,marks],{});self.scan(marks,positions,capacity)
  index=self.buffer('indices',capacity*8);blocks=self.buffer('blocks',columns*rows*16)
  self.dispatch('blocks',BLOCKS,columns*rows,[blocks],dict(uColumns=columns))
  self.dispatch('compact',COMPACT,capacity,[source,marks,positions,index,blocks,start_buffer],{})
  return blocks,index,columns,rows
