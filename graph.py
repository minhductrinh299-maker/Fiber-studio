"""Construction Graph numeric operators on GPU; topology/keys stay deterministic."""
from copy import deepcopy
import math
import numpy as np
from .evaluation import enabled,execute

FIELD=r'''
uniform int uScalarMode,uScalarAxis,uColorMode;uniform vec4 uScalar0,uScalar1,uColor0,uColor1,uField0,uField1;
uniform int uPaletteCount;layout(std430,binding=3) readonly buffer Palette{vec4 palette[];};
float scalar(vec2 p){if(uScalarMode==1)return clamp(mix(uScalar0.x,uScalar0.y,uScalarAxis==0?p.x:p.y),0.,3.);if(uScalarMode==2)return clamp(1.-length((p-uScalar0.xy)/max(uScalar0.zw,vec2(.001))),0.,1.);return uScalar0.x;}
vec3 pigment(vec2 p){if(uColorMode>3)return uColor0.rgb;
 if(uColorMode==0){float d=length(vec2((p.x-uField0.x-uField1.x*(p.y-uField0.y))/max(.001,uField0.z),(p.y-uField0.y)/max(.001,uField0.w)));vec3 c=uColor0.rgb;for(int i=0;i<uPaletteCount;++i)if(d<palette[i].w)c=palette[i].rgb;return c;}
 float angle=radians(uField0.y);float v=p.x*cos(angle)+p.y*sin(angle);
 if(uColorMode==1)v=sin(6.28318530718*(uField0.x*v+uField0.z))>uField0.w?1.:0.;
 else if(uColorMode==2)v=clamp(v,0.,1.);
 else{v=sin((p.x*127.1+p.y*311.7)*max(.1,uField0.x)*.03+uField0.z)*43758.5453;v=clamp((fract(v)-.25)*2.,0.,1.);}
 return mix(uColor0.rgb,uColor1.rgb,v);
}
'''
SOURCE=r'''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Points{vec4 points[];};
layout(std430,binding=1) readonly buffer Headers{vec4 headers[];};
layout(std430,binding=2) readonly buffer Owners{uint owners[];};
'''+FIELD+r'''
layout(std430,binding=4) writeonly buffer Output{vec4 outputData[];};
uniform int uCount,uOp;uniform vec4 uParams;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(uCount))return;vec4 h=headers[owners[i]];float t=float(i-uint(h.x))/max(1.,h.y-1.);vec4 a=points[i*2u],b=points[i*2u+1u];
if(uOp==0){float amount=uParams.x*t*(1.-t)*4.+uParams.y*sin(6.28318530718*uParams.z*t)*sin(3.14159265359*t);a.xy+=h.zw*amount*.12;}
else if(uOp==1)a.z*=mix(clamp(uParams.y,0.,1.),1.,pow(1.-t,clamp(uParams.z,.1,6.)))*uParams.x*scalar(a.xy);
else if(uOp==2)b.rgb=clamp(mix(b.rgb,pigment(a.xy),clamp(uParams.x,0.,1.)),0.,1.);
else if(uOp==3)b.w=scalar(a.xy);
outputData[i*2u]=a;outputData[i*2u+1u]=b;}
'''
PATH=r'''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) writeonly buffer Output{vec4 values[];};
uniform int uCount;uniform vec4 uEndpoints;uniform vec4 uStyle;uniform vec3 uColor;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(uCount))return;float t=float(i)/float(uCount-1);vec2 p=mix(uEndpoints.xy,uEndpoints.zw,t)+vec2(uStyle.x*sin(3.14159265359*t),0);values[i*2u]=vec4(p,max(.03,uStyle.y)*.5,.75);values[i*2u+1u]=vec4(uColor,0);}
'''
GROW=r'''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Roots{vec4 roots[];};
layout(std430,binding=1) readonly buffer Noise{float noise[];};
layout(std430,binding=2) readonly buffer Dummy{uint dummy[];};
'''+FIELD+r'''
layout(std430,binding=4) writeonly buffer Output{vec4 values[];};
uniform int uCount,uDetail,uEnvelope;uniform vec4 uParams,uShape;uniform vec3 uColor;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(uCount))return;uint root=i/uint(uDetail);vec4 a=roots[root*2u],b=roots[root*2u+1u];float t=float(i%uint(uDetail))/float(uDetail-1);
float angle=radians(uParams.z)*b.x;vec2 direction=mat2(cos(angle),sin(angle),-sin(angle),cos(angle))*a.zw;vec2 normal=vec2(-direction.y,direction.x);
float len=clamp(uParams.x,.001,1.9)*clamp(scalar(a.xy),0.,3.);if(uEnvelope!=0)len*=max(.09,pow(sin(3.14159265359*(.04+.92*b.y)),.68));
float perturb=uShape.x*sin(3.14159265359*t)*len+uShape.y*sin(6.28318530718*uShape.z*t)*sin(3.14159265359*t)*len+uShape.w*noise[i]*sin(3.14159265359*t)*len;
vec2 xy=a.xy+len*t*direction+perturb*normal;values[i*2u]=vec4(xy,max(.03,uParams.y)*.5,b.z);values[i*2u+1u]=vec4(uColor,0);}
'''

def _field(field,color=None):
 from ..geometry import hex_rgb
 p=field.params if field else {};mode=field.mode if field else 'constant'
 if field is not None and mode not in ('constant','gradient','radial'):raise ValueError('Unknown scalar field')
 values=dict(uScalarMode={'constant':0,'gradient':1,'radial':2}.get(mode,0),uScalarAxis=0 if p.get('axis','y')=='x' else 1,uScalar0=(float(p.get('value',1.)),0.,0.,0.),uScalar1=(0.,0.,0.,0.),uColorMode=2,uColor0=(0.,0.,0.,1.),uColor1=(1.,1.,1.,1.),uField0=(0.,90.,0.,0.),uField1=(0.,0.,0.,0.),uPaletteCount=0)
 if mode=='gradient':values['uScalar0']=(float(p.get('start',0)),float(p.get('end',1)),0.,0.)
 elif mode=='radial':values['uScalar0']=(float(p.get('cx',.5)),float(p.get('cy',.5)),float(p.get('rx',.3)),float(p.get('ry',.3)))
 palette=np.zeros((1,4),np.float32)
 if color:
  p=color.params;mode=color.mode;values['uColorMode']={'eye':0,'stripes':1,'gradient':2,'noise':3}.get(mode,4)
  values['uColor0']=tuple(hex_rgb(p.get('color_a','#b99764')))+(1.,);values['uColor1']=tuple(hex_rgb(p.get('color_b','#40bbd0')))+(1.,)
  values['uField0']=(float(p.get('frequency',7)),float(p.get('angle',90 if mode=='gradient' else 0)),float(p.get('phase',0)),float(p.get('threshold',0)))
  if mode=='eye':
   values['uField0']=(float(p.get('center_x',.5)),float(p.get('center_y',.36)),float(p.get('radius_x',.32)),float(p.get('radius_y',.24)));values['uField1']=(float(p.get('skew',0)),0.,0.,0.)
   palette=np.asarray([(*hex_rgb(c),float(edge)) for c,edge in zip(p.get('palette',['#b47c36','#06a7a2','#009bdd','#20165a']),p.get('rings',[1.,.76,.54,.30]))],np.float32).reshape(-1,4);values['uPaletteCount']=len(palette)
 return values,palette

def evaluate(op,p,inputs,seed,nodeid):
 if not enabled() or op not in ('path','grow','deform','taper','paint','filter','scatter','along','instance'):return None
 from ..construction.core import Curve,CurveSet
 from ..geometry import hex_rgb
 from ..model import stable_int
 try:
  if op in ('scatter','along','instance'):
   from .graph_topology import evaluate as topology
   return topology(op,p,inputs,seed,nodeid)
  if op=='path':
   n=max(4,min(160,int(p.get('points',24))));rgb=hex_rgb(p.get('color','#a98857'))
   data=execute('graph_path',PATH,[],(n,8),dict(uCount=n,uEndpoints=(float(p.get('x0',.5)),float(p.get('y0',.94)),float(p.get('x1',.5)),float(p.get('y1',.08))),uStyle=(float(p.get('bend',0)),float(p.get('width',3.2)),0.,0.),uColor=tuple(rgb)),n)
   return CurveSet([Curve(nodeid+':path',data[:,:2],data[:,2],data[:,4:7],.75,group='guide')])
  if op=='grow':
   points=inputs['points'].points;n=max(5,min(96,int(p.get('points_per_curve',12))))
   if not points:return CurveSet([])
   roots=np.asarray([((q.x,q.y,*q.tangent),(q.side,q.u,q.depth,0)) for q in points],np.float32).reshape(-1,4)
   noise=np.stack([np.random.default_rng(stable_int(seed,nodeid,q.key,int(p.get('seed',3)))).normal(0,.15,n) for q in points]).astype(np.float32)
   uniforms,palette=_field(inputs.get('length_field'));uniforms.update(uCount=len(points)*n,uDetail=n,uEnvelope=int(p.get('envelope','none')=='feather'),uParams=(float(p.get('length',.28)),float(p.get('width',1.7)),float(p.get('angle',66)),0.),uShape=(float(p.get('bend',.04)),float(p.get('curl',0)),float(p.get('waves',2)),float(p.get('frizz',0))),uColor=tuple(hex_rgb(p.get('color','#9b825f'))))
   data=execute('graph_grow',GROW,[roots,noise,np.zeros(1,np.uint32),palette],(len(points)*n,8),uniforms,len(points)*n)
   return CurveSet([Curve(f'{nodeid}:{q.key}',data[i*n:(i+1)*n,:2],data[i*n:(i+1)*n,2],data[i*n:(i+1)*n,4:7],q.depth,group=p.get('group','fiber')) for i,q in enumerate(points)])
  curves=inputs['curves'].curves
  if not curves:return CurveSet([])
  counts=np.asarray([len(c.coords) for c in curves],np.int32);offsets=np.r_[0,np.cumsum(counts)];data=np.zeros((int(offsets[-1]),8),np.float32);headers=[]
  for i,c in enumerate(curves):
   sl=slice(offsets[i],offsets[i+1]);data[sl,:2]=c.coords;data[sl,2]=c.radii;data[sl,3]=c.depth;data[sl,4:7]=c.colors
   direction=c.coords[-1]-c.coords[0];direction/=max(1e-7,float(np.linalg.norm(direction)));headers.append((offsets[i],counts[i],-direction[1],direction[0]))
  uniforms,palette=_field(inputs.get('width_field') if op=='taper' else inputs.get('mask'),inputs.get('color'))
  uniforms.update(uCount=len(data),uOp={'deform':0,'taper':1,'paint':2,'filter':3}[op],uParams=(0.,0.,0.,0.))
  if op=='deform':uniforms['uParams']=(float(p.get('bend',0)),float(p.get('curl',0)),float(p.get('waves',3)),0.)
  elif op=='taper':uniforms['uParams']=(float(p.get('root_scale',1)),float(p.get('tip_ratio',.07)),float(p.get('power',1.3)),0.)
  elif op=='paint':uniforms['uParams']=(float(p.get('strength',1)),0.,0.,0.)
  if op=='filter' and inputs.get('mask') is None:return inputs['curves']
  result=execute('graph_curve_ops',SOURCE,[data,np.asarray(headers,np.float32),np.repeat(np.arange(len(curves),dtype=np.uint32),counts),palette],data.shape,uniforms,len(data));out=[]
  for i,c in enumerate(curves):
   if op=='filter':
    value=result[offsets[i]+len(c.coords)//2,7]
    if (value>=float(p.get('threshold',.5)))==bool(p.get('invert',False)):continue
   n=deepcopy(c);sl=slice(offsets[i],offsets[i+1]);n.coords=result[sl,:2].copy();n.radii=result[sl,2].copy();n.colors=result[sl,4:7].copy();out.append(n)
  return CurveSet(out)
 except Exception as error:
  from ..construction.core import GraphError
  if isinstance(error,GraphError):raise
  import os
  if os.environ.get('FSD_FIBERBAKE_BACKEND')=='gpu':raise
  return None
