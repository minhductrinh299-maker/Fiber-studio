"""GPU topology-preserving distribution and instancing for Construction Graph."""
from copy import deepcopy
import math
import numpy as np
from .evaluation import execute

SCATTER=r'''#version 430 core
layout(local_size_x=64) in;layout(std430,binding=0) readonly buffer Random{vec4 randoms[];};layout(std430,binding=1) writeonly buffer Points{vec4 points[];};
uniform int uCount;uniform vec4 uBounds;uniform float uAngle;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(uCount))return;vec4 r=randoms[i];points[i*2u]=vec4(mix(uBounds.x,uBounds.y,r.x),mix(uBounds.z,uBounds.w,r.y),sin(uAngle),-cos(uAngle));points[i*2u+1u]=vec4(1,float(i)/max(1.,float(uCount)),r.z,0);}
'''
ALONG=r'''#version 430 core
layout(local_size_x=64) in;layout(std430,binding=0) readonly buffer Geometry{vec4 vertices[];};layout(std430,binding=1) readonly buffer Recipes{vec4 recipes[];};layout(std430,binding=2) writeonly buffer Points{vec4 points[];};uniform int uCount;uniform float uOffset;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(uCount))return;vec4 h=recipes[i];float at=h.z*(h.y-1.);uint a=min(uint(h.y)-2u,uint(at));float f=at-float(a);vec4 p=vertices[uint(h.x)+a],q=vertices[uint(h.x)+a+1u];vec2 tangent=q.xy-p.xy;tangent/=max(1e-7,length(tangent));vec2 xy=mix(p.xy,q.xy,f)+vec2(-tangent.y,tangent.x)*uOffset*h.w;points[i*2u]=vec4(xy,tangent);points[i*2u+1u]=vec4(h.w,h.z,p.z,0);}
'''
INSTANCE=r'''#version 430 core
layout(local_size_x=64) in;layout(std430,binding=0) readonly buffer Geometry{vec4 vertices[];};layout(std430,binding=1) readonly buffer Recipes{vec4 recipes[];};layout(std430,binding=2) readonly buffer Owners{uint owners[];};layout(std430,binding=3) writeonly buffer Output{vec4 outputData[];};uniform int uCount;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(uCount))return;uint id=owners[i];vec4 h=recipes[id*3u],target=recipes[id*3u+1u],style=recipes[id*3u+2u];uint j=i-uint(h.z),first=uint(h.x);vec4 a=vertices[(first+j)*2u],b=vertices[(first+j)*2u+1u],root=vertices[first*2u],tip=vertices[(first+uint(h.y)-1u)*2u];vec2 direction=tip.xy-root.xy;direction/=max(1e-7,length(direction));float angle=atan(target.w,target.z)-atan(direction.y,direction.x);mat2 rotation=mat2(cos(angle),sin(angle),-sin(angle),cos(angle));a.xy=rotation*(a.xy-root.xy)*style.x+target.xy;a.z*=style.x;if(style.z>.5)a.w=(a.w-root.w)*style.x+style.y;outputData[i*2u]=a;outputData[i*2u+1u]=b;}
'''

def evaluate(op,p,inputs,seed,nodeid):
 from ..construction.core import Point,PointSet,CurveSet,MAX_CURVES,MAX_POINTS,MAX_VERTICES,GraphError
 from ..model import stable_int
 if op=='scatter':
  count=max(1,min(MAX_POINTS,int(p.get('count',160))));x0,x1=sorted((float(p.get('x_min',.1)),float(p.get('x_max',.9))));y0,y1=sorted((float(p.get('y_min',.85)),float(p.get('y_max',.95))))
  randoms=np.empty((count,4),np.float32)
  for i in range(count):
   rng=np.random.default_rng(stable_int(seed,nodeid,'scatter',i,int(p.get('seed',1))));randoms[i]=(rng.random(),rng.random(),rng.uniform(.15,.85),0)
  data=execute('graph_scatter',SCATTER,[randoms],(count,8),dict(uCount=count,uBounds=(x0,x1,y0,y1),uAngle=math.radians(float(p.get('angle',0)))),count)
  return PointSet([Point(f'{nodeid}:s{i}',float(row[0]),float(row[1]),tuple(row[2:4]),1,float(row[5]),float(row[6])) for i,row in enumerate(data)])
 if op=='along':
  curves=inputs['curves'].curves;count=max(1,min(1500,int(p.get('count',12))));low=max(0,min(1,float(p.get('start',.06))));high=max(0,min(1,float(p.get('end',.91))));side=p.get('side','both');sides=(-1,1) if side=='both' else (-1,) if side=='left' else (1,)
  if len(curves)*count*len(sides)>MAX_POINTS:raise GraphError('Too many distributed points')
  recipes=[];geometry=[];keys=[];offset=0
  for curve in curves:
   geometry.append(np.column_stack((curve.coords,np.full(len(curve.coords),curve.depth),np.zeros(len(curve.coords)))))
   for j in range(count):
    u=low+(high-low)*(j+.5)/count
    for sd in sides:recipes.append((offset,len(curve.coords),u,sd));keys.append(f'{nodeid}:{curve.key}:a{j}:{sd}')
   offset+=len(curve.coords)
  if len(keys)>MAX_POINTS:raise GraphError('Too many distributed points')
  if not keys:return PointSet([])
  data=execute('graph_along',ALONG,[np.concatenate(geometry).astype(np.float32),np.asarray(recipes,np.float32)],(len(keys),8),dict(uCount=len(keys),uOffset=float(p.get('offset',0))),len(keys))
  return PointSet([Point(key,float(row[0]),float(row[1]),tuple(row[2:4]),int(row[4]),float(row[5]),float(row[6])) for key,row in zip(keys,data)])
 if op=='instance':
  curves=inputs['curves'].curves;points=inputs['points'].points;limit=max(1,min(MAX_CURVES,int(p.get('limit',120))));scale=float(p.get('scale',.5))
  if not curves or not points:return CurveSet([])
  offsets=np.r_[0,np.cumsum([len(c.coords) for c in curves])];total=int(offsets[-1]);source=np.zeros((total,8),np.float32)
  for i,c in enumerate(curves):
   sl=slice(offsets[i],offsets[i+1]);source[sl,:2]=c.coords;source[sl,2]=c.radii;source[sl,3]=c.depth_profile if c.depth_profile is not None else c.depth;source[sl,4:7]=c.colors
  recipes=[];meta=[];count=0
  for j,q in enumerate(points):
   for i,c in enumerate(curves):
    recipes.append(((offsets[i],len(c.coords),count,0),(q.x,q.y,*q.tangent),(scale,q.depth,int(c.depth_profile is not None),0)));meta.append((j,c,count));count+=len(c.coords)
    if len(meta)>=limit:break
   if len(meta)>=limit:break
  if count>MAX_VERTICES:raise GraphError('Exceeded vertex budget')
  owners=np.repeat(np.arange(len(meta),dtype=np.uint32),[len(c.coords) for _,c,_ in meta])
  result=execute('graph_instance',INSTANCE,[source,np.asarray(recipes,np.float32).reshape(-1,4),owners],(count,8),dict(uCount=count),count);out=[]
  for j,c,start in meta:
   n=deepcopy(c);data=result[start:start+len(c.coords)];n.coords=data[:,:2].copy();n.radii=data[:,2].copy();n.colors=data[:,4:7].copy();n.key=f'{nodeid}:{j}:{c.key}'
   if c.depth_profile is not None:n.depth_profile=data[:,3].copy()
   out.append(n)
  return CurveSet(out)
 return None
