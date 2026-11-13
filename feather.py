"""GPU feather/rachis/barb/barbule evaluation with unchanged source RNG streams."""
import math,os
import numpy as np
from .evaluation import enabled,execute

SOURCE=r'''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Recipes{vec4 recipes[];};
layout(std430,binding=1) buffer Output{vec4 vertices[];};
uniform int uCount,uShaftDetail,uBarbDetail;uniform vec4 uShape,uStyle,uProfile;uniform float uProfileLast;
float profile(float t){float f=t*4.;int a=min(3,int(f));float v=a==0?uProfile.x:a==1?uProfile.y:a==2?uProfile.z:uProfile.w;float w=a==0?uProfile.y:a==1?uProfile.z:a==2?uProfile.w:uProfileLast;return mix(v,w,f-float(a));}
float widthAt(float t,float taper,float tip,float bulge){return max(0.,(tip+(1.-tip)*pow(1.-t,taper))*max(.02,1.+bulge*4.*t*(1.-t)));}
vec2 shaft(float t){return vec2(.5+uShape.x*.12*sin(3.14159265359*t)*t,.95-uShape.y*t);}
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(uCount))return;
if(i==0u){for(int j=0;j<uShaftDetail;++j){float t=float(j)/float(uShaftDetail-1);vertices[j]=vec4(shaft(t),widthAt(t,.85,.08,.1)*max(.5,uStyle.x*.70)*profile(t),0);}return;}
uint r=(i-1u)*6u;vec4 header=recipes[r],source=recipes[r+1u],extra=recipes[r+2u];float u=source.x,side=source.y;
float f=u*float(uShaftDetail-1);int a=min(uShaftDetail-2,int(f));vec2 base=mix(shaft(float(a)/float(uShaftDetail-1)),shaft(float(a+1)/float(uShaftDetail-1)),f-float(a));
float width=uShape.z*.43*pow(sin(3.14159265359*u),.85)*(1.+side*uShape.w*.5)*source.z;if(u>.85)width*=max(.12,(1.-u)/.15);
vec2 tip=clamp(base+vec2(side*width,-uShape.y*(.07+.08*(1.-u))),vec2(.04),vec2(.96));float thick=max(.45,uStyle.x*.28)*extra.z;
for(int j=0;j<uBarbDetail;++j){float t=float(j)/float(uBarbDetail-1);vec2 p=clamp(base+(tip-base)*vec2(pow(t,.86),t),vec2(.035,.030),vec2(.965,.970));p.x=clamp(p.x+side*uStyle.y*.015*sin(t*9.42477796077+source.w)*pow(t,1.5),.035,.965);
float k=t*4.;int lo=min(3,int(k));vec2 wa=vec2(recipes[r+3u+uint(lo/2)][(lo%2)*2],recipes[r+3u+uint(lo/2)][(lo%2)*2+1]);int hi=lo+1;vec2 wb=hi==4?recipes[r+5u].xy:vec2(recipes[r+3u+uint(hi/2)][(hi%2)*2],recipes[r+3u+uint(hi/2)][(hi%2)*2+1]);p=clamp(p+mix(wa,wb,k-float(lo)),vec2(.035,.030),vec2(.965,.970));p=clamp(p+extra.xy,vec2(.035,.030),vec2(.965,.970));
vertices[uint(header.x)+uint(j)]=vec4(p,widthAt(t,.75,.01,.1)*profile(t)*thick,0);}
for(int j=0;j<int(header.w);++j){float bu=.20+.60*(float(j)+.5)/max(1.,header.w);float at=bu*float(uBarbDetail-1);int lo=min(uBarbDetail-2,int(at));vec2 p=mix(vertices[uint(header.x)+uint(lo)].xy,vertices[uint(header.x)+uint(lo+1)].xy,at-float(lo));float radius=max(.22,thick*.38);uint offset=uint(header.z)+uint(j)*2u;vertices[offset]=vec4(p,radius,0);vertices[offset+1u]=vec4(p+vec2(side*width*.09,-uShape.y*.008),radius*.01,0);}
}'''

def generate(tile,detail=24,cancel=None,skip_color=False):
 if not enabled():return None
 from ..geometry import Strand,hex_rgb,color_jitter,_vdc
 from ..model import stable_int
 s=tile.settings.sanitize();root,mid,tip=hex_rgb(s.color_root),hex_rgb(s.color_mid),hex_rgb(s.color_tip)
 shaft_count=max(12,int(detail));barb_count=max(9,int(detail)//2);offset=shaft_count;recipes=[];meta=[]
 for index in range(1,s.count+1):
  if cancel and index%32==0 and cancel():raise RuntimeError('Render cancelled')
  override=tile.overrides.get(str(index),{})
  if override.get('hidden',False):continue
  rng=np.random.default_rng(stable_int(tile.seed,'feather',index));side=-1 if index%2==0 else 1;u=float(np.clip(.11+.80*_vdc((index-1)//2+1),.04,.97));scale=float(np.clip(1+float(rng.normal(0,s.length_variation*.18)),.6,1.5));phase=float(rng.random())
  if skip_color:rng.normal(0,1,9);colors=(root,mid,tip)
  else:colors=tuple(color_jitter(c,rng,s.hue_variation,s.saturation_variation,s.value_variation) for c in (root,mid,tip))
  if override.get('color'):
   try:colors=(hex_rgb(override['color']),)*3
   except (ValueError,IndexError):pass
  depth=.1+float(rng.random())*.7;pattern=float(rng.random());recipe=np.zeros(24,np.float32)
  recipe[:4]=(offset,barb_count,offset+barb_count,s.feather_barbules);recipe[4:8]=(u,side,scale,phase);recipe[8:12]=(float(override.get('offset_x',0)),float(override.get('offset_y',0)),float(override.get('width_scale',1)),0)
  warp=override.get('curve_nodes')
  if isinstance(warp,list) and len(warp)==5:
   try:
    value=np.asarray(warp,np.float32)
    if value.shape==(5,2) and np.isfinite(value).all():recipe[12:22]=value.reshape(-1)
   except (ValueError,TypeError):pass
  recipes.append(recipe);meta.append((index,offset,colors,depth,pattern));offset+=barb_count+2*s.feather_barbules
 try:
  data=execute('feather_geometry',SOURCE,[np.asarray(recipes,np.float32).reshape(-1,24)],(offset,4),dict(uCount=len(recipes)+1,uShaftDetail=shaft_count,uBarbDetail=barb_count,uShape=(s.bend,min(.90,s.length),s.feather_width,s.feather_asymmetry),uStyle=(s.root_width,s.frizz,0.,0.),uProfile=tuple(s.width_profile[:4]),uProfileLast=s.width_profile[4]),len(recipes)+1)
  output=[Strand(f'{tile.id}:0',data[:shaft_count,:2].copy(),data[:shaft_count,2].copy(),root,mid,tip,.92,s.roughness,True,'rachis',opacity=s.strand_opacity)]
  for index,start,colors,depth,pattern in meta:
   output.append(Strand(f'{tile.id}:{index}',data[start:start+barb_count,:2].copy(),data[start:start+barb_count,2].copy(),*colors,depth,s.roughness,False,'barb',pattern,opacity=s.strand_opacity))
   for j in range(s.feather_barbules):
    a=start+barb_count+j*2;output.append(Strand(f'{tile.id}:{index}:{j}',data[a:a+2,:2].copy(),data[a:a+2,2].copy(),*colors,.04,s.roughness,False,'barbule',opacity=s.strand_opacity))
  return output
 except Exception:
  if os.environ.get('FSD_FIBERBAKE_BACKEND')=='gpu':raise
  return None
