"""Coverage quality and continuous-depth AO evaluated from source geometry."""
import re

AO_FUNCTION=r'''
float geometryWeights[32];float geometryMoments[32];
float sourceOcclusion(vec2 p,float receiverDepth,uint selfId,uint start,uint count){
 StrandGPU self=strands[selfId];float own=0.,ownDepth=0.,ownOpacity=0.;
 for(uint j=0u;j<self.ids.y;++j){uint index=self.ids.x+j;SegGPU seg=segs[index];vec2 d=seg.ab.zw-seg.ab.xy;float len2=dot(d,d);if(len2<1e-10)continue;
  float t=clamp(dot(p-seg.ab.xy,d)/len2,0.,1.);float prox=clamp(1.-max(0.,length(p-seg.ab.xy-t*d)-mix(seg.rt.x,seg.rt.y,t))/uHalo,0.,1.);
  if(prox>own){own=prox;ownDepth=mix(attrs[index*4u+1u].w,attrs[index*4u+3u].w,t);ownOpacity=mix(attrs[index*4u].x,attrs[index*4u+2u].x,t);}
 }
 // The receiver is removed from its moment bucket; no self-occlusion shortcut
 // or alpha-derived blur is used. 32 source-depth intervals bound shader work.
 int ownBin=clamp(int((ownDepth-uDepthRange.x)/uDepthRange.y*32.),0,31);float optical=0.;
 for(int z=0;z<32;++z){float w=max(0.,geometryWeights[z]-(z==ownBin?own*ownOpacity:0.));if(w<1e-8)continue;
  float m=geometryMoments[z]-(z==ownBin?own*ownOpacity*ownDepth:0.);float center=m/w;
  float radius=max(0.,min(.5*uDepthRange.y/32.,min(center-(uDepthRange.x+float(z)*uDepthRange.y/32.),uDepthRange.x+float(z+1)*uDepthRange.y/32.-center)));
  float gap=center-receiverDepth;float v=radius<1e-8?max(0.,gap):gap>=radius?gap:gap<=-radius?0.:(radius+gap)*(radius+gap)/(4.*radius);
  optical+=w*v;
 }
 return exp(-.45*optical);
}
''' 

def extend(source,channels=17):
    # Accumulate floating source contributions before final quantization.
    source=source.replace('void main() {', '''uniform int uSamples;uniform int uExactAO;uniform vec2 uDepthRange;
vec4 sampleAccum[16];vec4 sampleBackground[16];float coverageAccum;float currentCoverage;int sampleNumber;
uint idSamples[48];
void storeSample(int channel,vec4 value){sampleBackground[channel]=value;
 if(channel==1){sampleAccum[channel]+=value;return;}
 if(channel==2||channel==3)sampleAccum[channel]+=vec4((value.rgb*2.-1.)*currentCoverage,currentCoverage);
 else sampleAccum[channel]+=vec4(value.rgb*currentCoverage,currentCoverage);
}
void storeIdentity(int channel,uint value){int index=channel==6?0:channel==7?1:2;idSamples[index*16+sampleNumber]=value;}
'''+AO_FUNCTION+'\nvoid evaluateSample(vec2 offset) {',1)
    source=source.replace('float ao_buckets[8];','float ao_buckets[8]; for(int z=0;z<32;++z){geometryWeights[z]=0.;geometryMoments[z]=0.;}')
    source=source.replace('ao_buckets[zbin] += prox * near_op;','ao_buckets[zbin] += prox * near_op; int gz=clamp(int((near_depth-uDepthRange.x)/uDepthRange.y*32.),0,31);geometryWeights[gz]+=prox*near_op;geometryMoments[gz]+=prox*near_op*near_depth;')
    source=source.replace('vec2(float(px) + 0.5, float(py) + 0.5)','vec2(float(px),float(py))+offset')
    source=source.replace('bool empty = (weight <= 1e-6);','currentCoverage=weight;coverageAccum+=weight;bool empty = (weight <= 1e-6);')
    pattern=r'out_pixels\[uint\(uChannelSlots\[(\d+)\]\)\]\s*\*\s*npix\s*\+\s*pidx\]\s*=\s*pack_byte4\(([^;]+)\);'
    source=re.sub(pattern,lambda m:f'storeSample({m[1]},vec4({m[2]}));',source)
    pattern=r'out_pixels\[uint\(uChannelSlots\[(\d+)\]\)\]\s*\*\s*npix\s*\+\s*pidx\]\s*=\s*pack_u24_rgb\(([^;]+)\);'
    source=re.sub(pattern,lambda m:f'storeIdentity({m[1]},{m[2]});',source)
    source=source.replace('float ao_s = exp(-0.45 * optical);','float ao_s=uExactAO!=0?sourceOcclusion(p,a1.w,s_idx,bin_start,bin_count):exp(-0.45*optical);')
    source+=r'''
void main(){
 uint offset=gl_WorkGroupID.x>>2u;if(int(offset)>=uBlockCount)return;uvec4 block=blocks[uBlockBase+offset];uint quad=gl_WorkGroupID.x&3u;
 uint x=block.x*32u+(quad&1u)*16u+gl_LocalInvocationID.x,y=block.y*32u+(quad>>1u)*16u+gl_LocalInvocationID.y;
 if(x>=uint(uWidth)||y>=uint(uHeight))return;
 for(int c=0;c<16;++c)sampleAccum[c]=vec4(0);for(int k=0;k<48;++k)idSamples[k]=0u;
 coverageAccum=0.;int count=clamp(uSamples,1,16);int side=int(ceil(sqrt(float(count))));
 for(int i=0;i<count;++i){sampleNumber=i;vec2 p=count==1?vec2(.5):(vec2(i%side,i/side)+.5)/float(side);evaluateSample(p);}
 uint pidx=y*uint(uWidth)+x,npix=uint(uWidth)*uint(uHeight);
 for(int c=0;c<16;++c){if(uChannelSlots[c]<0)continue;uint encoded;
  if(c==6||c==7||c==15){int identity=c==6?0:c==7?1:2;uint chosen=0u;int votes=-1;
   for(int i=0;i<count;++i){uint value=idSamples[identity*16+i];int n=0;for(int j=0;j<count;++j)if(idSamples[identity*16+j]==value)++n;if(value!=0u&&n>votes){chosen=value;votes=n;}}
   encoded=pack_u24_rgb(chosen);
  }else{vec4 v=sampleAccum[c];
   if(c==1)v/=float(count);
   else if(coverageAccum>1e-6)v.rgb/=coverageAccum;
   else if(c==2)v.rgb=vec3(0,0,1);
   else if(c==3)v.rgb=vec3(0);
   else v.rgb=sampleBackground[c].rgb;
   if(c==2||c==3){float n=length(v.rgb);v.rgb=(n>1e-8?v.rgb/n:v.rgb)*.5+.5;}
   float alpha=c==0?coverageAccum/float(count):1.;encoded=pack_byte4(v.r,v.g,v.b,alpha);
  }
  out_pixels[uint(uChannelSlots[c])*npix+pidx]=encoded;
 }
}
'''
    return source.replace('sampleAccum[16]',f'sampleAccum[{channels}]').replace('sampleBackground[16]',f'sampleBackground[{channels}]').replace('c<16',f'c<{channels}')
