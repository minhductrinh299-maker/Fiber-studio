"""Direct shared-texture Qt presentation. All GL calls stay on the GUI owner."""
import ctypes as C
import math
from ..rendercore.context import WGLContext
from ..rendercore.glapi import GL

VERTEX='''#version 330 core
out vec2 uv;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.-1.,0,1);}'''
FRAGMENT='''#version 330 core
uniform sampler2D uMap;uniform vec4 uRect;uniform vec2 uScreen;uniform vec2 uPan;
uniform float uZoom;uniform float uAngle;uniform int uColor;
out vec4 outColor;
vec4 sampleMap(vec2 uv){if(uColor==0)return texture(uMap,uv);ivec2 size=textureSize(uMap,0);vec2 q=uv*vec2(size)-.5;ivec2 p=ivec2(floor(q));vec2 f=fract(q);vec4 sum=vec4(0);for(int y=0;y<2;++y)for(int x=0;x<2;++x){vec4 v=texelFetch(uMap,clamp(p+ivec2(x,y),ivec2(0),size-1),0);v.rgb*=v.a;sum+=v*(x==0?1.-f.x:f.x)*(y==0?1.-f.y:f.y);}if(sum.a>1e-8)sum.rgb/=sum.a;return sum;}
void main(){
 vec2 screen=vec2(gl_FragCoord.x,uScreen.y-gl_FragCoord.y);
 vec2 p=screen-uScreen*.5-uPan;
 float c=cos(uAngle),s=sin(uAngle);p=mat2(c,s,-s,c)*p/max(uZoom,1e-8)+uScreen*.5;
 vec2 t=(p-uRect.xy)/uRect.zw;
 vec3 bg=vec3(.067,.078,.094);
 if(any(lessThan(t,vec2(0)))||any(greaterThan(t,vec2(1)))){outColor=vec4(bg,1);return;}
 vec4 tex=sampleMap(t);
 if(uColor==1){float checker=mod(floor(p.x/12.)+floor(p.y/12.),2.);vec3 back=mix(vec3(.12),vec3(.17),checker);outColor=vec4(mix(back,tex.rgb,tex.a),1);}
 else outColor=vec4(tex.rgb,1);
}'''

class TexturePresenter:
 def __init__(self,viewport,packet):
  self.viewport=viewport;self.surface=viewport.live_surface();self.packet=None;self.fence=None
  self.context=WGLContext(int(self.surface.winId()),install_tk_paint_hook=False,share_with=packet.device.context)
  self.g=GL(self.context);self.program=self.g.program(VERTEX,FRAGMENT);self.vao=self.g.name('VertexArrays')
  self.overlay_texture=0;self.overlay_size=None;self.overlay_program=0
  self.set_packet(packet)
 def set_packet(self,packet):
  if self.packet:self.packet.close(self.fence);self.fence=None
  self.packet=packet
  self.viewport._gpu_texture_image_size=packet.result.size
 def frame(self):
  if not self.packet or self.packet.closed:return
  viewport=self.viewport
  if callable(getattr(viewport,'_map_canvas',None)) or viewport._renderer_callback is None:viewport.hide_live_surface();return
  self.context.activate();g=self.g
  self.surface.setGeometry(viewport.rect());self.surface.show();self.surface.raise_()
  ratio=viewport.devicePixelRatioF();w=round(viewport.width()*ratio);h=round(viewport.height()*ratio)
  self.context.width=w;self.context.height=h;g.glViewport(0,0,w,h);g.glDisable(0x0B71);g.glDisable(0x0BE2);g.glDisable(0x0C11)
  result=self.packet.result;self.viewport._gpu_texture_image_size=result.size;tw,th=result.size;scale=min((w-32*ratio)/tw,(h-32*ratio)/th);rw=tw*scale;rh=th*scale
  g.glUseProgram(self.program);loc=lambda name:g.glGetUniformLocation(self.program,name.encode())
  g.glUniform4f(loc('uRect'),(w-rw)/2,(h-rh)/2,rw,rh);g.glUniform2f(loc('uScreen'),w,h)
  g.glUniform2f(loc('uPan'),viewport._view_pan.x()*ratio,viewport._view_pan.y()*ratio)
  g.glUniform1f(loc('uZoom'),viewport._view_zoom);g.glUniform1f(loc('uAngle'),math.radians(viewport._view_angle))
  channel=next(iter(result.textures));g.glUniform1i(loc('uColor'),int(channel=='color'));g.glUniform1i(loc('uMap'),0)
  g.glActiveTexture(0x84C0);g.glBindTexture(0x0DE1,result.textures[channel]);g.glBindVertexArray(self.vao);g.glDrawArrays(4,0,3)
  self.draw_overlay()
  if self.fence:g.glDeleteSync(self.fence)
  self.fence=g.glFenceSync(0x9117,0);g.glFlush();self.context.present()
 def draw_overlay(self,atlas=None):
  from PySide6 import QtGui,QtCore
  v=self.viewport;ratio=v.devicePixelRatioF();width=round(v.width()*ratio);height=round(v.height()*ratio)
  image=QtGui.QImage(width,height,QtGui.QImage.Format.Format_RGBA8888);image.setDevicePixelRatio(ratio);image.fill(0)
  painter=QtGui.QPainter(image);painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
  if atlas is not None:atlas._paint_atlas_overlay(painter)
  else:
   overlay=getattr(v,'_map_overlay',None)
   if overlay:overlay(painter)
   if v._show_cube:v._draw_orientation_cube(painter,QtCore.QPointF(42,42))
  painter.setPen(QtGui.QColor('#7DD3C0'));painter.setFont(v.font())
  painter.drawText(v.rect().adjusted(14,64 if v._show_cube and atlas is None else 14,-14,-14),QtCore.Qt.AlignmentFlag.AlignTop|QtCore.Qt.AlignmentFlag.AlignLeft,v._hud_text);painter.end()
  g=self.g
  if not self.overlay_texture:
   self.overlay_texture=g.name('Textures');self.overlay_program=g.program(VERTEX,'''#version 330 core
in vec2 uv;uniform sampler2D uMap;out vec4 color;void main(){color=texture(uMap,vec2(uv.x,1.-uv.y));}''')
  g.glActiveTexture(0x84C0);g.glBindTexture(0x0DE1,self.overlay_texture)
  g.glTexParameteri(0x0DE1,0x2801,0x2601);g.glTexParameteri(0x0DE1,0x2800,0x2601)
  payload=C.c_void_p(C.addressof(C.c_char.from_buffer(image.bits())))
  if self.overlay_size!=(width,height):g.glTexImage2D(0x0DE1,0,0x8058,width,height,0,0x1908,0x1401,payload);self.overlay_size=(width,height)
  else:g.glTexSubImage2D(0x0DE1,0,0,0,width,height,0x1908,0x1401,payload)
  g.glUseProgram(self.overlay_program);g.glUniform1i(g.glGetUniformLocation(self.overlay_program,b'uMap'),0);g.glEnable(0x0BE2);g.glBlendFunc(0x0302,0x0303);g.glBindVertexArray(self.vao);g.glDrawArrays(4,0,3);g.glDisable(0x0BE2)
 def close(self):
  self.context.activate()
  if self.overlay_texture:
   name=C.c_uint(self.overlay_texture);self.g.glDeleteTextures(1,C.byref(name));self.g.glDeleteProgram(self.overlay_program)
  if self.packet:self.packet.close(self.fence);self.packet=None;self.fence=None
  arr=C.c_uint(self.vao);self.g.glDeleteVertexArrays(1,C.byref(arr));self.g.glDeleteProgram(self.program);self.context.close()


class AtlasPresenter(TexturePresenter):
 def __init__(self,owner,packets):
  self.owner=owner;self.packets={};super().__init__(owner.viewport,next(iter(packets.values())))
  self.packet=None;self.packets=packets
  self.atlas_program=self.g.program(VERTEX,FRAGMENT.replace('outColor=vec4(bg,1);return;','discard;').replace('uniform int uColor;','uniform int uColor;uniform float uTileAngle;uniform vec2 uSourceSize;').replace('vec4 tex=sampleMap(t);','float tc=cos(uTileAngle),ts=sin(uTileAngle);vec2 centered=t-.5;float q=abs(mod(uTileAngle,1.57079632679));t=(q<1e-5?mat2(tc,ts,-ts,tc)*centered:(mat2(tc,ts,-ts,tc)*(centered*uSourceSize))/uSourceSize)+.5;vec4 tex=any(lessThan(t,vec2(0)))||any(greaterThan(t,vec2(1)))?vec4(0):sampleMap(t);'))
 def replace_packets(self,packets):
  # One flushed consumer fence guards the complete previous packet set. The
  # service retires it before processing the subsequent unpin commands.
  fence=self.fence;self.fence=None
  for packet in self.packets.values():
   packet.close(fence);fence=None
  self.packets=packets
 def frame(self):
  owner=self.owner;v=self.viewport
  if not owner._is_atlas():v.hide_live_surface();return
  self.context.activate();g=self.g;ratio=v.devicePixelRatioF();width=round(v.width()*ratio);height=round(v.height()*ratio)
  self.surface.setGeometry(v.rect());self.surface.show();self.surface.raise_();g.glViewport(0,0,width,height);g.glDisable(0x0B71);g.glDisable(0x0BE2);g.glDisable(0x0C11);g.glClearColor(.067,.078,.094,1);g.glClear(0x4000)
  g.glUseProgram(self.atlas_program);loc=lambda name:g.glGetUniformLocation(self.atlas_program,name.encode())
  g.glUniform2f(loc('uScreen'),width,height);g.glUniform2f(loc('uPan'),v._view_pan.x()*ratio,v._view_pan.y()*ratio);g.glUniform1f(loc('uZoom'),v._view_zoom);g.glUniform1f(loc('uAngle'),math.radians(v._view_angle));g.glUniform1i(loc('uMap'),0)
  records=owner._atlas_records()
  if owner._atlas_drag and not owner._atlas_drag.get('box'):records.update(owner._atlas_drag['current'])
  left,top,side=owner._atlas_base();size=owner.controller.atlas_layout.size
  for tid,packet in self.packets.items():
   if tid not in records or packet.closed:continue
   x,y,w,h,rotation=records[tid];channel=next(iter(packet.result.textures))
   g.glUniform4f(loc('uRect'),(left+x/size*side)*ratio,(top+y/size*side)*ratio,w/size*side*ratio,h/size*side*ratio)
   g.glUniform2f(loc('uSourceSize'),*packet.result.size);g.glUniform1f(loc('uTileAngle'),math.radians(rotation));g.glUniform1i(loc('uColor'),int(channel=='color'));g.glActiveTexture(0x84C0);g.glBindTexture(0x0DE1,packet.result.textures[channel]);g.glBindVertexArray(self.vao);g.glDrawArrays(4,0,3)
  self.draw_overlay(owner)
  if not owner._atlas_timer.isActive():owner._atlas_timer.start(40)
  if self.fence:g.glDeleteSync(self.fence)
  self.fence=g.glFenceSync(0x9117,0);g.glFlush();self.context.present()
 def close(self):
  self.context.activate();self.g.glFinish()
  for packet in self.packets.values():packet.close()
  self.packets={}
  if self.fence:self.g.glDeleteSync(self.fence);self.fence=None
  self.g.glDeleteProgram(self.atlas_program);super().close()
