"""Context-owned service, explicit readback adapter, source/quality version registry."""
from concurrent.futures import Future
from threading import Thread,Event,Lock,get_ident,current_thread,main_thread
from queue import PriorityQueue,SimpleQueue,Empty,Full
from itertools import count
from types import SimpleNamespace
from dataclasses import dataclass
import ctypes as C
import atexit,sys,os,time
import numpy as np
from .gpu import GLBackend
from ..engine_v2.cache import ByteLRU

class DeviceService:
 def __init__(self,name='final',budget_mb=512,share_with=None):
  self.releases=SimpleQueue();self.queue=PriorityQueue(maxsize=32);self.serial=count();self.ready=Event();self.available=False;self.error='';self.context=None;self.backend=None;self.closed=False;self.name=name
  self.thread=Thread(target=self._run,args=(budget_mb,share_with),name='fiberbake-x-'+name,daemon=True);self.thread.start()
 def _run(self,budget,share_with):
  window=None;context=None
  try:
   if sys.platform!='win32':raise RuntimeError('WGL backend requires Windows; CPU fallback remains available')
   from ..rendercore.context import WGLContext
   user=C.WinDLL('user32',use_last_error=True);user.CreateWindowExW.restype=C.c_void_p
   user.CreateWindowExW.argtypes=[C.c_uint32,C.c_wchar_p,C.c_wchar_p,C.c_uint32,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p]
   user.DestroyWindow.argtypes=[C.c_void_p];user.DestroyWindow.restype=C.c_int
   window=user.CreateWindowExW(0,'STATIC','FiberBake X '+self.name,0x80000000|0x08000000,0,0,16,16,None,None,None,None)
   if not window:raise RuntimeError('Cannot create FiberBake X hidden surface')
   context=WGLContext(window,16,16,install_tk_paint_hook=False,share_with=share_with)
   self.context=context;self.backend=GLBackend(context,budget);self.available=True
  except Exception as error:self.error=str(error)
  finally:self.ready.set()
  try:
   if not self.available:return
   pending_releases=[]
   while True:
    while not self.releases.empty():pending_releases.append(self.releases.get())
    remaining=[]
    for index,(result,fence) in enumerate(pending_releases):
     try:
      if not self.backend.release(result,fence):
       # Atlas sends one fence followed by the rest of its lease group.
       # Preserve release ordering without stalling unrelated compute work.
       remaining=pending_releases[index:];break
     except Exception as error:
      self.error=str(error)
      # A failed group fence cannot authorize any subsequent unpin.
      self.quarantined_releases=getattr(self,'quarantined_releases',[])+pending_releases[index:]
      break
    pending_releases=remaining
    try:_,_,fn,future=self.queue.get(timeout=.005 if pending_releases else .02)
    except Empty:continue
    if fn is None:break
    if not future.set_running_or_notify_cancel():continue
    try:future.set_result(fn(self.backend))
    except BaseException as error:
     # A cancellation cannot let later jobs overwrite in-flight storage.
     try:self.backend.wait()
     except Exception:pass
     future.set_exception(error)
  finally:
   self.available=False
   try:
    if self.backend:self.backend.close()
   finally:
    try:
     if context:context.close()
    finally:
     if window:user.DestroyWindow(window)
 def call(self,fn,priority=10):
  if not self.ready.wait(15):raise RuntimeError('FiberBake X startup timeout')
  if not self.available or self.closed:raise RuntimeError(self.error or 'FiberBake X closed')
  if self.backend is not None and get_ident()==self.backend.owner:
   future=Future()
   try:future.set_result(fn(self.backend))
   except Exception as error:future.set_exception(error)
   return future
  future=Future();item=(priority,next(self.serial),fn,future)
  try:
   if current_thread() is main_thread():self.queue.put_nowait(item)
   else:self.queue.put(item,timeout=5)
  except Full:future.set_exception(RuntimeError('FiberBake X queue is full'))
  return future
 def release(self,result,fence=None):
  # Releasing a GUI lease never waits for queue capacity or GPU completion.
  if self.available and not self.closed:self.releases.put((result,fence))
 def close(self):
  if self.closed:return
  self.closed=True
  if self.available:
   while True:
    try:self.queue.put_nowait((-100,next(self.serial),None,Future()));break
    except __import__('queue').Full:
     try:_,_,_,pending=self.queue.get_nowait();pending.cancel()
     except Empty:pass

_devices={};_lock=Lock()

def device(interactive=False):
 name='interactive' if interactive else 'final'
 with _lock:
  if name not in _devices:
   _devices[name]=DeviceService(name,192 if interactive else 768)
  return _devices[name]

def close_devices():
 # Presenters close first; only the owner thread destroys its context/resources.
 for item in list(_devices.values()):item.close()
atexit.register(close_devices)


# Immutable source snapshots are shared by production passes and preview lanes.
# Byte budgets bound retained CPU bridge data; no tile/UI object is cached here.
_batches=ByteLRU(48*1024*1024)
_geometry_metadata=ByteLRU(24*1024*1024)

def canonical_batch(truth):
 batch=getattr(truth,'batch',None)
 if batch is not None:return batch
 n=len(truth);lengths=np.array([len(s.points) for s in truth],np.int32);offsets=np.r_[0,np.cumsum(lengths)].astype(np.int32)
 if not n:return None
 values=dict(count=n,offsets=offsets,points=np.concatenate([s.points for s in truth]),radii=np.concatenate([s.radii for s in truth]),t=np.concatenate([s.t for s in truth]),uniform_detail=int(lengths[0]) if np.all(lengths==lengths[0]) else 0)
 for key in ('roughness','opacity','specular','anisotropy','transmission','scatter','iridescence','depth','pattern_phase','cluster_id','numeric_id','color_root','color_mid','color_tip'):values[key]=np.asarray([getattr(s,key) for s in truth])
 sampled=[s.sampled_colors if s.sampled_colors is not None else np.zeros((len(s.points),3),np.float32) for s in truth]
 values['sampled_colors']=np.concatenate(sampled) if any(s.sampled_colors is not None for s in truth) else None
 values['has_sampled']=np.array([s.sampled_colors is not None for s in truth],np.uint8)
 return SimpleNamespace(**values)

class BatchedTruth:
 def __init__(self,truth):self.truth=truth;self.batch=canonical_batch(truth)
 def __len__(self):return len(self.truth)
 def __iter__(self):return iter(self.truth)
 def __getitem__(self,index):return self.truth[index]


def prepared_truth(truth,keys):
 if isinstance(truth,BatchedTruth):return truth
 key=tuple(sorted(keys.items()))
 item=_batches.get(key)
 # Keep the exact immutable snapshot alive; identity guards equal version keys
 # from a separately constructed source and prevents Python id reuse aliases.
 if item is not None and item.truth is truth:return item
 item=BatchedTruth(truth);_batches.put(key,item);return item


def geometry_metadata(truth,keys,size,cancel=lambda:False):
 from ..engine_v2.samples import strand_signature,bounds,AO_RADIUS
 key=(keys['geometry'],tuple(size));data=_geometry_metadata.get(key)
 if data is None:
  signatures=[];boxes=[]
  for index,strand in enumerate(truth):
   if index%128==0 and cancel():raise RuntimeError('Render cancelled')
   signatures.append(strand_signature(strand));boxes.append(bounds(strand,size,AO_RADIUS*size[0]/512.))
  data=({},tuple(signatures),tuple(boxes));_geometry_metadata.put(key,data)
 return data


def render_resident(tile,truth,keys,bins_data,size,channels,cancel=lambda:False,interactive=False,pin=False,progress=None):
 if os.environ.get('FSD_FIBERBAKE_BACKEND','auto').lower()=='cpu':raise NotImplementedError('CPU reference explicitly requested')
 d=device(interactive);truth=prepared_truth(truth,keys)
 if bins_data is None:
  bins_data=geometry_metadata(truth,keys,size,cancel)
 return d,d.call(lambda gpu:gpu.render(tile,truth,keys,bins_data,size,channels,cancel,progress,pin),0 if interactive else 10).result()


def try_bake(truth,bins_data,keys,size,channels,settings,cancel,progress=None):
 """Legacy RenderResult adapter: readback is explicit and runs on a worker, never Qt."""
 if os.environ.get('FSD_FIBERBAKE_BACKEND','auto').lower()=='cpu' or not len(truth):return None
 from PIL import Image
 tile=SimpleNamespace(id=truth[0].sid.split(':',1)[0],revision=0,settings=settings)
 images={}
 try:
  d=device(False);prepared=prepared_truth(truth,keys)
  data=bins_data if bins_data is not None else geometry_metadata(prepared,keys,size,cancel)
  for start in range(0,len(channels),4):
   if cancel():raise RuntimeError('Render cancelled')
   group=channels[start:start+4]
   # Read back each pass before the next; budget can evict earlier GPU channels.
   def run(gpu):
    result=gpu.render(tile,prepared,keys,data,size,group,cancel)
    return gpu.readback(result)
   arrays=d.call(run).result()
   for channel,array in arrays.items():
    if channel=='color':images[channel]=Image.fromarray(array,'RGBA')
    elif channel in ('normal','flow','id','cluster_id','material_id','material_mask','rgb_mask'):images[channel]=Image.fromarray(np.ascontiguousarray(array[:,:,:3]),'RGB')
    else:images[channel]=Image.fromarray(np.ascontiguousarray(array[:,:,0]),'L')
   if progress:progress(min(1.,(start+len(group))/len(channels)))
  return images
 except Exception as error:
  if cancel() or 'cancelled' in str(error).lower():raise RuntimeError('Render cancelled') from error
  if os.environ.get('FSD_FIBERBAKE_BACKEND')=='gpu':raise
  d=device(False);d.error=str(error)
  return None


class PreviewPacket:
 def __init__(self,device,result,truth):self.device=device;self.result=result;self.truth=truth;self.closed=False
 def close(self,fence=None):
  if self.closed:return
  self.closed=True;self.device.release(self.result,fence)
 def __del__(self):
  try:self.close()
  except Exception:pass

def prepare_preview(tile,size,channel,cancel):
 from ..engine_v2.engine import ENGINE
 from ..engine_v2.samples import build_bins
 truth,keys=ENGINE.truth(tile,cancel=cancel)
 bins=None
 if cancel():raise RuntimeError('Render cancelled')
 service,result=render_resident(tile,truth,keys,bins,size,(channel,),cancel,interactive=True,pin=True)
 packet=PreviewPacket(service,result,truth)
 if cancel():packet.close();raise RuntimeError('Render cancelled')
 return packet


def peek_device():
 value=_devices.get('final')
 return value if value is not None and value.available and not value.closed else None


def request_atlas_tiles(requests):
 """Lease only exact published production versions; no evaluation/bake on navigation."""
 d=peek_device()
 if d is None:return None
 def gather(gpu):
  packets={}
  for tid,channel,version in requests:
   from .atlas import source_for
   descriptor=source_for(tid,channel,version)
   if descriptor is not None:
    try:
     result=descriptor.ensure(gpu,(channel,),pin=True);packets[tid]=PreviewPacket(d,result,());continue
    except Exception as error:d.error=str(error)
   for (source_id,size),entry in reversed(gpu.entries.items()):
    if source_id!=tid or channel not in entry.textures or channel not in entry.versions:continue
    from ..engine_v2.dependencies import channel_key
    if channel_key(channel,entry.versions[channel],size)!=version:continue
    record=entry.textures[channel];record.pins+=1
    from .gpu import ResidentResult
    result=ResidentResult(tid,0,size,{channel:record.name},{channel:version},dict(readback_bytes=0), (record,))
    packets[tid]=PreviewPacket(d,result,());break
  return packets
 return d.call(gather,priority=1)
