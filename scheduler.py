"""Bounded priority/coalescing executor. Interactive and final use separate lanes."""
from concurrent.futures import Future,CancelledError
from threading import Condition,Thread,Event,current_thread
from itertools import count

class LatestScheduler:
 def __init__(self,workers=1,max_pending=32,name='fiberbake'):
  self.condition=Condition();self.pending={};self.running={};self.serial=count(1);self.closed=False;self.limit=max_pending
  self.threads=[Thread(target=self._run,name=f'{name}-{i}',daemon=True) for i in range(workers)]
  for thread in self.threads:thread.start()
 def submit(self,key,fn,priority=10,coalesce=True):
  future=Future();token=Event()
  with self.condition:
   if self.closed:raise RuntimeError('Scheduler closed')
   if coalesce:
    old=self.pending.pop(key,None)
    if old:old[3].set();old[4].cancel()
    if key in self.running:self.running[key].set()
   elif key in self.pending or key in self.running:raise ValueError('Final job keys must be unique')
   if len(self.pending)>=self.limit:
    victim=max(self.pending,key=lambda k:self.pending[k][:2]);old=self.pending.pop(victim);old[3].set();old[4].set_exception(CancelledError('Queue budget exceeded'))
   self.pending[key]=(priority,next(self.serial),fn,token,future);self.condition.notify_all()
  return future
 def _run(self):
  while True:
   with self.condition:
    self.condition.wait_for(lambda:self.closed or any(k not in self.running for k in self.pending))
    if self.closed:return
    key=min((k for k in self.pending if k not in self.running),key=lambda k:self.pending[k][:2])
    _,_,fn,token,future=self.pending.pop(key);self.running[key]=token
   try:
    if future.set_running_or_notify_cancel():
     if token.is_set():raise CancelledError()
     result=fn(token.is_set)
     if token.is_set():raise CancelledError()
     future.set_result(result)
   except BaseException as error:
    if not future.done():future.set_exception(error)
   finally:
    with self.condition:self.running.pop(key,None);self.condition.notify_all()
 def close(self,wait=False):
  with self.condition:
   self.closed=True
   for _,_,_,token,future in self.pending.values():token.set();future.cancel()
   for token in self.running.values():token.set()
   self.pending.clear();self.condition.notify_all()
  if wait:
   for thread in self.threads:
    if thread is not current_thread():thread.join(2)
