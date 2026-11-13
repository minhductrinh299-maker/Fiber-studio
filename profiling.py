"""Optional GPU timestamps. Collection only occurs after the owner's frame fence."""
import ctypes as C

class TimestampProfiler:
    def __init__(self,gl):self.gl=gl;self.queries={};self.active=set()
    def begin(self,label):
        if label not in self.queries:self.queries[label]=(self.gl.name('Queries'),self.gl.name('Queries'))
        self.gl.glQueryCounter(self.queries[label][0],0x8E28);self.active.add(label)
    def end(self,label):self.gl.glQueryCounter(self.queries[label][1],0x8E28)
    def collect(self):
        result={}
        for label in self.active:
            a,b=self.queries[label];low=C.c_uint64();high=C.c_uint64()
            self.gl.glGetQueryObjectui64v(a,0x8866,C.byref(low));self.gl.glGetQueryObjectui64v(b,0x8866,C.byref(high));result[label]=max(0,high.value-low.value)/1e6
        self.active.clear();return result
    def close(self):
        values=[name for pair in self.queries.values() for name in pair]
        if values:self.gl.glDeleteQueries(len(values),(C.c_uint*len(values))(*values))
        self.queries.clear()
