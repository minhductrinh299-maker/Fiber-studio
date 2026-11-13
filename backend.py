"""Render policies and backend boundary; no API exposes unowned GL names."""
from dataclasses import dataclass
from typing import Protocol,Callable,Mapping

@dataclass(frozen=True)
class Quality:
    samples:int=1
    ao_mode:str='proximity'
    @classmethod
    def preview(cls):return cls(1,'proximity')
    @classmethod
    def final(cls):return cls(16,'geometry')

@dataclass(frozen=True)
class BackendCapabilities:
    api:str
    compute:bool
    resident_textures:bool
    partial_updates:bool
    max_budget_bytes:int

class RasterBackend(Protocol):
    """OpenGL implements this boundary; native D3D12 can implement the same contract.

    A native backend must supply owner-thread execution, explicit fences and
    versioned image resources. A backend name alone never advertises support.
    """
    capabilities:BackendCapabilities
    def render(self,tile,truth,keys,bins_data,size,channels,cancel:Callable=...,progress:Callable|None=...,pin:bool=False):...
    def readback(self,result)->Mapping:...
    def release(self,result,fence=None)->None:...
    def close(self)->None:...
