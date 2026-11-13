"""Transitive source dependency graph and conservative old+new dirty footprints."""
from dataclasses import dataclass
from math import ceil
from ..engine_v2.dependencies import CHANNEL_SOURCES, ALL_CHANNELS, channel_key
from ..engine_v2.samples import BLOCK

# Leaves are canonical source domains, never output textures.
GRAPH={
 'curves':('geometry',),'footprint':('curves','sampling'),
 'coverage':('footprint','opacity'),'frame':('curves','orientation'),
 'color':('coverage','pigment'),'alpha':('coverage',),
 'normal':('coverage','frame'),'flow':('coverage','curves'),
 'depth':('coverage','curves'),'ao':('coverage','occluders','occlusion'),
 'occluders':('curves','opacity'),'id':('coverage',),
 'material_id':('coverage','material_id_source'),
 'material_mask':('coverage','material_id_source','material_mask_source'),
 'cluster_id':('coverage',),'rgb_mask':('coverage','curves'),
 **{c:('coverage',c+'_source') for c in ('roughness','specular','anisotropy','transmission','scatter','iridescence')},
 'pigment':('color_source',),
}
def leaves(node, path=()):
 if node in path:raise ValueError('Dependency cycle: '+str(path+(node,)))
 if node not in GRAPH:return frozenset((node,))
 return frozenset().union(*(leaves(n,path+(node,)) for n in GRAPH[node]))

def affected(changed):
 return frozenset(c for c in ALL_CHANNELS if leaves(c)&set(changed))

@dataclass(frozen=True)
class DirtyPlan:
 channels:tuple
 blocks:tuple
 full:bool
 reason:str

def plan(keys,previous_keys,bins,previous_bins,size,requested,changed_indices=None,threshold=.55):
 channels=tuple(c for c in requested if not previous_keys or channel_key(c,keys,size)!=channel_key(c,previous_keys,size))
 if not channels:return DirtyPlan((),(),False,'cache-hit')
 width,height=size;all_blocks={(x,y) for y in range(ceil(height/BLOCK)) for x in range(ceil(width/BLOCK))}
 if not previous_keys:return DirtyPlan(channels,tuple(sorted(all_blocks)),True,'cold')
 # Scalar defaults affect the empty background as well as occupied bins.
 defaults_changed=any(c in ('roughness','specular','anisotropy','transmission','scatter','iridescence') and keys[c]!=previous_keys.get(c) for c in channels)
 if defaults_changed or keys['sampling']!=previous_keys.get('sampling'):
  return DirtyPlan(channels,tuple(sorted(all_blocks)),True,'sampling/default')
 old=previous_bins or {};new=bins or {}
 if changed_indices is None:dirty=set(old)|set(new)
 else:
  changed=set(changed_indices)
  dirty={block for mapping in (old,new) for block,ids in mapping.items() if changed.intersection(ids)}
 full=len(dirty)>len(all_blocks)*threshold
 return DirtyPlan(channels,tuple(sorted(all_blocks if full else dirty)),full,'threshold' if full else 'old+new footprint with AO halo')


def plan_bounds(keys,previous_keys,boxes,previous_boxes,size,requested,changed_indices=None):
    """Dirty rectangles use old+new canonical bounds; GPU owns bin membership."""
    channels=tuple(c for c in requested if not previous_keys or channel_key(c,keys,size)!=channel_key(c,previous_keys,size))
    if not channels:return DirtyPlan((),(),False,'cache-hit')
    width,height=size
    if not previous_keys or changed_indices is None or keys['sampling']!=previous_keys.get('sampling'):
        blocks=tuple((x,y) for y in range(ceil(height/BLOCK)) for x in range(ceil(width/BLOCK)))
        return DirtyPlan(channels,blocks,True,'source/full')
    dirty=set()
    for index in changed_indices:
        for records in (boxes,previous_boxes or ()):
            if index>=len(records):continue
            x0,y0,x1,y1=records[index]
            for y in range(max(0,int(y0)//BLOCK),min(ceil(height/BLOCK),ceil(y1/BLOCK))):
                for x in range(max(0,int(x0)//BLOCK),min(ceil(width/BLOCK),ceil(x1/BLOCK))):dirty.add((x,y))
    full=len(dirty)>ceil(width/BLOCK)*ceil(height/BLOCK)*.55
    if full:dirty={(x,y) for y in range(ceil(height/BLOCK)) for x in range(ceil(width/BLOCK))}
    return DirtyPlan(channels,tuple(sorted(dirty)),full,'old+new geometry bounds')
