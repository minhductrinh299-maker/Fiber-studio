"""Context-thread buffer deltas. Fixed-capacity resources grow geometrically."""
import numpy as np

def changed_ranges(previous,current,max_ranges=64):
 current=np.ascontiguousarray(current)
 if previous is None or previous.shape!=current.shape or previous.dtype!=current.dtype:return [(0,current.nbytes)] if current.nbytes else []
 # Compare exact source bytes, coalesce adjacent changed records.
 stride=current.dtype.itemsize*(int(np.prod(current.shape[1:])) if current.ndim>1 else 1)
 if not len(current):return []
 rows=(current.view(np.uint8).reshape(len(current),stride)!=previous.view(np.uint8).reshape(len(current),stride)).any(axis=1)
 indices=np.flatnonzero(rows)
 if not len(indices):return []
 starts=np.r_[indices[0],indices[1:][np.diff(indices)>1]];ends=np.r_[indices[:-1][np.diff(indices)>1]+1,indices[-1]+1]
 if len(starts)>max_ranges:return [(0,current.nbytes)]
 return [(int(a)*stride,int(b-a)*stride) for a,b in zip(starts,ends)]
