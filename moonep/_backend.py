"""Select the implementation supplied by the installed extension build."""

# Load PyTorch's shared libraries before importing its C++ extension.
import torch

from . import _C


IS_PPU = bool(getattr(_C, "TILELANG_PPU_BUILD", False))

if IS_PPU:
    from .tilelang import api
    from .tilelang.buffer import pad_to_granularity
else:
    from . import api
    from .buffer import pad_to_granularity

Buffer = api.Buffer
MoonEPCommPlan = api.MoonEPCommPlan
allocate_planning_outputs = api.allocate_planning_outputs
create_nvl_dist_tensor = api.create_nvl_dist_tensor
pad_dim0_for_alignment = api.pad_dim0_for_alignment
launch_planning = api.launch_planning
launch_dispatch = api.launch_dispatch
launch_dispatch_epilogue = api.launch_dispatch_epilogue
launch_combine = api.launch_combine
launch_combine_prologue = api.launch_combine_prologue
launch_inter_rank_sync = api.launch_inter_rank_sync
launch_prefetch = api.launch_prefetch
launch_grad_reduce = api.launch_grad_reduce
