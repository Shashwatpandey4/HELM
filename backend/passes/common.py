import torch

def get_node_name(node):
    return node.name

# Dummy distributed functions for visualization if functional ops are missing
# Real distributed ops wrapper
def dist_all_reduce(tensor, op='sum', group=None):
    if torch.distributed.is_initialized():
        # Map string op to object if needed
        # dist.all_reduce is in-place
        op_obj = torch.distributed.ReduceOp.SUM # simplified
        torch.distributed.all_reduce(tensor, op=op_obj, group=group)
        return tensor
    else:
        return tensor
    
def dist_send(tensor, dst, group=None):
    if torch.distributed.is_initialized():
        # dist.send is synchronous or async with wait?
        # functional send/recv might be preferred but let's use standard
        torch.distributed.send(tensor, dst=dst, group=group)
    return tensor
    
def dist_recv(src, shape=None, dtype=None, group=None):
    if torch.distributed.is_initialized():
        if shape is None:
            # Fallback or error? Real execution needs shape.
            # Maybe use a small tensor header? Too complex for now.
            raise ValueError("Runtime dist_recv requires shape")
            
        device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
        if dtype is None:
            dtype = torch.get_default_dtype()
        tensor = torch.zeros(shape, device=device, dtype=dtype)
        torch.distributed.recv(tensor, src=src, group=group)
        return tensor
    else:
        if shape:
            return torch.zeros(shape, dtype=dtype or torch.float32) 
        return torch.empty(1)
