import torch
import torch.fx as fx

def device_placement_pass(gm: torch.fx.GraphModule):
    """
    Reads 'target_rank' annotation and enforces device placement 'to()' ops locally.
    Wait, 'target_rank' tells us which PROCESS assumes ownership.
    Inside that process, we map to the local device.
    
    If we are strictly running 1 rank per GPU, the device is always 'cuda:0' (local relative)
    or 'cuda:N' (global absolute).
    
    Standard DDP/PP usually uses 'cuda:local_rank'.
    
    However, this pass is conceptually "Assign Ops to Devices".
    If we are looking at the GLOBAL graph, we might say:
    Node A -> Rank 0 -> cuda:0
    Node B -> Rank 1 -> cuda:1
    
    But when we run 'topology_pass', we slice the graph so each rank only sees its own nodes.
    So the *final* code on Rank 1 should probably say `.to('cuda:0')` (if using CUDA_VISIBLE_DEVICES)
    or `.to('cuda:1')`.
    
    Let's assume Global Addressing for now to align with "Device Placement".
    """
    print("\n>> [Pass] Running Device Placement...")
    
    # We Iterate and apply 'device' metadata based on target_rank
    # We do NOT insert .to() ops yet if we plan to cut the graph.
    # The Topology pass handles the cutting.
    # This pass essentially refines the 'device' meta-data to be concrete.
    
    for node in gm.graph.nodes:
        rank = node.meta.get('target_rank', 0)
        
        # Map rank to device
        # For now, 1-to-1 mapping
        if torch.cuda.is_available():
            # In a real distributed run, each process sees 'cuda:0' usually if env vars are set.
            # But for GLOBAL representation, we might want 'cuda:rank'.
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")
            
        node.meta["device"] = device
        
    return gm
