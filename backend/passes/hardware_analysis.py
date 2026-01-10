import torch
import torch.fx as fx

def hardware_analysis_pass(gm: torch.fx.GraphModule):
    """
    Analyzes the current hardware environment and attaches metadata to the GraphModule.
    Collects:
    - Number of GPUs
    - Per-GPU details: Name, SM Count, Memory, Clock Rate (if approx available/inferred)
    """
    print("\n>> [Pass] Running Hardware Analysis...")
    
    hardware_info = {
        "device_count": 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        hardware_info["device_count"] = count
        print(f"   + Detected {count} CUDA devices.")
        
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            # props has: name, major, minor, total_memory, multi_processor_count
            
            # Clock rate isn't directly exposed in basic properties in all versions easily 
            # without pynvml, but we can capture what we have.
            
            # Compute capability
            compute_cap = f"{props.major}.{props.minor}"
            
            info = {
                "id": i,
                "name": props.name,
                "sm_count": props.multi_processor_count,
                "total_memory_mb": props.total_memory / (1024**2),
                "compute_capability": compute_cap
            }
            hardware_info["devices"].append(info)
            
            print(f"   + GPU {i}: {props.name} | SMs: {props.multi_processor_count} | Mem: {info['total_memory_mb']:.0f} MB | Cap: {compute_cap}")
            
    else:
        print("   + No CUDA devices detected.")
        hardware_info["device_count"] = 0
        
    # Attach to GraphModule metadata
    gm.meta["hardware_info"] = hardware_info
    
    return gm
