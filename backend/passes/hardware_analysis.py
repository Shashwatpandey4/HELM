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
            
            # Heuristics for TFLOPS estimation
            # Cores per SM based on arch (approx)
            cores_per_sm = 64 # Default
            if props.major == 6: cores_per_sm = 64 # Pascal (e.g. 1080: 128 actually? No 1080 is 128/SM? Pascal GP104 is 128.)
            # Corrections:
            # Pascal (6.1): 128 cores/SM
            # Volta (7.0): 64 FP32 cores/SM
            # Turing (7.5): 64 FP32 cores/SM
            # Ampere (8.0/8.6): 64 FP32 + 64 INT32/FP32 = 128 cores/SM effective for FP32? Or 64? 
            # Ampere is complex. conservatively 128.
            # Hopper (9.0): 128
            
            if props.major == 6: cores_per_sm = 128 # Pascal
            elif props.major == 7: cores_per_sm = 64 # Volta/Turing
            elif props.major == 8: cores_per_sm = 128 # Ampere
            elif props.major >= 9: cores_per_sm = 128 # Hopper
            
            # Clock estimation (Boost clock in GHz)
            clock_ghz = 1.5 # Conservative base
            
            # TFLOPS = SM * Cores_per_SM * 2 (FMA) * Clock / 1000
            tflops = props.multi_processor_count * cores_per_sm * 2 * clock_ghz / 1000.0
            
            info = {
                "id": i,
                "name": props.name,
                "sm_count": props.multi_processor_count,
                "memory_mb": props.total_memory / (1024**2),
                "compute_capability": compute_cap,
                "throughput_tflops": tflops,
                "bandwidth_gbps": 16.0 # PCIe Gen3/4 heuristic, placeholder
            }
            hardware_info["devices"].append(info)
            
            print(f"   + GPU {i}: {props.name} | SMs: {props.multi_processor_count} | Mem: {info['memory_mb']:.0f} MB | Est. TFLOPS: {tflops:.2f}")
            
    else:
        print("   + No CUDA devices detected.")
        hardware_info["device_count"] = 0
        
    # Attach to GraphModule metadata
    gm.meta["hardware_info"] = hardware_info
    
    return gm

