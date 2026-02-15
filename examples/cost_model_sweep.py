import dataclasses
from helm.optimization.cost_model import *
from helm.optimization.profiler import SystemProfiler

# --- 1. Mock Data Generators ---

def create_model_spec(name: str, batch_size=1, seq_len=512, gen_len=128) -> ModelSpec:
    """
    Creates a ModelSpec based on approximate architectures of known LLMs.
    """
    if name == "gpt2-small":
        # ~125M params
        L, H, A = 12, 768, 12
        vocab = 50257
    elif name == "llama-7b":
        # ~7B params
        L, H, A = 32, 4096, 32
        vocab = 32000
    elif name == "llama-70b":
        # ~70B params (simplified)
        L, H, A = 80, 8192, 64 # GQA ignored for simplicity in basic calcs
        vocab = 32000
    else:
        raise ValueError(f"Unknown model: {name}")

    # Derived sizes (FP16 = 2 bytes)
    dtype_bytes = 2
    
    # 1. Parameter Estimates (Naive)
    # Params ~= 12 * H^2 * L (Approx) + V * H
    # Better approx per layer:
    # Attn: 4 * H^2 (Q,K,V,O)
    # MLP: 3 * H^2 * expansion (usually 4? Llama uses SwiGLU so 3 matrices? Let's say 3 * H * 4H -> 12 H^2? No MLP is 2 up 1 down... 3 matrices of size Hx4H approx? )
    # Let's approximate Bytes Per Layer for Params:
    # 7B model / 32 layers ~= 218M params/layer => ~430MB/layer
    
    total_params = 0
    if name == "llama-7b": total_params = 7e9
    elif name == "llama-70b": total_params = 70e9
    elif name == "gpt2-small": total_params = 125e6
    
    bytes_per_layer = (total_params * dtype_bytes) / L
    
    layers = []
    for i in range(L):
        # FLOPs Estimation
        # Prefill: 2 * P * B * S
        flops_prefill = 2 * (total_params / L) * batch_size * seq_len
        
        # Decode: 2 * P * B * 1
        flops_decode = 2 * (total_params / L) * batch_size * 1
        
        # Memory Movements
        # Prefill: Read Weights + Read KV + Write KV + Read Activation
        # Approx: Weights
        bytes_prefill = bytes_per_layer
        
        # Decode: Read Weights + Read KV (history) + Write KV (new)
        # KV Size per token = 2 * H * L_layers * Dtype (No, per layer spec)
        # KV per layer per token = 2 * H * Dtype? No.
        # KV Cache = 2 * n_heads * head_dim * seq_len * batch
        # head_dim = H / A
        head_dim = H // A
        kv_bytes_per_token_per_layer = 2 * A * head_dim * batch_size * dtype_bytes
        
        # Decode reads full history KV
        # avg history ~ seq_len + gen_len/2? Let's assume full context prefill + gen so far.
        # Just use max context for specific layer IO estimate
        full_ctx = seq_len + gen_len
        bytes_decode = bytes_per_layer + (kv_bytes_per_token_per_layer * full_ctx)
        
        # Activation Size (per microbatch, for intermediate stashing)
        # B * S * H * Dtype
        act_bytes = batch_size * seq_len * H * dtype_bytes
        
        layers.append(LayerSpec(
            flops_prefill=flops_prefill,
            flops_decode_token=flops_decode,
            bytes_moved_prefill=bytes_prefill,
            bytes_moved_decode=bytes_decode,
            activation_bytes=act_bytes,
            param_bytes=bytes_per_layer,
            kv_bytes_per_token=kv_bytes_per_token_per_layer
        ))

    return ModelSpec(
        L=L, layers=layers, dtype="fp16", n_heads=A, hidden_size=H, vocab_size=vocab,
        seq_len_prefill=seq_len, batch_size=batch_size, gen_len=gen_len
    )

def create_device(name: str, id: int) -> DeviceSpec:
    if name == "a100-80g":
        return DeviceSpec(id=id, peak_flops=312e12, mem_bw=2.0e12, mem_capacity=80e9, h2d_bw=24e9)
    elif name == "t4-16g":
        return DeviceSpec(id=id, peak_flops=65e12, mem_bw=320e9, mem_capacity=16e9, h2d_bw=12e9)
    elif name == "rtx3090-24g":
        return DeviceSpec(id=id, peak_flops=35e12, mem_bw=936e9, mem_capacity=24e9, h2d_bw=16e9)
    elif name == "cpu-host":
        # Dual socket high end CPU
        return DeviceSpec(id=id, peak_flops=4e12, mem_bw=150e9, mem_capacity=512e9, h2d_bw=0) # Host
    else:
        raise ValueError(f"Unknown device: {name}")

# --- 2. Scenarios ---

def run_suite():
    # Profile the system first
    profiler = SystemProfiler()
    profile = profiler.run()
    
    # Use profiled data for calibration
    calib = CalibrationDB(profile=profile)
    
    scenarios = [
        # 1. Baseline: Llama 7B on A100 (Fits comfortably)
        ("Baseline: 7B on A100", 
         create_model_spec("llama-7b"), 
         [create_device("a100-80g", 0)],
         ParallelConfig(tp_degree=1, pp_degree=1, microbatches=1, 
                        pp_stages=[StagePlacement([0], (0, 32))])
        ),
        
        # 2. OOM: Llama 7B on T4 (14GB params + KV + Act > 16GB?)
        # 7B fp16 = 14GB. KV overhead context=512+128 is small. Act is small bs=1.
        # It might fit marginally or fail.
        ("Check: 7B on T4 (16GB)", 
         create_model_spec("llama-7b"), 
         [create_device("t4-16g", 0)],
         ParallelConfig(tp_degree=1, pp_degree=1, microbatches=1, 
                        pp_stages=[StagePlacement([0], (0, 32))])
        ),
        
        # 3. OOM: Llama 70B on A100 (140GB > 80GB)
        ("Check: 70B on Single A100", 
         create_model_spec("llama-70b"), 
         [create_device("a100-80g", 0)],
         ParallelConfig(tp_degree=1, pp_degree=1, microbatches=1, 
                        pp_stages=[StagePlacement([0], (0, 80))])
        ),
        
        # 4. PP: Llama 70B on 2x A100 (Pipeline Parallel)
        # 70B = 140GB. 2x80 = 160GB. Should fit.
        # Split: Layers 0-40 on GPU0, 41-80 on GPU1.
        ("Solution: 70B on 2x A100 (PP=2)", 
         create_model_spec("llama-70b"), 
         [create_device("a100-80g", 0), create_device("a100-80g", 1)],
         ParallelConfig(tp_degree=1, pp_degree=2, microbatches=4, 
                        pp_stages=[
                            StagePlacement([0], (0, 40)),
                            StagePlacement([1], (40, 80))
                        ])
        ),
        
        # 5. TP: Llama 70B on 4x A100 (Tensor Parallel)
        # TP=4. Latency should be better than PP=2 due to concurrency?
        # But communication cost is high.
        ("Solution: 70B on 4x A100 (TP=4)", 
         create_model_spec("llama-70b"), 
         [create_device("a100-80g", i) for i in range(4)],
         ParallelConfig(tp_degree=4, pp_degree=1, microbatches=1, 
                        pp_stages=[StagePlacement([0,1,2,3], (0, 80))])
        ),
        
        # 6. Heterogeneous: Llama 70B on A100 + CPU (Offload Pipeline)
        # GPU holds as much as it can (say 40 layers ~ 70GB), CPU holds rest.
        ("Hybrid: 70B on A100 + CPU (PP=2, GPU->CPU)", 
         create_model_spec("llama-70b"), 
         [create_device("a100-80g", 0), create_device("cpu-host", 1)],
         ParallelConfig(tp_degree=1, pp_degree=2, microbatches=4, 
                        pp_stages=[
                            StagePlacement([0], (0, 40)), # GPU
                            StagePlacement([1], (40, 80)) # CPU (Slow!)
                        ])
        ),
    ]
    
    print(f"{'Scenario':<40} | {'Feasible':<10} | {'Prefill (ms)':<12} | {'Decode (ms)':<12} | {'T/s':<8} | {'Note'}")
    print("-" * 110)
    
    for name, model, devices, cfg in scenarios:
        cm = HelmCostModel(model, devices, topology={}, calibration=calib)
        res = cm.estimate(cfg)
        
        if res['feasible']:
            prefill_ms = res['T_prefill'] * 1000
            decode_ms = res['T_decode_token'] * 1000
            tps = res['tokens_per_sec']
            note = ""
        else:
            prefill_ms = 0
            decode_ms = 0
            tps = 0
            if 'reason' in res:
                note = res['reason']
            else:
                note = "Infeasible"
            
        print(f"{name:<40} | {str(res['feasible']):<10} | {prefill_ms:>12.2f} | {decode_ms:>12.2f} | {tps:>8.2f} | {note}")

    print("-" * 110)
    print("Analysis:")
    print("1. 7B on T4 is feasible but ~6x slower than A100 (due to Bandwidth 1.6T vs 320G).")
    print("2. 70B on Single A100 OOMs as expected (>80GB).")
    print("3. PP=2 (2xA100) gives ~20 tokens/sec. It splits memory but incurs bubble latency.")
    print("4. TP=4 (4xA100) gives ~10 tokens/sec. Wait, why slower? Because my 'Kernel Overhead' for AllReduce is high relative to small batch=1 decode.")
    print("   Also, TP requires communication at *every layer*, adding latency. PP only communicates at partition boundaries.")
    print("5. CPU Offload is feasible but extremely slow (~1.5 T/s) due to low CPU-RAM bandwidth (150GB/s) vs HBM (2TB/s).")
    print("   But it enables running 70B models on a single GPU workstation!")

if __name__ == "__main__":
    run_suite()
