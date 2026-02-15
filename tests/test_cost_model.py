from helm.passes.cost_model import *
import pytest

def test_cost_model_basic():
    # 1. Setup Mock Hardware
    dev0 = DeviceSpec(id=0, peak_flops=312e12, mem_bw=1.6e12, mem_capacity=80e9) # A100-80G
    dev1 = DeviceSpec(id=1, peak_flops=312e12, mem_bw=1.6e12, mem_capacity=80e9) 
    
    # 2. Setup Mock Model (Qwen-7B approx)
    # 7B Params ~ 14GB FP16
    # 32 Layers
    model = ModelSpec(
        L=32, layers=[], dtype="fp16", n_heads=32, hidden_size=4096, vocab_size=32000,
        seq_len_prefill=512, batch_size=1, gen_len=128
    )
    
    # Fill layers
    for _ in range(32):
        l = LayerSpec(flops_prefill=1e9, flops_decode_token=1e6, 
                      bytes_moved_prefill=1e6, bytes_moved_decode=1e4,
                      activation_bytes=1e6, param_bytes=400e6, kv_bytes_per_token=1e3)
        model.layers.append(l)
        
    # 3. Setup Config: PP=2, TP=1
    # Stage 0: Layers 0-15 on Dev0
    # Stage 1: Layers 16-31 on Dev1
    
    stage0 = StagePlacement(devices=[0], layer_range=(0, 16))
    stage1 = StagePlacement(devices=[1], layer_range=(16, 32))
    
    cfg = ParallelConfig(tp_degree=1, pp_degree=2, microbatches=4, pp_stages=[stage0, stage1])
    
    # 4. Run Model
    cm = HelmCostModel(model, devices=[dev0, dev1], topology={}, calibration=CalibrationDB())
    
    result = cm.estimate(cfg)
    
    print("\n--- Cost Model Result ---")
    print(result)
    
    assert result['feasible'] == True
    assert result['T_total'] > 0

def test_oom_check():
    # Force OOM: Tiny GPU
    dev0 = DeviceSpec(id=0, peak_flops=312e12, mem_bw=1.6e12, mem_capacity=1e9) # 1GB
    model = ModelSpec(L=32, layers=[], dtype="fp16", n_heads=32, hidden_size=4096, vocab_size=32000,
                      seq_len_prefill=512, batch_size=1, gen_len=128)
    for _ in range(32):
        l = LayerSpec(flops_prefill=1, flops_decode_token=1, bytes_moved_prefill=1, bytes_moved_decode=1,
                      activation_bytes=1e6, param_bytes=400e6, kv_bytes_per_token=1e3) # 400MB params per layer
        model.layers.append(l)
        
    stage0 = StagePlacement(devices=[0], layer_range=(0, 32)) # All on Dev0 -> 12GB needed
    cfg = ParallelConfig(tp_degree=1, pp_degree=1, microbatches=1, pp_stages=[stage0])
    
    cm = HelmCostModel(model, devices=[dev0], topology={}, calibration=CalibrationDB())
    result = cm.estimate(cfg)
    
    print("\n--- OOM Check Result ---")
    print(result)
    
    assert result['feasible'] == False
    assert "OOM" in result['reason']

if __name__ == "__main__":
    test_cost_model_basic()
    test_oom_check()
