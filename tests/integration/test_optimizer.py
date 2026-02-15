from helm.optimization.cost_model import *
from helm.optimization.optimizer import ParallelOptimizer
import pytest

def test_optimizer_basic():
    # 1. Setup Mock Hardware
    dev0 = DeviceSpec(id=0, peak_flops=312e12, mem_bw=1.6e12, mem_capacity=80e9) 
    dev1 = DeviceSpec(id=1, peak_flops=312e12, mem_bw=1.6e12, mem_capacity=80e9) 
    
    # 2. Setup Mock Model (70B, fits on 2x A100)
    model = ModelSpec(
        L=80, layers=[], dtype="fp16", n_heads=64, hidden_size=8192, vocab_size=32000,
        seq_len_prefill=512, batch_size=1, gen_len=128
    )
    
    for _ in range(80):
        l = LayerSpec(flops_prefill=1e9, flops_decode_token=1e6, 
                      bytes_moved_prefill=1e6, bytes_moved_decode=1e4,
                      activation_bytes=1e6, param_bytes=140e9/80, kv_bytes_per_token=1e3)
        model.layers.append(l)
        
    optimizer = ParallelOptimizer(model, [dev0, dev1], topology={}, calibration=CalibrationDB())
    
    # 3. Optimize
    # Expect: PP=2 or TP=2 (Since Single A100 OOMs)
    best_cfg = optimizer.optimize()
    
    print("\n--- Optimizer Result ---")
    if best_cfg:
        print(f"Best: TP={best_cfg.tp_degree}, PP={best_cfg.pp_degree}")
        for s in best_cfg.pp_stages:
            print(f"  Stage: {s.layer_range} on {s.devices}")
            
    assert best_cfg is not None
    assert best_cfg.tp_degree * best_cfg.pp_degree <= 2
    # Ensure it didn't pick single GPU (infeasible)
    assert not (best_cfg.pp_degree == 1 and best_cfg.tp_degree == 1)

if __name__ == "__main__":
    test_optimizer_basic()
