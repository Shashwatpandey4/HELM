import torch
import torch.distributed as dist
import torch.cuda
import time
import json
import os
from typing import Dict, List, Optional

class SystemProfiler:
    """
    Profiles the hardware to measure actual:
    1. HBM Bandwidth (Device-to-Device Copy)
    2. Compute TFLOPS (FP16 GEMM)
    3. Communication Bandwidth (AllReduce / P2P)
    4. PCIe Bandwidth (Host-to-Device)
    
    Results are cached in .helm_profile.json to avoid re-running.
    """
    def __init__(self, cache_file: str = ".helm_profile.json"):
        self.cache_file = cache_file
        self.profile = {}

    def run(self, force: bool = False) -> Dict:
        if not force and os.path.exists(self.cache_file):
            print(f"[SystemProfiler] Loading profile from {self.cache_file}...")
            try:
                with open(self.cache_file, "r") as f:
                    self.profile = json.load(f)
                return self.profile
            except Exception as e:
                print(f"[SystemProfiler] Failed to load cache: {e}. Re-profiling.")
        
        print("\n[SystemProfiler] Starting Hardware Micro-benchmarks...")
        
        if not torch.cuda.is_available():
            print("[SystemProfiler] No GPU detected. Returning CPU-only profile.")
            self.profile = {"gpu_count": 0, "device_hbm": 0, "device_flops": 0, "p2p_bw": 0, "pcie_bw": 10e9} # Fallback
            return self.profile
            
        gpu_count = torch.cuda.device_count()
        self.profile["gpu_count"] = gpu_count
        
        # 1. HBM Bandwidth (Device 0)
        self.profile["device_hbm"] = self._measure_hbm(0)
        
        # 2. Compute TFLOPS (Device 0)
        self.profile["device_flops"] = self._measure_flops(0)
        
        # 3. PCIe Bandwidth (Host -> Device 0)
        self.profile["pcie_bw"] = self._measure_pcie(0)

        # 4. Inter-GPU Communication (if applicable)
        if gpu_count > 1:
            self.profile["p2p_bw"] = self._measure_p2p(0, 1) # Assumes GPUs are uniform
            # AllReduce? Need distributed initialization.
            # Skipping AllReduce benchmark for now to avoid process spawning complexity in single script.
            # Using P2P as proxy for link bandwidth.
        else:
            self.profile["p2p_bw"] = 0 # No inter-GPU comm
            
        # Save to cache
        print(f"[SystemProfiler] Saving profile to {self.cache_file}...")
        with open(self.cache_file, "w") as f:
            json.dump(self.profile, f, indent=2)
            
        return self.profile

    def _measure_hbm(self, device_id: int) -> float:
        """Measure Device HBM Bandwidth (GB/s)"""
        size_bytes = 1024 * 1024 * 256 # 256MB
        device = torch.device(f"cuda:{device_id}")
        t = torch.randn(size_bytes // 4, device=device, dtype=torch.float32)
        
        # Warmup
        t_out = t.clone()
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            t_out = t.clone()
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 10
        bw = size_bytes / avg_time
        print(f"  [HBM] {bw / 1e9:.2f} GB/s")
        return bw

    def _measure_flops(self, device_id: int) -> float:
        """Measure FP16 TFLOPS (Matrix Multiplication)"""
        N = 4096
        device = torch.device(f"cuda:{device_id}")
        a = torch.randn(N, N, device=device, dtype=torch.float16)
        b = torch.randn(N, N, device=device, dtype=torch.float16)
        
        # Warmup
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        start = time.time()
        iters = 10
        for _ in range(iters):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / iters
        flops = (2 * N**3) / avg_time
        print(f"  [Compute] {flops / 1e12:.2f} TFLOPS")
        return flops

    def _measure_pcie(self, device_id: int) -> float:
        """Measure Host-to-Device Bandwidth (GB/s)"""
        size_bytes = 1024 * 1024 * 128 # 128MB
        host_t = torch.randn(size_bytes // 4, dtype=torch.float32)
        device = torch.device(f"cuda:{device_id}")
        
        # Warmup
        dev_t = host_t.to(device)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(5):
            dev_t = host_t.to(device, non_blocking=True)
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 5
        bw = size_bytes / avg_time
        print(f"  [PCIe H2D] {bw / 1e9:.2f} GB/s")
        return bw

    def _measure_p2p(self, src: int, dst: int) -> float:
        """Measure P2P Copy Bandwidth (GB/s)"""
        size_bytes = 1024 * 1024 * 128 # 128MB
        src_dev = torch.device(f"cuda:{src}")
        dst_dev = torch.device(f"cuda:{dst}")
        
        t = torch.randn(size_bytes // 4, device=src_dev, dtype=torch.float32)
        
        # Warmup
        t_out = t.to(dst_dev)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            t_out = t.to(dst_dev)
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 10
        bw = size_bytes / avg_time
        print(f"  [P2P {src}->{dst}] {bw / 1e9:.2f} GB/s")
        return bw

if __name__ == "__main__":
    prof = SystemProfiler()
    prof.run(force=True)
