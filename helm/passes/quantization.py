from ..graph import HelmGraph, HelmNode
import torch.nn as nn
import torch

class QuantizationPass:
    """
    Quantization Pass: Converts model to lower precision.
    
    Supported dtypes:
    - fp16, bf16, fp32: Standard dtype conversion
    - int8: Weight-only INT8 quantization (per-channel)
    - int4: Placeholder (falls back to INT8)
    """
    def __init__(self, graph: HelmGraph, gm: torch.fx.GraphModule, dtype: str = "fp16"):
        self.graph = graph
        self.gm = gm
        self.dtype = dtype

    def run(self):
        print(f"[QuantizationPass] Converting model to {self.dtype}...")
        
        if self.dtype in ["fp16", "bf16", "fp32"]:
            # Standard dtype conversion
            target_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32
            }[self.dtype]
            
            for name, module in self.gm.named_modules():
                if len(list(module.parameters(recurse=False))) > 0:
                    module.to(dtype=target_dtype)
                    
            print(f"[QuantizationPass] Converted to {self.dtype}.")
            
        elif self.dtype == "int8":
            # INT8 weight-only quantization
            from ..quantization import quantize_model_int8, estimate_memory_savings
            
            # Estimate savings
            savings = estimate_memory_savings(self.gm)
            print(f"  Original size: {savings['original_mb']:.2f} MB")
            print(f"  Quantized size: {savings['quantized_mb']:.2f} MB")
            print(f"  Savings: {savings['savings_pct']:.1f}%")
            
            # Apply quantization
            quantize_model_int8(self.gm, inplace=True)
            print(f"[QuantizationPass] INT8 quantization complete.")
            
        elif self.dtype == "int4":
            # INT4 placeholder
            print(f"[QuantizationPass] WARNING: INT4 quantization requires custom CUDA kernels.")
            print(f"  Falling back to INT8 for now.")
            from ..quantization import quantize_model_int8
            quantize_model_int8(self.gm, inplace=True)
            
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
