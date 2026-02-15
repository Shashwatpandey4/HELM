import torch
import torch.nn as nn
from typing import Optional

class QuantizedLinear(nn.Module):
    """
    INT8 Weight-Only Quantized Linear Layer.
    
    Quantization Strategy:
    - Weights: INT8 per-channel (symmetric quantization)
    - Activations: FP16/FP32 (no activation quantization)
    - Forward: Dequantize weights -> FP matmul
    
    This is suitable for inference where memory is the bottleneck.
    Compute is still FP16, so no custom CUDA kernels needed.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Quantized weight (INT8)
        self.register_buffer(
            'weight_quantized',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        # Per-channel scales (FP16 for efficiency)
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, dtype=torch.float16)
        )
        
        # Bias (FP16)
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with runtime dequantization.
        """
        # Dequantize weights: W_fp = W_int8 * scale
        weight_fp = self.weight_quantized.to(x.dtype) * self.weight_scales.unsqueeze(1).to(x.dtype)
        
        # Standard linear operation
        output = torch.nn.functional.linear(x, weight_fp, self.bias.to(x.dtype) if self.bias is not None else None)
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8):
        """
        Convert a standard nn.Linear to QuantizedLinear.
        
        Quantization Formula (Symmetric):
            scale = max(abs(W)) / 127
            W_int8 = round(W / scale).clamp(-128, 127)
        """
        quantized = cls(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            bits=bits
        )
        
        # Quantize weights per output channel (row)
        weight_fp = linear.weight.data
        
        # Compute per-channel scales
        max_vals = weight_fp.abs().max(dim=1, keepdim=True)[0]
        scales = max_vals / 127.0  # Symmetric quantization
        scales = scales.squeeze(1).to(torch.float16)
        
        # Avoid division by zero
        scales = torch.clamp(scales, min=1e-8)
        
        # Quantize
        weight_int8 = torch.round(weight_fp / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
        
        # Store
        quantized.weight_quantized.copy_(weight_int8)
        quantized.weight_scales.copy_(scales)
        
        if linear.bias is not None:
            quantized.bias.copy_(linear.bias.to(torch.float16))
            
        return quantized
    
    def memory_footprint(self) -> int:
        """Return memory usage in bytes."""
        weight_bytes = self.weight_quantized.numel() * 1  # INT8 = 1 byte
        scale_bytes = self.weight_scales.numel() * 2  # FP16 = 2 bytes
        bias_bytes = self.bias.numel() * 2 if self.bias is not None else 0
        return weight_bytes + scale_bytes + bias_bytes


def quantize_model_int8(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    Recursively quantize all nn.Linear layers in a model to INT8.
    
    Args:
        model: PyTorch model to quantize
        inplace: If True, modify model in place. Otherwise return a copy.
    
    Returns:
        Quantized model
    """
    if not inplace:
        model = model.copy()
    
    # Recursively replace Linear layers
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            # Replace with QuantizedLinear
            quantized = QuantizedLinear.from_linear(module, bits=8)
            setattr(model, name, quantized)
            print(f"  Quantized: {name} ({module.in_features}x{module.out_features})")
        else:
            # Recurse
            quantize_model_int8(module, inplace=True)
    
    return model


def estimate_memory_savings(model: nn.Module) -> dict:
    """
    Estimate memory savings from INT8 quantization.
    
    Returns dict with:
        - original_mb: Original model size in MB
        - quantized_mb: Quantized model size in MB
        - savings_pct: Percentage reduction
    """
    original_bytes = 0
    quantized_bytes = 0
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Original: FP16 weights + bias
            original_bytes += module.weight.numel() * 2
            if module.bias is not None:
                original_bytes += module.bias.numel() * 2
            
            # Quantized: INT8 weights + FP16 scales + FP16 bias
            quantized_bytes += module.weight.numel() * 1  # INT8
            quantized_bytes += module.weight.shape[0] * 2  # Scales (per output channel)
            if module.bias is not None:
                quantized_bytes += module.bias.numel() * 2
        elif isinstance(module, QuantizedLinear):
            quantized_bytes += module.memory_footprint()
    
    original_mb = original_bytes / (1024 ** 2)
    quantized_mb = quantized_bytes / (1024 ** 2)
    savings_pct = (1 - quantized_mb / original_mb) * 100 if original_mb > 0 else 0
    
    return {
        'original_mb': original_mb,
        'quantized_mb': quantized_mb,
        'savings_pct': savings_pct
    }
