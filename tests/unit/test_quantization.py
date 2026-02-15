import torch
import torch.nn as nn
from helm.core.quantization import QuantizedLinear, quantize_model_int8, estimate_memory_savings

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_quantized_linear_creation():
    """Test creating QuantizedLinear from nn.Linear."""
    linear = nn.Linear(128, 512)
    quantized = QuantizedLinear.from_linear(linear, bits=8)
    
    assert quantized.in_features == 128
    assert quantized.out_features == 512
    assert quantized.weight_quantized.dtype == torch.int8
    assert quantized.weight_scales.dtype == torch.float16
    print("✓ QuantizedLinear creation works")


def test_quantized_linear_forward():
    """Test forward pass numerical accuracy."""
    torch.manual_seed(42)
    linear = nn.Linear(128, 512)
    quantized = QuantizedLinear.from_linear(linear, bits=8)
    
    # Test input
    x = torch.randn(4, 128)
    
    # Original output
    with torch.no_grad():
        orig_out = linear(x)
        quant_out = quantized(x)
    
    # Check outputs are close (INT8 has quantization error)
    max_diff = (orig_out - quant_out).abs().max().item()
    mean_diff = (orig_out - quant_out).abs().mean().item()
    
    print(f"  Max diff: {max_diff:.4f}")
    print(f"  Mean diff: {mean_diff:.4f}")
    
    # Tolerance: INT8 quantization should be within ~1% error
    assert mean_diff < 0.1, f"Mean error too high: {mean_diff}"
    print("✓ QuantizedLinear forward pass is accurate")


def test_quantize_model():
    """Test quantizing entire model."""
    model = SimpleModel()
    
    # Count original Linear layers
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    assert linear_count == 2
    
    # Quantize
    quantize_model_int8(model, inplace=True)
    
    # Count quantized layers
    quant_count = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    assert quant_count == 2
    
    print(f"✓ Quantized {quant_count} Linear layers")


def test_memory_savings():
    """Test memory estimation."""
    model = SimpleModel()
    savings = estimate_memory_savings(model)
    
    print(f"  Original: {savings['original_mb']:.2f} MB")
    print(f"  Quantized: {savings['quantized_mb']:.2f} MB")
    print(f"  Savings: {savings['savings_pct']:.1f}%")
    
    # Should save ~40-50% (INT8 vs FP16)
    assert savings['savings_pct'] > 30, "Should save at least 30%"
    assert savings['savings_pct'] < 60, "Savings should be realistic"
    print("✓ Memory savings estimation is reasonable")


def test_quantized_model_inference():
    """Test end-to-end inference with quantized model."""
    torch.manual_seed(42)
    
    # Original model
    model_orig = SimpleModel()
    model_orig.eval()
    
    # Quantized model
    model_quant = SimpleModel()
    model_quant.load_state_dict(model_orig.state_dict())
    quantize_model_int8(model_quant, inplace=True)
    model_quant.eval()
    
    # Test input
    x = torch.randn(8, 128)
    
    with torch.no_grad():
        out_orig = model_orig(x)
        out_quant = model_quant(x)
    
    # Check similarity
    mse = ((out_orig - out_quant) ** 2).mean().item()
    print(f"  MSE: {mse:.6f}")
    
    assert mse < 0.01, f"MSE too high: {mse}"
    print("✓ Quantized model inference is accurate")


if __name__ == "__main__":
    test_quantized_linear_creation()
    test_quantized_linear_forward()
    test_quantize_model()
    test_memory_savings()
    test_quantized_model_inference()
    print("\n✅ All quantization tests passed!")
