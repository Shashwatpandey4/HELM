import torch
import torch.nn as nn
import torch.fx
import os
from backend.passes.parallelism import pipeline_parallel_pass, tensor_parallel_pass
from backend.passes.hardware_analysis import hardware_analysis_pass
from backend.passes.soft_analysis import soft_analysis_pass
from torch.fx.passes.shape_prop import ShapeProp

class ToyTransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, d_model) # Simplified attention (projection)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        # Sublayer 1
        resid = x
        x = self.ln1(x)
        x = self.attn(x)
        x = x + resid
        
        # Sublayer 2
        resid = x
        x = self.ln2(x)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = x + resid
        return x

class ToyTransformer(nn.Module):
    def __init__(self, d_model=16, num_layers=2):
        super().__init__()
        self.emb = nn.Linear(10, d_model) # Dummy embedding
        self.layers = nn.ModuleList([ToyTransformerBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, 10)
        
    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x

def save_graph(gm, filename):
    os.makedirs("tests/visual_outputs", exist_ok=True)
    filepath = os.path.join("tests/visual_outputs", filename)
    with open(filepath, "w") as f:
        f.write("Metadata:\n")
        f.write(str(gm.meta))
        f.write("\n\n" + "="*20 + "\n\n")
        f.write(str(gm.graph))
        f.write("\n\n" + "="*20 + "\n\n")
        f.write(gm.code)
    print(f"Saved graph to {filepath}")

def main():
    print("Initializing Toy Transformer...")
    model = ToyTransformer()
    
    # Trace
    print("Tracing model...")
    tracer = torch.fx.Tracer()
    graph = tracer.trace(model)
    gm = torch.fx.GraphModule(model, graph)
    
    # Run Shape Propagation
    print("Running Shape Propagation...")
    example_input = torch.randn(2, 10) # Batch size 2, input dim 10 (matching embedding)
    ShapeProp(gm).propagate(example_input)
    
    save_graph(gm, "01_original.txt")
    
    # Run Hardware Analysis
    print("Applying Hardware Analysis Pass...")
    gm = hardware_analysis_pass(gm)
    save_graph(gm, "02_after_hardware.txt")
    
    # Run Soft Analysis
    print("Applying Soft Analysis Pass (FLOPs & IO)...")
    gm = soft_analysis_pass(gm)
    save_graph(gm, "03_after_soft_analysis.txt")
    
    # Run Cost Model (Heuristic)
    from backend.passes.cost_model import cost_model_pass
    print("Applying Cost Model (Heuristics)...")
    gm = cost_model_pass(gm, world_size=None) # Default
    # No graph change here, just metadata, but we can save to inspect
    save_graph(gm, "04_after_cost_model.txt")
    
    # 1. Pipeline Parallelism
    # Auto-detected from metadata
    print("Applying PP (Auto)...")
    gm = pipeline_parallel_pass(gm) # No manual split nodes
    gm.recompile()
    save_graph(gm, "05_after_pp.txt")

    # 2. Tensor Parallelism
    # Auto-detected from metadata
    print("Applying TP (Auto)...")
    gm = tensor_parallel_pass(gm) # No manual node/strategy
    gm.recompile()
    save_graph(gm, "06_after_tp.txt")

if __name__ == "__main__":
    main()
