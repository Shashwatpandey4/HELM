import torch
import torch.nn as nn
import torch.fx
from helm.graph import HelmGraph
from helm.passes.pipeline_split import PipelineSplitPass
from helm.passes.execution import ExecutionPass
from helm.pipeline import PipelineExecutor

# --- Mock Partitioner (Same as before) ---
class ForcedSplitPartitioner:
    def __init__(self, graph: HelmGraph):
        self.graph = graph
        
    def run(self):
        print("\n[ForcedSplitPartitioner] Creating artificial split...")
        count = 0
        limit = len(self.graph.nodes) // 2
        for node in self.graph.nodes:
            if count < limit:
                node.device = "cuda:0"
            else:
                node.device = "cpu" 
            count += 1
        print(f"  Split logic applied: First half -> cuda:0, Second half -> cpu")

def simple_fn(x, w):
    # Stage 0
    a = torch.matmul(x, w) 
    b = torch.relu(a)
    # Stage 1
    c = b * 2
    d = c + 1
    return d

class Model(nn.Module):
    def forward(self, x, w):
        return simple_fn(x, w)

def test_pipeline_executor():
    print("Starting Pipeline Executor Integration Test...")
    
    # 1. Setup
    model = Model()
    gm = torch.fx.symbolic_trace(model)
    helm_graph = HelmGraph(gm.graph)
    
    # 2. Partition
    partitioner = ForcedSplitPartitioner(helm_graph)
    partitioner.run()
    
    # 3. Execution Pass (Insert .to)
    # Use distinct dimensions to avoid accidental splitting of weights in executor heuristic
    # Batch=16, InputDim=32. Weights=32x16.
    inputs = [torch.randn(16, 32, device='cuda:0'), torch.randn(32, 16, device='cuda:0')]
    ExecutionPass(helm_graph, gm, inputs).run()
    
    # 4. Split
    PipelineSplitPass(helm_graph, gm).run()
    
    # 5. Initialize Executor
    # Microbatch size 4 (so 16 -> 4 mbs)
    executor = PipelineExecutor(gm, micro_batch_size=4)
    
    # 6. Run
    # Run forward (warmup if needed, no just run)
    x_full, w_full = inputs
    
    print("\n[Test] Running Pipeline Executor...")
    pipeline_output = executor.run_forward(x_full, w_full)
    
    # 7. Verification
    print("\n[Test] Verifying Output...")
    with torch.no_grad():
        expected = model(x_full, w_full)
    
    # Check if output is on device of last stage (CPU) or collected?
    # Our executor collation returns it as constructed.
    # Stage 1 is CPU in this test, so output should be CPU tensor.
    # Expected is CUDA tensor (since model ran on CUDA inputs).
    
    pipeline_out_cuda = pipeline_output.to('cuda:0')
    
    if torch.allclose(expected, pipeline_out_cuda, atol=1e-5):
        print("✅ Pipeline Executor Output Matches Reference!")
    else:
        print("❌ Output Mismatch!")
        # Debug
        print(f"Expected shape: {expected.shape}")
        print(f"Actual shape: {pipeline_output.shape}")
        if expected.shape == pipeline_output.shape:
             print("Expected:", expected[0, :5])
             print("Actual (moved to cuda):", pipeline_out_cuda[0, :5])

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_pipeline_executor()
    else:
        print("Skipping test (requires CUDA)")
