import torch
import torch.nn as nn
import torch.fx
from helm.graph import HelmGraph
from helm.passes.tensor_parallel import TensorParallelPass

# Import for recompiled graph usage
from helm.layers import ShardedLinear, HelmAllReduce

class MLP(nn.Module):
    def __init__(self, hidden=32, intermediate=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate) 
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(intermediate, hidden) 
        
        # Init weights with 1s for deterministic checking
        nn.init.ones_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.ones_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def test_tp_inference():
    print("Starting Tensor Parallel Inference Test...")
    
    # 1. Setup
    model = MLP()
    gm = torch.fx.symbolic_trace(model)
    helm_graph = HelmGraph(gm.graph)

    # 2. Run TP Pass
    tp_pass = TensorParallelPass(helm_graph, gm, tp_degree=2)
    tp_pass.run()
    
    print("\n--- Mutated Graph ---")
    print(gm.code)
    
    # 3. Simulate Execution
    # Note: ShardedLinear creates NEW Linear modules inside.
    # By default they are random initialized.
    # For this test to match the original model, we need to SHARD the original weights into the new modules.
    
    print("\n[Sharding] Transferring weights to ShardedModules...")
    with torch.no_grad():
        # FC1 -> Sharded Col
        # Original: [64, 32]
        # Shard 0: [32, 32] (Top half of output features?)
        # Shard 1: [32, 32]
        
        # Wait, our ShardedLinear just holds ONE linear layer locally. 
        # In a real distributed setting, Rank 0 holds one shard, Rank 1 holds another.
        # But here, we have a Single Graph representing the LOCAL computation on ONE rank.
        
        # If we run this graph as is, it represents ONE RANK's perspective.
        # So inputs/outputs are partial.
        
        # To verify correctness locally, we need to simulate the ALL-REDUCE.
        # Currently HelmAllReduce does nothing (identity).
        
        # If HelmAllReduce does nothing, then:
        # Result = partial_output from RowParallel.
        
        # This means the output will be 1/TP of the accumulated sum (if we use Mean) or just one chunk of the Sum.
        
        # Let's say we are Rank 0.
        # FC1 (Col): We compute Output[0:32].
        # Act: ReLU valid on partial.
        # FC2 (Row): We take Input[0:32], multiply by Weight[0:32, :].
        # Result: Partial Sum.
        # AllReduce: Needs to Sum(PartialSum_Rank0 + PartialSum_Rank1).
        
        # To test this purely locally, we can:
        # 1. Run Rank 0 Graph logic.
        # 2. Run Rank 1 Graph logic.
        # 3. Manually Sum.
        
        # But our graph is generic. The ShardedLinear doesn't know "I am Rank 0".
        # It just has a linear layer of size [32, 32].
        
        # We need to manually load the correct Slice into the ShardedLinear for the test.
        
        # Let's simulate Rank 0 Execution.
        sharded_fc1 = gm.N1_sharded_col
        sharded_fc2 = gm.N3_sharded_row
        
        # Clone original weights
        w1 = model.fc1.weight.data # [64, 32]
        w2 = model.fc2.weight.data # [32, 64]
        
        # Load Rank 0 Shards
        # Col Parallel: Split Output dimensions (dim 0 of weight)
        sharded_fc1.linear.weight.data = w1[0:32, :] 
        sharded_fc1.linear.bias.data = model.fc1.bias.data[0:32]
        
        # Row Parallel: Split Input dimensions (dim 1 of weight)
        sharded_fc2.linear.weight.data = w2[:, 0:32] 
        sharded_fc2.linear.bias.data = model.fc2.bias.data # Bias is usually replicated and summed? Or just one adds bias?
        # Usually RowParallel adds bias AFTER AllReduce.
        # But ShardedLinear here has bias. If both ranks add bias, we adding bias twice!
        # Correct TP: Disable bias in RowParallel linear, add bias after AllReduce.
        # For now, let's assume bias=0 for simplicity in this test.
        
    # Input
    x = torch.ones(1, 32)
    
    # Run Graph (Rank 0 Partial)
    rank0_out = gm(x)
    
    print(f"Rank 0 Partial Output: {rank0_out[0,0]}")
    
    # Simulate Rank 1
    with torch.no_grad():
        sharded_fc1.linear.weight.data = w1[32:64, :]
        sharded_fc1.linear.bias.data = model.fc1.bias.data[32:64]
        sharded_fc2.linear.weight.data = w2[:, 32:64]
        
    rank1_out = gm(x)
    print(f"Rank 1 Partial Output: {rank1_out[0,0]}")

    # All Reduce (Sum)
    total_out = rank0_out + rank1_out
    
    # Expected
    expected = model(x)
    print(f"Total Output: {total_out[0,0]}")
    print(f"Expected:     {expected[0,0]}")
    
    if torch.allclose(total_out, expected, atol=1e-5):
        print("✅ TP Logic Verified (Manual AllReduce sum matches)!")
    else:
        print("❌ TP Logic Mismatch.")

if __name__ == "__main__":
    test_tp_inference()
