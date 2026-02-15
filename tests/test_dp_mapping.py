import torch
import torch.nn as nn
import torch.fx
from unittest.mock import patch, MagicMock
from helm.graph import HelmGraph
from helm.passes.pipeline_split import PipelineSplitPass
from helm.pipeline import PipelineExecutor
from helm.backend.mesh import DeviceMesh

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.l2(self.l1(x))

def test_dp_mapping():
    # DP=2, PP=2, TP=1 (Total 4 devices)
    # We will simulate DP=2 by creating two Executors, one for each Replica.
    
    mesh = DeviceMesh(dp=2, pp=2, tp=1, device_type="cpu") # Mock CPU mesh for test
    
    model = SimpleModel()
    gm = torch.fx.symbolic_trace(model)
    helm_graph = HelmGraph(gm.graph)
    
    # 1. Artificially Split Graph into 2 Stages
    # Node l1 -> Stage 0
    # Node l2 -> Stage 1
    # We force metadata since we are skipping full Partitioner run
    
    # Create Submodules (like PipelineSplitPass does)
    # Actually, let's just make a dummy GraphModule that HAS submodules.
    # It's easier than running the pass.
    
    submod_0 = torch.fx.GraphModule(model, torch.fx.Graph()) # Empty dummy
    submod_0.meta_device = "cuda:0" # Original "Logical" intention
    
    submod_1 = torch.fx.GraphModule(model, torch.fx.Graph())
    submod_1.meta_device = "cuda:1"
    
    class MockSplitGM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.submod_0 = submod_0
            self.submod_1 = submod_1
            
    mock_gm = MockSplitGM()
    
    # Mock torch.cuda.Stream so it doesn't crash on non-existent GPU
    with patch('torch.cuda.Stream', return_value=MagicMock()):
        
        # Override mesh to force identity mapping (simulate enough GPUs)
        # We need to monkeypath the INSTANCE method, not the class method here easily since we pass instance.
        # But we can just set mesh.get_physical_device_id = ...
        mesh.get_physical_device_id = lambda rank: rank 
        
        # 2. Initialize Replica 0 Executor
        print("\n--- Testing Replica 0 ---")
        # Initialize
        exec0 = PipelineExecutor(mock_gm, device_mesh=mesh, replica_id=0)
        
        # Note: PipelineExecutor converts string "cuda:0" to device object.
        # torch.device("cuda:0") fails? No.
        
        print(f"Replica 0 Stages: {exec0.stages[0].device}, {exec0.stages[1].device}")
        
        # Verify
        assert exec0.stages[0].device.index == 0
        assert exec0.stages[1].device.index == 1
        
        # 3. Initialize Replica 1 Executor
        print("\n--- Testing Replica 1 ---")
        exec1 = PipelineExecutor(mock_gm, device_mesh=mesh, replica_id=1)
        
        print(f"Replica 1 Stages: {exec1.stages[0].device}, {exec1.stages[1].device}")
        
        assert exec1.stages[0].device.index == 2
        assert exec1.stages[1].device.index == 3
    
    print("✅ DP Mapping Logic Verified!")

if __name__ == "__main__":
    test_dp_mapping()
