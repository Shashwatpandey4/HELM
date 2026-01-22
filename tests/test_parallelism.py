import torch
import torch.nn as nn
import torch.fx
from backend.passes.parallelism import pipeline_parallel_pass, tensor_parallel_pass

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def test_pipeline_parallel():
    print("Testing Pipeline Parallelism...")
    model = SimpleMLP()
    tracer = torch.fx.Tracer()
    graph = tracer.trace(model)
    gm = torch.fx.GraphModule(model, graph)
    
    # Identify nodes
    nodes = list(gm.graph.nodes)
    # Split after relu
    # fc1 -> relu (stage 0) | fc2 (stage 1)
    # nodes: input, fc1, relu, fc2, output
    
    # Let's find relu node
    split_node = None
    for n in nodes:
        if n.target == "relu" or (n.op == 'call_module' and 'relu' in n.target):
             split_node = n
             break
             
    if split_node:
        print(f"Splitting after node: {split_node.name}")
        gm = pipeline_parallel_pass(gm, [split_node])
        
        gm.recompile()
        print("Graph Nodes:", [n.name for n in gm.graph.nodes])
        print(gm.code)
        
        # Verify
        code = gm.code
        if "dist_send" in code and "dist_recv" in code:
            print("PASS: found dist_send and dist_recv")
        else:
            print("FAIL: missing dist_send or dist_recv")
    else:
        print("FAIL: Could not find split node")

def test_tensor_parallel():
    print("\nTesting Tensor Parallelism...")
    model = SimpleMLP()
    tracer = torch.fx.Tracer()
    graph = tracer.trace(model)
    gm = torch.fx.GraphModule(model, graph)
    
    # Target fc1 for TP
    target_node = None
    for n in gm.graph.nodes:
        if n.op == 'call_module' and 'fc1' in n.target:
            target_node = n
            break
            
    if target_node:
        print(f"Applying TP to node: {target_node.name}")
        gm = tensor_parallel_pass(gm, target_node, strategy='col')
        gm.recompile()
        print(gm.code)
        
        code = gm.code
        if "dist_all_reduce" in code:
             print("PASS: found dist_all_reduce")
        else:
             print("FAIL: missing dist_all_reduce")
    else:
        print("FAIL: Could not find target node")

if __name__ == "__main__":
    test_pipeline_parallel()
    test_tensor_parallel()
