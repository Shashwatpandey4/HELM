import torch
import torch.fx
from ..graph import HelmGraph, HelmNode

class ExecutionPass:
    """
    Pass 6: Execution / Graph Mutation
    Applies the partitioning plan to the FX Graph by inserting `.to(device)` ops.
    """
    def __init__(self, graph: HelmGraph, gm: torch.fx.GraphModule, example_inputs):
        self.helm_graph = graph
        self.gm = gm
        self.example_inputs = example_inputs
        self.fx_to_helm = graph.fx_to_helm

    def run(self):
        print("\n[ExecutionPass] Applying Partition Plan to FX Graph...")
        
        # 1. Sync Placeholders with Reality
        # The Partitioner may have assigned placeholders (weights) to GPU.
        # But physically, they enter functions as CPU tensors (if model is CPU).
        # We must treat placeholders as residing on their *actual* input device
        # so that we generate the necessary transfers to the compute device.
        
        placeholders = [n for n in self.gm.graph.nodes if n.op == 'placeholder']
        
        if len(placeholders) != len(self.example_inputs):
            print(f"Warning: Placeholder count ({len(placeholders)}) != Example Inputs ({len(self.example_inputs)}). Skipping device sync.")
        else:
            for node, inp in zip(placeholders, self.example_inputs):
                if node in self.fx_to_helm:
                    helm_node = self.fx_to_helm[node]
                    
                    # Determine real device
                    if isinstance(inp, torch.Tensor):
                        real_device = str(inp.device)
                    else:
                        real_device = "cpu" # Default for non-tensors
                    
                    # Force HelmNode to match reality for the source
                    helm_node.device = real_device
                    
        # 2. Mutate Graph
        # We cache transfers to avoid duplicate .to() calls for the same value
        # Map: (Node, target_device_str) -> NewNode
        transfer_cache = {}
        
        insertions = 0
        
        for node in self.gm.graph.nodes:
            if node.op == 'output':
                # Output device policy: Leave it on the device of the result?
                # Or force to CPU? For now, let's leave it.
                continue
                
            if node.op == 'placeholder':
                continue
                
            if node not in self.fx_to_helm:
                continue
                
            helm_node = self.fx_to_helm[node]
            target_device = helm_node.device
            
            # Helper to transform args
            def transform_arg(arg):
                nonlocal insertions
                if isinstance(arg, torch.fx.Node):
                    if arg in self.fx_to_helm:
                        source_helm = self.fx_to_helm[arg]
                        source_device = source_helm.device
                        
                        # Normalize device strings (cuda:0 vs cuda)
                        s_dev = source_device.split(':')[0] if 'cuda' in source_device else source_device
                        t_dev = target_device.split(':')[0] if 'cuda' in target_device else target_device
                        
                        if s_dev != t_dev:
                            # NEED TRANSFER
                            key = (arg, target_device)
                            if key in transfer_cache:
                                return transfer_cache[key]
                            
                            # Insert .to()
                            # We insert after the definition of 'arg' to keep top-sort valid
                            # BUT if we are processing 'node', we are after 'arg'.
                            # We can just insert before 'node'.
                            
                            with self.gm.graph.inserting_before(node):
                                new_node = self.gm.graph.call_method("to", args=(arg,), kwargs={"device": target_device})
                                
                                # We should index this new node in transfer_cache
                                transfer_cache[key] = new_node
                                insertions += 1
                                return new_node
                                
                    return arg
                else:
                    return arg
            
            # Update Args
            new_args = torch.fx.node.map_arg(node.args, transform_arg)
            new_kwargs = torch.fx.node.map_arg(node.kwargs, transform_arg)
            
            node.args = new_args
            node.kwargs = new_kwargs
            
        print(f"  Inserted {insertions} device transfer ops.")
        
        # 3. Recompile
        self.gm.recompile()
        # print(self.gm.code) # Debug
