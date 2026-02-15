import torch
import torch.fx
from ..graph import HelmGraph, HelmNode

class PipelineSplitPass:
    """
    Pass: Pipeline Splitting
    Splits the main GraphModule into multiple submodules (StageModules)
    based on the assigned devices of the nodes.
    """
    def __init__(self, graph: HelmGraph, gm: torch.fx.GraphModule):
        self.helm_graph = graph
        self.gm = gm
        self.fx_to_helm = graph.fx_to_helm
    
    def run(self):
        print("\n[PipelineSplitPass] Splitting Graph into Stages...")
        
        # 1. Identify Stages (Contiguous blocks of same device)
        stages = [] # List of {'device': str, 'nodes': [fx_node], 'name': str}
        current_stage = None
        stage_idx = 0
        
        # Helper to get stage name
        def get_stage_name():
            nonlocal stage_idx
            name = f"submod_{stage_idx}"
            stage_idx += 1
            return name

        # Helper to infer device of an FX Node
        def get_device(fx_node):
            # 1. Check HelmGraph (legacy/original nodes)
            if fx_node in self.helm_graph.fx_to_helm:
                return self.helm_graph.fx_to_helm[fx_node].device
            
            # 2. Check if it's a new .to() node
            if fx_node.op == 'call_method' and fx_node.target == 'to':
                # Try args/kwargs for device
                # args[0] is tensor, args[1] might be device, or kwargs['device']
                dev = fx_node.kwargs.get('device')
                if dev:
                    return str(dev)
                # Check args
                if len(fx_node.args) > 1:
                    # heuristic: 2nd arg is often device/dtype
                    # tricky to guess if it's device object or string
                    pass
            
            # 3. Fallback: Inherit from previous stage (or default to cpu)
            # This is risky but often correct for ops inserted *between* stages
            return None

        # Iterate FX Graph directly
        for fx_node in self.gm.graph.nodes:
            if fx_node.op == 'output':
                continue
                
            device = get_device(fx_node)
            
            if device is None:
                # If we couldn't determine device, assume it belongs to current stage
                if current_stage:
                    device = current_stage['device']
                else:
                    device = "cpu" # Default start
            
            # Stage Boundaries
            if current_stage is None or current_stage['device'] != device:
                # New Stage
                if current_stage:
                    stages.append(current_stage)
                current_stage = {'device': device, 'nodes': [], 'name': get_stage_name()}
                
            current_stage['nodes'].append(fx_node)
            
        if current_stage:
            stages.append(current_stage)
            
        print(f"  Identified {len(stages)} pipeline stages.")
        # Print details
        # for idx, stage in enumerate(stages):
        #    print(f"    Stage {idx}: {stage['device']} ({len(stage['nodes'])} nodes)")

        if len(stages) <= 1:
            print("  Single stage detected. Skipping physical split.")
            return

        # 2. Build Submodules
        # We need to create a new Graph for each stage.
        # And a new Main Graph that calls them.
        
        new_main_graph = torch.fx.Graph()
        # Map original nodes to new main graph nodes (if they become outputs of stages)
        node_remap = {} 
        
        # Copy inputs to main graph placeholders
        input_nodes = [n for n in self.gm.graph.nodes if n.op == 'placeholder']
        for inp in input_nodes:
            new_node = new_main_graph.placeholder(inp.name)
            node_remap[inp] = new_node

        for stage_info in stages:
            stage_nodes = stage_info['nodes']
            stage_device = stage_info['device']
            stage_name = stage_info['name']
            
            # --- Create Stage Graph ---
            sub_graph = torch.fx.Graph()
            # Map original nodes to sub_graph nodes
            sub_node_map = {}
            
            # Identify Inputs: Nodes used in this stage but defined BEFORE (in prev stages or global inputs)
            stage_inputs = []
            for node in stage_nodes:
                for arg in node.all_input_nodes: # This gets fx nodes if using standard FX, but HelmNode has wrapper
                     # We need to check actual FX args
                     pass
                
                # Scan args for dependencies outside this stage
                def check_dep(target):
                    if isinstance(target, torch.fx.Node):
                        if target not in stage_nodes:
                            # External input!
                            if target not in sub_node_map:
                                # Create placeholder in sub_graph
                                ph = sub_graph.placeholder(target.name)
                                sub_node_map[target] = ph
                                stage_inputs.append(target)
                
                torch.fx.node.map_arg(node.args, check_dep)
                torch.fx.node.map_arg(node.kwargs, check_dep)

            # Copy Nodes to Sub Graph
            for node in stage_nodes:
                if node.op == 'placeholder':
                    # If it's a global placeholder used in this stage
                    if node not in sub_node_map:
                        # Add it as an input to this stage
                        ph = sub_graph.placeholder(node.name)
                        sub_node_map[node] = ph
                        stage_inputs.append(node)
                    continue
                if node.op == 'output':
                    # We handle output at the end of stage
                    continue
                
                # Clone node
                new_node = sub_graph.node_copy(node, lambda x: sub_node_map.get(x, x))
                sub_node_map[node] = new_node
            
            # Identify Outputs: Nodes defined here but used LATER
            stage_outputs = []
            for node in stage_nodes:
                for user in node.users:
                    if user not in stage_nodes:
                        # Used outside!
                        stage_outputs.append(node)
                        break
            
            # Deduplicate outputs
            stage_outputs = list(dict.fromkeys(stage_outputs))
            
            # Create Output Node in Sub Graph
            output_values = tuple(sub_node_map[n] for n in stage_outputs)
            if len(output_values) == 1:
                sub_graph.output(output_values[0])
            else:
                sub_graph.output(output_values)
                
            # Compile Submodule
            sub_gm = torch.fx.GraphModule(self.gm, sub_graph, class_name=stage_name)
            
            # Attach Metadata for Runtime
            sub_gm.meta_device = stage_device
            print(f"    Submodule {stage_name} attached to device: {stage_device}")
            
            # Register Submodule in Main Module
            self.gm.add_submodule(stage_name, sub_gm)
            
            # --- Add Call in Main Graph ---
            # Prepare args from main graph
            call_args = tuple(node_remap[n] for n in stage_inputs)
            
            call_node = new_main_graph.call_module(stage_name, args=call_args)
            
            # Map outputs for next stages
            if len(stage_outputs) == 1:
                node_remap[stage_outputs[0]] = call_node
            else:
                for i, out_node in enumerate(stage_outputs):
                    # We need getitem to unpack tuple
                    val_node = new_main_graph.call_function(import_getitem, args=(call_node, i))
                    node_remap[out_node] = val_node
                    
            print(f"    Created {stage_name}: Inputs={len(stage_inputs)}, Outputs={len(stage_outputs)}")

        # Handle Global Output
        orig_output = [n for n in self.gm.graph.nodes if n.op == 'output'][0]
        # Map args
        new_output_args = torch.fx.node.map_arg(orig_output.args, lambda n: node_remap.get(n, n))
        new_main_graph.output(new_output_args[0])
        
        # Replace Main Graph
        self.gm.graph = new_main_graph
        self.gm.recompile()
        print("[PipelineSplitPass] Graph splitting complete.")

# Helper for getitem
import operator
def import_getitem(a, b):
    return operator.getitem(a, b)

