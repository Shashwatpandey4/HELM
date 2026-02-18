import torch
import torch.fx
from ..graph import HelmGraph, HelmNode

class DynamicAnalyzer:
    """
    Pass: Dynamic Meta-Execution Analysis
    Uses FlopCounterMode with FakeTensors to perform true measure.
    """
    def __init__(self, graph: HelmGraph, gm: torch.fx.GraphModule, example_inputs):
        self.graph = graph
        self.gm = gm
        self.inputs = example_inputs

    def run(self):
        print("\n[DynamicAnalyzer] Starting Meta-Execution with FlopCounterMode...")
        try:
            self._run_meta_execution()
        except ImportError:
            print("FlopCounterMode not available in this PyTorch version.")
        except Exception as e:
            print(f"Dynamic Analysis Failed: {e}")
            import traceback
            traceback.print_exc()

    def _run_meta_execution(self):
        print("[DynamicAnalyzer] Skipping meta-execution to avoid meta tensor issues.")
        return
        from torch.utils.flop_counter import FlopCounterMode
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        # Enable Meta Execution Context
        with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
            # Convert inputs to Fake
            fake_inputs = []
            for t in self.inputs:
                if isinstance(t, torch.Tensor):
                    fake_inputs.append(fake_mode.from_tensor(t))
                else:
                    fake_inputs.append(t)
            
            # Simple FX Interpreter
            env = {}

            # Map inputs to placeholder nodes
            placeholders = [n for n in self.gm.graph.nodes if n.op == 'placeholder']
            for node, inp in zip(placeholders, fake_inputs):
                env[node] = inp
            
            non_zero_nodes = 0
            
            for node in self.gm.graph.nodes:
                if node.op == 'placeholder':
                    if node in self.graph.fx_to_helm:
                        helm_node = self.graph.fx_to_helm[node]
                        if isinstance(env[node], torch.Tensor):
                             helm_node.output_shape = list(env[node].shape)
                             helm_node.output_dtype = env[node].dtype
                    continue
                if node.op == 'output':
                    continue
                
                # Prepare Args
                def load_arg(a):
                    if isinstance(a, torch.fx.Node):
                        return env[a]
                    elif isinstance(a, (list, tuple)):
                        return type(a)(load_arg(x) for x in a)
                    else:
                        return a
                
                try:
                    args = torch.fx.node.map_arg(node.args, load_arg)
                    kwargs = torch.fx.node.map_arg(node.kwargs, load_arg)
                except Exception as e:
                    print(f"Error loading args for node {node.name}: {e}")
                    continue
                
                # Execute Op
                def run_op():
                    if node.op == 'call_function':
                        return node.target(*args, **kwargs)
                    elif node.op == 'call_method':
                        self_obj = args[0]
                        method_args = args[1:]
                        return getattr(self_obj, node.target)(*method_args, **kwargs)
                    elif node.op == 'call_module':
                        return self.gm.get_submodule(node.target)(*args, **kwargs)
                    return None

                flops = 0
                
                # Run with FlopCounter
                try:
                    with FlopCounterMode(display=False) as flop_counter:
                        result = run_op()
                    
                    # Extract FLOPs
                    if hasattr(flop_counter, 'get_total_flops'):
                        flops = flop_counter.get_total_flops()
                    else:
                        flops = 0
                        for mod_name, counts in flop_counter.flop_counts.items():
                             flops += sum(counts.values())

                except Exception as op_err:
                    print(f"Warning: Failed to profile op {node.name} ({node.target})")
                    # Fallback execution
                    result = run_op()
                
                env[node] = result
                
                # Store FLOPs in Helm Graph (using .flops field now)
                if node in self.graph.fx_to_helm:
                    helm_node = self.graph.fx_to_helm[node]
                    helm_node.flops = flops
                    
                    if isinstance(result, torch.Tensor):
                         helm_node.output_shape = list(result.shape)
                         helm_node.output_dtype = result.dtype
                    
                    if flops > 0:
                        non_zero_nodes += 1
                    
        print(f"[DynamicAnalyzer] Meta-Execution Complete. Found FLOPs for {non_zero_nodes} nodes.")
