import torch
import torch.fx as fx
from functools import reduce
import operator

def safe_prod(iterable):
    return reduce(operator.mul, iterable, 1)

def estimate_flops_for_node(node):
    val = node.meta.get('val')
    if val is None:
        val = node.meta.get('example_value')
        
    if val is None:
        return 0
    
    if not isinstance(val, torch.Tensor):
        return 0
        
    input_shapes = []
    for arg in node.args:
        # Check node.args, but also node.meta['val'] of the args if they are nodes
        if isinstance(arg, fx.Node):
            arg_val = arg.meta.get('val')
            if arg_val is None:
                arg_val = arg.meta.get('example_value')
                
            if isinstance(arg_val, torch.Tensor):
                input_shapes.append(arg_val.shape)
            else:
                input_shapes.append(None)
        # Handle constants (not Nodes) if needed? usually not huge flops source
        else:
            input_shapes.append(None)
            
    flops = 0
    target_str = str(node.target)
    
    
    # Check if we hit any logic
    matched = False


    # Matrix Multiply
    if 'mm' in target_str or 'matmul' in target_str or 'linear' in target_str or 'bmm' in target_str:
        # torch._C._nn.linear is a common one seen in the output!
        if len(input_shapes) >= 2 and input_shapes[0] is not None:
             shape_a = input_shapes[0]
             # Linear: (input, weight, bias). Weight is [Out, In]. Input is [..., In]
             # Ops can be 'aten.linear', 'torch.nn.functional.linear', or 'torch._C._nn.linear'
             
             if 'linear' in target_str: 
                 # Arg 0: Input, Arg 1: Weight.
                 # Check Arg 1 is a Node (weights usually are nodes in FX graph, get_attr or such)
                 shape_w = input_shapes[1] if len(input_shapes) > 1 else None
                 if shape_a and shape_w:
                     # shape_a: [..., In]
                     # shape_w: [Out, In]
                     # FLOPs = 2 * Batch * In * Out
                     # Batch is product of shape_a[:-1]
                     batch = safe_prod(shape_a[:-1])
                     in_dim = shape_a[-1]
                     out_dim = safe_w = shape_w[0]
                     flops = 2 * batch * in_dim * out_dim
             else:
                # MM/BMM/Matmul
                shape_b = input_shapes[1] if len(input_shapes) > 1 else None
                if shape_a and shape_b:
                    # Last two dims
                    M = shape_a[-2]
                    K = shape_a[-1]
                    N = shape_b[-1]
                    batch = safe_prod(shape_a[:-2])
                    if len(shape_b) > 2:
                        batch = max(batch, safe_prod(shape_b[:-2]))
                    flops = 2 * batch * M * N * K

    # Elementwise Add/Mul/Sub
    elif 'add' in target_str or 'sub' in target_str or 'mul' in target_str:
        # Be careful not to double count huge broadcasts or scalar ops
        if 'addmm' not in target_str:
             flops = safe_prod(val.shape)
             matched = True

    if not matched and ('linear' in target_str or 'matmul' in target_str):
        # We missed it, likely due to inputs being None or shapes weird
        pass

    return int(flops)

def flops_analysis_pass(gm: torch.fx.GraphModule):
    print("\n>> [Pass] Running FLOPs Analysis...")
    total_flops = 0
    
    for node in gm.graph.nodes:
        if node.op in ['call_function', 'call_method', 'call_module']:
             flops = estimate_flops_for_node(node)
             node.meta['flops'] = flops
             total_flops += flops

             if flops > 0:
                 # Debug print to ensure we see it happens
                 print(f"   + Node {node.name} ({node.target}) -> {flops:.2e} FLOPs")
    
    print(f">> [Pass] Total Estimated FLOPs: {total_flops:.4e}")
    gm.meta['total_flops'] = total_flops
    return gm
