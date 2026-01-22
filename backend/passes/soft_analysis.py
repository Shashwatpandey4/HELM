import torch
import torch.fx as fx
import torch.nn as nn
from functools import reduce
import operator

def safe_prod(iterable):
    return reduce(operator.mul, iterable, 1)

def get_shape_and_element_size(node):
    """
    Extracts (shape, element_size_in_bytes) from node metadata.
    """
    # 1. Check tensor_meta (ShapeProp)
    tm = node.meta.get('tensor_meta')
    if tm is not None:
        if hasattr(tm, 'shape') and hasattr(tm, 'dtype'):
            element_size = getattr(tm.dtype, 'itemsize', 4)
            return tm.shape, element_size
            
    # 2. Check val / example_value
    val = node.meta.get('val') or node.meta.get('example_value')
    if isinstance(val, torch.Tensor):
        return val.shape, val.element_size()
        
    return None, 0

def estimate_node_metrics(node, gm):
    """
    Estimates FLOPs and Memory IO for a given node.
    Returns (flops, memory_bytes)
    """
    out_shape, out_elem_size = get_shape_and_element_size(node)
    
    # 1. Output Size
    out_bytes = 0
    if out_shape is not None:
        out_bytes = safe_prod(out_shape) * out_elem_size
    
    # 2. Input Size and Shapes
    in_bytes = 0
    input_shapes = []
    
    for arg in node.args:
        if isinstance(arg, fx.Node):
            shape, elem_size = get_shape_and_element_size(arg)
            if shape is not None:
                size = safe_prod(shape) * elem_size
                in_bytes += size
                input_shapes.append(shape)
            else:
                input_shapes.append(None)
        else:
             input_shapes.append(None)

    total_mem_bytes = in_bytes + out_bytes
    
    # FLOPs Estimation
    flops = 0
    target_str = str(node.target)
    
    # Check Module Type if call_module
    module_type = None
    if node.op == 'call_module':
        try:
            submod = gm.get_submodule(node.target)
            module_type = type(submod)
        except:
            pass
            
    # Logic
    is_linear = False
    is_matmul = False
    is_elementwise = False
    is_layernorm = False
    
    # 1. Identify Op Type
    if module_type is not None:
        if issubclass(module_type, nn.Linear):
            is_linear = True
        elif issubclass(module_type, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            is_layernorm = True
        elif issubclass(module_type, (nn.ReLU, nn.GELU, nn.SiLU)):
            is_elementwise = True # zero flops or small? ReLU is 1 cmp per element
    else:
        # String matching for functionals
        if 'linear' in target_str: is_linear = True
        elif 'mm' in target_str or 'matmul' in target_str or 'bmm' in target_str: is_matmul = True
        elif 'layer_norm' in target_str: is_layernorm = True
        elif 'add' in target_str or 'sub' in target_str or 'mul' in target_str: is_elementwise = True

    # 2. Calculate FLOPs
    if is_linear:
         if len(input_shapes) >= 1 and input_shapes[0] is not None:
             shape_a = input_shapes[0]
             # Linear layer weights are in the module, not usually in args[1] for call_module
             # We can get weight shape from module if available
             weight_shape = None
             if node.op == 'call_module':
                 submod = gm.get_submodule(node.target)
                 if hasattr(submod, 'weight') and submod.weight is not None:
                     weight_shape = submod.weight.shape
             elif len(input_shapes) > 1:
                 weight_shape = input_shapes[1]
                 
             if shape_a and weight_shape:
                 # shape_a: [..., In]
                 # weight: [Out, In]
                 batch = safe_prod(shape_a[:-1])
                 out_dim = weight_shape[0]
                 in_dim = weight_shape[1]
                 flops = 2 * batch * in_dim * out_dim
                 
    elif is_matmul:
        if len(input_shapes) >= 2 and input_shapes[0] is not None and input_shapes[1] is not None:
            shape_a = input_shapes[0]
            shape_b = input_shapes[1]
            if len(shape_a) >= 2 and len(shape_b) >= 2:
                M = shape_a[-2]
                K = shape_a[-1]
                N = shape_b[-1]
                batch = safe_prod(shape_a[:-2])
                if len(shape_b) > 2:
                     batch = max(batch, safe_prod(shape_b[:-2]))
                flops = 2 * batch * M * N * K
                
    elif is_layernorm:
        if out_shape is not None:
             flops = safe_prod(out_shape) * 5
             
    elif is_elementwise:
        if out_shape is not None:
             flops = safe_prod(out_shape)

    return int(flops), int(total_mem_bytes)

def soft_analysis_pass(gm: torch.fx.GraphModule):
    """
    Annotates nodes with computation (FLOPs) and data read-write (Memory Bytes).
    """
    print("\n>> [Pass] Running Soft Analysis (Computation & IO)...")
    total_flops = 0
    total_mem = 0
    
    node_stats = {}
    
    for node in gm.graph.nodes:
        if node.op in ['call_function', 'call_method', 'call_module']:
             flops, mem = estimate_node_metrics(node, gm)
             
             node.meta['soft_analysis'] = {
                 'flops': flops,
                 'mem_bytes': mem
             }
             
             total_flops += flops
             total_mem += mem
             
             node_stats[node.name] = {'flops': flops, 'mem_bytes': mem}
             
             if flops > 0 or mem > 0:
                  print(f"   + {node.name}: {flops:.2e} FLOPs, {mem/1024:.2f} KB RW")

    # Attach summary to GraphModule
    gm.meta['soft_analysis_summary'] = {
        'total_flops': total_flops,
        'total_mem_bytes': total_mem,
        'node_stats': node_stats
    }
    
    print(f">> [Pass] Total: {total_flops:.4e} FLOPs, {total_mem/(1024**2):.2f} MB RW")
    return gm
