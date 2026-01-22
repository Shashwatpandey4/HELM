import torch
import torch.fx
from .common import dist_send, dist_recv, dist_all_reduce

def pipeline_parallel_pass(gm: torch.fx.GraphModule, split_nodes: list[torch.fx.Node] = None):
    """
    Splits the graph into stages.
    Prioritizes `split_nodes` arg.
    If `split_nodes` is None/Empty, looks for `node.meta['pipeline_split']`.
    """
    graph = gm.graph
    
    # Collect split nodes from metadata if not provided
    if not split_nodes:
        split_nodes = []
        for node in graph.nodes:
            if node.meta.get('pipeline_split', False):
                split_nodes.append(node)
                
    if not split_nodes:
        # No splits found
        return gm

    # 1. Map nodes to stages
    node_to_stage = {}
    current_stage = 0
    split_node_set = set(split_nodes)
    
    for node in graph.nodes:
        node_to_stage[node] = current_stage
        if node in split_node_set:
            current_stage += 1
            
    print(f"DEBUG: split_nodes={[n.name for n in split_nodes]}")
    # print(f"DEBUG: node_to_stage={{n.name: s for n, s in node_to_stage.items()}}")

    # 2. Identify cross-stage dependencies
    processed_nodes = list(graph.nodes) 
    replacements = {} 
    
    for node in processed_nodes:
        # Check args
        for arg in node.all_input_nodes:
            producer = arg
            consumer = node
            
            s_prod = node_to_stage.get(producer, 0)
            s_cons = node_to_stage.get(consumer, 0)
            
            if s_prod < s_cons:
                # Need communication
                key = (producer, s_cons)
                if key in replacements:
                    recv_node = replacements[key]
                else:
                    with graph.inserting_after(producer):
                        graph.call_function(dist_send, args=(producer, s_cons))
                        
                    with graph.inserting_before(consumer):
                        shape = None
                        # Convert ShapeProp tensor_meta to shape if available
                        if 'tensor_meta' in producer.meta:
                            tm = producer.meta['tensor_meta']
                            if hasattr(tm, 'shape'): shape = tm.shape
                        elif 'val' in producer.meta:
                             val = producer.meta['val']
                             if hasattr(val, 'shape'): shape = val.shape
                             
                        recv_node = graph.call_function(dist_recv, args=(s_prod, shape)) 
                        node_to_stage[recv_node] = s_cons
                        
                    replacements[key] = recv_node
                
                consumer.replace_input_with(producer, recv_node)

    graph.lint()
    return gm

def tensor_parallel_pass(gm: torch.fx.GraphModule, node: torch.fx.Node = None, strategy: str = None):
    """
    Applies Tensor Parallelism.
    If `node` is provided, applies to that node.
    If `node` is None, scans graph for nodes with `node.meta['sharding_strategy']`.
    """
    graph = gm.graph
    
    nodes_to_shard = []
    if node:
        nodes_to_shard.append((node, strategy or 'col'))
    else:
        for n in graph.nodes:
            strat = n.meta.get('sharding_strategy')
            if strat:
                nodes_to_shard.append((n, strat))
    
    if not nodes_to_shard:
        return gm
        
    for target_node, strat in nodes_to_shard:
        if target_node not in graph.nodes:
            continue
            
        if strat == 'col':
            print(f"DEBUG: Applying col strategy to {target_node.name}")
            # Insert AllReduce after the node
            # Ensure we haven't already inserted one (avoid dupes if re-running)
            
            # Simple check: is the first user already an all_reduce?
            users = list(target_node.users.keys())
            already_sharded = False
            for u in users:
                if 'dist_all_reduce' in str(u.target):
                    already_sharded = True
                    break
            
            if not already_sharded:
                with graph.inserting_after(target_node):
                    all_reduce_node = graph.call_function(dist_all_reduce, args=(target_node,), kwargs={'op': 'sum'})
                    
                    for user in users:
                        if user != all_reduce_node:
                            user.replace_input_with(target_node, all_reduce_node)
                    
    graph.lint()
    return gm
