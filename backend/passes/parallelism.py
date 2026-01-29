import torch
import torch.fx as fx
from .common import dist_send, dist_recv, dist_all_reduce
from .data_analysis import get_shape_and_element_size, safe_prod

def pipeline_parallel_pass(gm: fx.GraphModule, rank: int, world_size: int):
    """
    Splits the graph into partitions based on 'placement' metadata.
    Ensures SEND/RECV ordering is globally consistent across ranks.
    """
    print(f"\n>> [Pass] Running Pipeline Parallelism (Partitioning for Rank {rank})...")
    
    graph = gm.graph
    
    # 1. Map each node to its topological index for consistent sorting
    node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
    
    # 2. Identify all cross-rank dependencies
    # We want unique (src_node, dst_rank) pairs
    cross_dependencies = []
    seen_deps = set()
    
    for node in graph.nodes:
        dst_rank = node.meta.get('placement', 0)
        for arg in node.all_input_nodes:
            src_rank = arg.meta.get('placement', 0)
            
            if src_rank != dst_rank:
                dep = (arg, dst_rank)
                if dep not in seen_deps:
                    cross_dependencies.append(dep)
                    seen_deps.add(dep)
                    
    # 3. Sort dependencies by source node's topological index
    # This ensures both sender and receiver insert ops in the SAME order.
    cross_dependencies.sort(key=lambda x: node_to_idx[x[0]])
    
    # 4. Insert Communication Nodes
    # Map: (src_node, dst_rank) -> comm_node
    
    # We collect all Recvs for this rank to insert them in a single block at the earliest point needed.
    my_recvs = []
    
    for src_node, dst_rank in cross_dependencies:
        src_rank = src_node.meta.get('placement', 0)
        
        if src_rank == rank:
            # I am the sender. Insert SEND after production.
            # Using inserting_after(src_node) is fine as production happens in graph order.
            with graph.inserting_after(src_node):
                send_node = graph.call_function(dist_send, args=(src_node, dst_rank))
                send_node.meta['placement'] = rank
                print(f"   [Partition] Rank {rank}: Inserted SEND of '{src_node.name}' to Rank {dst_rank}")
                
        if dst_rank == rank:
            # I am the receiver. 
            # We first identify the earliest consumer to find the block's insertion point.
            earliest_consumer = None
            for user in src_node.users:
                if user.meta.get('placement', 0) == rank:
                    if earliest_consumer is None or node_to_idx[user] < node_to_idx[earliest_consumer]:
                        earliest_consumer = user
            
            # We record this dependency for the receiver-side block insertion
            if earliest_consumer:
                 my_recvs.append({
                     'src_node': src_node,
                     'src_rank': src_rank,
                     'earliest_consumer': earliest_consumer
                 })

    # Receiver Side Block Insertion
    if my_recvs:
        # Find the absolute earliest consumer for ANY recv
        first_consumer = None
        for r in my_recvs:
            if first_consumer is None or node_to_idx[r['earliest_consumer']] < node_to_idx[first_consumer]:
                first_consumer = r['earliest_consumer']
        
        # Insert all RECVs in the sorted cross_dependencies order just before the first consumer
        with graph.inserting_before(first_consumer):
            for r_info in my_recvs:
                src_node = r_info['src_node']
                src_rank = r_info['src_rank']
                
                shape, _, dtype = get_shape_and_element_size(src_node, gm)
                if shape is None: shape = (1,)
                
                recv_node = graph.call_function(dist_recv, args=(src_rank, shape, dtype))
                recv_node.meta['placement'] = rank
                print(f"   [Partition] Rank {rank}: Inserted RECV from Rank {src_rank} for '{src_node.name}' (Sequence-aligned)")
                
                # Replace all uses of src_node on THIS rank with recv_node
                for user in list(src_node.users.keys()):
                    if user.meta.get('placement', 0) == rank and user != recv_node:
                        user.replace_input_with(src_node, recv_node)

    # 5. Pruning and Output Management
    count = 0
    # Iterate in reverse topological order
    for node in reversed(list(graph.nodes)):
        if node.op == 'output':
             # Ensure output node is on every rank, but returns dummies for non-owned results
             owned_rank = node.meta.get('placement', 0)
             if owned_rank != rank:
                 new_args = []
                 for arg in node.args[0]:
                     if isinstance(arg, fx.Node) and arg.meta.get('placement', 0) != rank:
                         with graph.inserting_before(node):
                             shape, _, dtype = get_shape_and_element_size(arg, gm)
                             if shape is None: shape = (1,)
                             dummy = graph.call_function(torch.zeros, args=(shape,), kwargs={'dtype': dtype, 'device': f"cuda:{rank}"})
                             dummy.meta['placement'] = rank
                             new_args.append(dummy)
                     else:
                         new_args.append(arg)
                 node.args = (tuple(new_args),)
             continue

        node_rank = node.meta.get('placement', 0)
        if node_rank != rank:
            # Safely erase: point-to-point uses have been replaced by RECV
            # External users are in other partitions
            node.users.clear() 
            graph.erase_node(node)
            count += 1
            
    print(f"   [Partition] Rank {rank}: Pruned {count} nodes.")
    
    graph.lint()
    gm.recompile()
    return gm

def tensor_parallel_pass(gm: fx.GraphModule, node: fx.Node = None, strategy: str = None):
    return gm
