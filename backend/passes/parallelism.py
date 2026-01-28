import torch
import torch.fx as fx
from .common import dist_send, dist_recv, dist_all_reduce
from .soft_analysis import get_shape_and_element_size

def pipeline_parallel_pass(gm: fx.GraphModule, rank: int, world_size: int):
    """
    Splits the graph into partitions based on 'placement' metadata.
    Returns the GraphModule containing ONLY the nodes for `rank`.
    Inserts Send/Recv for cross-rank edges.
    """
    print(f"\n>> [Pass] Running Pipeline Parallelism (Partitioning for Rank {rank})...")
    
    graph = gm.graph
    
    # 1. Identify Cross-Rank Dependencies
    # We map (src_node, consumer_node) -> communication required
    
    nodes_to_remove = []
    
    # Keep track of replacements: (src_node, dst_rank) -> recv_node
    # Used to deduplicate recvs if multiple nodes in this rank need the same external input
    recv_replacements = {} 
    
    # Iterate copy of nodes since we modify graph
    for node in list(graph.nodes):
        node_rank = node.meta.get('placement', 0)
        
        # If node belongs to this rank, check its inputs
        if node_rank == rank:
            for arg in node.all_input_nodes:
                src_rank = arg.meta.get('placement', 0)
                
                if src_rank != rank:
                    # Dependency: Arg (Rank src) -> Node (Rank me)
                    # We need a RECV node here.
                    
                    key = (arg, rank)
                    if key in recv_replacements:
                        recv_node = recv_replacements[key]
                    else:
                        # Insert RECV
                        with graph.inserting_before(node):
                            # Try to get shape from metadata for the receive buffer
                            shape, _ = get_shape_and_element_size(arg)
                            if shape is None:
                                # Fallback or warning
                                shape = (1,) 
                                
                            recv_node = graph.call_function(dist_recv, args=(src_rank, shape))
                            recv_node.meta['placement'] = rank
                            recv_replacements[key] = recv_node
                            print(f"   [Partition] Rank {rank}: Inserted RECV from Rank {src_rank} for '{arg.name}'")
                            
                    # Replace input
                    node.replace_input_with(arg, recv_node)
                    
        # If node belongs to OTHER rank, check if it feeds US or OTHERS
        # Actually simplest way:
        # If I am PRODUCING a value used by another rank, I need to SEND it.
        else:
            # Node is NOT on my rank.
            # But maybe I need to SEND it? 
            # Wait, if node_rank != rank, I don't execute this node. 
            # I only execute nodes where node_rank == rank.
            # So I can't send it.
            # The logic is:
            # If I AM the producer (node_rank == rank), check users.
            pass
            
    # Second Pass: Send Insertion
    # We iterate again? Or do it in one pass?
    # Better to iterate nodes I OWN.
    
    for node in list(graph.nodes):
        node_rank = node.meta.get('placement', 0)
        
        if node_rank == rank:
            # I own this node. Check who uses it.
            # Convert users dict to list to avoid modification issues if any
            for user in list(node.users.keys()):
                user_rank = user.meta.get('placement', 0)
                
                if user_rank != rank:
                    # I produce 'node', 'user' runs on 'user_rank'
                    # I must SEND 'node' to 'user_rank'
                    
                    # Insert SEND after producer
                    # Check if we already sent to this rank?
                    # Ideally execute Send once per dst rank
                    
                    # We insert the send. The send returns 'tensor' (passthrough) or None.
                    # Send doesn't replace the user edge. The user edge is implicit in finding the Recv on the other side.
                    
                    with graph.inserting_after(node):
                         send_node = graph.call_function(dist_send, args=(node, user_rank))
                         send_node.meta['placement'] = rank
                         print(f"   [Partition] Rank {rank}: Inserted SEND of '{node.name}' to Rank {user_rank}")
                         
    # 3. Pruning
    # Remove all nodes not on this rank
    # Care with PLH (Placeholders)
    # Global Inputs (placeholders) are usually needed only by Rank 0, 
    # UNLESS they are used by Rank X directly.
    # But if `pipeline_placement` put input on Rank 0, and Rank 1 uses it, 
    # Rank 1 will have inserted a RECV from Rank 0. Rank 0 will SEND it.
    # So we can safely remove placeholders if they are marked Rank 0 and we are Rank 1.
    
    for node in list(graph.nodes):
        node_rank = node.meta.get('placement', 0)
        
        # Special case: inserted comms nodes inherit rank (handled above)
        
        if node_rank != rank:
            # Ensure it has no users in this graph?
            # If we did replacements correctly, all users in *this* rank now use RECV.
            # Users in *other* ranks are irrelevant (they are in the other partition).
            # So we can remove.
            
            # EXCEPT: If we remove a node, FX checks if it has users.
            # In a full graph, it has users (the nodes on other ranks).
            # But we are splitting.
            # We must destroy the edges first.
            
            # Since we return a partial graph intended for this rank, 
            # we simply erase the node. FX `graph.erase_node` fails if users exist.
            # We need to aggressively prune users that are also being removed.
            
            pass
            
    # Topological delete is standard:
    # Remove unused nodes repeatedly?
    # Or just replace all uses with a Dummy?
    # Better: Construct a NEW graph.
    
    # ... Refactor to use NEW GRAPH construction or robust deletion ...
    # Deletion is tricky in-place.
    # Let's try simple deletion loop.
    
    count = 0
    # Iterate in reverse topological order (outputs first) to handle dependencies?
    for node in reversed(list(graph.nodes)):
        node_rank = node.meta.get('placement', 0)
        if node_rank != rank:
            # Check users. 
            # If users exist, they MUST be on other ranks (since we replaced all local users with Recv).
            # So we can force remove?
            # FX erase_node checks users.
            # We can clear users first.
            node.users.clear() # Violent but effective for partitioning?
            graph.erase_node(node)
            count += 1
            
    print(f"   [Partition] Rank {rank}: Pruned {count} nodes.")
    
    graph.lint()
    gm.recompile()
    return gm

def tensor_parallel_pass(gm: fx.GraphModule, node: fx.Node = None, strategy: str = None):
    # Dummy placeholder for TP
    return gm
