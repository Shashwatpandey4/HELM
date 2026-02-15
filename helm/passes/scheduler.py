from typing import List, Dict, Any
from ..graph import HelmGraph, HelmNode

class ScheduledOp:
    def __init__(self, op_type: str, details: str, device: str):
        self.op_type = op_type # 'COMPUTE', 'TRANSFER'
        self.details = details
        self.device = device
        
    def __repr__(self):
        return f"[{self.op_type}] {self.device}: {self.details}"

class HelmScheduler:
    """
    Pass 5: Scheduling
    Infers the execution schedule and data transfers based on node placement.
    """
    def __init__(self, graph: HelmGraph):
        self.graph = graph
        self.schedule: List[ScheduledOp] = []
        
    def run(self):
        print("\n[HelmScheduler] Inferring Schedule...")
        
        transfer_count = 0
        transfer_bytes = 0
        
        # We iterate in topological order (which is the order of nodes in the graph)
        for node in self.graph.nodes:
            # Check inputs for transfers
            target_device = node.device
            
            for dep in node.all_input_nodes:
                source_device = dep.device
                
                if source_device != target_device:
                    # Transfer needed!
                    # In a real compiler, we would insert a TransferNode here.
                    # For now, we just record it in the schedule.
                    
                    bytes_to_transfer = dep.output_bytes
                    transfer_info = f"Transfer {dep.name} ({dep.get_output_bytes_str()}) from {source_device} -> {target_device}"
                    
                    self.schedule.append(ScheduledOp("TRANSFER", transfer_info, f"{source_device}->{target_device}"))
                    
                    transfer_count += 1
                    transfer_bytes += bytes_to_transfer
            
            # Record the computation
            self.schedule.append(ScheduledOp("COMPUTE", f"{node.name} ({node.op_type})", node.device))
            
        print(f"  Schedule Inferred: {len(self.schedule)} ops")
        print(f"  Total Transfers: {transfer_count}")
        if transfer_bytes > 0:
            print(f"  Total Data Movement: {transfer_bytes / (1024**3):.4f} GB")
        else:
             print(f"  Total Data Movement: 0 GB")
        
        # Determine bottlenecks? (Simple heuristic)
        # Only meaningful if we have transfers.
