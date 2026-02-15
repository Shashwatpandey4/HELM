import torch
import re
from ..graph import HelmGraph, HelmNode

class HelmPartitioner:
    """
    Pass 4: Partitioning
    Assigns each node to a device (CPU or GPU) based on available memory.
    Strategy: 
    - If Config provided: Config-Driven (Layer Ranges -> Devices).
    - If No Config: Greedy - Fill GPU until VRAM limit is reached, then spill to CPU.
    """
    def __init__(self, graph: HelmGraph, config=None):
        self.graph = graph
        self.config = config
        self.device_map = {} # node -> device_str
    
    def run(self):
        print("\n[HelmPartitioner] Starting Partitioning...")
        
        if self.config:
            print(f"  Using Optimal Config (PP={self.config.pp_degree}). Applying Plan...")
            self._apply_config()
        else:
            print(f"  No Config provided. Using Greedy Strategy.")
            self._apply_greedy()
            
    def _apply_config(self):
        """
        Maps nodes to devices based on Layer Index -> Stage -> Device.
        """
        # Create Loopup: Layer Index -> Device
        layer_to_device = {}
        
        # Iterate stages
        for stage in self.config.pp_stages:
            # Stage.devices is a list (e.g., [0, 1] for TP=2).
            # For partitioning metadata, we assign the node to the "primary" device of the stage (e.g. cuda:0).
            # The runtime handle TP distribution.
            # However, if we assign "cuda:0", PipelineSplit will put it in Stage 0.
            # If Stage 1 is on "cuda:1", it puts it in Stage 1.
            # If Stage 0 is on "cuda:0" AND "cuda:1"?
            # PipelineSplitPass splits by transition in `meta_device`. 
            # If all nodes in Stage 0 have `meta_device="cuda:0"`, they form one submodule.
            # Then PipelineExecutor sees "cuda:0" and initiates TP group [0, 1]. This works.
            
            primary_dev_id = stage.devices[0]
            dev_str = f"cuda:{primary_dev_id}"
            
            start, end = stage.layer_range # [start, end)
            for l in range(start, end):
                layer_to_device[l] = dev_str
                
        # Fallback for non-layer nodes (Embeddings, Norm, Head)
        # Embeddings -> First Stage
        # Head -> Last Stage
        first_stage_dev = f"cuda:{self.config.pp_stages[0].devices[0]}"
        last_stage_dev = f"cuda:{self.config.pp_stages[-1].devices[0]}"
        
        for node in self.graph.nodes:
            # Extract Layer Index
            # Pattern: layers.12. or h.12. or blocks.12.
            match = re.search(r"\.(\d+)\.", node.name)
            
            if match:
                layer_idx = int(match.group(1))
                if layer_idx in layer_to_device:
                    node.device = layer_to_device[layer_idx]
                else:
                    # Layer out of range? (Maybe Head is L+1?)
                    node.device = last_stage_dev
            else:
                # No layer index found.
                # If "embed" in name -> First Stage
                # If "head" or "lm_head" -> Last Stage
                # Else? Likely early ops -> First Stage
                if "head" in node.name:
                    node.device = last_stage_dev
                else:
                    node.device = first_stage_dev
                    
        print(f"  Applied Config-Driven Partitioning.")

    def _apply_greedy(self):
        # 1. Get Hardware Constraints
        gpu_devices = []
        if self.graph.hardware_meta.get('gpu_available'):
             # Create list of (device_name, capacity_bytes)
             for i, gpu in enumerate(self.graph.hardware_meta['gpus']):
                 capacity = int(gpu['total_memory_gb'] * (1024**3))
                 gpu_devices.append({
                     "device": f"cuda:{i}",
                     "capacity": capacity,
                     "usage": 0,
                     "node_count": 0
                 })
             print(f"  Detected {len(gpu_devices)} GPU(s).")
        else:
             print("  No GPU detected. All nodes -> CPU.")
             # Fallback: everything on CPU
             for node in self.graph.nodes:
                 node.device = "cpu"
             return
 
        # 2. Greedy Fill Strategy (Multi-GPU)
        # We fill GPU 0, then GPU 1, ..., then CPU.
        
        # 1GB Safety Margin
        SAFE_MARGIN = 1.0 * (1024**3) 
        
        current_device_idx = 0
        
        for node in self.graph.nodes:
            # Check if it's a parameter/buffer placeholder (persistent memory)
            is_param = (node.op_type == 'placeholder') 
            node_size = node.output_bytes
            
            assigned = False
            
            # Try to fit in current or subsequent GPUs
            while current_device_idx < len(gpu_devices):
                device_info = gpu_devices[current_device_idx]
                effective_limit = device_info['capacity'] - SAFE_MARGIN
                
                # Check if it fits
                if device_info['usage'] + node_size < effective_limit:
                    # Assign to this GPU
                    node.device = device_info['device']
                    if is_param:
                        device_info['usage'] += node_size
                    device_info['node_count'] += 1
                    assigned = True
                    break
                else:
                    # Current GPU full! Move to next one.
                    print(f"  [Info] {device_info['device']} full ({device_info['usage']/(1024**3):.2f} GB). Switching to next device.")
                    current_device_idx += 1
            
            if not assigned:
                # All GPUs full -> CPU
                node.device = "cpu"
                
        print(f"  Partitioning Complete.")
        for dev in gpu_devices:
            print(f"  {dev['device']}: {dev['node_count']} nodes, {dev['usage']/(1024**3):.2f} GB used")
        
        cpu_nodes = len([n for n in self.graph.nodes if n.device == 'cpu'])
        print(f"  cpu: {cpu_nodes} nodes")
