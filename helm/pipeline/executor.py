import torch
import torch.nn as nn
from typing import List, Any, Dict

class PipelineStage:
    """
    Represents a single stage in the pipeline (a wrapped nn.Module or GraphModule).
    """
    def __init__(self, stage_idx: int, module: nn.Module, device: str):
        self.stage_idx = stage_idx
        self.module = module
        self.device = torch.device(device)
        self.stream = None
        
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=self.device)

    def run(self, *args, **kwargs):
        """Run the stage, using stream if available."""
        if self.stream:
            with torch.cuda.stream(self.stream):
                return self.module(*args, **kwargs)
        else:
            return self.module(*args, **kwargs)
            
    def record_event(self):
        """Record an event on the stream."""
        if self.stream:
            event = torch.cuda.Event()
            event.record(self.stream)
            return event
        return None
        
    def wait_event(self, event):
        """Make this stage's stream wait for an event."""
        if self.stream and event:
            self.stream.wait_event(event)
            
    def synchronize(self):
        """Wait for execution to complete."""
        if self.stream:
            self.stream.synchronize()

class PipelineExecutor:
    """
    The runtime driver that executes a split GraphModule in pipeline parallel fashion.
    Implements wavefront scheduling with CUDA Events for dependency management.
    Supports Data Parallelism via DeviceMesh mapping.
    """
    def __init__(self, gm: torch.fx.GraphModule, micro_batch_size: int = 1, 
                 max_seq_len: int = 2048, device_mesh=None, replica_id: int = 0):
        self.gm = gm
        self.micro_batch_size = micro_batch_size
        self.max_seq_len = max_seq_len  # Maximum sequence length for capacity planning
        self.device_mesh = device_mesh
        self.replica_id = replica_id
        self.stages: List[PipelineStage] = []
        
        # 1. Detect Stages
        idx = 0
        while True:
            stage_name = f"submod_{idx}"
            if hasattr(gm, stage_name):
                submod = getattr(gm, stage_name)
                
                # Determine Device
                if self.device_mesh:
                    # Use Mesh Resolution
                    # Map (Replica, Stage) -> Physical Device
                    # We assume Stage Index maps 1:1 to PP Rank for now
                    # and TP=0 (Tensor Parallelism handled inside stage)
                    # We need to map Logical Stage -> PP Rank.
                    # Simple assumption: Stage 0 is PP Rank 0.
                    
                    # Need a way to convert (dp, pp, tp) -> physical ID
                    # We'll assume the mesh has a lookup or we calculate it.
                    # Since Mesh logic is strict:
                    # physical_id = mesh.get_physical_ipv4... no wait
                    # Let's trust the mesh's linear mapping for now or import helper.
                    
                    # Coordinate: (dp=self.replica_id, pp=idx, tp=0)
                    # Note: This assumes TP is handled inside the module (internally sharded).
                    # But wait, if TP>1, the stage itself spans multiple devices?
                    # Yes. PipelineStage typically manages the detailed execution.
                    # For basic PP+DP, we map to the "Primary" device of the stage (TP rank 0).
                    
                    # We need the mesh to give us the device ID.
                    # Let's assume mesh has get_global_rank and get_physical_device_id
                    global_rank = self.device_mesh.get_global_rank(self.replica_id, idx, 0)
                    phy_id = self.device_mesh.get_physical_device_id(global_rank)
                    
                    # Check if CPU override intended?
                    # If meta_device was "cpu", generally we respect it for offload.
                    # But for GPU-GPU pipeline, we overwrite.
                    meta_dev = getattr(submod, "meta_device", "cuda")
                    if "cpu" in str(meta_dev):
                        device_str = "cpu"
                    else:
                        device_str = f"cuda:{phy_id}"
                        
                else:
                    # Legacy / Independent Logic
                    # Retrieve metadata attached by PipelineSplitPass
                    device_str = getattr(submod, "meta_device", "cpu")
                    
                    # Fallback purely for robustness (e.g. legacy tests)
                    if not hasattr(submod, "meta_device"):
                         if idx == 0: device_str = "cuda:0"
                         else: device_str = "cpu"

                stage = PipelineStage(idx, submod, device_str)
                self.stages.append(stage)
                idx += 1
            else:
                break
                
        print(f"[PipelineExecutor] Initialized Replica {self.replica_id} with {len(self.stages)} stages.")

    def run_forward(self, *inputs):
        """
        Executes the pipeline (Inference Mode).
        inputs: Tuple of full-batch input tensors.
        """
        # Detect actual sequence length and adjust micro-batch size if needed
        reference_tensor = next((x for x in inputs if isinstance(x, torch.Tensor)), None)
        if reference_tensor is not None and len(reference_tensor.shape) > 1:
            actual_seq_len = reference_tensor.shape[1]  # Assume (batch, seq, ...)
            effective_mb = self._adjust_micro_batch_size(actual_seq_len)
        else:
            effective_mb = self.micro_batch_size
        
        # 1. Chunk Inputs (Micro-batching)
        chunked_inputs = self._chunk_inputs(inputs, effective_mb)
        num_micro_batches = len(chunked_inputs)
        num_stages = len(self.stages)
        
        print(f"[PipelineExecutor] Batch processing: {num_micro_batches} microbatches (MB size={effective_mb}) on {num_stages} stages.")

        # buffers[stage_idx][mb_idx] stores output of stage_idx for mb_idx
        # buffers[0] is input to Stage 0
        buffers = [[None] * num_micro_batches for _ in range(num_stages + 1)]
        buffers[0] = chunked_inputs
        
        # Events [stage_idx][mb_idx] -> Event signaling completion
        events = [[None] * num_micro_batches for _ in range(num_stages)]
        
        # 2. Wavefront / 1F1B Scheduling Check
        # To maximize overlapping, we want to launch tasks as soon as possible.
        # Simple loop: for mb in range(M): for stage in range(N):
        # This issues: (S0, MB0), (S1, MB0), (S0, MB1), (S1, MB1)...
        # Constraint: S1 can't start MB0 until S0 finishes MB0.
        
        for mb_i in range(num_micro_batches):
            for stage in self.stages:
                idx = stage.stage_idx
                
                # Input args for this stage/mb
                inp_args = buffers[idx][mb_i]
                
                # Verify Logic:
                # If idx > 0 (not first stage), we must wait for previous stage.
                if idx > 0:
                    prev_event = events[idx-1][mb_i]
                    if prev_event:
                        stage.wait_event(prev_event)
                
                # Run
                # Note: inp_args might be on previous device.
                # The module likely contains .to() instructions at start.
                if isinstance(inp_args, tuple):
                     out = stage.run(*inp_args)
                else:
                     out = stage.run(inp_args)
                
                # Store output
                buffers[idx+1][mb_i] = out
                
                # Record completion
                events[idx][mb_i] = stage.record_event()
        
        # 3. Synchronize All
        for stage in self.stages:
            stage.synchronize()
            
        # 4. Concatenate Outputs
        return self._collate_outputs(buffers[-1])

    
    def _adjust_micro_batch_size(self, actual_seq_len: int) -> int:
        """
        Dynamically adjust micro-batch size based on sequence length.
        
        Longer sequences -> smaller micro-batches to avoid OOM.
        """
        if actual_seq_len > self.max_seq_len * 0.8:
            # Long sequences: reduce MB size
            adjusted = max(1, self.micro_batch_size // 2)
            print(f"  [Dynamic Batching] Long sequence ({actual_seq_len} tokens) -> reducing MB size to {adjusted}")
            return adjusted
        elif actual_seq_len < self.max_seq_len * 0.3:
            # Short sequences: can increase MB size
            adjusted = min(self.micro_batch_size * 2, 16)
            print(f"  [Dynamic Batching] Short sequence ({actual_seq_len} tokens) -> increasing MB size to {adjusted}")
            return adjusted
        else:
            return self.micro_batch_size

    def _chunk_inputs(self, inputs, micro_batch_size=None):
        if micro_batch_size is None:
            micro_batch_size = self.micro_batch_size
            
        # Flatten inputs to list
        input_list = list(inputs)
        chunked = [] 
        
        # Determine batch size from first tensor input
        # TODO: More robust check for scalar vs tensor args
        reference_tensor = next((x for x in input_list if isinstance(x, torch.Tensor)), None)
        if reference_tensor is None:
             # No tensors? Just return as is (1 MB)
             return [inputs]
             
        batch_dim_size = reference_tensor.shape[0] 
        num_micro_batches = batch_dim_size // micro_batch_size
        if batch_dim_size % micro_batch_size != 0:
            num_micro_batches += 1
            
        for i in range(num_micro_batches):
            mb_args = []
            start = i * micro_batch_size
            end = min(start + micro_batch_size, batch_dim_size)
            
            for arg in input_list:
                if isinstance(arg, torch.Tensor) and arg.shape[0] == batch_dim_size:
                    # simplistic batch dim check
                    mb_args.append(arg[start:end])
                else:
                    mb_args.append(arg)
            chunked.append(tuple(mb_args))
        return chunked

    def _collate_outputs(self, outputs):
        if not outputs: return None
        first = outputs[0]
        
        if isinstance(first, torch.Tensor):
            return torch.cat(outputs, dim=0)
        elif isinstance(first, (tuple, list)):
            transposed = list(zip(*outputs))
            final_res = []
            for col in transposed:
                if isinstance(col[0], torch.Tensor):
                     # Check dim match?
                     final_res.append(torch.cat(col, dim=0))
                else:
                    # Can't cat scalars, return list?
                    final_res.append(col)
            return tuple(final_res)
        return outputs
