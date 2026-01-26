import ray
import torch
import torch.distributed as dist
import os
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.backend import helm
@ray.remote(num_gpus=0.5)
class PipelineStage:
    def __init__(self, rank, world_size, master_addr, master_port, model_factory):
        self.rank = rank
        self.world_size = world_size
        self.model_factory = model_factory
        
        # Set environment variables for process group
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        # Initialize Process Group
        # Ray sets CUDA_VISIBLE_DEVICES, so we use device 0 of the visible ones
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        
        print(f"[Rank {rank}] Initializing Process Group...")
        print(f"[Rank {rank}] Physical Device: {torch.cuda.get_device_name(0)}")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Process Group Initialized.")
        
        self.model = None
        self.opt_model = None

    def compile_model(self):
        print(f"[Rank {self.rank}] Loading and Compiling Model...")
        # Load Model via Factory
        # Expects factory to return (model, example_input) or just model (if we can infer input)
        # But for tracing we definitely need input.
        res = self.model_factory()
        if isinstance(res, tuple):
            self.model, example_input = res
        else:
            self.model = res
            # Fallback if factory doesn't return input?
            # We used to use dummy 8x1024. Now we really should enforce tuple return.
            # Assuming factory returns (model, input_tensor/args)
            example_input = torch.randn(8, 1024, device=self.device) # Fallback

        self.model = self.model.to(self.device)
        # We don't always want to force eval, but typically yes for inference
        self.model.eval()
        
        def helm_backend(gm, inputs):
            return helm(gm, inputs, world_size=self.world_size, rank=self.rank)
        
        self.opt_model = torch.compile(self.model, backend=helm_backend)
        
        # Trigger compilation with example input
        print(f"[Rank {self.rank}] Warmup/Compile Trigger...")
        
        # Ensure example_input is on device
        if isinstance(example_input, torch.Tensor):
            example_input = example_input.to(self.device)
        
        try:
            with torch.no_grad():
                self.opt_model(example_input)
        except Exception as e:
            # Rank 0 is EXPECTED to fail if it produces no output but wrapper expects one
            if "tuple index out of range" in str(e) or "NoneType" in str(e):
                print(f"[Rank {self.rank}] Warmup completed (caught expected output error: {e})")
            else:
                print(f"[Rank {self.rank}] Warmup exception: {e}")
        
        # Synchronize
        torch.cuda.synchronize()
        print(f"[Rank {self.rank}] Compilation Done.")

    def run_batch(self, input_tensor):
        """
        Runs a batch.
        Rank 0 receives input_tensor.
        Other ranks might need dummy input/trigger.
        """
        # Ensure input is on device
        if input_tensor is not None:
            input_tensor = input_tensor.to(self.device)
        else:
            # Intermediate ranks need "some" input to trigger the forward
            # For now, we create a dummy tensor of typical shape?
            # Ideally this should also be provided by the factory or metadata?
            # Or we can pass a dummy empty tensor if the graph doesn't check values.
             input_tensor = torch.empty(1, 1, device=self.device) # Minimal dummy

        try:
            with torch.no_grad():
                out = self.opt_model(input_tensor)
            
            # If we are the last Stage, we expect a valid output
            if self.rank == self.world_size - 1:
                return out.cpu() # Return to head node
            else:
                return "Stage Completed"
        except Exception as e:
            if "tuple index out of range" in str(e):
                return "Stage Completed (No Output)"
            raise e

class RayPipelineRuntime:
    def __init__(self, model_factory, world_size=2):
        self.world_size = world_size
        ray.init(ignore_reinit_error=True)
        
        master_addr = "localhost"
        master_port = 29506
        
        self.stages = []
        for i in range(world_size):
            stage = PipelineStage.remote(i, world_size, master_addr, master_port, model_factory)
            self.stages.append(stage)
            
        # Compile SEQUENTIALLY to avoid OOM (RAM spike)
        # If we launch all at once, every worker tries to load the full 4B model into RAM simultaneously.
        # By doing it one by one, the previous worker finishes compilation (and hopefully releases some temp memory)
        # before the next one starts.
        for i, s in enumerate(self.stages):
            print(f"initializing Stage {i}...")
            ray.get(s.compile_model.remote())
        print("All stages compiled.")

    def run(self, input_tensor):
        print("Orchestrating Run...")
        futures = []
        for i, stage in enumerate(self.stages):
            # Only rank 0 gets the actual data
            inp = input_tensor if i == 0 else None
            futures.append(stage.run_batch.remote(inp))
            
        results = ray.get(futures)
        return results[-1] # Return result of last stage

    def shutdown(self):
        ray.shutdown()
