
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from backend.backend import helm

print("Loading dist_runtime module...", flush=True)

def _worker_loop(rank, world_size, master_addr, master_port, model_factory, input_queue, output_queue, control_queue):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Use device corresponding to rank (assuming 1 GPU per rank approx, or reuse logic)
    # Simple mapping for now: rank % device_count
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}] Initializing Process Group on {device}...", flush=True)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Process Group Initialized.", flush=True)

    model = None
    opt_model = None

    # Helper for compilation
    def compile_model():
        nonlocal model, opt_model
        print(f"[Rank {rank}] Loading and Compiling Model...", flush=True)
        res = model_factory()
        if isinstance(res, tuple):
            model, example_input = res
        else:
            model = res
            example_input = torch.randn(8, 1024, device=device)

        # If model is on meta device, don't move it to CUDA yet (avoids OOM or NotImplementedError)
        is_meta = any(p.device.type == 'meta' for p in model.parameters())
        if is_meta:
             print(f"[Rank {rank}] Model is on META device. Skipping .to(device). Using META input.", flush=True)
             example_input = example_input.to("meta")
        elif not is_meta:
            print(f"[Rank {rank}] Moving model to device...", flush=True)
            model = model.to(device)
            print(f"[Rank {rank}] Model on device.", flush=True)
        else:
             print(f"[Rank {rank}] Model is on META device. Skipping .to(device).", flush=True)

        def helm_backend(gm, inputs):
            return helm(gm, inputs, world_size=world_size, rank=rank)

        print(f"[Rank {rank}] Calling torch.compile...", flush=True)
        opt_model = torch.compile(model, backend=helm_backend, fullgraph=True)
        print(f"[Rank {rank}] torch.compile returned.", flush=True)
        
        # Warmup
        print(f"[Rank {rank}] Warmup/Compile Trigger (Input Shape: {example_input.shape})...", flush=True)
        if isinstance(example_input, torch.Tensor) and not is_meta:
            example_input = example_input.to(device)
        elif is_meta:
            print(f"[Rank {rank}] Skipping input .to(device) (Meta Mode).", flush=True)
            
        try:
            with torch.no_grad():
                opt_model(example_input)
        except Exception as e:
            # Expected error for intermediate ranks if they return None/Stage Completed
            pass
        torch.cuda.synchronize()
        print(f"[Rank {rank}] Compilation Done.", flush=True)

    print(f"[Rank {rank}] Entering Worker Loop...", flush=True)
    while True:
        # Wait for command
        try:
            print(f"[Rank {rank}] Waiting for command...", flush=True)
            cmd_data = control_queue.get() # Blocking get
            print(f"[Rank {rank}] Received command: {cmd_data}", flush=True)
        except Exception as e:
             print(f"[Rank {rank}] Queue error: {e}", flush=True)
             continue

        cmd_type = cmd_data[0]
        
        if cmd_type == "SHUTDOWN":
            break
            
        if cmd_type == "COMPILE":
            compile_model()
            output_queue.put("DONE")
            
        if cmd_type == "RUN":
            inp = None
            if rank == 0:
                 inp = input_queue.get()
                 inp = inp.to(device)
            else:
                 inp = torch.empty(1, 1, device=device)

            try:
                with torch.no_grad():
                    out = opt_model(inp)
                
                if rank == world_size - 1:
                    output_queue.put(out.cpu())
            except Exception as e:
                pass
                
    dist.destroy_process_group()

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

class Runtime:
    def __init__(self, model_factory, world_size=2):
        self.world_size = world_size
        self.mp_ctx = mp.get_context('spawn')
        self.input_queue = self.mp_ctx.Queue()
        self.output_queue = self.mp_ctx.Queue()
        
        self.control_queues = [self.mp_ctx.Queue() for _ in range(world_size)]
        
        self.processes = []
        master_addr = "127.0.0.1"
        master_port = find_free_port()
        print(f"Using Master Port: {master_port}", flush=True)
        
        for i in range(world_size):
            p = self.mp_ctx.Process(
                target=_worker_loop,
                args=(i, world_size, master_addr, master_port, model_factory, 
                      self.input_queue, self.output_queue, self.control_queues[i])
            )
            p.start()
            self.processes.append(p)
            
        # Compile
        print("Sending COMPILE command to workers...", flush=True)
        for q in self.control_queues:
            q.put(("COMPILE",))
        
        print("Waiting for workers to compile...", flush=True)
        # Wait for all to finish compiling
        for _ in range(world_size):
            self.output_queue.get() 
        print("All processes compiled.", flush=True)

    def run(self, input_tensor):
        self.input_queue.put(input_tensor)
        for q in self.control_queues:
            q.put(("RUN",))
            
        # Wait for result from last rank
        return self.output_queue.get()

    def shutdown(self):
        for q in self.control_queues:
            q.put(("SHUTDOWN",))
        for p in self.processes:
            p.join()
