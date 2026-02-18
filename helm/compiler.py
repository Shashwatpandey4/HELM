import torch
import os
from typing import List
from .graph import HelmGraph, HelmNode
from .passes import DynamicAnalyzer
from .passes import HardwareAnalyzer
from .passes import HelmPartitioner
from .passes import HelmScheduler
from .passes import ExecutionPass
from .passes import PipelineSplitPass
from .passes import TensorParallelPass
from .runtime.executor import PipelineExecutor
from .passes import QuantizationPass

def helm_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], **kwargs):
    """
    A custom torch.compile backend for HELM.
    kwargs accepts 'options' dictionary.
    Keys:
      - 'dtype': 'fp16', 'bf16', 'int8', 'fp32' (Default: 'fp16')
      - 'tp_degree': int (Force TP)
      - 'micro_batch_size': int (Force MB)
    """
    options = kwargs.get('options', {}) if kwargs.get('options') else {}
    print(f"\n[HELM Compiler] Received GraphModule. Options: {options}")

    # Pass 0: Quantization / Precision Adjustment
    # We do this FIRST so all downstream analysis sees the correct dtype.
    target_dtype = options.get("dtype", "fp16")
    quant_pass = QuantizationPass(None, gm, dtype=target_dtype)
    # quant_pass.run() 
    print("[HELM Compiler] Skipped QuantizationPass to avoid meta tensor issues.")
    
    
    node_count = len(gm.graph.nodes)
    print(f"[HELM Compiler] Total FX Nodes: {node_count}")
    print("--- Graph Nodes ---")
    for n in gm.graph.nodes:
        print(f"  {n.op}: {n.name} ({n.target})")
    print("-------------------")
    
    # Pass 1: Create Mirror Graph (IR)
    helm_graph = HelmGraph(gm.graph)
    
    # Pass 2: Hardware Analysis
    hw_analyzer = HardwareAnalyzer(helm_graph)
    hw_analyzer.run()
    
    # Pass 3: Dynamic Analysis (Meta-Execution)
    dynamic_analyzer = DynamicAnalyzer(helm_graph, gm, example_inputs)
    dynamic_analyzer.run()
    
    # Pass 4: Strategy Selection via ParallelOptimizer
    # Extract constraints from options
    forced_tp = options.get("tp_degree")
    forced_mb = options.get("micro_batch_size")
    forced_pp = options.get("pp_degree")
    
    
    # Estimate layer count from model config if available, otherwise fallback to graph heuristics
    layer_count = 1  # Default fallback
    
    # Try to extract from options first (passed from benchmark)
    if "model_config" in options:
        cfg = options["model_config"]
        if hasattr(cfg, 'num_hidden_layers'):
            layer_count = cfg.num_hidden_layers
            print(f"[HELM Compiler] Detected {layer_count} transformer layers from model config")
        elif hasattr(cfg, 'n_layer'):
            layer_count = cfg.n_layer
            print(f"[HELM Compiler] Detected {layer_count} transformer layers from model config")
    # Try to extract from model config (if model was passed with config)
    elif hasattr(gm, 'config') and hasattr(gm.config, 'num_hidden_layers'):
        layer_count = gm.config.num_hidden_layers
        print(f"[HELM Compiler] Detected {layer_count} transformer layers from model config")
    elif hasattr(gm, 'config') and hasattr(gm.config, 'n_layer'):
        layer_count = gm.config.n_layer
        print(f"[HELM Compiler] Detected {layer_count} transformer layers from model config")
    else:
        # Fallback: count nodes with 'layer' in name (unreliable for meta device)
        layer_count = len([n for n in helm_graph.nodes if 'layer' in n.name.lower()])
        layer_count = max(layer_count, 1)
        print(f"[HELM Compiler] WARNING: Could not detect layer count from config, using heuristic: {layer_count}")

    # Calculate total parameters from model config (more reliable than graph introspection)
    # For transformer models: params ≈ 12 × n_layers × d_model²
    # But we can get a better estimate from config if available
    if "model_config" in options and hasattr(options["model_config"], 'num_parameters'):
        total_params = options["model_config"].num_parameters
    elif "model_config" in options:
        cfg = options["model_config"]
        # Estimate from architecture
        if hasattr(cfg, 'hidden_size') and hasattr(cfg, 'num_hidden_layers'):
            d_model = cfg.hidden_size
            n_layers = cfg.num_hidden_layers
            n_heads = getattr(cfg, 'num_attention_heads', 12)
            d_ff = getattr(cfg, 'intermediate_size', 4 * d_model)
            vocab_size = getattr(cfg, 'vocab_size', 50000)
            
            # Rough parameter count formula for transformer
            # Embeddings: vocab_size × d_model
            # Per layer: 4 × d_model² (QKV + O) + 3 × d_model × d_ff (FFN)
            # LM head: d_model × vocab_size
            params_per_layer = (4 * d_model * d_model) + (3 * d_model * d_ff)
            total_params = (vocab_size * d_model) + (n_layers * params_per_layer) + (d_model * vocab_size)
            print(f"[HELM Compiler] Estimated {total_params / 1e9:.2f}B parameters from config")
        else:
            total_params = 1e9  # Fallback: assume 1B params
    else:
        # Fallback: try to extract from graph (won't work on meta device)
        def get_param_bytes(node):
            if hasattr(node, 'meta') and 'val' in node.meta:
                val = node.meta['val']
                if isinstance(val, torch.Tensor):
                    return val.numel() * val.element_size()
            if hasattr(node, 'fx_node') and hasattr(node.fx_node, 'meta') and 'val' in node.fx_node.meta:
                val = node.fx_node.meta['val']
                if isinstance(val, torch.Tensor):
                    return val.numel() * val.element_size()
            return 0
        total_params = sum(get_param_bytes(node) for node in helm_graph.nodes)
        if total_params == 0:
            print("[HELM Compiler] WARNING: Could not extract parameters from graph, using fallback")
            total_params = 1e9  # Fallback
    
    total_flops = sum((node.flops or 0) for node in helm_graph.nodes if hasattr(node, 'flops'))
    
    # Use optimizer if no forced strategy
    best_config = None
    if not forced_tp and not forced_pp:
        print("[HELM Compiler] Running ParallelOptimizer to find optimal strategy...")
        
        from .optimization.optimizer import ParallelOptimizer
        from .optimization.cost_model import ModelSpec, LayerSpec, DeviceSpec
        
        # Create simplified ModelSpec for optimization
        # This is a rough approximation - real implementation would parse model config
        batch_size = example_inputs[0].shape[0] if example_inputs and len(example_inputs) > 0 else 1
        seq_len = example_inputs[0].shape[1] if example_inputs and len(example_inputs) > 0 and len(example_inputs[0].shape) > 1 else 512
        
        # Create uniform layer specs (simplified)
        avg_flops_per_layer = total_flops / layer_count if layer_count > 0 else 1e9
        avg_params_per_layer = total_params / layer_count if layer_count > 0 else 1e6
        
        layers = [
            LayerSpec(
                flops_prefill=avg_flops_per_layer,
                flops_decode_token=avg_flops_per_layer / seq_len,
                bytes_moved_prefill=avg_params_per_layer * 2,  # Read + write
                bytes_moved_decode=avg_params_per_layer,
                activation_bytes=1e6,  # Rough estimate
                param_bytes=avg_params_per_layer,
                kv_bytes_per_token=1024
            )
            for _ in range(layer_count)
        ]
        
        model_spec = ModelSpec(
            L=layer_count,
            layers=layers,
            dtype=target_dtype,
            n_heads=32,  # Default
            hidden_size=4096,  # Default
            vocab_size=50000,  # Default
            seq_len_prefill=seq_len,
            max_seq_len=seq_len,
            batch_size=batch_size,
            gen_len=1
        )
        
        # Get hardware info
        devices = []
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            # Use profiler if available
            from .optimization.profiler import SystemProfiler
            profiler = SystemProfiler()
            profile = profiler.run()
            
            # Add GPU devices
            for i in range(gpu_count):
                devices.append(DeviceSpec(
                    id=i,
                    peak_flops=profile.get('device_flops', 30e12),
                    mem_bw=profile.get('device_hbm', 1e12),
                    mem_capacity=profile.get('device_mem', 8e9),
                    h2d_bw=profile.get('pcie_bw', 16e9),
                    d2h_bw=profile.get('pcie_bw', 16e9)
                ))
            
            # Add CPU as a fallback device for offloading
            # CPU has much larger memory but slower compute
            import psutil
            cpu_mem = psutil.virtual_memory().available
            devices.append(DeviceSpec(
                id=-1,  # Special ID for CPU
                peak_flops=1e12,  # ~1 TFLOPS (much slower than GPU)
                mem_bw=50e9,  # ~50 GB/s DDR4
                mem_capacity=cpu_mem,  # Use available system RAM
                h2d_bw=profile.get('pcie_bw', 16e9),  # PCIe bandwidth
                d2h_bw=profile.get('pcie_bw', 16e9)
            ))
            print(f"[HELM Compiler] Devices: {gpu_count} GPU(s) + 1 CPU (offload)")
        else:
            # CPU-only mode
            import psutil
            cpu_mem = psutil.virtual_memory().available
            devices = [
                DeviceSpec(
                    id=-1,
                    peak_flops=1e12,
                    mem_bw=50e9,
                    mem_capacity=cpu_mem,
                    h2d_bw=10e9,
                    d2h_bw=10e9
                )
            ]
            print("[HELM Compiler] CPU-only mode")
        
        # Run optimizer
        optimizer = ParallelOptimizer(model_spec, devices)
        best_config = optimizer.optimize()
        
        if best_config is None:
            print("[HELM Compiler] ERROR: No feasible parallelism strategy found!")
            print("[HELM Compiler] Model requires more memory than available.")
            print("[HELM Compiler] Suggestions:")
            print("  1. Use quantization (FP8/INT8) to reduce memory by 2-4x")
            print("  2. Enable KV cache offloading to CPU")
            print("  3. Add more GPUs for pipeline parallelism")
            print("  4. Use a smaller model or reduce batch size")
            raise RuntimeError("No feasible parallelism configuration found. Model too large for available hardware.")
        
        tp_degree = best_config.tp_degree
        pp_degree = best_config.pp_degree
        
        print(f"[HELM Compiler] Optimizer selected: TP={tp_degree}, PP={pp_degree}")
    else:
        # Use forced strategy
        from .optimization.cost_model import ParallelConfig, StagePlacement

        tp_degree = int(forced_tp) if forced_tp else int(os.environ.get("HELM_TP_DEGREE", "1"))
        pp_degree = int(forced_pp) if forced_pp else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        print(f"[HELM Compiler] Using forced strategy: TP={tp_degree}, PP={pp_degree}")

        # Construct Manual Config
        # Heuristic: Even split of layers across PP stages
        layers_per_stage = layer_count // pp_degree
        rem = layer_count % pp_degree
        
        stages = []
        start = 0
        device_pool = list(range(pp_degree)) # Logical device IDs 0, 1, ...
        
        for i in range(pp_degree):
            count = layers_per_stage + (1 if i < rem else 0)
            end = start + count
            # Each stage needs `tp_degree` devices. 
            # In forced mode without detailed topology, we assume logical mapping:
            # Stage 0 -> Devices [0..TP-1]
            # Stage 1 -> Devices [0..TP-1] ??? No.
            # PP splits across devices. TP splits within device (or node).
            # If PP=2, TP=1: S0->Dev0, S1->Dev1.
            # If PP=2, TP=2: S0->[Dev0,Dev1], S1->[Dev2,Dev3].
            
            # Simple linear assignment for now
            # Stage i uses devices [i*TP : (i+1)*TP]
            dev_start = i * tp_degree
            stage_devs = list(range(dev_start, dev_start + tp_degree))
            
            stages.append(StagePlacement(devices=stage_devs, layer_range=(start, end)))
            start = end
            
        mb_size = int(forced_mb) if forced_mb else int(os.environ.get("HELM_MICRO_BATCH", "4"))
        
        best_config = ParallelConfig(
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            microbatches=mb_size,
            pp_stages=stages
        )
    
    # Initialize distributed if needed
    if tp_degree > 1 or pp_degree > 1:
        from .backend.distributed import get_dist_manager
        dm = get_dist_manager()
        
        if not dm.initialized:
            # Get rank and world_size from environment (set by torchrun)
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            world_size = int(os.environ.get("WORLD_SIZE", str(tp_degree * pp_degree)))
            
            print(f"[HELM Compiler] Initializing distributed: rank={rank}, world_size={world_size}")
            dm.initialize(rank, world_size)
        else:
            rank = dm.get_rank()
            world_size = dm.get_world_size()
    else:
        rank = 0
        world_size = 1
        
    print(f"[HELM Compiler] Final Strategy: PP={pp_degree}, TP={tp_degree}, Rank={rank}/{world_size}")

    # Pass 5: Tensor Parallelism (Graph Rewrite)
    if tp_degree > 1:
        print(f"[HELM Compiler] Applying Tensor Parallelism (Degree={tp_degree}, Rank={rank})...")
        tp_pass = TensorParallelPass(helm_graph, gm, tp_degree=tp_degree, rank=rank)
        tp_pass.run()
        
        # Refresh HelmGraph after mutation because TP Pass adds FX nodes
        helm_graph = HelmGraph(gm.graph) 

    # Pass 6: Partitioning
    # Update partitioner to target pp_degree stages
    # Note: Partitioner currently doesn't accept num_stages arg, likely just spills greedy.
    # We should update Partitioner to be stage-aware.
    print(f"[HELM Compiler] Running Partitioning...")
    # PASS THE CONFIG!
    partitioner = HelmPartitioner(helm_graph, config=best_config)
    partitioner.run()

    # Pass 7: Scheduling (Infer dependencies and transfers)
    scheduler = HelmScheduler(helm_graph)
    scheduler.run()
    
    # Pass 8: Execution (Insert .to)
    execution = ExecutionPass(helm_graph, gm, example_inputs)
    execution.run()
    
    # Pass 9: Pipeline Split (Physical breakdown into methods/submodules)
    pipeline_split = PipelineSplitPass(helm_graph, gm)
    pipeline_split.run()
    
    # Pass 10: Runtime Construction
    # Determine Micro-batch size from config/env
    mb_size = int(forced_mb) if forced_mb else int(os.environ.get("HELM_MICRO_BATCH", "4"))
    
    print(f"[HELM Compiler] Compilation Complete. Constructing PipelineExecutor (MB={mb_size}).")
    lazy_path = options.get("lazy_load_path")
    executor = PipelineExecutor(gm, micro_batch_size=mb_size, checkpoint_path=lazy_path)
    
    # Return the executor's run method strictly
    return executor.run_forward
