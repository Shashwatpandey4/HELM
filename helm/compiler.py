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
from .pipeline.executor import PipelineExecutor
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
    quant_pass.run()
    
    # 0. Dump Original FX Graph
    if os.environ.get("HELM_DUMP_GRAPH", "0") == "1":
        print("[HELM Compiler] Dumping raw FX graph to 'qwen_fx_graph.py'...")
        with open("qwen_fx_graph.py", "w") as f:
            f.write(gm.code)
    
    node_count = len(gm.graph.nodes)
    print(f"[HELM Compiler] Total FX Nodes: {node_count}")
    
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
    
    # Use optimizer if no forced strategy
    if not forced_tp and not forced_pp:
        print("[HELM Compiler] Running ParallelOptimizer to find optimal strategy...")
        
        from .passes.optimizer import ParallelOptimizer
        from .passes.cost_model import ModelSpec, LayerSpec, DeviceSpec
        
        # Build ModelSpec from graph analysis
        # Use dynamic analyzer results
        total_params = sum(node.param_bytes for node in helm_graph.nodes if hasattr(node, 'param_bytes'))
        total_flops = sum(node.flops for node in helm_graph.nodes if hasattr(node, 'flops'))
        
        # Estimate layer count from graph
        layer_count = len([n for n in helm_graph.nodes if 'layer' in n.name.lower()])
        layer_count = max(layer_count, 1)
        
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
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            # Use profiler if available
            from .passes.profiler import SystemProfiler
            profiler = SystemProfiler()
            profile = profiler.run()
            
            devices = [
                DeviceSpec(
                    id=i,
                    peak_flops=profile.get('device_flops', 30e12),
                    mem_bw=profile.get('device_hbm', 1e12),
                    mem_capacity=torch.cuda.get_device_properties(i).total_memory,
                    h2d_bw=profile.get('pcie_bw', 16e9),
                    d2h_bw=profile.get('pcie_bw', 16e9)
                )
                for i in range(gpu_count)
            ]
        else:
            gpu_count = 1
            devices = [
                DeviceSpec(
                    id=0,
                    peak_flops=1e12,
                    mem_bw=100e9,
                    mem_capacity=16e9,
                    h2d_bw=10e9,
                    d2h_bw=10e9
                )
            ]
        
        # Run optimizer
        optimizer = ParallelOptimizer(model_spec, devices)
        best_config = optimizer.optimize()
        
        tp_degree = best_config.tp_degree
        pp_degree = best_config.pp_degree
        
        print(f"[HELM Compiler] Optimizer selected: TP={tp_degree}, PP={pp_degree}")
    else:
        # Use forced strategy
        tp_degree = int(forced_tp) if forced_tp else int(os.environ.get("HELM_TP_DEGREE", "1"))
        pp_degree = int(forced_pp) if forced_pp else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        print(f"[HELM Compiler] Using forced strategy: TP={tp_degree}, PP={pp_degree}")
    
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
    partitioner = HelmPartitioner(helm_graph) # TODO: Pass num_stages
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
    executor = PipelineExecutor(gm, micro_batch_size=mb_size)
    
    # Return the executor's run method strictly
    return executor.run_forward
