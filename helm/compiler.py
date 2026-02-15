import torch
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
    
    # Pass 4: Strategy Selection
    # Extract constraints from options
    forced_tp = options.get("tp_degree")
    forced_mb = options.get("micro_batch_size")
    
    tp_degree = int(forced_tp) if forced_tp else int(os.environ.get("HELM_TP_DEGREE", "1"))
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        pp_degree = max(1, gpu_count) # Use all GPUs for PP by default
    else:
        pp_degree = 1
        
    print(f"[HELM Compiler] Strategy Selected: PP={pp_degree}, TP={tp_degree}")

    # Pass 5: Tensor Parallelism (Graph Rewrite)
    if tp_degree > 1:
        print(f"[HELM Compiler] Applying Tensor Parallelism (Degree={tp_degree})...")
        tp_pass = TensorParallelPass(helm_graph, gm, tp_degree=tp_degree)
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
