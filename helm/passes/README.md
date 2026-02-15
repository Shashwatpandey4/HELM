# HELM Compiler Passes

This directory contains the core transformation passes that optimize and rewrite the neural network graph.

## Passes

The compiler executes these passes sequentially:

1.  **`analysis.py` (`DynamicAnalyzer`)**: Profiles a model by meta-execution to estimate FLOPs and Activation Memory usage.
2.  **`hardware.py` (`HardwareAnalyzer`)**: Detects system topology (GPUs, VRAM, RAM, PCIe/NVLink Bandwidth).
3.  **`optimizer.py` (`ParallelOptimizer`)**: The brain of the compiler. It uses the Cost Model to search for the optimal parallel strategy (e.g., Should I use TP=4 or PP=2? Which layers go on GPU 0?).
4.  **`quantization.py` (`QuantizationPass`)**: Handles model precision conversion and quantization.
    *   FP16/BF16/FP32: Standard dtype conversion
    *   INT8: Weight-only per-channel quantization (~49% memory savings)
    *   INT4: Placeholder (falls back to INT8)
5.  **`tensor_parallel.py` (`TensorParallelPass`)**: Applies 1D Megatron-LM style tensor parallelism.
    *   Shards `nn.Linear` layers (pattern-based: q/k/v→col, o/down→row)
    *   Shards `nn.Embedding` layers (vocab parallelism)
    *   Replicates `nn.LayerNorm` (no sharding)
    *   Inserts `HelmAllReduce` communication nodes
6.  **`partitioner.py` (`HelmPartitioner`)**: Assigns each graph node to a specific device (`cuda:0`, `cpu`, etc.) based on the Optimizer's plan. Can fall back to a Greedy strategy if no plan exists.
7.  **`pipeline_split.py` (`PipelineSplitPass`)**: Physically cuts the single FX graph into multiple `PipelineStage` sub-modules based on device placement boundaries.
8.  **`execution.py` (`ExecutionPass`)**: Inserts explicit `.to(device)` data movement calls between nodes that cross device boundaries.
9.  **`scheduler.py` (`HelmScheduler`)**: Infers execution dependencies and topological ordering.

## Utilities

*   **`cost_model.py` (`HelmCostModel`)**: An analytical simulator (Oracle) that predicts the latency (Prefill/Decode) of a given parallel configuration using a roofline model.
