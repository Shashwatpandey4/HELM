# HELM Core

This directory contains the main source code for the HELM compiler and runtime.

## File Structure

*   **`compiler.py`**: The main entry point. It implements `helm_backend`, which hooks into `torch.compile`. It orchestrates the sequence of compiler passes (Analysis, Optimization, Partitioning, Execution).
*   **`layers.py`**: Custom PyTorch modules for distributed primitives.
    *   `ShardedLinear`: Represents a linear layer sliced across multiple devices (TP).
    *   `ShardedEmbedding`: Represents an embedding table sharded across vocabulary dimension (vocab parallelism).
    *   `HelmAllReduce`: Represents a collective communication operation.
*   **`quantization.py`**: INT8 weight-only quantization infrastructure.
    *   `QuantizedLinear`: Linear layer with INT8 weights and FP16 scales.
    *   `quantize_model_int8()`: Recursive model quantization utility.
*   **`graph.py`**: The Intermediate Representation (IR). Defines `HelmGraph` and `HelmNode`, which wrap PyTorch FX nodes with extra metadata (device placement, memory size, FLOPs).

## Sub-Modules

*   **`passes/`**: Contains the compiler passes that transform the graph.
*   **`pipeline/`**: Contains the runtime executor for pipeline parallelism.
*   **`backend/`**: Contains distributed communication utilities.
