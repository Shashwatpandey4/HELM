# HELM: Hardware-Aware Efficient Learning Model Compiler

HELM is a research prototype for an **automatic model parallelism compiler**. It analyzes PyTorch models and the underlying hardware to automatically determine and apply optimal **Pipeline Parallelism (PP)** strategies.

Unlike standard tools that require manual device placement (e.g., `device_map="auto"` or manual `to(device)` calls), HELM uses an analytical cost model to mathematically predict the best split points to maximize throughput and minimize memory potential OOMs.

## Key Features

*   **Hardware Detection**: Automatically detects GPU compute capacity (TFLOPS), memory bandwidth, and VRAM limits.
*   **Robust Graph Analysis**: Uses structural analysis (tracing `scaled_dot_product_attention`) to accurately detect Transformer architectures (Layers, Hidden Dimensions) directly from the `torch.fx` graph, without relying on module names.
*   **Analytical Cost Model**: Predicts the optimal pipeline split point ($k$) by modeling compute ($T_{comp}$) and communication ($T_{comm}$) costs, balancing pipeline stages.
*   **Automatic Partitioning**:
    *   **Device Placement**: Maps logical split decisions to physical graph nodes using trace-back logic.
    *   **Parameter Sharding**: Automatically moves model weights to the device where they are consumed, pruning them from other devices to save memory.
    *   **Graph Slicing**: Generates standalone, rank-specific computation graphs with sequence-aligned communication to prevent deadlocks.
*   **Distributed Runtime**: Integrated with `torchrun` for real-time multi-GPU execution using optimized P2P primitives.

## Architecture

The compilation pipeline consists of 5 sequential passes:

1.  **Hardware Analysis Pass**: Probes the system for available GPUs and their specifications.
2.  **Data Analysis Pass**: Propagates shapes, estimates FLOPs/Bytes for every operation, and deduces high-level model configuration (e.g., L=32, d_model=4096).
3.  **Cost Model Pass**: Runs an analytical solver to find the optimal layer split index that minimizes pipeline bubble overhead and fits within memory constraints.
4.  **Device Placement Pass**: Annotates every node in the graph with a Rank ID (`placement`) based on the split decision. Validates correct placement of weights and inputs.
5.  **Pipeline Parallelism Pass**: Physically splits the graph and inserts globally ordered `dist.send`/`dist.recv` nodes to ensure deadlock-free communication.

## Installation

HELM requires PyTorch 2.0+ (for `torch.fx` and `torch.compile`) and `transformers`.

```bash
pip install torch transformers accelerate
# Recommended: install 'uv' for fast project management
pip install uv
```

## Usage

### Single-Rank Compilation Mock
Demonstrates the compilation flow without requiring multiple GPUs.
```bash
export HF_TOKEN=your_hf_token
uv run runner/compile_llama.py
```

### Distributed Multi-GPU Execution
Run the model across 2 GPUs using `torchrun`.
```bash
export HF_TOKEN=your_hf_token
uv run torchrun --nproc_per_node=2 runner/run_distributed.py
```

## Performance Benchmarks

Measured with a **Batch Size of 1** and **Sequence Length of 128**:

| Setup | Model | Layers | Avg Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **2x NVIDIA RTX A6000** | Llama-2-7b | 32 | **16.42 ms** | **7,793 tokens/sec** |
| **2x NVIDIA RTX A6000** | Llama-2-13b | 40 | **28.32 ms** | **4,519 tokens/sec** |

*Note: Benchmarks represent steady-state distributed execution via `torch.compile` and HELM, excluding initial compilation overhead.*

## Implementation Details

- **Stability**: Bypasses `ShapeProp` during compilation to avoid `functorch` internal stack corruption during `torch.compile`.
- **Communication Alignment**: Uses a globally consistent topological sort for communication ops to ensure SEND/RECV pairs always match their peer's execution order.
- **DType Support**: Supports `bfloat16` and `float32` communication buffers automatically.
