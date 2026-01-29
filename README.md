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
    *   **Graph Slicing**: Generates standalone, rank-specific computation graphs with explicit `dist.send`/`dist.recv` communication instructions.

## Architecture

The compilation pipeline consists of 5 sequential passes:

1.  **Hardware Analysis Pass**: Probes the system for available GPUs and their specifications.
2.  **Data Analysis Pass**: Propagates shapes, estimates FLOPs/Bytes for every operation, and deduces high-level model configuration (e.g., L=32, d_model=4096).
3.  **Cost Model Pass**: Runs an analytical solver to find the optimal layer split index that minimizes pipeline bubble overhead and fits within memory constraints.
4.  **Device Placement Pass**: Annotates every node in the graph with a Rank ID (`placement`) based on the split decision. Validates correct placement of weights and inputs.
5.  **Pipeline Parallelism Pass**: Physically splits the graph.
    *   **Rank 0**: Contains layers $0 \dots k-1$. Ends with `dist.send`.
    *   **Rank 1**: Contains layers $k \dots L$. Starts with `dist.recv`.

## Installation

HELM requires PyTorch 2.0+ (for `torch.fx` and `torch.compile`) and `transformers`.

```bash
pip install torch transformers accelerate
# Recommended: install 'uv' for fast script management
pip install uv
```

## Usage

The repository provides a runner script to demonstrate the compilation of Llama-2-7b.

**Note:** You will need a Hugging Face token to load Llama-2.

```bash
# Export your token
export HF_TOKEN=your_hf_token

# Run the compilation demo
uv run python runner/compile_llama.py
```

### Expected Output

The script will:
1.  Load Llama-2-7b on the `Meta` device (no memory usage).
2.  Run the HELM pipeline.
3.  Print the detected configuration and optimal split (e.g., "Split before Layer 16").
4.  Simulate the execution of the partitioned graphs to verify connectivity and parameter sharding.

## Current Status

This repository is a stripped-down **Core Compiler** implementation. It generates partitioned Intermediate Representations (IR) suitable for distributed execution. Integration with a distributed runtime (e.g., `torch.distributed.run` or Ray) is the intended next step for actual multi-GPU deployment.
