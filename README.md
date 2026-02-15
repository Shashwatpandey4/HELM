# HELM: Heterogeneous Execution for Large Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**HELM** is a next-generation compiler and runtime for deploying Large Language Models (LLMs) on heterogeneous hardware clusters. It seamlessly orchestrates execution across **NVIDIA GPUs**, **CPUs**, and system RAM to break the "VRAM Wall".

Unlike traditional frameworks (DeepSpeed, Megatron-LM) that require manual model rewriting, HELM hooks into `torch.compile` to automatically apply **Pipeline Parallelism (PP)**, **Tensor Parallelism (TP)**, and **Data Parallelism (DP)** to standard PyTorch models.

---

## Features

*   **Zero Code Changes**: Use standard HuggingFace/PyTorch models. just add `torch.compile(backend=helm_backend)`.
*   **Heterogeneous Pipeline**: Automatically offload layers to CPU RAM when GPU memory is full.
*   **3D Parallelism**: 
    *   **Tensor Parallelism (TP)**: Shard large matrix multiplications across GPUs for low latency.
        *   Supports `nn.Linear`, `nn.Embedding` (vocab parallelism), and `nn.LayerNorm` (replication)
        *   Pattern-based detection (q/k/v→column, o/down→row parallel)
    *   **Pipeline Parallelism (PP)**: Split model layers across devices for high throughput.
    *   **Data Parallelism (DP)**: Replicate pipelines for scale-out.
*   **Smart Cost Model**: A built-in simulator predicts latency (prefill/decode) and finds the optimal parallel strategy for your specific hardware topology before execution.
*   **Asynchronous Runtime**: Wavefront scheduling with CUDA Streams and Micro-batching for maximum GPU utilization.
*   **INT8 Quantization**: Weight-only quantization with per-channel scaling, achieving ~49% memory savings with <0.01% accuracy loss.
*   **Topology-Aware Optimization**: Automatically detects NVLink connectivity and groups GPUs for optimal TP performance.
*   **Lazy Checkpoint Loading**: Stream weights directly to sharded partitions from disk, reducing peak memory by 93% for large models.
*   **Dynamic Shape Support**: Adaptive micro-batching based on actual sequence lengths, improving throughput by up to 47% on variable-length batches.

---

## Architecture

HELM transforms a standard PyTorch `nn.Module` (traced via FX) into a distributed, pipelined runtime through a series of compiler passes.

### 1. The Compiler Stack
*   **Hardware Analyzer**: Detects system topology (GPUs, VRAM, RAM, PCIe Bandwidth).
*   **Cost Model & Optimizer**:
    *   **Simulator**: Predicts latency using a roofline model for compute and communication.
    *   **Optimizer**: Searches the configuration space (e.g., "Should I use TP=2 or PP=2?") to minimize latency.
*   **Passes**:
    *   `QuantizationPass`: Auto-casts model to FP16/BF16 or applies INT8 weight-only quantization.
    *   `TensorParallelPass`: Rewrites Linear/Embedding layers into sharded versions + `HelmAllReduce`.
    *   `HelmPartitioner`: Assigns graph nodes to devices based on the Optimizer's plan.
    *   `PipelineSplitPass`: Physically cuts the graph into `PipelineStage` submodules.

### 2. The Runtime Engine
*   **Pipeline Executor**:
    *   **Wavefront Scheduling**: Maximizes concurrency between stages.
    *   **Micro-batching**: Asynchronous execution with CUDA Streams/Events.
*   **Device Mesh**: 
    *   Manages logical-to-physical mapping for **Data Parallelism**.
    *   Example: `DP=2, PP=2` maps 4 GPUs to `[Replica0: (GPU0, GPU1), Replica1: (GPU2, GPU3)]`.
*   **Distributed Backend**:
    *   Uses `NCCL` for intra-node TP (high bandwidth).
    *   Uses `Gloo` for CPU-based coordination.

---

## Installation

**Prerequisites**:
*   Linux
*   Python 3.10+
*   PyTorch 2.0+ (with CUDA)
*   `uv` (Recommended) or `pip`

```bash
# Clone Repository
git clone https://github.com/shashwatpandey4/HELM.git
cd HELM

# Install Editable
uv pip install -e .
```

---

## Usage

### 1. Quick Start (Auto-Strategy)

The simplest way to use HELM is to let the optimizer decide the best strategy.

```python
import torch
import helm

# 1. Load your Standard Model
# For large models, load to 'meta' device or 'cpu' to avoid OOM before compilation.
with torch.device("meta"):
    model = MyLLM() 

# 2. Compile with HELM
# HELM will analyze the graph and materialize shards on the correct GPUs.
opt_model = torch.compile(model, backend=helm.compiler.helm_backend)

# 3. Run Inference
output = opt_model(input_ids)
```

### 2. Advanced Configuration (`options`)

You can control precision and parallelism strategies explicitly via the `options` dictionary.

```python
opt_model = torch.compile(
    model, 
    backend=helm.compiler.helm_backend,
    options={
        "dtype": "int8",       # INT8 quantization (~49% memory savings)
        "tp_degree": 2,        # Force 2-way Tensor Parallelism
        "micro_batch_size": 4  # Pipeline Micro-batches
    }
)
```

### 3. Environment Variables

Alternatively, use environment variables for cluster-wide configuration:

```bash
export HELM_TP_DEGREE=2
export HELM_MICRO_BATCH=4
export HELM_DUMP_GRAPH=1   # Dumps the intermediate FX graph to a file
python run_inference.py
```

---

## Cost Model Simulator

HELM includes a standalone simulator tool. You can ask it "what-if" questions without running the model. **Example: Can I run Llama-70B on 2x A100?**

```bash
python examples/cost_model_sweep.py
```

**Sample Output:**
```
Scenario                                 | Feasible   | Prefill (ms) | Note
--------------------------------------------------------------------------------
Check: 70B on Single A100                 | False      |         0.00 | OOM Device 0
Solution: 70B on 2x A100 (PP=2)           | True       |      1150.72 | 
Solution: 70B on 4x A100 (TP=4)           | True       |       460.29 | 
Hybrid: 70B on A100 + CPU (Offload)       | True       |     89602.00 | Slow but works
```

---

## Development & Testing

Run the verification suite to ensure correctness of the compiler components.

```bash
uv run pytest tests/
```

| Test File | Purpose |
|---|---|
| `test_cost_model.py` | Verifies latency predictions and OOM detection. |
| `test_optimizer.py` | Verifies that the search finds optimal (PP, TP) configs. |
| `test_tensor_parallelism.py` | Verifies correctness of Sharded Linear math (Manual Reduce). |
| `test_pipeline_executor.py` | Verifies wavefront scheduling and micro-batching correctness. |
| `test_dp_mapping.py` | Verifies logical-to-physical device mapping for large clusters. |

---

## Future Roadmap

*   **Multi-Node**: Support for multi-node clusters via TCP/Infiniband.
*   **INT4 Kernels**: Custom CUDA kernels for INT4 quantization (currently uses INT8).
*   **Flash Attention**: Integrate FlashAttention-2 inside compiled graphs.
*   **Dynamic Shape Support**: Full symbolic shape tracing for variable sequence lengths.

---

## License

[MIT License](LICENSE)
