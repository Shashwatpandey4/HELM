# HELM: Heterogeneous Execution of Language Models

**HELM** is a research project aiming to develop a **workload and hardware-aware compiler** and **distributed runtime** for PyTorch. Our goal is to create a system that automatically analyzes deep learning workloads and partitions them across heterogeneous hardware resources based on their capabilities (compute, memory, bandwidth). We pose this as a compiler technique, leveraging `torch.compile` and FX Graph transformations to democratize efficient model parallelism.

## Current State & Capabilities

HELM currently implements an end-to-end pipeline for **Automatic Pipeline Parallelism**.

### 1. The Helm Backend (Compiler)
The custom compiler backend (`helm`) transforms standard, single-device PyTorch models into distributed graphs.

*   **Hardware Analysis Pass**: Automatically queries the environment to detect GPU resources (Count, SMs, Memory, Compute Capability).
*   **Cost Model (FLOPs Analysis)**: Traces the model graph to estimate computational costs for key operators (Linear, MatMul).
*   **Heuristic Partitioning**: A greedy algorithm that uses hardware and cost metadata to slice the model graph into balanced stages matching the number of available GPUs.
*   **Topology Transformation**: Automatically injects distributed communication primitives (`dist.send`, `dist.recv`) at stage boundaries and prunes the graph so each rank executes only its assigned partition.

### 2. The Helm Runtime
We have developed a custom distributed runtime to orchestrate the execution of the partitioned graphs.

*   **Ray Orchestration**: Uses Ray to manage the lifecycle of distributed workers (Actors).
*   **NCCL Communication**: Uses standard PyTorch `distributed` (NCCL) for high-performance tensor communication between stages.
*   **Execution**: Handles initialization, graph compilation on remote workers, and data flow management.

## Usage

### Benchmarking
To verify the backend against standard `torch.compile` (Inductor):
```bash
uv run python runner/compare_backends.py
```

### Running Distributed Pipeline
To run the full distributed runtime with Ray:
```bash
uv run python runtime/ray_runtime.py
```

## Work In Progress (WIP)

We are actively working on extending the compiler to support **Hybrid Parallelism Topologies**. This includes:
*   Simultaneous **Tensor Parallelism (TP)** and **Pipeline Parallelism (PP)**.
*   Advanced partitioning strategies to automatically discover optimal hybrid configurations.
*   Compiling complex topologies for heterogeneous clusters.
