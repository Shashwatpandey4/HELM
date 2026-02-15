# HELM Runtime Engine

This directory contains the core execution logic for running compiled pipelines.

## File Structure

*   **`executor.py` (`PipelineExecutor`)**: The runtime driver that executes the compiled FX graph.
    *   **Wavefront Scheduling**: It schedules independent micro-batches across multiple stages concurrently using CUDA streams.
    *   **CUDA Events**: Inserts explicit synchronization events at stage boundaries to ensure correct pipelining.
    *   **PipelineStage**: A lightweight wrapper around a `torch.nn.Module` (a shard of the original network) that ties it to a specific device and CUDA stream.

## Usage

This module is instantiated automatically by the `compiler.py`. However, it can be tested in isolation:

```python
from helm.pipeline import PipelineExecutor
# gm is a split FX graph module
executor = PipelineExecutor(gm, micro_batch_size=4)
output = executor.run_forward(inputs)
```
