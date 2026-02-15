# HELM Test Suite

This directory contains verification tests for the HELM compiler and runtime.

## Unit Tests

Run these tests to ensure individual components behave correctly:

*   **`test_cost_model.py`**: Verifies that the Cost Model Simulator correctly predicts latency and detects memory overruns.
*   **`test_optimizer.py`**: Verifies that the Optimization Search algorithm (Optimizer/Oracle) can find a feasible parallel configuration for a given model/hardware.
*   **`test_tensor_parallelism.py`**: Verifies that the `TensorParallelPass` correctly shards `nn.Linear` layers and that the `ShardedLinear` math (plus `HelmAllReduce`) matches non-sharded execution.
*   **`test_tp_extensions.py`**: Verifies TP extensions for `nn.Embedding` (vocab parallelism), `nn.LayerNorm` (replication), and pattern-based Linear detection.
*   **`test_quantization.py`**: Verifies INT8 weight-only quantization correctness, memory savings (~49%), and numerical accuracy.
*   **`test_pipeline_executor.py`**: Verifies the core `PipelineExecutor` using wavefront scheduling and CUDA streams, confirming proper data flow between Stage 0 (GPU) and Stage 1 (CPU).
*   **`test_dp_mapping.py`**: Test the logical `DeviceMesh` and coordinate mapping for Data Parallelism scenarios (PP=2, TP=2, DP=2).

## Run All Tests

```bash
uv run pytest tests/
```
