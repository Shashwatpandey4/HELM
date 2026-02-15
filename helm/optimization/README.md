# HELM Optimization

This directory contains optimization and cost modeling components.

## Modules

### `optimizer.py`
Parallelization strategy optimizer:
- `ParallelOptimizer`: Finds optimal TP/PP/MB configuration
- Macro search over parallelism degrees
- TP group enumeration with topology awareness
- PP partitioning with beam search

### `cost_model.py`
Performance cost modeling:
- `HelmCostModel`: Estimates latency and throughput
- `ModelSpec`, `LayerSpec`, `DeviceSpec`: Specification classes
- `CalibrationDB`: Hardware calibration database
- Overlap modeling for TP communication

### `profiler.py`
Hardware profiling:
- `SystemProfiler`: Micro-benchmarks for GPU hardware
- HBM bandwidth measurement
- Compute TFLOPS measurement
- P2P bandwidth matrix
- Heterogeneity detection

### `topology.py`
GPU topology detection:
- `GPUTopology`: NVLink/PCIe connectivity detection
- nvidia-smi topology parsing
- Bandwidth matrix construction
- TP group selection with greedy clique search
