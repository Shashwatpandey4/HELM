# HELM Runtime

This directory contains runtime execution components.

## Modules

### `executor.py`
Pipeline execution runtime:
- `PipelineExecutor`: Wavefront scheduling for pipeline parallelism
- Dynamic micro-batching based on sequence length
- CUDA event-based dependency management
- Adaptive batch size adjustment

### `distributed.py`
Distributed process management:
- `DistributedManager`: Singleton for managing torch.distributed
- Auto-initialization from torchrun environment variables
- NCCL/Gloo backend support
- Rank and world size management
