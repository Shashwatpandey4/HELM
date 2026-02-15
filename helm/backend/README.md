# HELM Backend (Distributed)

This directory implements the distributed communication primitives used for Data Parallelism (DP) and low-level synchronization.

## File Structure

*   **`distributed.py` (`DistributedManager`)**: Initializes the `torch.distributed` process groups (NCCL/Gloo).
    *   Automatically falls back to `gloo` if CUDA is unavailable.
*   **`mesh.py` (`DeviceMesh`)**: Maps logical ranks to physical hardware coordinates.
    *   Example: `(DP=1, PP=2, TP=2)` maps 4 GPUs to specific ranks.
    *   Essential for large-scale replication of the model pipeline.
*   **`topology.py` (`GPUTopology`)**: Detects GPU connectivity and NVLink topology.
    *   Builds bandwidth matrix using P2P capability detection
    *   Provides greedy clique search for optimal TP group selection
    *   Enables topology-aware device grouping for better multi-GPU utilization

## Usage

This module is typically internal to `PipelineExecutor` but can be used directly for low-level configuration:

```python
from helm.backend import DistributedManager
manager = DistributedManager()
manager.initialize(rank=0, world_size=4)
```
