# HELM Core Components

This directory contains the core building blocks of HELM.

## Modules

### `layers.py`
Tensor-parallel layer implementations:
- `ShardedLinear`: Column/row parallel linear layers
- `ShardedEmbedding`: Vocab-parallel embedding layers
- `HelmAllReduce`: Communication primitive for TP

### `quantization.py`
Quantization utilities:
- `QuantizedLinear`: INT8 weight-only quantized linear layer
- `quantize_model()`: Model-wide quantization
- Per-channel symmetric quantization

### `checkpoint.py`
Checkpoint loading and saving:
- `ShardedCheckpoint`: Lazy checkpoint loader for sharded models
- `save_sharded_checkpoint()`: Save model in sharded format
- Real weight sharding (not replication)
- Layer detection and range computation
