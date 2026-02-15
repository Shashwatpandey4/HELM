# HELM Tools

This directory contains utilities for working with HELM.

## launch_distributed.py

Launches HELM scripts in distributed mode using `torchrun`.

### Usage

```bash
# Launch with all available GPUs
python -m helm.tools.launch_distributed your_script.py

# Specify number of processes
python -m helm.tools.launch_distributed your_script.py --nproc-per-node=4

# Multi-node execution
python -m helm.tools.launch_distributed your_script.py \
  --nnodes=2 \
  --node-rank=0 \
  --master-addr=192.168.1.1 \
  --master-port=29500
```

### Environment Variables

The launcher sets:
- `RANK`: Global rank of this process
- `LOCAL_RANK`: Local rank on this node
- `WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: Master node address
- `MASTER_PORT`: Master node port

HELM's compiler auto-detects these and initializes distributed execution.

---

## convert_checkpoint.py

Converts standard PyTorch/HuggingFace checkpoints to HELM's sharded format for lazy loading.

### Usage

```bash
# Convert HuggingFace model
python -m helm.tools.convert_checkpoint \
  --input gpt2 \
  --output ./gpt2-sharded \
  --tp-degree 2 \
  --pp-degree 1

# Convert local PyTorch checkpoint
python -m helm.tools.convert_checkpoint \
  --input ./my_model.pt \
  --output ./my_model-sharded \
  --tp-degree 4 \
  --pp-degree 2 \
  --model-type pytorch
```

### Output Format

Creates a directory with:
- `metadata.json`: Model config and shard mapping
- `stage_X_rank_Y.safetensors`: Weight shards for each PP stage and TP rank

### Loading Sharded Checkpoints

```python
from helm.checkpoint_loader import ShardedCheckpoint

# Load sharded checkpoint
checkpoint = ShardedCheckpoint("./gpt2-sharded", parallel_config)
checkpoint.load_into_model(model, rank=0)
```
