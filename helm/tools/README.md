# HELM Tools

This directory contains utilities for working with HELM.

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
