"""
Lazy Checkpoint Loading for Sharded Models.

Enables streaming weights directly from disk into sharded partitions,
avoiding the need to load entire model into memory first.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import safetensors.torch

from .passes.cost_model import ParallelConfig


@dataclass
class ShardMetadata:
    """Metadata for a single shard."""
    stage_id: int
    tp_rank: int
    layer_range: tuple  # (start, end)
    device: str
    parameters: Dict[str, tuple]  # param_name -> (shape, dtype)


class ShardedCheckpoint:
    """
    Manages loading of sharded checkpoints.
    
    Checkpoint Format:
        checkpoint_dir/
            metadata.json          # Shard mapping, model config
            stage_0_rank_0.safetensors
            stage_0_rank_1.safetensors
            ...
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path], parallel_config: ParallelConfig):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.parallel_config = parallel_config
        
        # Load metadata
        metadata_path = self.checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.model_config = self.metadata.get('model_config', {})
        self.shard_map = self._build_shard_map()
        
    def _build_shard_map(self) -> Dict[str, ShardMetadata]:
        """Build mapping of parameter names to shard files."""
        shard_map = {}
        
        for shard_info in self.metadata['shards']:
            shard_meta = ShardMetadata(
                stage_id=shard_info['stage_id'],
                tp_rank=shard_info['tp_rank'],
                layer_range=tuple(shard_info['layer_range']),
                device=shard_info['device'],
                parameters=shard_info['parameters']
            )
            
            for param_name in shard_meta.parameters:
                shard_map[param_name] = shard_meta
        
        return shard_map
    
    def get_shard_path(self, stage_id: int, tp_rank: int) -> Path:
        """Get path to shard file."""
        return self.checkpoint_dir / f"stage_{stage_id}_rank_{tp_rank}.safetensors"
    
    def load_into_model(self, model: nn.Module, rank: int = 0, strict: bool = True):
        """
        Load weights from sharded checkpoint into model.
        
        Only loads weights that belong to this rank based on parallel_config.
        
        Args:
            model: PyTorch model (can be on meta device)
            rank: Current process rank
            strict: Whether to require all parameters to be loaded
        """
        print(f"[ShardedCheckpoint] Loading checkpoint from {self.checkpoint_dir}")
        print(f"  Parallel Config: TP={self.parallel_config.tp_degree}, PP={self.parallel_config.pp_degree}")
        
        # Determine which stage and TP rank this process owns
        stage_id, tp_rank = self._get_stage_and_tp_rank(rank)
        
        print(f"  Rank {rank} -> Stage {stage_id}, TP Rank {tp_rank}")
        
        # Load the shard file
        shard_path = self.get_shard_path(stage_id, tp_rank)
        
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        
        print(f"  Loading shard: {shard_path}")
        state_dict = safetensors.torch.load_file(str(shard_path))
        
        # Load into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if strict and missing_keys:
            raise RuntimeError(f"Missing keys in checkpoint: {missing_keys}")
        
        print(f"  Loaded {len(state_dict)} parameters")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        return model
    
    def _get_stage_and_tp_rank(self, global_rank: int) -> tuple:
        """Convert global rank to (stage_id, tp_rank)."""
        tp_degree = self.parallel_config.tp_degree
        pp_degree = self.parallel_config.pp_degree
        
        # Assume layout: [Stage0_TP0, Stage0_TP1, ..., Stage1_TP0, ...]
        stage_id = global_rank // tp_degree
        tp_rank = global_rank % tp_degree
        
        return stage_id, tp_rank


def save_sharded_checkpoint(
    model: nn.Module,
    checkpoint_dir: Union[str, Path],
    parallel_config: ParallelConfig,
    model_config: Optional[Dict] = None
):
    """
    Save model as sharded checkpoint.
    
    Args:
        model: PyTorch model to save
        checkpoint_dir: Directory to save checkpoint
        parallel_config: Parallelism configuration
        model_config: Optional model configuration dict
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[ShardedCheckpoint] Saving checkpoint to {checkpoint_dir}")
    
    # Get model state dict
    state_dict = model.state_dict()
    
    # Partition parameters based on parallel config
    shards = _partition_state_dict(state_dict, parallel_config)
    
    # Save each shard
    shard_metadata = []
    for (stage_id, tp_rank), shard_state in shards.items():
        shard_path = checkpoint_dir / f"stage_{stage_id}_rank_{tp_rank}.safetensors"
        
        # Save using safetensors
        safetensors.torch.save_file(shard_state, str(shard_path))
        
        # Record metadata
        shard_meta = {
            'stage_id': stage_id,
            'tp_rank': tp_rank,
            'layer_range': [0, 0],  # TODO: Compute from param names
            'device': 'cuda',
            'parameters': {
                name: [list(tensor.shape), str(tensor.dtype)]
                for name, tensor in shard_state.items()
            }
        }
        shard_metadata.append(shard_meta)
        
        print(f"  Saved shard: {shard_path} ({len(shard_state)} params)")
    
    # Save metadata
    metadata = {
        'model_config': model_config or {},
        'parallel_config': {
            'tp_degree': parallel_config.tp_degree,
            'pp_degree': parallel_config.pp_degree,
            'microbatches': parallel_config.microbatches
        },
        'shards': shard_metadata
    }
    
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[ShardedCheckpoint] Saved {len(shards)} shards")


def _partition_state_dict(
    state_dict: Dict[str, torch.Tensor],
    parallel_config: ParallelConfig
) -> Dict[tuple, Dict[str, torch.Tensor]]:
    """
    Partition state dict into shards based on parallel config.
    
    Returns: {(stage_id, tp_rank): {param_name: tensor}}
    """
    tp_degree = parallel_config.tp_degree
    pp_degree = parallel_config.pp_degree
    
    shards = {}
    
    # Simple strategy: Distribute layers evenly across PP stages
    # For TP: Shard Linear/Embedding weights
    
    for stage_id in range(pp_degree):
        for tp_rank in range(tp_degree):
            shards[(stage_id, tp_rank)] = {}
    
    # Assign parameters to shards
    # This is a simplified version - real implementation would use layer detection
    total_params = len(state_dict)
    params_per_stage = total_params // pp_degree
    
    for idx, (name, tensor) in enumerate(state_dict.items()):
        stage_id = min(idx // params_per_stage, pp_degree - 1)
        
        # For now, replicate across TP ranks (no actual sharding)
        # Real implementation would shard Linear/Embedding weights
        for tp_rank in range(tp_degree):
            shards[(stage_id, tp_rank)][name] = tensor.clone()
    
    return shards
