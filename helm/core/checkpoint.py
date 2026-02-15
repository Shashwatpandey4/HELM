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
        # Compute layer range from parameter names
        layer_range = _compute_layer_range(list(shard_state.keys()))
        
        shard_meta = {
            'stage_id': stage_id,
            'tp_rank': tp_rank,
            'layer_range': layer_range,
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
    
    Implements actual weight sharding:
    - Linear layers: Column/Row parallelism based on name patterns
    - Embedding layers: Vocab parallelism
    - Other parameters: Replicated
    
    Returns: {(stage_id, tp_rank): {param_name: tensor}}
    """
    tp_degree = parallel_config.tp_degree
    pp_degree = parallel_config.pp_degree
    
    shards = {}
    for stage_id in range(pp_degree):
        for tp_rank in range(tp_degree):
            shards[(stage_id, tp_rank)] = {}
    
    # Detect layer structure from parameter names
    layer_params = _group_parameters_by_layer(state_dict)
    total_layers = len(layer_params['layers'])
    
    # Distribute layers across PP stages
    layers_per_stage = total_layers // pp_degree if total_layers > 0 else 0
    
    print(f"[Checkpoint Sharding] Total layers: {total_layers}, PP stages: {pp_degree}, TP degree: {tp_degree}")
    
    # Assign embedding/lm_head to first/last stage
    for name, tensor in layer_params['embeddings'].items():
        stage_id = 0  # Embeddings go to first stage
        _shard_parameter(name, tensor, shards, stage_id, tp_degree, param_type='embedding')
    
    for name, tensor in layer_params['lm_head'].items():
        stage_id = pp_degree - 1  # LM head goes to last stage
        _shard_parameter(name, tensor, shards, stage_id, tp_degree, param_type='linear')
    
    # Assign transformer layers to stages
    for layer_idx, layer_state in layer_params['layers'].items():
        stage_id = min(layer_idx // layers_per_stage, pp_degree - 1) if layers_per_stage > 0 else 0
        
        for name, tensor in layer_state.items():
            param_type = _detect_parameter_type(name)
            _shard_parameter(name, tensor, shards, stage_id, tp_degree, param_type)
    
    # Replicate other parameters across all shards
    for name, tensor in layer_params['other'].items():
        for stage_id in range(pp_degree):
            for tp_rank in range(tp_degree):
                shards[(stage_id, tp_rank)][name] = tensor.clone()
    
    return shards


def _group_parameters_by_layer(state_dict: Dict[str, torch.Tensor]) -> Dict:
    """
    Group parameters by layer index.
    
    Returns:
        {
            'embeddings': {...},
            'lm_head': {...},
            'layers': {0: {...}, 1: {...}, ...},
            'other': {...}
        }
    """
    import re
    
    grouped = {
        'embeddings': {},
        'lm_head': {},
        'layers': {},
        'other': {}
    }
    
    for name, tensor in state_dict.items():
        # Match layer index patterns: layers.0., model.layers.0., h.0., etc.
        layer_match = re.search(r'(?:layers?|h|blocks?)\.(\d+)\.', name)
        
        if 'embed' in name.lower() and 'lm_head' not in name.lower():
            grouped['embeddings'][name] = tensor
        elif 'lm_head' in name.lower() or 'output' in name.lower():
            grouped['lm_head'][name] = tensor
        elif layer_match:
            layer_idx = int(layer_match.group(1))
            if layer_idx not in grouped['layers']:
                grouped['layers'][layer_idx] = {}
            grouped['layers'][layer_idx][name] = tensor
        else:
            grouped['other'][name] = tensor
    
    return grouped


def _detect_parameter_type(name: str) -> str:
    """
    Detect parameter type from name for sharding strategy.
    
    Returns: 'linear_col', 'linear_row', 'embedding', 'layernorm', or 'other'
    """
    name_lower = name.lower()
    
    # Embedding
    if 'embed' in name_lower and 'weight' in name_lower:
        return 'embedding'
    
    # LayerNorm / RMSNorm (replicate)
    if any(x in name_lower for x in ['norm', 'ln_']):
        return 'layernorm'
    
    # Linear layers - detect split style
    if 'weight' in name_lower:
        # Column parallel patterns
        col_patterns = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'fc1', 'c_attn']
        if any(p in name_lower for p in col_patterns):
            return 'linear_col'
        
        # Row parallel patterns
        row_patterns = ['o_proj', 'down_proj', 'fc2', 'c_proj', 'out_proj']
        if any(p in name_lower for p in row_patterns):
            return 'linear_row'
        
        # Default: treat as linear column parallel
        if 'linear' in name_lower or 'fc' in name_lower:
            return 'linear_col'
    
    return 'other'


def _shard_parameter(
    name: str,
    tensor: torch.Tensor,
    shards: Dict[tuple, Dict],
    stage_id: int,
    tp_degree: int,
    param_type: str
):
    """
    Shard a single parameter across TP ranks.
    
    Args:
        name: Parameter name
        tensor: Parameter tensor
        shards: Output dictionary to populate
        stage_id: PP stage ID
        tp_degree: Tensor parallelism degree
        param_type: Type of parameter (linear_col, linear_row, embedding, etc.)
    """
    if param_type == 'embedding' and len(tensor.shape) == 2:
        # Vocab parallelism: shard dimension 0 (vocabulary)
        vocab_size = tensor.shape[0]
        vocab_per_rank = vocab_size // tp_degree
        
        for tp_rank in range(tp_degree):
            start = tp_rank * vocab_per_rank
            end = (tp_rank + 1) * vocab_per_rank if tp_rank < tp_degree - 1 else vocab_size
            shards[(stage_id, tp_rank)][name] = tensor[start:end].clone()
    
    elif param_type == 'linear_col' and len(tensor.shape) == 2:
        # Column parallel: shard dimension 0 (output features)
        out_features = tensor.shape[0]
        out_per_rank = out_features // tp_degree
        
        for tp_rank in range(tp_degree):
            start = tp_rank * out_per_rank
            end = (tp_rank + 1) * out_per_rank if tp_rank < tp_degree - 1 else out_features
            shards[(stage_id, tp_rank)][name] = tensor[start:end].clone()
    
    elif param_type == 'linear_row' and len(tensor.shape) == 2:
        # Row parallel: shard dimension 1 (input features)
        in_features = tensor.shape[1]
        in_per_rank = in_features // tp_degree
        
        for tp_rank in range(tp_degree):
            start = tp_rank * in_per_rank
            end = (tp_rank + 1) * in_per_rank if tp_rank < tp_degree - 1 else in_features
            shards[(stage_id, tp_rank)][name] = tensor[:, start:end].clone()
    
    elif param_type == 'linear_col' and 'bias' in name and len(tensor.shape) == 1:
        # Column parallel bias: shard along dimension 0
        out_features = tensor.shape[0]
        out_per_rank = out_features // tp_degree
        
        for tp_rank in range(tp_degree):
            start = tp_rank * out_per_rank
            end = (tp_rank + 1) * out_per_rank if tp_rank < tp_degree - 1 else out_features
            shards[(stage_id, tp_rank)][name] = tensor[start:end].clone()
    
    elif param_type == 'linear_row' and 'bias' in name:
        # Row parallel bias: only rank 0 keeps it
        for tp_rank in range(tp_degree):
            if tp_rank == 0:
                shards[(stage_id, tp_rank)][name] = tensor.clone()
            else:
                shards[(stage_id, tp_rank)][name] = torch.zeros_like(tensor)
    
    else:
        # Replicate (LayerNorm, other parameters)
        for tp_rank in range(tp_degree):
            shards[(stage_id, tp_rank)][name] = tensor.clone()


def _compute_layer_range(param_names: list) -> list:
    """
    Compute layer range [start, end) from parameter names.
    
    Args:
        param_names: List of parameter names in this shard
        
    Returns:
        [start_layer, end_layer] or [0, 0] if no layers found
    """
    import re
    
    layer_indices = []
    
    for name in param_names:
        # Match layer index patterns
        layer_match = re.search(r'(?:layers?|h|blocks?)\.(\d+)\.', name)
        if layer_match:
            layer_indices.append(int(layer_match.group(1)))
    
    if not layer_indices:
        return [0, 0]
    
    return [min(layer_indices), max(layer_indices) + 1]
