import torch
import torch.nn as nn
from typing import Optional

class ShardedEmbedding(nn.Module):
    """
    Vocab-Parallel Embedding for Tensor Parallelism.
    
    Shards the embedding table across the vocabulary dimension.
    Each TP rank holds a slice of the vocabulary.
    
    Example:
        vocab_size=50000, tp_degree=4
        Rank 0: tokens [0, 12500)
        Rank 1: tokens [12500, 25000)
        Rank 2: tokens [25000, 37500)
        Rank 3: tokens [37500, 50000)
    """
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        tp_degree: int = 1,
        rank: int = 0,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_degree = tp_degree
        self.rank = rank
        self.padding_idx = padding_idx
        
        # Calculate shard boundaries
        vocab_per_rank = num_embeddings // tp_degree
        self.vocab_start = rank * vocab_per_rank
        self.vocab_end = (rank + 1) * vocab_per_rank if rank < tp_degree - 1 else num_embeddings
        
        # Local embedding table (only this rank's slice)
        self.local_embedding = nn.Embedding(
            self.vocab_end - self.vocab_start,
            embedding_dim,
            padding_idx=padding_idx if padding_idx is not None and self.vocab_start <= padding_idx < self.vocab_end else None
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with vocab masking.
        
        For tokens outside this rank's range, return zeros.
        The AllReduce across TP group will sum contributions.
        """
        # Create mask for tokens in local range
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        
        # Shift input_ids to local range
        local_ids = input_ids - self.vocab_start
        local_ids = torch.clamp(local_ids, 0, self.vocab_end - self.vocab_start - 1)
        
        # Lookup embeddings
        embeddings = self.local_embedding(local_ids)
        
        # Zero out embeddings for tokens outside range
        embeddings = embeddings * mask.unsqueeze(-1).float()
        
        return embeddings
    
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding, tp_degree: int, rank: int):
        """Create ShardedEmbedding from existing nn.Embedding."""
        sharded = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            tp_degree=tp_degree,
            rank=rank,
            padding_idx=embedding.padding_idx
        )
        
        # Copy the relevant slice of weights
        vocab_per_rank = embedding.num_embeddings // tp_degree
        start = rank * vocab_per_rank
        end = (rank + 1) * vocab_per_rank if rank < tp_degree - 1 else embedding.num_embeddings
        
        with torch.no_grad():
            sharded.local_embedding.weight.copy_(embedding.weight[start:end])
            
        return sharded


class ShardedLinear(nn.Module):
    """
    Tensor-Parallel Linear Layer.
    Supports both Column and Row parallelism.
    """
    def __init__(self, in_features, out_features, bias=True, split_style='col', tp_degree=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_style = split_style
        self.tp_degree = tp_degree
        
        # Local linear layer (sharded)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x):
        return self.linear(x)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, split_style: str, tp_degree: int, rank: int):
        """Create ShardedLinear from existing nn.Linear."""
        if split_style == 'col':
            # Column parallel: shard output dimension
            out_per_rank = linear.out_features // tp_degree
            start = rank * out_per_rank
            end = (rank + 1) * out_per_rank if rank < tp_degree - 1 else linear.out_features
            
            sharded = cls(linear.in_features, end - start, bias=(linear.bias is not None), 
                         split_style='col', tp_degree=tp_degree)
            
            with torch.no_grad():
                sharded.linear.weight.copy_(linear.weight[start:end])
                if linear.bias is not None:
                    sharded.linear.bias.copy_(linear.bias[start:end])
                    
        elif split_style == 'row':
            # Row parallel: shard input dimension
            in_per_rank = linear.in_features // tp_degree
            start = rank * in_per_rank
            end = (rank + 1) * in_per_rank if rank < tp_degree - 1 else linear.in_features
            
            sharded = cls(end - start, linear.out_features, bias=(linear.bias is not None),
                         split_style='row', tp_degree=tp_degree)
            
            with torch.no_grad():
                sharded.linear.weight.copy_(linear.weight[:, start:end])
                if linear.bias is not None and rank == 0:
                    # Only rank 0 keeps bias for row parallel
                    sharded.linear.bias.copy_(linear.bias)
                elif linear.bias is not None:
                    sharded.linear.bias.zero_()
        else:
            raise ValueError(f"Unknown split_style: {split_style}")
            
        return sharded


class HelmAllReduce(nn.Module):
    """
    AllReduce communication primitive for TP.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Check if distributed is initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(x)
        return x
