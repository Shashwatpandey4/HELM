"""
Utilities for handling dynamic shapes and padding.
"""

import torch
from typing import List, Tuple, Optional


def pad_to_multiple(tensor: torch.Tensor, multiple: int, dim: int = 1) -> Tuple[torch.Tensor, int]:
    """
    Pad tensor to nearest multiple along specified dimension.
    
    Args:
        tensor: Input tensor
        multiple: Pad to this multiple
        dim: Dimension to pad (default: 1 for sequence length)
        
    Returns:
        (padded_tensor, original_length)
    """
    original_size = tensor.size(dim)
    
    if original_size % multiple == 0:
        return tensor, original_size
    
    target_size = ((original_size + multiple - 1) // multiple) * multiple
    pad_size = target_size - original_size
    
    # Create padding specification
    # PyTorch pad works from last dim backwards
    num_dims = tensor.dim()
    pad_spec = [0, 0] * num_dims
    pad_spec[(num_dims - dim - 1) * 2 + 1] = pad_size
    
    padded = torch.nn.functional.pad(tensor, pad_spec, value=0)
    
    return padded, original_size


def create_attention_mask(seq_lens: List[int], max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create attention mask for variable-length sequences.
    
    Args:
        seq_lens: List of sequence lengths
        max_len: Maximum sequence length (default: max of seq_lens)
        
    Returns:
        Attention mask of shape (batch_size, max_len)
        1 for valid tokens, 0 for padding
    """
    if max_len is None:
        max_len = max(seq_lens)
    
    batch_size = len(seq_lens)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, length in enumerate(seq_lens):
        mask[i, :length] = 1
    
    return mask


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive generation.
    
    Args:
        seq_len: Sequence length
        device: Target device
        
    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.bool()


def batch_sequences(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    pad_to_multiple: Optional[int] = None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Batch variable-length sequences with padding.
    
    Args:
        sequences: List of 1D tensors with different lengths
        padding_value: Value to use for padding
        pad_to_multiple: If set, pad to nearest multiple of this value
        
    Returns:
        (batched_tensor, original_lengths)
    """
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    
    if pad_to_multiple is not None:
        max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    
    batch_size = len(sequences)
    batched = torch.full((batch_size, max_len), padding_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        batched[i, :len(seq)] = seq
    
    return batched, lengths


def compute_effective_batch_size(seq_lens: List[int], max_tokens: int) -> int:
    """
    Compute effective batch size to stay under max_tokens budget.
    
    Useful for dynamic batching where we want to maximize throughput
    while staying within memory constraints.
    
    Args:
        seq_lens: List of sequence lengths
        max_tokens: Maximum total tokens allowed
        
    Returns:
        Effective batch size
    """
    sorted_lens = sorted(seq_lens, reverse=True)
    
    total_tokens = 0
    effective_bs = 0
    
    for length in sorted_lens:
        if total_tokens + length <= max_tokens:
            total_tokens += length
            effective_bs += 1
        else:
            break
    
    return max(1, effective_bs)


class DynamicBatcher:
    """
    Dynamic batching utility for variable-length sequences.
    
    Automatically groups sequences to maximize GPU utilization
    while staying within memory constraints.
    """
    
    def __init__(self, max_tokens: int, max_batch_size: int = 32):
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        
    def create_batches(self, sequences: List[torch.Tensor]) -> List[List[int]]:
        """
        Group sequences into batches.
        
        Args:
            sequences: List of input sequences
            
        Returns:
            List of batches (each batch is a list of sequence indices)
        """
        # Sort by length (descending) for better packing
        indexed_seqs = [(i, len(seq)) for i, seq in enumerate(sequences)]
        indexed_seqs.sort(key=lambda x: x[1], reverse=True)
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, length in indexed_seqs:
            # Check if adding this sequence would exceed limits
            new_tokens = current_tokens + length
            
            if (len(current_batch) < self.max_batch_size and 
                new_tokens <= self.max_tokens):
                current_batch.append(idx)
                current_tokens = new_tokens
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_tokens = length
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
