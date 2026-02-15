import torch
import torch.nn as nn
from helm.utils.padding import (
    pad_to_multiple,
    create_attention_mask,
    create_causal_mask,
    batch_sequences,
    compute_effective_batch_size,
    DynamicBatcher
)


def test_pad_to_multiple():
    """Test padding tensor to multiple."""
    tensor = torch.randn(2, 13, 64)  # batch=2, seq=13, hidden=64
    
    padded, original_len = pad_to_multiple(tensor, multiple=8, dim=1)
    
    assert padded.shape == (2, 16, 64), f"Expected (2, 16, 64), got {padded.shape}"
    assert original_len == 13
    print("✓ pad_to_multiple works")


def test_create_attention_mask():
    """Test attention mask creation."""
    seq_lens = [5, 8, 3]
    mask = create_attention_mask(seq_lens, max_len=10)
    
    assert mask.shape == (3, 10)
    assert mask[0, :5].all() and not mask[0, 5:].any()
    assert mask[1, :8].all() and not mask[1, 8:].any()
    assert mask[2, :3].all() and not mask[2, 3:].any()
    print("✓ create_attention_mask works")


def test_batch_sequences():
    """Test batching variable-length sequences."""
    seqs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 7, 8, 9])
    ]
    
    batched, lengths = batch_sequences(seqs, padding_value=0)
    
    assert batched.shape == (3, 4)  # max_len=4
    assert lengths == [3, 2, 4]
    assert (batched[0] == torch.tensor([1, 2, 3, 0])).all()
    assert (batched[1] == torch.tensor([4, 5, 0, 0])).all()
    print("✓ batch_sequences works")


def test_compute_effective_batch_size():
    """Test effective batch size computation."""
    seq_lens = [100, 200, 150, 300, 50]
    max_tokens = 500
    
    effective_bs = compute_effective_batch_size(seq_lens, max_tokens)
    
    # Should fit: 300 + 200 = 500 (2 sequences)
    assert effective_bs == 2
    print(f"✓ compute_effective_batch_size: {effective_bs} sequences fit in {max_tokens} tokens")


def test_dynamic_batcher():
    """Test dynamic batching."""
    sequences = [
        torch.randint(0, 1000, (length,))
        for length in [50, 100, 150, 200, 80, 120, 30, 250]
    ]
    
    batcher = DynamicBatcher(max_tokens=400, max_batch_size=4)
    batches = batcher.create_batches(sequences)
    
    print(f"  Created {len(batches)} batches from {len(sequences)} sequences")
    
    # Verify no batch exceeds limits
    for batch_indices in batches:
        total_tokens = sum(len(sequences[i]) for i in batch_indices)
        assert total_tokens <= 400, f"Batch exceeds max_tokens: {total_tokens}"
        assert len(batch_indices) <= 4, f"Batch exceeds max_batch_size: {len(batch_indices)}"
    
    print("✓ DynamicBatcher works")


if __name__ == "__main__":
    test_pad_to_multiple()
    test_create_attention_mask()
    test_batch_sequences()
    test_compute_effective_batch_size()
    test_dynamic_batcher()
    print("\n✅ All dynamic shape tests passed!")
