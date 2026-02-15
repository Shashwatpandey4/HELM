import torch
import torch.nn as nn
import pytest
from helm.passes.tensor_parallel import TensorParallelPass
from helm.graph import HelmGraph
from helm.layers import ShardedEmbedding, ShardedLinear

class SimpleTransformerBlock(nn.Module):
    """Minimal transformer block for testing TP."""
    def __init__(self, vocab_size=1000, hidden_size=128, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)
        
        # LayerNorm
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.ln1(x)
        
        # Attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q + k + v)  # Simplified attention
        
        x = x + attn_out
        x = self.ln2(x)
        
        # MLP
        x = x + self.down_proj(torch.relu(self.gate_proj(x)))
        
        return x


def test_tp_embedding_detection():
    """Test that TensorParallelPass detects and shards Embedding layers."""
    model = SimpleTransformerBlock()
    
    # Trace model
    example_input = torch.randint(0, 1000, (2, 10))  # batch=2, seq=10
    gm = torch.fx.symbolic_trace(model)
    
    # Create HelmGraph
    helm_graph = HelmGraph(gm.graph)
    
    # Apply TP
    tp_pass = TensorParallelPass(helm_graph, gm, tp_degree=2)
    tp_pass.run()
    
    # Verify: Should have replaced nn.Embedding with ShardedEmbedding
    found_sharded_emb = False
    for name, module in gm.named_modules():
        if isinstance(module, ShardedEmbedding):
            found_sharded_emb = True
            assert module.tp_degree == 2
            print(f"✓ Found ShardedEmbedding: {name}")
    
    assert found_sharded_emb, "TensorParallelPass should create ShardedEmbedding"


def test_tp_linear_pattern_matching():
    """Test that Linear layers are correctly classified as col/row parallel."""
    model = SimpleTransformerBlock()
    example_input = torch.randint(0, 1000, (2, 10))
    gm = torch.fx.symbolic_trace(model)
    helm_graph = HelmGraph(gm.graph)
    
    tp_pass = TensorParallelPass(helm_graph, gm, tp_degree=2)
    
    # Test pattern detection
    assert tp_pass._detect_split_style("q_proj") == "col"
    assert tp_pass._detect_split_style("k_proj") == "col"
    assert tp_pass._detect_split_style("v_proj") == "col"
    assert tp_pass._detect_split_style("gate_proj") == "col"
    assert tp_pass._detect_split_style("o_proj") == "row"
    assert tp_pass._detect_split_style("down_proj") == "row"
    assert tp_pass._detect_split_style("unknown_layer") == "unknown"
    
    print("✓ Pattern matching works correctly")


def test_tp_layernorm_replication():
    """Test that LayerNorm is detected but not sharded (replicated)."""
    model = SimpleTransformerBlock()
    example_input = torch.randint(0, 1000, (2, 10))
    gm = torch.fx.symbolic_trace(model)
    helm_graph = HelmGraph(gm.graph)
    
    tp_pass = TensorParallelPass(helm_graph, gm, tp_degree=2)
    tp_pass.run()
    
    # Verify: LayerNorm should still be nn.LayerNorm (not sharded)
    layernorm_count = 0
    for name, module in gm.named_modules():
        if isinstance(module, nn.LayerNorm):
            layernorm_count += 1
    
    assert layernorm_count == 2, "LayerNorm should remain replicated (not sharded)"
    print(f"✓ Found {layernorm_count} replicated LayerNorm modules")


def test_sharded_embedding_forward():
    """Test ShardedEmbedding forward pass correctness."""
    vocab_size = 1000
    embed_dim = 128
    tp_degree = 4
    
    # Create original embedding
    orig_emb = nn.Embedding(vocab_size, embed_dim)
    
    # Create sharded versions for each rank
    sharded_embs = [
        ShardedEmbedding.from_embedding(orig_emb, tp_degree, rank)
        for rank in range(tp_degree)
    ]
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (2, 10))
    
    # Original output
    orig_out = orig_emb(input_ids)
    
    # Sharded output (sum across ranks simulates AllReduce)
    sharded_out = sum(emb(input_ids) for emb in sharded_embs)
    
    # Verify outputs match
    assert torch.allclose(orig_out, sharded_out, atol=1e-5), "Sharded embedding output should match original"
    print("✓ ShardedEmbedding forward pass is correct")


if __name__ == "__main__":
    test_tp_embedding_detection()
    test_tp_linear_pattern_matching()
    test_tp_layernorm_replication()
    test_sharded_embedding_forward()
    print("\n✅ All TP extension tests passed!")
