import torch
import torch.nn as nn
from ..graph import HelmGraph, HelmNode
from ..layers import ShardedLinear, HelmAllReduce

class TensorParallelPass:
    """
    Pass: Tensor Parallelism (TP)
    Identifies layers suitable for TP and splits them in the HelmGraph.
    
    Supported Modules:
    - nn.Linear: Column/Row parallelism based on name patterns
    - nn.Embedding: Vocab parallelism
    - nn.LayerNorm: Replicated (not sharded)
    
    Strategy:
    1. Embedding: Shard vocabulary dimension
    2. Linear (q/k/v/gate/up): Column parallel
    3. Linear (o/down): Row parallel + AllReduce
    4. LayerNorm: Replicate across all TP ranks
    """
    def __init__(self, graph: HelmGraph, gm: torch.fx.GraphModule, tp_degree: int = 1):
        self.graph = graph
        self.gm = gm
        self.tp_degree = tp_degree
        
    def run(self):
        if self.tp_degree <= 1:
            print("[TensorParallelPass] TP Degree <= 1. Skipping.")
            return

        print(f"\n[TensorParallelPass] Analyzing Graph for TP (Degree={self.tp_degree})...")
        
        # 1. Identify Candidates
        linear_candidates = []
        embedding_candidates = []
        layernorm_candidates = []
        
        for node in self.graph.nodes:
            if node.op_type == 'call_module':
                target_name = node.target
                try:
                    mod = self.gm.get_submodule(target_name)
                    
                    if isinstance(mod, nn.Linear):
                        print(f"  Found Linear: {node.name} ({target_name}) Shape: {mod.weight.shape}")
                        linear_candidates.append((node, mod, target_name))
                    elif isinstance(mod, nn.Embedding):
                        print(f"  Found Embedding: {node.name} ({target_name}) Vocab: {mod.num_embeddings}")
                        embedding_candidates.append((node, mod, target_name))
                    elif isinstance(mod, nn.LayerNorm):
                        print(f"  Found LayerNorm: {node.name} ({target_name}) - Will Replicate")
                        layernorm_candidates.append((node, mod, target_name))
                except AttributeError:
                    pass
        
        print(f"  Summary: {len(linear_candidates)} Linear, {len(embedding_candidates)} Embedding, {len(layernorm_candidates)} LayerNorm")
        
        
        # 2. Process Embeddings (Vocab Parallelism)
        for node, mod, target_name in embedding_candidates:
            print(f"  [Mutate] {node.name} -> Vocab Parallel Embedding")
            self._apply_vocab_parallel(node, mod, target_name)
        
        # 3. Process Linear Layers (Smart Pattern Matching)
        processed_nodes = set()
        
        for node, mod, target_name in linear_candidates:
            if node in processed_nodes: continue
            
            # Pattern-based detection
            split_style = self._detect_split_style(target_name)
            
            if split_style == 'col':
                print(f"  [Mutate] {node.name} -> Column Parallel (Pattern: {target_name})")
                self._apply_col_parallel(node, mod)
            elif split_style == 'row':
                print(f"  [Mutate] {node.name} -> Row Parallel (Pattern: {target_name})")
                self._apply_row_parallel(node, mod)
            else:
                print(f"  [Skip] {node.name} - No clear pattern, skipping TP")
                
            processed_nodes.add(node)
        
        # 4. LayerNorm: No action needed (replicated by default)
        print(f"  LayerNorm modules will be replicated across TP ranks (no sharding).")
            
        print("[TensorParallelPass] Mutation Complete.")

    def _detect_split_style(self, target_name: str) -> str:
        """
        Detect whether a Linear layer should be column or row parallel.
        Based on common naming patterns in Transformers.
        """
        name_lower = target_name.lower()
        
        # Column parallel patterns (input replication)
        col_patterns = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'fc1']
        for pattern in col_patterns:
            if pattern in name_lower:
                return 'col'
        
        # Row parallel patterns (output reduction)
        row_patterns = ['o_proj', 'down_proj', 'fc2', 'out_proj']
        for pattern in row_patterns:
            if pattern in name_lower:
                return 'row'
        
        # Fallback: unknown
        return 'unknown'

    def _apply_vocab_parallel(self, node, mod, target_name):
        """
        Apply Vocab Parallelism to Embedding layer.
        Replace nn.Embedding with ShardedEmbedding + AllReduce.
        """
        from ..layers import ShardedEmbedding, HelmAllReduce
        
        # For prototype: assume rank 0 (single process)
        # In real distributed: get rank from torch.distributed.get_rank()
        rank = 0
        
        sharded_emb = ShardedEmbedding(
            mod.num_embeddings,
            mod.embedding_dim,
            tp_degree=self.tp_degree,
            rank=rank,
            padding_idx=mod.padding_idx
        )
        
        # Copy weights
        sharded_emb = ShardedEmbedding.from_embedding(mod, self.tp_degree, rank)
        
        # Add to graph
        sharded_name = f"{target_name}_sharded_vocab"
        self.gm.add_submodule(sharded_name, sharded_emb)
        
        # AllReduce
        all_reduce_mod = HelmAllReduce()
        all_reduce_name = f"{target_name}_vocab_allreduce"
        self.gm.add_submodule(all_reduce_name, all_reduce_mod)
        
        # Replace in FX graph
        original_fx_node = node.fx_node
        
        with self.gm.graph.inserting_after(original_fx_node):
            new_emb_node = self.gm.graph.call_module(sharded_name, args=original_fx_node.args, kwargs=original_fx_node.kwargs)
            
        with self.gm.graph.inserting_after(new_emb_node):
            all_reduce_node = self.gm.graph.call_module(all_reduce_name, args=(new_emb_node,))
            original_fx_node.replace_all_uses_with(all_reduce_node)
            self.gm.graph.erase_node(original_fx_node)
        
        self.gm.recompile()
        print(f"    Replaced {target_name} with ShardedEmbedding + AllReduce [TP={self.tp_degree}]")
    def _apply_col_parallel(self, node, mod):
        # Physical Graph Rewrite:
        # Replace original nn.Linear with ShardedLinear(col)
        
        # 1. Create Sharded Module
        sharded_mod = ShardedLinear(mod.in_features, mod.out_features // self.tp_degree, 
                                    bias=(mod.bias is not None), 
                                    split_style='col', 
                                    tp_degree=self.tp_degree)
        
        # 2. Add to GraphModule
        sharded_name = f"{node.name}_sharded_col"
        self.gm.add_submodule(sharded_name, sharded_mod)
        
        # 3. Replace Node in FX Graph
        # We need to find the specific fx node
        # node.fx_node is the key
        original_fx_node = node.fx_node
        
        with self.gm.graph.inserting_after(original_fx_node):
            # Create new call_module
            new_node = self.gm.graph.call_module(sharded_name, args=original_fx_node.args, kwargs=original_fx_node.kwargs)
            original_fx_node.replace_all_uses_with(new_node)
            
            # Remove old node from graph
            self.gm.graph.erase_node(original_fx_node)
            
        print(f"    Replaced {node.name} with ShardedLinear(col) [TP={self.tp_degree}]")

    def _apply_row_parallel(self, node, mod):
        # Physical Graph Rewrite:
        # Replace original nn.Linear with ShardedLinear(row) -> AllReduce
        
        # 1. Create Sharded Module
        sharded_mod = ShardedLinear(mod.in_features // self.tp_degree, mod.out_features, 
                                    bias=(mod.bias is not None), 
                                    split_style='row', 
                                    tp_degree=self.tp_degree)
        
        # 2. Add to GraphModule
        sharded_name = f"{node.name}_sharded_row"
        self.gm.add_submodule(sharded_name, sharded_mod)
        
        # 3. AllReduce Module
        all_reduce_mod = HelmAllReduce()
        all_reduce_name = f"{node.name}_all_reduce"
        self.gm.add_submodule(all_reduce_name, all_reduce_mod)
        
        # 4. Replace Node in FX Graph
        original_fx_node = node.fx_node
        
        with self.gm.graph.inserting_after(original_fx_node):
            # A. Sharded Linear
            new_linear_node = self.gm.graph.call_module(sharded_name, args=original_fx_node.args, kwargs=original_fx_node.kwargs)
            
        with self.gm.graph.inserting_after(new_linear_node):
            # B. AllReduce (Takes output of linear)
            all_reduce_node = self.gm.graph.call_module(all_reduce_name, args=(new_linear_node,))
            
            # Replace uses
            original_fx_node.replace_all_uses_with(all_reduce_node)
            
            # Remove old node
            self.gm.graph.erase_node(original_fx_node)
            
        print(f"    Replaced {node.name} with ShardedLinear(row) -> AllReduce [TP={self.tp_degree}]")
        
        # Recompile graph to finalize changes
        self.gm.recompile()
