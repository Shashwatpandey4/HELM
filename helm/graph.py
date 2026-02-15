import torch
from typing import List, Dict, Any, Union, Optional
from functools import reduce

class HelmNode:
    """
    Represents a node in the Helm Graph, mirroring an FX node.
    """
    def __init__(self, name: str, fx_node: torch.fx.Node):
        self.name = name
        self.fx_node = fx_node
        self.op_type = fx_node.op
        self.target = fx_node.target
        self.args: List[Union['HelmNode', Any]] = [] 
        self.kwargs: Dict[str, Any] = {}
        self.users: List['HelmNode'] = []
        self.all_input_nodes: List['HelmNode'] = [] 
        
        # Metrics Storage
        self.flops: Optional[int] = None
        self.output_shape: Optional[List[int]] = None
        self.output_dtype: Optional[torch.dtype] = None
        
        # Backend / Partitioning
        self.device: str = "cpu" # Default placement

    @property
    def output_bytes(self) -> int:
        """Calculate total bytes of the output tensor."""
        if self.output_shape and self.output_dtype:
            num_elements = reduce(lambda x, y: x * y, self.output_shape, 1)
            return num_elements * self.output_dtype.itemsize
        return 0

    def get_output_bytes_str(self) -> str:
        """Formatted string for output bytes (e.g. '4.00 MB')."""
        b = self.output_bytes
        if b == 0:
            return ""
        if b < 1024:
            return f"{b} B"
        elif b < 1024**2:
            return f"{b/1024:.2f} KB"
        elif b < 1024**3:
            return f"{b/(1024**2):.2f} MB"
        else:
            return f"{b/(1024**3):.2f} GB"

    def __repr__(self):
        # Helper to format args nicely
        fmt_args = []
        for arg in self.args:
            if isinstance(arg, HelmNode):
                fmt_args.append(arg.name)
            else:
                fmt_args.append(str(arg))
        return f"{self.name} = {self.op_type}({self.target}, args={fmt_args})"

class HelmGraph:
    """
    A mirrored graph representation of the FX Graph.
    """
    def __init__(self, fx_graph: torch.fx.Graph):
        self.nodes: List[HelmNode] = []
        self.fx_to_helm: Dict[torch.fx.Node, HelmNode] = {}
        self.fx_graph = fx_graph # Keep ref
        
        # Global Metadata
        self.hardware_meta: Dict[str, Any] = {}
        
        self._build_from_fx(fx_graph)

    def _extract_dependencies(self, arg: Any) -> Any:
        found_deps = []
        
        def recursive_map(x):
            if isinstance(x, torch.fx.Node):
                if x in self.fx_to_helm:
                    helm_node = self.fx_to_helm[x]
                    found_deps.append(helm_node)
                    return helm_node
                else:
                    return x 
            elif isinstance(x, (list, tuple)):
                return type(x)(recursive_map(item) for item in x)
            elif isinstance(x, dict):
                return {k: recursive_map(v) for k, v in x.items()}
            else:
                return x

        mapped_arg = recursive_map(arg)
        return mapped_arg, found_deps

    def _build_from_fx(self, fx_graph: torch.fx.Graph):
        idx = 0
        for fx_node in fx_graph.nodes:
            helm_name = f"N{idx}"
            helm_node = HelmNode(helm_name, fx_node)
            self.nodes.append(helm_node)
            self.fx_to_helm[fx_node] = helm_node
            idx += 1

        for helm_node in self.nodes:
            original_args = helm_node.fx_node.args
            
            new_args = []
            all_deps = []
            
            for arg in original_args:
                mapped_arg, deps = self._extract_dependencies(arg)
                new_args.append(mapped_arg)
                all_deps.extend(deps)
            
            helm_node.args = new_args
            helm_node.all_input_nodes = all_deps 

            for dep in all_deps:
                dep.users.append(helm_node)
    
    def print_graph(self):
        print("\n--- Helm Graph (Mirrored) ---")
        if self.hardware_meta:
            print(f"Hardware Context: {self.hardware_meta}")
        for node in self.nodes:
            print(f"{node} [Depends on: {[d.name for d in node.all_input_nodes]}]")
        print("-----------------------------\n")
