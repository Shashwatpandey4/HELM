import torch
import torch.fx as fx
import graphviz
import os

def visualize_fx_graph(gm: torch.fx.GraphModule, title: str, filename: str):
    """
    Visualizes a torch.fx.GraphModule using graphviz.
    
    Args:
        gm: The GraphModule to visualize.
        title: Title of the graph.
        filename: Filename to save the PNG to (without extension, or with).
    """
    dot = graphviz.Digraph(comment=title)
    dot.attr(label=title)
    
    # Map nodes to string IDs
    node_map = {}
    
    for node in gm.graph.nodes:
        node_id = str(id(node))
        node_map[node] = node_id
        
        # Label construction
        label = f"{node.name}\nno: {node.op}"
        if node.op == 'call_function':
            label += f"\ntarget: {node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)}"
        elif node.op == 'call_method':
            label += f"\ntarget: {node.target}"
        elif node.op == 'call_module':
            label += f"\ntarget: {node.target}"
        
        # Metadata (device, flops)
        if 'device' in node.meta:
            label += f"\ndevice: {node.meta['device']}"
        if 'flops' in node.meta:
            label += f"\nflops: {node.meta['flops']:.2e}"
            
        # Color coding
        color = 'white'
        style = 'filled'
        if node.op == 'placeholder':
            color = 'lightgrey'
            shape = 'invhouse'
        elif node.op == 'output':
            color = 'lightblue'
            shape = 'house'
        elif node.op == 'call_module':
            color = 'lightyellow'
            shape = 'box'
        elif 'dist_' in str(node.target) or 'all_reduce' in str(node.target) or 'send' in str(node.target) or 'recv' in str(node.target):
             color = 'lightcoral'
             shape = 'hexagon'
        else:
            shape = 'ellipse'
            
        dot.node(node_id, label, shape=shape, style=style, fillcolor=color)
        
        # Edges (Inputs)
        for arg in node.args:
            if isinstance(arg, fx.Node):
                dot.edge(node_map[arg], node_id)
            elif isinstance(arg, (tuple, list)):
                for item in arg:
                    if isinstance(item, fx.Node):
                        dot.edge(node_map[item], node_id)
        
        # Edges (Kwargs)
        for k, arg in node.kwargs.items():
            if isinstance(arg, fx.Node):
                 dot.edge(node_map[arg], node_id, label=k)

    # Save
    output_path = f"{filename}"
    # render(filename, view=False, cleanup=True, format='png') will save filename.png
    try:
        dot.render(output_path, view=False, cleanup=True, format='png')
        print(f"Graph saved to {output_path}.png")
    except Exception as e:
        print(f"Failed to save graph: {e}")
