"""
Checkpoint Conversion Tool.

Converts standard PyTorch/HuggingFace checkpoints to HELM sharded format.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from helm.checkpoint_loader import save_sharded_checkpoint
from helm.passes.cost_model import ParallelConfig, PPStage


def convert_checkpoint(
    input_path: str,
    output_path: str,
    tp_degree: int = 1,
    pp_degree: int = 1,
    model_type: str = "auto"
):
    """
    Convert checkpoint to sharded format.
    
    Args:
        input_path: Path to input checkpoint (HuggingFace model ID or local path)
        output_path: Path to output sharded checkpoint directory
        tp_degree: Tensor parallelism degree
        pp_degree: Pipeline parallelism degree
        model_type: Model type (auto, hf, pytorch)
    """
    print(f"[ConvertCheckpoint] Converting {input_path} -> {output_path}")
    print(f"  TP={tp_degree}, PP={pp_degree}")
    
    # Load model
    if model_type == "auto" or model_type == "hf":
        model, config = load_huggingface_model(input_path)
    elif model_type == "pytorch":
        model, config = load_pytorch_checkpoint(input_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create parallel config
    # For simplicity, create even PP stages
    pp_stages = []
    for i in range(pp_degree):
        stage = PPStage(
            stage_id=i,
            devices=list(range(i * tp_degree, (i + 1) * tp_degree)),
            layer_range=(0, 0)  # Will be computed during partitioning
        )
        pp_stages.append(stage)
    
    parallel_config = ParallelConfig(
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        microbatches=1,
        pp_stages=pp_stages
    )
    
    # Save sharded checkpoint
    save_sharded_checkpoint(
        model=model,
        checkpoint_dir=output_path,
        parallel_config=parallel_config,
        model_config=config
    )
    
    print(f"[ConvertCheckpoint] Conversion complete!")
    print(f"  Output: {output_path}")


def load_huggingface_model(model_id: str):
    """Load HuggingFace model."""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")
    
    print(f"  Loading HuggingFace model: {model_id}")
    
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    config_dict = config.to_dict()
    
    return model, config_dict


def load_pytorch_checkpoint(checkpoint_path: str):
    """Load PyTorch checkpoint."""
    print(f"  Loading PyTorch checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model and config
    if isinstance(checkpoint, dict):
        model_state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        config = checkpoint.get('config', {})
    else:
        model_state = checkpoint
        config = {}
    
    # Create a simple wrapper model
    class CheckpointModel(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.load_state_dict(state_dict, strict=False)
    
    model = CheckpointModel(model_state)
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to HELM sharded format")
    parser.add_argument("--input", required=True, help="Input checkpoint path or HF model ID")
    parser.add_argument("--output", required=True, help="Output directory for sharded checkpoint")
    parser.add_argument("--tp-degree", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--pp-degree", type=int, default=1, help="Pipeline parallelism degree")
    parser.add_argument("--model-type", default="auto", choices=["auto", "hf", "pytorch"],
                       help="Model type (auto, hf, pytorch)")
    
    args = parser.parse_args()
    
    convert_checkpoint(
        input_path=args.input,
        output_path=args.output,
        tp_degree=args.tp_degree,
        pp_degree=args.pp_degree,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()
