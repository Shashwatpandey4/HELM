#!/usr/bin/env python3
"""
Distributed launcher for HELM.

Wraps torchrun to launch multi-process distributed execution.
"""

import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Launch HELM in distributed mode")
    parser.add_argument("script", help="Python script to run")
    parser.add_argument("--nproc-per-node", type=int, default=None,
                       help="Number of processes per node (default: num GPUs)")
    parser.add_argument("--nnodes", type=int, default=1,
                       help="Number of nodes")
    parser.add_argument("--node-rank", type=int, default=0,
                       help="Rank of this node")
    parser.add_argument("--master-addr", default="localhost",
                       help="Master node address")
    parser.add_argument("--master-port", default="29500",
                       help="Master node port")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                       help="Arguments to pass to script")
    
    args = parser.parse_args()
    
    # Determine nproc_per_node
    if args.nproc_per_node is None:
        import torch
        if torch.cuda.is_available():
            args.nproc_per_node = torch.cuda.device_count()
        else:
            args.nproc_per_node = 1
    
    print(f"[HELM Launcher] Starting distributed execution:")
    print(f"  Script: {args.script}")
    print(f"  Processes per node: {args.nproc_per_node}")
    print(f"  Nodes: {args.nnodes}")
    print(f"  Master: {args.master_addr}:{args.master_port}")
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node-rank={args.node_rank}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port}",
        args.script
    ] + args.script_args
    
    print(f"\n[HELM Launcher] Running: {' '.join(cmd)}\n")
    
    # Execute
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
