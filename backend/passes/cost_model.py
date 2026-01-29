import torch
import torch.fx as fx
import re
import math
import copy

# --- Ported Constants & Functions from User Script ---

# We will try to deduce these from the graph, but keep defaults for fallback
DEFAULT_MODEL_CONFIG = {
    "name": "Generic-LLM",
    "L": 32,
    "d_model": 4096,
    "intermediate": 11008,
    "vocab": 32000,
    "S_max": 2048,
    "precision_bytes": 2, # FP16
    "n_heads": 32,
    "n_kv_heads": 32,
    "head_dim": 128
}

USABLE_VRAM_FRAC = 0.9
KAPPA_GLOBAL = 0.45   # Calibration factor
BW_EFFICIENCY = 0.85  # PCIe efficiency
LINK_LATENCY_S = 10e-6

ETA_TABLE_DEFAULT = {
  1: 0.55, 2: 0.65, 4: 0.75, 8: 0.85, 16: 0.92, 32: 0.95
}

def get_eta(b):
    bucket = 1
    for k in sorted(ETA_TABLE_DEFAULT.keys()):
        if b >= k: bucket = k
        else: break
    return ETA_TABLE_DEFAULT[bucket]

def kv_bytes_per_layer(B, S, model_config):
    n_kv = model_config.get("n_kv_heads", 32)
    h_dim = model_config.get("head_dim", 128)
    p = model_config["precision_bytes"]
    return 2 * B * S * n_kv * h_dim * p

def weight_bytes_per_layer(model_config):
    d = model_config["d_model"]
    i = model_config["intermediate"]
    p = model_config["precision_bytes"]
    # 4*d^2 (attn) + 3*d*i (mlp) + 2*d (norm)
    params = (4 * d * d) + (3 * d * i) + (2 * d)
    return params * p

def embedding_bytes(model_config):
    return model_config["vocab"] * model_config["d_model"] * model_config["precision_bytes"]

def lm_head_bytes(model_config):
    return model_config["d_model"] * model_config["vocab"] * model_config["precision_bytes"]

def flops_per_layer_decode(B, S, model_config): 
    d = model_config["d_model"]
    i = model_config["intermediate"]
    flops_mlp = 6 * B * d * i
    flops_attn_proj = 8 * B * d * d
    flops_attn_op = 4 * B * S * d
    return flops_mlp + flops_attn_proj + flops_attn_op

def pp_comm_time(B, d, link_bw, precision_bytes):
    if link_bw == 0: return float('inf')
    bytes_to_transfer = B * d * precision_bytes
    transfer_time = bytes_to_transfer / link_bw
    return transfer_time + LINK_LATENCY_S

# --- Graph Extraction Helper ---

def extract_model_config(gm: fx.GraphModule):
    """
    Generic extraction of model config (d_model, L, etc.) from the graph.
    Detects the main layer stack by identifying the longest sequence of numerically indexed modules.
    """
    config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    
    # 1. Check Metadata first (from Soft Analysis)
    if 'model_config' in gm.meta:
        print("   [CostModel] Using Deducted Metadata Config.")
        # Merge with default to ensure all keys exist
        for k, v in gm.meta['model_config'].items():
            config[k] = v
        return config

    # 2. Detect Layer Stack Pattern
    # We look for patterns like "layers.0", "h.0", "blocks.0", "encoder.layer.0"
    # Strategy: Group modules by "parent path" and count how many indexed children they have.
    
    parent_counts = {} # "parent_name" -> set of indices
    
    for name, _ in gm.named_modules():
        parts = name.split('.')
        # Check if last part is integer (index)
        if parts[-1].isdigit():
            idx = int(parts[-1])
            parent = ".".join(parts[:-1])
            if parent not in parent_counts:
                parent_counts[parent] = set()
            parent_counts[parent].add(idx)
            
    # Find the parent with the most indices -> likely the layer stack
    best_parent = None
    max_layers = 0
    
    for parent, indices in parent_counts.items():
        count = len(indices)
        if count > max_layers:
            max_layers = count
            best_parent = parent
            
    if best_parent:
        config["L"] = max_layers
        config["layer_prefix"] = best_parent # Store for later use
        print(f"   [CostModel] Detected {max_layers} layers (Prefix: '{best_parent}').")
    else:
        # Fallback for flattened graphs where modules are gone?
        # Or try to parse node targets?
        # print(f"   [CostModel] WARNING: Could not detect layer stack. Using default L={config['L']}.")
        config["layer_prefix"] = "layers" # Default

    # 2. Estimate d_model / intermediate from first layer
    # Look inside {best_parent}.0
    try:
        layer0_prefix = f"{config['layer_prefix']}.0"
        
        for name, mod in gm.named_modules():
            if name.startswith(layer0_prefix):
                # Heuristics for d_model (Attention Input/Output)
                # Look for Linear layers with common names
                if hasattr(mod, 'in_features'):
                    # q_proj, k_proj, v_proj usually take d_model
                    if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'c_attn']):
                        config["d_model"] = mod.in_features
                        
                    # MLP expansion usually outputs intermediate
                    if any(x in name for x in ['up_proj', 'gate_proj', 'fc1', 'c_fc']):
                        config["intermediate"] = mod.out_features
                        
    except Exception as e:
        print(f"   [CostModel] Warning during config extraction: {e}")
        
    print(f"   [CostModel] Extracted Config: d_model={config['d_model']}, intermediate={config['intermediate']}")
    return config

# --- The Main Pass ---

def cost_model_pass(gm: fx.GraphModule, world_size: int = None):
    print("\n>> [Pass] Running Analytical Cost Model (User Provided)...")
    
    # 0. Hardware Analysis Info
    hw_info = gm.meta.get("hardware_info", None)
    if not hw_info:
        print("   [CostModel] No HW info. Skipping PP analysis.")
        return gm
    
    # Use provided world_size or detected count
    effective_world_size = world_size if world_size else hw_info["device_count"]
    if effective_world_size < 2:
        print(f"   [CostModel] World size {effective_world_size} < 2. Skipping PP analysis.")
        return gm

    # We assume uniform GPUs for the N-way split heuristic
    gpu_props = hw_info["devices"][0]
    tflops = gpu_props.get("throughput_tflops", 10.0) * 1e12 * KAPPA_GLOBAL
    vram = (gpu_props.get("memory_mb", 0) * 1024**2) * USABLE_VRAM_FRAC
    link_bw = gpu_props.get("bandwidth_gbps", 16.0) * 1024**3 * BW_EFFICIENCY

    print(f"   [CostModel] Hardware (Uniform Assumption): {tflops/1e12:.2f} TFLOPS, {vram/1024**3:.2f} GiB, {link_bw/1024**3:.2f} GiB/s", flush=True)

    # 1. Extract Config
    config = extract_model_config(gm)
    L = config["L"]
    S = config["S_max"]

    # Precompute constants
    w_layer = weight_bytes_per_layer(config)
    w_embed = embedding_bytes(config) 
    w_head = lm_head_bytes(config)
    kv_coef = kv_bytes_per_layer(1, S, config)

    # 2. Evaluate Strategies
    print(f"   [CostModel] Evaluating L={L} for World Size={effective_world_size}...")
    
    # Baseline: Single-GPU
    # GPU 0 takes all
    fixed_mem_single = L * w_layer + w_embed + w_head
    b_limit = int((vram - fixed_mem_single) / (L * kv_coef)) if vram > fixed_mem_single else 0
    tps_single = 0
    b_single = 0
    if b_limit >= 1:
        # Optimizing batch for single GPU
        for b in sorted(list(set(range(1, 33)).union({b_limit}))):
            if b > b_limit: continue
            lat = (L * flops_per_layer_decode(b, S, config)) / (tflops * get_eta(b))
            tps = b / lat
            if tps > tps_single:
                tps_single = tps
                b_single = b

    # Sharding: Split into N stages
    # Heuristic: Balanced Split L / N
    layers_per_rank = L / effective_world_size
    fixed_mem_pp = (layers_per_rank * w_layer) + max(w_embed, w_head) # conservative
    b_limit_pp = int((vram - fixed_mem_pp) / (layers_per_rank * kv_coef)) if vram > fixed_mem_pp else 0
    tps_pp = 0
    b_pp = 0
    if b_limit_pp >= 1:
        for b in sorted(list(set(range(1, 33)).union({b_limit_pp}))):
            if b > b_limit_pp: continue
            # Latency for sequential execution: sum of compute + (N-1) skips
            compute_lat = (L * flops_per_layer_decode(b, S, config)) / (tflops * get_eta(b))
            comm_lat = (effective_world_size - 1) * pp_comm_time(b, config["d_model"], link_bw, config["precision_bytes"])
            lat = compute_lat + comm_lat
            tps = b / lat
            if tps > tps_pp:
                tps_pp = tps
                b_pp = b

    print(f"   [CostModel] Candidate TPS: Single-GPU={tps_single:.2f}, {effective_world_size}-way PP={tps_pp:.2f}")

    # Decision
    if tps_single >= tps_pp and tps_single > 0:
        print(f"   [CostModel] WINNER: Single-GPU (Staying local to avoid communication overhead)")
        split_k = L # Rank 0 takes all
        final_tps = tps_single
        final_b = b_single
    elif tps_pp > 0:
        # Basic Balanced Split
        split_k = int(L / effective_world_size)
        print(f"   [CostModel] WINNER: {effective_world_size}-way PP (Sharding required or throughput advantageous)")
        final_tps = tps_pp
        final_b = b_pp
    else:
        print("   [CostModel] No feasible strategy found (OOM at Batch 1).")
        return gm

    # Record Decision
    gm.meta['split_config'] = {
        "split_k": split_k,
        "tps": final_tps,
        "b_opt": final_b
    }


    return gm
