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
    
    # 0. Check Hardware Info (from previous pass)
    hw_info = gm.meta.get("hardware_info", None)
    if not hw_info or hw_info["device_count"] < 2:
        print("   [CostModel] Less than 2 GPUs detected or no HW info. Skipping PP analysis.")
        return gm
        
    gpu1_info = hw_info["devices"][0]
    gpu2_info = hw_info["devices"][1] # Assuming 2 GPUs for simplicity as per pairs.json #5
    
    # Get TFLOPS (estimated in previous pass)
    tflops1 = gpu1_info.get("throughput_tflops", 10.0) * 1e12
    tflops2 = gpu2_info.get("throughput_tflops", 10.0) * 1e12

    tflops1 = gpu1_info.get("throughput_tflops", 10.0) * 1e12
    tflops2 = gpu2_info.get("throughput_tflops", 10.0) * 1e12

    # Get VRAM
    vram1 = (gpu1_info.get("memory_mb", 0) * 1024**2) * USABLE_VRAM_FRAC
    vram2 = (gpu2_info.get("memory_mb", 0) * 1024**2) * USABLE_VRAM_FRAC
    
    # Get Bandwidth (heuristic default if missing)
    link_bw = gpu1_info.get("bandwidth_gbps", 16.0) * 1024**3 * BW_EFFICIENCY

    print(f"   [CostModel] GPU1: {tflops1/1e12:.2f} TFLOPS, {vram1/1024**3:.2f} GiB", flush=True)
    print(f"   [CostModel] GPU2: {tflops2/1e12:.2f} TFLOPS, {vram2/1024**3:.2f} GiB", flush=True)
    print(f"   [CostModel] Link: {link_bw/1024**3:.2f} GiB/s", flush=True)

    # 1. Extract Config
    config = extract_model_config(gm)
    L = config["L"]
    S = config["S_max"]
    
    # 2. Evaluate all splits
    best_pp = None
    max_pp_tps = -1.0
    
    # Precompute constants
    w_layer = weight_bytes_per_layer(config)
    w_embed = embedding_bytes(config) 
    w_head = lm_head_bytes(config)
    kv_coef = kv_bytes_per_layer(1, S, config)
    
    print(f"   [CostModel] Evaluation Loop (L={L})...")
    
    for k in range(1, L): # Split at k (1 to L-1), not 0 or L (single GPU)
        layers1 = k
        layers2 = L - k
        
        # Memory Check
        # GPU1: layers 0..k-1 + Embed
        mem1_fixed = (layers1 * w_layer) + w_embed
        rem_mem1 = vram1 - mem1_fixed
        if rem_mem1 <= 0: continue
        b_cap1 = int(rem_mem1 / (layers1 * kv_coef))
        
        # GPU2: layers k..L-1 + Head
        mem2_fixed = (layers2 * w_layer) + w_head
        rem_mem2 = vram2 - mem2_fixed
        if rem_mem2 <= 0: continue
        b_cap2 = int(rem_mem2 / (layers2 * kv_coef))
        
        b_capacity = min(b_cap1, b_cap2)
        if b_capacity < 1: continue
        
        # Throughput Search
        local_best_tps = -1
        local_best_b = 0
        
        # Optimization: Don't search every B, just search logarithmic + near capacity? 
        # Or just search reasonable range.
        search_space = list(range(1, 33)) + [b_capacity] 
        search_space = [b for b in search_space if b <= b_capacity]
        search_space = sorted(list(set(search_space)))

        for b_curr in search_space:
             eta1 = get_eta(b_curr) # Using default eta table for now
             eta2 = get_eta(b_curr)
             
             flops_1token = flops_per_layer_decode(b_curr, S, config)
             
             t1 = (layers1 * flops_1token) / (tflops1 * eta1)
             t2 = (layers2 * flops_1token) / (tflops2 * eta2)
             
             comm = pp_comm_time(b_curr, config["d_model"], link_bw, config["precision_bytes"])
             
             latency = max(t1+comm, t2+comm)
             tps = b_curr / latency
             
             if tps > local_best_tps:
                 local_best_tps = tps
                 local_best_b = b_curr
        
        if local_best_tps > max_pp_tps:
            max_pp_tps = local_best_tps
            best_pp = {
                "split_k": k,
                "tps": local_best_tps,
                "b_opt": local_best_b
            }

    if best_pp:
        k = best_pp["split_k"]
        msg = f"   [CostModel] Optimal Split Found: Layer {k} (TPS: {best_pp['tps']:.2f}, Batch: {best_pp['b_opt']})\n"
        print(msg, flush=True)
        # 3. Write Split to Metadata (No Graph Modification Here)
        gm.meta['split_config'] = {
            "split_k": k, # Layer Index (0-indexed layer to split BEFORE)
            "tps": best_pp['tps'],
            "b_opt": best_pp['b_opt']
        }
        print(f"   [CostModel] Split Decision Recorded in Metadata: Split before Layer {k}")

    else:
        print("   [CostModel] No feasible PP split found or Single GPU is better (not implemented check).")

    return gm
