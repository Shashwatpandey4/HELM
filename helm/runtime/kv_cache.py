import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
import math

@dataclass
class KVPage:
    page_id: int
    layer_id: int
    start_token: int
    used_tokens: int
    capacity_tokens: int
    k_tensor: torch.Tensor
    v_tensor: torch.Tensor
    device: torch.device
    state: str  # 'FREE', 'GPU', 'CPU'

class LayerKVCache:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.pages: List[KVPage] = []
        self.tail_page: Optional[KVPage] = None
        self.total_tokens: int = 0

    def append_page(self, page: KVPage):
        self.pages.append(page)
        self.tail_page = page

class KVCacheManager:
    """
    KVCacheManager owns all layer KV state.
    Implements append semantics, residency tracking, and high-level 
    prefill/decode integration.
    """
    def __init__(self, allocator, gpu_high_watermark_bytes: int = 0.25 * 1024 * 1024):
        self.allocator = allocator
        self.layers = {}
        self.gpu_high_watermark_bytes = gpu_high_watermark_bytes

    def initialize_layer_caches(self, num_layers: int):
        for i in range(num_layers):
            self.layers[i] = LayerKVCache(layer_id=i)

    def append_prefill(self, layer_id: int, K_prefill: torch.Tensor, V_prefill: torch.Tensor):
        """Append prefill KV tensor chunk to the layer cache in page-sized blocks."""
        if layer_id not in self.layers:
            self.layers[layer_id] = LayerKVCache(layer_id)
        
        layer = self.layers[layer_id]
        seq_len = K_prefill.size(2)
        page_size = self.allocator.page_size
        device = K_prefill.device
        
        for start_idx in range(0, seq_len, page_size):
            chunk_len = min(page_size, seq_len - start_idx)
            end_idx = start_idx + chunk_len

            k_chunk = K_prefill[:, :, start_idx:end_idx, :]
            v_chunk = V_prefill[:, :, start_idx:end_idx, :]

            page = self.allocator.allocate(device)
            page.layer_id = layer_id
            page.start_token = layer.total_tokens
            page.used_tokens = chunk_len

            page.k_tensor[:, :, :chunk_len, :].copy_(k_chunk)
            page.v_tensor[:, :, :chunk_len, :].copy_(v_chunk)

            layer.append_page(page)
            layer.total_tokens += chunk_len

            # Enforce after every page so GPU KV never spikes beyond watermark
            # even during long prefills (previously only ran once at end of layer).
            self._enforce_residency_policy()

    def append_decode(self, layer_id: int, K_new: torch.Tensor, V_new: torch.Tensor):
        """Append a single token KV pair for the given layer."""
        if layer_id not in self.layers:
            self.layers[layer_id] = LayerKVCache(layer_id)
            
        layer = self.layers[layer_id]
        device = K_new.device
        
        if layer.tail_page is None or layer.tail_page.used_tokens == layer.tail_page.capacity_tokens:
            page = self.allocator.allocate(device)
            page.layer_id = layer_id
            page.start_token = layer.total_tokens
            page.used_tokens = 0
            layer.append_page(page)
            
        page = layer.tail_page
        idx = page.used_tokens
        
        page.k_tensor[:, :, idx:idx+1, :].copy_(K_new)
        page.v_tensor[:, :, idx:idx+1, :].copy_(V_new)
        
        page.used_tokens += 1
        layer.total_tokens += 1
        
        self._enforce_residency_policy()

    def iterate_layer_pages(self, layer_id: int):
        if layer_id not in self.layers:
            return []
        # Return a shallow copy of the list so it isn't mutated during iteration
        return list(self.layers[layer_id].pages)

    def evict_pages(self, bytes_to_evict: int):
        """Global oldest page first eviction."""
        if bytes_to_evict <= 0:
            return

        gpu_pages = []
        for layer in self.layers.values():
            for page in layer.pages:
                if page.state == 'GPU' and page.device.type == 'cuda':
                    # Protect tail page
                    if page != layer.tail_page:
                        # Only evict fully written pages
                        if page.used_tokens == page.capacity_tokens:
                            gpu_pages.append((layer, page))
                            
        # Sort by oldest allocation globally (using start_token instead of page_id for chronological age)
        gpu_pages.sort(key=lambda t: t[1].start_token)
        
        evicted_bytes = 0
        for layer, page in gpu_pages:
            if evicted_bytes >= bytes_to_evict:
                break
            
            # move_page returns a newly allocated CPU page with copied data
            original_start_token = page.start_token
            new_page = self.allocator.move_page(page, torch.device('cpu'))
            
            # Ensure ordering invariants are protected
            assert new_page.start_token == original_start_token, "Page invariant broken during migration"
            
            # Replace old page with new page in the layer's list
            idx = layer.pages.index(page)
            layer.pages[idx] = new_page
            
            # Keep tail page reference correct if it happened to be the one (though protected above)
            if layer.tail_page is page:
                layer.tail_page = new_page
                
            evicted_bytes += new_page.k_tensor.numel() * new_page.k_tensor.element_size() * 2

    def _enforce_residency_policy(self):
        # Note: We only record bytes for pages that are fully written and not the active tail.
        # Thus, the watermark strictly applies to evictable pages, not total GPU KV footprint.
        gpu_bytes = 0
        for layer in self.layers.values():
            for page in layer.pages:
                if page.state == 'GPU' and page.device.type == 'cuda':
                    if page != layer.tail_page and page.used_tokens == page.capacity_tokens:
                        gpu_bytes += page.k_tensor.numel() * page.k_tensor.element_size() * 2
        
        if gpu_bytes > self.gpu_high_watermark_bytes:
            self.evict_pages(gpu_bytes - self.gpu_high_watermark_bytes)

    def seq_len(self):
        """Return the current cached sequence length (from layer 0)."""
        if 0 in self.layers:
            return self.layers[0].total_tokens
        return 0

    def num_layers(self):
        return len(self.layers)

    def clear(self):
        for layer in self.layers.values():
            for page in layer.pages:
                # DEAD pages had their tensors replaced with placeholders and
                # were never added to any pool — skip them here.
                if page.state != 'DEAD':
                    self.allocator.free(page)
            layer.pages.clear()
            layer.tail_page = None
            layer.total_tokens = 0
        self.layers.clear()

    def layers_keys(self):
        return sorted(self.layers.keys())

    def report_bytes_per_tier(self) -> dict:
        gpu_bytes = 0
        cpu_bytes = 0
        for layer in self.layers.values():
            for page in layer.pages:
                if page.state == 'DEAD':
                    continue
                page_bytes = page.k_tensor.numel() * page.k_tensor.element_size() * 2
                if page.state == 'GPU':
                    gpu_bytes += page_bytes
                elif page.state == 'CPU':
                    cpu_bytes += page_bytes
                    
        return {
            "gpu_active_bytes": gpu_bytes,
            "cpu_active_bytes": cpu_bytes
        }
        
        
# Persistent copy stream for async CPU→GPU KV page prefetch.
# Lazily initialised on first call to perform_streaming_attention that needs it.
_kv_copy_stream: Optional[torch.cuda.Stream] = None


def perform_streaming_attention(query: torch.Tensor, pages: List[KVPage], scale: float = None):
    """
    Runtime attention helper that streams over KV pages using online softmax.

    Four execution paths depending on page residency:

    1. All GPU  — single F.scaled_dot_product_attention call (flash-attention
                  backend on CUDA; no Python loop, TC-enabled fp16 kernels).
    2. All CPU  — CPU-stage layers: query and all pages on CPU.  Concatenate
                  all pages and issue a single SDPA call; avoids the per-page
                  Python loop and repeated small-tensor allocations.
    3. Mixed    — GPU-stage layers with some pages evicted to CPU.  GPU pages
                  are batched into one matmul.  CPU pages are prefetched
                  asynchronously: all H2D copies are submitted upfront on a
                  dedicated copy stream with per-page CUDA events; the compute
                  stream waits for each event individually, overlapping the
                  H2D transfer of page N+1 with the matmul for page N.
    4. Mixed fallback — same as 3 but without CUDA (should never fire in
                  normal operation since CPU pages with a CUDA query only arise
                  from GPU-stage attention).
    """
    global _kv_copy_stream
    device = query.device
    num_q_heads = query.size(1)

    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    dtype = query.dtype

    # Partition active pages by device.  Exclude DEAD pages (evicted GPU pages
    # whose tensors have been replaced with empty placeholders).
    gpu_pages = [p for p in pages if p.used_tokens > 0 and p.state == 'GPU' and p.device.type == "cuda"]
    cpu_pages = [p for p in pages if p.used_tokens > 0 and p.state == 'CPU' and p.device.type != "cuda"]

    if not gpu_pages and not cpu_pages:
        return torch.zeros_like(query)

    def _expand_kv(k: torch.Tensor, v: torch.Tensor):
        """Expand KV heads to match Q heads for grouped-query attention."""
        num_kv = k.size(1)
        if num_kv != num_q_heads:
            groups = num_q_heads // num_kv
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        return k, v

    # ── Path 1: all pages GPU-resident ───────────────────────────────────────
    # Single SDPA call — flash-attention backend, TC-enabled fp16, no loop.
    if not cpu_pages:
        K = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2)
        V = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2)
        K, V = _expand_kv(K, V)
        return F.scaled_dot_product_attention(query, K, V, scale=scale)

    # ── Path 2: all pages CPU-resident, query on CPU ─────────────────────────
    # CPU-stage layers: no PCIe involved.  Concatenate all pages and call SDPA
    # once — avoids the N-iteration Python loop and per-page tensor allocations.
    if not gpu_pages and device.type != "cuda":
        K = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in cpu_pages], dim=2).float()
        V = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in cpu_pages], dim=2).float()
        K, V = _expand_kv(K, V)
        return F.scaled_dot_product_attention(query.float(), K, V, scale=scale).to(dtype)

    # ── Path 3: mixed residency, CUDA query (GPU-stage with evicted pages) ────
    # Submit all CPU→GPU copies upfront on a dedicated copy stream.  Record a
    # per-page CUDA event after each copy+cast.  The compute stream then waits
    # on each event in order, so it only stalls until *that page* is ready —
    # the H2D transfer for page N+1 runs concurrently with the matmul for page N.
    q32 = query.float()
    bsz, _, q_len, _ = query.shape
    out         = torch.zeros(bsz, num_q_heads, q_len, query.size(-1),
                              device=device, dtype=torch.float32)
    running_max = torch.full((bsz, num_q_heads, q_len, 1), -float("inf"),
                             device=device, dtype=torch.float32)
    running_sum = torch.zeros((bsz, num_q_heads, q_len, 1),
                              device=device, dtype=torch.float32)

    # GPU pages: one fused matmul
    if gpu_pages:
        K_gpu = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2).float()
        V_gpu = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in gpu_pages], dim=2).float()
        K_gpu, V_gpu = _expand_kv(K_gpu, V_gpu)

        scores    = torch.matmul(q32, K_gpu.transpose(-2, -1)) * scale
        block_max = scores.amax(dim=-1, keepdim=True)
        new_max   = torch.maximum(running_max, block_max)
        old_scale = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)
        block_exp = torch.exp(scores - new_max)

        out         = out * old_scale + torch.matmul(block_exp, V_gpu)
        running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
        running_max = new_max

    # Async prefetch all CPU pages to GPU, recording a per-page event.
    if _kv_copy_stream is None:
        _kv_copy_stream = torch.cuda.Stream(device=device)

    kv_bufs: List[tuple] = []
    for page in cpu_pages:
        with torch.cuda.stream(_kv_copy_stream):
            k_buf = page.k_tensor[:, :, :page.used_tokens, :].to(device, non_blocking=True).float()
            v_buf = page.v_tensor[:, :, :page.used_tokens, :].to(device, non_blocking=True).float()
            ev = torch.cuda.Event()
            ev.record()
        kv_bufs.append((k_buf, v_buf, ev))

    # Process each page: wait only for that page's event, so H2D for page N+1
    # overlaps with the matmul for page N.
    for k_buf, v_buf, ev in kv_bufs:
        torch.cuda.current_stream().wait_event(ev)
        k, v = _expand_kv(k_buf, v_buf)

        scores    = torch.matmul(q32, k.transpose(-2, -1)) * scale
        block_max = scores.amax(dim=-1, keepdim=True)
        new_max   = torch.maximum(running_max, block_max)
        old_scale = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)
        block_exp = torch.exp(scores - new_max)

        out         = out * old_scale + torch.matmul(block_exp, v)
        running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
        running_max = new_max

    out = out / torch.clamp(running_sum, min=1e-12)
    return out.to(dtype)
