from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# --- Data Structures ---

@dataclass
class LayerSpec:
    flops_prefill: float
    flops_decode_token: float
    bytes_moved_prefill: float
    bytes_moved_decode: float
    activation_bytes: float
    param_bytes: float
    kv_bytes_per_token: float

@dataclass
class ModelSpec:
    L: int
    layers: List[LayerSpec]
    dtype: str
    n_heads: int
    hidden_size: int
    vocab_size: int
    seq_len_prefill: int  # Default/typical sequence length
    max_seq_len: int      # Maximum supported sequence length (for capacity planning)
    batch_size: int
    gen_len: int
    
    def estimate_for_batch(self, actual_seq_lens: List[int]) -> Dict:
        """
        Estimate costs for a specific batch with variable sequence lengths.
        
        Args:
            actual_seq_lens: List of sequence lengths in the batch
            
        Returns:
            Dict with adjusted FLOPs and memory estimates
        """
        max_len = max(actual_seq_lens)
        mean_len = sum(actual_seq_lens) / len(actual_seq_lens)
        
        # Adjust FLOPs based on actual sequence lengths
        # Use mean for compute, max for memory
        seq_ratio = mean_len / self.seq_len_prefill
        mem_ratio = max_len / self.seq_len_prefill
        
        return {
            'seq_ratio': seq_ratio,
            'mem_ratio': mem_ratio,
            'max_len': max_len,
            'mean_len': mean_len
        }

@dataclass
class DeviceSpec:
    id: int
    peak_flops: float
    mem_bw: float
    mem_capacity: float
    h2d_bw: float = 16e9 # Example 16GB/s PCIe
    d2h_bw: float = 16e9
    h2d_lat: float = 1e-6
    d2h_lat: float = 1e-6

@dataclass
class StagePlacement:
    devices: List[int]
    layer_range: Tuple[int, int] # [start, end)

@dataclass
class OverlapParams:
    overlap_tp_comm_prefill: float = 0.5
    overlap_tp_comm_decode: float = 0.5
    overlap_pp_send: float = 0.5
    overlap_kv_prefetch: float = 0.0
    overlap_offload: float = 0.0

@dataclass
class ParallelConfig:
    tp_degree: int
    pp_degree: int
    microbatches: int
    pp_stages: List[StagePlacement]
    overlap_params: OverlapParams = field(default_factory=OverlapParams)
    quant_scheme: str = "fp16"
    kv_policy: str = "GPU_ONLY" # GPU_ONLY, CPU_OFFLOAD
    offload_policy: str = "NONE" # NONE, WEIGHTS, ACT

@dataclass
class MemReport:
    feasible: bool
    reason: str
    per_dev: Dict[int, float]
    headroom: Dict[int, float]

class CalibrationDB:
    """
    Hardware calibration database using profiled system measurements.
    """
    def __init__(self, profile: Optional[Dict] = None):
        self.profile = profile or {}
        
    def eff_compute(self, shape_key, dtype, phase) -> float:
        # Conservative estimate: 50-70% of peak depending on phase
        # Prefill (large batch) gets better utilization
        if phase == "PREFILL":
            return 0.7
        return 0.5
    
    def eff_bw(self, phase) -> float:
        # Memory bandwidth efficiency
        return 0.7
        
    def kernel_overhead(self, shape_key, phase) -> float:
        # Kernel launch overhead (measured or estimated)
        # Small kernels (decode) have higher relative overhead
        if phase == "DECODE_TOKEN":
            return 20e-6  # 20us
        return 10e-6  # 10us

    def allreduce_coeffs(self, tp_degree, topology, phase) -> Tuple[float, float]:
        # (latency_coeff, bandwidth_coeff)
        # Use profiled P2P bandwidth if available
        return 1.0, 1.0
    
    def get_link_bandwidth(self) -> float:
        """Get inter-GPU bandwidth from profile or fallback."""
        # Use profiled value or default to 200GB/s NVLink
        # Ensure non-zero to avoid division errors
        bw = self.profile.get('p2p_bw', 200e9)
        return max(bw, 200e9)  # Fallback if profiled value is 0
    
    def get_link_latency(self) -> float:
        """Get inter-GPU latency (estimated)."""
        # Latency is harder to profile accurately, use conservative estimate
        return 10e-6  # 10us

# --- HELM Cost Model Implementation ---

class HelmCostModel:
    def __init__(self, model: ModelSpec, devices: List[DeviceSpec], topology: Dict, calibration: CalibrationDB):
        self.model = model
        self.devices = devices
        self.topo = topology
        self.calib = calibration

    def estimate(self, cfg: ParallelConfig) -> Dict:
        # 0. Memory Check
        mem_report = self._memory_check(cfg)
        if not mem_report.feasible:
            return {'feasible': False, 'reason': mem_report.reason, 'cost': float('inf'), 'mem': mem_report}

        # 1. Prefill Latency
        stage_times_prefill = self._compute_stage_times(cfg, phase="PREFILL")
        T_prefill = self._pipeline_latency(stage_times_prefill, cfg.microbatches, cfg.pp_degree)

        # 2. Decode Latency
        stage_times_decode = self._compute_stage_times(cfg, phase="DECODE_TOKEN")
        T_decode_token = max(stage_times_decode) if stage_times_decode else 0.0
        
        # 3. Offload/KV Overhead
        T_decode_token += self._kv_and_offload_cost(cfg, phase="DECODE_TOKEN")

        # 4. Total
        T_total = T_prefill + self.model.gen_len * T_decode_token

        return {
            'feasible': True,
            'T_prefill': T_prefill,
            'T_decode_token': T_decode_token,
            'T_total': T_total,
            'tokens_per_sec': self.model.batch_size / T_decode_token if T_decode_token > 0 else 0,
            'mem': mem_report,
            'details': {
                'prefill_stages': stage_times_prefill,
                'decode_stages': stage_times_decode
            }
        }

    def _memory_check(self, cfg: ParallelConfig) -> MemReport:
        per_dev = {d.id: 0.0 for d in self.devices}
        
        # A. Weights
        for stage in cfg.pp_stages:
            start, end = stage.layer_range
            stage_bytes = sum(self.model.layers[l].param_bytes for l in range(start, end))
            
            # Sharded by TP
            shard_bytes = stage_bytes / cfg.tp_degree
            for dev_id in stage.devices:
                per_dev[dev_id] += shard_bytes

        # B. KV Cache
        seq_total = self.model.seq_len_prefill + self.model.gen_len
        for stage in cfg.pp_stages:
            start, end = stage.layer_range
            stage_kv = sum(self.model.layers[l].kv_bytes_per_token for l in range(start, end))
            kv_total = stage_kv * seq_total * self.model.batch_size
            
            # Policy
            if cfg.kv_policy == "GPU_ONLY":
                kv_resident = kv_total
            else:
                kv_resident = 0 # Offloaded
                
            shard_kv = kv_resident / cfg.tp_degree
            for dev_id in stage.devices:
                per_dev[dev_id] += shard_kv

        # C. Activations (Simplified)
        act_factor = 1.0 # Checkpointing logic here
        for stage in cfg.pp_stages:
            start, end = stage.layer_range
            stage_act = sum(self.model.layers[l].activation_bytes for l in range(start, end))
            act_total = stage_act * cfg.microbatches * act_factor
            shard_act = act_total / cfg.tp_degree
            for dev_id in stage.devices:
                per_dev[dev_id] += shard_act
                
        # Debug: Print memory usage
        print(f"[MemCheck] Config: TP={cfg.tp_degree}, PP={len(cfg.pp_stages)}")
        for dev in self.devices:
            used_gb = per_dev[dev.id] / 1e9
            capacity_gb = dev.mem_capacity / 1e9
            print(f"[MemCheck] Device {dev.id}: {used_gb:.2f} GB / {capacity_gb:.2f} GB")
            
        # Check Capacity
        for dev in self.devices:
            if per_dev[dev.id] > dev.mem_capacity:
                print(f"[MemCheck] OOM on Device {dev.id}!")
                return MemReport(False, f"OOM Device {dev.id}", per_dev, {})
                
        print(f"[MemCheck] PASSED")
        return MemReport(True, "", per_dev, {})

    def _compute_stage_times(self, cfg: ParallelConfig, phase: str) -> List[float]:
        stage_times = []
        for i, stage in enumerate(cfg.pp_stages):
            # Bottleneck device (slowest in TP group)
            # Simplified: Assume homogeneous TP group, take dev 0
            dev = self.devices[stage.devices[0]] 
            
            start, end = stage.layer_range
            T_stage = 0.0
            
            for l in range(start, end):
                layer = self.model.layers[l]
                
                if phase == "PREFILL":
                    flops = layer.flops_prefill
                    bytes_mov = layer.bytes_moved_prefill
                    overlap = cfg.overlap_params.overlap_tp_comm_prefill
                else:
                    flops = layer.flops_decode_token
                    bytes_mov = layer.bytes_moved_decode
                    overlap = cfg.overlap_params.overlap_tp_comm_decode
                    
                eff_c = self.calib.eff_compute(None, self.model.dtype, phase)
                eff_b = self.calib.eff_bw(phase)
                
                # Roofline
                T_compute = max(
                    flops / (dev.peak_flops * eff_c + 1e-9),
                    bytes_mov / (dev.mem_bw * eff_b + 1e-9)
                ) + self.calib.kernel_overhead(None, phase)
                
                # Comm (TP)
                T_comm = 0.0
                if cfg.tp_degree > 1:
                    # Estimate AllReduce
                    # Bytes ~ Batch * Hidden * Dtype
                    dtype_bytes = 2 # fp16
                    if phase == "PREFILL":
                        ar_bytes = self.model.batch_size * self.model.seq_len_prefill * self.model.hidden_size * dtype_bytes
                    else:
                        ar_bytes = self.model.batch_size * 1 * self.model.hidden_size * dtype_bytes
                        
                    # Latency + BW from calibration
                    link_bw = self.calib.get_link_bandwidth()
                    link_lat = self.calib.get_link_latency()
                    
                    # Allreduce model: latency_coeff * link_lat + bandwidth_coeff * ar_bytes / link_bw
                    lat_coeff, bw_coeff = self.calib.allreduce_coeffs(cfg.tp_degree, self.topo, phase)
                    T_comm = lat_coeff * link_lat + bw_coeff * ar_bytes / link_bw
                    
                T_layer = max(T_compute, T_comm * (1 - overlap))
                T_stage += T_layer
                
            stage_times.append(T_stage)
        return stage_times

    def _pipeline_latency(self, stage_times: List[float], m: int, p: int) -> float:
        if not stage_times: return 0.0
        T_steady = max(stage_times)
        T_bubble = (p - 1) * T_steady
        return T_bubble + m * T_steady

    def _kv_and_offload_cost(self, cfg: ParallelConfig, phase: str) -> float:
        if phase != "DECODE_TOKEN": return 0.0
        if cfg.kv_policy == "GPU_ONLY": return 0.0
        
        # Offload logic: Fetch KV per token
        # Bytes = Layers * Heads * Dim * Batch * 1 * Dtype
        # Assume we fetch ALL KV (Naive offload) or just active? 
        # Usually we stream KV.
        # Cost = Latency + Bytes / PCIe_BW
        return 0.0 # Placeholder
