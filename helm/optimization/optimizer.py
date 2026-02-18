from typing import List, Dict, Tuple, Optional
import itertools
from helm.optimization.cost_model import HelmCostModel, ParallelConfig, StagePlacement, ModelSpec, DeviceSpec, CalibrationDB

class ParallelOptimizer:
    """
    Finds the optimal parallelization strategy (PP, TP, Microbatches) for a given model and hardware.
    Implements a 3-level search:
    1. Macro Search (Degrees of Parallelism)
    2. TP Group Enumeration
    3. PP Partitioning (Beam Search)
    """
    def __init__(self, model: ModelSpec, devices: List[DeviceSpec], topology: Dict = None, calibration: CalibrationDB = None):
        self.model = model
        self.devices = devices
        # Default empty topology/calib if None
        self.topo = topology if topology else {}
        self.calib = calibration if calibration else CalibrationDB()
        self.cost_model = HelmCostModel(model, devices, self.topo, self.calib)

    def optimize(self) -> Optional[ParallelConfig]:
        """
        Simplified optimizer: Only search Pipeline Parallelism (PP).
        TP=1 (no tensor parallelism), MB=1 (no microbatching).
        
        Use beam search to find optimal layer splits for each PP degree.
        """
        print("[ParallelOptimizer] Starting PP-Only Optimization...")
        
        num_devs = len(self.devices)
        
        best_cfg = None
        min_cost = float('inf')
        
        # Search PP degrees: 1, 2, 4, 8, ... up to num_devs
        possible_pp = [1, 2, 4, 8, 16, 32]
        possible_pp = [pp for pp in possible_pp if pp <= num_devs]
        
        print(f"[ParallelOptimizer] Searching PP degrees: {possible_pp}")
        
        for pp in possible_pp:
            # Fixed: TP=1, MB=1
            tp = 1
            mb = 1
            
            # Form device groups (one device per stage for PP)
            # Use actual device IDs from self.devices
            if pp > len(self.devices):
                continue  # Can't have more stages than devices
            
            # Select first pp devices (prioritize GPUs, then CPU)
            selected_devices = [dev.id for dev in self.devices[:pp]]
            groups = [[dev_id] for dev_id in selected_devices]
            
            print(f"  [PP={pp}] Device groups: {groups}")
            
            # Beam search for optimal layer partitioning
            pp_stages = self._beam_search_partition(groups, pp)
            
            if not pp_stages:
                print(f"  [PP={pp}] Failed to partition layers")
                continue
            
            # Build config
            cfg = ParallelConfig(
                tp_degree=tp,
                pp_degree=pp,
                microbatches=mb,
                pp_stages=pp_stages,
                kv_policy="GPU_ONLY"
            )
            
            # Evaluate with cost model
            res = self.cost_model.estimate(cfg)
            
            if res['feasible']:
                cost = res['T_total']
                
                if cost < min_cost:
                    min_cost = cost
                    best_cfg = cfg
                    print(f"  ✓ New Best (PP={pp}) → Latency: {cost*1000:.2f}ms")
                else:
                    print(f"  ✓ Feasible (PP={pp}) → Latency: {cost*1000:.2f}ms")
            else:
                print(f"  ✗ Infeasible (PP={pp}): {res['reason']}")
        
        if best_cfg:
            print(f"[ParallelOptimizer] Optimal: PP={best_cfg.pp_degree}, Latency={min_cost*1000:.2f}ms")
        else:
            print("[ParallelOptimizer] No feasible configuration found!")
            print("[ParallelOptimizer] Suggestions:")
            print("  - Use quantization (FP8/INT8) to reduce memory")
            print("  - Enable KV cache offloading")
            print("  - Add more GPUs for pipeline parallelism")
        
        return best_cfg

    def _form_tp_groups_linear(self, tp: int, pp: int) -> List[List[int]]:
        """
        Chunks devices linearly. Assumes self.devices is sorted by topology.
        Returns List of [dev_ids] for each stage.
        Length must be pp.
        """
        groups = []
        needed = tp * pp
        if len(self.devices) < needed: return []
        
        # Use first N devices
        used_devices = [d.id for d in self.devices[:needed]]
        
        for i in range(pp):
            stage_devs = used_devices[i*tp : (i+1)*tp]
            groups.append(stage_devs)
            
        return groups

    def _beam_search_partition(self, device_groups: List[List[int]], pp_degree: int) -> List[StagePlacement]:
        """
        Finds cut points for L layers into pp_degree stages using Beam Search.
        Objective: Minimize pipeline latency (bottleneck stage latency).
        Constraint: Each stage must fit in memory.
        """
        num_layers = self.model.L
        BEAM_WIDTH = 5  # Keep top 5 candidates per stage
        
        # State: (current_layer_idx, list_of_stages, max_stage_latency)
        # Initial State: At layer 0, no stages, 0 latency.
        beam = [(0, [], 0.0)] 
        
        # Verify pp_degree matches device groups
        if len(device_groups) != pp_degree:
            print(f"[Optimizer] Error: Device groups {len(device_groups)} != PP degree {pp_degree}")
            return []

        # Iterate through each pipeline stage index (0 to PP-1)
        for stage_idx in range(pp_degree):
            next_beam = []
            
            devices = device_groups[stage_idx]
            is_last_stage = (stage_idx == pp_degree - 1)
            
            for start_layer, current_stages, current_max_lat in beam:
                # Determine possible end layers for this stage
                # Constraint: Must leave at least 1 layer for each remaining stage
                remaining_stages = pp_degree - 1 - stage_idx
                min_end = start_layer + 1
                max_end = num_layers - remaining_stages
                
                if is_last_stage:
                    # Last stage MUST consume all remaining layers
                    candidates = [num_layers]
                else:
                    # Heuristic pruning: Don't check every single layer if L is huge.
                    # Check every 1 layer for small models, or stride for large.
                    candidates = range(min_end, max_end + 1)

                for end_layer in candidates:
                    # Form Candidate Stage
                    stage = StagePlacement(devices=devices, layer_range=(start_layer, end_layer))
                    
                    # Evaluate Cost & Feasibility
                    feasible, latency, reason = self._evaluate_candidate_stage(stage, pp_degree)
                    
                    if feasible:
                        new_max_lat = max(current_max_lat, latency)
                        new_stages = current_stages + [stage]
                        
                        # Add to next beam
                        next_beam.append((end_layer, new_stages, new_max_lat))
                    else:
                        # If minimal split is infeasible, this path is dead.
                        # (But maybe a larger split is feasible? Unlikely for memory, likely for latency? 
                        # Actually larger split = more memory. So if small split OOMs, larger will too.)
                        pass

            if not next_beam:
                print(f"[Optimizer] Dead end at stage {stage_idx}. No feasible splits found.")
                return []
                
            # Prune Beam: Sort by lowest max_latency
            next_beam.sort(key=lambda x: x[2])
            beam = next_beam[:BEAM_WIDTH]
            
        # Return best config's stages
        if not beam:
            return []
            
        best_candidate = beam[0]
        return best_candidate[1] # Return list of stages

    def _evaluate_candidate_stage(self, stage: StagePlacement, total_pp_degree: int) -> Tuple[bool, float, str]:
        """
        Evaluates a single stage for memory feasibility and latency.
        Returns: (feasible, latency, reason)
        """
        # Create a dummy config with just this stage
        # We assume TP degree is len(stage.devices)
        tp = len(stage.devices)
        
        dummy_cfg = ParallelConfig(
            tp_degree=tp,
            pp_degree=1, # Treat as single stage for calculation
            microbatches=1, # MB doesn't affect per-stage latency calculation much in _compute_stage_times
            pp_stages=[stage],
            kv_policy="GPU_ONLY"
        )
        
        # 1. Check Memory
        mem_report = self.cost_model._memory_check(dummy_cfg)
        if not mem_report.feasible:
            return False, float('inf'), mem_report.reason
            
        # 2. Check Latency (Prefill + Decode)
        # We consider the maximum of prefill/decode stage time as the bottleneck?
        # Or usually decode is the bottleneck. Let's use T_decode per token.
        # Ideally we minimize End-to-End time.
        # But pipeline bottleneck is defined by the slowest stage.
        
        # Note: _compute_stage_times returns [T_stage]
        try:
            times_prefill = self.cost_model._compute_stage_times(dummy_cfg, "PREFILL")
            times_decode = self.cost_model._compute_stage_times(dummy_cfg, "DECODE_TOKEN")
            
            t_prefill = times_prefill[0] if times_prefill else 0
            t_decode = times_decode[0] if times_decode else 0
            
            # Weighted metric? For serving, Decode dominates.
            # Let's optimize for Decode Latency (Bottleneck)
            return True, t_decode, ""
        except Exception as e:
            return False, float('inf'), str(e)
