from typing import List, Dict, Tuple, Optional
import itertools
from helm.passes.cost_model import HelmCostModel, ParallelConfig, StagePlacement, ModelSpec, DeviceSpec, CalibrationDB

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
        Main entry point. Returns the best feasible ParallelConfig found.
        """
        print("[ParallelOptimizer] Starting Optimization...")
        
        # 1. Macro Search Space
        # Enumerate feasible (tp, pp) pairs
        # Constraint: tp * pp <= num_devices
        # Constraint: tp should likely be a power of 2 (1, 2, 4, 8)
        # Constraint: pp can be anything
        num_devs = len(self.devices)
        
        best_cfg = None
        min_cost = float('inf')
        
        # Heuristic range for TP: 1, 2, 4, 8 (if supported)
        possible_tp = [1, 2, 4, 8]
        possible_pp = range(1, num_devs + 1)
        
        # Microbatch sweep? Only affect pipeline bubble vs utilization.
        # Heuristic: 1, 2, 4, 8
        possible_mb = [1, 4] 
        
        candidates = []
        
        for tp in possible_tp:
            if tp > num_devs: continue
            
            for pp in possible_pp:
                total_req = tp * pp
                if total_req > num_devs: continue
                
                # Check if this combo makes sense?
                # E.g. skip if total_req < num_devs/2 (underutilization), unless specific constraint.
                # For now search all.
                
                for mb in possible_mb:
                    if pp == 1 and mb > 1: continue # No pipeline bubble to hide, MB useless (adds overhead)? Or helps memory?
                    
                    # 2. Structured Subsearch
                    # We need to form PP stages where each stage has size TP.
                    # This means grouping devices.
                    
                    # Grouping Strategy:
                    # Simple: Linear chunking of device ID list.
                    # [0, 1, 2, 3] -> TP=2, PP=2 -> Stage0=[0,1], Stage1=[2,3]
                    # This assumes devices list is sorted by topology proximity (e.g. 0-1 are on same node).
                    # A robust optimizer would enumerate valid topology groupings.
                    # We stick to linear chunking for prototype.
                    
                    groups = self._form_tp_groups_linear(tp, pp)
                    if not groups: continue
                    
                    # 3. PP Partitioning
                    # Given the sequence of groups (Stages), how to cut layers?
                    # We use Beam Search to find split points.
                    pp_stages = self._beam_search_partition(groups, pp)
                    
                    if not pp_stages: continue
                    
                    # Build Config
                    cfg = ParallelConfig(
                        tp_degree=tp,
                        pp_degree=pp,
                        microbatches=mb,
                        pp_stages=pp_stages,
                        kv_policy="GPU_ONLY" # Fixed for now
                    )
                    
                    # Evaluate
                    res = self.cost_model.estimate(cfg)
                    
                    if res['feasible']:
                        cost = res['T_total'] # Minimizing Latency? Or T_decode?
                        # Let's minimize Total Latency (Prefill + Gen).
                        
                        candidates.append((cfg, cost))
                        
                        if cost < min_cost:
                            min_cost = cost
                            best_cfg = cfg
                            print(f"  New Best (TP={tp}, PP={pp}, MB={mb}) -> Latency: {cost*1000:.2f}ms")
                        else:
                            # print(f"  Feasible (TP={tp}, PP={pp}) -> {cost*1000:.2f}ms")
                            pass
                    else:
                        # print(f"  Infeasible (TP={tp}, PP={pp}): {res['reason']}")
                        pass

        if best_cfg:
            print(f"[ParallelOptimizer] Optimal Found: TP={best_cfg.tp_degree}, PP={best_cfg.pp_degree}")
        else:
            print("[ParallelOptimizer] No feasible configuration found.")
            
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
        Finds cut points for L layers into pp_degree stages.
        """
        num_layers = self.model.L
        
        # If PP=1, trivial
        if pp_degree == 1:
            return [StagePlacement(devices=device_groups[0], layer_range=(0, num_layers))]
            
        # Brute Force / Beam limits
        # Since number of layers is usually small (32-80), and PP degree is small (2-8),
        # we can just iterate uniform splits or use a simplified heuristic for now.
        
        # Heuristic: Even split of layers
        # TODO: Implement actual cost-aware beam search (evaluating memory imbalance).
        
        # Simple Logic: Even split
        layers_per_stage = num_layers // pp_degree
        rem = num_layers % pp_degree
        
        stages = []
        start = 0
        for i in range(pp_degree):
            count = layers_per_stage + (1 if i < rem else 0)
            end = start + count
            stages.append(StagePlacement(devices=device_groups[i], layer_range=(start, end)))
            start = end
            
        return stages
