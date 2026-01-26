import os
import time
import subprocess
import threading
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Force RTX 3090 (Confirmed as Index 0 in CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
BATCH_SIZES = [1, 4, 8, 16, 32]
FIXED_PROMPT_LENGTH = 128
VARIED_PROMPT_RANGE = (128, 512)
GEN_LENGTH = 128
WARMUP_ITER = 2
BENCHMARK_ITER = 10

def get_gpu_util():
    try:
        # Note: We are only interested in GPU 1 (RTX 3090) but nvidia-smi with CUDA_VISIBLE_DEVICES=1 might show it as index 0 or 1 depending on query
        # To be safe, we query all and pick the one with usage
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        utils = [int(x) for x in res.decode().strip().split('\n')]
        return utils
    except Exception:
        return [0, 0]

class UtilMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.utils = []

    def run(self):
        while not self.stop_event.is_set():
            # Query all GPUs. When CUDA_VISIBLE_DEVICES=0, physical 3090 is usually the only one visible or first.
            self.utils.append(get_gpu_util())
            time.sleep(0.5)

    def stop(self):
        self.stop_event.set()

    def get_avg(self):
        if not self.utils:
            return [0, 0]
        # Since indices can be confusing, we report the max of any active GPU as "active" 
        # and assume the rest are idle. 
        # Based on our checks: 3090 is physical 0 (under C_V_D=0) or physical 1 (global).
        # We will report the first two indices from nvidia-smi.
        avg_0 = sum(u[0] for u in self.utils) / len(self.utils)
        avg_1 = sum(u[1] for u in self.utils) / len(self.utils) if len(self.utils[0]) > 1 else 0
        return [avg_0, avg_1]

def run_suite(llm, tokenizer, scenario_name, prompt_config):
    print(f"\nScenario: {scenario_name}")
    print("-" * 110)
    header = "| BS | Toks/s | RPS | Avg TTFT (ms) | Avg TPOT (ms) | GPU0 % | GPU1 % |"
    print(header)
    print("-" * 110)

    results = []
    sampling_params = SamplingParams(
        max_tokens=GEN_LENGTH,
        min_tokens=GEN_LENGTH,
        temperature=0.0 # Greedy
    )

    for bs in BATCH_SIZES:
        print(f"Running BS={bs}...", end="\r")
        
        # Prepare prompts
        is_varied = isinstance(prompt_config, tuple)
        prompts = []
        for _ in range(bs):
            if is_varied:
                length = np.random.randint(prompt_config[0], prompt_config[1] + 1)
            else:
                length = prompt_config
            # Generate dummy tokens and decode to text (vLLM likes text or token IDs)
            token_ids = np.random.randint(0, tokenizer.vocab_size, (length,)).tolist()
            prompts.append({"prompt_token_ids": token_ids})

        # Warmup (shared across BS)
        if bs == BATCH_SIZES[0]:
            for _ in range(WARMUP_ITER):
                llm.generate(prompts[:1], sampling_params, use_tqdm=False)

        latencies = []
        total_toks = 0
        
        monitor = UtilMonitor()
        monitor.start()
        
        start_time = time.time()
        for _ in range(BENCHMARK_ITER):
            iter_start = time.time()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            iter_end = time.time()
            
            latencies.append((iter_end - iter_start) * 1000)
            for output in outputs:
                total_toks += len(output.outputs[0].token_ids)

        end_time = time.time()
        monitor.stop()
        monitor.join()

        avg_latency = np.mean(latencies)
        # vLLM 'latency' here is for the whole batch. 
        # In continuous batching, TTFT is often very fast. 
        # Without AsyncEngine, we can't easily get TTFT per request.
        # We'll report the batch latency / GEN_LENGTH as an estimate for TPOT, 
        # and we'll leave TTFT as 'N/A' or a separate measurement if possible.
        # Actually, let's just use the average iteration time / GEN_LENGTH for TPOT.
        
        toks_per_sec = total_toks / (end_time - start_time)
        rps = (bs * BENCHMARK_ITER) / (end_time - start_time)
        
        # Estimate: In vLLM offline, TTFT is usually < 50ms for small prompts.
        # Without AsyncEngine, we can't easily get TTFT per request.
        avg_ttft = "N/A"
        avg_tpot = (avg_latency) / (GEN_LENGTH)

        # Map utils back to physical GPUs
        # GPU0 (1080) should be 0. GPU1 (3090) should be active.
        avg_utils = monitor.get_avg()
        # Since we set CUDA_VISIBLE_DEVICES=1, nvidia-smi might show it as the only GPU or as index 1.
        # We'll just report what we found.
        
        print(f"| {bs:2d} | {toks_per_sec:7.2f} | {rps:5.2f} | {avg_ttft:>13s} | {avg_tpot:13.2f} | {avg_utils[0]:6.1f} | {avg_utils[1]:6.1f} |")
        
        results.append({
            "bs": bs,
            "toks_sec": toks_per_sec,
            "rps": rps,
            "ttft": avg_ttft,
            "tpot": avg_tpot,
            "gpu0": avg_utils[0],
            "gpu1": avg_utils[1]
        })
    return results

def benchmark():
    print(f"Initializing vLLM with model: {MODEL_ID}")
    # Initialize vLLM
    # We use tensor_parallel_size=1 since we only have one compatible GPU
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=True # Avoid graph overhead for short benchmarks
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\nStarting Benchmark Suite (vLLM)")
    print(f"Iterations: {BENCHMARK_ITER}, Warmup: {WARMUP_ITER}")
    
    fixed_results = run_suite(llm, tokenizer, "Fixed Prompt (128)", FIXED_PROMPT_LENGTH)
    varied_results = run_suite(llm, tokenizer, "Varied Prompt (128-512)", VARIED_PROMPT_RANGE)

    return {"fixed": fixed_results, "varied": varied_results}

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"\nError during benchmark: {e}")
