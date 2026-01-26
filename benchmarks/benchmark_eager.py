import torch
import time
import subprocess
import threading
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            self.utils.append(get_gpu_util())
            time.sleep(0.2)

    def stop(self):
        self.stop_event.set()

    def get_avg(self):
        if not self.utils:
            return [0, 0]
        avg_0 = sum(u[0] for u in self.utils) / len(self.utils)
        avg_1 = sum(u[1] for u in self.utils) / len(self.utils)
        return [avg_0, avg_1]

def run_suite(model, tokenizer, scenario_name, prompt_config):
    print(f"\nScenario: {scenario_name}")
    print("-" * 110)
    header = "| BS | Toks/s | RPS | Avg TTFT (ms) | Avg TPOT (ms) | GPU0 % | GPU1 % |"
    print(header)
    print("-" * 110)

    results = []
    
    # Initialize padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for bs in BATCH_SIZES:
        print(f"Running BS={bs}...", end="\r")
        ttfts = []
        tpots = []
        total_toks = 0
        total_time = 0
        
        # Determine fixed vs varied
        is_varied = isinstance(prompt_config, tuple)
        
        # Warmup
        warmup_len = prompt_config[1] if is_varied else prompt_config
        input_ids = torch.randint(0, tokenizer.vocab_size, (bs, warmup_len)).to(model.device)
        attn_mask = torch.ones_like(input_ids)
        for _ in range(WARMUP_ITER):
            with torch.no_grad():
                _ = model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=10, do_sample=False)
        
        monitor = UtilMonitor()
        monitor.start()
        
        for _ in range(BENCHMARK_ITER):
            # Generate input
            if is_varied:
                # Varied lengths padded to max_len
                min_len, max_len = prompt_config
                lengths = torch.randint(min_len, max_len + 1, (bs,))
                input_ids = torch.full((bs, max_len), tokenizer.pad_token_id, dtype=torch.long)
                attn_mask = torch.zeros((bs, max_len), dtype=torch.long)
                for i, l in enumerate(lengths):
                    input_ids[i, :l] = torch.randint(0, tokenizer.vocab_size, (l,))
                    attn_mask[i, :l] = 1
                input_ids = input_ids.to(model.device)
                attn_mask = attn_mask.to(model.device)
            else:
                input_ids = torch.randint(0, tokenizer.vocab_size, (bs, prompt_config)).to(model.device)
                attn_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                # Measure TTFT (Prefill)
                t0 = time.time()
                _ = model(input_ids, attention_mask=attn_mask)
                t1 = time.time()
                ttft = (t1 - t0) * 1000
                ttfts.append(ttft)
                
                # Measure Total Generation
                t_gen_start = time.time()
                gen_out = model.generate(
                    input_ids, 
                    attention_mask=attn_mask,
                    max_new_tokens=GEN_LENGTH, 
                    min_new_tokens=GEN_LENGTH, 
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                t_gen_end = time.time()
                
                total_duration = t_gen_end - t_gen_start
                total_time += total_duration
                
                new_toks = (gen_out.shape[1] - input_ids.shape[1]) * bs
                total_toks += new_toks
                
                # TPOT = (Total Gen Time - (Initial Prefill estimate during generate)) / GEN_LENGTH
                # We use the total duration as a better metric for throughput comparison
                tpot = (total_duration * 1000) / (GEN_LENGTH)
                tpots.append(tpot)

        avg_gpu_utils = monitor.get_avg()
        monitor.stop()
        monitor.join()

        avg_ttft = np.mean(ttfts)
        avg_tpot = np.mean(tpots)
        toks_per_sec = total_toks / total_time
        rps = (bs * BENCHMARK_ITER) / total_time

        print(f"| {bs:2d} | {toks_per_sec:7.2f} | {rps:5.2f} | {avg_ttft:13.2f} | {avg_tpot:13.2f} | {avg_gpu_utils[0]:6.1f} | {avg_gpu_utils[1]:6.1f} |")
        
        results.append({
            "bs": bs,
            "toks_sec": toks_per_sec,
            "rps": rps,
            "ttft": avg_ttft,
            "tpot": avg_tpot,
            "gpu0": avg_gpu_utils[0],
            "gpu1": avg_gpu_utils[1]
        })
    return results

def benchmark():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (RTX 3090)")
    print(f"Loading model: {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": device}, 
        trust_remote_code=True
    )
    model.eval()

    print("\nStarting Benchmark Suite (PyTorch Eager)")
    print(f"Iterations: {BENCHMARK_ITER}, Warmup: {WARMUP_ITER}")
    
    fixed_results = run_suite(model, tokenizer, "Fixed Prompt (128)", FIXED_PROMPT_LENGTH)
    varied_results = run_suite(model, tokenizer, "Varied Prompt (128-512)", VARIED_PROMPT_RANGE)

    return {"fixed": fixed_results, "varied": varied_results}

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"Error during benchmark: {e}")
