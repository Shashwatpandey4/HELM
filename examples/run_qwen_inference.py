import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import psutil
import csv
import numpy as np
import gc

model_id = "Qwen/Qwen2.5-7B-Instruct"
csv_file = "qwen_metrics.csv"

def check_memory():
    print("\n--- Current Memory Status ---")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU VRAM: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    try:
        ram = psutil.virtual_memory()
        print(f"System RAM: Used={ram.used/(1024**3):.2f}GB, Available={ram.available/(1024**3):.2f}GB, Total={ram.total/(1024**3):.2f}GB")
    except:
        pass

def save_metrics(metrics):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"\nMetrics saved to {csv_file}")

def synchronize_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def run_model():
    print(f"--- Attempting to load {model_id} in FP16 with Averaged Run Metrics ---")
    check_memory()
    
    start_time = synchronize_time()
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nStep 1: Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Step 2: Loading Model in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="offload_fp16"
        )
        
        load_time = synchronize_time() - start_time
        print(f"\nSuccessfully loaded in {load_time:.2f} seconds.")
        check_memory()
        
        prompt = "Explain why the sky is blue in one sentence."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print(f"\nStep 3: Running inference loop (10 iterations)...")
        
        # Lists to store metrics across all 10 runs
        all_ttfts = []
        all_tpots = []
        all_e2e_latencies = []
        
        gc.disable()
        
        for i in range(10):
            print(f"  Run {i+1}/10...", end="", flush=True)
            
            input_ids = model_inputs.input_ids
            attention_mask = model_inputs.attention_mask
            
            run_start = synchronize_time()
            token_times = []
            
            with torch.no_grad():
                # TTFT
                t0 = synchronize_time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                t1 = synchronize_time()
                
                ttft = t1 - t0
                all_ttfts.append(ttft)
                
                generated_ids = [next_token.item()]
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=model.device)], dim=-1)

                # Generate 15 more tokens (kept relatively short to save time)
                max_new_tokens = 50
                for _ in range(max_new_tokens - 1):
                    t_token_start = synchronize_time()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    t_token_end = synchronize_time()
                    
                    token_times.append(t_token_end - t_token_start)
                    generated_ids.append(next_token.item())
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=model.device)], dim=-1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            run_end = synchronize_time()
            all_e2e_latencies.append(run_end - run_start)
            
            # Calculate avg TPOT for this specific run
            if token_times:
                all_tpots.extend(token_times)
                
            print(f" Done. (TTFT: {ttft:.2f}s)")

        gc.enable()
        
        # Final Averaged Calculations
        avg_ttft = np.mean(all_ttfts)
        avg_tpot = np.mean(all_tpots)
        p50_tpot = np.percentile(all_tpots, 50)
        p99_tpot = np.percentile(all_tpots, 99)
        avg_e2e = np.mean(all_e2e_latencies)

        metrics = {
            "load_time": f"{load_time:.4f}",
            "avg_ttft": f"{avg_ttft:.4f}",
            "avg_tpot": f"{avg_tpot:.4f}",
            "p50_tpot": f"{p50_tpot:.4f}",
            "p99_tpot": f"{p99_tpot:.4f}",
            "avg_e2e_latency": f"{avg_e2e:.4f}",
            "total_runs": 10
        }

        print("\n--- Final Averaged Metrics (Over 10 Runs) ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
        save_metrics(metrics)

    except torch.cuda.OutOfMemoryError as e:
        print("\n OOM Error: The combined 24GB (8 GPU + 16 CPU) wasn't enough.")
        check_memory()
    except Exception as e:
        print(f"\n Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists("offload_fp16"):
        os.makedirs("offload_fp16")
    run_model()
