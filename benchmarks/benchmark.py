import time
import threading
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import matplotlib.pyplot as plt
import json

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    HAS_GPU_MONITORING = True
except ImportError:
    HAS_GPU_MONITORING = False

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.timestamps = []
        self.cpu_usages = []
        self.gpu_usages = []
        self.gpu_mem_usages = []
        self.start_time = None
        
        if HAS_GPU_MONITORING:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
            except Exception as e:
                print(f"Failed to initialize NVML: {e}")
                self.device_count = 0
        else:
            self.device_count = 0

    def run(self):
        self.start_time = time.time()
        while not self.stop_event.is_set():
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            
            # CPU Usage
            self.cpu_usages.append(psutil.cpu_percent(interval=None))
            
            # GPU Usage
            if self.device_count > 0:
                current_gpu_util = []
                current_gpu_mem = []
                for handle in self.handles:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used = mem_info.used / 1024**2 # MB
                        current_gpu_util.append(util)
                        current_gpu_mem.append(mem_used)
                    except Exception:
                        pass
                
                if current_gpu_util:
                    self.gpu_usages.append(sum(current_gpu_util) / len(current_gpu_util)) # Average across GPUs
                    self.gpu_mem_usages.append(sum(current_gpu_mem)) # Total memory
                else:
                    self.gpu_usages.append(0)
                    self.gpu_mem_usages.append(0)
            else:
                self.gpu_usages.append(0)
                self.gpu_mem_usages.append(0)
            
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        if HAS_GPU_MONITORING and self.device_count > 0:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def get_stats(self):
        avg_cpu = sum(self.cpu_usages) / len(self.cpu_usages) if self.cpu_usages else 0
        avg_gpu = sum(self.gpu_usages) / len(self.gpu_usages) if self.gpu_usages else 0
        max_gpu_mem = max(self.gpu_mem_usages) if self.gpu_mem_usages else 0
        return {
            "avg_cpu_utilization": avg_cpu,
            "avg_gpu_utilization": avg_gpu,
            "max_gpu_memory_mb": max_gpu_mem,
            "timestamps": self.timestamps,
            "cpu_usages": self.cpu_usages,
            "gpu_usages": self.gpu_usages,
            "gpu_mem_usages": self.gpu_mem_usages
        }

def plot_results(stats, output_file="benchmark_results.png"):
    timestamps = stats["timestamps"]
    cpu_usages = stats["cpu_usages"]
    gpu_usages = stats["gpu_usages"]
    gpu_mem_usages = stats["gpu_mem_usages"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Utilization (%)', color='black')
    ax1.plot(timestamps, cpu_usages, label='CPU Utilization', color='blue')
    ax1.plot(timestamps, gpu_usages, label='GPU Utilization', color='green')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('GPU Memory (MB)', color='red')  # we already handled the x-label with ax1
    ax2.plot(timestamps, gpu_mem_usages, label='GPU Memory', color='red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.title('System Resource Utilization During Inference')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

import os

def save_json_results(results, output_file="benchmark_results.json"):
    data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                else:
                    data = [content]
        except json.JSONDecodeError:
            pass
    
    data.append(results)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results appended to {output_file}")

def benchmark_inference(model_name="Qwen/Qwen3-4B-Instruct-2507", num_tokens=100):
    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded successfully.")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

    prompt = "Hello, can you tell me a story about a brave knight?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    print("Warming up...")
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    print(f"Starting benchmark for {num_tokens} tokens...")
    
    monitor = ResourceMonitor()
    monitor.start()
    
    start_time = time.time()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=num_tokens, 
            do_sample=True, # Use sampling for more realistic load
            temperature=0.7
        )
    
    end_time = time.time()
    monitor.stop()
    monitor.join()

    total_time = end_time - start_time
    generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = generated_tokens / total_time
    
    stats = monitor.get_stats()

    # Create results dictionary
    results_data = {
        "model_name": model_name,
        "configuration": {
            "num_tokens": num_tokens,
            "device_map": str(model.hf_device_map) if hasattr(model, 'hf_device_map') else "N/A"
        },
        "performance": {
            "total_time_seconds": total_time,
            "generated_tokens": generated_tokens,
            "tokens_per_second": tokens_per_sec
        },
        "resources": {
            "avg_cpu_utilization": stats['avg_cpu_utilization'],
            "avg_gpu_utilization": stats['avg_gpu_utilization'],
            "max_gpu_memory_mb": stats['max_gpu_memory_mb']
        }
    }

    print("\n" + "="*30)
    print("BENCHMARK RESULTS")
    print("="*30)
    print(f"Model: {model_name}")
    print(f"Tokens generated: {generated_tokens}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"Avg CPU Utilization: {stats['avg_cpu_utilization']:.1f}%")
    print(f"Avg GPU Utilization: {stats['avg_gpu_utilization']:.1f}%")
    print(f"Max GPU Memory Used: {stats['max_gpu_memory_mb']:.0f} MB")
    print("="*30)
    
    plot_results(stats)
    save_json_results(results_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM Inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Model name or path")
    parser.add_argument("--tokens", type=int, default=100, help="Number of new tokens to generate")
    args = parser.parse_args()
    
    benchmark_inference(args.model, args.tokens)
