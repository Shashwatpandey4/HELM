# HELM Benchmarks

This directory contains benchmarking results for LLM inference performance on a heterogeneous GPU system.

## System Configuration
- **GPU 0**: NVIDIA GeForce GTX 1080
- **GPU 1**: NVIDIA GeForce RTX 3090

## Benchmark Methodology
We are benchmarking inference performance using various batch sizes to evaluate throughput and latency metrics.

### Model
- **Model Name**: `Qwen/Qwen3-4B-Instruct-2507`
- **Framework**: PyTorch Eager (Initial Test)

### Metrics
1. **Toks/sec**: Total tokens generated per second.
2. **Requests per second (RPS)**: Number of requests processed per second.
3. **Avg. TTFT**: Average Time To First Token.
4. **Avg. TPOT**: Average Time Per Output Token.
5. **Prefill (TTFT)**: Latency for the prefill stage.
6. **Decode (TPOT)**: Latency for the decoding stage.

## Benchmark Results

> [!IMPORTANT]
> **GPU Compatibility Alert**: The GTX 1080 (sm_61) is not compatible with the current PyTorch installation (requires sm_70+). All benchmarks below were run on the **RTX 3090 (GPU 1)**.
> 
> **Configuration**:
> - **Iterations**: 10 (averaged)
> - **Warmup**: 2
> - **Gen Length**: 128 tokens

### Scenario: Fixed Prompt (128 tokens)
| Framework | Batch Size | Toks/sec | RPS | Avg. TTFT | Avg. TPOT | GPU 0 Util | GPU 1 Util |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch Eager | **1** | 62.18 | 0.49 | 12.18 ms | 16.08 ms | 0.0% | 98.0% |
| PyTorch Eager | **4** | 229.14 | 1.79 | 37.06 ms | 17.46 ms | 0.0% | 98.1% |
| PyTorch Eager | **8** | 418.29 | 3.27 | 78.93 ms | 19.13 ms | 0.0% | 98.1% |
| PyTorch Eager | **16** | 715.00 | 5.59 | 146.59 ms | 22.38 ms | 0.0% | 98.4% |
| PyTorch Eager | **32** | 1153.92 | 9.02 | 282.32 ms | 27.73 ms | 0.0% | 98.7% |
| torch.compile | **1** | 62.03 | 0.48 | 1278.85 ms | 16.12 ms | 0.0% | 64.1% |
| torch.compile | **4** | 227.16 | 1.77 | 2119.73 ms | 17.61 ms | 0.0% | 54.5% |
| torch.compile | **8** | 410.24 | 3.21 | 16.81 ms | 19.50 ms | 0.0% | 98.2% |
| torch.compile | **16** | 688.43 | 5.38 | 4.88 ms | 23.24 ms | 0.0% | 98.3% |
| torch.compile | **32** | 1090.11 | 8.52 | 4.73 ms | 29.35 ms | 0.0% | 98.7% |
| HF Accelerate | **1** | 62.27 | 0.49 | 12.16 ms | 16.06 ms | 0.0% | 98.0% |
| HF Accelerate | **4** | 229.00 | 1.79 | 37.10 ms | 17.47 ms | 0.0% | 98.1% |
| HF Accelerate | **8** | 418.32 | 3.27 | 79.00 ms | 19.12 ms | 0.0% | 98.2% |
| HF Accelerate | **16** | 715.14 | 5.59 | 146.98 ms | 22.37 ms | 0.0% | 98.4% |
| HF Accelerate | **32** | 1155.50 | 9.03 | 282.33 ms | 27.69 ms | 0.0% | 98.7% |
| vLLM | **1** | 87.56 | 0.68 | N/A | 11.42 ms | 0.0% | 99.9% |
| vLLM | **4** | 326.92 | 2.55 | N/A | 12.24 ms | 0.0% | 95.0% |
| vLLM | **8** | 649.98 | 5.08 | N/A | 12.31 ms | 0.0% | 97.9% |
| vLLM | **16** | 1151.49 | 9.00 | N/A | 13.89 ms | 0.0% | 99.4% |
| vLLM | **32** | 2204.14 | 17.22 | N/A | 14.52 ms | 0.0% | 97.6% |

### Scenario: Varied Prompt (128-512 tokens)
| Framework | Batch Size | Toks/sec | RPS | Avg. TTFT | Avg. TPOT | GPU 0 Util | GPU 1 Util |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch Eager | **1** | 49.35 | 0.39 | 42.46 ms | 20.27 ms | 0.0% | 93.4% |
| PyTorch Eager | **4** | 145.05 | 1.13 | 168.31 ms | 27.58 ms | 0.0% | 95.4% |
| PyTorch Eager | **8** | 217.25 | 1.70 | 318.69 ms | 36.82 ms | 0.0% | 96.6% |
| PyTorch Eager | **16** | 288.93 | 2.26 | 602.49 ms | 55.38 ms | 0.0% | 97.5% |
| PyTorch Eager | **32** | 351.56 | 2.75 | 1201.35 ms | 91.02 ms | 0.0% | 98.5% |
| torch.compile | **1** | 49.64 | 0.39 | 2292.18 ms | 20.14 ms | 0.0% | 53.1% |
| torch.compile | **4** | 139.75 | 1.09 | 2850.87 ms | 28.62 ms | 0.0% | 57.6% |
| torch.compile | **8** | 205.21 | 1.60 | 4.74 ms | 38.98 ms | 0.0% | 96.5% |
| torch.compile | **16** | 270.40 | 2.11 | 4.90 ms | 59.17 ms | 0.0% | 97.5% |
| torch.compile | **32** | 325.86 | 2.55 | 4.75 ms | 98.20 ms | 0.0% | 98.5% |
| HF Accelerate | **1** | 49.43 | 0.39 | 41.16 ms | 20.23 ms | 0.0% | 93.3% |
| HF Accelerate | **4** | 144.67 | 1.13 | 169.81 ms | 27.65 ms | 0.0% | 95.1% |
| HF Accelerate | **8** | 217.26 | 1.70 | 319.83 ms | 36.82 ms | 0.0% | 96.6% |
| HF Accelerate | **16** | 288.82 | 2.26 | 603.91 ms | 55.40 ms | 0.0% | 97.5% |
| HF Accelerate | **32** | 351.04 | 2.74 | 1203.50 ms | 91.16 ms | 0.0% | 98.5% |
| vLLM | **1** | 86.71 | 0.68 | N/A | 11.53 ms | 0.0% | 99.9% |
| vLLM | **4** | 329.09 | 2.57 | N/A | 12.15 ms | 0.0% | 96.7% |
| vLLM | **8** | 624.31 | 4.88 | N/A | 12.81 ms | 0.0% | 96.6% |
| vLLM | **16** | 1092.83 | 8.54 | N/A | 14.64 ms | 0.0% | 97.1% |
| vLLM | **32** | 2021.49 | 15.79 | N/A | 15.83 ms | 0.0% | 97.3% |
