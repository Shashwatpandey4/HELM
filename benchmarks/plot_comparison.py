import json
import matplotlib.pyplot as plt
import os

def plot_comparison(input_file="benchmark_results.json", output_file="benchmark_comparison.png"):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode {input_file}")
        return

    if not isinstance(data, list):
        data = [data]

    model_names = []
    throughputs = []
    gpu_utils = []

    # Process data to handle duplicates (keep latest) or just plot all
    # Here we'll just plot all runs found
    for run_idx, run in enumerate(data):
        name = run.get("model_name", "Unknown")
        # Distinguish runs if same model appears multiple times
        display_name = f"{name}\n(Run {run_idx+1})"
        
        tps = run["performance"]["tokens_per_second"]
        gpu = run["resources"]["avg_gpu_utilization"]
        
        model_names.append(display_name)
        throughputs.append(tps)
        gpu_utils.append(gpu)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Model Run')
    ax1.set_ylabel('Throughput (tokens/sec)', color=color)
    bars1 = ax1.bar(model_names, throughputs, color=color, alpha=0.6, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('Avg GPU Utilization (%)', color=color)
    # Plot as line or points to distinguish from bars, or narrow bars
    ax2.plot(model_names, gpu_utils, color=color, marker='o', linestyle='-', linewidth=2, label='GPU Util')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Benchmark Comparison: Throughput & GPU Usage')
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    plot_comparison()
