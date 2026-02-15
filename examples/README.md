# HELM Examples

This directory contains executable scripts demonstrating how to use HELM to compile and run large language models.

## Demo Scripts

*   **`run_qwen.py`**: Compiles and runs the Qwen-7B model using the standard HELM backend.
*   **`generate_qwen_graph.py`**: Specialized script to dump the intermediate FX graph of Qwen-7B for analysis.
*   **`cost_model_sweep.py`**: A simulator script that lets you predict the performance of various models (Llama-7B, Llama-70B, GPT-2) on arbitrary hardware configurations (A100, T4, CPU) without actually running them. It outputs feasible parallel strategies (PP vs TP vs Offload).

## Usage

To run the cost model simulator:

```bash
uv run python examples/cost_model_sweep.py
```

To run an inference demo (requires HuggingFace token and GPU):

```bash
uv run python examples/run_qwen.py
```
