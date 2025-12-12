# =========================
# HELM Makefile
# =========================

HF_HOME ?= $(PWD)/.hf_cache
HF_TOKEN ?= 

RESULTS_DIR ?= results

MODEL_LLAMA3 ?= meta-llama/Llama-3.2-3B-Instruct
DTYPE ?= bf16
MAX_NEW_TOKENS ?= 128

.PHONY: help setup bench bench-single clean

help:
	@echo "HELM Makefile targets:"
	@echo "  setup          - install dependencies with uv"
	@echo "  bench-single   - run single-GPU Llama-3 benchmark"
	@echo "  bench          - alias for bench-single"
	@echo "  clean          - remove results and caches"

# -------------------------
# Setup
# -------------------------
setup:
	uv sync

# -------------------------
# Benchmarks
# -------------------------
bench-single:
	@mkdir -p $(RESULTS_DIR)
	HF_HOME=$(HF_HOME) \
	HF_TOKEN=$(HF_TOKEN) \
	uv run python -m benchmarks.single_gpu \
		--model $(MODEL_LLAMA3) \
		--dtype $(DTYPE) \
		--max-new-tokens $(MAX_NEW_TOKENS) \
		--out $(RESULTS_DIR)/single_gpu.jsonl

bench: bench-single

# -------------------------
# Cleanup
# -------------------------
clean:
	rm -rf $(RESULTS_DIR)
	rm -rf .hf_cache
