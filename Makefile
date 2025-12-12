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
SWEEP_PROMPT_LENS ?= 16,64,256,512,1024

bench-sweep-prompt:
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "ERROR: HF_TOKEN is not set. Please export HF_TOKEN=hf_..."; \
		exit 1; \
	fi
	@mkdir -p $(RESULTS_DIR)
	HF_HOME=$(HF_HOME) \
	HF_TOKEN=$(HF_TOKEN) \
	uv run python -m benchmarks.single_gpu \
		--model $(MODEL_LLAMA3) \
		--dtype $(DTYPE) \
		--max-new-tokens $(MAX_NEW_TOKENS) \
		--sweep "$(SWEEP_PROMPT_LENS)" \
		--out $(RESULTS_DIR)/prompt_sweep.jsonl


bench: bench-sweep-prompt



# -------------------------
# Cleanup
# -------------------------
clean:
	rm -rf $(RESULTS_DIR)
	rm -rf .hf_cache
