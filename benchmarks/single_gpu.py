# bench/single_gpu_llama3.py
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchResult:
    model: str
    dtype: str
    quant: Optional[str]
    device: str
    prompt_len: int
    max_new_tokens: int
    ttft_s: float
    decode_toks_per_s: float
    total_time_s: float
    peak_vram_mb: float
    torch_version: str
    transformers_version: str


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument(
        "--prompt",
        type=str,
        default="Write a short explanation of heterogeneous LLM inference.",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    p.add_argument(
        "--quant", type=str, default="none", choices=["none", "8bit", "4bit"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dtype selection
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model load
    quant = None if args.quant == "none" else args.quant
    load_kwargs = dict(torch_dtype=torch_dtype, device_map="auto")

    if quant is not None:
        if quant == "8bit":
            load_kwargs.update(dict(load_in_8bit=True))
        elif quant == "4bit":
            load_kwargs.update(dict(load_in_4bit=True))
        else:
            raise ValueError(f"unknown quant {quant}")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    # warmup + benchmark input
    inputs = tok(args.prompt, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # warmup (short)
    _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    _sync()

    # --- TTFT measurement ---
    _sync()
    t0 = time.perf_counter()

    # prefill
    out = model(**inputs, use_cache=True)
    past = out.past_key_values

    # first decode step
    next_inp = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    out2 = model(input_ids=next_inp, use_cache=True, past_key_values=past)

    _sync()
    ttft = time.perf_counter() - t0

    # --- decode throughput measurement ---
    remaining = args.max_new_tokens - 1
    if remaining < 1:
        remaining = 1

    # start from the token we just produced
    cur = next_inp
    past = out2.past_key_values

    _sync()
    t1 = time.perf_counter()
    for _i in range(remaining):
        out_step = model(input_ids=cur, use_cache=True, past_key_values=past)
        past = out_step.past_key_values
        cur = out_step.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    _sync()
    t2 = time.perf_counter()

    decode_time = t2 - t1
    decode_tps = remaining / decode_time if decode_time > 0 else float("inf")
    total_time = ttft + decode_time

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # lightweight versions
    import transformers as _tf  # noqa

    res = BenchResult(
        model=args.model,
        dtype=args.dtype,
        quant=args.quant if args.quant != "none" else None,
        device=device,
        prompt_len=prompt_len,
        max_new_tokens=args.max_new_tokens,
        ttft_s=ttft,
        decode_toks_per_s=decode_tps,
        total_time_s=total_time,
        peak_vram_mb=peak_mb,
        torch_version=torch.__version__,
        transformers_version=_tf.__version__,
    )

    print(json.dumps(asdict(res), indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(res)) + "\n")


if __name__ == "__main__":
    main()
