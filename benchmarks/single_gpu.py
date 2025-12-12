# benchmarks/single_gpu.py
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

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


def _parse_csv_ints(s: str) -> List[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def _make_synthetic_prompt(tokenizer, target_len: int) -> str:
    """
    Create a prompt that tokenizes to *approximately* target_len tokens.
    For planner modeling, exact length isn't critical, but we try to be close.
    """
    # A token-friendly base word; repeated words typically map to 1 token each for many tokenizers.
    base = " hello"
    prompt = ""
    # Build and check length iteratively (cheap for these sizes)
    while True:
        prompt += base * 64
        toks = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        if toks >= target_len:
            break
        if toks > target_len * 4:
            break  # safety
    # Trim down by binary-ish shrinking
    while True:
        toks = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        if toks <= target_len:
            break
        # remove some chunk
        prompt = prompt[: max(0, len(prompt) - len(base) * 16)]
        if len(prompt) == 0:
            break
    # If we're short, add a few tokens
    while True:
        toks = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        if toks >= target_len:
            break
        prompt += base
    return prompt.strip()


@torch.no_grad()
def run_once(args, tok, model, prompt: str) -> BenchResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tok(prompt, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # warmup
    _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    _sync()

    # TTFT = prefill + first decode step
    _sync()
    t0 = time.perf_counter()

    out = model(**inputs, use_cache=True)
    past = out.past_key_values

    next_inp = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    out2 = model(input_ids=next_inp, use_cache=True, past_key_values=past)

    _sync()
    ttft = time.perf_counter() - t0

    remaining = args.max_new_tokens - 1
    if remaining < 1:
        remaining = 1

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

    import transformers as _tf  # noqa

    return BenchResult(
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument(
        "--prompt",
        type=str,
        default="Write a short explanation of heterogeneous LLM inference.",
    )
    p.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        help="If >0, generate a synthetic prompt of ~this token length.",
    )
    p.add_argument(
        "--sweep",
        type=str,
        default="",
        help='Comma-separated prompt lengths, e.g. "16,64,256,512,1024".',
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

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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

    # Determine runs
    sweep_lens: List[int] = []
    if args.sweep.strip():
        sweep_lens = _parse_csv_ints(args.sweep)
    elif args.prompt_len and args.prompt_len > 0:
        sweep_lens = [args.prompt_len]

    results = []

    if sweep_lens:
        for L in sweep_lens:
            prompt = _make_synthetic_prompt(tok, L)
            res = run_once(args, tok, model, prompt)
            results.append(res)
            print(json.dumps(asdict(res), indent=2))
            if args.out:
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(res)) + "\n")
    else:
        res = run_once(args, tok, model, args.prompt)
        print(json.dumps(asdict(res), indent=2))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(res)) + "\n")


if __name__ == "__main__":
    main()
