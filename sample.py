"""Generate Toki Pona text from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from tokenizer import TokiPonaTokenizer
from train import pick_device


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from a trained Toki Pona LM")
    p.add_argument("checkpoint", type=Path, help="Path to saved checkpoint directory")
    p.add_argument("--prompt", default="tenpo pini la", help="Starting Toki Pona text")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    tokenizer = TokiPonaTokenizer()
    model = LlamaForCausalLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    ids = tokenizer.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"[{i + 1}] {text}\n")


if __name__ == "__main__":
    main()
