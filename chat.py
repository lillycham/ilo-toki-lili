"""Interactive multi-turn Toki Pona chat with an SFT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from arch import generate_tokens, get_max_ctx, load_causal_lm
from tokenizer import TokiPonaTokenizer
from train import pick_device


def main() -> None:
    p = argparse.ArgumentParser(description="Chat in Toki Pona with an SFT model")
    p.add_argument("checkpoint", nargs="?", type=Path, default=Path("checkpoints/14m-chat"))
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    tokenizer = TokiPonaTokenizer()
    model = load_causal_lm(args.checkpoint, device)
    model.eval()

    nl_id = tokenizer.tok2id["\n"]
    max_ctx = get_max_ctx(model)
    history: list[tuple[str, str]] = []

    print(f"Loaded {args.checkpoint} on {device}. Type Toki Pona (or /reset, /quit).")
    while True:
        try:
            user = input("jan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "pini"):
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)")
            continue

        history.append(("jan", " ".join(user.split())))

        # Build prompt, dropping oldest turns if it exceeds context window.
        while True:
            ids = [tokenizer.bos_token_id]
            for role, text in history:
                ids.extend(tokenizer.encode(f"{role}: {text}\n"))
            ids.extend(tokenizer.encode("ilo:"))
            if len(ids) + args.max_new_tokens <= max_ctx or len(history) <= 1:
                break
            history = history[2:] if len(history) > 2 else history[1:]

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = generate_tokens(
                model=model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, nl_id],
            )
        gen_ids = out[0][len(ids) :].tolist()
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"ilo: {reply}")
        history.append(("ilo", reply))


if __name__ == "__main__":
    main()
