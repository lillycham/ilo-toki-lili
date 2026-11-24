"""Train a tiny monolingual Toki Pona LlamaForCausalLM."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from data import load_or_build_corpus
from tokenizer import TokiPonaTokenizer

DEFAULT_PROMPTS = [
    "tenpo pini la",
    "mi wile",
    "jan Sonja li",
    "tomo tawa ni li",
]


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    max_start = len(tokens) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    offsets = torch.arange(seq_len)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    return tokens[idx].to(device=device, dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: LlamaForCausalLM,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_batches: int = 25,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x = sample_batch(val_tokens, batch_size, seq_len, device)
        out = model(input_ids=x, labels=x)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_samples(
    model: LlamaForCausalLM,
    tokenizer: TokiPonaTokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 48,
    temperature: float = 0.8,
) -> list[tuple[str, str]]:
    model.eval()
    results: list[tuple[str, str]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        results.append((prompt, text))
    model.train()
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny Toki Pona LM")
    p.add_argument(
        "--datasets",
        default="poki-lapo,wikipesija,tatoeba,monolingual,lipu-kule,lipu-sewi",
        help="Comma-separated corpora",
    )
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--decay-steps", type=int, default=500, help="Cooldown steps for WSD")
    p.add_argument("--save-stable", type=Path, default=None, help="Optional path to save pre-decay WSD checkpoint")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--decay-eval-every", type=int, default=100, help="Finer eval cadence during WSD decay")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/14m-v2"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"Using device: {device} (bf16={args.bf16})")

    tokenizer = TokiPonaTokenizer()
    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    train_tokens, val_tokens = load_or_build_corpus(
        dataset_names, tokenizer, seed=args.seed
    )
    print(
        f"Vocab size: {len(tokenizer)} | "
        f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}"
    )

    padded_vocab = ((len(tokenizer) + 63) // 64) * 64
    intermediate = args.intermediate_size or (args.d_model * 8 // 3)
    intermediate = ((intermediate + 63) // 64) * 64

    config = LlamaConfig(
        vocab_size=padded_vocab,
        hidden_size=args.d_model,
        intermediate_size=intermediate,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_heads,
        max_position_embeddings=args.seq_len,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    embed_params = model.model.embed_tokens.weight.numel()
    print(
        f"Model params: {total_params:,} total "
        f"({total_params - embed_params:,} non-embedding, {embed_params:,} embedding)"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    decay_start = max(args.warmup_steps, args.steps - args.decay_steps)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        if args.schedule == "wsd":
            if step < decay_start:
                return 1.0
            progress = (step - decay_start) / max(1, args.steps - decay_start)
            # 1 - sqrt cooldown (Hu et al. / Hagele et al. WSD)
            return max(0.0, 1.0 - math.sqrt(min(1.0, progress)))
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.bf16
        else nullcontext()
    )

    train_time = 0.0
    running_loss = 0.0
    best_val = float("inf")
    log_interval = max(1, min(100, args.eval_every))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list] = {"train": [], "val": []}

    model.train()
    for step in range(1, args.steps + 1):
        t_step = time.time()
        x = sample_batch(train_tokens, args.batch_size, args.seq_len, device)
        with autocast_ctx:
            out = model(input_ids=x, labels=x)
            loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        train_time += time.time() - t_step

        if args.schedule == "wsd" and args.save_stable and step == decay_start:
            args.save_stable.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.save_stable)
            tokenizer.save(args.save_stable / "tokenizer.json")
            print(f"Saved pre-decay stable checkpoint to {args.save_stable} @ step {step}")

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            running_loss = 0.0
            tok_per_sec = (step * args.batch_size * args.seq_len) / max(train_time, 1e-6)
            lr_now = scheduler.get_last_lr()[0]
            metrics["train"].append([step, round(avg_loss, 4), lr_now])
            print(
                f"step {step:5d}/{args.steps} | "
                f"train_loss {avg_loss:.3f} | "
                f"lr {lr_now:.2e} | "
                f"{tok_per_sec:,.0f} tok/s"
            )

        in_decay = args.schedule == "wsd" and step > decay_start
        do_eval = (
            step % args.eval_every == 0
            or (in_decay and step % args.decay_eval_every == 0)
            or step == args.steps
        )
        if do_eval:
            val_loss = evaluate(model, val_tokens, args.batch_size, args.seq_len, device)
            ppl = math.exp(min(val_loss, 20.0))
            metrics["val"].append([step, round(val_loss, 4)])
            saved_tag = ""
            if val_loss < best_val:
                best_val = val_loss
                model.save_pretrained(args.out_dir)
                tokenizer.save(args.out_dir / "tokenizer.json")
                saved_tag = " [saved best]"
            print(
                f"--- eval @ step {step}: val_loss {val_loss:.3f} | "
                f"val_ppl {ppl:.2f}{saved_tag} ---"
            )
            if step % args.eval_every == 0 or step == args.steps:
                for prompt, text in generate_samples(model, tokenizer, DEFAULT_PROMPTS, device):
                    print(f"  [{prompt}] -> {text}")
            print("---")

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Done. Best val_loss={best_val:.3f} (ppl {math.exp(best_val):.2f}) saved to {args.out_dir}")


if __name__ == "__main__":
    main()
